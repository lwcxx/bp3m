"""
Step 1: Download Gaia DR3 data for a sky region.

Queries the Gaia TAP archive in parallel magnitude bins (to stay within row
limits), applies quality flags, and saves a single CSV to:
    {output_dir}/{field_name}/Gaia/{field_name}_ra{ra}_{dec}_w{w}_h{h}_G{min}[_{max}].csv

A JSON sidecar with the same stem records all query parameters so that cache
validity can be checked precisely on subsequent runs.

Adapted from GaiaHub (del Pino et al.) — query structure, magnitude binning,
and quality-flag logic follow the original implementation exactly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from multiprocessing import Pool
from math import log10, floor

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from .catalog_utils import correct_flux_excess_factor, apply_quality_flags


# ── ADQL column lists ────────────────────────────────────────────────────────

_ASTROMETRIC_COLS = (
    "l, b, ra, ra_error, dec, dec_error, "
    "parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, "
    "radial_velocity, radial_velocity_error, "
    "ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr, "
    "dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr, "
    "parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr"
)

_PHOTOMETRIC_COLS = (
    "phot_g_mean_flux, "
    "phot_g_mean_mag AS gmag, "
    "(1.086*phot_g_mean_flux_error/phot_g_mean_flux) AS gmag_error, "
    "phot_bp_mean_mag AS bpmag, "
    "(1.086*phot_bp_mean_flux_error/phot_bp_mean_flux) AS bpmag_error, "
    "phot_rp_mean_mag AS rpmag, "
    "(1.086*phot_rp_mean_flux_error/phot_rp_mean_flux) AS rpmag_error, "
    "bp_rp, "
    "sqrt(power((1.086*phot_bp_mean_flux_error/phot_bp_mean_flux),2) "
    "    + power((1.086*phot_rp_mean_flux_error/phot_rp_mean_flux),2)) AS bp_rp_error"
)

_QUALITY_COLS = (
    "ecl_lat, pseudocolour, nu_eff_used_in_astrometry, "
    "visibility_periods_used, astrometric_excess_noise_sig, "
    "astrometric_params_solved, astrometric_n_good_obs_al, astrometric_chi2_al, "
    "phot_bp_rp_excess_factor, ruwe, "
    "(phot_bp_n_blended_transits+phot_rp_n_blended_transits)*1.0/"
    "(phot_bp_n_obs+phot_rp_n_obs) AS beta, "
    "ipd_gof_harmonic_amplitude, "
    "phot_bp_n_contaminated_transits, phot_rp_n_contaminated_transits, "
    "ref_epoch"
)


# ── Query helpers ────────────────────────────────────────────────────────────

def resolve_target(name):
    """Resolve a target name to (ra_deg, dec_deg) via Simbad."""
    from astroquery.simbad import Simbad
    custom = Simbad()
    custom.add_votable_fields('dim')
    tbl = custom.query_object(name)
    if tbl is None:
        raise ValueError(f"Simbad could not resolve '{name}'")

    # Newer astroquery returns lowercase columns with ra/dec already in degrees.
    # Older versions return uppercase 'RA'/'DEC' as sexagesimal strings.
    if 'ra' in tbl.colnames:
        ra  = float(tbl['ra'][0])
        dec = float(tbl['dec'][0])
    else:
        coo = SkyCoord(ra=tbl['RA'], dec=tbl['DEC'], unit=(u.hourangle, u.deg))
        ra  = float(coo.ra.deg[0])
        dec = float(coo.dec.deg[0])

    # Try to read angular size for auto search radius
    search_radius = None
    for col in ('galdim_majaxis', 'GALDIM_MAJAXIS'):
        if col not in tbl.colnames:
            continue
        try:
            val = tbl[col][0]
            if val is not None and float(val) > 0:
                search_radius = max(round(float(2. * val / 60.), 2), 0.1)
        except Exception:
            pass
        break
    return ra, dec, search_radius


def _build_query(source_table, ra, dec, width, height):
    """Build ADQL query string for a rectangular sky region."""
    box = (f"CONTAINS(POINT('ICRS',{source_table}.ra,{source_table}.dec),"
           f"BOX('ICRS',{ra:.8f},{dec:.8f},{width:.8f},{height:.8f}))=1")
    qcols = _QUALITY_COLS
    if 'dr3' in source_table and 'ruwe' not in qcols:
        qcols = 'ruwe, ' + qcols
    cols = f", {_ASTROMETRIC_COLS}, {_PHOTOMETRIC_COLS}, {qcols}"
    return f"SELECT source_id {cols} FROM {source_table} WHERE {box}"


def _mag_bins(min_mag, max_mag, area):
    """Log-spaced magnitude bins matching GaiaHub's density-adaptive scheme."""
    # n is the number of bin edges; minimum 2 to produce at least 1 bin.
    n = max(2, round((max_mag - min_mag) * max_mag**2 * area * 5e-5))
    return 1.0 + max_mag - np.logspace(
        log10(1.0), log10(1.0 + max_mag - min_mag), num=int(n))


def _query_mag_bin(args):
    """Worker: launch one Gaia TAP query for a magnitude slice."""
    from astroquery.gaia import Gaia
    query, min_g, max_g, ind_dir, field, n, n_total = args
    full_q = (query +
              f" AND (phot_g_mean_mag > {min_g:.4f})"
              f" AND (phot_g_mean_mag <= {max_g:.4f})")

    cache_path = None
    if ind_dir is not None:
        cache_path = Path(ind_dir) / f"{field}_G_{min_g:.4f}_{max_g:.4f}.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path)

    job = Gaia.launch_job_async(full_q)
    result = job.get_results().to_pandas()
    try:
        Gaia.remove_jobs([job.jobid])
    except Exception:
        pass

    if cache_path is not None:
        result.to_csv(cache_path, index=False)
    print(f"  Bin {n}/{n_total}: {len(result)} stars  (G {min_g:.2f}–{max_g:.2f})")
    return result


# ── Cache helpers ────────────────────────────────────────────────────────────

def _cache_stem(field_name: str, ra: float, dec: float,
                search_width: float, search_height: float,
                min_gmag: float, max_gmag: float | None) -> str:
    """Build informative filename stem for the Gaia cache files.

    The stem always ends in '_gaia' so the CSV matches the '*_gaia.csv' glob
    used by bp3m's data_loader_flc.
    """
    base = (f"{field_name}_ra{ra:.4f}_dec{dec:+.4f}"
            f"_w{search_width:.4f}_h{search_height:.4f}"
            f"_G{min_gmag:.1f}")
    if max_gmag is not None:
        base += f"_{max_gmag:.1f}"
    return base + "_gaia"


def _query_metadata(ra: float, dec: float, search_width: float,
                    search_height: float, min_gmag: float,
                    max_gmag: float | None, source_table: str,
                    sigma_flux_excess: float, only_5p: bool,
                    adql: str) -> dict:
    """Build the sidecar metadata dict to be saved alongside the CSV."""
    return {
        "ra": ra,
        "dec": dec,
        "search_width": search_width,
        "search_height": search_height,
        "min_gmag": min_gmag,
        "max_gmag": max_gmag,
        "source_table": source_table,
        "sigma_flux_excess": sigma_flux_excess,
        "only_5p": only_5p,
        "adql": adql,
    }


def _check_cache(csv_path: Path, meta_path: Path,
                 current_meta: dict) -> tuple[bool, list[str]]:
    """
    Return (cache_valid, list_of_differences).

    cache_valid is True only when both files exist and all metadata values match.
    """
    if not csv_path.exists():
        return False, []
    if not meta_path.exists():
        return False, ["no sidecar — cannot verify query match"]

    try:
        saved = json.loads(meta_path.read_text())
    except Exception as e:
        return False, [f"could not read sidecar: {e}"]

    diffs = []
    for key, cur_val in current_meta.items():
        if key == "adql":
            continue  # checked separately below (verbose)
        saved_val = saved.get(key, "<missing>")
        if saved_val != cur_val:
            diffs.append(f"  {key}: saved={saved_val!r}  current={cur_val!r}")

    if saved.get("adql") != current_meta["adql"]:
        diffs.append("  adql: query string differs")

    return len(diffs) == 0, diffs


# ── Public API ───────────────────────────────────────────────────────────────

def download_gaia(
    ra: float,
    dec: float,
    search_width: float,
    search_height: float,
    output_dir: Path,
    field_name: str,
    min_gmag: float = 16.0,
    max_gmag: float | None = None,
    source_table: str = 'gaiadr3.gaia_source',
    sigma_flux_excess: float = 3.0,
    only_5p: bool = False,
    n_processes: int = 4,
    force_redownload: bool = False,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Download and quality-filter Gaia stars in a rectangular sky region.

    Cached results are always used when available. The output CSV filename
    encodes the query geometry and magnitude limits; a JSON sidecar records all
    parameters so cache validity can be checked on subsequent runs.  Pass
    ``force_redownload=True`` to re-query the archive regardless.

    Individual magnitude-bin query results are always cached under
    ``{gaia_dir}/individual_queries/`` to speed up re-runs.

    Parameters
    ----------
    ra, dec            : centre coordinates (degrees)
    search_width/height: box size (degrees)
    output_dir         : pipeline root directory
    field_name         : subdirectory name (e.g. 'Sculptor_dSph')
    min_gmag           : brightest G magnitude to query
    max_gmag           : faintest G magnitude to query (None = no limit)
    source_table       : Gaia TAP table name
    sigma_flux_excess  : threshold for flux excess factor clipping
    only_5p            : restrict to 5-param astrometric solutions only
    n_processes        : parallel query workers (1 = serial)
    force_redownload   : ignore cache and re-query the Gaia archive
    quiet              : suppress interactive prompts

    Returns
    -------
    pd.DataFrame  — quality-filtered Gaia catalogue with all required columns
    """
    gaia_dir = Path(output_dir) / field_name / "Gaia"
    gaia_dir.mkdir(parents=True, exist_ok=True)

    stem     = _cache_stem(field_name, ra, dec, search_width, search_height,
                           min_gmag, max_gmag)
    out_path = gaia_dir / f"{stem}.csv"
    meta_path = gaia_dir / f"{stem}.query.json"

    # Gaia DR3 survey limit is ~G=21.5; using 99 as a sentinel would
    # generate a huge number of unnecessary magnitude bins.  G=22 gives
    # a small margin beyond the survey limit without blowing up the bin count.
    _max_gmag = max_gmag if max_gmag is not None else 22.0
    query = _build_query(source_table, ra, dec, search_width, search_height)
    current_meta = _query_metadata(ra, dec, search_width, search_height,
                                   min_gmag, max_gmag, source_table,
                                   sigma_flux_excess, only_5p, query)

    if not force_redownload:
        cache_ok, diffs = _check_cache(out_path, meta_path, current_meta)
        if cache_ok:
            print(f"[Gaia] Loading cached catalogue: {out_path}")
            return pd.read_csv(out_path)
        elif out_path.exists():
            if diffs == ["no sidecar — cannot verify query match"]:
                print(f"[Gaia] WARNING: cached CSV found but no query sidecar — "
                      f"loading anyway: {out_path}")
                return pd.read_csv(out_path)
            else:
                print(f"[Gaia] Cached query differs from current request "
                      f"— re-downloading:")
                for d in diffs:
                    print(d)

    print("\n" + "─"*50)
    print("Step 1: Downloading Gaia data")
    print("─"*50)
    print(f"  Centre:  ({ra:.4f}, {dec:.4f}) deg")
    print(f"  Box:     {search_width:.4f} × {search_height:.4f} deg")
    max_str = str(max_gmag) if max_gmag is not None else 'no limit'
    print(f"  G range: {min_gmag} – {max_str}")
    print(f"  Table:   {source_table}")

    ind_dir = gaia_dir / "individual_queries"
    ind_dir.mkdir(exist_ok=True)

    area  = search_width * search_height * abs(np.cos(np.deg2rad(dec)))
    bins  = _mag_bins(min_gmag, _max_gmag, area)
    n_bins = len(bins) - 1

    print(f"  Magnitude bins: {n_bins}  (area {area:.4f} deg²)")

    args = [
        (query, bins[i+1], bins[i], ind_dir, field_name, i+1, n_bins)
        for i in range(n_bins)
    ]

    if n_bins > 1 and n_processes > 1:
        workers = min(n_bins, 20, n_processes * 2)
        with Pool(workers) as pool:
            chunks = pool.map(_query_mag_bin, args)
    else:
        chunks = [_query_mag_bin(a) for a in args]

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=['SOURCE_ID'])
    print(f"  Raw stars before quality filter: {len(df)}")

    # Apply quality flags (needs phot_bp_rp_excess_factor column)
    if 'phot_bp_rp_excess_factor' in df.columns and 'bp_rp' in df.columns:
        df = apply_quality_flags(df, sigma_flux_excess=sigma_flux_excess,
                                  use_5p=only_5p)
    else:
        df['clean_label'] = True

    # Remove sidecar BEFORE writing the CSV so an interrupted write leaves no
    # stale sidecar that could mark a partial file as valid on the next run.
    if meta_path.exists():
        meta_path.unlink()

    df.to_csv(out_path, index=False)
    meta_path.write_text(json.dumps(current_meta, indent=2))

    # Individual bin files are now redundant — the merged CSV is the source of truth
    for f in ind_dir.glob("*.csv"):
        f.unlink()
    ind_dir.rmdir()

    n_clean = df['clean_label'].sum()
    print(f"  Stars after quality filter: {n_clean} / {len(df)}")
    print(f"  Saved: {out_path}")
    print(f"  Query metadata: {meta_path}")
    return df
