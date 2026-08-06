"""
Step 2: Search MAST and download HST (or future JWST) images for a sky region.

Uses astroquery.mast to find FLC exposures, presents the observations table
for review, and downloads selected products to:
    {output_dir}/{field_name}/HST/mastDownload/HST/{obs_id}/{obs_id}_flc.fits

The directory layout produced here is exactly what bp3m's data_loader_flc and
the cross-matcher expect.

Extension note
--------------
JWST support will be added by passing telescope='JWST' and the appropriate
instrument list. The download path will then use:
    {output_dir}/{field_name}/JWST/mastDownload/...
The psf_fitting and cross_match modules accept a ``telescope`` argument that
selects instrument-specific behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from astroquery.mast import Observations


# Instruments supported per telescope (MAST instrument_name values)
_INSTRUMENTS = {
    'HST':  ['ACS/WFC', 'WFC3/UVIS', 'ACS/HRC'],
    'JWST': ['NIRCAM', 'NIRISS', 'MIRI'],
}

# Map MAST instrument_name → STDPSFs/STDGDCs subdirectory name
# NIRCam uses mixed-case 'NIRCam' on disk to match the STScI directory naming.
_INST_TO_LIBDIR = {
    'ACS/WFC':   'ACSWFC',
    'WFC3/UVIS': 'WFC3UV',
    'ACS/HRC':   'ACSHRC',
    'NIRCAM':    'NIRCam',
    'NIRISS':    'NIRISS',
    'MIRI':      'MIRI',
}

# Map NIRCam detector names → library channel subdirectory (LWC or SWC).
# Keys cover both uppercase (FITS DETECTOR keyword) and lowercase (filename suffix).
# LWC: FITS uses NRCALONG/NRCBLONG; PSF library abbreviates these as NRCAL/NRCBL.
# SWC: FITS uses NRCA1–NRCA4 / NRCB1–NRCB4; filenames use the same in lowercase.
_NIRCAM_DETECTOR_TO_CHANNEL: dict[str, str] = {
    'nrca1': 'SWC', 'nrca2': 'SWC', 'nrca3': 'SWC', 'nrca4': 'SWC',
    'nrcb1': 'SWC', 'nrcb2': 'SWC', 'nrcb3': 'SWC', 'nrcb4': 'SWC',
    'NRCA1': 'SWC', 'NRCA2': 'SWC', 'NRCA3': 'SWC', 'NRCA4': 'SWC',
    'NRCB1': 'SWC', 'NRCB2': 'SWC', 'NRCB3': 'SWC', 'NRCB4': 'SWC',
    'nrcalong': 'LWC', 'nrcblong': 'LWC',
    'NRCALONG': 'LWC', 'NRCBLONG': 'LWC',
}

# LWC detector names are abbreviated in PSF/GDC filenames (NRCALONG → NRCAL).
# SWC detector names are unchanged, so .get(det, det.upper()) handles both cases.
_NIRCAM_LWC_LIBNAME: dict[str, str] = {
    'NRCALONG': 'NRCAL', 'nrcalong': 'NRCAL',
    'NRCBLONG': 'NRCBL', 'nrcblong': 'NRCBL',
}

# Default Gaia DR3 reference epoch as MJD (2017-05-28)
_GAIA_DR3_MJD = Time('2017-05-28').mjd

# Normalise filter names from PSF/GDC filenames to MAST canonical names.
# STScI PSF files occasionally use abbreviated names that differ from MAST.
_PSF_FILTER_NORM = {
    'F850L':  'F850LP',   # STDPSF_ACSWFC_F850L.fits → MAST F850LP
}

def _normalise_filter(name: str) -> str:
    """Map a PSF/GDC filename filter token to the MAST canonical name."""
    return _PSF_FILTER_NORM.get(name, name)


def _clean_mast_filter(raw: str, telescope: str = 'HST') -> str:
    """Return the science filter from a MAST filter string.

    MAST returns paired-filter strings separated by ';'. We split on ';', drop
    any token that is empty or starts with 'CLEAR', and return one of the
    remaining tokens. Falls back to the raw string if nothing survives.

    HST: paired strings like 'F814W;CLEAR1L' or 'CLEAR2L;F606W' — return the
    first remaining (science) token.
    JWST: paired strings like 'F277W;CLEAR' (drop CLEAR) or 'F150W;F150W2'
    (two science filters — pupil + filter wheel) — return the last token.
    """
    tokens = [t.strip() for t in raw.split(';')]
    science = [t for t in tokens if t and not t.upper().startswith('CLEAR')]
    if not science:
        return raw.strip()
    return science[-1] if telescope == 'JWST' else science[0]


def _query_params_sidecar(hst_dir: Path, field_name: str) -> Path:
    return hst_dir / f"{field_name}_obs_params.json"


def _make_query_params(
    ra, dec, search_width, search_height,
    hst_filters, t_exptime_min, t_exptime_max,
    time_baseline_days, date_second_epoch_mjd,
    obs_date_min, obs_date_max, im_type, telescope, instruments,
    lib_dir,
) -> dict:
    return {
        "ra":                    ra,
        "dec":                   dec,
        "search_width":          search_width,
        "search_height":         search_height,
        "hst_filters":           sorted(hst_filters) if hst_filters else None,
        "t_exptime_min":         t_exptime_min,
        "t_exptime_max":         float(t_exptime_max) if np.isfinite(t_exptime_max) else None,
        "time_baseline_days":    time_baseline_days,
        "date_second_epoch_mjd": date_second_epoch_mjd,
        "obs_date_min":          obs_date_min,
        "obs_date_max":          obs_date_max,
        "im_type":               im_type,
        "telescope":             telescope.upper(),
        "instruments":           sorted(instruments) if instruments else None,
        "lib_dir":               str(lib_dir) if lib_dir else None,
    }


def _nircam_filters(nircam_dir: Path, prefix: str) -> set[str]:
    """Collect filter names from NIRCam's two-level LWC/SWC library structure.

    NIRCam/LWC/ — files are flat: {prefix}_NRC{A|B}L_{filter}.fits
    NIRCam/SWC/ — one subdir per filter: SWC/{filter}/{prefix}_NRC{A|B}{1-4}_{filter}.fits
    """
    filters: set[str] = set()

    lwc_dir = nircam_dir / 'LWC'
    if lwc_dir.exists():
        for f in lwc_dir.glob(f'{prefix}_*.fits'):
            parts = f.stem.split('_')
            if len(parts) >= 3:
                filters.add(_normalise_filter(parts[2]))

    swc_dir = nircam_dir / 'SWC'
    if swc_dir.exists():
        for filt_dir in sorted(swc_dir.iterdir()):
            if filt_dir.is_dir() and any(filt_dir.glob(f'{prefix}_*.fits')):
                filters.add(filt_dir.name)

    return filters


def get_available_psf_gdc_combos(lib_dir: str | Path) -> dict[str, set[str]]:
    """
    Scan a lib/ directory to find instrument+filter combinations that
    have BOTH a STDPSF and a STDGDC file.

    Parameters
    ----------
    lib_dir : path to lib/ directory containing STDPSFs/ and STDGDCs/

    Returns
    -------
    dict mapping MAST instrument_name → set of filter strings that have both
    PSF and GDC.  E.g. ``{'ACS/WFC': {'F606W', 'F814W', ...}, ...}``
    """
    lib_dir = Path(lib_dir)
    psf_root = lib_dir / "STDPSFs"
    gdc_root = lib_dir / "STDGDCs"

    if not psf_root.exists() or not gdc_root.exists():
        return {}

    # Build reverse map: libdir name → MAST instrument name
    libdir_to_inst = {v: k for k, v in _INST_TO_LIBDIR.items()}

    result: dict[str, set[str]] = {}

    for psf_sub in sorted(psf_root.iterdir()):
        if not psf_sub.is_dir():
            continue
        det = psf_sub.name                              # e.g. 'ACSWFC', 'NIRCam'
        inst = libdir_to_inst.get(det)
        if inst is None:
            continue

        gdc_sub = gdc_root / det
        if not gdc_sub.exists():
            continue

        if det == 'NIRCam':
            # NIRCam has a two-level structure: LWC/ (flat files) and SWC/ (per-filter subdirs)
            psf_filters = _nircam_filters(psf_sub, 'STDPSF')
            gdc_filters = _nircam_filters(gdc_sub, 'STDGDC')
        else:
            # Collect filters with a PSF file (ignore _SM3/_SM4 variants and 'vintage')
            psf_filters: set[str] = set()
            for f in psf_sub.glob("STDPSF_*.fits"):
                parts = f.stem.split('_')
                # STDPSF_ACSWFC_F814W, STDPSF_ACSWFC_F814W_SM4, STDPSF_ACSWFC_F850L_SM3
                # SM-suffixed files are valid PSFs (e.g. F850L only exists as SM3).
                # Using a set means multiple variants for the same filter don't double-count.
                if len(parts) >= 3 and parts[-1] != 'vintage':
                    psf_filters.add(_normalise_filter(parts[2]))

            # Collect filters with a GDC file
            gdc_filters: set[str] = set()
            for f in gdc_sub.glob("STDGDC_*.fits"):
                parts = f.stem.split('_')
                if len(parts) >= 3 and 'OFFICIAL' not in parts and 'JFRAME' not in parts:
                    gdc_filters.add(_normalise_filter(parts[2]))

        common = psf_filters & gdc_filters
        if common:
            result[inst] = common

    return result


def search_mast(
    ra: float,
    dec: float,
    search_width: float,
    search_height: float,
    hst_filters: list[str] | None = None,
    project: list[str] | None = None,
    t_exptime_min: float = 2.0,
    t_exptime_max: float = np.inf,
    time_baseline_days: float | None = None,
    date_second_epoch_mjd: float = _GAIA_DR3_MJD,
    obs_date_min: str | None = None,
    obs_date_max: str | None = None,
    im_type: str | None = None,
    telescope: str = 'HST',
    instruments: list[str] | None = None,
    available_combos: dict[str, set[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Query MAST for science images of a sky region.

    Parameters
    ----------
    time_baseline_days: minimum telescope–Gaia time baseline in days. None means no
                        minimum (HST: all images up to date_second_epoch_mjd are kept;
                        JWST: all images from date_second_epoch_mjd onward are kept).
    obs_date_min    : earliest observation date to include (ISO string, e.g. '2005-01-01').
                      None means no lower bound.
    obs_date_max    : latest observation date to include (ISO string). None means no upper bound.
    instruments     : list of MAST instrument_name values to restrict to
                      (e.g. ['ACS/WFC']). None uses all supported instruments.
    available_combos: output of get_available_psf_gdc_combos(); if provided,
                      filters to only instrument+filter combos that have PSF+GDC.

    Returns
    -------
    obs_table          : one row per observation set (with i_exptime, t_baseline)
    data_products_table: one row per image file (with URI, parent_obsid, …)
    """
    if im_type is None:
        im_type = '_cal' if telescope == 'JWST' else '_flc'

    allowed_inst = _INSTRUMENTS.get(telescope.upper())
    if allowed_inst is None:
        raise ValueError(f"Unsupported telescope '{telescope}'. "
                         f"Choose from {list(_INSTRUMENTS)}")

    # Apply user instrument filter
    if instruments is not None:
        requested = [i.upper() for i in instruments]
        allowed_inst = [i for i in allowed_inst
                        if i.replace('/', '').upper() in requested
                        or i.upper() in requested]
        if not allowed_inst:
            raise ValueError(f"No supported instruments match {instruments}. "
                             f"Available: {_INSTRUMENTS[telescope.upper()]}")

    # Apply PSF+GDC availability filter
    if available_combos:
        allowed_inst = [i for i in allowed_inst if i in available_combos]
        if not allowed_inst:
            print("  WARNING: No instruments have both PSF and GDC files in lib_dir.")

    if hst_filters is None or hst_filters == ['any']:
        if available_combos:
            # Union of all filters available across all allowed instruments
            hst_filters = sorted(set.union(*[available_combos[i]
                                              for i in allowed_inst
                                              if i in available_combos]) or set())
        if not hst_filters:
            if telescope == 'JWST':
                hst_filters = ['F090W', 'F115W', 'F140M', 'F150W', 'F158M', 'F200W', 'F277W',
                               'F356W', 'F380M', 'F430M', 'F444W', 'F480W', 'F560W', 'F770W', 'F1000W']
            else:
                hst_filters = ['F435W', 'F475W', 'F555W', 'F606W',
                               'F625W', 'F658N', 'F775W', 'F814W', 'F850LP']
    if project is None:
        project = [telescope]

    # Shrink (HST) / grow (JWST) box slightly to avoid edge artefacts
    cos_dec = np.cos(np.deg2rad(dec))
    margin_ra  = 0.056 / cos_dec
    margin_dec = 0.056
    if telescope == 'JWST':
        ra1  = ra  - search_width  / 2 - margin_ra
        ra2  = ra  + search_width  / 2 + margin_ra
        dec1 = dec - search_height / 2 - margin_dec
        dec2 = dec + search_height / 2 + margin_dec
    else:
        ra1  = ra  - search_width  / 2 + margin_ra
        ra2  = ra  + search_width  / 2 - margin_ra
        dec1 = dec - search_height / 2 + margin_dec
        dec2 = dec + search_height / 2 - margin_dec

    # Build MAST time bounds.
    # HST predates the Gaia DR3 epoch, so when time_baseline_days is None all
    # images up to the Gaia epoch are kept, and t_max_mjd is pushed earlier by
    # time_baseline_days when given. JWST postdates Gaia DR3, so all JWST
    # images up to "now" are kept, and time_baseline_days instead raises the
    # minimum obs date.
    t_min_bound = 0
    if obs_date_min is not None:
        t_min_bound = Time(obs_date_min).mjd
    if telescope == 'JWST':
        t_max_mjd = Time.now().mjd
        if time_baseline_days is not None:
            t_min_bound = max(t_min_bound, _GAIA_DR3_MJD + time_baseline_days)
    else:
        if time_baseline_days is not None:
            t_max_mjd = _GAIA_DR3_MJD - time_baseline_days
        else:
            t_max_mjd = (Time.now()-366*u.day).mjd

    print(f"  Querying MAST (this can take a minute)...")
    import time as _time
    _mast_retries = 5
    _mast_delay   = 10  # seconds between retries
    for _attempt in range(_mast_retries):
        try:
            obs_raw = Observations.query_criteria(
                dataproduct_type=['image'],
                obs_collection=[telescope],
                s_ra=[ra1, ra2],
                s_dec=[dec1, dec2],
                instrument_name=allowed_inst,
                t_max=[t_min_bound, t_max_mjd],
                filters=hst_filters,
                project=project,
            )
            break
        except Exception as _e:
            if _attempt < _mast_retries - 1:
                print(f"  MAST query failed (attempt {_attempt+1}/{_mast_retries}): {_e}")
                print(f"  Retrying in {_mast_delay}s ...")
                _time.sleep(_mast_delay)
                _mast_delay *= 2  # exponential back-off
            else:
                raise

    if len(obs_raw) == 0:
        return pd.DataFrame(), pd.DataFrame()

    _delay2 = 10
    for _attempt in range(_mast_retries):
        try:
            prod_raw = Observations.get_product_list(obs_raw)
            break
        except Exception as _e:
            if _attempt < _mast_retries - 1:
                print(f"  MAST get_product_list failed (attempt {_attempt+1}/{_mast_retries}): {_e}")
                print(f"  Retrying in {_delay2}s ...")
                _time.sleep(_delay2)
                _delay2 *= 2
            else:
                raise
    im_sub   = im_type[1:].upper()   # '_flc' → 'FLC'
    if telescope == 'JWST':
        # JWST has no DRZ-equivalent product at the FLC level.
        mask = (
            (prod_raw['productSubGroupDescription'] == im_sub) &
            (prod_raw['obs_collection'] == telescope)
        )
    else:
        mask = (
            ((prod_raw['productSubGroupDescription'] == im_sub) |
             (prod_raw['productSubGroupDescription'] == 'DRZ')) &
            (prod_raw['obs_collection'] == telescope)
        )
    prod_df = prod_raw[mask].to_pandas()
    obs_df  = obs_raw.to_pandas()

    # Drop HAP pipeline products. Note: JWST does not have HAP products.
    prod_df = prod_df[~prod_df['project'].str.contains('HAP', na=False)]

    # Count exposures per observation to compute individual exposure time
    n_exp = (prod_df[prod_df['productSubGroupDescription'] == im_sub]
             .groupby('parent_obsid')['parent_obsid']
             .count()
             .rename('n_exp'))
    obs_df['obsid'] = obs_df['obsid'].astype(str)
    n_exp.index = n_exp.index.astype(str)
    obs_df = obs_df.merge(n_exp.rename_axis('obsid'), on='obsid', how='inner')
    obs_df['i_exptime'] = obs_df['t_exptime'] / obs_df['n_exp']

    obs_time = Time(obs_df['t_max'].values, format='mjd')
    obs_time.format = 'iso'; obs_time.out_subfmt = 'date'
    obs_df['obs_time']   = obs_time.value
    if telescope == 'JWST':
        obs_df['t_baseline'] = np.round(
            -(date_second_epoch_mjd - obs_df['t_max'].values) / 365.2422, 2)
    else:
        obs_df['t_baseline'] = np.round(
            (date_second_epoch_mjd - obs_df['t_max'].values) / 365.2422, 2)
    obs_df['filters'] = obs_df['filters'].apply(lambda f: _clean_mast_filter(f, telescope=telescope))

    # Merge exposure-time info into products table
    meta = obs_df[['obsid', 'i_exptime', 'filters', 't_baseline', 's_ra', 's_dec']]
    prod_df['parent_obsid'] = prod_df['parent_obsid'].astype(str)
    prod_df = prod_df.merge(
        meta.rename(columns={'obsid': 'parent_obsid'}), on='parent_obsid', how='left')

    # Filter by exposure time and (optionally) time baseline
    t_base_yr = time_baseline_days / 365.2422 if time_baseline_days is not None else -np.inf
    obs_df = obs_df[
        (obs_df['i_exptime'] >= t_exptime_min) &
        (obs_df['i_exptime'] <= t_exptime_max) &
        (obs_df['t_baseline'] >= t_base_yr)
    ]
    prod_df = prod_df[
        (prod_df['i_exptime'] >= t_exptime_min) &
        (prod_df['i_exptime'] <= t_exptime_max) &
        (prod_df['t_baseline'] >= t_base_yr)
    ]

    # Post-query date filter for obs_date_max
    if obs_date_max is not None:
        t_max_iso = Time(obs_date_max).mjd
        obs_df  = obs_df[obs_df['t_max'] <= t_max_iso]
        prod_df = prod_df[prod_df['t_baseline'].notna()]   # already merged; filter via obsid
        keep_ids = set(obs_df['obsid'].astype(str))
        prod_df = prod_df[prod_df['parent_obsid'].isin(keep_ids)]

    # Filter products to only PSF+GDC-available instrument+filter combos
    if available_combos:
        def _combo_ok(row):
            inst_name = row.get('instrument_name', '')
            if telescope == 'JWST':
                inst_name = inst_name.split('/')[0]  # 'NIRCAM/IMAGE' → 'NIRCAM'
            filt_name = row.get('filters', '')
            if inst_name not in available_combos:
                return False
            return filt_name in available_combos[inst_name]

        if 'instrument_name' in obs_df.columns and 'filters' in obs_df.columns:
            mask_obs = obs_df.apply(_combo_ok, axis=1)
            dropped = obs_df[~mask_obs]
            if not dropped.empty:
                dropped_combos = sorted(set(
                    f"{r['instrument_name']}/{r['filters']}"
                    for _, r in dropped.iterrows()
                ))
                print(f"  WARNING: {len(dropped)} observation(s) dropped — no PSF+GDC "
                      f"in lib_dir for: {', '.join(dropped_combos)}")
            obs_df  = obs_df[mask_obs]
            keep_ids = set(obs_df['obsid'].astype(str))
            prod_df = prod_df[prod_df['parent_obsid'].isin(keep_ids)]

    return obs_df.reset_index(drop=True), prod_df.reset_index(drop=True)


def download_hst_images(
    ra: float,
    dec: float,
    search_width: float,
    search_height: float,
    output_dir: Path,
    field_name: str,
    hst_filters: list[str] | None = None,
    project: list[str] | None = None,
    t_exptime_min: float = 2.0,
    t_exptime_max: float = np.inf,
    time_baseline_days: float | None = None,
    date_second_epoch_mjd: float = _GAIA_DR3_MJD,
    obs_date_min: str | None = None,
    obs_date_max: str | None = None,
    im_type: str | None = None,
    telescope: str = 'HST',
    instruments: list[str] | None = None,
    lib_dir: str | Path | None = None,
    gaia_df: 'pd.DataFrame | None' = None,
    field_ids: list[int] | None = None,
    quiet: bool = False,
    force_redownload: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Search MAST for images of a field and download them.

    Creates:
        {output_dir}/{field_name}/{telescope}/mastDownload/{telescope}/{obs_id}/...

    Parameters
    ----------
    obs_date_min    : earliest observation date (ISO string, e.g. '2005-01-01'). None = no limit.
    obs_date_max    : latest observation date (ISO string). None = no limit.
    instruments     : restrict to these instrument/detector names (None = all supported).
    lib_dir         : path to lib/ containing STDPSFs/ and STDGDCs/. If provided, only
                      instruments+filters with both PSF and GDC files are kept.
    gaia_df         : Gaia catalogue DataFrame (used to count stars per footprint). Optional.
    field_ids       : list of 1-based field IDs to download (integers shown in the table).
                      None = all. Pass [0] to skip download and just return the table.
    force_redownload: if True, re-query MAST and re-download even if cached files exist.

    Returns
    -------
    obs_table, data_products_table  (both pd.DataFrame)
    """
    if im_type is None:
        im_type = '_cal' if telescope == 'JWST' else '_flc'
    tel_upper  = telescope.upper()
    hst_dir    = Path(output_dir) / field_name / tel_upper
    hst_dir.mkdir(parents=True, exist_ok=True)
    obs_csv    = hst_dir / f"{field_name}_obs.csv"
    prod_csv   = hst_dir / f"{field_name}_data_products.csv"

    print("\n" + "─"*50)
    print(f"Step 2: Searching MAST for {tel_upper} images")
    print("─"*50)

    # Load PSF+GDC availability if lib_dir provided
    available_combos: dict[str, set[str]] | None = None
    if lib_dir is not None:
        available_combos = get_available_psf_gdc_combos(lib_dir)
        if available_combos:
            total = sum(len(v) for v in available_combos.values())
            print(f"  PSF+GDC availability: {total} instrument+filter combos in lib_dir")
        else:
            print(f"  WARNING: No PSF+GDC combos found in {lib_dir}")

    current_params = _make_query_params(
        ra, dec, search_width, search_height,
        hst_filters, t_exptime_min, t_exptime_max,
        time_baseline_days, date_second_epoch_mjd,
        obs_date_min, obs_date_max, im_type, telescope, instruments, lib_dir,
    )
    params_sidecar = _query_params_sidecar(hst_dir, field_name)

    use_cache = (
        not force_redownload
        and obs_csv.exists() and prod_csv.exists()
        and params_sidecar.exists()
    )
    if use_cache:
        stored_params = json.loads(params_sidecar.read_text())
        diffs = [f"    {k}: {stored_params.get(k)!r} → {current_params[k]!r}"
                 for k in current_params
                 if current_params[k] != stored_params.get(k)]
        if diffs:
            print("  Query params changed — re-searching MAST:")
            for d in diffs:
                print(d)
            use_cache = False

    if use_cache:
        print(f"  Loading cached observation table from {hst_dir}")
        obs_df  = pd.read_csv(obs_csv)
        prod_df = pd.read_csv(prod_csv)
    else:
        obs_df, prod_df = search_mast(
            ra, dec, search_width, search_height,
            hst_filters=hst_filters, project=project,
            t_exptime_min=t_exptime_min, t_exptime_max=t_exptime_max,
            time_baseline_days=time_baseline_days,
            date_second_epoch_mjd=date_second_epoch_mjd,
            obs_date_min=obs_date_min, obs_date_max=obs_date_max,
            im_type=im_type, telescope=telescope,
            instruments=instruments,
            available_combos=available_combos,
        )
        obs_df.to_csv(obs_csv,   index=False)
        prod_df.to_csv(prod_csv, index=False)
        params_sidecar.write_text(json.dumps(current_params, indent=2))

    if obs_df.empty:
        print("  No observations found matching the criteria.")
        # Write empty manifest so downstream steps know to exit gracefully.
        manifest = hst_dir / f"{field_name}_selected_obsids.json"
        manifest.write_text("[]")
        return obs_df, prod_df

    # Attach field_id (1-based sequential index shown to the user)
    obs_df = obs_df.reset_index(drop=True)
    obs_df.insert(0, 'field_id', np.arange(1, len(obs_df) + 1))

    # Count Gaia stars in each footprint if catalog is available
    if gaia_df is not None and 's_region' in obs_df.columns:
        obs_df['n_gaia'] = _count_gaia_in_footprints(obs_df, gaia_df)
    elif 'n_gaia' not in obs_df.columns:
        obs_df['n_gaia'] = -1   # unknown

    # Propagate n_gaia and field_id to products table via obsid
    obs_df['obsid'] = obs_df['obsid'].astype(str)
    prod_df['parent_obsid'] = prod_df['parent_obsid'].astype(str)
    id_map = obs_df.set_index('obsid')[['field_id']].rename_axis('parent_obsid')
    prod_df = prod_df.merge(id_map.reset_index(), on='parent_obsid', how='left')

    print(f"\n  Found {len(obs_df)} observation(s):")
    _print_obs_table(obs_df)

    # Save footprint plot
    footprint_png = hst_dir / f"{field_name}_footprints.png"
    try:
        plot_footprints(obs_df, footprint_png,
                        gaia_df=gaia_df, field_name=field_name,
                        ra=ra, dec=dec,
                        search_width=search_width,
                        search_height=search_height,
                        telescope=telescope)
    except Exception as _e:
        print(f"  WARNING: footprint plot failed — {_e}")

    # Select which observations to download
    if field_ids == 'all':
        print("  Downloading all observations.")

    elif field_ids is not None:
        if 0 in field_ids:
            print("  Skipping download (field_id 0).")
            return obs_df, prod_df
        selected_obsids = set(
            obs_df.loc[obs_df['field_id'].isin(field_ids), 'obsid'].astype(str)
        )
        obs_df  = obs_df[obs_df['obsid'].astype(str).isin(selected_obsids)]
        prod_df = prod_df[prod_df['parent_obsid'].isin(selected_obsids)]

    elif not quiet:
        choice = input(
            "\n  Enter field IDs to download (space-separated, e.g. '1 3 5'), "
            "or 'y' for all, 'n' to skip: "
        ).strip()
        if choice.lower() == 'n':
            return obs_df, prod_df
        elif choice.lower() not in ('y', ''):
            try:
                ids = [int(x) for x in choice.split()]
            except ValueError:
                print("  Invalid input — downloading all.")
                ids = list(obs_df['field_id'])
            selected_obsids = set(
                obs_df.loc[obs_df['field_id'].isin(ids), 'obsid'].astype(str)
            )
            obs_df  = obs_df[obs_df['obsid'].astype(str).isin(selected_obsids)]
            prod_df = prod_df[prod_df['parent_obsid'].isin(selected_obsids)]

    # Download image products only
    flc_sub = im_type[1:].upper()
    to_dl   = prod_df[prod_df['productSubGroupDescription'] == flc_sub].copy()
    if to_dl.empty:
        print(f"  No {im_type} products to download.")
        return obs_df, prod_df

    # Skip already-downloaded files unless force_redownload
    failed_obsids: dict[str, str] = {}  # obs_id → reason (kept on disk, skipped downstream)
    if not force_redownload and 'dataURI' in to_dl.columns:
        from astropy.io import fits
        mast_root = hst_dir / "mastDownload" / tel_upper
        already = []
        broken = []
        for _, row in to_dl.iterrows():
            # MAST download path mirrors the URI structure:
            # mast:HST/product/jXXX_flc.fits → mastDownload/HST/jXXX/jXXX_flc.fits
            fname = Path(row['dataURI']).name
            obs_id = row.get('obs_id', '')
            dest = mast_root / obs_id / fname
            if not dest.exists():
                continue

            disk_size = dest.stat().st_size
            expected_size = row.get('size', None)

            # Fast size check: catches empty files and truncated downloads
            if disk_size == 0 or (expected_size and disk_size != expected_size):
                reason = "empty" if disk_size == 0 else f"size {disk_size} != expected {expected_size}"
                print(f"  WARNING: {fname} is corrupt ({reason}) — will re-download.")
                dest.unlink()
                _invalidate_psf_cache(dest)
                broken.append(fname)
                continue

            # FITS integrity check for files that passed the size check
            try:
                with fits.open(dest, memmap=False) as hdul:
                    hdul.verify('exception')
            except Exception as e:
                print(f"  WARNING: {fname} failed FITS check ({e}) — will re-download.")
                dest.unlink()
                _invalidate_psf_cache(dest)
                broken.append(fname)
                continue

            # Failed-observation check: EXPTIME=0 means no real sky data collected.
            # Keep the file on disk but exclude from downstream processing.
            fail_reason = _check_exptime(dest, telescope=telescope)
            if fail_reason:
                print(f"  WARNING: {fname} is a failed observation ({fail_reason}) — "
                      f"skipping all downstream steps.")
                _invalidate_psf_cache(dest)
                failed_obsids[obs_id] = fail_reason
                already.append(row.name)  # don't re-download
                continue

            already.append(row.name)

        if already:
            n_valid = len(already) - len(failed_obsids)
            print(f"  {n_valid} file(s) already cached and verified; skipping re-download.")
        if broken:
            print(f"  {len(broken)} broken file(s) removed; will re-download.")
        to_dl = to_dl.drop(index=already)

    if to_dl.empty:
        print("  All files already downloaded.")
        if failed_obsids:
            print(f"  NOTE: {len(failed_obsids)} failed observation(s) excluded from processing: "
                  + ", ".join(sorted(failed_obsids)))
        _write_selected_obsids(prod_df, hst_dir, field_name, im_type, failed_obsids)
        return obs_df, prod_df

    # Invalidate PSF caches for every file about to be downloaded so that a
    # partially-completed or interrupted download never leaves stale PSF outputs.
    if 'dataURI' in to_dl.columns:
        _mast_root = hst_dir / "mastDownload" / tel_upper
        for _, _row in to_dl.iterrows():
            _dest = _mast_root / _row.get('obs_id', '') / Path(_row['dataURI']).name
            _invalidate_psf_cache(_dest)

    print(f"\n  Downloading {len(to_dl)} {im_type} file(s) to {hst_dir}...")
    import time as _time
    _dl_delay = 10
    for _dl_attempt in range(5):
        try:
            try:
                Observations.download_products(
                    Table.from_pandas(to_dl), download_dir=str(hst_dir))
            except Exception:
                Observations.download_products(to_dl, download_dir=str(hst_dir))
            break
        except Exception as _e:
            if _dl_attempt < 4:
                print(f"  Download failed (attempt {_dl_attempt+1}/5): {_e}")
                print(f"  Retrying in {_dl_delay}s ...")
                _time.sleep(_dl_delay)
                _dl_delay *= 2
            else:
                raise

    print("  Download complete.")

    # Validate newly downloaded files for failed observations (EXPTIME=0).
    if 'dataURI' in to_dl.columns:
        mast_root_nd = hst_dir / "mastDownload" / tel_upper
        for _, row in to_dl.iterrows():
            obs_id = row.get('obs_id', '')
            if obs_id in failed_obsids:
                continue
            fname = Path(row['dataURI']).name
            dest = mast_root_nd / obs_id / fname
            if not dest.exists():
                continue
            fail_reason = _check_exptime(dest, telescope=telescope)
            if fail_reason:
                print(f"  WARNING: {fname} is a failed observation ({fail_reason}) — "
                      f"skipping all downstream steps.")
                _invalidate_psf_cache(dest)
                failed_obsids[obs_id] = fail_reason

    if failed_obsids:
        print(f"  NOTE: {len(failed_obsids)} failed observation(s) excluded from processing: "
              + ", ".join(sorted(failed_obsids)))
    _write_selected_obsids(prod_df, hst_dir, field_name, im_type, failed_obsids)
    return obs_df, prod_df


# Filter → display colour mapping (approximate true-colour ordering)
_FILTER_COLORS = {
    # HST
    'F220W': '#9b59b6', 'F225W': '#8e44ad', 'F275W': '#6c3483',
    'F330W': '#4a235a', 'F336W': '#7d3c98', 'F350LP': '#aab7b8',
    'F390M': '#2471a3', 'F390W': '#2e86c1', 'F435W': '#1a5276',
    'F438W': '#154360', 'F467M': '#1f618d', 'F475W': '#2980b9',
    'F502N': '#148f77', 'F547M': '#1e8449', 'F550M': '#1d8348',
    'F555W': '#27ae60', 'F600LP': '#d4ac0d', 'F606W': '#f1c40f',
    'F621M': '#e67e22', 'F625W': '#d35400', 'F658N': '#cb4335',
    'F660N': '#c0392b', 'F775W': '#e74c3c', 'F814W': '#922b21',
    'F850LP': '#641e16', 'F850L': '#641e16',
    # JWST NIRCam short wavelength (0.9–2.0 μm) — dark reds → purples
    'F090W':  '#922b21',
    'F115W':  '#7b241c',
    'F140W':  '#5b2c6f',
    'F150W':  '#6c3483',
    'F158W':  '#7d3c98',
    'F200W':  '#4a235a',
    # JWST NIRCam long wavelength (2.7–4.8 μm) — navy → blue → teal → green → gold
    'F277W':  '#154360',
    'F356W':  '#1a5276',
    'F380W':  '#1f618d',
    'F430W':  '#148f77',
    'F444W':  '#1e8449',
    'F480W':  '#d4ac0d',
    # JWST MIRI (5.6–10 μm) — orange → red
    'F560W':  '#e67e22',
    'F770W':  '#d35400',
    'F1000W': '#c0392b',
}
_DEFAULT_COLOR = '#95a5a6'


def plot_footprints(
    obs_df: pd.DataFrame,
    save_path: str | Path,
    gaia_df: 'pd.DataFrame | None' = None,
    field_name: str = '',
    ra: float | None = None,
    dec: float | None = None,
    search_width: float | None = None,
    search_height: float | None = None,
    telescope: str = 'HST',
) -> None:
    """
    Plot HST/JWST image footprints on the sky with Gaia stars in the background.

    Footprints are coloured by filter, labelled with their field_id, and
    a legend shows which filter maps to which colour.  Saves a PNG to
    ``save_path``.

    Parameters
    ----------
    obs_df        : observations DataFrame with columns field_id, s_region, filters,
                    proposal_id, instrument_name, obs_time.
    save_path     : output PNG path.
    gaia_df       : optional Gaia catalogue; plotted as background scatter.
    field_name    : used in the figure title.
    ra, dec       : field centre (degrees).  When provided together with
                    search_width / search_height, the axes are fixed to the
                    user-specified search box, preventing wildly-offset WCS
                    entries from zooming the plot out.
    search_width, search_height : search box full-width in degrees.
    telescope     : used in the figure title.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots(figsize=(9, 8))

    # ── Background Gaia stars ─────────────────────────────────────────────────
    if gaia_df is not None and len(gaia_df) > 0:
        gmag = gaia_df['gmag'].values if 'gmag' in gaia_df.columns else None
        ax.scatter(gaia_df['ra'].values, gaia_df['dec'].values,
                   c=gmag, cmap='Greys', vmin=16, vmax=22,
                   s=2, alpha=0.5, rasterized=True, zorder=1)

    # ── Footprint polygons ────────────────────────────────────────────────────
    filter_patches: dict[str, mpatches.Patch] = {}   # for legend

    for _, row in obs_df.iterrows():
        filt    = str(row.get('filters', '')).strip().upper()
        fid     = int(row.get('field_id', 0))
        color   = _FILTER_COLORS.get(filt, _DEFAULT_COLOR)
        s_region = str(row.get('s_region', ''))

        polygons = _parse_polygons(s_region)
        if not polygons:
            continue

        for poly_verts in polygons:
            patch = MplPolygon(poly_verts, closed=True,
                               facecolor='none', edgecolor=color,
                               lw=1.5, zorder=2)
            ax.add_patch(patch)

        # Label at centroid of first polygon
        verts = polygons[0]
        cx = np.mean(verts[:, 0])
        cy = np.mean(verts[:, 1])
        ax.text(cx, cy, str(fid), ha='center', va='center',
                fontsize=7, fontweight='bold', color=color,
                zorder=4,
                bbox=dict(boxstyle='round,pad=0.15', fc='white',
                          ec='none', alpha=0.6))

        if filt not in filter_patches:
            filter_patches[filt] = mpatches.Patch(
                facecolor='none', edgecolor=color, lw=1.5,
                label=filt)

    # ── Axes limits ───────────────────────────────────────────────────────────
    # Strategy: use data-derived limits (zoom in to where HST actually points)
    # but clamp to the user's cutout so that wildly-offset WCS entries cannot
    # zoom the plot out beyond the requested field of view.
    #
    # When the search box is available:
    #   - Only footprint centroids that fall inside the cutout contribute to
    #     the data bounds (filters out corrupt WCS entries).
    #   - Final limits = data bounds with 8 % padding, clamped to cutout.
    # When the search box is not available: use raw data bounds as before.

    pad_factor = 0.08

    if ra is not None and dec is not None and search_width and search_height:
        cos_d = np.cos(np.deg2rad(dec))
        cut_ra_lo  = ra  - search_width  / 2 / cos_d
        cut_ra_hi  = ra  + search_width  / 2 / cos_d
        cut_dec_lo = dec - search_height / 2
        cut_dec_hi = dec + search_height / 2

        # Collect bounds from footprints whose centroid lies inside the cutout.
        all_ra, all_dec = [], []
        for _, row in obs_df.iterrows():
            bbox = _footprint_bbox(str(row.get('s_region', '')))
            if not bbox:
                continue
            cra  = (bbox[0] + bbox[1]) / 2
            cdec = (bbox[2] + bbox[3]) / 2
            if cut_ra_lo <= cra <= cut_ra_hi and cut_dec_lo <= cdec <= cut_dec_hi:
                all_ra  += [bbox[0], bbox[1]]
                all_dec += [bbox[2], bbox[3]]
        if gaia_df is not None and len(gaia_df) > 0:
            all_ra  += list(gaia_df['ra'].values)
            all_dec += list(gaia_df['dec'].values)

        if all_ra:
            span_ra  = max(all_ra)  - min(all_ra)
            span_dec = max(all_dec) - min(all_dec)
            data_ra_lo  = min(all_ra)  - span_ra  * pad_factor
            data_ra_hi  = max(all_ra)  + span_ra  * pad_factor
            data_dec_lo = min(all_dec) - span_dec * pad_factor
            data_dec_hi = max(all_dec) + span_dec * pad_factor
            # Clamp: zoom in freely, but never exceed the cutout.
            ra_lo  = max(cut_ra_lo,  data_ra_lo)
            ra_hi  = min(cut_ra_hi,  data_ra_hi)
            dec_lo = max(cut_dec_lo, data_dec_lo)
            dec_hi = min(cut_dec_hi, data_dec_hi)
        else:
            # No footprints inside cutout — show full cutout.
            ra_lo, ra_hi   = cut_ra_lo,  cut_ra_hi
            dec_lo, dec_hi = cut_dec_lo, cut_dec_hi

        center_dec = (dec_lo + dec_hi) / 2
    else:
        all_ra, all_dec = [], []
        for _, row in obs_df.iterrows():
            bbox = _footprint_bbox(str(row.get('s_region', '')))
            if bbox:
                all_ra  += [bbox[0], bbox[1]]
                all_dec += [bbox[2], bbox[3]]
        if gaia_df is not None and len(gaia_df) > 0:
            all_ra  += list(gaia_df['ra'].values)
            all_dec += list(gaia_df['dec'].values)
        if not all_ra:
            plt.close(fig)
            return
        pad_ra  = (max(all_ra)  - min(all_ra))  * pad_factor
        pad_dec = (max(all_dec) - min(all_dec)) * pad_factor
        ra_lo, ra_hi   = min(all_ra) - pad_ra, max(all_ra) + pad_ra
        dec_lo, dec_hi = min(all_dec) - pad_dec, max(all_dec) + pad_dec
        center_dec = (dec_lo + dec_hi) / 2

    ax.set_xlim(ra_hi, ra_lo)   # RA right-to-left
    ax.set_ylim(dec_lo, dec_hi)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(center_dec)), adjustable='box')

    # ── Legend, labels, title ─────────────────────────────────────────────────
    if filter_patches:
        ax.legend(handles=list(filter_patches.values()),
                  title='Filter', fontsize=8, title_fontsize=8,
                  loc='best', framealpha=0.8)

    ax.set_xlabel('R.A. (deg)')
    ax.set_ylabel('Dec. (deg)')
    title = f'{field_name} — {telescope} footprints' if field_name else f'{telescope} footprints'
    ax.set_title(title, fontsize=12)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Footprint plot saved: {save_path}")


def _parse_polygons(s_region: str) -> list[np.ndarray]:
    """
    Parse a MAST s_region string into a list of (N,2) vertex arrays.
    Handles strings with one or more POLYGON blocks.
    """
    polys = []
    # Split on 'POLYGON' keyword (case-insensitive)
    parts = s_region.upper().split('POLYGON')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            nums = [float(x) for x in part.split()]
            if len(nums) < 6:
                continue
            verts = np.array(nums).reshape(-1, 2)
            polys.append(verts)
        except (ValueError, TypeError):
            continue
    return polys


def find_flc_images(output_dir: Path, field_name: str,
                    telescope: str = 'HST', im_type: str = '_flc') -> list[Path]:
    """
    Return sorted list of downloaded FLC (HST) or _cal (JWST) FITS paths for a field.

    Expected structure:
        {output_dir}/{field}/{telescope}/mastDownload/{telescope}/{obs_id}/{obs_id}{im_type}.fits
    """
    root = Path(output_dir) / field_name / telescope.upper() / "mastDownload" / telescope.upper()
    suffix = f"{im_type}.fits"
    found = sorted(root.rglob(f"*{suffix}")) if root.exists() else []
    return found


def _footprint_bbox(s_region: str) -> tuple[float, float, float, float] | None:
    """
    Parse a MAST s_region string (one or more POLYGON vertices) and return
    (ra_min, ra_max, dec_min, dec_max).  Returns None if unparseable.
    """
    try:
        tokens = s_region.upper().replace('POLYGON', ' ').split()
        coords = [float(t) for t in tokens]
        if len(coords) < 4:
            return None
        ras  = coords[0::2]
        decs = coords[1::2]
        return min(ras), max(ras), min(decs), max(decs)
    except Exception:
        return None


def _count_gaia_in_footprints(obs_df: pd.DataFrame,
                               gaia_df: pd.DataFrame) -> pd.Series:
    """
    For each row in obs_df, count Gaia stars whose (ra, dec) falls within
    the image footprint bounding box derived from s_region.
    Returns a Series of integer counts aligned to obs_df's index.
    """
    ra_g  = gaia_df['ra'].values
    dec_g = gaia_df['dec'].values
    counts = []
    for _, row in obs_df.iterrows():
        bbox = _footprint_bbox(str(row.get('s_region', '')))
        if bbox is None:
            counts.append(-1)
            continue
        ra_min, ra_max, dec_min, dec_max = bbox
        n = int(np.sum(
            (ra_g  >= ra_min) & (ra_g  <= ra_max) &
            (dec_g >= dec_min) & (dec_g <= dec_max)
        ))
        counts.append(n)
    return pd.Series(counts, index=obs_df.index)


def _invalidate_psf_cache(flc_path: Path) -> None:
    """Delete PSF and cross-match caches for a given FLC/CAL path, if they exist."""
    for p in (flc_path.parent / f"{flc_path.stem}_catalog.fits",
              flc_path.parent / "psf_params.json",
              flc_path.parent / "matched_gaia.csv",
              flc_path.parent / "xmatch_params.json"):
        if p.exists():
            p.unlink()


def _check_exptime(flc_path: Path, telescope: str = 'HST') -> str | None:
    """Return a failure reason string if the image is a failed observation, else None.

    HST — checks three conditions in priority order (file is kept on disk in all cases):
    1. EXPTIME == 0 — shutter open but no real sky signal (e.g. EXCESSIVE DOWNTIME).
    2. EXPFLAG != 'NORMAL' — any non-nominal exposure flag indicates compromised data.
       Known values seen in practice:
         'EXCESSIVE DOWNTIME'    — guide-star loss; EXPTIME typically 0
         'TDF-DOWN AT EXPSTART'  — science telemetry unavailable; data may be corrupt
         'INTERRUPTED'           — exposure cut short by HST safing or guide-star loss
       Any other non-NORMAL value is also flagged.
    3. No HDRLET extensions — image was not fully processed by the HST calibration
       pipeline (drizzle/astrodrizzle did not apply WCS corrections). These images
       have less trustworthy astrometry and tend to produce poor cross-matches.

    JWST — checks four conditions in priority order:
    1. EFFEXPTM == 0   — no effective exposure time; no real sky signal collected.
    2. ENG_QUAL != 'OK' — guide-star or engineering problem during the exposure.
    3. DATAPROB == True — pipeline detected a data problem.
    4. VISITSTA != 'SUCCESSFUL' — visit did not complete successfully.
    """
    from astropy.io import fits
    if telescope == 'JWST':
        try:
            with fits.open(flc_path, memmap=False) as hdul:
                h = hdul[0].header
                effexptm = h.get('EFFEXPTM', None)
                eng_qual = h.get('ENG_QUAL', '').strip()
                dataprob = h.get('DATAPROB', False)
                visitsta = h.get('VISITSTA', '').strip()
            if effexptm is not None and float(effexptm) == 0.0:
                return "EFFEXPTM=0.0"
            if eng_qual and eng_qual != 'OK':
                return f"ENG_QUAL='{eng_qual}'"
            if dataprob:
                return "DATAPROB=True"
            if visitsta and visitsta != 'SUCCESSFUL':
                return f"VISITSTA='{visitsta}'"
        except Exception:
            pass
        return None

    try:
        with fits.open(flc_path, memmap=False) as hdul:
            exptime = hdul[0].header.get('EXPTIME', None)
            expflag = hdul[0].header.get('EXPFLAG', '').strip()
            has_hdrlet = any(hdu.name == 'HDRLET' for hdu in hdul)
        if exptime is not None and float(exptime) == 0.0:
            reason = f"EXPTIME=0.0 (EXPFLAG='{expflag}')" if expflag else "EXPTIME=0.0"
            return reason
        if expflag and expflag != 'NORMAL':
            return f"EXPFLAG='{expflag}'"
        if not has_hdrlet:
            return "no HDRLET/WCS corrections (image not fully pipeline-calibrated)"
    except Exception:
        pass
    return None


def _write_selected_obsids(prod_df: pd.DataFrame, hst_dir: Path,
                            field_name: str, im_type: str,
                            failed_obsids: dict[str, str] | None = None) -> None:
    """Save individual FLC image obs_ids to a JSON manifest.

    These are the per-exposure obs_ids (e.g. 'jbjm03llq') that match the
    directory names under mastDownload/ and the image names used by BP3M —
    not the parent observation obsids.

    failed_obsids, if given, maps obs_id → reason string for images that were
    downloaded but must be skipped (e.g. EXPTIME=0 failed observations).
    These are written to a separate {field}_failed_obsids.json manifest and
    excluded from the selected manifest.
    """
    flc_sub = im_type[1:].upper()   # '_flc' → 'FLC'
    flc_rows = prod_df[prod_df['productSubGroupDescription'] == flc_sub]
    all_obsids = sorted(set(flc_rows['obs_id'].astype(str)))
    bad = set(failed_obsids or {})
    obsids = [o for o in all_obsids if o not in bad]
    manifest = hst_dir / f"{field_name}_selected_obsids.json"
    manifest.write_text(json.dumps(obsids, indent=2))
    failed_manifest = hst_dir / f"{field_name}_failed_obsids.json"
    if failed_obsids:
        failed_manifest.write_text(json.dumps(failed_obsids, indent=2))
    elif failed_manifest.exists():
        failed_manifest.unlink()


def _print_obs_table(obs_df: pd.DataFrame) -> None:
    """Print the observations table with field_id, proposal_id, n_gaia, n_exp."""
    display_cols = {
        'field_id':      'ID',
        'proposal_id':   'PropID',
        'obs_time':      'Date',
        'instrument_name': 'Instrument',
        'filters':       'Filter',
        'i_exptime':     'ExpTime(s)',
        'n_exp':         'N_exp',
        't_baseline':    'Baseline(yr)',
        'n_gaia':        'N_Gaia',
    }
    present = {k: v for k, v in display_cols.items() if k in obs_df.columns}
    disp = obs_df[list(present.keys())].rename(columns=present).copy()
    # Format floats nicely
    for col in ('ExpTime(s)', 'Baseline(yr)'):
        if col in disp.columns:
            disp[col] = disp[col].map(lambda x: f'{x:.1f}' if pd.notna(x) else '?')
    if 'N_Gaia' in disp.columns:
        disp['N_Gaia'] = disp['N_Gaia'].map(lambda x: str(x) if x >= 0 else '?')
    print(disp.to_string(index=False))
