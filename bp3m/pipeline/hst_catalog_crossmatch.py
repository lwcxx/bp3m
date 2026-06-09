#!/usr/bin/env python3
"""
hst_catalog_crossmatch.py  —  Cross-match ALL HST sources between images.

Uses the BP3M alignment solution to project every detected HST source to
RA,Dec with full uncertainty propagation, then cross-matches sources across
images of the same filter (and later across filters), extending the catalog
to much fainter magnitudes than the Gaia-limited pipeline.

Three-phase algorithm
---------------------
1. Within-filter  : match sources across images using ZP-corrected magnitudes
                    and a position/PM consistency check (<= max_pm_masyr).
                    Produces a per-filter master catalog with fitted PM + position.
2. Cross-filter   : match filter master catalogs using Phase 1 positions/PMs.
3. Gaia recovery  : re-attempt Gaia matching for sources in the master catalog
                    that were not matched in the standard pipeline.

Outputs (written to <field_dir>/hst_xmatch/)
---------------------------------------------
detections_{filter}.csv      all sources in RA,Dec from every image of that filter
master_{filter}.csv          within-filter master catalog (position + PM fit)
master_combined.csv          cross-filter merged catalog
gaia_recovered.csv           newly recovered Gaia matches

Usage
-----
python -m bp3m.pipeline.hst_catalog_crossmatch \\
    --field_dir /path/to/Leo_I \\
    [--output_dir /path/to/output] \\   # default: field_dir/hst_xmatch
    [--max_pm_masyr 100] \\
    [--match_n_sigma 5] \\
    [--mag_n_sigma 3.0] \\
    [--mag_floor 0.10] \\
    [--min_detections 2]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time as AstropyTime
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

# ── bp3m imports ──────────────────────────────────────────────────────────────
try:
    from bp3m.astro_utils import (
        build_X_matrix,
        build_U_matrix,
        get_tele_position,
        get_parallax_factors,
        plane_project,
        plane_project_jacobian,
        compute_poly_jacobian,
        hst_position_cov,
        n_r_from_poly_order,
        michalik_sigma_plx_prior,
        DEG2RAD,
        RAD2MAS,
        GAIA_SYS_DICT,
    )
    from bp3m.coords import plane_project_inverse
except ImportError as exc:
    sys.exit(f"Cannot import bp3m: {exc}\nEnsure bp3m is installed (pip install bp3m).")

# ── Constants ─────────────────────────────────────────────────────────────────

_AMP_Y_SPLITS = {
    ('ACS',  'WFC'):  2048,
    ('WFC3', 'UVIS'): 2051,
    ('WFC3', 'IR'):   512,
}

MJD_YR = 365.25       # days per year
_MJD_J2000  = 51544.5                   # J2000.0 in MJD
_MJD0_YR    = _MJD_J2000 / MJD_YR      # ≈ 141.12  (J2000 in epoch_yr units)

# Centering pivot used by BP3M solver (hardcoded in solver.py line 308)
_SOLVER_XO = _SOLVER_YO = 2048.0


# ── Helpers: instrument / header ─────────────────────────────────────────────

def _get_filter(h0) -> str:
    filt = h0.get('FILTER', '')
    if filt:
        return str(filt).strip()
    f1 = str(h0.get('FILTER1', '')).strip()
    f2 = str(h0.get('FILTER2', '')).strip()
    if 'CLEAR' in f1.upper():
        return f2
    return f1


def _get_y_split(instrument: str, detector: str) -> int:
    return _AMP_Y_SPLITS.get((instrument.upper(), detector.upper()), 2048)


# ── Helpers: coordinate transform ────────────────────────────────────────────

def _project_to_radec(
    x_gdc: np.ndarray,
    y_gdc: np.ndarray,
    cov_xx: np.ndarray,
    cov_yy: np.ndarray,
    cov_xy: np.ndarray,
    r_j: np.ndarray,
    C_r_j: np.ndarray,
    ra0: float,
    dec0: float,
    pscale: float,
    poly_order: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert GDC-corrected HST pixel positions to (RA, Dec) with full
    uncertainty propagation (r-vector uncertainty + PSF measurement noise).

    Uses the same centering (Xo=Yo=2048) and math as BP3M's hst_to_radec().

    Returns
    -------
    ra, dec          : degrees
    sigma_ra, sigma_dec : mas  (1-sigma, marginalised)
    cov_rd           : mas²   (RA–Dec off-diagonal covariance)
    """
    n = len(x_gdc)
    n_r = len(r_j)
    X_c = x_gdc - _SOLVER_XO
    Y_c = y_gdc - _SOLVER_YO

    # Build design matrices (n, 2, n_r) — vectorised; tangent-point derivs are 0 here
    X_mats = np.zeros((n, 2, n_r))
    X_mats[:, 0, 0] = X_c;  X_mats[:, 0, 1] = Y_c;  X_mats[:, 0, 4] = 1.0
    X_mats[:, 1, 2] = X_c;  X_mats[:, 1, 3] = Y_c;  X_mats[:, 1, 5] = 1.0
    # columns 6,7 (tangent-point derivatives) stay 0
    _S = 2048.0
    _col = 8
    for _deg in range(2, poly_order + 1):
        _sc = _S ** (_deg - 1)
        for _j in range(_deg + 1):
            X_mats[:, 0, _col] = X_c ** (_deg - _j) * Y_c ** _j / _sc
            _col += 1
        for _j in range(_deg + 1):
            X_mats[:, 1, _col] = X_c ** (_deg - _j) * Y_c ** _j / _sc
            _col += 1

    # Predicted pseudo-image position (n, 2) in pixels
    x_gaia = np.einsum('nkl,l->nk', X_mats, r_j)

    # Convert to RA, Dec
    ra, dec = plane_project_inverse(x_gaia[:, 0], x_gaia[:, 1], ra0, dec0, pscale)

    # Jacobian of (x_pix, y_pix) w.r.t. (RA_mas, Dec_mas) → shape (n, 2, 2)
    J     = plane_project_jacobian(ra, dec, ra0, dec0, pscale)
    J_inv = np.linalg.inv(J)

    # C_xy from r-vector uncertainty: X_mat @ C_r_j @ X_mat^T  (n, 2, 2) pix²
    XCr  = np.einsum('nij,jk->nik', X_mats, C_r_j)
    C_xy = np.einsum('nik,njk->nij', XCr, X_mats)

    # Add HST PSF measurement noise (propagated through the transformation) — vectorised
    sig_x = np.sqrt(np.maximum(cov_xx, 0.))
    sig_y = np.sqrt(np.maximum(cov_yy, 0.))
    denom = sig_x * sig_y
    corr  = np.where(denom > 0., cov_xy / denom, 0.)
    corr  = np.clip(corr, -0.9999, 0.9999)
    J_trans = compute_poly_jacobian(r_j, X_c, Y_c, poly_order)   # (n, 2, 2)
    _FLOOR2 = 0.001 ** 2
    C_hst_all = np.zeros((n, 2, 2))
    C_hst_all[:, 0, 0] = sig_x ** 2 + _FLOOR2
    C_hst_all[:, 1, 1] = sig_y ** 2 + _FLOOR2
    _xy_cov             = sig_x * sig_y * corr
    C_hst_all[:, 0, 1]  = _xy_cov
    C_hst_all[:, 1, 0]  = _xy_cov
    _JC   = np.einsum('nij,njk->nik', J_trans, C_hst_all)
    C_xy += np.einsum('nij,nkj->nik', _JC, J_trans)

    # Propagate to RA,Dec:  J^{-1} @ C_xy @ J^{-T}  [mas²]
    JiCxy      = np.einsum('nij,njk->nik', J_inv, C_xy)
    cov_radec  = np.einsum('nij,nkj->nik', JiCxy, J_inv)          # (n, 2, 2) mas²
    sigma_ra   = np.sqrt(np.maximum(cov_radec[:, 0, 0], 0.))      # mas
    sigma_dec  = np.sqrt(np.maximum(cov_radec[:, 1, 1], 0.))      # mas
    cov_rd     = cov_radec[:, 0, 1]                                # mas²

    return ra, dec, sigma_ra, sigma_dec, cov_rd


# ── Phase 0: load all detections and project to RA,Dec ───────────────────────

def _load_all_detections(field_dir: Path,
                         bp3m_results_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """
    For every processed image, load the full PSF catalog and project all
    detected sources to (RA, Dec) using the BP3M posterior transformation.

    Returns a DataFrame with one row per source-detection (source × image),
    or None if the BP3M results directory does not exist.

    Key columns
    -----------
    sub_name       : BP3M image name (e.g. 'j9gz01orq_hi')
    base_name      : base image name ('j9gz01orq')
    filter, instrument, detector
    epoch_mjd      : mid-exposure MJD
    catalog_index  : row index in *_flc_catalog.fits
    ra, dec        : degrees
    sigma_ra, sigma_dec : mas
    cov_ra_dec     : mas²
    mag_gdc, mag_err_gdc
    mag_zp         : calibrated magnitude = mag_st_gdc + cross_image_zp (or just mag_st_gdc if no pre-ZP file)
    is_star_candidate : bool
    qfit, chi2
    x_gdc, y_gdc  : GDC pixel positions
    has_gaia_match : bool
    gaia_source_id : int64 (0 if no match)
    """
    bp3m_dir = Path(bp3m_results_dir) if bp3m_results_dir is not None else field_dir / 'BP3M_results'
    hst_root = field_dir / 'HST' / 'mastDownload' / 'HST'

    if not bp3m_dir.exists():
        print(f"  Error: BP3M_results not found at {bp3m_dir}")
        return None

    img_csv   = bp3m_dir / 'image_transformations.csv'
    c_r_path  = bp3m_dir / 'C_r.npy'
    if not img_csv.exists() or not c_r_path.exists():
        print("  Error: image_transformations.csv or C_r.npy missing in BP3M_results/")
        return None

    transform_df = pd.read_csv(img_csv)
    C_r          = np.load(c_r_path)
    n_sub        = len(transform_df)
    n_r          = C_r.shape[0] // n_sub
    poly_order   = _infer_poly_order(n_r)

    # Use run_config.json if available (written by run_alignment._save_results)
    run_cfg_path = bp3m_dir / 'run_config.json'
    if run_cfg_path.exists():
        import json as _json
        with open(run_cfg_path) as _f:
            run_cfg = _json.load(_f)
        # Override poly_order from config if present
        if 'poly_order' in run_cfg:
            poly_order = int(run_cfg['poly_order'])
        # Reorder transform_df to match C_r block order if image_names is stored
        if 'image_names' in run_cfg:
            saved_names = run_cfg['image_names']
            if set(saved_names) == set(transform_df['image_name'].tolist()):
                transform_df = transform_df.set_index('image_name').loc[saved_names].reset_index()

    print(f"  {n_sub} sub-images, N_R={n_r} (poly_order={poly_order})")

    # Pre-normalise mag_zp using cross_image_zp from magnitude_zp_offsets.csv.
    # These ZPs are computed from Gaia-matched bright sources (reliable) and put
    # all images on the same photometric scale before Phase 3.  Phase 3 then
    # measures only small residual ZPs rather than large spurious offsets that
    # arise when matching across long time baselines using all faint sources.
    # Convention: mag_norm = mag_st_gdc + cross_image_zp  →  add (not subtract).
    zp_df = _load_zp_offsets(field_dir)
    zp_legacy: dict[str, float] = {}
    _apply_pre_zp = False
    if zp_df is not None and 'cross_image_zp' in zp_df.columns:
        for _, row in zp_df.iterrows():
            img_name = str(row['image'])
            zp_val   = float(row['cross_image_zp'])
            zp_legacy[img_name] = zp_val
        if zp_legacy:
            _apply_pre_zp = True
            n_zp = sum(1 for v in zp_legacy.values() if abs(v) > 0.2)
            print(f"  Pre-ZP from magnitude_zp_offsets.csv: {len(zp_legacy)} images, "
                  f"{n_zp} with |ZP| > 0.2 mag")

    # Load alpha (chi2 inflation) factors per sub-image from image_transformations.csv
    # alpha^2 inflates HST position covariances to account for systematic residuals
    alpha_lookup: dict[str, float] = {}
    if 'alpha' in transform_df.columns:
        for _, row in transform_df.iterrows():
            alpha_lookup[row['image_name']] = float(row['alpha'])
        print(f"  Loaded alpha inflation factors for {len(alpha_lookup)} sub-images")

    # r-vectors per sub-image
    r_vecs = {}
    for j, row in enumerate(transform_df.itertuples()):
        cs         = j * n_r
        sub_name   = row.image_name
        r_j        = np.array([getattr(row, p) for p in ('a', 'b', 'c', 'd', 'w', 'z',
                                                           'delta_ra0_mas', 'delta_dec0_mas')])
        C_r_j      = C_r[cs:cs + n_r, cs:cs + n_r]
        r_vecs[sub_name] = (r_j, C_r_j)

    # ── Per-sub-image processing ──────────────────────────────────────────────
    # Closures over r_vecs, zp_legacy, alpha_lookup, hst_root, n_r, poly_order —
    # all read-only.  FITS I/O (astropy) and numpy ops release the GIL so threads
    # give real parallelism even for CPU-heavy projection work.
    def _process_one_sub_image(j_trow: tuple) -> Optional[pd.DataFrame]:
        _, trow = j_trow
        sub_name = trow.image_name
        if sub_name.endswith('_hi'):
            base   = sub_name[:-3]
            suffix = '_hi'
        elif sub_name.endswith('_lo'):
            base   = sub_name[:-3]
            suffix = '_lo'
        else:
            base   = sub_name
            suffix = None

        img_dir  = hst_root / base
        cat_path = img_dir / f'{base}_flc_catalog.fits'
        tran_csv = img_dir / 'transformation.csv'

        if not cat_path.exists() or not tran_csv.exists():
            return None

        # Image metadata from transformation.csv
        try:
            tdf    = pd.read_csv(tran_csv).set_index('parameter')['value']
            ra0    = float(tdf['ra_cen'])
            dec0   = float(tdf['dec_cen'])
            pscale = float(tdf['pixel_scale']) * 1000.0   # mas/pix
        except Exception as exc:
            print(f"  Warning: skipping {base} (transformation.csv error): {exc}")
            return None

        # Instrument/detector/epoch from FITS header
        flt = instrument = detector = 'UNKNOWN'
        epoch_mjd = np.nan
        y_split   = 2048
        flc_path  = img_dir / f'{base}_flc.fits'
        if flc_path.exists():
            try:
                with fits.open(flc_path, memmap=False) as hdu:
                    h0         = hdu[0].header
                    instrument = str(h0.get('INSTRUME', 'UNKNOWN')).strip()
                    detector   = str(h0.get('DETECTOR', 'UNKNOWN')).strip()
                    flt        = _get_filter(h0)
                    expstart   = float(h0.get('EXPSTART', 0))
                    expend     = float(h0.get('EXPEND',   expstart))
                    epoch_mjd  = 0.5 * (expstart + expend)
                    y_split    = _get_y_split(instrument, detector)
            except Exception:
                pass

        # Load PSF catalog
        try:
            with fits.open(cat_path, memmap=False) as hdu:
                tbl = hdu[1].data
        except Exception as exc:
            print(f"  Warning: cannot open catalog {cat_path}: {exc}")
            return None

        # Build numpy arrays from catalog table
        cat_y_raw  = np.asarray(tbl['y'],            float)
        cat_xgdc   = np.asarray(tbl['x_gdc'],        float)
        cat_ygdc   = np.asarray(tbl['y_gdc'],        float)
        cat_cov_xx = np.asarray(tbl['cov_xx_gdc'],   float)
        cat_cov_yy = np.asarray(tbl['cov_yy_gdc'],   float)
        cat_cov_xy = np.asarray(tbl['cov_xy_gdc'],   float)
        cat_mag    = np.asarray(tbl['mag_gdc'],       float)
        cat_magerr = np.asarray(tbl['mag_err_gdc'],  float)
        # Require mag_st_gdc (py1pass STMAG calibrated via PHOTFLAM/EXPTIME + GDC
        # pixel-area correction).  Catalogs without it were produced by an older
        # version of py1pass and will produce large spurious inter-image ZP offsets
        # during within-filter matching.  Delete the stale catalog so py1pass will
        # re-run on the next pipeline invocation.
        if 'mag_st_gdc' not in tbl.dtype.names:
            print(f"  ERROR: {sub_name}: catalog missing 'mag_st_gdc' column — "
                  f"stale py1pass output. Deleting catalog so it will be re-fitted.")
            try:
                cat_path.unlink()
                sidecar = cat_path.parent / 'psf_params.json'
                if sidecar.exists():
                    sidecar.unlink()
            except Exception as _del_exc:
                print(f"    (deletion failed: {_del_exc})")
            return None
        cat_mag_cal = np.asarray(tbl['mag_st_gdc'], float)
        cat_qfit = np.asarray(tbl['qfit'],  float)
        cat_chi2 = np.asarray(tbl['chi2'],  float)
        cat_nsat = np.asarray(tbl['n_sat'], int)
        cat_star = np.asarray(tbl['is_star_candidate'], bool) \
                   if 'is_star_candidate' in tbl.dtype.names else \
                   np.ones(len(tbl), bool)

        # Select sources for this sub-image (hi/lo chip split)
        if suffix == '_hi':
            mask = cat_y_raw > y_split
        elif suffix == '_lo':
            mask = cat_y_raw <= y_split
        else:
            mask = np.ones(len(cat_y_raw), bool)

        mask &= (cat_nsat == 0)
        mask &= (cat_qfit < 2.0)
        mask &= np.isfinite(cat_mag)

        idx = np.where(mask)[0]
        if len(idx) == 0:
            return None

        # Gaia match lookup (hst_index → gaia_source_id)
        gaia_match = _load_gaia_match_lookup(img_dir)

        r_j, C_r_j = r_vecs[sub_name]

        # Alpha inflation: scale HST covariances by alpha^2 before projecting.
        alpha  = alpha_lookup.get(sub_name, 1.0)
        alpha2 = alpha ** 2

        # Project to RA,Dec
        try:
            ra, dec, sigma_ra, sigma_dec, cov_rd = _project_to_radec(
                cat_xgdc[idx], cat_ygdc[idx],
                cat_cov_xx[idx] * alpha2, cat_cov_yy[idx] * alpha2,
                cat_cov_xy[idx] * alpha2,
                r_j, C_r_j, ra0, dec0, pscale, poly_order=poly_order,
            )
        except Exception as exc:
            print(f"  Warning: projection failed for {sub_name}: {exc}")
            return None

        # Add cross_image_zp (from magnitude_zp_offsets.csv) to normalise to the
        # common reference scale before Phase 3 within-filter matching.
        pre_zp_corr = zp_legacy.get(base, 0.0) if _apply_pre_zp else 0.0

        # Hard cap: drop sources with degenerate covariances (failed PSF fits)
        _good = (sigma_ra < 50.) & (sigma_dec < 50.)   # 50 mas ≈ 1 px
        idx      = idx[_good]
        ra       = ra[_good]
        dec      = dec[_good]
        sigma_ra = sigma_ra[_good]
        sigma_dec= sigma_dec[_good]
        cov_rd   = cov_rd[_good]

        if len(idx) == 0:
            return None

        # Build per-image DataFrame from arrays — no per-source Python loop
        gaia_ids = np.array([gaia_match.get(int(ci), 0) for ci in idx],
                            dtype=np.int64)
        return pd.DataFrame({
            'sub_name':          sub_name,
            'base_name':         base,
            'filter':            flt,
            'instrument':        instrument,
            'detector':          detector,
            'epoch_mjd':         epoch_mjd,
            'catalog_index':     idx.astype(np.int64),
            'ra':                ra,
            'dec':               dec,
            'sigma_ra':          sigma_ra,
            'sigma_dec':         sigma_dec,
            'cov_ra_dec':        cov_rd,
            'mag_gdc':           cat_mag[idx],
            'mag_err_gdc':       cat_magerr[idx],
            'mag_zp':            cat_mag_cal[idx] + pre_zp_corr,
            'alpha':             float(alpha),
            'is_star_candidate': cat_star[idx],
            'qfit':              cat_qfit[idx],
            'chi2':              cat_chi2[idx],
            'x_gdc':             cat_xgdc[idx],
            'y_gdc':             cat_ygdc[idx],
            'cov_xx_raw':        cat_cov_xx[idx],   # pre-alpha, py1pass chi2-scaled
            'cov_yy_raw':        cat_cov_yy[idx],
            'cov_xy_raw':        cat_cov_xy[idx],
            'has_gaia_match':    gaia_ids != 0,
            'gaia_source_id':    gaia_ids,
        })

    # Dispatch across sub-images with threads — FITS I/O and numpy projection
    # both release the GIL; up to 16 workers for I/O-bound fields.
    jobs     = list(enumerate(transform_df.itertuples()))
    n_workers = min(16, os.cpu_count() or 1)
    if n_workers > 1 and len(jobs) >= 4:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            img_dfs = list(executor.map(_process_one_sub_image, jobs))
    else:
        img_dfs = [_process_one_sub_image(jt) for jt in jobs]

    per_image_dfs = [df for df in img_dfs if df is not None]
    if not per_image_dfs:
        return None
    result = pd.concat(per_image_dfs, ignore_index=True)
    result.attrs['pre_zp_applied'] = _apply_pre_zp
    return result


def _infer_poly_order(n_r: int) -> int:
    """Infer polynomial order from N_R (parameters per image)."""
    for p in range(1, 6):
        if n_r_from_poly_order(p) == n_r:
            return p
    raise ValueError(f"Cannot infer poly_order from N_R={n_r}")


def _load_zp_offsets(field_dir: Path) -> Optional[pd.DataFrame]:
    """Load magnitude_zp_offsets.csv if present."""
    path = field_dir / 'magnitude_zp_offsets.csv'
    if path.exists():
        return pd.read_csv(path)
    return None


def _compute_gaia_zp_per_image(hst_root: Path, image_names: list[str]) -> dict[str, float]:
    """
    Compute per-image ZP = median(gaia_gmag − hst_mag_gdc) from matched_gaia.csv.

    Returns base_name → ZP offset (to be added to hst_mag_gdc to get Gaia-calibrated mag).
    Images with fewer than 5 matched stars are skipped; a per-filter median fallback is
    applied afterwards for those images.
    """
    raw_zp: dict[str, float]   = {}
    flt_map: dict[str, str]    = {}   # base → filter name
    zp_by_filter: dict[str, list[float]] = {}

    bases_seen = set()
    for nm in image_names:
        base = nm[:-3] if nm.endswith(('_hi', '_lo')) else nm
        bases_seen.add(base)

    for base in sorted(bases_seen):
        img_dir    = hst_root / base
        match_path = img_dir / 'matched_gaia.csv'
        if not match_path.exists():
            continue
        try:
            mdf = pd.read_csv(match_path)
            if 'gaia_gmag' not in mdf.columns or 'hst_mag_gdc' not in mdf.columns:
                continue
            diffs = mdf['gaia_gmag'].values - mdf['hst_mag_gdc'].values
            ok    = np.isfinite(diffs)
            if ok.sum() < 5:
                continue
            raw_zp[base] = float(np.median(diffs[ok]))
        except Exception:
            continue

        # Read filter for grouping
        flc_path = img_dir / f'{base}_flc.fits'
        if flc_path.exists():
            try:
                with fits.open(flc_path, memmap=False) as h:
                    flt = _get_filter(h[0].header)
                flt_map[base] = flt
            except Exception:
                pass

    # Build per-filter median ZP for fallback
    for base, zp in raw_zp.items():
        flt = flt_map.get(base, 'UNK')
        zp_by_filter.setdefault(flt, []).append(zp)

    fallback_zp = {flt: float(np.median(vals)) for flt, vals in zp_by_filter.items()}
    overall_fallback = float(np.median(list(raw_zp.values()))) if raw_zp else 0.0

    # Assign ZP to every base image
    result: dict[str, float] = {}
    for base in bases_seen:
        if base in raw_zp:
            result[base] = raw_zp[base]
        else:
            flt = flt_map.get(base, 'UNK')
            result[base] = fallback_zp.get(flt, overall_fallback)

    n_direct   = sum(1 for b in bases_seen if b in raw_zp)
    n_fallback = len(bases_seen) - n_direct
    print(f"  Gaia ZP: {n_direct} images direct, {n_fallback} fallback; "
          f"range [{min(result.values()):.2f}, {max(result.values()):.2f}]")
    return result


def _load_gaia_match_lookup(img_dir: Path) -> dict[int, int]:
    """Return dict {catalog_index: gaia_source_id} from matched_gaia.csv."""
    match_path = img_dir / 'matched_gaia.csv'
    if not match_path.exists():
        return {}
    try:
        # Read gaia_source_id as int64 directly to avoid float64 precision loss.
        # iterrows() in pandas 3.x converts int64 elements via float64, silently
        # corrupting 19-digit Gaia IDs (same bug as in data_loader_master.py).
        df = pd.read_csv(match_path, dtype={'gaia_source_id': np.int64})
        hi_arr  = df['hst_index'].to_numpy(dtype=np.int64)
        gid_arr = df['gaia_source_id'].to_numpy(dtype=np.int64)
        return {int(hi): int(gid) for hi, gid in zip(hi_arr, gid_arr)}
    except Exception:
        return {}


# ── Tangent-plane utilities for matching ─────────────────────────────────────

def _to_tangent_plane(ra: np.ndarray, dec: np.ndarray,
                       ra0: float, dec0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Project (ra, dec) onto a local tangent plane centred at (ra0, dec0).
    Returns (x, y) in mas.  Accurate for offsets up to ~1 degree.
    """
    dra  = (ra - ra0) * np.cos(dec0 * DEG2RAD)  # degrees
    ddec = (dec - dec0)                          # degrees
    return dra * 3.6e6, ddec * 3.6e6             # mas


def _from_tangent_plane(x_mas: np.ndarray, y_mas: np.ndarray,
                         ra0: float, dec0: float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of _to_tangent_plane."""
    ra  = ra0  + (x_mas / 3.6e6) / np.cos(dec0 * DEG2RAD)
    dec = dec0 + (y_mas / 3.6e6)
    return ra, dec


# ── Astrometric fitting per source ───────────────────────────────────────────

def _fit_astrometry(ra: np.ndarray, dec: np.ndarray,
                    sigma_ra: np.ndarray, sigma_dec: np.ndarray,
                    epochs_yr: np.ndarray,
                    ra0: float, dec0: float) -> dict:
    """
    Fit a linear proper-motion model to a set of detections:
        ra*(t)  = ra*_0  + pmra  * (t - t_ref)
        dec(t)  = dec_0  + pmdec * (t - t_ref)

    where ra* = ra * cos(dec0) in the same coordinate as sigma_ra.

    Returns dict with keys:
        ra0, dec0           : reference position (degrees)
        pmra, pmdec         : proper motion (mas/yr)
        sigma_pmra, sigma_pmdec : 1-sigma uncertainties (mas/yr)
        sigma_ra0, sigma_dec0   : reference-epoch position uncertainties (mas)
        epoch_ref           : reference epoch (yr)
        chi2_ra, chi2_dec   : chi2 per DOF of the linear fit
        n_detect            : number of detections
    """
    n = len(epochs_yr)
    t_ref = np.mean(epochs_yr)
    dt    = epochs_yr - t_ref

    # Convert (ra, dec) to local offsets in mas
    x_mas = (ra  - ra0) * np.cos(dec0 * DEG2RAD) * 3.6e6
    y_mas = (dec - dec0)                          * 3.6e6

    # Weighted least-squares: [offset_i] = [1, dt_i] @ [pos0, pm]
    # Convert t_ref (MJD/MJD_YR units) to decimal year
    epoch_ref_decyr = 2000.0 + (t_ref - _MJD0_YR)

    if n == 1:
        return {
            'ra0': float(ra[0]), 'dec0': float(dec[0]),
            'pmra': 0., 'pmdec': 0.,
            'sigma_pmra': np.nan, 'sigma_pmdec': np.nan,
            'sigma_ra0': float(sigma_ra[0]), 'sigma_dec0': float(sigma_dec[0]),
            'epoch_ref': float(epoch_ref_decyr), 'chi2_ra': np.nan, 'chi2_dec': np.nan,
            'n_detect': 1,
        }

    A   = np.column_stack([np.ones(n), dt])

    # Check if PM is constrainable (at least 2 distinct epochs)
    has_pm_dof = np.unique(dt).size >= 2

    # RA axis — use lstsq for robustness against degenerate epochs
    w_ra    = 1. / np.maximum(sigma_ra,  1e-6) ** 2
    AW_ra   = A.T * w_ra
    lhs_ra  = AW_ra @ A
    rhs_ra  = AW_ra @ x_mas
    p_ra, _, _, _ = np.linalg.lstsq(lhs_ra, rhs_ra, rcond=None)
    # Covariance: (A^T W A)^{-1} via lstsq on identity RHS
    try:
        cov_ra = np.linalg.inv(lhs_ra)
    except np.linalg.LinAlgError:
        cov_ra = np.full((2, 2), np.nan)
    res_ra  = x_mas - A @ p_ra
    chi2_ra = float(np.sum(res_ra**2 * w_ra) / max(n - 2, 1))

    # Dec axis
    w_dec   = 1. / np.maximum(sigma_dec, 1e-6) ** 2
    AW_dec  = A.T * w_dec
    lhs_dec = AW_dec @ A
    rhs_dec = AW_dec @ y_mas
    p_dec, _, _, _ = np.linalg.lstsq(lhs_dec, rhs_dec, rcond=None)
    try:
        cov_dec = np.linalg.inv(lhs_dec)
    except np.linalg.LinAlgError:
        cov_dec = np.full((2, 2), np.nan)
    res_dec = y_mas - A @ p_dec
    chi2_dec = float(np.sum(res_dec**2 * w_dec) / max(n - 2, 1))

    ra_fitted, dec_fitted = _from_tangent_plane(
        np.array([p_ra[0]]), np.array([p_dec[0]]), ra0, dec0
    )

    return {
        'ra0':         float(ra_fitted[0]),
        'dec0':        float(dec_fitted[0]),
        'pmra':        float(p_ra[1]),
        'pmdec':       float(p_dec[1]),
        'sigma_pmra':  float(np.sqrt(cov_ra[1, 1])),
        'sigma_pmdec': float(np.sqrt(cov_dec[1, 1])),
        'sigma_ra0':   float(np.sqrt(cov_ra[0, 0])),
        'sigma_dec0':  float(np.sqrt(cov_dec[0, 0])),
        'epoch_ref':   float(epoch_ref_decyr),
        'chi2_ra':     chi2_ra,
        'chi2_dec':    chi2_dec,
        'n_detect':    n,
    }


# ── Phase 0b: V1 BP3M-anchored Gaia detection ────────────────────────────────

def _phase0b_anchor_gaia_stars(
    det_df: pd.DataFrame,
    bp3m_results_dir: Path,
    search_radius_px: float = 50.0,
    n_candidates: int = 5,
    mag_n_sigma: float = 3.0,
    mag_floor: float = 0.15,
) -> pd.DataFrame:
    """
    Use V1 BP3M stellar astrometry to find every V1 Gaia-matched star in
    every HST sub-image, maximising the number of detections per star.

    For EVERY V1 Gaia-matched star (not just missing ones) and EVERY sub-image
    where that star is not yet labelled as a Gaia match, this function:
      1. Predicts the star's sky position using V1 PM + Gaia epoch propagation.
      2. Searches for up to ``n_candidates`` unmatched detections within
         ``search_radius_px`` pixels of the predicted position.
      3. Applies a colour/magnitude filter using the per-sub-image ZP offset
         estimated from existing Gaia matches (same logic as Phase 0's
         per-image Gaia matching).
      4. Assigns the best-passing candidate as a Gaia match.

    This ensures that stars detected in only 1 image per filter (which would
    be dropped by Phase 1's min_detections cut) survive, and that stars already
    matched in some images also get their detections in the remaining images.
    Particularly valuable for 2p Gaia stars (no Gaia PM) whose positions can
    now be propagated across epochs using V1 BP3M PM estimates.

    Returns an updated copy of det_df.
    """
    astrom_path = bp3m_results_dir / 'stellar_astrometry.csv'
    xform_path  = bp3m_results_dir / 'image_transformations.csv'

    if not astrom_path.exists() or not xform_path.exists():
        return det_df

    astrom = pd.read_csv(astrom_path, dtype={'Gaia_id': np.int64})

    # All V1 Gaia-matched stars (positive Gaia_id, has RA/Dec)
    gaia_stars = astrom[astrom['Gaia_id'].astype(np.int64) > 0].copy()
    if len(gaia_stars) == 0:
        return det_df

    GAIA_EPOCH_YR = 2016.0
    PLATE_SCALE_DEG = 50.0 / 3.6e6   # 50 mas/px → degrees/px
    search_deg = search_radius_px * PLATE_SCALE_DEG
    cos_dec_global = np.cos(np.radians(det_df['dec'].median()))

    det_df = det_df.copy()

    # ── Per-sub-image ZP offset from existing Gaia matches ───────────────────
    gaia_g_lookup = (gaia_stars.set_index('Gaia_id')['gmag'].to_dict()
                     if 'gmag' in gaia_stars.columns else {})
    zp_per_sub: dict[str, tuple[float, float]] = {}
    matched_det = det_df[det_df['has_gaia_match']].copy()
    if len(matched_det) > 0:
        matched_det['_gid'] = matched_det['gaia_source_id'].astype(np.int64)
        gaia_g_ser = (gaia_stars.set_index('Gaia_id')['gmag']
                      if 'gmag' in gaia_stars.columns else pd.Series(dtype=float))
        matched_det['_gmag'] = matched_det['_gid'].map(gaia_g_ser)
        valid = matched_det.dropna(subset=['_gmag', 'mag_zp'])
        for sub, grp in valid.groupby('sub_name'):
            diffs = grp['_gmag'].values - grp['mag_zp'].values
            if len(diffs) >= 5:
                med   = float(np.median(diffs))
                sigma = float(np.median(np.abs(diffs - med)) / 0.6745)
                zp_per_sub[sub] = (med, max(sigma, 0.1))

    # ── Per-sub-image KD-trees of UNMATCHED detections ───────────────────────
    # Only search detections not yet assigned to any Gaia source.
    # A detection already assigned to star A is not a candidate for star B.
    sub_trees: dict[str, tuple] = {}
    for sub, grp in det_df[~det_df['has_gaia_match']].groupby('sub_name'):
        if len(grp) == 0:
            continue
        ra_s  = grp['ra'].values
        dec_s = grp['dec'].values
        tree  = cKDTree(np.column_stack([ra_s * cos_dec_global, dec_s]))
        sub_trees[sub] = (tree, ra_s, dec_s, grp['mag_zp'].values,
                          grp.index.values)

    epoch_lookup = (det_df.groupby('sub_name')['epoch_mjd'].first() / MJD_YR
                    + _MJD0_YR).to_dict()

    # ── Build set of (gid, sub_name) pairs already in det_df ─────────────────
    # Skip sub-images where this star is already correctly labelled.
    already_labelled: set[tuple[int, str]] = set()
    for row in det_df[det_df['has_gaia_match']].itertuples():
        already_labelled.add((int(row.gaia_source_id), row.sub_name))

    n_already_found = len(set(
        det_df.loc[det_df['has_gaia_match'], 'gaia_source_id'].astype(np.int64)))
    n_missing = sum(1 for gid in gaia_stars['Gaia_id'].values
                    if int(gid) not in {int(g) for g in
                                        det_df.loc[det_df['has_gaia_match'],
                                                   'gaia_source_id']})
    print(f"  Phase 0b: searching all {len(gaia_stars)} V1 Gaia stars in all "
          f"{len(sub_trees)} sub-images "
          f"({n_already_found} already matched, {n_missing} not yet found)")

    n_anchored  = 0
    n_stars_improved = 0

    def _try_match(gid, ra0_g, dec0_g, pmra, pmdec, g_mag, sub_name, epoch_yr):
        """Try to find star (gid) in sub_name; return det_df row index or None."""
        if (gid, sub_name) in already_labelled or sub_name not in sub_trees:
            return None
        dt      = epoch_yr - GAIA_EPOCH_YR
        ra_pred = ra0_g + pmra  * dt / (np.cos(np.radians(dec0_g)) * 3.6e6)
        dec_pred = dec0_g + pmdec * dt / 3.6e6
        tree, ra_s, dec_s, mag_s, idx_s = sub_trees[sub_name]
        k = min(n_candidates, len(ra_s))
        dists, ii = tree.query([[ra_pred * cos_dec_global, dec_pred]], k=k)
        dists = dists[0]; ii = ii[0]
        ok = dists < search_deg
        if not ok.any():
            return None
        zp_info = zp_per_sub.get(sub_name)
        best_row = None; best_sep = np.inf
        for ki, dist_i in zip(ii[ok], dists[ok]):
            row_idx = idx_s[ki]
            if np.isfinite(g_mag) and zp_info is not None:
                zp_med, zp_sig = zp_info
                hst_mag = float(mag_s[ki]) if np.isfinite(mag_s[ki]) else np.nan
                if np.isfinite(hst_mag):
                    resid = abs(hst_mag - (g_mag - zp_med))
                    if resid > max(mag_n_sigma * zp_sig, mag_floor):
                        continue
            if dist_i < best_sep:
                best_sep = dist_i; best_row = row_idx
        return best_row

    for _, star in gaia_stars.iterrows():
        gid    = int(star['Gaia_id'])
        ra0_g  = float(star['ra'])
        dec0_g = float(star['dec'])
        pmra   = float(star.get('pmra_bp3m_cond',  star.get('pmra',  0) or 0) or 0)
        pmdec  = float(star.get('pmdec_bp3m_cond', star.get('pmdec', 0) or 0) or 0)
        g_mag  = gaia_g_lookup.get(gid, np.nan)

        star_added = 0
        for sub_name, epoch_yr in epoch_lookup.items():
            row_idx = _try_match(gid, ra0_g, dec0_g, pmra, pmdec, g_mag,
                                 sub_name, epoch_yr)
            if row_idx is None:
                continue
            det_df.iat[row_idx, det_df.columns.get_loc('has_gaia_match')] = True
            det_df.iat[row_idx, det_df.columns.get_loc('gaia_source_id')] = np.int64(gid)
            already_labelled.add((gid, sub_name))
            # Remove from tree so it can't be claimed by another star
            tree, ra_s, dec_s, mag_s, idx_s = sub_trees[sub_name]
            keep = idx_s != row_idx
            if keep.any():
                new_tree = cKDTree(np.column_stack(
                    [ra_s[keep] * cos_dec_global, dec_s[keep]]))
                sub_trees[sub_name] = (new_tree, ra_s[keep], dec_s[keep],
                                       mag_s[keep], idx_s[keep])
            else:
                del sub_trees[sub_name]
            star_added += 1
            n_anchored += 1

        if star_added > 0:
            n_stars_improved += 1

    print(f"  Phase 1: added {n_anchored} detections across "
          f"{n_stars_improved} V1 Gaia stars")
    return det_df


def _phase2_gaia_catalog_anchor(
    det_df: pd.DataFrame,
    gaia_csv: Optional[Path],
    anchor_gaia_ids: set,
    search_radius_px: float = 50.0,
    n_candidates: int = 5,
    mag_n_sigma: float = 3.0,
    mag_floor: float = 0.15,
) -> pd.DataFrame:
    """
    Phase 2: Search for Gaia catalog stars NOT covered by Phase 1 (V1 BP3M
    anchoring) in all sub-images, using Gaia catalog PM for propagation.

    ``anchor_gaia_ids`` is the set of Gaia source IDs already handled by
    Phase 1.  Only Gaia stars NOT in this set are searched here.

    This ensures that new Gaia stars (not in the original BP3M v1 run, e.g.
    faint stars recovered by the improved crossmatch or 2p stars whose PM was
    not measured in v1) also get their detections labelled before Phase 3
    (within-filter crossmatch), so they survive the min_detections cut.
    """
    if gaia_csv is None or not Path(gaia_csv).exists():
        return det_df

    try:
        gaia_df = pd.read_csv(gaia_csv)
        gaia_df.columns = [c.lower() for c in gaia_df.columns]
    except Exception:
        return det_df

    id_col = 'source_id' if 'source_id' in gaia_df.columns else 'SOURCE_ID'.lower()
    if id_col not in gaia_df.columns:
        return det_df
    gaia_df[id_col] = pd.to_numeric(gaia_df[id_col], errors='coerce').astype(np.int64)

    # Restrict to Gaia stars not already anchored by Phase 1
    already_in_det = set(
        det_df.loc[det_df['has_gaia_match'], 'gaia_source_id']
        .astype(np.int64).values
    )
    targets = gaia_df[~gaia_df[id_col].isin(anchor_gaia_ids | already_in_det)].copy()
    if len(targets) == 0:
        return det_df

    GAIA_EPOCH_YR = 2016.0
    PLATE_SCALE_DEG = 50.0 / 3.6e6
    search_deg = search_radius_px * PLATE_SCALE_DEG
    cos_dec_global = np.cos(np.radians(det_df['dec'].median()))

    # Build ZP offsets and KD-trees (same as Phase 1)
    gaia_g_lookup = gaia_df.set_index(id_col)['gmag'].to_dict() if 'gmag' in gaia_df.columns else {}
    zp_per_sub: dict[str, tuple[float, float]] = {}
    matched_det = det_df[det_df['has_gaia_match']].copy()
    if len(matched_det) > 0:
        matched_det['_gid'] = matched_det['gaia_source_id'].astype(np.int64)
        gaia_g_ser = gaia_df.set_index(id_col)['gmag'] if 'gmag' in gaia_df.columns else pd.Series(dtype=float)
        matched_det['_gmag'] = matched_det['_gid'].map(gaia_g_ser)
        valid = matched_det.dropna(subset=['_gmag', 'mag_zp'])
        for sub, grp in valid.groupby('sub_name'):
            diffs = grp['_gmag'].values - grp['mag_zp'].values
            if len(diffs) >= 5:
                med = float(np.median(diffs))
                sigma = float(np.median(np.abs(diffs - med)) / 0.6745)
                zp_per_sub[sub] = (med, max(sigma, 0.1))

    det_df = det_df.copy()
    sub_trees: dict[str, tuple] = {}
    for sub, grp in det_df[~det_df['has_gaia_match']].groupby('sub_name'):
        if len(grp) == 0:
            continue
        ra_s = grp['ra'].values; dec_s = grp['dec'].values
        tree = cKDTree(np.column_stack([ra_s * cos_dec_global, dec_s]))
        sub_trees[sub] = (tree, ra_s, dec_s, grp['mag_zp'].values, grp.index.values)

    epoch_lookup = (det_df.groupby('sub_name')['epoch_mjd'].first() / MJD_YR + _MJD0_YR).to_dict()
    already_labelled: set[tuple[int, str]] = set()
    for row in det_df[det_df['has_gaia_match']].itertuples():
        already_labelled.add((int(row.gaia_source_id), row.sub_name))

    n_anchored = 0; n_stars_improved = 0
    for _, star in targets.iterrows():
        gid = int(star[id_col])
        ra0_g = float(star.get('ra', np.nan))
        dec0_g = float(star.get('dec', np.nan))
        if not (np.isfinite(ra0_g) and np.isfinite(dec0_g)):
            continue
        pmra  = float(star.get('pmra',  0) or 0) if np.isfinite(star.get('pmra',  np.nan)) else 0.0
        pmdec = float(star.get('pmdec', 0) or 0) if np.isfinite(star.get('pmdec', np.nan)) else 0.0
        g_mag = gaia_g_lookup.get(gid, np.nan)

        star_added = 0
        for sub_name, epoch_yr in epoch_lookup.items():
            if (gid, sub_name) in already_labelled or sub_name not in sub_trees:
                continue
            dt = epoch_yr - GAIA_EPOCH_YR
            ra_pred  = ra0_g + pmra  * dt / (np.cos(np.radians(dec0_g)) * 3.6e6)
            dec_pred = dec0_g + pmdec * dt / 3.6e6
            tree, ra_s, dec_s, mag_s, idx_s = sub_trees[sub_name]
            k = min(n_candidates, len(ra_s))
            dists, ii = tree.query([[ra_pred * cos_dec_global, dec_pred]], k=k)
            dists = dists[0]; ii = ii[0]
            ok = dists < search_deg
            if not ok.any():
                continue
            zp_info = zp_per_sub.get(sub_name)
            best_row = None; best_sep = np.inf
            for ki, dist_i in zip(ii[ok], dists[ok]):
                row_idx = idx_s[ki]
                if np.isfinite(g_mag) and zp_info is not None:
                    zp_med, zp_sig = zp_info
                    hst_mag = float(mag_s[ki]) if np.isfinite(mag_s[ki]) else np.nan
                    if np.isfinite(hst_mag):
                        resid = abs(hst_mag - (g_mag - zp_med))
                        if resid > max(mag_n_sigma * zp_sig, mag_floor):
                            continue
                if dist_i < best_sep:
                    best_sep = dist_i; best_row = row_idx
            if best_row is None:
                continue
            det_df.iat[best_row, det_df.columns.get_loc('has_gaia_match')] = True
            det_df.iat[best_row, det_df.columns.get_loc('gaia_source_id')] = np.int64(gid)
            already_labelled.add((gid, sub_name))
            keep = idx_s != best_row
            if keep.any():
                sub_trees[sub_name] = (
                    cKDTree(np.column_stack([ra_s[keep]*cos_dec_global, dec_s[keep]])),
                    ra_s[keep], dec_s[keep], mag_s[keep], idx_s[keep])
            else:
                del sub_trees[sub_name]
            star_added += 1; n_anchored += 1
        if star_added > 0:
            n_stars_improved += 1

    print(f"  Phase 2: added {n_anchored} detections across "
          f"{n_stars_improved} non-V1 Gaia stars ({len(targets)} candidates)")
    return det_df


# ── Phase 3: within-filter matching ──────────────────────────────────────────

def _match_two_sets(
    ra_a: np.ndarray, dec_a: np.ndarray,
    sigma_ra_a: np.ndarray, sigma_dec_a: np.ndarray,
    mag_zp_a: np.ndarray, mag_err_a: np.ndarray,
    ra_b: np.ndarray, dec_b: np.ndarray,
    sigma_ra_b: np.ndarray, sigma_dec_b: np.ndarray,
    mag_zp_b: np.ndarray, mag_err_b: np.ndarray,
    dt_yr,                  # scalar or 1-D array (per master source)
    ra0: float, dec0: float,
    max_pm_masyr: float,
    match_n_sigma: float,
    mag_n_sigma: float  = 3.0,
    mag_floor: float    = 0.10,
    n_candidates: int   = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match sources in set A against sources in set B using a vectorised KDTree.

    dt_yr may be a scalar (all master sources share the same time gap) or a
    1-D array of length len(ra_a) giving the time gap between each master
    source's running mean epoch and the current image epoch.  The PM
    contribution to the match radius is max_pm_masyr * dt_yr per source.

    Magnitude cut: |Δmag| < max(mag_n_sigma × √(σ_a² + σ_b²), mag_floor).
    With mag_st_gdc the ZP is identical across images in the same filter, so
    the combined photometric uncertainty is the only relevant scale.  The floor
    prevents rejection of perfectly calibrated bright-star pairs due to
    floating-point noise.

    n_candidates controls how many nearest neighbours are fetched per source in A.
    Using n_candidates > 1 means the nearest neighbour in B can be rejected on
    the magnitude cut without losing the source — the next-closest surviving
    candidate is used instead.  One-to-one assignment is resolved greedily by
    separation across all (a, b) candidate pairs that pass both cuts.

    Returns
    -------
    idx_a, idx_b : matched indices into A and B (one-to-one, shortest separation first)
    """
    if len(ra_a) == 0 or len(ra_b) == 0:
        return np.empty(0, int), np.empty(0, int)

    # Project to tangent plane (mas)
    xa, ya = _to_tangent_plane(ra_a, dec_a, ra0, dec0)
    xb, yb = _to_tangent_plane(ra_b, dec_b, ra0, dec0)

    # pm_contrib is scalar or per-source array depending on dt_yr type
    pm_contrib   = np.abs(max_pm_masyr * np.asarray(dt_yr, dtype=float))
    sigma_med_b  = float(np.median(np.sqrt(sigma_ra_b**2 + sigma_dec_b**2) / np.sqrt(2)))

    # Per-source match radius (using median sigma_b for vectorised query)
    sigma_a_eff  = np.sqrt(0.5 * (sigma_ra_a**2 + sigma_dec_a**2))
    r_per_a      = match_n_sigma * np.sqrt(sigma_a_eff**2 + sigma_med_b**2 + pm_contrib**2)
    r_global     = float(r_per_a.max())   # worst-case for KDTree

    tree_b = cKDTree(np.column_stack([xb, yb]))

    # Query up to n_candidates nearest neighbours per source in A
    k_eff = min(n_candidates, len(xb))
    raw_d, raw_i = tree_b.query(
        np.column_stack([xa, ya]), k=k_eff,
        distance_upper_bound=r_global,
    )
    # Normalise to 2-D: k=1 returns 1-D arrays
    if k_eff == 1:
        raw_d = raw_d[:, np.newaxis]
        raw_i = raw_i[:, np.newaxis]

    # Flatten to candidate pairs (one row per a-source × neighbour slot)
    n_a     = len(xa)
    ca_flat = np.repeat(np.arange(n_a), k_eff)
    cb_flat = raw_i.ravel()
    d_flat  = raw_d.ravel()
    # KDTree fills empty slots with index=len(xb) and dist=inf
    valid = (d_flat < r_global) & (cb_flat < len(xb))
    ca   = ca_flat[valid]
    cb   = cb_flat[valid]
    seps = d_flat[valid]

    if len(ca) == 0:
        return np.empty(0, int), np.empty(0, int)

    # Per-pair position cut using actual per-source sigma_b
    pm_contrib_pair = pm_contrib[ca] if pm_contrib.ndim > 0 else pm_contrib
    sigma_b_eff  = np.sqrt(0.5 * (sigma_ra_b[cb]**2 + sigma_dec_b[cb]**2))
    r_pair       = match_n_sigma * np.sqrt(sigma_a_eff[ca]**2 + sigma_b_eff**2 + pm_contrib_pair**2)
    pos_ok       = seps < r_pair

    # Magnitude cut: N-sigma using per-source photometric errors, with a floor
    dmag      = np.abs(mag_zp_b[cb] - mag_zp_a[ca])
    _comb_sig = np.sqrt(mag_err_a[ca]**2 + mag_err_b[cb]**2)
    mag_ok    = dmag < np.maximum(mag_n_sigma * _comb_sig, mag_floor)

    keep = pos_ok & mag_ok
    ca   = ca[keep]
    cb   = cb[keep]
    seps = seps[keep]

    if len(ca) == 0:
        return np.empty(0, int), np.empty(0, int)

    # One-to-one: greedy by separation across all surviving candidate pairs.
    # With n_candidates > 1, a source in A may appear multiple times; the
    # greedy sort ensures the shortest-separation pair is always preferred,
    # and the next candidate is used if the preferred one is already claimed.
    order    = np.argsort(seps)
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched_a: list[int] = []
    matched_b: list[int] = []
    for oi in order:
        i, j = int(ca[oi]), int(cb[oi])
        if i not in used_a and j not in used_b:
            matched_a.append(i)
            matched_b.append(j)
            used_a.add(i)
            used_b.add(j)

    return np.array(matched_a, int), np.array(matched_b, int)


def _order_images_greedy(
    candidates: list[str],
    fdf: pd.DataFrame,
    img_epochs: "pd.Series",
    seed_epoch_mjd: float,
    seed_ra: np.ndarray,
    seed_dec: np.ndarray,
    overlap_scale_arcmin: float = 4.0,
    time_scale_yr: float = 5.0,
) -> list[str]:
    """
    Greedy ordering that maximises sky overlap with the growing master footprint
    while preferring images close in time to the seed.

    Uses image *centers* rather than individual sources for all distance
    calculations, making this O(N_images²) instead of O(N_images² × N_sources).
    For 158 images this is ~25 K distance evaluations total — negligible.

    At each step the next image scores highest on::

        score = overlap_weight(min_dist_to_footprint) * time_weight(Δt)

    where overlap_weight decays linearly from 1 → 0 as center-to-footprint
    distance grows from 0 → overlap_scale.  The time_weight = exp(-|Δt|/τ)
    with τ = time_scale_yr.  A small epsilon on time_weight ensures
    non-overlapping images are still ranked by time proximity.

    Parameters
    ----------
    candidates           : sub-image names to order (seed already excluded)
    fdf                  : full detection DataFrame for this filter
    img_epochs           : Series mapping sub_name → epoch_mjd
    seed_epoch_mjd       : MJD of the seed image
    seed_ra/dec          : RA/Dec of seed image sources (degrees); center used
    overlap_scale_arcmin : distance at which overlap_weight → 0 (arcmin).
                           Set to roughly the image FOV diameter.
    time_scale_yr        : e-folding time for the time-weight term (years)
    """
    if not candidates:
        return []
    if len(candidates) == 1:
        return list(candidates)

    overlap_scale_deg = overlap_scale_arcmin / 60.0

    sub_name_arr = fdf['sub_name'].values
    ra_arr       = fdf['ra'].values
    dec_arr      = fdf['dec'].values

    # Image centers (median RA/Dec per sub-image) — one point per image.
    all_names = [*candidates, '_seed_']
    ctr_ra  = {s: float(np.median(ra_arr[sub_name_arr  == s])) for s in candidates}
    ctr_dec = {s: float(np.median(dec_arr[sub_name_arr == s])) for s in candidates}
    seed_ctr_ra  = float(np.median(seed_ra))
    seed_ctr_dec = float(np.median(seed_dec))

    # cos(dec) scale factor (fixed at field center; accurate enough for ordering).
    cos_dec = np.cos(np.deg2rad(seed_ctr_dec))

    def _center_dist(s, sel_ctrs_ra, sel_ctrs_dec):
        """Minimum angular distance from image s center to any selected center."""
        dra  = (ctr_ra[s]  - sel_ctrs_ra)  * cos_dec
        ddec =  ctr_dec[s] - sel_ctrs_dec
        return float(np.min(np.hypot(dra, ddec)))

    # Initialise footprint with seed center.
    sel_ctrs_ra  = np.array([seed_ctr_ra])
    sel_ctrs_dec = np.array([seed_ctr_dec])

    remaining = list(candidates)
    ordered   = []

    while remaining:
        if len(remaining) == 1:
            ordered.append(remaining[0])
            break

        best_sub, best_score = None, -1.0
        for s in remaining:
            min_dist = _center_dist(s, sel_ctrs_ra, sel_ctrs_dec)
            overlap_w = max(0.0, 1.0 - min_dist / overlap_scale_deg)

            dt_yr = abs(float(img_epochs[s]) - seed_epoch_mjd) / 365.25
            time_w = np.exp(-dt_yr / time_scale_yr)

            # Overlap-dominant when images share sky; time-dominant otherwise.
            score = (overlap_w + 1e-3) * time_w
            if score > best_score:
                best_score = score
                best_sub   = s

        ordered.append(best_sub)
        remaining.remove(best_sub)

        # Grow footprint with the selected image's center.
        sel_ctrs_ra  = np.append(sel_ctrs_ra,  ctr_ra[best_sub])
        sel_ctrs_dec = np.append(sel_ctrs_dec, ctr_dec[best_sub])

    return ordered


def _within_filter_match(
    det_df: pd.DataFrame,
    max_pm_masyr: float   = 100.,
    match_n_sigma: float  = 5.,
    mag_n_sigma: float    = 3.0,
    mag_floor: float      = 0.10,
    min_detections: int   = 2,
    mag_outlier_sigma: float = 5.0,
    mag_outlier_floor: float = 0.5,
    nonstar_mag_relax: float = 1.5,
    pre_zp_applied: bool  = False,
) -> dict[str, pd.DataFrame]:
    """
    Build a per-filter master catalog by iteratively matching sources across
    images.  Returns {filter: master_DataFrame}.

    Two-pass matching per sub-image
        Pass 1: match is_star_candidate=True detections against the current
        master using the sigma-based mag cut (mag_n_sigma × combined σ).
        Stars get priority on claiming master entries, ensuring the master
        is anchored to reliable point-source photometry.
        Pass 2: match non-star-candidate detections against the remaining
        (unmatched) master entries using mag_n_sigma × nonstar_mag_relax,
        then seed any still-unmatched detections as new entries.

    mag_outlier_sigma / mag_outlier_floor
        After grouping, reject per-detection magnitude outliers.  A detection
        is flagged if |mag_i − weighted_mean_mag| > max(mag_outlier_sigma × MAD,
        mag_outlier_floor).  Only applied when a group has ≥ 3 detections and at
        least min_detections would survive.  Astrometry is re-fit on the pruned set.

    Each detection receives a 'source_id' label after matching; statistics
    are then computed per source_id with pandas groupby (fast).
    """
    ra0  = float(det_df['ra'].mean())
    dec0 = float(det_df['dec'].mean())

    results: dict[str, pd.DataFrame] = {}

    def _process_one_filter(filt: str, fdf_raw: pd.DataFrame) -> Optional[pd.DataFrame]:
        # Reset to clean 0-based index (all position lookups use iloc)
        fdf = fdf_raw.reset_index(drop=True).copy()

        # Merge split-CCD halves (_hi/_lo) into a single sub_name so that both
        # chips of the same exposure are treated as one image throughout: seed
        # selection, greedy ordering, ZP estimation, and master matching.
        # Original names are preserved in orig_sub_name for the output catalogue
        # (where they must match the r-vector keys used by BP3M).
        fdf['orig_sub_name'] = fdf['sub_name']
        if fdf['sub_name'].str.endswith(('_hi', '_lo')).any():
            fdf['sub_name'] = fdf['sub_name'].apply(
                lambda s: s[:-3] if s.endswith(('_hi', '_lo')) else s
            )
        n_chips  = fdf['orig_sub_name'].nunique()
        n_images = fdf['sub_name'].nunique()
        _chip_note = f' ({n_chips} chips)' if n_chips != n_images else ''
        print(f"  Filter {filt}: {len(fdf)} detections in "
              f"{n_images} images{_chip_note}")

        # Sort sub-images by epoch
        img_epochs = (fdf.groupby('sub_name')['epoch_mjd'].first()
                      .sort_values())
        sub_names  = img_epochs.index.tolist()
        if len(sub_names) == 0:
            return None

        # Assign unique source_id per detection row; -1 = unassigned
        source_ids = np.full(len(fdf), -1, dtype=np.int64)
        is_star_arr = fdf['is_star_candidate'].values.astype(bool) \
                      if 'is_star_candidate' in fdf.columns \
                      else np.ones(len(fdf), bool)

        # Start with the sub-image that has the most sources as reference.
        # Seed star candidates first so master entries 0..n_star-1 are anchored
        # to reliable point sources; non-stars follow.
        n_per   = fdf.groupby('sub_name').size()
        ref_sub = n_per.idxmax()

        ref_mask         = fdf['sub_name'].values == ref_sub
        ref_star_pos     = np.where(ref_mask &  is_star_arr)[0]
        ref_nonstar_pos  = np.where(ref_mask & ~is_star_arr)[0]
        if len(ref_star_pos) == 0:
            ref_star_pos = np.where(ref_mask)[0]   # fallback if no stars
            ref_nonstar_pos = np.empty(0, int)

        n_ref   = len(ref_star_pos)
        source_ids[ref_star_pos]    = np.arange(n_ref)
        source_ids[ref_nonstar_pos] = np.arange(n_ref, n_ref + len(ref_nonstar_pos))
        next_id = n_ref + len(ref_nonstar_pos)

        ref_all_pos = np.concatenate([ref_star_pos, ref_nonstar_pos]).astype(int)

        # Running position/sigma/epoch/star-flag per master source.
        # master_epoch is the weighted-mean MJD so the PM-contribution to the
        # match radius uses the actual time gap to each source's mean epoch, not
        # a fixed reference epoch.
        ref_epoch_mjd  = float(img_epochs[ref_sub])
        master_ra      = fdf['ra'].values[ref_all_pos].copy()
        master_dec     = fdf['dec'].values[ref_all_pos].copy()
        master_sig_ra  = fdf['sigma_ra'].values[ref_all_pos].copy()
        master_sig_dec = fdf['sigma_dec'].values[ref_all_pos].copy()
        master_mag     = fdf['mag_zp'].values[ref_all_pos].copy()
        master_mag_err = fdf['mag_err_gdc'].values[ref_all_pos].copy()
        master_epoch   = np.full(len(ref_all_pos), ref_epoch_mjd)
        master_is_star = is_star_arr[ref_all_pos].copy()   # True if any detection is a star

        # Order remaining images: maximise sky overlap with the growing master
        # footprint while preferring images close in time to the seed.
        # This ensures ZP calibration is anchored to well-overlapping images
        # before processing non-overlapping tiles, and keeps PM displacements
        # small between matched images.
        candidates = [s for s in sub_names if s != ref_sub]
        remaining = _order_images_greedy(
            candidates  = candidates,
            fdf         = fdf,
            img_epochs  = img_epochs,
            seed_epoch_mjd = ref_epoch_mjd,
            seed_ra     = fdf['ra'].values[ref_all_pos],
            seed_dec    = fdf['dec'].values[ref_all_pos],
        )
        n_imgs = len(sub_names)
        if n_imgs > 1:
            print(f"    image processing order: {ref_sub} (seed) → "
                  + " → ".join(remaining[:min(5, len(remaining))])
                  + (" → ..." if len(remaining) > 5 else ""))

        def _update_master(idx_m, idx_c, cur_vals_ra, cur_vals_dec,
                           cur_sig_ra, cur_sig_dec, cur_mag_vals, cur_mag_err_vals,
                           cur_epoch, cur_is_star):
            """Weighted-mean update of master arrays for matched pairs."""
            w_old_ra = 1. / np.maximum(master_sig_ra[idx_m], 1e-9)**2
            w_new_ra = 1. / np.maximum(cur_sig_ra[idx_c],    1e-9)**2
            w_tot_ra = w_old_ra + w_new_ra
            master_ra[idx_m]     = (w_old_ra * master_ra[idx_m]    + w_new_ra * cur_vals_ra[idx_c])  / w_tot_ra
            master_epoch[idx_m]  = (w_old_ra * master_epoch[idx_m] + w_new_ra * cur_epoch)           / w_tot_ra
            master_sig_ra[idx_m] = 1. / np.sqrt(w_tot_ra)

            w_old_d = 1. / np.maximum(master_sig_dec[idx_m], 1e-9)**2
            w_new_d = 1. / np.maximum(cur_sig_dec[idx_c],    1e-9)**2
            w_tot_d = w_old_d + w_new_d
            master_dec[idx_m]     = (w_old_d * master_dec[idx_m]    + w_new_d * cur_vals_dec[idx_c])  / w_tot_d
            master_sig_dec[idx_m] = 1. / np.sqrt(w_tot_d)

            # Inverse-variance weighted magnitude and propagated error
            w_old_m = 1. / np.maximum(master_mag_err[idx_m],     1e-9)**2
            w_new_m = 1. / np.maximum(cur_mag_err_vals[idx_c],   1e-9)**2
            w_tot_m = w_old_m + w_new_m
            master_mag[idx_m]     = (w_old_m * master_mag[idx_m] + w_new_m * cur_mag_vals[idx_c]) / w_tot_m
            master_mag_err[idx_m] = 1. / np.sqrt(w_tot_m)
            master_is_star[idx_m] |= cur_is_star[idx_c]

        def _extend_master(new_pos, cur_epoch_mjd):
            nonlocal master_ra, master_dec, master_sig_ra, master_sig_dec
            nonlocal master_mag, master_mag_err, master_epoch, master_is_star, next_id
            n_new = len(new_pos)
            if n_new == 0:
                return
            new_ids = np.arange(next_id, next_id + n_new)
            source_ids[new_pos] = new_ids
            next_id += n_new
            master_ra      = np.concatenate([master_ra,      fdf['ra'].values[new_pos]])
            master_dec     = np.concatenate([master_dec,     fdf['dec'].values[new_pos]])
            master_sig_ra  = np.concatenate([master_sig_ra,  fdf['sigma_ra'].values[new_pos]])
            master_sig_dec = np.concatenate([master_sig_dec, fdf['sigma_dec'].values[new_pos]])
            master_mag     = np.concatenate([master_mag,     fdf['mag_zp'].values[new_pos]])
            master_mag_err = np.concatenate([master_mag_err, fdf['mag_err_gdc'].values[new_pos]])
            master_epoch   = np.concatenate([master_epoch,   np.full(n_new, cur_epoch_mjd)])
            master_is_star = np.concatenate([master_is_star, is_star_arr[new_pos]])

        _ZP_MIN_MASTER = 500   # master sources required
        _ZP_MIN_IMG    = 50    # star candidates required in current image
        _ZP_MIN_INLIER = 30    # mode-inliers required
        _ZP_MAX_CORR   = 1.0

        for sub_name in remaining:
            cur_mask = fdf['sub_name'].values == sub_name
            cur_pos  = np.where(cur_mask)[0]
            if len(cur_pos) == 0:
                continue

            cur_ra_v      = fdf['ra'].values[cur_pos]
            cur_dec_v     = fdf['dec'].values[cur_pos]
            cur_sig_ra    = fdf['sigma_ra'].values[cur_pos]
            cur_sig_dec   = fdf['sigma_dec'].values[cur_pos]
            cur_mag_v     = fdf['mag_zp'].values[cur_pos]
            cur_mag_err_v = fdf['mag_err_gdc'].values[cur_pos]
            cur_is_star   = is_star_arr[cur_pos]
            cur_epoch_mjd = float(img_epochs[sub_name])

            # ── Per-image ZP estimation (k=1, position-only, star candidates) ─
            _zp_offset = 0.0
            _sl_zp = np.where(cur_is_star)[0]
            if len(_sl_zp) >= _ZP_MIN_IMG and len(master_ra) >= _ZP_MIN_MASTER:
                _dt_zp = np.abs(master_epoch - cur_epoch_mjd) / MJD_YR
                _im_q, _ic_q = _match_two_sets(
                    master_ra, master_dec, master_sig_ra, master_sig_dec,
                    master_mag, master_mag_err,
                    cur_ra_v[_sl_zp], cur_dec_v[_sl_zp],
                    cur_sig_ra[_sl_zp], cur_sig_dec[_sl_zp],
                    cur_mag_v[_sl_zp], cur_mag_err_v[_sl_zp],
                    _dt_zp, ra0, dec0, max_pm_masyr, match_n_sigma,
                    mag_n_sigma=999., mag_floor=999., n_candidates=1,
                )
                if len(_im_q) >= _ZP_MIN_INLIER:
                    _dmag = cur_mag_v[_sl_zp][_ic_q] - master_mag[_im_q]
                    _dmag = _dmag[np.abs(_dmag) < 1.0]
                    if len(_dmag) >= _ZP_MIN_INLIER:
                        # Mode-based ZP estimation:
                        # 1. Build histogram with 0.05 mag bins and smooth
                        _bin_w   = 0.05
                        _bins    = np.arange(-1.025, 1.076, _bin_w)
                        _hist, _edges = np.histogram(_dmag, bins=_bins)
                        _hist_sm = np.convolve(_hist.astype(float),
                                               np.array([1., 2., 1.]) / 4.,
                                               mode='same')
                        # 2. Mode = centre of the smoothed peak bin
                        _pk   = int(np.argmax(_hist_sm))
                        _mode = 0.5 * (_edges[_pk] + _edges[_pk + 1])
                        # 3. Narrow cut ±0.15 mag around mode, then median
                        _dmag_cut = _dmag[np.abs(_dmag - _mode) < 0.15]
                        if len(_dmag_cut) >= _ZP_MIN_INLIER:
                            _zp_raw    = float(np.median(_dmag_cut))
                            _w68       = float(np.percentile(_dmag_cut, 84) -
                                               np.percentile(_dmag_cut, 16))
                            _zp_offset = float(np.clip(_zp_raw, -_ZP_MAX_CORR, _ZP_MAX_CORR))
                            if abs(_zp_raw) > 0.01:
                                _cap_note = (f" [capped from {_zp_raw:+.3f}]"
                                             if abs(_zp_raw) > _ZP_MAX_CORR else "")
                                print(f"      ZP [{sub_name}]: {_zp_offset:+.3f} mag "
                                      f"({len(_im_q)} pairs, {len(_dmag_cut)} inliers, "
                                      f"68%w={_w68:.3f}{_cap_note})")

            # Apply ZP correction in-place so _extend_master also sees corrected values.
            if _zp_offset != 0.0:
                fdf.iloc[cur_pos, fdf.columns.get_loc('mag_zp')] -= _zp_offset
                cur_mag_v = fdf['mag_zp'].values[cur_pos]   # refresh

            # ── Pass 1: star candidates → master (standard mag cut) ──────────
            star_local   = np.where(cur_is_star)[0]          # indices into cur_pos
            star_cur_pos = cur_pos[star_local]
            matched_master_ids: set[int] = set()

            if len(star_local) > 0:
                dt_arr = np.abs(master_epoch - cur_epoch_mjd) / MJD_YR
                idx_m_s, idx_c_s = _match_two_sets(
                    master_ra, master_dec, master_sig_ra, master_sig_dec,
                    master_mag, master_mag_err,
                    cur_ra_v[star_local], cur_dec_v[star_local],
                    cur_sig_ra[star_local], cur_sig_dec[star_local],
                    cur_mag_v[star_local], cur_mag_err_v[star_local],
                    dt_arr, ra0, dec0, max_pm_masyr, match_n_sigma,
                    mag_n_sigma=mag_n_sigma, mag_floor=mag_floor,
                )
                if len(idx_m_s) > 0:
                    source_ids[star_cur_pos[idx_c_s]] = idx_m_s
                    _update_master(idx_m_s, idx_c_s,
                                   cur_ra_v[star_local], cur_dec_v[star_local],
                                   cur_sig_ra[star_local], cur_sig_dec[star_local],
                                   cur_mag_v[star_local], cur_mag_err_v[star_local],
                                   cur_epoch_mjd, cur_is_star[star_local])
                    matched_master_ids.update(idx_m_s.tolist())
                unmatched_star_local = np.setdiff1d(np.arange(len(star_local)), idx_c_s)
                _extend_master(star_cur_pos[unmatched_star_local], cur_epoch_mjd)
            else:
                unmatched_star_local = np.empty(0, int)

            # ── Pass 2: non-stars → master (relaxed mag cut, skip taken entries) ─
            nonstar_local   = np.where(~cur_is_star)[0]
            nonstar_cur_pos = cur_pos[nonstar_local]

            if len(nonstar_local) > 0:
                dt_arr = np.abs(master_epoch - cur_epoch_mjd) / MJD_YR  # recompute: master may have grown
                idx_m_ns, idx_c_ns = _match_two_sets(
                    master_ra, master_dec, master_sig_ra, master_sig_dec,
                    master_mag, master_mag_err,
                    cur_ra_v[nonstar_local], cur_dec_v[nonstar_local],
                    cur_sig_ra[nonstar_local], cur_sig_dec[nonstar_local],
                    cur_mag_v[nonstar_local], cur_mag_err_v[nonstar_local],
                    dt_arr, ra0, dec0, max_pm_masyr, match_n_sigma,
                    mag_n_sigma=mag_n_sigma * nonstar_mag_relax, mag_floor=mag_floor,
                )
                # Drop any match that claims a master entry already taken by a star
                if len(idx_m_ns) > 0:
                    keep = np.array([idx_m_ns[k] not in matched_master_ids
                                     for k in range(len(idx_m_ns))])
                    idx_m_ns = idx_m_ns[keep]
                    idx_c_ns = idx_c_ns[keep]
                if len(idx_m_ns) > 0:
                    source_ids[nonstar_cur_pos[idx_c_ns]] = idx_m_ns
                    _update_master(idx_m_ns, idx_c_ns,
                                   cur_ra_v[nonstar_local], cur_dec_v[nonstar_local],
                                   cur_sig_ra[nonstar_local], cur_sig_dec[nonstar_local],
                                   cur_mag_v[nonstar_local], cur_mag_err_v[nonstar_local],
                                   cur_epoch_mjd, cur_is_star[nonstar_local])
                unmatched_ns_local = np.setdiff1d(np.arange(len(nonstar_local)), idx_c_ns)
                _extend_master(nonstar_cur_pos[unmatched_ns_local], cur_epoch_mjd)

        n_unique = int(source_ids.max()) + 1
        print(f"    {n_unique} unique sources found")

        # Attach source_ids to detections for groupby statistics
        fdf = fdf.copy()
        fdf['source_id'] = source_ids

        # Keep only assigned and multi-detected sources; reset index for contiguous numpy access
        fdf = fdf[fdf['source_id'] >= 0].copy()
        n_per_src = fdf.groupby('source_id').size()
        keep_ids  = n_per_src[n_per_src >= min_detections].index

        # Always keep Gaia-matched sources even with fewer than min_detections.
        # These were identified by Phase 0 / Phase 0b and are known real objects;
        # dropping them because they appear in only one image would lose cluster
        # members and other valuable sources that V1 BP3M confirmed.
        if 'has_gaia_match' in fdf.columns:
            gaia_anchored_ids = fdf.loc[fdf['has_gaia_match'].astype(bool),
                                        'source_id'].unique()
            keep_ids = keep_ids.union(pd.Index(gaia_anchored_ids))
        fdf       = fdf[fdf['source_id'].isin(keep_ids)].reset_index(drop=True)

        if len(fdf) == 0:
            print(f"    No sources with >= {min_detections} detections")
            return None

        # ── Inter-image ZP consistency check ─────────────────────────────────
        # Compare per-image magnitude residuals against the per-source
        # cross-image mean.  This is immune to completeness differences between
        # shallow and deep images (a short-exposure image detects only bright
        # stars, so its raw median magnitude is brighter — but its residuals
        # relative to each star's own cross-image mean are still ~0 if the ZP
        # is correct).  A real ZP error shifts every star in the image by the
        # same amount, showing up as a non-zero median residual.
        _star_mask = fdf['is_star_candidate'].values.astype(bool) \
                     if 'is_star_candidate' in fdf.columns else np.ones(len(fdf), bool)
        _star_fdf  = fdf[_star_mask].copy()
        if len(_star_fdf) > 0 and _star_fdf['source_id'].nunique() >= 5:
            # Per-source weighted-mean mag from all detections
            _src_mean = (
                _star_fdf
                .groupby('source_id')
                .apply(lambda g: np.average(g['mag_zp'], weights=1.0 / np.maximum(g['mag_err_gdc'], 1e-6)**2),
                       include_groups=False)
            )
            _star_fdf = _star_fdf.copy()
            _star_fdf['_src_mean'] = _star_fdf['source_id'].map(_src_mean)
            _star_fdf['_residual'] = _star_fdf['mag_zp'] - _star_fdf['_src_mean']

            # Require >= 5 matched stars per image to compute a meaningful offset
            _img_groups = _star_fdf.groupby('sub_name')
            _per_img = _img_groups['_residual'].agg(
                lambda r: float(np.median(r)) if len(r) >= 5 else np.nan
            ).dropna()

            if len(_per_img) >= 2:
                _max_dev = float(_per_img.abs().max())
                if _max_dev > 0.1:
                    _tag = "ERROR" if _max_dev > 0.5 else "WARNING"
                    _offenders = _per_img[_per_img.abs() > 0.1].reindex(
                        _per_img.abs().sort_values(ascending=False).index
                    ).dropna()
                    print(f"    {_tag} [{filt}]: inter-image ZP spread = {_max_dev:.3f} mag. "
                          f"Expected ~0 mag with mag_st_gdc. "
                          f"Likely stale catalog or miscalibration:")
                    for _sn, _off in _offenders.items():
                        _n = int(_img_groups.get_group(_sn)['source_id'].nunique()) \
                             if _sn in _img_groups.groups else 0
                        print(f"      {_sn}: ZP offset = {_off:+.3f} mag  (N={_n} stars)")

        # Compute master catalog via groupby (fast, vectorised)
        master_rows = []
        grp_arrays = {
            'ra':              fdf['ra'].values,
            'dec':             fdf['dec'].values,
            'sigma_ra':        fdf['sigma_ra'].values,
            'sigma_dec':       fdf['sigma_dec'].values,
            'epoch_yr':        fdf['epoch_mjd'].values / MJD_YR,
            'mag_zp':          fdf['mag_zp'].values,
            'mag_err':         fdf['mag_err_gdc'].values,
            'is_star':         fdf['is_star_candidate'].values,
            'has_gaia':        fdf['has_gaia_match'].values,
            'gaia_id':         fdf['gaia_source_id'].values,
            'sub_name':        fdf['orig_sub_name'].values,
            'source_id':       fdf['source_id'].values,
            'catalog_index':   fdf['catalog_index'].values,
            'x_gdc':           fdf['x_gdc'].values,
            'y_gdc':           fdf['y_gdc'].values,
            'cov_xx_raw':      fdf['cov_xx_raw'].values,
            'cov_yy_raw':      fdf['cov_yy_raw'].values,
            'cov_xy_raw':      fdf['cov_xy_raw'].values,
            'alpha_val':       fdf['alpha'].values,
            'epoch_mjd_raw':   fdf['epoch_mjd'].values,
            'base_name':       fdf['base_name'].values,
        }

        src_ids_sorted = fdf.groupby('source_id').groups   # {src_id: [pos_in_fdf]}
        for src_id, grp_idx in src_ids_sorted.items():
            gi = np.array(grp_idx)
            n  = len(gi)

            ra_g      = grp_arrays['ra'][gi]
            dec_g     = grp_arrays['dec'][gi]
            sig_ra_g  = grp_arrays['sigma_ra'][gi]
            sig_dec_g = grp_arrays['sigma_dec'][gi]
            ep_yr_g   = grp_arrays['epoch_yr'][gi]

            astrom = _fit_astrometry(ra_g, dec_g, sig_ra_g, sig_dec_g,
                                     ep_yr_g, ra0, dec0)

            mag_g  = grp_arrays['mag_zp'][gi]
            merr_g = np.maximum(grp_arrays['mag_err'][gi], 1e-4)
            w_mag  = 1. / merr_g**2
            mag_wm = float(np.dot(w_mag, mag_g) / w_mag.sum())
            mag_werr = float(1.0 / np.sqrt(w_mag.sum()))
            mag_median = float(np.median(mag_g))
            pct16, pct84 = np.percentile(mag_g, [16., 84.])
            mag_scatter = float(0.5 * (pct84 - pct16))

            # Post-grouping magnitude outlier rejection.
            # For groups with ≥3 detections, reject any detection whose mag deviates
            # by more than max(mag_outlier_sigma × MAD, mag_outlier_floor) from the
            # weighted-mean magnitude.  Re-fit astrometry on the pruned set.
            n_mag_out = 0
            if n >= 3:
                mad = float(np.median(np.abs(mag_g - mag_wm))) * 1.4826
                thresh = max(mag_outlier_sigma * mad, mag_outlier_floor)
                bad = np.abs(mag_g - mag_wm) > thresh
                n_bad = int(bad.sum())
                if n_bad > 0 and (n - n_bad) >= min_detections:
                    keep_mask = ~bad
                    gi_clean  = gi[keep_mask]
                    ra_c      = grp_arrays['ra'][gi_clean]
                    dec_c     = grp_arrays['dec'][gi_clean]
                    sig_ra_c  = grp_arrays['sigma_ra'][gi_clean]
                    sig_dec_c = grp_arrays['sigma_dec'][gi_clean]
                    ep_yr_c   = grp_arrays['epoch_yr'][gi_clean]
                    astrom = _fit_astrometry(ra_c, dec_c, sig_ra_c, sig_dec_c,
                                            ep_yr_c, ra0, dec0)
                    # Update magnitude stats on cleaned set
                    mag_g    = grp_arrays['mag_zp'][gi_clean]
                    merr_g   = np.maximum(grp_arrays['mag_err'][gi_clean], 1e-4)
                    w_mag    = 1. / merr_g**2
                    mag_wm   = float(np.dot(w_mag, mag_g) / w_mag.sum())
                    mag_werr = float(1.0 / np.sqrt(w_mag.sum()))
                    mag_median = float(np.median(mag_g))
                    pct16, pct84 = np.percentile(mag_g, [16., 84.])
                    mag_scatter  = float(0.5 * (pct84 - pct16))
                    gi   = gi_clean
                    n    = len(gi)
                    n_mag_out = n_bad

            gaia_mask = grp_arrays['has_gaia'][gi]
            gaia_id   = int(grp_arrays['gaia_id'][gi[gaia_mask][0]]) if gaia_mask.any() else 0

            pm_size = np.sqrt(astrom['pmra']**2 + astrom['pmdec']**2)

            # Build hst_indices string: "sub_name:catalog_index,..."
            snames  = grp_arrays['sub_name'][gi]
            cidxs   = grp_arrays['catalog_index'][gi]
            hst_indices = ','.join(f'{s}:{c}' for s, c in zip(snames, cidxs))
            # Mean MJD for this source (raw float, not converted)
            epoch_ref_mjd = float(np.mean(grp_arrays['epoch_mjd_raw'][gi]))

            master_rows.append({
                'source_id':     int(src_id),
                'filter':        filt,
                'n_detect':      n,
                'sub_names':     ','.join(snames.tolist()),
                'hst_indices':   hst_indices,
                'ra0':           astrom['ra0'],
                'dec0':          astrom['dec0'],
                'pmra':          astrom['pmra'],
                'pmdec':         astrom['pmdec'],
                'pm_size_masyr': float(pm_size),
                'sigma_ra0':     astrom['sigma_ra0'],
                'sigma_dec0':    astrom['sigma_dec0'],
                'sigma_pmra':    astrom['sigma_pmra'],
                'sigma_pmdec':   astrom['sigma_pmdec'],
                'epoch_ref':     astrom['epoch_ref'],
                'epoch_ref_mjd': epoch_ref_mjd,
                'chi2_ra':       astrom['chi2_ra'],
                'chi2_dec':      astrom['chi2_dec'],
                'mag_wmean':   mag_wm,
                'mag_werr':    mag_werr,
                'mag_median':  mag_median,
                'mag_scatter': mag_scatter,
                'is_star_all':     bool(grp_arrays['is_star'][gi].all()),
                'is_star_any':     bool(grp_arrays['is_star'][gi].any()),
                'has_gaia_match':  gaia_id != 0,
                'gaia_source_id':  gaia_id,
                'n_mag_outliers':  n_mag_out,
            })

        if not master_rows:
            return None

        master_df = pd.DataFrame(master_rows)
        n_with_pm = int(np.isfinite(master_df['sigma_pmra']).sum())
        print(f"    {len(master_df)} sources with >={min_detections} detections "
              f"({n_with_pm} with PM fit, "
              f"{master_df['has_gaia_match'].sum()} Gaia-matched)")
        return master_df

    # ── Parallel dispatch across filters (filters are fully independent) ──────
    filter_groups = list(det_df.groupby('filter'))
    n_filters = len(filter_groups)
    if n_filters > 1:
        with ThreadPoolExecutor(max_workers=n_filters) as executor:
            master_dfs = list(executor.map(
                lambda p: _process_one_filter(p[0], p[1]), filter_groups
            ))
        for (filt, _), mdf in zip(filter_groups, master_dfs):
            if mdf is not None:
                results[filt] = mdf
    else:
        for filt, fdf_raw in filter_groups:
            mdf = _process_one_filter(filt, fdf_raw)
            if mdf is not None:
                results[filt] = mdf

    return results


# ── Phase 2: cross-filter matching ───────────────────────────────────────────

def _deduplicate_merged(df: pd.DataFrame, pos_threshold_mas: float = 50.0) -> pd.DataFrame:
    """
    Remove source-level duplicates from the Phase 2 merged catalog.

    Two duplicate categories are handled:

    1. **Duplicate Gaia source IDs**: same non-zero gaia_source_id in multiple rows.
       Root cause: Phase 2 matched an F814W Gaia source to the wrong nearby F606W
       source, leaving the original F814W-only Gaia row intact alongside the merged
       F606W+F814W row — both carrying the same gaia_source_id.
       Fix: merge all rows sharing a gaia_source_id.  For each filter, keep the
       hst_indices from whichever row has more detections in that filter (never
       concatenate conflicting same-filter indices so no detection appears twice).

    2. **Cross-filter merge failures** (complementary-filter close pairs): two rows
       within `pos_threshold_mas` whose filter sets are *disjoint* and share no
       sub_image names.  Phase 2 missed matching them because one source was already
       claimed by a different neighbor.
       Fix: merge the pair — combine hst_indices columns, use the richer row's
       astrometry.

    Same-filter close pairs are left untouched (genuine blends / close real pairs
    in dense fields — merging them would silently drop detections).

    Invariant preserved: every (sub_name, catalog_index) detection appears in
    exactly one output row.
    """
    hsi_cols = [c for c in df.columns if c.startswith('hst_indices_')]

    def _sub_names_of_row(row) -> set[str]:
        """Return the set of sub_image names referenced in any hst_indices_* column."""
        snames: set[str] = set()
        for col in hsi_cols:
            val = row.get(col)
            if not val or (isinstance(val, float) and np.isnan(val)):
                continue
            for tok in str(val).split(';'):
                tok = tok.strip()
                if ':' in tok:
                    snames.add(tok.rsplit(':', 1)[0].strip())
        return snames

    def _n_det_col(row, col) -> int:
        val = row.get(col)
        if not val or (isinstance(val, float) and np.isnan(val)):
            return 0
        return len([t for t in str(val).split(';') if ':' in t])

    def _hsi_tokens(val) -> list[str]:
        """Parse an hst_indices value into a list of 'sname:idx' token strings."""
        if not val or (isinstance(val, float) and np.isnan(val)):
            return []
        return [t.strip() for t in str(val).split(';') if ':' in t.strip()]

    def _merge_two_rows(primary: dict, secondary: dict) -> dict:
        """
        Merge secondary into primary.  For each hst_indices_* column:
          - Only secondary has it → copy to primary (with mag/n_detect columns).
          - Both have it → concatenate token lists if they share no sub_names
            (different images of the same star); otherwise keep primary's.
        Astrometry (ra0, dec0, pmra, …) stays from primary.  filter_list rebuilt.
        """
        out = dict(primary)
        for col in hsi_cols:
            p_toks = _hsi_tokens(out.get(col))
            s_toks = _hsi_tokens(secondary.get(col))
            filt = col.replace('hst_indices_', '')
            if s_toks and not p_toks:
                # Only secondary has this filter — copy everything
                out[col] = ';'.join(s_toks)
                for prefix in ('mag_wmean_', 'mag_werr_', 'mag_median_',
                               'mag_scatter_', 'n_detect_', 'sub_names_'):
                    scol = prefix + filt
                    if scol in secondary and (scol not in out or
                            (isinstance(out.get(scol), float) and np.isnan(out[scol]))):
                        out[scol] = secondary[scol]
            elif s_toks and p_toks:
                # Both have this filter — concatenate only if no shared sub_names
                p_snames = {t.rsplit(':', 1)[0] for t in p_toks}
                s_snames = {t.rsplit(':', 1)[0] for t in s_toks}
                if not (p_snames & s_snames):
                    combined = p_toks + s_toks
                    out[col] = ';'.join(combined)
                    # Update n_detect and mag stats for this filter
                    out[f'n_detect_{filt}'] = (out.get(f'n_detect_{filt}') or 0) + len(s_toks)
                # else: shared sub_names → keep primary's (genuinely different detections)
        # Rebuild filter_list and n_detect from the merged hst_indices columns
        present_filts = sorted(
            col.replace('hst_indices_', '') for col in hsi_cols
            if (_hsi_tokens(out.get(col)))
        )
        out['filter_list'] = ','.join(present_filts)
        n_det_cols = [f'n_detect_{f}' for f in present_filts if f'n_detect_{f}' in out]
        out['n_detect'] = int(sum(out.get(c, 0) or 0 for c in n_det_cols))
        out['n_filters'] = len(present_filts)
        out['is_star_all'] = bool(primary.get('is_star_all', True)) and bool(secondary.get('is_star_all', True))
        out['is_star_any'] = bool(primary.get('is_star_any', False)) or bool(secondary.get('is_star_any', False))
        return out

    rows = df.to_dict('records')
    drop_indices: set[int] = set()

    # ── Pass 1: duplicate Gaia source IDs ────────────────────────────────────
    from collections import defaultdict
    gaia_groups: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        gid = row.get('gaia_source_id', 0)
        try:
            gid_int = int(gid) if gid and not (isinstance(gid, float) and np.isnan(gid)) else 0
        except (ValueError, TypeError):
            gid_int = 0
        if gid_int != 0:
            gaia_groups[gid_int].append(i)

    n_gaia_merges = 0
    for gid, idxs in gaia_groups.items():
        if len(idxs) < 2:
            continue
        # Keep the row with the most detections as primary
        primary_i = max(idxs, key=lambda i: rows[i].get('n_detect', 0) or 0)
        for sec_i in idxs:
            if sec_i == primary_i:
                continue
            rows[primary_i] = _merge_two_rows(rows[primary_i], rows[sec_i])
            drop_indices.add(sec_i)
            n_gaia_merges += 1

    # ── Pass 2: cross-filter merge failures (complementary filters, close position) ──
    if pos_threshold_mas > 0:
        # Work only on surviving rows
        surviving = [(i, row) for i, row in enumerate(rows) if i not in drop_indices]
        if len(surviving) >= 2:
            s_idx  = [i   for i, _ in surviving]
            s_rows = [row for _, row in surviving]
            s_ra   = np.array([r.get('ra0', np.nan) for r in s_rows], dtype=float)
            s_dec  = np.array([r.get('dec0', np.nan) for r in s_rows], dtype=float)
            good   = np.isfinite(s_ra) & np.isfinite(s_dec)
            if good.sum() >= 2:
                cos_dec = np.cos(np.nanmedian(s_dec[good]) * np.pi / 180)
                gidxs   = np.where(good)[0]
                coords  = np.column_stack([s_ra[good] * cos_dec, s_dec[good]])
                tree    = cKDTree(coords)
                pairs   = list(tree.query_pairs(r=pos_threshold_mas / 3.6e6))

                n_cf_merges = 0
                already_merged: set[int] = set()   # local surviving indices that have been merged
                for (a, b) in sorted(pairs, key=lambda p: (
                        abs(s_ra[gidxs[p[0]]] - s_ra[gidxs[p[1]]]) * cos_dec +
                        abs(s_dec[gidxs[p[0]]] - s_dec[gidxs[p[1]]]))):
                    ia, ib = gidxs[a], gidxs[b]
                    ri_idx, rj_idx = s_idx[ia], s_idx[ib]
                    if ri_idx in drop_indices or rj_idx in drop_indices:
                        continue
                    if ia in already_merged or ib in already_merged:
                        continue
                    ri, rj = rows[ri_idx], rows[rj_idx]
                    # Only merge if filter sets are disjoint
                    fi = set(col.replace('hst_indices_', '') for col in hsi_cols
                             if (ri.get(col) and not (isinstance(ri.get(col), float) and np.isnan(ri.get(col)))))
                    fj = set(col.replace('hst_indices_', '') for col in hsi_cols
                             if (rj.get(col) and not (isinstance(rj.get(col), float) and np.isnan(rj.get(col)))))
                    # Same-filter pairs where both rows have only that one filter are
                    # likely genuine blends / real close pairs — leave them untouched.
                    if fi == fj and len(fi) == 1:
                        continue
                    # Check no shared sub_image names (would indicate different detections of the same image)
                    sni = _sub_names_of_row(ri)
                    snj = _sub_names_of_row(rj)
                    if sni & snj:
                        continue  # share an image — can't be a merge failure
                    # Merge: keep primary as the one with more detections
                    pri_idx, sec_idx = (ri_idx, rj_idx) if (ri.get('n_detect', 0) or 0) >= (rj.get('n_detect', 0) or 0) else (rj_idx, ri_idx)
                    rows[pri_idx] = _merge_two_rows(rows[pri_idx], rows[sec_idx])
                    drop_indices.add(sec_idx)
                    already_merged.add(ia if sec_idx == rj_idx else ib)
                    n_cf_merges += 1

    total_dropped = len(drop_indices)
    if total_dropped > 0:
        print(f"  Deduplication: removed {total_dropped} duplicate rows "
              f"({n_gaia_merges} Gaia-ID merges, {total_dropped - n_gaia_merges} cross-filter position merges)")

    # Rebuild DataFrame preserving the original index so callers can align on df.index
    surviving_list_indices = [i for i in range(len(rows)) if i not in drop_indices]
    out_rows = [rows[i] for i in surviving_list_indices]
    out_index = [df.index[i] for i in surviving_list_indices]
    return pd.DataFrame(out_rows, index=out_index)


def _cross_filter_match(
    filter_masters: dict[str, pd.DataFrame],
    match_n_sigma: float = 5.,
    match_radius_mas: float = 200.,
) -> pd.DataFrame:
    """
    Match sources between different filter master catalogs using sky position.
    Sources are matched on (ra0, dec0) since ZP offsets don't apply across filters.
    Uses match_radius_mas as the fixed position tolerance (overrides sigma-based matching
    since cross-filter astrometric uncertainties are already well-characterised by Phase 1).

    Returns a combined DataFrame with one row per unique source (across all filters).
    Per-filter columns are added for magnitude, detection count, sub-names, and hst_indices.
    For sources that only appear in one filter, those per-filter columns for the other
    filters are NaN/empty.
    """
    if len(filter_masters) == 0:
        return pd.DataFrame()

    filters = sorted(filter_masters.keys())

    if len(filter_masters) == 1:
        filt = filters[0]
        df = filter_masters[filt].copy()
        df['n_filters']  = 1
        df['filter_list'] = df['filter']
        df['n_detect']   = df.get('n_detect', pd.Series(np.nan, index=df.index))
        df['is_star_all'] = df.get('is_star_all', pd.Series(True, index=df.index))
        df['is_star_any'] = df.get('is_star_any', pd.Series(True, index=df.index))
        # Per-filter columns
        df[f'mag_wmean_{filt}']   = df['mag_wmean']
        df[f'mag_werr_{filt}']   = df['mag_werr']
        df[f'mag_median_{filt}'] = df['mag_median']
        df[f'mag_scatter_{filt}'] = df['mag_scatter']
        df[f'n_detect_{filt}']   = df['n_detect']
        df[f'sub_names_{filt}']  = df['sub_names']
        df[f'hst_indices_{filt}'] = df['hst_indices'] if 'hst_indices' in df.columns else ''
        return df.reset_index(drop=True)

    ra0  = np.concatenate([df['ra0'].values  for df in filter_masters.values()]).mean()
    dec0 = np.concatenate([df['dec0'].values for df in filter_masters.values()]).mean()

    # Convert all masters to tangent plane
    masters_tp: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for filt, df in filter_masters.items():
        xa, ya = _to_tangent_plane(df['ra0'].values, df['dec0'].values, ra0, dec0)
        masters_tp[filt] = (xa, ya)

    # Build merged catalog row by row.
    # Each output row tracks which filter indices it merges, and we populate per-filter columns.
    # merged_rows: list of dicts (one per merged source)
    # merged_x, merged_y: tangent-plane coords for matching
    merged_rows: list[dict] = []
    merged_x_list: list[float] = []
    merged_y_list: list[float] = []

    # Per-filter column names we want to carry through
    _PER_FILTER_COLS = ('mag_wmean', 'mag_werr', 'mag_median', 'mag_scatter', 'n_detect', 'sub_names', 'hst_indices')

    def _make_base_row(row: pd.Series, filt: str) -> dict:
        """Create a base merged row from a single filter source."""
        d: dict = {}
        for col in ('ra0', 'dec0', 'pmra', 'pmdec', 'pm_size_masyr',
                    'sigma_ra0', 'sigma_dec0', 'sigma_pmra', 'sigma_pmdec',
                    'epoch_ref', 'epoch_ref_mjd', 'chi2_ra', 'chi2_dec',
                    'has_gaia_match', 'gaia_source_id'):
            if col in row.index:
                d[col] = row[col]
        d['is_star_all'] = bool(row.get('is_star_all', True))
        d['is_star_any'] = bool(row.get('is_star_any', True))
        d['filter_list'] = filt
        # Per-filter columns
        d[f'mag_wmean_{filt}']   = row.get('mag_wmean',  np.nan)
        d[f'mag_werr_{filt}']   = row.get('mag_werr',   np.nan)
        d[f'mag_median_{filt}'] = row.get('mag_median', np.nan)
        d[f'mag_scatter_{filt}'] = row.get('mag_scatter', np.nan)
        d[f'n_detect_{filt}']    = row.get('n_detect', np.nan)
        d[f'sub_names_{filt}']   = row.get('sub_names', '')
        d[f'hst_indices_{filt}'] = row.get('hst_indices', '')
        return d

    # Start with the first filter
    ref_filt = filters[0]
    for _, row in filter_masters[ref_filt].iterrows():
        merged_rows.append(_make_base_row(row, ref_filt))
    mx = masters_tp[ref_filt][0].tolist()
    my = masters_tp[ref_filt][1].tolist()
    merged_x_list.extend(mx)
    merged_y_list.extend(my)

    # Iterate over remaining filters and merge
    for filt in filters[1:]:
        cur_df = filter_masters[filt]
        cur_x, cur_y = masters_tp[filt]

        if len(merged_rows) == 0 or len(cur_x) == 0:
            for _, row in cur_df.iterrows():
                merged_rows.append(_make_base_row(row, filt))
            merged_x_list.extend(cur_x.tolist())
            merged_y_list.extend(cur_y.tolist())
            continue

        mx_arr = np.array(merged_x_list)
        my_arr = np.array(merged_y_list)
        tree = cKDTree(np.column_stack([mx_arr, my_arr]))
        dists, idxs = tree.query(np.column_stack([cur_x, cur_y]),
                                 k=1, distance_upper_bound=match_radius_mas)

        matched_cur  = dists < match_radius_mas

        # Populate per-filter columns on matched merged rows
        for ci, mi in zip(np.where(matched_cur)[0], idxs[matched_cur]):
            row = cur_df.iloc[ci]
            mrow = merged_rows[mi]
            # Append filter to filter_list
            mrow['filter_list'] = str(mrow['filter_list']) + ',' + filt
            # Per-filter columns
            mrow[f'mag_wmean_{filt}']   = row.get('mag_wmean',  np.nan)
            mrow[f'mag_werr_{filt}']   = row.get('mag_werr',   np.nan)
            mrow[f'mag_median_{filt}'] = row.get('mag_median', np.nan)
            mrow[f'mag_scatter_{filt}'] = row.get('mag_scatter', np.nan)
            mrow[f'n_detect_{filt}']    = row.get('n_detect', np.nan)
            mrow[f'sub_names_{filt}']   = row.get('sub_names', '')
            mrow[f'hst_indices_{filt}'] = row.get('hst_indices', '')
            # Update is_star flags across filters
            mrow['is_star_all'] = bool(mrow['is_star_all']) and bool(row.get('is_star_all', True))
            mrow['is_star_any'] = bool(mrow['is_star_any']) or  bool(row.get('is_star_any', False))
            # Prefer the filter with the most detections for primary astrometry
            # (update ra0, dec0, etc. if current filter has more detections)
            cur_n = row.get('n_detect', 0) or 0
            old_n_col = f'n_detect_{mrow["filter_list"].split(",")[0]}'
            old_n = mrow.get(old_n_col, 0) or 0
            if cur_n > old_n:
                for col in ('ra0', 'dec0', 'pmra', 'pmdec', 'pm_size_masyr',
                            'sigma_ra0', 'sigma_dec0', 'sigma_pmra', 'sigma_pmdec',
                            'epoch_ref', 'epoch_ref_mjd', 'chi2_ra', 'chi2_dec'):
                    if col in row.index:
                        mrow[col] = row[col]
            # Update Gaia match from whichever filter has it
            if not mrow.get('has_gaia_match', False) and row.get('has_gaia_match', False):
                mrow['has_gaia_match'] = True
                mrow['gaia_source_id'] = row.get('gaia_source_id', 0)

        # Add unmatched sources from current filter as new rows
        for ci in np.where(~matched_cur)[0]:
            row = cur_df.iloc[ci]
            new_row = _make_base_row(row, filt)
            merged_rows.append(new_row)
            merged_x_list.append(float(cur_x[ci]))
            merged_y_list.append(float(cur_y[ci]))

    # Build the final DataFrame
    merged_df = pd.DataFrame(merged_rows)

    # Compute n_detect (total across all filters) and n_filters
    n_detect_cols = [c for c in merged_df.columns if c.startswith('n_detect_') and c != 'n_detect']
    if n_detect_cols:
        merged_df['n_detect'] = merged_df[n_detect_cols].fillna(0).sum(axis=1).astype(int)
    elif 'n_detect' not in merged_df.columns:
        merged_df['n_detect'] = np.nan
    merged_df['n_filters'] = merged_df['filter_list'].str.count(',') + 1

    merged_df = _deduplicate_merged(merged_df.reset_index(drop=True))
    return merged_df.reset_index(drop=True)


# ── Phase 3: Gaia recovery ───────────────────────────────────────────────────

def _recover_gaia_matches(
    master_df: pd.DataFrame,
    gaia_csv: Optional[Path],
    match_radius_mas: float = 100.,
    n_candidates: int       = 5,
    color_tolerance: float  = 3.0,
) -> pd.DataFrame:
    """
    For sources in master_df without a Gaia match, attempt to find a Gaia
    counterpart using the improved sky positions from Phase 1/2.

    Loads the Gaia CSV, propagates Gaia positions to the mean HST epoch, and
    matches using up to n_candidates nearest Gaia neighbours per HST source.

    Matching logic:
      1. Query the n_candidates nearest Gaia sources within match_radius_mas.
      2. Apply a HST−G colour cut: learn the expected colour offset from sources
         that already have Gaia matches, then reject candidates where
         |colour − expected_offset| > color_tolerance.  If no colour information
         is available, skip the colour cut.
      3. Enforce one-to-one assignment greedily by separation (shortest first),
         so a Gaia source cannot be claimed by two HST sources.

    Returns a DataFrame of newly recovered matches (master columns + Gaia ID).
    """
    if gaia_csv is None or not Path(gaia_csv).exists():
        print("  No Gaia CSV provided; skipping Gaia recovery")
        return pd.DataFrame()

    unmatched = master_df[~master_df['has_gaia_match']].copy().reset_index(drop=True)
    if len(unmatched) == 0:
        print("  All master sources already have Gaia matches")
        return pd.DataFrame()

    print(f"  Attempting Gaia recovery for {len(unmatched)} unmatched sources "
          f"(k={n_candidates} candidates, ±{color_tolerance:.1f} mag colour cut)")

    try:
        gaia_df = pd.read_csv(gaia_csv)
        gaia_df.columns = [c.lower() for c in gaia_df.columns]
    except Exception as exc:
        print(f"  Warning: cannot load Gaia CSV: {exc}")
        return pd.DataFrame()

    # ── Propagate Gaia positions to mean HST epoch ────────────────────────────
    mean_epoch_yr = float(unmatched['epoch_ref'].mean()) if 'epoch_ref' in unmatched else 2015.0
    gaia_df = gaia_df.copy()
    dt_yr = mean_epoch_yr - 2015.5   # Gaia DR3 reference epoch
    gaia_df['ra_prop']  = gaia_df['ra'].copy()
    gaia_df['dec_prop'] = gaia_df['dec'].copy()
    has_pm = np.isfinite(gaia_df['pmra']) & np.isfinite(gaia_df['pmdec'])
    if has_pm.any():
        gaia_df.loc[has_pm, 'ra_prop']  = (
            gaia_df.loc[has_pm, 'ra']
            + dt_yr * gaia_df.loc[has_pm, 'pmra'] / 3.6e6
              / np.cos(gaia_df.loc[has_pm, 'dec'] * DEG2RAD)
        )
        gaia_df.loc[has_pm, 'dec_prop'] = (
            gaia_df.loc[has_pm, 'dec']
            + dt_yr * gaia_df.loc[has_pm, 'pmdec'] / 3.6e6
        )

    # ── Learn HST−G colour offsets from already-matched sources (all filters) ──
    # For each available filter, learn the median HST−G colour from existing
    # Gaia-matched sources.  The colour cut during candidate filtering requires
    # ALL available filters to pass; candidates with no HST magnitude in a given
    # filter are exempt from that filter's cut.
    mag_filt_cols = sorted(
        [c for c in master_df.columns if c.startswith('mag_wmean_')],
        key=lambda c: master_df[c].notna().sum(),
        reverse=True,
    )
    color_offsets: dict[str, float] = {}   # col → median(HST − G)

    if mag_filt_cols and 'gmag' in gaia_df.columns:
        existing = master_df[master_df['has_gaia_match']].copy()
        if len(existing) > 10:
            gaia_gmag_lookup = (
                gaia_df[['source_id', 'gmag']]
                .dropna(subset=['gmag'])
                .set_index('source_id')['gmag']
            )
            existing_gmag = existing['gaia_source_id'].map(gaia_gmag_lookup)
            for col in mag_filt_cols:
                existing_hst = existing[col]
                valid_pairs  = existing_hst.notna() & existing_gmag.notna()
                if valid_pairs.sum() >= 5:
                    colors = existing_hst[valid_pairs].values - existing_gmag[valid_pairs].values
                    color_offsets[col] = float(np.median(colors))
                    filt_label = col.replace('mag_wmean_', '')
                    print(f"  Colour offset ({filt_label} − G) = {color_offsets[col]:+.2f} mag "
                          f"(from {valid_pairs.sum()} existing matches)")

    # ── Build KDTree and query k candidates ───────────────────────────────────
    ra0  = float(unmatched['ra0'].mean())
    dec0 = float(unmatched['dec0'].mean())
    xm, ym = _to_tangent_plane(unmatched['ra0'].values, unmatched['dec0'].values, ra0, dec0)
    xg, yg = _to_tangent_plane(gaia_df['ra_prop'].values, gaia_df['dec_prop'].values, ra0, dec0)

    tree_gaia = cKDTree(np.column_stack([xg, yg]))
    k_eff = min(n_candidates, len(gaia_df))
    raw_d, raw_i = tree_gaia.query(
        np.column_stack([xm, ym]), k=k_eff,
        distance_upper_bound=match_radius_mas,
    )
    if k_eff == 1:
        raw_d = raw_d[:, np.newaxis]
        raw_i = raw_i[:, np.newaxis]

    # Flatten to candidate pairs
    n_unm    = len(unmatched)
    rows_m   = np.repeat(np.arange(n_unm), k_eff)
    rows_g   = raw_i.ravel()
    seps     = raw_d.ravel()
    valid    = (seps < match_radius_mas) & (rows_g < len(gaia_df))
    rows_m   = rows_m[valid]
    rows_g   = rows_g[valid]
    seps     = seps[valid]

    if len(rows_m) == 0:
        print("  No new Gaia matches found")
        return pd.DataFrame()

    # ── Colour cut (all available filters must pass) ──────────────────────────
    n_before_color = len(rows_m)
    if color_offsets and 'gmag' in gaia_df.columns:
        g_mags   = gaia_df['gmag'].values[rows_g]
        keep_color = np.ones(len(rows_m), dtype=bool)
        for col, offset in color_offsets.items():
            hst_mags  = unmatched[col].values[rows_m]
            color     = hst_mags - g_mags
            color_ok  = np.isfinite(color) & (np.abs(color - offset) < color_tolerance)
            no_hst    = ~np.isfinite(hst_mags)   # exempt if no measurement
            keep_color &= (color_ok | no_hst)
        rows_m = rows_m[keep_color]
        rows_g = rows_g[keep_color]
        seps   = seps[keep_color]
        n_filt = len(color_offsets)
        print(f"  Colour cut ({n_filt} filter{'s' if n_filt > 1 else ''}): "
              f"{n_before_color} → {len(rows_m)} candidates "
              f"({n_before_color - len(rows_m)} rejected)")

    if len(rows_m) == 0:
        print("  No new Gaia matches found after colour cut")
        return pd.DataFrame()

    # ── One-to-one greedy assignment by separation ────────────────────────────
    gaia_source_ids = gaia_df['source_id'].values
    gaia_gmags      = gaia_df['gmag'].values if 'gmag' in gaia_df.columns else np.full(len(gaia_df), np.nan)

    order  = np.argsort(seps)
    used_m: set[int] = set()
    used_g: set[int] = set()
    recovered_rows: list[dict] = []
    for oi in order:
        m, g = int(rows_m[oi]), int(rows_g[oi])
        if m not in used_m and g not in used_g:
            row = unmatched.iloc[m].to_dict()
            row['gaia_source_id']  = int(gaia_source_ids[g])
            row['gaia_gmag']       = float(gaia_gmags[g])
            row['gaia_sep_mas']    = float(seps[oi])
            row['recovery_method'] = 'hst_xmatch'
            recovered_rows.append(row)
            used_m.add(m)
            used_g.add(g)

    if not recovered_rows:
        print("  No new Gaia matches found")
        return pd.DataFrame()

    recovered_df = pd.DataFrame(recovered_rows)
    print(f"  Recovered {len(recovered_df)} new Gaia matches")
    return recovered_df


# ── Phase 4: proper astrometry with full C_r treatment ───────────────────────

def _parse_hst_indices_columns(combined_df: pd.DataFrame) -> dict[int, list[tuple[str, int]]]:
    """
    Parse all hst_indices_* columns in combined_df into a dict:
        row_index → [(sub_name, catalog_index), ...]
    """
    hst_idx_cols = [c for c in combined_df.columns if c.startswith('hst_indices_')]
    result: dict[int, list[tuple[str, int]]] = {}
    for row_i, row in combined_df.iterrows():
        pairs: list[tuple[str, int]] = []
        for col in hst_idx_cols:
            val = row.get(col, '')
            if not val or (isinstance(val, float) and np.isnan(val)):
                continue
            for token in str(val).split(','):
                token = token.strip()
                if ':' not in token:
                    continue
                sname, cidx_str = token.rsplit(':', 1)
                try:
                    pairs.append((sname.strip(), int(cidx_str.strip())))
                except ValueError:
                    continue
        result[row_i] = pairs
    return result


def _build_gaia_cov5(row: pd.Series) -> tuple:
    """Build the inflated 5×5 Gaia covariance (mas², mas²/yr²) matching BP3M.

    Parameter order: (Δα*, Δδ, pmra, pmdec, parallax).
    Inflation and source-type classification follow bp3m/solver.py exactly:
        gaia_6p = np.isfinite(pseudocolour)
        gaia_5p = np.isfinite(pmra) and not gaia_6p
        gaia_2p = has ra_error and not gaia_5p

    Returns
    -------
    (C, is_full_astrometry) where C is the 5×5 covariance (or None on failure)
    and is_full_astrometry is True for 5p/6p sources (False for 2p).
    """
    for c in ['ra_error', 'dec_error']:
        if c not in row or not np.isfinite(float(row[c])):
            return None, False

    ra_e   = float(row.get('ra_error',  np.nan))   # mas
    dec_e  = float(row.get('dec_error', np.nan))   # mas
    pm_ra_e  = float(row.get('pmra_error',     np.nan))   # mas/yr
    pm_dec_e = float(row.get('pmdec_error',    np.nan))   # mas/yr
    plx_e    = float(row.get('parallax_error', np.nan))   # mas

    # Source-type classification — identical to bp3m/solver.py:
    #   gaia_6p = np.isfinite(g['pseudocolour'])
    #   gaia_5p = np.isfinite(g['pmra']) & ~gaia_6p
    is_6p = np.isfinite(float(row.get('pseudocolour', np.nan)))
    is_5p = np.isfinite(float(row.get('pmra',         np.nan))) and not is_6p
    is_full_astrometry = is_5p or is_6p   # has PM + parallax from Gaia

    # Correlation coefficients (may be NaN for 2-param solutions)
    def _corr(name):
        v = row.get(name, np.nan)
        return float(v) if np.isfinite(float(v)) else 0.0

    c_ra_dec  = _corr('ra_dec_corr')
    c_ra_pmra = _corr('ra_pmra_corr')
    c_ra_pmd  = _corr('ra_pmdec_corr')
    c_d_pmra  = _corr('dec_pmra_corr')
    c_d_pmd   = _corr('dec_pmdec_corr')
    c_pm      = _corr('pmra_pmdec_corr')

    # Use fallback sigmas for missing PM/parallax (2-param solutions)
    if not np.isfinite(pm_ra_e):
        pm_ra_e = pm_dec_e = 1e6   # flat prior via huge sigma
        c_ra_pmra = c_ra_pmd = c_d_pmra = c_d_pmd = c_pm = 0.0
    if not np.isfinite(plx_e):
        plx_e = 1e6

    sigmas = np.array([ra_e, dec_e, pm_ra_e, pm_dec_e, plx_e])
    if not np.all(np.isfinite(sigmas)):
        return None, is_full_astrometry

    # Build correlation matrix (order: ra*, dec, pmra, pmdec, plx)
    corr = np.eye(5)
    corr[0, 1] = corr[1, 0] = c_ra_dec
    corr[0, 2] = corr[2, 0] = c_ra_pmra
    corr[0, 3] = corr[3, 0] = c_ra_pmd
    corr[1, 2] = corr[2, 1] = c_d_pmra
    corr[1, 3] = corr[3, 1] = c_d_pmd
    corr[2, 3] = corr[3, 2] = c_pm

    C = np.outer(sigmas, sigmas) * corr

    # Apply BP3M inflation (same as solver.py)
    if is_6p:
        C *= GAIA_SYS_DICT['mult_6p']
    elif is_5p:
        C *= GAIA_SYS_DICT['mult_5p']
    # 2p: multiply by 1.0 (no change)

    # Add systematic floors (in variance) for PM and parallax
    C[2, 2] += GAIA_SYS_DICT['pm_sys_err'] ** 2
    C[3, 3] += GAIA_SYS_DICT['pm_sys_err'] ** 2
    C[4, 4] += GAIA_SYS_DICT['parallax_sys_err'] ** 2

    return C, is_full_astrometry


# Diffuse prior constants — applied only to Gaia 2p and HST-only sources.
# Gaia 5p/6p sources use their Gaia covariance as the sole prior (no diffuse added).
# Parallax prior uses Michalik et al. (2015) magnitude/direction-dependent sigma
# (10 * sigma_F90) rather than a fixed value.
_SIGMA_POS_DIFFUSE = 1e6     # mas  — effectively flat (all non-5p/6p sources)
_SIGMA_PM_DIFFUSE  = 100.0   # mas/yr  (Gaia 2p and HST-only)


def _make_diffuse_prior_inv(ra_deg: float, dec_deg: float, g_mag: float) -> np.ndarray:
    """
    Build 5×5 diagonal prior precision matrix for a Gaia 2p or HST-only source.
    Uses Michalik et al. (2015) for the parallax sigma; flat on position; 100 mas/yr on PM.
    """
    sigma_plx = michalik_sigma_plx_prior(ra_deg, dec_deg, g_mag)
    return np.diag([
        _SIGMA_POS_DIFFUSE**-2, _SIGMA_POS_DIFFUSE**-2,
        _SIGMA_PM_DIFFUSE**-2,  _SIGMA_PM_DIFFUSE**-2,
        float(sigma_plx)**-2,
    ])

# Gaia DR3 reference epoch (J2016.0)
_GAIA_T_REF_MJD = 57388.5   # MJD of J2016.0
_GAIA_T_REF_YR  = 2016.0


def _measure_astrometry_proper(
    combined_df: pd.DataFrame,
    det_df: pd.DataFrame,
    r_hat_arr: np.ndarray,
    C_r: np.ndarray,
    image_names: list,
    n_r: int,
    poly_order: int,
    ra0_field: float,
    dec0_field: float,
    pscale: float,
    sub_img_meta: Optional[dict] = None,
    gaia_df: Optional[pd.DataFrame] = None,
    outlier_sigma: float = 5.0,
    min_hst_epochs: int = 2,
    _det_lookup: Optional[dict] = None,
    _tele_xyz_cache: Optional[dict] = None,
    _src_detections: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Measure proper astrometry for each source by solving the linearised
    problem entirely in tangent-plane pixel space and marginalising over
    the transformation posterior (via C_r).

    Stellar unknowns: u = (Δα*, Δδ, pmra, pmdec, parallax) in mas.
    - (Δα*, Δδ): correction at J2016.0 relative to Gaia prior (or mean obs)
    - pmra, pmdec: correction relative to Gaia prior PM (mas/yr)
    - parallax: correction relative to Gaia prior parallax (mas)

    Design matrix H_j = J_j @ U_j in pix/mas, where U_j is the 2×5 matrix:
        U_j = [[1, 0, dt_j, 0,     f_ra_j ],
               [0, 1, 0,    dt_j,  f_dec_j]]
    dt_j = epoch_j (decimal year) − 2016.0; f_ra_j, f_dec_j are the parallax
    factors at image epoch j computed from the Earth/HST barycentric position
    via get_tele_position + get_parallax_factors (same as BP3M solver.py).

    Normal equations (units: 1/mas²):
        (ΣⱼHⱼᵀ BigCⱼ⁻¹ Hⱼ + C_gaia⁻¹ + C_diffuse⁻¹) u = ΣⱼHⱼᵀ BigCⱼ⁻¹ δⱼ

    BigCⱼ (2N×2N, pix²): Xⱼ C_r Xⱼᵀ cross-image blocks + α²Jt C_hst Jtᵀ diagonal.

    Returns a DataFrame indexed like combined_df.
    """
    _t0_phase = time.perf_counter()
    print(f"  [phase4] setup: {len(combined_df)} sources, "
          f"{len(det_df)} detections, {len(image_names)} sub-images")

    # ── Pre-build lookups ─────────────────────────────────────────────────────
    if _det_lookup is not None:
        det_lookup = _det_lookup
        print(f"  [phase4] det_lookup:       reused ({len(det_lookup)} entries)")
    else:
        _t = time.perf_counter()
        det_lookup = {}
        for row in det_df.to_dict('records'):
            det_lookup[(str(row['sub_name']), int(row['catalog_index']))] = row
        print(f"  [phase4] det_lookup:       {time.perf_counter()-_t:.2f}s  "
              f"({len(det_lookup)} entries)")

    img_idx_lookup: dict[str, int] = {name: i for i, name in enumerate(image_names)}

    _t = time.perf_counter()
    # Gaia lookup: source_id (int64) → row dict.
    # iterrows() recasts int64 source IDs to float64, silently rounding large IDs.
    # Use to_numpy(np.int64) for IDs and to_dict('records') for row data instead.
    gaia_lookup: dict[int, dict] = {}
    if gaia_df is not None and 'source_id' in gaia_df.columns:
        _gaia_ids  = gaia_df['source_id'].to_numpy(dtype=np.int64, na_value=0)
        _gaia_rows = gaia_df.to_dict('records')
        for sid, row in zip(_gaia_ids, _gaia_rows):
            if sid != 0:
                gaia_lookup[int(sid)] = row
    print(f"  [phase4] gaia_lookup:      {time.perf_counter()-_t:.2f}s  "
          f"({len(gaia_lookup)} Gaia entries)")

    if _src_detections is not None:
        src_detections = _src_detections
        print(f"  [phase4] src_detections:   reused ({len(src_detections)} entries)")
    else:
        _t = time.perf_counter()
        src_detections = _parse_hst_indices_columns(combined_df)
        print(f"  [phase4] src_detections:   {time.perf_counter()-_t:.2f}s  "
              f"({len(src_detections)} entries)")

    # MAS_PER_DEG constant (3.6e6 mas / degree)
    _MAS_PER_DEG = 3.6e6

    # ── Pre-compute Earth barycentric position per image ──────────────────────
    # get_tele_position is an astropy ephemeris call — expensive per invocation,
    # so we cache tele_xyz keyed by sub_name.  Many sources share the same image,
    # so this reduces N_images ephemeris calls instead of N_sources × N_detections.
    if _tele_xyz_cache is not None:
        tele_xyz_cache = _tele_xyz_cache
        print(f"  [phase4] tele_xyz_cache:   reused ({len(tele_xyz_cache)} images)")
    else:
        _t = time.perf_counter()
        _img_mjd: dict[str, float] = {}
        for row in det_df[['sub_name', 'epoch_mjd']].drop_duplicates('sub_name').to_dict('records'):
            _img_mjd[str(row['sub_name'])] = float(row['epoch_mjd'])

        tele_xyz_cache: dict[str, np.ndarray] = {}

        def _fetch_xyz(sname_mjd):
            sname, mjd = sname_mjd
            try:
                return sname, get_tele_position(AstropyTime(mjd, format='mjd'), curr_id='earth')
            except Exception:
                return sname, np.zeros(3)

        with ThreadPoolExecutor(max_workers=min(8, len(_img_mjd))) as _pool:
            for sname, xyz in _pool.map(_fetch_xyz, _img_mjd.items()):
                tele_xyz_cache[sname] = xyz
        print(f"  [phase4] tele_xyz_cache:   {time.perf_counter()-_t:.2f}s  "
              f"({len(tele_xyz_cache)} images)")

    # rename to match closure usage below
    _tele_xyz_cache = tele_xyz_cache

    # Pre-extract gaia_source_id to avoid N combined_df.loc[] calls.
    # Use to_numpy(int64) — Series.items() may yield float64 for nullable columns,
    # silently rounding large Gaia source IDs.
    _t = time.perf_counter()
    _src_gaia_ids: dict = {}
    if 'gaia_source_id' in combined_df.columns:
        _ids = combined_df['gaia_source_id'].to_numpy(dtype=np.int64, na_value=0)
        for _row_i, _sid in zip(combined_df.index, _ids):
            _src_gaia_ids[_row_i] = int(_sid)
    print(f"  [phase4] src_gaia_ids:     {time.perf_counter()-_t:.2f}s")

    _ZERO3 = np.zeros(3)   # fallback telescope position

    # ── Per-source fit (called in parallel from thread pool) ──────────────────
    # Closures over the pre-built lookup dicts and arrays above.
    # Each call is fully independent: no shared mutable state.
    # Threads give real parallelism here because numpy LAPACK (linalg.inv/solve)
    # releases the GIL.
    def _fit_one_source(row_i) -> dict:
        pairs = src_detections.get(row_i, [])
        valid_pairs = [(s, c) for (s, c) in pairs if (s, c) in det_lookup]

        base_row: dict = {
            'ra_xmatch':               np.nan,
            'dec_xmatch':              np.nan,
            'pmra_xmatch':             np.nan,
            'pmdec_xmatch':            np.nan,
            'parallax_xmatch':         np.nan,
            'sigma_ra_xmatch':         np.nan,
            'sigma_dec_xmatch':        np.nan,
            'sigma_pmra_xmatch':       np.nan,
            'sigma_pmdec_xmatch':      np.nan,
            'sigma_parallax_xmatch':   np.nan,
            'corr_pmra_pmdec_xmatch':  np.nan,
            'n_detect_fit':            0,
            'n_outliers_xmatch':       0,
            'outlier_images':          '',
            'det_chi2':                '',
            'epoch_ref_xmatch':        _GAIA_T_REF_YR,
            'chi2_xmatch':             np.nan,
        }

        if not valid_pairs:
            return base_row

        # ── Gaia prior for this source ────────────────────────────────────────
        gaia_row: Optional[pd.Series] = None
        sid_int = _src_gaia_ids.get(row_i, 0)
        if sid_int > 0:
            gaia_row = gaia_lookup.get(sid_int)

        has_gaia = gaia_row is not None
        if has_gaia:
            ra_g    = float(gaia_row['ra'])   # degrees, J2016.0
            dec_g   = float(gaia_row['dec'])  # degrees, J2016.0
            pmra_g  = float(gaia_row.get('pmra',     0.0) or 0.0)
            pmdec_g = float(gaia_row.get('pmdec',    0.0) or 0.0)
            plx_g   = float(gaia_row.get('parallax', 0.0) or 0.0)
            if not np.isfinite(pmra_g):  pmra_g  = 0.0
            if not np.isfinite(pmdec_g): pmdec_g = 0.0
            if not np.isfinite(plx_g):   plx_g   = 0.0
            C_gaia, has_full_gaia_astrometry = _build_gaia_cov5(gaia_row)
            gaia_g_mag = float(gaia_row.get('gmag', 20.0))
            if not np.isfinite(gaia_g_mag):
                gaia_g_mag = 20.0
        else:
            ra_g = dec_g = pmra_g = pmdec_g = plx_g = 0.0
            C_gaia = None
            has_full_gaia_astrometry = False
            gaia_g_mag = 20.0  # cap — HST-only sources are always faint

        # ── Collect per-detection data ────────────────────────────────────────
        # For poly_order=1 (the common case) we skip build_X_matrix entirely:
        # X_mat is reconstructed vectorised inside _build_system, and y_obs is
        # computed inline.  For higher orders we fall back to the full call.
        det_data: list[dict] = []
        _poly1 = (poly_order == 1)
        for (sname, cidx) in valid_pairs:
            j_idx = img_idx_lookup.get(sname, -1)
            if j_idx < 0:
                continue
            d   = det_lookup[(sname, cidx)]
            x_c = float(d['x_gdc']) - _SOLVER_XO
            y_c = float(d['y_gdc']) - _SOLVER_YO
            cs  = j_idx * n_r
            r_blk = r_hat_arr[cs:cs + n_r]
            if _poly1:
                # y_obs = X_mat @ r_blk for poly_order=1 without allocating X_mat
                y_obs = np.array([r_blk[0]*x_c + r_blk[1]*y_c + r_blk[4],
                                  r_blk[2]*x_c + r_blk[3]*y_c + r_blk[5]])
                X_mat = None   # reconstructed in _build_system
            else:
                X_mat = build_X_matrix(x_c, y_c, 0., 0., 0., 0., poly_order=poly_order)
                y_obs = X_mat @ r_blk
            det_data.append({
                'sname':      sname,
                'j_idx':      j_idx,
                'cs':         cs,
                'x_c':        x_c,
                'y_c':        y_c,
                'X_mat':      X_mat,
                'y_obs':      y_obs,
                'epoch_yr':   2000.0 + (float(d['epoch_mjd']) - _MJD_J2000) / MJD_YR,
                'epoch_mjd':  float(d['epoch_mjd']),
                'cov_xx_raw': float(d['cov_xx_raw']),
                'cov_yy_raw': float(d['cov_yy_raw']),
                'cov_xy_raw': float(d['cov_xy_raw']),
                'alpha':      float(d['alpha']),
            })

        if not det_data:
            return base_row

        # ── Reference sky position ────────────────────────────────────────────
        # Per-image tangent-plane metadata (ra0, dec0, pscale) matching BP3M.
        # sub_img_meta maps sub-image name → (ra0_deg, dec0_deg, pscale_mas).
        _meta = sub_img_meta or {}

        def _img_meta(sname: str) -> tuple[float, float, float]:
            return _meta.get(sname, (ra0_field, dec0_field, pscale))

        if has_gaia:
            ref_ra, ref_dec = ra_g, dec_g
        else:
            # Average sky position across observations using per-image inversions
            sky_ras, sky_decs = [], []
            for dd in det_data:
                ra0_k, dec0_k, ps_k = _img_meta(dd['sname'])
                try:
                    ra_k, dec_k = plane_project_inverse(
                        np.array([dd['y_obs'][0]]), np.array([dd['y_obs'][1]]),
                        ra0_k, dec0_k, ps_k)
                    sky_ras.append(float(ra_k[0]))
                    sky_decs.append(float(dec_k[0]))
                except Exception:
                    pass
            if not sky_ras:
                return base_row
            ref_ra  = float(np.mean(sky_ras))
            ref_dec = float(np.mean(sky_decs))

        # ── Build BigC (2N×2N, pix²) and residuals (2N,) ─────────────────────
        def _build_system(ddata: list[dict]) -> Optional[dict]:
            N  = len(ddata)
            if N < 1:
                return None

            # Fix 3: vectorise all per-detection work — replaces the O(N) Python
            # loop with a handful of numpy calls on (N,) arrays.

            snames    = [da['sname']     for da in ddata]
            cs_arr    = np.array([da['cs']         for da in ddata], dtype=np.intp)
            x_c_arr   = np.array([da['x_c']        for da in ddata])
            y_c_arr   = np.array([da['y_c']        for da in ddata])
            y_obs_all = np.array([da['y_obs']      for da in ddata])   # (N, 2)
            dt_arr    = np.array([da['epoch_yr'] - _GAIA_T_REF_YR for da in ddata])
            cov_xx    = np.array([da['cov_xx_raw'] for da in ddata])
            cov_yy    = np.array([da['cov_yy_raw'] for da in ddata])
            cov_xy    = np.array([da['cov_xy_raw'] for da in ddata])
            alpha_arr = np.array([da['alpha']      for da in ddata])

            ra0_arr  = np.array([_img_meta(s)[0]  for s in snames])
            dec0_arr = np.array([_img_meta(s)[1]  for s in snames])
            ps_arr   = np.array([_img_meta(s)[2]  for s in snames])

            # Reference sky positions (epoch-propagated for Gaia stars)
            if has_gaia:
                cos_dec_g   = np.cos(dec_g * DEG2RAD)
                ra_ref_arr  = ra_g  + pmra_g  * dt_arr / (cos_dec_g * _MAS_PER_DEG)
                dec_ref_arr = dec_g + pmdec_g * dt_arr / _MAS_PER_DEG
            else:
                ra_ref_arr  = np.full(N, ref_ra)
                dec_ref_arr = np.full(N, ref_dec)

            # Deltas: one plane_project call for all N detections
            xr_all, yr_all = plane_project(ra_ref_arr, dec_ref_arr,
                                            ra0_arr, dec0_arr, ps_arr)
            # Fall back to y_obs where projection is non-finite
            xr_all = np.where(np.isfinite(xr_all), xr_all, y_obs_all[:, 0])
            yr_all = np.where(np.isfinite(yr_all), yr_all, y_obs_all[:, 1])
            deltas = (y_obs_all - np.stack([xr_all, yr_all], axis=1)).ravel()  # (2N,)

            # H_stack: J_j @ U_j vectorised across all N detections
            J_all = plane_project_jacobian(ra_ref_arr, dec_ref_arr,
                                           ra0_arr, dec0_arr, ps_arr)  # (N, 2, 2)
            # Replace non-finite rows with field-centre fallback
            bad_J = ~np.all(np.isfinite(J_all), axis=(-2, -1))
            if bad_J.any():
                J_fb = plane_project_jacobian(
                    np.full(bad_J.sum(), ref_ra), np.full(bad_J.sum(), ref_dec),
                    np.full(bad_J.sum(), ra0_field), np.full(bad_J.sum(), dec0_field),
                    np.full(bad_J.sum(), pscale))
                J_all[bad_J] = J_fb

            tele_xyz_T = np.array([_tele_xyz_cache.get(s, _ZERO3)
                                    for s in snames]).T            # (3, N)
            plx_ra_all, plx_dec_all = get_parallax_factors(ref_ra, ref_dec, tele_xyz_T)

            U_all = np.zeros((N, 2, 5))
            U_all[:, 0, 0] = 1.;  U_all[:, 1, 1] = 1.
            U_all[:, 0, 2] = dt_arr;  U_all[:, 1, 3] = dt_arr
            U_all[:, 0, 4] = plx_ra_all;  U_all[:, 1, 4] = plx_dec_all
            H_stack = np.einsum('nij,njk->nik', J_all, U_all).reshape(2*N, 5)

            # Diagonal C_hst blocks: J_trans @ (alpha² C_hst) @ J_trans.T  (N, 2, 2)
            sig_x   = np.sqrt(np.maximum(cov_xx, 0.))
            sig_y   = np.sqrt(np.maximum(cov_yy, 0.))
            denom   = sig_x * sig_y
            corr    = np.where(denom > 0,
                               np.clip(cov_xy / denom, -0.9999, 0.9999), 0.)
            sxsy    = sig_x * sig_y * corr
            C_hst_all        = np.zeros((N, 2, 2))
            C_hst_all[:, 0, 0] = sig_x**2
            C_hst_all[:, 1, 1] = sig_y**2
            C_hst_all[:, 0, 1] = sxsy
            C_hst_all[:, 1, 0] = sxsy

            r_blk_all = r_hat_arr[cs_arr[:, None] + np.arange(n_r)]  # (N, n_r)
            if poly_order == 1:
                J_trans_all = np.zeros((N, 2, 2))
                J_trans_all[:, 0, 0] = r_blk_all[:, 0]   # a
                J_trans_all[:, 0, 1] = r_blk_all[:, 1]   # b
                J_trans_all[:, 1, 0] = r_blk_all[:, 2]   # c
                J_trans_all[:, 1, 1] = r_blk_all[:, 3]   # d
            else:
                J_trans_all = np.array([
                    compute_poly_jacobian(r_blk_all[a],
                                          np.array([x_c_arr[a]]),
                                          np.array([y_c_arr[a]]), poly_order)[0]
                    for a in range(N)
                ])

            alpha2       = (alpha_arr ** 2)[:, None, None]
            diag_blocks  = np.einsum('nij,njk,nlk->nil',
                                     J_trans_all, alpha2 * C_hst_all, J_trans_all)

            # X_mat array: for poly_order=1 reconstruct from x_c/y_c (no alloc in loop)
            if poly_order == 1:
                X_arr = np.zeros((N, 2, n_r))
                X_arr[:, 0, 0] = x_c_arr;  X_arr[:, 0, 1] = y_c_arr;  X_arr[:, 0, 4] = 1.
                X_arr[:, 1, 2] = x_c_arr;  X_arr[:, 1, 3] = y_c_arr;  X_arr[:, 1, 5] = 1.
            else:
                X_arr = np.array([da['X_mat'] for da in ddata])  # (N, 2, n_r)

            # ── Assemble Big_C ────────────────────────────────────────────────
            Big_C = np.zeros((2 * N, 2 * N))

            # Diagonal C_hst blocks via advanced indexing (no Python loop)
            a_idx = np.arange(N)
            for i in range(2):
                for j in range(2):
                    Big_C[2*a_idx + i, 2*a_idx + j] += diag_blocks[:, i, j]

            # C_r contribution: X_flat @ C_r_sub @ X_flat.T  (one BLAS call)
            row_idx      = cs_arr[:, None] + np.arange(n_r)        # (N, n_r)
            row_idx_flat = row_idx.ravel()
            C_r_sub      = C_r[np.ix_(row_idx_flat, row_idx_flat)] # (N*n_r, N*n_r)
            X_flat       = np.zeros((2 * N, N * n_r))
            for a in range(N):
                X_flat[2*a:2*a+2, a*n_r:(a+1)*n_r] = X_arr[a]
            Big_C += X_flat @ C_r_sub @ X_flat.T

            # Prior information matrix in mas⁻²
            # Gaia 5p/6p: no diffuse prior — use Gaia covariance only.
            # Gaia 2p and HST-only: Michalik parallax prior + 100 mas/yr PM prior,
            #   combined with the (near-uninformative) Gaia 2p position covariance.
            if has_full_gaia_astrometry and C_gaia is not None:
                try:
                    C_prior_inv = np.linalg.inv(C_gaia)
                except np.linalg.LinAlgError:
                    C_prior_inv = _make_diffuse_prior_inv(ref_ra, ref_dec, gaia_g_mag)
            else:
                C_prior_inv = _make_diffuse_prior_inv(ref_ra, ref_dec, gaia_g_mag)
                if has_gaia and C_gaia is not None:
                    try:
                        C_prior_inv = C_prior_inv + np.linalg.inv(C_gaia)
                    except np.linalg.LinAlgError:
                        pass

            try:
                Big_C_inv = np.linalg.inv(Big_C)
            except np.linalg.LinAlgError:
                return None

            # Normal equations: (HᵀC⁻¹H + C_prior⁻¹) u = HᵀC⁻¹δ
            AtCiA = H_stack.T @ Big_C_inv @ H_stack + C_prior_inv  # (5, 5) mas⁻²
            AtCib = H_stack.T @ Big_C_inv @ deltas                   # (5,)  mas⁻¹
            try:
                u     = np.linalg.solve(AtCiA, AtCib)   # (5,) mas
                C_u   = np.linalg.inv(AtCiA)             # (5, 5) mas²
            except np.linalg.LinAlgError:
                return None

            return {
                'u': u, 'C_u': C_u, 'N': N,
                'Big_C': Big_C, 'Big_C_inv': Big_C_inv,
                'H_stack': H_stack, 'deltas': deltas,
            }

        # Initial solve
        fit = _build_system(det_data)
        if fit is None:
            return base_row

        # Outlier rejection (up to 3 iterations)
        outlier_snames: list[str] = []
        for _iter in range(3):
            residuals = fit['deltas'] - fit['H_stack'] @ fit['u']
            N_fit = fit['N']

            # Vectorised chi² per detection: analytical batch 2×2 inverse
            # avoids N_fit separate linalg.inv calls.
            _ai = np.arange(N_fit)
            _a  = fit['Big_C'][2*_ai,   2*_ai  ]
            _b  = fit['Big_C'][2*_ai,   2*_ai+1]
            _c  = fit['Big_C'][2*_ai+1, 2*_ai  ]
            _d  = fit['Big_C'][2*_ai+1, 2*_ai+1]
            _det = _a * _d - _b * _c
            _safe = np.abs(_det) > 0
            _inv_det = np.where(_safe, 1.0 / np.where(_safe, _det, 1.0), 0.0)
            _r0 = residuals[2*_ai]
            _r1 = residuals[2*_ai + 1]
            chi2_per = ((_d*_r0 - _b*_r1)*_r0 + (-_c*_r0 + _a*_r1)*_r1) * _inv_det

            outlier_mask = chi2_per > outlier_sigma**2
            if not outlier_mask.any():
                break
            for i in np.where(outlier_mask)[0]:
                outlier_snames.append(det_data[i]['sname'])
            det_data = [d for i, d in enumerate(det_data) if not outlier_mask[i]]
            if not det_data:
                break
            fit = _build_system(det_data)
            if fit is None:
                det_data = []
                break

        n_outliers = len(outlier_snames)
        if fit is None or not det_data:
            base_row['n_outliers_xmatch'] = n_outliers
            base_row['outlier_images']    = ','.join(set(outlier_snames))
            return base_row

        u   = fit['u']    # (5,) mas: corrections relative to Gaia reference
        C_u = fit['C_u']  # (5, 5) mas²

        # ── Convert corrections to absolute astrometric values ────────────────
        # u = (Δα*, Δδ, Δpmra, Δpmdec, Δplx)  all in mas / mas/yr / mas
        if has_gaia:
            ra_out  = ra_g  + u[0] / (np.cos(dec_g * DEG2RAD) * _MAS_PER_DEG)
            dec_out = dec_g + u[1] / _MAS_PER_DEG
            pmra_out  = pmra_g  + u[2]
            pmdec_out = pmdec_g + u[3]
            plx_out   = plx_g   + u[4]
        else:
            # Reference is mean sky-pixel position; convert to sky coords
            # u[0:2] is an additional offset in mas from that reference
            ra_out  = ref_ra  + u[0] / (np.cos(ref_dec * DEG2RAD) * _MAS_PER_DEG)
            dec_out = ref_dec + u[1] / _MAS_PER_DEG
            pmra_out  = u[2]
            pmdec_out = u[3]
            plx_out   = u[4]

        # Uncertainties (mas, mas/yr, mas)
        sigma_ra     = float(np.sqrt(max(C_u[0, 0], 0.)))
        sigma_dec    = float(np.sqrt(max(C_u[1, 1], 0.)))
        sigma_pmra   = float(np.sqrt(max(C_u[2, 2], 0.)))
        sigma_pmdec  = float(np.sqrt(max(C_u[3, 3], 0.)))
        sigma_plx    = float(np.sqrt(max(C_u[4, 4], 0.)))
        denom_corr   = sigma_pmra * sigma_pmdec
        corr_pm      = float(C_u[2, 3] / denom_corr) if denom_corr > 0 else np.nan

        # Null out PM if its uncertainty exceeds _SIGMA_PM_DIFFUSE/10 (10 mas/yr) —
        # below this threshold the measurement is not useful for most science.
        _pm_null = sigma_pmra >= _SIGMA_PM_DIFFUSE / 10.0 or sigma_pmdec >= _SIGMA_PM_DIFFUSE / 10.0
        if _pm_null:
            pmra_out = pmdec_out = np.nan
            sigma_pmra = sigma_pmdec = corr_pm = np.nan

        # Overall chi2/dof  (residuals in tangent-plane pixels)
        res_final = fit['deltas'] - fit['H_stack'] @ u
        chi2_total = float(res_final @ fit['Big_C_inv'] @ res_final)
        dof = max(2 * fit['N'] - 5, 1)
        chi2_dof = chi2_total / dof

        # Per-detection chi2 for surviving inliers: "sname:chi2,..."
        det_chi2_parts: list[str] = []
        for i, d in enumerate(det_data):
            r_i = res_final[2*i:2*i+2]
            try:
                C_i_inv = np.linalg.inv(fit['Big_C'][2*i:2*i+2, 2*i:2*i+2])
                c2 = float(r_i @ C_i_inv @ r_i)
            except np.linalg.LinAlgError:
                c2 = 0.0
            det_chi2_parts.append(f"{d['sname']}:{c2:.4f}")
        det_chi2_str = ','.join(det_chi2_parts)

        base_row.update({
            'ra_xmatch':               float(ra_out),
            'dec_xmatch':              float(dec_out),
            'pmra_xmatch':             float(pmra_out),
            'pmdec_xmatch':            float(pmdec_out),
            'parallax_xmatch':         float(plx_out),
            'sigma_ra_xmatch':         sigma_ra,
            'sigma_dec_xmatch':        sigma_dec,
            'sigma_pmra_xmatch':       sigma_pmra,
            'sigma_pmdec_xmatch':      sigma_pmdec,
            'sigma_parallax_xmatch':   sigma_plx,
            'corr_pmra_pmdec_xmatch':  corr_pm,
            'n_detect_fit':            fit['N'],
            'n_outliers_xmatch':       n_outliers,
            'outlier_images':          ','.join(set(outlier_snames)),
            'det_chi2':                det_chi2_str,
            'epoch_ref_xmatch':        _GAIA_T_REF_YR,
            'chi2_xmatch':             chi2_dof,
        })
        return base_row

    # ── Parallel dispatch (threads release GIL during LAPACK calls) ──────────
    all_indices = list(combined_df.index)
    n_total     = len(all_indices)
    n_workers   = min(8, os.cpu_count() or 1)

    print(f"  [phase4] dispatching {n_total} sources across {n_workers} threads ...")
    _t_dispatch = time.perf_counter()

    # Thread-safe progress counter
    _progress_lock   = threading.Lock()
    _progress_done   = [0]
    _report_interval = max(1, n_total // 20)   # report every ~5%

    def _report_progress(n_just_done: int) -> None:
        with _progress_lock:
            _progress_done[0] += n_just_done
            done = _progress_done[0]
            if done % _report_interval < n_just_done or done == n_total:
                elapsed = time.perf_counter() - _t_dispatch
                rate    = done / elapsed if elapsed > 0 else 0.0
                eta     = (n_total - done) / rate if rate > 0 else float('inf')
                print(f"  [phase4]   {done:>7}/{n_total}  "
                      f"({100*done/n_total:.0f}%)  "
                      f"{rate:.0f} src/s  "
                      f"ETA {eta:.0f}s", flush=True)

    if n_workers > 1 and n_total >= 200:
        # Sort by detection count descending (heaviest first), then round-robin
        # assign to chunks — LPT heuristic for near-optimal load balance.
        all_indices.sort(key=lambda ri: len(src_detections.get(ri, [])), reverse=True)
        chunks = [[] for _ in range(n_workers)]
        for i, row_i in enumerate(all_indices):
            chunks[i % n_workers].append(row_i)

        def _process_chunk(chunk_indices: list) -> list[dict]:
            _t_chunk = time.perf_counter()
            results  = [_fit_one_source(row_i) for row_i in chunk_indices]
            _report_progress(len(chunk_indices))
            return results

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_process_chunk, chunks))
        out_rows = [r for chunk in results for r in chunk]
    else:
        out_rows = []
        for row_i in all_indices:
            out_rows.append(_fit_one_source(row_i))
            _report_progress(1)

    elapsed_total = time.perf_counter() - _t_dispatch
    rate_total    = n_total / elapsed_total if elapsed_total > 0 else 0.0
    print(f"  [phase4] fit done: {elapsed_total:.1f}s total  "
          f"({rate_total:.0f} src/s avg)")
    print(f"  [phase4] total phase time: "
          f"{time.perf_counter() - _t0_phase:.1f}s")

    return pd.DataFrame(out_rows, index=combined_df.index)


# ── Phase 5: PM-guided second-pass cross-match ────────────────────────────────

def _second_pass_match(
    det_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    ra0: float,
    dec0: float,
    max_pm_unc_masyr: float = 10.0,
    match_n_sigma: float = 5.0,
    mag_n_sigma: float = 3.0,
    mag_floor: float = 0.10,
    min_detections: int = 2,
    min_match_mas: float = 10.0,
) -> pd.DataFrame:
    """
    Second-pass cross-match using first-pass PM estimates to propagate source
    positions to each image epoch, then re-match detections.

    Motivation
    ----------
    The first pass anchors each image's matching to the previous image's
    measured positions.  For sources with significant proper motions and long
    time baselines between images, the accumulated position offset can exceed
    the match radius, causing the same star to appear as two separate master
    entries.  The second pass uses the first-pass PM model to predict where
    each well-measured source should be at each image epoch, closing those gaps.

    Algorithm
    ---------
    1. Select "template" sources: those with well-measured PMs
       (max(sigma_pmra, sigma_pmdec) < max_pm_unc_masyr).
    2. For each image epoch in the detection catalog:
         a. Predict each template's sky position at that epoch.
         b. Compute predicted uncertainty = sqrt(sigma_pos² + (sigma_pm * dt)²).
         c. Estimate the ZP offset for same-filter images using the template
            detections already established; skip for cross-filter (position-only).
         d. Match detections to predicted positions (Hungarian-style, same as
            Phase 1) using the propagated uncertainty as the match radius.
    3. Reassign detections to template sources (templates take priority over
       the first-pass assignment).
    4. Re-fit astrometry (_fit_astrometry) for all sources whose detection
       sets changed.  Sources losing all detections to templates are dropped.
    5. Return an updated combined_df with v2 PM and position columns.

    Only detections in images NOT already covered by a template's first-pass
    hst_indices are eligible for reassignment; existing first-pass detections
    are kept unless a cross-image merge is identified.

    Returns
    -------
    pd.DataFrame  (same columns as combined_df, plus '_v2' suffix on
    re-measured astrometry columns)
    """
    # ── Choose the best available PM columns ─────────────────────────────────
    if 'pmra_xmatch' in combined_df.columns and combined_df['pmra_xmatch'].notna().any():
        pm_ra_col    = 'pmra_xmatch'
        pm_dec_col   = 'pmdec_xmatch'
        spm_ra_col   = 'sigma_pmra_xmatch'
        spm_dec_col  = 'sigma_pmdec_xmatch'
        ra0_col      = 'ra0'
        dec0_col     = 'dec0'
        sra0_col     = 'sigma_ra0'
        sdec0_col    = 'sigma_dec0'
        epoch_col    = 'epoch_ref'         # decimal year
    else:
        pm_ra_col    = 'pmra'
        pm_dec_col   = 'pmdec'
        spm_ra_col   = 'sigma_pmra'
        spm_dec_col  = 'sigma_pmdec'
        ra0_col      = 'ra0'
        dec0_col     = 'dec0'
        sra0_col     = 'sigma_ra0'
        sdec0_col    = 'sigma_dec0'
        epoch_col    = 'epoch_ref'

    needed = [pm_ra_col, pm_dec_col, spm_ra_col, spm_dec_col,
              ra0_col, dec0_col, sra0_col, sdec0_col, epoch_col]
    # hst_indices may be stored as bare 'hst_indices' (single-filter) or as
    # per-filter 'hst_indices_F814W' etc. after cross-filter merging.
    has_hst_indices = any(c == 'hst_indices' or c.startswith('hst_indices_')
                          for c in combined_df.columns)
    if not has_hst_indices:
        needed.append('hst_indices')   # will trigger the missing-column exit below
    missing = [c for c in needed if c not in combined_df.columns]
    if missing:
        print(f"  Phase 5: skipped — missing columns: {missing}")
        return combined_df

    # ── Select templates ──────────────────────────────────────────────────────
    ok_pm = (
        combined_df[spm_ra_col].notna() &
        combined_df[spm_dec_col].notna() &
        combined_df[pm_ra_col].notna() &
        (combined_df[spm_ra_col] < max_pm_unc_masyr) &
        (combined_df[spm_dec_col] < max_pm_unc_masyr)
    )
    templates = combined_df[ok_pm].copy().reset_index(drop=True)
    print(f"  Phase 5: {len(templates)} template sources "
          f"(PM unc < {max_pm_unc_masyr} mas/yr) of {len(combined_df)} total")
    if len(templates) == 0:
        return combined_df

    # ── Build detection lookup: (sub_name, catalog_index) → row index in det_df ─
    det_key = det_df['sub_name'].astype(str) + ':' + det_df['catalog_index'].astype(str)
    key_to_detrow = {k: i for i, k in enumerate(det_key)}

    # Parse all hst_indices / hst_indices_* columns into a per-template lookup.
    # _parse_hst_indices_columns handles both bare 'hst_indices' and per-filter
    # 'hst_indices_F814W' columns produced after cross-filter merging.
    hst_pairs_lookup: dict[int, list[tuple[str, int]]] = _parse_hst_indices_columns(templates)

    # Set of "sub_name:catalog_index" strings claimed by each template in the
    # first pass, used to skip images where the template already has a match.
    template_claimed: dict[int, set] = {}   # template_row_idx → set of det_keys
    for ti in range(len(templates)):
        claimed = {f'{s}:{c}' for s, c in hst_pairs_lookup.get(ti, [])}
        template_claimed[ti] = claimed

    # ── All images we need to search ─────────────────────────────────────────
    all_sub_names = sorted(det_df['sub_name'].unique())
    img_epoch_mjd = (det_df.groupby('sub_name')['epoch_mjd']
                           .first())          # Series: sub_name → epoch_mjd
    img_filter    = (det_df.groupby('sub_name')['filter']
                           .first())          # Series: sub_name → filter name

    # Convert template epoch_ref (decimal year) to MJD for arithmetic
    # epoch_ref = 2000 + (epoch_mjd / MJD_YR - _MJD0_YR)
    # → epoch_mjd_ref = (_MJD0_YR + epoch_ref - 2000) * MJD_YR
    tmpl_epoch_mjd = (templates[epoch_col] - 2000.0 + _MJD0_YR) * MJD_YR

    cos_dec0 = np.cos(np.deg2rad(dec0))
    MAS_TO_DEG = 1.0 / 3.6e6

    # ── Per-image matching ────────────────────────────────────────────────────
    # new_assignments: list of (det_row_idx, template_ti) pairs
    new_assignments: list[tuple[int, int]] = []
    claimed_globally: set[int] = set()   # det_row indices already claimed

    # Process images in epoch order (order doesn't affect correctness here
    # since we match against a fixed catalog, not a growing master)
    for sub_name in sorted(all_sub_names,
                           key=lambda s: float(img_epoch_mjd[s])):
        img_filt      = str(img_filter[sub_name])
        img_epoch     = float(img_epoch_mjd[sub_name])
        img_epoch_yr  = 2000.0 + (img_epoch / MJD_YR - _MJD0_YR)

        # Detections in this image not yet claimed
        sub_mask  = det_df['sub_name'].values == sub_name
        sub_rows  = np.where(sub_mask)[0]
        free_rows = [r for r in sub_rows if r not in claimed_globally]
        if not free_rows:
            continue
        free_rows = np.array(free_rows)

        # ── Predict template positions at this epoch ──────────────────────
        dt_yr = (img_epoch - tmpl_epoch_mjd.values) / 365.25   # (n_tmpl,)

        # Predicted RA/Dec (degrees); PM in mas/yr
        pred_ra  = (templates[ra0_col].values
                    + templates[pm_ra_col].values * MAS_TO_DEG / cos_dec0 * dt_yr)
        pred_dec = (templates[dec0_col].values
                    + templates[pm_dec_col].values * MAS_TO_DEG * dt_yr)

        # Predicted positional sigma (mas → deg)
        pred_sig_ra  = np.hypot(templates[sra0_col].values,
                                templates[spm_ra_col].values * np.abs(dt_yr)) * MAS_TO_DEG
        pred_sig_dec = np.hypot(templates[sdec0_col].values,
                                templates[spm_dec_col].values * np.abs(dt_yr)) * MAS_TO_DEG
        # Floor: min_match_mas
        pred_sig_ra  = np.maximum(pred_sig_ra,  min_match_mas * MAS_TO_DEG)
        pred_sig_dec = np.maximum(pred_sig_dec, min_match_mas * MAS_TO_DEG)

        det_ra   = det_df['ra'].values[free_rows]
        det_dec  = det_df['dec'].values[free_rows]
        det_sra  = det_df['sigma_ra'].values[free_rows]
        det_sdec = det_df['sigma_dec'].values[free_rows]
        det_mag  = det_df['mag_zp'].values[free_rows]
        det_merr = det_df['mag_err_gdc'].values[free_rows]

        # ── ZP correction and per-filter template magnitudes ─────────────
        # Use the per-filter magnitude column (mag_wmean_{img_filt}) if the
        # template has photometry in this image's filter — even for cross-filter
        # templates.  Fall back to the primary mag_wmean only for same-filter.
        tmpl_mag_filt_col = f'mag_wmean_{img_filt}'
        tmpl_merr_filt_col = f'mag_werr_{img_filt}'
        has_filt_col = tmpl_mag_filt_col in templates.columns

        # Per-template mag for this filter: per-filter column where available,
        # else primary mag_wmean for same-filter templates, else NaN.
        if has_filt_col:
            tmpl_mag_for_img = templates[tmpl_mag_filt_col].values.astype(float)
            tmpl_merr_for_img = (
                templates[tmpl_merr_filt_col].values.astype(float)
                if tmpl_merr_filt_col in templates.columns
                else np.full(len(templates), 0.05)
            )
        elif 'filter' in templates.columns:
            same_filt_arr = templates['filter'].values == img_filt
            tmpl_mag_for_img = np.where(
                same_filt_arr,
                templates['mag_wmean'].values.astype(float) if 'mag_wmean' in templates.columns
                else np.full(len(templates), np.nan),
                np.nan,
            )
            tmpl_merr_for_img = np.where(
                same_filt_arr,
                templates['mag_werr'].values.astype(float) if 'mag_werr' in templates.columns
                else np.full(len(templates), 0.05),
                0.05,
            )
        else:
            tmpl_mag_for_img  = np.full(len(templates), np.nan)
            tmpl_merr_for_img = np.full(len(templates), 0.05)

        # Templates with a usable magnitude for this filter (any filter)
        has_mag_mask = np.isfinite(tmpl_mag_for_img)

        # ── ZP estimation: use templates with magnitude in this filter ────
        zp_offset = 0.0
        if has_mag_mask.sum() >= 3:
            tree_tmpl = cKDTree(
                np.column_stack([pred_ra[has_mag_mask] * cos_dec0,
                                 pred_dec[has_mag_mask]]))
            tree_det_arr  = np.column_stack([det_ra * cos_dec0, det_dec])
            nn_d, nn_i = tree_tmpl.query(tree_det_arr, k=1)
            # Pairs within 2 arcsec (rough ZP estimate)
            close = nn_d < (2.0 / 3600.0)
            if close.sum() >= 3:
                tmpl_mags = tmpl_mag_for_img[has_mag_mask][nn_i[close]]
                det_mags  = det_mag[close]
                delta_mags = det_mags - tmpl_mags
                delta_mags = delta_mags[np.abs(delta_mags) < 1.0]
                if len(delta_mags) >= 3:
                    zp_offset = float(np.median(delta_mags))

        det_mag_zp = det_mag - zp_offset

        # ── Match: for each template, find nearest unmatched detection ────
        # Use a KD-tree over detections, query per-template with its own radius.
        cos_img = np.cos(np.deg2rad(np.mean(det_dec))) if len(det_dec) else cos_dec0
        xy_det  = np.column_stack([det_ra * cos_img, det_dec])
        tree_det_kd = cKDTree(xy_det)

        local_claimed: dict[int, int] = {}  # free_row_idx → template_ti

        for ti in range(len(templates)):
            # Skip templates that already have a detection in this image
            sub_claimed = template_claimed[ti]
            prefix = sub_name + ':'
            if any(k.startswith(prefix) for k in sub_claimed):
                continue

            radius_deg = match_n_sigma * max(pred_sig_ra[ti], pred_sig_dec[ti])
            xy_pred = np.array([[pred_ra[ti] * cos_img, pred_dec[ti]]])
            idxs = tree_det_kd.query_ball_point(xy_pred[0], r=radius_deg)

            if not idxs:
                continue

            # Among candidates, find best by combined position + magnitude score
            best_det_local, best_score = None, np.inf
            for local_idx in idxs:
                if local_idx in local_claimed:
                    continue
                # Elliptical distance in sigma
                dra  = (det_ra[local_idx]  - pred_ra[ti])  * cos_img
                ddec =  det_dec[local_idx] - pred_dec[ti]
                sig  = max(pred_sig_ra[ti], pred_sig_dec[ti], 1e-12)
                pos_score = (dra**2 + ddec**2) / sig**2

                # Magnitude match: use the template's magnitude in this filter
                # if known (per-filter column or same-filter primary mag).
                # Skip magnitude check when the template has no magnitude for
                # this filter (true cross-filter with no overlap photometry).
                if has_mag_mask[ti]:
                    combined_sig_mag = max(
                        mag_floor,
                        mag_n_sigma * float(np.hypot(
                            det_merr[local_idx],
                            tmpl_merr_for_img[ti],
                        ))
                    )
                    if abs(det_mag_zp[local_idx] - tmpl_mag_for_img[ti]) > combined_sig_mag:
                        continue

                score = pos_score
                if score < best_score:
                    best_score  = score
                    best_det_local = local_idx

            if best_det_local is not None:
                # Resolve conflicts: if two templates want the same detection,
                # the closer one wins (lower pos_score).
                existing = local_claimed.get(best_det_local)
                if existing is None:
                    local_claimed[best_det_local] = ti
                else:
                    # Already claimed — compare scores
                    prev_ti = existing
                    dra_prev = (det_ra[best_det_local] - pred_ra[prev_ti]) * cos_img
                    ddec_prev = det_dec[best_det_local] - pred_dec[prev_ti]
                    sig_prev = max(pred_sig_ra[prev_ti], pred_sig_dec[prev_ti], 1e-12)
                    prev_score = (dra_prev**2 + ddec_prev**2) / sig_prev**2
                    if best_score < prev_score:
                        local_claimed[best_det_local] = ti

        # Commit local_claimed: map local free_row indices back to global det rows
        for local_idx, ti in local_claimed.items():
            det_row = int(free_rows[local_idx])
            new_assignments.append((det_row, ti))
            claimed_globally.add(det_row)
            # Record claim in template
            det_key_val = (f"{sub_name}:"
                           f"{det_df['catalog_index'].values[det_row]}")
            template_claimed[ti].add(det_key_val)

    # ── Rebuild detection sets for each template ──────────────────────────────
    # Build a dict: template_ti → list of det_row indices
    tmpl_det_rows: dict[int, list[int]] = {ti: [] for ti in range(len(templates))}

    # Seed with existing first-pass detections
    for ti in range(len(templates)):
        for sname, cidx in hst_pairs_lookup.get(ti, []):
            det_row = key_to_detrow.get(f'{sname}:{cidx}')
            if det_row is not None:
                tmpl_det_rows[ti].append(det_row)

    # Add new second-pass matches
    for det_row, ti in new_assignments:
        if det_row not in tmpl_det_rows[ti]:
            tmpl_det_rows[ti].append(det_row)

    # Det rows already in the first pass (across all templates)
    first_pass_keys: set[int] = set()
    for ti in range(len(templates)):
        for sname, cidx in hst_pairs_lookup.get(ti, []):
            k = key_to_detrow.get(f'{sname}:{cidx}')
            if k is not None:
                first_pass_keys.add(k)

    n_new_total = sum(1 for det_row, _ in new_assignments
                      if det_row not in first_pass_keys)

    # Per-template first-pass detection count (for summary comparison)
    first_pass_n = np.array([
        sum(1 for sname, cidx in hst_pairs_lookup.get(ti, [])
            if key_to_detrow.get(f'{sname}:{cidx}') is not None)
        for ti in range(len(templates))
    ])

    # ── Re-fit astrometry for each template ──────────────────────────────────
    v2_rows = []
    for ti, trow in templates.iterrows():
        det_rows_ti = tmpl_det_rows[ti]
        if len(det_rows_ti) < min_detections:
            # Not enough detections after merge; dropped (falls back to v1)
            v2_rows.append(None)
            continue

        snames_g = det_df['sub_name'].values[np.array(det_rows_ti)]
        cidxs_g  = det_df['catalog_index'].values[np.array(det_rows_ti)]
        det_indices_str = ','.join(f'{s}:{c}' for s, c in zip(snames_g, cidxs_g))

        row: dict = {'n_detect_v2': len(det_rows_ti)}

        # pass2_hst_indices is set only when the detection set changed so that
        # run_hst_crossmatch knows which sources need a Phase 4 re-fit.
        # _parse_hst_indices_columns won't pick this up (doesn't start with
        # 'hst_indices_'), so there is no risk of double-counting detections.
        if len(det_rows_ti) != first_pass_n[ti]:
            row['pass2_hst_indices'] = det_indices_str

        v2_rows.append(row)

    # ── Attach v2 columns to templates, then merge back into combined_df ─────
    v2_df = pd.DataFrame(
        [r if r is not None else {} for r in v2_rows],
        index=templates.index
    )
    templates_v2 = pd.concat([templates, v2_df], axis=1)

    # Non-template sources stay as-is (no v2 columns)
    non_templates = combined_df[~ok_pm].copy()
    for col in v2_df.columns:
        non_templates[col] = np.nan

    result = pd.concat([templates_v2, non_templates], ignore_index=True)

    # ── Summary statistics ────────────────────────────────────────────────────
    v2_n        = np.array([len(tmpl_det_rows[ti]) for ti in range(len(templates))])
    gained_mask = v2_n > first_pass_n
    lost_mask   = (v2_n < first_pass_n) & (v2_n >= min_detections)
    drop_mask   = v2_n < min_detections
    n_gained    = int(gained_mask.sum())
    n_lost      = int(lost_mask.sum())
    n_dropped   = int(drop_mask.sum())
    n_unchanged = len(templates) - n_gained - n_lost - n_dropped
    det_added   = int(np.sum(np.maximum(v2_n - first_pass_n, 0)))
    det_removed = int(np.sum(np.maximum(first_pass_n - v2_n, 0)[~drop_mask]))

    print(f"\n  Phase 5 summary")
    print(f"  {'─'*50}")
    print(f"  Template sources:  {len(templates)}")
    print(f"    gained ≥1 det :  {n_gained}  (+{det_added} total detections added)")
    print(f"    lost   ≥1 det :  {n_lost}  (-{det_removed} detections removed)")
    print(f"    unchanged     :  {n_unchanged}")
    print(f"    dropped (<{min_detections}):  {n_dropped}")
    if n_gained > 0:
        gain_vals = (v2_n - first_pass_n)[gained_mask]
        print(f"    det gain (med/max): +{np.median(gain_vals):.0f} / +{gain_vals.max()}")
    print(f"  New assignments:  {n_new_total} genuinely new  "
          f"({len(new_assignments)} total, {len(new_assignments)-n_new_total} already had)")
    print(f"  Images searched:  {len(all_sub_names)}")
    print(f"  (Phase 4 proper astrometry will be re-run for {n_gained + n_lost} "
          f"changed sources in run_hst_crossmatch)")
    print(f"  {'─'*50}")

    return result


# ── Main entry point ──────────────────────────────────────────────────────────

def run_hst_crossmatch(
    field_dir: Path,
    output_dir: Optional[Path]  = None,
    gaia_csv: Optional[Path]    = None,
    max_pm_masyr: float         = 100.,
    match_n_sigma: float        = 5.,
    mag_n_sigma: float          = 3.0,
    mag_floor: float            = 0.10,
    min_detections: int         = 2,
    cross_filter_radius_mas: float = 200.,
    gaia_recovery_radius_mas: float = 100.,
    save_detections: bool       = True,
    run_second_pass: bool       = True,
    second_pass_max_pm_unc: float = 10.0,
    bp3m_results_dir: Optional[Path] = None,
    anchor_bp3m_dir: Optional[Path] = None,
    phase4_outlier_sigma: float = 3.5,
    cycle_id: int = 0,
) -> dict[str, pd.DataFrame]:
    """
    Run the full HST cross-match pipeline for a field.

    Parameters
    ----------
    field_dir        : root directory of the processed field
    output_dir       : where to write outputs (default: field_dir/hst_xmatch)
    gaia_csv         : path to the Gaia catalog CSV for Gaia recovery (optional)
    max_pm_masyr     : maximum allowed proper motion for within-filter matching
    match_n_sigma    : match radius in units of combined astrometric sigma
    mag_n_sigma      : magnitude match threshold in units of combined photometric sigma
    mag_floor        : minimum magnitude tolerance regardless of photometric error (mag)
    min_detections   : minimum detections for a source to appear in the master catalog
    cross_filter_radius_mas  : match radius for cross-filter association (mas)
    gaia_recovery_radius_mas : match radius for Gaia recovery (mas)
    save_detections  : write per-filter detection catalogs to disk
    bp3m_results_dir : override BP3M results directory (default: field_dir/BP3M_results)
    anchor_bp3m_dir  : directory with V1 BP3M results used for Phase 0b Gaia anchoring
                       (default: same as bp3m_results_dir).  Pass the V1 BP3M dir here
                       when calling the second crossmatch (with v2 bp3m_results_dir) so
                       the anchor step always recovers the original V1 Gaia matches.
    phase4_outlier_sigma : per-detection chi2 sigma threshold for Phase 4 / 4b / 5
        outlier rejection (default 3.5; formerly hardcoded at 5.0)
    cycle_id : int, default 0
        Refinement cycle number appended to plot filenames so successive runs
        don't overwrite each other.  0 = initial crossmatch (using v1 BP3M),
        1 = after first v2 alignment, 2 = after second v2 alignment, etc.

    Returns
    -------
    dict with keys: 'detections', 'filter_masters', 'combined', 'gaia_recovered'
    """
    field_dir = Path(field_dir)
    if output_dir is None:
        output_dir = field_dir / 'hst_xmatch'
    output_dir = Path(output_dir)
    if bp3m_results_dir is None:
        bp3m_results_dir = field_dir / 'BP3M_results'
    bp3m_results_dir = Path(bp3m_results_dir)
    # anchor_bp3m_dir defaults to the same as bp3m_results_dir but can be
    # overridden to always anchor from V1 even when doing the v2 crossmatch.
    _anchor_dir = Path(anchor_bp3m_dir) if anchor_bp3m_dir is not None else bp3m_results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"HST catalog cross-match: {field_dir.name}")
    print(f"{'='*60}")

    # ── Phase 0: load all detections ─────────────────────────────────────────
    print("\nPhase 0: Loading all HST detections and projecting to RA,Dec ...")
    det_df = _load_all_detections(field_dir, bp3m_results_dir=bp3m_results_dir)
    if det_df is None or len(det_df) == 0:
        print("  No detections found.")
        return {}

    print(f"  Loaded {len(det_df)} detections from "
          f"{det_df['sub_name'].nunique()} sub-images "
          f"in {det_df['filter'].nunique()} filter(s)")

    if save_detections:
        for filt, fdf in det_df.groupby('filter'):
            out_path = output_dir / f'detections_{filt}.csv'
            fdf.to_csv(out_path, index=False)
            print(f"  Saved {len(fdf)} detections → {out_path.name}")

    # ── Resolve Gaia CSV early (needed by both Phase 2 and Phase 5) ──────────
    if gaia_csv is None:
        gaia_dir = field_dir / 'Gaia'
        if gaia_dir.exists():
            gaia_files = sorted(f for f in gaia_dir.glob('*_gaia.csv')
                                if not f.name.startswith('._'))
            if gaia_files:
                gaia_csv = gaia_files[0]
                print(f"  Using Gaia catalog: {gaia_csv.name}")

    # ── Gaia catalog consistency check ───────────────────────────────────────
    # Every Gaia source_id in det_df must exist in the local Gaia CSV.
    # A mismatch indicates a Gaia ID precision bug (float64 roundtrip corrupting
    # 19-digit IDs).  Raise clearly rather than producing a silent wrong answer.
    if gaia_csv is not None and Path(gaia_csv).exists() and 'gaia_source_id' in det_df.columns:
        try:
            _gcat_ids = pd.read_csv(gaia_csv,
                                    usecols=lambda c: c.lower() in ('source_id',),
                                    dtype={'source_id': np.int64},
                                    low_memory=False)
            if _gcat_ids.empty:
                # Try uppercase column name
                _gcat_ids = pd.read_csv(gaia_csv,
                                        usecols=lambda c: c.upper() == 'SOURCE_ID',
                                        low_memory=False)
                _gcat_ids.columns = ['source_id']
                _gcat_ids['source_id'] = _gcat_ids['source_id'].astype(np.int64)
            _local_gids = set(_gcat_ids['source_id'].values)
            _has_match  = det_df['has_gaia_match'].values.astype(bool)
            _gids       = det_df['gaia_source_id'].astype(np.int64).values
            _not_local  = _has_match & (_gids > 0) & ~np.isin(_gids, list(_local_gids))
            n_bad = int(_not_local.sum())
            if n_bad > 0:
                print(f"  ERROR: {n_bad} per-image Gaia matches reference source_ids "
                      f"NOT in the local Gaia CSV — likely a Gaia ID int64 precision "
                      f"bug in matched_gaia.csv or Gaia CSV (float64 roundtrip).  "
                      f"Check that matched_gaia.csv was saved with int64 source_ids.")
                # Demote the bad ones so they don't propagate as false Gaia matches.
                det_df = det_df.copy()
                det_df.loc[_not_local, 'has_gaia_match'] = False
                det_df.loc[_not_local, 'gaia_source_id'] = np.int64(0)
        except Exception as _e:
            print(f"  Warning: Gaia consistency check failed: {_e}")

    # ── Phase 1: V1 BP3M-anchored Gaia detection ─────────────────────────────
    # For every V1 Gaia-matched star, search ALL sub-images using V1 BP3M PM
    # for position propagation.  This ensures all Gaia stars from V1 are found
    # in the maximum number of images before Phase 3's min_detections cut.
    print("\nPhase 1: V1 BP3M-anchored Gaia star detection ...")
    # Collect the V1 Gaia IDs so Phase 2 can skip them
    _v1_anchor_ids: set = set()
    _v1_astrom_path = _anchor_dir / 'stellar_astrometry.csv'
    if _v1_astrom_path.exists():
        _v1_a = pd.read_csv(_v1_astrom_path, usecols=['Gaia_id'],
                             dtype={'Gaia_id': np.int64})
        _v1_anchor_ids = set(int(g) for g in _v1_a['Gaia_id'].values if int(g) > 0)
    det_df = _phase0b_anchor_gaia_stars(
        det_df, _anchor_dir, search_radius_px=50.0, n_candidates=5)

    # ── Phase 2: Gaia catalog anchoring for non-V1-BP3M stars ────────────────
    # For Gaia catalog stars NOT covered by Phase 1, search using Gaia catalog
    # PM.  Includes new Gaia sources and 2p stars not measured in V1 BP3M.
    print("\nPhase 2: Gaia catalog anchoring (non-V1 stars) ...")
    det_df = _phase2_gaia_catalog_anchor(
        det_df, gaia_csv=gaia_csv,
        anchor_gaia_ids=_v1_anchor_ids,
        search_radius_px=50.0, n_candidates=5)

    # ── Phase 3: within-filter matching ──────────────────────────────────────
    print("\nPhase 3: Within-filter cross-matching ...")
    filter_masters = _within_filter_match(
        det_df,
        max_pm_masyr=max_pm_masyr,
        match_n_sigma=match_n_sigma,
        mag_n_sigma=mag_n_sigma,
        mag_floor=mag_floor,
        min_detections=min_detections,
        pre_zp_applied=det_df.attrs.get('pre_zp_applied', False),
    )
    for filt, master_df in filter_masters.items():
        out_path = output_dir / f'master_{filt}.csv'
        if 'gaia_source_id' in master_df.columns:
            master_df['gaia_source_id'] = master_df['gaia_source_id'].fillna(0).astype(np.int64)
        master_df.to_csv(out_path, index=False)
        print(f"  Saved {len(master_df)} sources → {out_path.name}")

    # ── Phase 2: cross-filter matching ───────────────────────────────────────
    print("\nPhase 4: Cross-filter matching ...")
    combined_df = _cross_filter_match(
        filter_masters,
        match_n_sigma=match_n_sigma,
        match_radius_mas=cross_filter_radius_mas,
    )
    if len(combined_df) > 0:
        out_path = output_dir / 'master_combined.csv'
        if 'gaia_source_id' in combined_df.columns:
            combined_df['gaia_source_id'] = combined_df['gaia_source_id'].fillna(0).astype(np.int64)
        combined_df.to_csv(out_path, index=False)
        print(f"  {len(combined_df)} unique sources across all filters")
        print(f"  Saved → {out_path.name}")

    # ── Phase 5: Gaia recovery (post-crossmatch) ─────────────────────────────
    # Re-attempt Gaia matching for sources that survived Phases 3-4 but were
    # not labelled in Phases 1-2 (e.g. sources first seen in weak detections
    # that required the PM fit from Phase 4 to confirm their identity).
    print("\nPhase 5: Gaia recovery (post-crossmatch) ...")
    # Load Gaia catalog for CMD plots (and pass to recovery)
    gaia_df: Optional[pd.DataFrame] = None
    if gaia_csv is not None and Path(gaia_csv).exists():
        try:
            gaia_df = pd.read_csv(gaia_csv)
            gaia_df.columns = [c.lower() for c in gaia_df.columns]
        except Exception:
            pass

    recovered_df = _recover_gaia_matches(
        combined_df if len(combined_df) > 0 else pd.concat(list(filter_masters.values()), ignore_index=True),
        gaia_csv=gaia_csv,
        match_radius_mas=gaia_recovery_radius_mas,
    )
    if len(recovered_df) > 0:
        out_path = output_dir / 'gaia_recovered.csv'
        recovered_df.to_csv(out_path, index=False)
        print(f"  Saved {len(recovered_df)} recovered sources → {out_path.name}")

    # ── Phase 4: proper astrometry with full C_r treatment ───────────────────
    print("\nPhase 6: Proper astrometry with full transformation covariance ...")
    bp3m_dir = bp3m_results_dir
    astrom_df = pd.DataFrame()
    _p4: dict = {}   # stashed for Phase 5 re-use
    try:
        img_csv  = bp3m_dir / 'image_transformations.csv'
        c_r_path = bp3m_dir / 'C_r.npy'
        if not img_csv.exists() or not c_r_path.exists():
            print("  Skipping Phase 4: image_transformations.csv or C_r.npy missing")
        elif len(combined_df) == 0:
            print("  Skipping Phase 4: no sources in combined catalog")
        else:
            import json as _json
            transform_df4 = pd.read_csv(img_csv)
            C_r_full      = np.load(c_r_path)
            n_sub4        = len(transform_df4)
            n_r4          = C_r_full.shape[0] // n_sub4
            poly_order4   = _infer_poly_order(n_r4)

            # Reorder if run_config.json is available
            run_cfg_path = bp3m_dir / 'run_config.json'
            if run_cfg_path.exists():
                with open(run_cfg_path) as _f:
                    run_cfg4 = _json.load(_f)
                if 'poly_order' in run_cfg4:
                    poly_order4 = int(run_cfg4['poly_order'])
                if 'image_names' in run_cfg4:
                    saved_names = run_cfg4['image_names']
                    if set(saved_names) == set(transform_df4['image_name'].tolist()):
                        transform_df4 = (transform_df4
                                         .set_index('image_name')
                                         .loc[saved_names]
                                         .reset_index())

            # Build r_hat array from image_transformations.csv
            r_params = ['a', 'b', 'c', 'd', 'w', 'z', 'delta_ra0_mas', 'delta_dec0_mas']
            r_hat4   = np.concatenate([
                np.array([getattr(row, p) for p in r_params])
                for row in transform_df4.itertuples()
            ])
            image_names4 = transform_df4['image_name'].tolist()

            # Build per-sub-image metadata: sub_name → (ra0_deg, dec0_deg, pscale_mas)
            # transformation.csv is per obs_id directory; both _hi and _lo chips share it.
            hst_root4 = field_dir / 'HST' / 'mastDownload' / 'HST'
            sub_img_meta4: dict[str, tuple[float, float, float]] = {}
            pscale4 = 50.0  # ACS/WFC default; overwritten per image below
            for img_dir4 in sorted(hst_root4.iterdir()):
                t4 = img_dir4 / 'transformation.csv'
                if not t4.exists():
                    continue
                try:
                    try:
                        tdf4 = pd.read_csv(t4).set_index('parameter')['value']
                        ra4   = float(tdf4['ra_cen'])
                        dec4  = float(tdf4['dec_cen'])
                        ps4   = float(tdf4['pixel_scale']) * 1000.0  # arcsec → mas
                    except (KeyError, ValueError):
                        tdf4 = pd.read_csv(t4)
                        ra4   = float(tdf4['ra_cen'].iloc[0])
                        dec4  = float(tdf4['dec_cen'].iloc[0])
                        ps4   = float(tdf4['pixel_scale'].iloc[0]) * 1000.0
                    pscale4 = ps4  # remember last valid pscale as fallback
                    base4 = img_dir4.name
                    for sfx in ('_hi', '_lo', ''):
                        sub_img_meta4[base4 + sfx] = (ra4, dec4, ps4)
                    sub_img_meta4[base4] = (ra4, dec4, ps4)
                except Exception:
                    pass

            # Field centre fallback (used only for sub-images without a transformation.csv)
            ra0_field  = float(combined_df['ra0'].mean()) if 'ra0' in combined_df.columns else 0.0
            dec0_field = float(combined_df['dec0'].mean()) if 'dec0' in combined_df.columns else 0.0

            n_meta = len(sub_img_meta4)
            print(f"  {len(combined_df)} sources, {len(image_names4)} sub-images, "
                  f"{n_meta} with per-image transform metadata, poly_order={poly_order4}")

            # Pre-build the expensive lookups once; all subsequent calls reuse them
            print("  [phase4] pre-building shared lookups ...")
            _t_pre = time.perf_counter()
            _shared_det_lookup = {}
            for _row in det_df.to_dict('records'):
                _shared_det_lookup[(str(_row['sub_name']), int(_row['catalog_index']))] = _row
            print(f"  [phase4]   det_lookup built: {time.perf_counter()-_t_pre:.2f}s  "
                  f"({len(_shared_det_lookup)} entries)")

            _t_pre = time.perf_counter()
            _img_mjd_pre: dict[str, float] = {}
            for _row in det_df[['sub_name', 'epoch_mjd']].drop_duplicates('sub_name').to_dict('records'):
                _img_mjd_pre[str(_row['sub_name'])] = float(_row['epoch_mjd'])

            def _fetch_xyz_pre(sname_mjd):
                sname, mjd = sname_mjd
                try:
                    return sname, get_tele_position(AstropyTime(mjd, format='mjd'), curr_id='earth')
                except Exception:
                    return sname, np.zeros(3)

            _shared_tele_xyz: dict[str, np.ndarray] = {}
            with ThreadPoolExecutor(max_workers=min(8, len(_img_mjd_pre))) as _pool:
                for _sname, _xyz in _pool.map(_fetch_xyz_pre, _img_mjd_pre.items()):
                    _shared_tele_xyz[_sname] = _xyz
            print(f"  [phase4]   tele_xyz built:   {time.perf_counter()-_t_pre:.2f}s  "
                  f"({len(_shared_tele_xyz)} images)")

            _t_pre = time.perf_counter()
            _shared_src_detections = _parse_hst_indices_columns(combined_df)
            print(f"  [phase4]   src_detections:   {time.perf_counter()-_t_pre:.2f}s  "
                  f"({len(_shared_src_detections)} entries)")

            astrom_df = _measure_astrometry_proper(
                combined_df      = combined_df,
                det_df           = det_df,
                r_hat_arr        = r_hat4,
                C_r              = C_r_full,
                image_names      = image_names4,
                n_r              = n_r4,
                poly_order       = poly_order4,
                ra0_field        = ra0_field,
                dec0_field       = dec0_field,
                pscale           = pscale4,
                sub_img_meta     = sub_img_meta4,
                gaia_df          = gaia_df,
                outlier_sigma    = phase4_outlier_sigma,
                _det_lookup      = _shared_det_lookup,
                _tele_xyz_cache  = _shared_tele_xyz,
                _src_detections  = _shared_src_detections,
            )

            n_good = int(np.isfinite(astrom_df['ra_xmatch']).sum())
            n_pm   = int(np.isfinite(astrom_df['pmra_xmatch']).sum())
            print(f"  Phase 4: {n_good} position fits, {n_pm} with PM constraints")

            # Merge back onto combined_df
            combined_df = combined_df.join(astrom_df, how='left')

            # Stash all Phase 4 ingredients so Phase 5 can re-use them
            _p4 = dict(
                r_hat           = r_hat4,
                C_r             = C_r_full,
                image_names     = image_names4,
                n_r             = n_r4,
                poly_order      = poly_order4,
                sub_img_meta    = sub_img_meta4,
                ra0_field       = ra0_field,
                dec0_field      = dec0_field,
                pscale          = pscale4,
                det_lookup      = _shared_det_lookup,
                tele_xyz_cache  = _shared_tele_xyz,
                src_detections  = _shared_src_detections,
            )

            # ── Phase 4b: post-astrometry deduplication (ra_xmatch positions) ────
            # Phase 2 deduplication uses Phase-1 ra0/dec0 positions, which can
            # have large inter-filter offsets (hundreds of mas) for fields where
            # different filter groups come from programs with different pointings.
            # After Phase 4 applies the BP3M transformation, ra_xmatch/dec_xmatch
            # are accurate to ~mas level, so a second dedup pass catches the
            # remaining cross-filter close pairs that Phase 2 missed.
            has_ra_xmatch = 'ra_xmatch' in combined_df.columns and combined_df['ra_xmatch'].notna().any()
            if has_ra_xmatch:
                n_before_4b = len(combined_df)
                # Temporarily map ra0/dec0 to ra_xmatch/dec_xmatch for position query
                _tmp = combined_df.copy()
                _valid_xm = _tmp['ra_xmatch'].notna() & _tmp['dec_xmatch'].notna()
                _tmp.loc[_valid_xm, 'ra0']  = _tmp.loc[_valid_xm, 'ra_xmatch']
                _tmp.loc[_valid_xm, 'dec0'] = _tmp.loc[_valid_xm, 'dec_xmatch']
                combined_dedup4b = _deduplicate_merged(_tmp, pos_threshold_mas=50.0)
                n_merged_4b = n_before_4b - len(combined_dedup4b)
                if n_merged_4b > 0:
                    # Restore original ra0/dec0 on surviving rows (index-aligned)
                    for col in ('ra0', 'dec0', 'pmra', 'pmdec'):
                        if col in combined_df.columns:
                            combined_dedup4b[col] = combined_df.loc[
                                combined_dedup4b.index, col]
                    # Re-run Phase 4 only on sources whose detection count grew
                    # (index is preserved so this comparison is exact)
                    _old_ndet = combined_df.loc[combined_dedup4b.index, 'n_detect']
                    _changed_4b = combined_dedup4b['n_detect'] > _old_ndet.fillna(0)
                    n_refit_4b = int(_changed_4b.sum())
                    if n_refit_4b > 0:
                        _cdf_4b = combined_dedup4b[_changed_4b].copy()
                        _drop_astrom = [c for c in _cdf_4b.columns
                                        if c in ('ra_xmatch', 'dec_xmatch',
                                                 'pmra_xmatch', 'pmdec_xmatch',
                                                 'parallax_xmatch',
                                                 'sigma_ra_xmatch', 'sigma_dec_xmatch',
                                                 'sigma_pmra_xmatch', 'sigma_pmdec_xmatch',
                                                 'sigma_parallax_xmatch')]
                        _cdf_4b = _cdf_4b.drop(columns=_drop_astrom, errors='ignore')
                        _astrom_4b = _measure_astrometry_proper(
                            combined_df=_cdf_4b, det_df=det_df,
                            r_hat_arr=_p4['r_hat'], C_r=_p4['C_r'],
                            image_names=_p4['image_names'], n_r=_p4['n_r'],
                            poly_order=_p4['poly_order'], ra0_field=_p4['ra0_field'],
                            dec0_field=_p4['dec0_field'], pscale=_p4['pscale'],
                            sub_img_meta=_p4['sub_img_meta'], gaia_df=gaia_df,
                            outlier_sigma=phase4_outlier_sigma,
                            _det_lookup=_p4.get('det_lookup'),
                            _tele_xyz_cache=_p4.get('tele_xyz_cache'),
                            _src_detections=_p4.get('src_detections'),
                        )
                        for _col in _astrom_4b.columns:
                            combined_dedup4b.loc[_cdf_4b.index, _col] = _astrom_4b[_col].values
                    combined_df = combined_dedup4b.reset_index(drop=True)
                    print(f"  Phase 4b: merged {n_merged_4b} duplicate rows using ra_xmatch positions"
                          + (f"; re-fitted {n_refit_4b} sources" if n_refit_4b > 0 else ""))

            # Merge chi2_hst from BP3M stellar_astrometry.csv for Gaia-matched rows
            _astrom_csv = bp3m_dir / 'stellar_astrometry.csv'
            if _astrom_csv.exists() and 'gaia_source_id' in combined_df.columns:
                try:
                    _astrom = pd.read_csv(_astrom_csv, usecols=lambda c: c in
                                          ('Gaia_id', 'chi2_hst', 'n_det_chi2', 'chi2_hst_red'))
                    if 'chi2_hst' in _astrom.columns:
                        _astrom = _astrom[_astrom['Gaia_id'] > 0].copy()
                        _astrom['Gaia_id'] = _astrom['Gaia_id'].astype(np.int64)
                        _id_col = combined_df['gaia_source_id'].copy()
                        _id_col = pd.to_numeric(_id_col, errors='coerce').fillna(0).astype(np.int64)
                        _id_map = _astrom.set_index('Gaia_id')
                        for _c in ('chi2_hst', 'n_det_chi2', 'chi2_hst_red'):
                            if _c in _id_map.columns:
                                combined_df[_c] = _id_col.map(_id_map[_c])
                except Exception as _chi2_exc:
                    pass  # non-critical: chi2 columns simply absent

            # Re-save combined catalog with Phase 4 columns
            out_path = output_dir / 'master_combined.csv'
            if 'gaia_source_id' in combined_df.columns:
                combined_df['gaia_source_id'] = combined_df['gaia_source_id'].fillna(0).astype(np.int64)
            combined_df.to_csv(out_path, index=False)
            print(f"  Updated master_combined.csv with Phase 4 astrometry columns")

    except Exception as exc:
        print(f"  Warning: Phase 4 failed: {exc}")
        import traceback
        traceback.print_exc()

    # ── Phase 5: PM-guided second-pass cross-match ───────────────────────────
    combined_v2_df = combined_df
    if run_second_pass:
        print("\nPhase 7: PM-guided second-pass cross-match ...")
        try:
            _ra0_v2  = float(combined_df['ra0'].mean())  if 'ra0'  in combined_df.columns \
                       else float(det_df['ra'].mean())
            _dec0_v2 = float(combined_df['dec0'].mean()) if 'dec0' in combined_df.columns \
                       else float(det_df['dec'].mean())
            combined_v2_df = _second_pass_match(
                det_df           = det_df,
                combined_df      = combined_df,
                ra0              = _ra0_v2,
                dec0             = _dec0_v2,
                max_pm_unc_masyr = second_pass_max_pm_unc,
                match_n_sigma    = match_n_sigma,
                mag_n_sigma      = mag_n_sigma,
                mag_floor        = mag_floor,
                min_detections   = min_detections,
            )

            # ── Re-run Phase 4 proper astrometry on changed sources ───────────
            # Sources whose detection set changed have a 'pass2_hst_indices'
            # column set.  Re-run _measure_astrometry_proper with their updated
            # detection set so the v2 xmatch columns carry the same Gaia prior
            # and C_r marginalisation as Phase 4 — no WLS shortcuts.
            if _p4 and 'pass2_hst_indices' in combined_v2_df.columns:
                _changed = combined_v2_df['pass2_hst_indices'].notna()
                n_changed = int(_changed.sum())
                if n_changed > 0:
                    print(f"  Re-running Phase 4 proper astrometry on "
                          f"{n_changed} sources with updated detection sets ...")
                    # Build a minimal combined_df for the changed sources where
                    # the only hst_indices column is the updated pass2 set.
                    # This prevents _parse_hst_indices_columns from double-
                    # counting the old per-filter detection sets.
                    _cdf_changed = combined_v2_df[_changed].copy()
                    _drop_idx_cols = [c for c in _cdf_changed.columns
                                      if c.startswith('hst_indices_')
                                      or c == 'hst_indices']
                    _cdf_changed = _cdf_changed.drop(columns=_drop_idx_cols)
                    # Name it so _parse_hst_indices_columns picks it up
                    _cdf_changed['hst_indices_pass2'] = _cdf_changed['pass2_hst_indices']

                    _astrom_v2 = _measure_astrometry_proper(
                        combined_df     = _cdf_changed,
                        det_df          = det_df,
                        r_hat_arr       = _p4['r_hat'],
                        C_r             = _p4['C_r'],
                        image_names     = _p4['image_names'],
                        n_r             = _p4['n_r'],
                        poly_order      = _p4['poly_order'],
                        ra0_field       = _p4['ra0_field'],
                        dec0_field      = _p4['dec0_field'],
                        pscale          = _p4['pscale'],
                        sub_img_meta    = _p4['sub_img_meta'],
                        _det_lookup     = _p4.get('det_lookup'),
                        _tele_xyz_cache = _p4.get('tele_xyz_cache'),
                        _src_detections = None,  # different column structure for pass2
                        gaia_df      = gaia_df,
                        outlier_sigma = phase4_outlier_sigma,
                    )
                    # Overwrite xmatch columns for changed sources
                    _changed_idx = combined_v2_df.index[_changed]
                    for _col in _astrom_v2.columns:
                        combined_v2_df.loc[_changed_idx, _col] = _astrom_v2[_col].values
                    _n_pm_v2 = int(np.isfinite(
                        combined_v2_df.loc[_changed_idx, 'pmra_xmatch']).sum())
                    print(f"  Phase 5: {_n_pm_v2}/{n_changed} re-fitted sources "
                          f"have full PM constraints")

            _out_v2 = output_dir / 'master_combined_v2.csv'
            if 'gaia_source_id' in combined_v2_df.columns:
                combined_v2_df['gaia_source_id'] = combined_v2_df['gaia_source_id'].fillna(0).astype(np.int64)
            combined_v2_df.to_csv(_out_v2, index=False)
            print(f"  Saved → {_out_v2.name}")
        except Exception as _exc_v2:
            print(f"  Warning: Phase 5 failed: {_exc_v2}")
            import traceback; traceback.print_exc()
            combined_v2_df = combined_df

    # ── Diagnostic figures ────────────────────────────────────────────────────
    field_name = Path(field_dir).name

    # Restrict sky and CMD plots to sources with well-measured PMs only,
    # and further exclude |PM| > 100 mas/yr for plot clarity (measurements
    # are retained in combined_df; only the plot subset is filtered).
    _ok_pm = (np.isfinite(combined_df['pmra_xmatch'].values)
              & np.isfinite(combined_df['pmdec_xmatch'].values)) \
             if 'pmra_xmatch' in combined_df.columns else \
             np.zeros(len(combined_df), bool)
    if _ok_pm.sum() > 0:
        _pmra_v  = combined_df['pmra_xmatch'].values
        _pmdec_v = combined_df['pmdec_xmatch'].values
        _ok_pm  &= (np.abs(_pmra_v) <= 50.) & (np.abs(_pmdec_v) <= 50.)
    good_pm_df = combined_df[_ok_pm].reset_index(drop=True)
    _pm_size = np.sqrt(good_pm_df['pmra_xmatch'].values**2
                       + good_pm_df['pmdec_xmatch'].values**2) \
               if len(good_pm_df) > 0 else np.empty(0)

    _cy = f'_cycle{cycle_id}'   # e.g. '_cycle0', '_cycle1', …

    try:
        _plot_sky(good_pm_df,
                  output_dir / f'sky_distribution{_cy}.png',
                  title=f'{field_name} (cycle {cycle_id})',
                  pm_size=_pm_size)
        print(f"  sky_distribution{_cy}.png written")
    except Exception as _e:
        print(f"  Warning: sky_distribution{_cy}.png failed: {_e}")

    try:
        _plot_vpd(good_pm_df,
                  output_dir / f'vpd{_cy}.png',
                  title=f'{field_name} (cycle {cycle_id})')
        print(f"  vpd{_cy}.png written")
    except Exception as _e:
        print(f"  Warning: vpd{_cy}.png failed: {_e}")

    try:
        _plot_cmds(good_pm_df, gaia_df,
                   output_dir / f'cmds{_cy}.png',
                   title=f'{field_name} (cycle {cycle_id})',
                   pm_size=_pm_size)
        print(f"  cmds{_cy}.png written")
    except Exception as _e:
        print(f"  Warning: cmds{_cy}.png failed: {_e}")

    if gaia_df is not None and 'gaia_source_id' in combined_df.columns:
        try:
            _plot_gaia_comparison(good_pm_df, gaia_df,
                                  output_dir / f'gaia_comparison{_cy}.png',
                                  title=f'{field_name} (cycle {cycle_id})')
            print(f"  gaia_comparison{_cy}.png written")
        except Exception as _e:
            print(f"  Warning: gaia_comparison{_cy}.png failed: {_e}")
            import traceback; traceback.print_exc()

    # ── Phase 5 diagnostic figures (v2 catalogue) ────────────────────────────
    if run_second_pass and 'pass2_hst_indices' in combined_v2_df.columns:
        # Use Phase 4 xmatch PMs throughout — the v2 WLS refit is lower quality
        # (no Gaia prior, no C_r marginalisation) and would degrade the plots.
        # The v2 catalogue's value is the updated source list; sky positions are
        # updated from ra0_v2/dec0_v2 where available.
        _pv2 = combined_v2_df.copy()
        for _dst, _src in [('ra_xmatch', 'ra0_v2'), ('dec_xmatch', 'dec0_v2')]:
            if _src in _pv2.columns and _dst in _pv2.columns:
                _pv2[_dst] = _pv2[_src].combine_first(_pv2[_dst])

        _ok_pm_v2 = (
            np.isfinite(_pv2['pmra_xmatch'].values)
            & np.isfinite(_pv2['pmdec_xmatch'].values)
            & (np.abs(_pv2['pmra_xmatch'].values) <= 50.)
            & (np.abs(_pv2['pmdec_xmatch'].values) <= 50.)
        ) if 'pmra_xmatch' in _pv2.columns else np.zeros(len(_pv2), bool)
        _good_pm_v2 = _pv2[_ok_pm_v2].reset_index(drop=True)
        _pm_size_v2 = (
            np.sqrt(_good_pm_v2['pmra_xmatch'].values**2
                    + _good_pm_v2['pmdec_xmatch'].values**2)
            if len(_good_pm_v2) > 0 else np.empty(0)
        )

        try:
            _plot_sky(_good_pm_v2, output_dir / f'sky_distribution_v2{_cy}.png',
                      title=f'{field_name} v2 (cycle {cycle_id})', pm_size=_pm_size_v2)
            print(f"  sky_distribution_v2{_cy}.png written")
        except Exception as _e:
            print(f"  Warning: sky_distribution_v2{_cy}.png failed: {_e}")

        try:
            _plot_vpd(_good_pm_v2, output_dir / f'vpd_v2{_cy}.png',
                      title=f'{field_name} v2 (cycle {cycle_id})')
            print(f"  vpd_v2{_cy}.png written")
        except Exception as _e:
            print(f"  Warning: vpd_v2{_cy}.png failed: {_e}")

        try:
            _plot_cmds(_good_pm_v2, gaia_df,
                       output_dir / f'cmds_v2{_cy}.png',
                       title=f'{field_name} v2 (cycle {cycle_id})', pm_size=_pm_size_v2)
            print(f"  cmds_v2{_cy}.png written")
        except Exception as _e:
            print(f"  Warning: cmds_v2{_cy}.png failed: {_e}")

        if gaia_df is not None and 'gaia_source_id' in _good_pm_v2.columns:
            try:
                _plot_gaia_comparison(_good_pm_v2, gaia_df,
                                      output_dir / f'gaia_comparison_v2{_cy}.png',
                                      title=f'{field_name} v2 (cycle {cycle_id})')
                print(f"  gaia_comparison_v2{_cy}.png written")
            except Exception as _e:
                print(f"  Warning: gaia_comparison_v2{_cy}.png failed: {_e}")

    print(f"\nOutputs written to: {output_dir}")
    return {
        'detections':     det_df,
        'filter_masters': filter_masters,
        'combined':       combined_df,
        'combined_v2':    combined_v2_df,
        'gaia_recovered': recovered_df,
    }


# ── Diagnostic figures ────────────────────────────────────────────────────────

def _plot_sky(combined_df: pd.DataFrame, output_path: Path, title: str = '',
              pm_size: Optional[np.ndarray] = None) -> None:
    """RA/Dec scatter coloured by log(PM size) when pm_size is provided."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    ra  = combined_df['ra_xmatch'].values  if 'ra_xmatch'  in combined_df.columns else combined_df.get('ra',  None)
    dec = combined_df['dec_xmatch'].values if 'dec_xmatch' in combined_df.columns else combined_df.get('dec', None)
    if ra is None or dec is None:
        return

    ok = np.isfinite(ra) & np.isfinite(dec)
    if pm_size is not None and len(pm_size) == len(ra):
        ok &= np.isfinite(pm_size) & (pm_size > 0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor('#e8e8e8')
    if pm_size is not None and ok.sum() > 0:
        c = pm_size[ok]
        vmin, vmax = np.percentile(c, [2, 98])
        vmin = max(vmin, 0.01)
        sc = ax.scatter(ra[ok], dec[ok], c=c, s=1.5, alpha=0.6,
                        norm=LogNorm(vmin=vmin, vmax=vmax),
                        cmap='plasma', rasterized=True)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(r'$|\mu|$ (mas/yr)')
    else:
        ax.scatter(ra[ok], dec[ok], s=1.5, alpha=0.4, color='steelblue', rasterized=True)
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    ax.invert_xaxis()
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, which='major', lw=0.4, alpha=0.5)
    ax.grid(True, which='minor', lw=0.2, alpha=0.3, linestyle=':')
    ax.minorticks_on()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def _plot_vpd(combined_df: pd.DataFrame, output_path: Path, title: str = '') -> None:
    """VPD with four zoom panels plus σ_PM_geom vs magnitude per filter."""
    import matplotlib.pyplot as plt

    pmra  = combined_df['pmra_xmatch'].values  if 'pmra_xmatch'  in combined_df.columns else None
    pmdec = combined_df['pmdec_xmatch'].values if 'pmdec_xmatch' in combined_df.columns else None
    if pmra is None or pmdec is None:
        return

    ok_pm = np.isfinite(pmra) & np.isfinite(pmdec)
    if ok_pm.sum() < 3:
        return
    x, y = pmra[ok_pm], pmdec[ok_pm]

    # For zoom panels, work within ±50 mas/yr so percentile widths capture
    # the slow-moving population rather than being dominated by fast outliers.
    _inner = (np.abs(x) <= 50.) & (np.abs(y) <= 50.)
    x_zoom = x[_inner] if _inner.sum() >= 3 else x
    y_zoom = y[_inner] if _inner.sum() >= 3 else y

    # σ_PM geometric mean: sqrt(σ_pmra × σ_pmdec)
    _sra = combined_df.get('sigma_pmra_xmatch', pd.Series(dtype=float)).values.astype(float)
    _sdc = combined_df.get('sigma_pmdec_xmatch', pd.Series(dtype=float)).values.astype(float)
    sigma_geom = np.where(
        np.isfinite(_sra) & np.isfinite(_sdc) & (_sra > 0) & (_sdc > 0),
        np.sqrt(_sra * _sdc), np.nan)

    # Collect per-filter magnitude columns, sorted by wavelength
    _WL_VPD: dict[str, float] = {
        'F225W': 237, 'F275W': 271, 'F336W': 336, 'F390W': 392,
        'F435W': 432, 'F438W': 438, 'F475W': 476, 'F555W': 531,
        'F606W': 592, 'F625W': 626, 'F775W': 764, 'F814W': 803,
        'F850LP': 918, 'G': 673,
    }
    def _wl(b):
        if b in _WL_VPD:
            return _WL_VPD[b]
        import re; m = re.search(r'(\d{3,4})', b)
        return float(m.group(1)) if m else 1000.

    filt_mags: list[tuple[str, np.ndarray]] = sorted(
        [(c.replace('mag_wmean_', ''), combined_df[c].values.astype(float))
         for c in combined_df.columns if c.startswith('mag_wmean_')],
        key=lambda t: _wl(t[0]))

    n_filt = len(filt_mags)
    ncols  = max(4, n_filt)
    nrows  = 2 if n_filt > 0 else 1

    fig, axes_all = plt.subplots(nrows, ncols,
                                 figsize=(4 * ncols, 4 * nrows),
                                 squeeze=False)
    for _ax in axes_all.ravel():
        _ax.set_facecolor('#e8e8e8')

    # ── Row 0: VPD zoom panels ────────────────────────────────────────────────
    zoom_fracs  = [1.0, 0.95, 0.64, 0.50]
    zoom_labels = ['Full', '95%', '64%', '50%']

    # Build zoom limits iteratively: each level is centered on the previous
    # level's median and sized from the previous level's visible sources.
    zoom_xlims: list = [None]   # Full panel has no limits
    zoom_ylims: list = [None]
    _zx, _zy = x_zoom, y_zoom   # sources visible at current level
    for frac in zoom_fracs[1:]:
        lo = (1 - frac) / 2 * 100
        hi = 100 - lo
        if len(_zx) >= 5:
            cx = float(np.median(_zx))
            cy = float(np.median(_zy))
            hw = max(float(np.percentile(_zx, hi) - np.percentile(_zx, lo)),
                     float(np.percentile(_zy, hi) - np.percentile(_zy, lo))) * 0.55
            xlim: tuple = (cx - hw, cx + hw)
            ylim: tuple = (cy - hw, cy + hw)
        else:
            xlim = ylim = (-5.0, 5.0)
        zoom_xlims.append(xlim)
        zoom_ylims.append(ylim)
        _vis = ((_zx >= xlim[0]) & (_zx <= xlim[1]) &
                (_zy >= ylim[0]) & (_zy <= ylim[1]))
        _zx = _zx[_vis]
        _zy = _zy[_vis]

    # σ_PM colour norm — use values for the ok_pm subset, log-scaled
    from matplotlib.colors import LogNorm
    sigma_geom_pm = sigma_geom[ok_pm]   # aligned with x, y
    _use_sig_color = np.isfinite(sigma_geom_pm).sum() > 3
    if _use_sig_color:
        _sv, _sv2 = np.nanpercentile(sigma_geom_pm, [2, 98])
        _sv = max(_sv, 1e-3)
        _sig_norm = LogNorm(vmin=_sv, vmax=_sv2)
        _sig_cmap = 'plasma_r'

    _last_sc_vpd = None
    for col, (frac, label) in enumerate(zip(zoom_fracs, zoom_labels)):
        ax = axes_all[0, col]
        if _use_sig_color:
            sc = ax.scatter(x, y, c=sigma_geom_pm, s=2, alpha=0.5,
                            norm=_sig_norm, cmap=_sig_cmap, rasterized=True)
            _last_sc_vpd = sc
        else:
            ax.scatter(x, y, s=1.5, alpha=0.4, color='steelblue', rasterized=True)
        _xl, _yl = zoom_xlims[col], zoom_ylims[col]
        if _xl is not None:
            ax.set_xlim(*_xl)
            ax.set_ylim(*_yl)
        ax.set_xlabel(r'$\mu_\alpha^*$ (mas/yr)')
        ax.set_ylabel(r'$\mu_\delta$ (mas/yr)')
        ax.set_title(label)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, which='major', lw=0.4, alpha=0.5)
        ax.grid(True, which='minor', lw=0.2, alpha=0.3, linestyle=':')
        ax.minorticks_on()
    for col in range(4, ncols):
        axes_all[0, col].set_visible(False)

    # Attach colorbar to the rightmost visible VPD panel
    if _use_sig_color and _last_sc_vpd is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        _last_vpd_ax = axes_all[0, min(3, ncols - 1)]
        _div = make_axes_locatable(_last_vpd_ax)
        _cax = _div.append_axes('right', size='5%', pad=0.06)
        fig.colorbar(_last_sc_vpd, cax=_cax, label=r'$\sigma_\mu^{\rm geom}$ (mas/yr)')

    # ── Row 1: σ_PM_geom vs magnitude per filter ─────────────────────────────
    if n_filt > 0:
        for col, (flt, mag_vals) in enumerate(filt_mags):
            ax = axes_all[1, col]
            ok = np.isfinite(mag_vals) & np.isfinite(sigma_geom)
            if ok.sum() > 0:
                if _use_sig_color:
                    ax.scatter(mag_vals[ok], sigma_geom[ok],
                               c=sigma_geom[ok], s=2, alpha=0.5,
                               norm=_sig_norm, cmap=_sig_cmap, rasterized=True)
                else:
                    ax.scatter(mag_vals[ok], sigma_geom[ok],
                               s=1.5, alpha=0.35, color='steelblue', rasterized=True)
                # Running median
                m_ok = mag_vals[ok]
                s_ok = sigma_geom[ok]
                bins  = np.percentile(m_ok, np.linspace(5, 95, 20))
                bins  = np.unique(bins)
                if len(bins) >= 4:
                    bx = 0.5 * (bins[:-1] + bins[1:])
                    by = np.array([np.median(s_ok[(m_ok >= lo) & (m_ok < hi)])
                                   for lo, hi in zip(bins[:-1], bins[1:])])
                    bv = np.isfinite(by)
                    if bv.sum() >= 2:
                        ax.plot(bx[bv], by[bv], color='tomato', lw=1.5, zorder=3)
            ax.set_xlabel(f'{flt} (mag)')
            ax.set_ylabel(r'$\sigma_{\mu}^{\rm geom}$ (mas/yr)')
            ax.set_title(f'{flt}')
            ax.set_yscale('log')
            ax.grid(True, which='major', lw=0.4, alpha=0.5)
            ax.grid(True, which='minor', lw=0.2, alpha=0.3, linestyle=':')
            ax.minorticks_on()
        for col in range(n_filt, ncols):
            axes_all[1, col].set_visible(False)

    if title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_cmds(combined_df: pd.DataFrame, gaia_df: Optional[pd.DataFrame],
               output_path: Path, title: str = '',
               pm_size: Optional[np.ndarray] = None) -> None:
    """All pairwise CMDs from available HST filters plus Gaia G.

    Colours are always blue_filter − red_filter (shorter λ minus longer λ).
    Y-axis is the redder (longer wavelength) band for each pair.
    Points are coloured by log(|PM|) when pm_size is provided.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # Approximate effective wavelengths (nm) for sorting blue→red
    _WL: dict[str, float] = {
        'F225W': 237, 'F275W': 271, 'F300X': 282, 'F336W': 336,
        'F390W': 392, 'F435W': 432, 'F438W': 438, 'F475W': 476,
        'F555W': 531, 'F606W': 592, 'F600LP': 747, 'F625W': 626,
        'F775W': 764, 'F814W': 803, 'F850LP': 918,
        'G': 673,   # Gaia G effective wavelength
    }

    def _wl(band: str) -> float:
        if band in _WL:
            return _WL[band]
        # Try to parse e.g. F606W → 606
        import re
        m = re.search(r'(\d{3,4})', band)
        return float(m.group(1)) if m else 1000.0

    # ── Collect HST magnitude columns ────────────────────────────────────────
    mag_cols   = sorted(c for c in combined_df.columns if c.startswith('mag_wmean_'))
    band_labels = [c.replace('mag_wmean_', '') for c in mag_cols]
    mags: dict[str, np.ndarray] = {
        lbl: combined_df[col].values.astype(float)
        for lbl, col in zip(band_labels, mag_cols)
    }

    # ── Add Gaia G and BP-RP for sources with a Gaia match ───────────────────
    # G and BP-RP are NaN for sources without a Gaia match so CMD panels
    # involving them show only the matched subset automatically.
    g_added   = False
    bp_rp_vals: Optional[np.ndarray] = None
    if gaia_df is not None and 'gaia_source_id' in combined_df.columns:
        gdf_lower = gaia_df.rename(columns=str.lower)
        sids = combined_df['gaia_source_id'].values
        if 'gmag' in gdf_lower.columns and 'source_id' in gdf_lower.columns:
            g_lookup = gdf_lower.set_index('source_id')['gmag']
            g_vals = np.where(sids != 0,
                              pd.Series(sids).map(g_lookup).values.astype(float),
                              np.nan)
            if np.isfinite(g_vals).sum() > 10:
                mags['G'] = g_vals
                band_labels.append('G')
                g_added = True
        if g_added and 'bp_rp' in gdf_lower.columns:
            bprp_lookup = gdf_lower.set_index('source_id')['bp_rp']
            _bprp = np.where(sids != 0,
                             pd.Series(sids).map(bprp_lookup).values.astype(float),
                             np.nan)
            if np.isfinite(_bprp).sum() > 10:
                bp_rp_vals = _bprp
    if not g_added and 'gaia_gmag' in combined_df.columns:
        g_vals = combined_df['gaia_gmag'].values.astype(float)
        if np.isfinite(g_vals).sum() > 10:
            mags['G'] = g_vals
            band_labels.append('G')
            g_added = True

    # ── Sort bands by wavelength (blue → red) ────────────────────────────────
    bands = sorted(band_labels, key=_wl)
    n = len(bands)
    if n < 2:
        return

    # PM colouring setup
    use_pm_color = (pm_size is not None
                    and len(pm_size) == len(combined_df)
                    and np.isfinite(pm_size).sum() > 0)
    if use_pm_color:
        _pm = pm_size.copy()
        _pm[~np.isfinite(_pm) | (_pm <= 0)] = np.nan
        _vmin = np.nanpercentile(_pm, 2)
        _vmax = np.nanpercentile(_pm, 98)
        _vmin = max(_vmin, 0.01)
        _norm = LogNorm(vmin=_vmin, vmax=_vmax)
        _cmap = 'plasma'

    pm_finite = np.isfinite(_pm) if use_pm_color else np.ones(len(combined_df), bool)

    # ── Per-band y-axis limits ────────────────────────────────────────────────
    band_ylim: dict[str, tuple[float, float]] = {}
    for b in bands:
        m = mags[b]
        ok_b = np.isfinite(m) & pm_finite
        if ok_b.sum() < 2:
            continue
        cy_hi = np.percentile(m[ok_b], 99.5)
        cy_lo = m[ok_b].min()
        pad   = 0.10 * abs(cy_hi - cy_lo) if cy_hi > cy_lo else 0.5
        band_ylim[b] = (cy_hi + pad, cy_lo - pad)   # (faint, bright) — inverted for matplotlib

    # ── Common y-axis limits across all panels ────────────────────────────────
    if band_ylim:
        _global_faint  = max(v[0] for v in band_ylim.values())  # faintest (largest mag)
        _global_bright = min(v[1] for v in band_ylim.values())  # brightest (smallest mag)
        global_ylim: tuple[float, float] | None = (_global_faint, _global_bright)
    else:
        global_ylim = None

    # ── Build panel_data before sizing the figure ─────────────────────────────
    # Gaia G vs BP-RP is prepended as the leftmost panel when available.
    # colour = bluer − redder (wavelength order); b_y = G when G is in pair.
    panel_data = []

    if bp_rp_vals is not None and 'G' in mags:
        ok_bprp = np.isfinite(bp_rp_vals) & np.isfinite(mags['G'])
        if use_pm_color:
            ok_bprp &= pm_finite
        if ok_bprp.sum() >= 2:
            panel_data.append(('BP', 'RP', 'G', bp_rp_vals, mags['G'], ok_bprp))

    # Pairwise HST (+ G) panels — sorted G-involved first, pre-filtered to data-bearing pairs
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def _g_sort_key(pair):
        bi, bj = bands[pair[0]], bands[pair[1]]
        return 0 if (bi == 'G' or bj == 'G') else 1
    pairs = sorted(pairs, key=_g_sort_key)

    for _i, _j in pairs:
        b_blu = bands[_i]; b_red = bands[_j]
        b_y   = 'G' if (b_blu == 'G' or b_red == 'G') else b_red
        colour = mags[b_blu] - mags[b_red]
        mag_y  = mags[b_y]
        ok = np.isfinite(colour) & np.isfinite(mag_y)
        if use_pm_color:
            ok &= pm_finite
        if ok.sum() >= 2:
            panel_data.append((b_blu, b_red, b_y, colour, mag_y, ok))

    n_pairs = len(panel_data)
    if n_pairs == 0:
        return
    ncols = min(4, n_pairs)

    nrows_sc    = (n_pairs + ncols - 1) // ncols   # rows of scatter panels
    nrows_total = 2 * nrows_sc                      # + matching hist2d rows below

    fig, axes = plt.subplots(nrows_total, ncols,
                             figsize=(4 * ncols, 4 * nrows_total),
                             squeeze=False)
    axes_sc = axes[:nrows_sc].ravel()    # scatter panels
    axes_h2 = axes[nrows_sc:].ravel()   # hist2d panels
    for _ax in axes.ravel():
        _ax.set_facecolor('#e8e8e8')

    # ── Row 0…nrows_sc-1: scatter panels ──────────────────────────────────────
    last_sc    = None
    panel_xlim = {}   # save x limits for hist2d row
    for k, (b_blu, b_red, b_y, colour, mag_y, ok) in enumerate(panel_data):
        ax = axes_sc[k]
        if ok.sum() < 2:
            ax.set_visible(False)
            axes_h2[k].set_visible(False)
            continue
        if use_pm_color:
            sc = ax.scatter(colour[ok], mag_y[ok], c=_pm[ok], s=2, alpha=0.6,
                            norm=_norm, cmap=_cmap, rasterized=True)
            last_sc = sc
        else:
            ax.scatter(colour[ok], mag_y[ok], s=1.5, alpha=0.35,
                       color='steelblue', rasterized=True)
        if global_ylim is not None:
            ax.set_ylim(*global_ylim)
        cx_lo, cx_hi = np.percentile(colour[ok], [0.5, 99.5])
        pad_c = 0.25 * (cx_hi - cx_lo) if cx_hi > cx_lo else 0.5
        ax.set_xlim(cx_lo - pad_c, cx_hi + pad_c)
        panel_xlim[k] = ax.get_xlim()
        ax.set_xlabel(f'{b_blu} − {b_red}')
        ax.set_ylabel(b_y)

    for k in range(n_pairs, len(axes_sc)):
        axes_sc[k].set_visible(False)
        axes_h2[k].set_visible(False)

    # ── Row nrows_sc…nrows_total-1: hist2d panels ─────────────────────────────
    last_h2d = None
    for k, (b_blu, b_red, b_y, colour, mag_y, ok) in enumerate(panel_data):
        ax = axes_h2[k]
        if ok.sum() < 2:
            continue
        xlim = panel_xlim.get(k, (float(np.nanmin(colour)), float(np.nanmax(colour))))
        ylim = global_ylim if global_ylim is not None else (float(np.nanmax(mag_y)), float(np.nanmin(mag_y)))
        ylo, yhi = min(ylim), max(ylim)   # hist2d needs ascending range
        _, _, _, img = ax.hist2d(
            colour[ok], mag_y[ok], bins=80,
            range=[[xlim[0], xlim[1]], [ylo, yhi]],
            norm=LogNorm(vmin=1), cmap='viridis', rasterized=True)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)   # restore inverted y-axis
        ax.set_xlabel(f'{b_blu} − {b_red}')
        ax.set_ylabel(b_y)
        last_h2d = img

    # Grid lines + minor ticks on all visible panels
    for k in range(n_pairs):
        for ax in (axes_sc[k], axes_h2[k]):
            if ax.get_visible():
                ax.grid(True, which='major', lw=0.4, alpha=0.5)
                ax.grid(True, which='minor', lw=0.2, alpha=0.3, linestyle=':')
                ax.minorticks_on()

    # Scatter colorbar — attached to last visible scatter panel via standard
    # fig.colorbar (compatible with constrained_layout, avoids row gap issues)
    if use_pm_color and last_sc is not None:
        last_sc_k = max(k for k in range(n_pairs) if axes_sc[k].get_visible())
        fig.colorbar(last_sc, ax=axes_sc[last_sc_k],
                     label=r'$|\mu|$ (mas/yr)', fraction=0.046, pad=0.04)

    # Hist2d colorbar
    if last_h2d is not None:
        last_h2_k = max(k for k in range(n_pairs) if axes_h2[k].get_visible())
        fig.colorbar(last_h2d, ax=axes_h2[last_h2_k],
                     label='N', fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_gaia_comparison(combined_df: pd.DataFrame, gaia_df: pd.DataFrame,
                           output_path: Path, title: str = '') -> None:
    """Compare new xmatch PMs to Gaia PMs for sources with Gaia matches.

    Panels:
      Row 0: pmra one-to-one | pmdec one-to-one | σ_PM_geom vs G (both)
      Row 1: VPD Gaia        | VPD xmatch       | PM improvement histogram
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # ── Join Gaia covariance columns onto combined_df ─────────────────────────
    gdf = gaia_df.rename(columns=str.lower)
    needed = {'source_id', 'gmag', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error',
              'pmra_pmdec_corr'}
    if not needed.issubset(set(gdf.columns)):
        print("  Warning: gaia_df missing required columns for comparison plot")
        return

    sids = combined_df['gaia_source_id'].values
    has_match = (sids != 0) & np.isfinite(sids.astype(float))
    if has_match.sum() < 5:
        return

    gdf_idx = gdf.set_index('source_id')

    def _lookup(col):
        return np.where(has_match,
                        pd.Series(sids).map(gdf_idx[col]).values.astype(float),
                        np.nan)

    gmag_g    = _lookup('gmag')
    pmra_g    = _lookup('pmra')
    pmdec_g   = _lookup('pmdec')
    sig_pmra_g_raw  = _lookup('pmra_error')
    sig_pmdec_g_raw = _lookup('pmdec_error')
    corr_pm_g       = _lookup('pmra_pmdec_corr')

    # Inflate Gaia PM uncertainties exactly as BP3M does, so that the pull
    # distribution (pm_xmatch − pm_gaia) / σ_Gaia_inflated is comparable to
    # N(0,1) rather than appearing too wide.
    # Inflation: C_inflated = mult * C_raw + diag(pm_sys_err²)
    # → σ_inflated = sqrt(mult * σ_raw² + pm_sys_err²)
    # mult depends on astrometric solution type:
    #   6-param (has pseudocolour): mult_6p = 1.22
    #   5-param (has pmra_error)  : mult_5p = 1.05
    #   2-param (no pmra_error)   : mult_2p = 1.00  (no PM, irrelevant here)
    has_pseudocolour = np.isfinite(_lookup('pseudocolour')) \
                       if 'pseudocolour' in gdf.columns else np.zeros(len(combined_df), bool)
    has_pm_g = np.isfinite(sig_pmra_g_raw) & (sig_pmra_g_raw > 0)
    mult = np.where(has_pseudocolour, GAIA_SYS_DICT['mult_6p'],
           np.where(has_pm_g,         GAIA_SYS_DICT['mult_5p'],
                                      GAIA_SYS_DICT['mult_2p']))
    _pm_sys2 = GAIA_SYS_DICT['pm_sys_err'] ** 2
    sig_pmra_g  = np.where(has_pm_g,
                            np.sqrt(mult * sig_pmra_g_raw**2  + _pm_sys2),
                            sig_pmra_g_raw)
    sig_pmdec_g = np.where(has_pm_g,
                            np.sqrt(mult * sig_pmdec_g_raw**2 + _pm_sys2),
                            sig_pmdec_g_raw)

    # Gaia geometric mean PM uncertainty (inflated, matching BP3M)
    sig_pm_g_geom = np.where(
        np.isfinite(sig_pmra_g) & np.isfinite(sig_pmdec_g) & (sig_pmra_g > 0) & (sig_pmdec_g > 0),
        np.sqrt(sig_pmra_g * sig_pmdec_g), np.nan)

    pmra_x    = combined_df['pmra_xmatch'].values.astype(float)
    pmdec_x   = combined_df['pmdec_xmatch'].values.astype(float)
    sig_pmra_x  = combined_df['sigma_pmra_xmatch'].values.astype(float) \
                  if 'sigma_pmra_xmatch' in combined_df.columns else np.full(len(combined_df), np.nan)
    sig_pmdec_x = combined_df['sigma_pmdec_xmatch'].values.astype(float) \
                  if 'sigma_pmdec_xmatch' in combined_df.columns else np.full(len(combined_df), np.nan)
    sig_pm_x_geom = np.where(
        np.isfinite(sig_pmra_x) & np.isfinite(sig_pmdec_x) & (sig_pmra_x > 0) & (sig_pmdec_x > 0),
        np.sqrt(sig_pmra_x * sig_pmdec_x), np.nan)

    # ── Mask to sources with both Gaia PM and xmatch PM, |PM|<=100 on both ───
    ok = (has_match
          & np.isfinite(pmra_g) & np.isfinite(pmdec_g)
          & np.isfinite(pmra_x) & np.isfinite(pmdec_x)
          & (np.abs(pmra_g)  <= 50.) & (np.abs(pmdec_g)  <= 50.)
          & (np.abs(pmra_x)  <= 50.) & (np.abs(pmdec_x)  <= 50.))

    if ok.sum() < 5:
        return

    G = gmag_g[ok];         pmra_g_ok  = pmra_g[ok];   pmdec_g_ok  = pmdec_g[ok]
    pmra_x_ok  = pmra_x[ok]; pmdec_x_ok = pmdec_x[ok]
    sig_g_ok   = sig_pm_g_geom[ok];    sig_x_ok   = sig_pm_x_geom[ok]
    sig_pmra_g_ok  = sig_pmra_g[ok];   sig_pmdec_g_ok  = sig_pmdec_g[ok]
    sig_pmra_x_ok  = sig_pmra_x[ok];   sig_pmdec_x_ok  = sig_pmdec_x[ok]
    corr_pm_g_ok   = corr_pm_g[ok]

    # Improvement factor  σ_Gaia / σ_xmatch  (>1 = improvement)
    improv = np.where(
        np.isfinite(sig_g_ok) & np.isfinite(sig_x_ok) & (sig_x_ok > 0),
        sig_g_ok / sig_x_ok, np.nan)

    # Normalised residuals using Gaia 2×2 PM covariance + xmatch 2×2 covariance
    # Since xmatch uses Gaia as prior they are correlated; we use Gaia σ as the
    # reference scale (how many Gaia sigmas did the PM shift?).
    pull_pmra  = np.where(sig_pmra_g_ok  > 0, (pmra_x_ok  - pmra_g_ok)  / sig_pmra_g_ok,  np.nan)
    pull_pmdec = np.where(sig_pmdec_g_ok > 0, (pmdec_x_ok - pmdec_g_ok) / sig_pmdec_g_ok, np.nan)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), squeeze=False)
    for _ax in axes.ravel():
        _ax.set_facecolor('#e8e8e8')
    _g_norm = Normalize(vmin=np.nanpercentile(G, 2), vmax=np.nanpercentile(G, 98))
    _g_cmap = 'viridis_r'

    def _add_grid(ax):
        ax.grid(True, which='major', lw=0.4, alpha=0.5)
        ax.grid(True, which='minor', lw=0.2, alpha=0.3, linestyle=':')
        ax.minorticks_on()

    # ── Panel (0,0): pmra one-to-one ─────────────────────────────────────────
    ax = axes[0, 0]
    sc = ax.scatter(pmra_g_ok, pmra_x_ok, c=G, s=4, alpha=0.5,
                    norm=_g_norm, cmap=_g_cmap, rasterized=True)
    lim = max(np.abs(pmra_g_ok).max(), np.abs(pmra_x_ok).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1, zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(r'$\mu_\alpha^*$ Gaia (mas/yr)')
    ax.set_ylabel(r'$\mu_\alpha^*$ xmatch (mas/yr)')
    ax.set_aspect('equal', adjustable='box')
    _add_grid(ax)

    # ── Panel (0,1): pmdec one-to-one ────────────────────────────────────────
    ax = axes[0, 1]
    ax.scatter(pmdec_g_ok, pmdec_x_ok, c=G, s=4, alpha=0.5,
               norm=_g_norm, cmap=_g_cmap, rasterized=True)
    lim = max(np.abs(pmdec_g_ok).max(), np.abs(pmdec_x_ok).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1, zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(r'$\mu_\delta$ Gaia (mas/yr)')
    ax.set_ylabel(r'$\mu_\delta$ xmatch (mas/yr)')
    ax.set_aspect('equal', adjustable='box')
    _add_grid(ax)
    cb = fig.colorbar(sc, ax=axes[0, 1], pad=0.02, label='G (mag)')

    # ── Panel (0,2): σ_PM_geom vs G ──────────────────────────────────────────
    ax = axes[0, 2]
    ok_g = np.isfinite(G) & np.isfinite(sig_g_ok) & (sig_g_ok > 0)
    ok_x = np.isfinite(G) & np.isfinite(sig_x_ok) & (sig_x_ok > 0)
    if ok_g.sum() > 0:
        ax.scatter(G[ok_g], sig_g_ok[ok_g], s=2, alpha=0.3, color='royalblue',
                   rasterized=True, label='Gaia')
    if ok_x.sum() > 0:
        ax.scatter(G[ok_x], sig_x_ok[ok_x], s=2, alpha=0.3, color='tomato',
                   rasterized=True, label='xmatch')
    # Running medians
    for _ok, _s, _col in [(ok_g, sig_g_ok, 'royalblue'), (ok_x, sig_x_ok, 'tomato')]:
        if _ok.sum() < 5:
            continue
        _m = G[_ok]; _v = _s[_ok]
        _bins = np.unique(np.percentile(_m, np.linspace(5, 95, 20)))
        if len(_bins) >= 4:
            _bx = 0.5 * (_bins[:-1] + _bins[1:])
            _by = np.array([np.median(_v[(_m >= lo) & (_m < hi)])
                            for lo, hi in zip(_bins[:-1], _bins[1:])])
            _bv = np.isfinite(_by)
            if _bv.sum() >= 2:
                ax.plot(_bx[_bv], _by[_bv], color=_col, lw=2, zorder=4)
    ax.set_xlabel('G (mag)')
    ax.set_ylabel(r'$\sigma_\mu^{\rm geom}$ (mas/yr)')
    ax.set_yscale('log')
    ax.legend(markerscale=3, fontsize=9)
    _add_grid(ax)

    # ── Panel (1,0): VPD Gaia ────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.scatter(pmra_g_ok, pmdec_g_ok, s=2, alpha=0.4, color='royalblue', rasterized=True)
    ax.set_xlabel(r'$\mu_\alpha^*$ Gaia (mas/yr)')
    ax.set_ylabel(r'$\mu_\delta$ Gaia (mas/yr)')
    ax.set_title('Gaia VPD')
    ax.set_aspect('equal', adjustable='box')
    _add_grid(ax)

    # ── Panel (1,1): VPD xmatch ──────────────────────────────────────────────
    ax = axes[1, 1]
    ax.scatter(pmra_x_ok, pmdec_x_ok, s=2, alpha=0.4, color='tomato', rasterized=True)
    ax.set_xlabel(r'$\mu_\alpha^*$ xmatch (mas/yr)')
    ax.set_ylabel(r'$\mu_\delta$ xmatch (mas/yr)')
    ax.set_title('xmatch VPD')
    ax.set_aspect('equal', adjustable='box')
    _add_grid(ax)
    # Match axis limits to Gaia VPD
    _lx = list(axes[1, 0].get_xlim()); _ly = list(axes[1, 0].get_ylim())
    _hw = max(abs(_lx[1] - _lx[0]), abs(_ly[1] - _ly[0])) * 0.5
    _cx = 0.5 * (_lx[0] + _lx[1]); _cy = 0.5 * (_ly[0] + _ly[1])
    for _a in axes[1, :2]:
        _a.set_xlim(_cx - _hw, _cx + _hw)
        _a.set_ylim(_cy - _hw, _cy + _hw)

    # ── Panel (1,2): pull distribution ───────────────────────────────────────
    ax = axes[1, 2]
    ok_pull = np.isfinite(pull_pmra) & np.isfinite(pull_pmdec)
    if ok_pull.sum() > 3:
        _bins = np.linspace(-5, 5, 41)
        ax.hist(pull_pmra[ok_pull],  bins=_bins, histtype='step', color='steelblue',
                lw=1.5, label=r'$\mu_\alpha^*$', density=True)
        ax.hist(pull_pmdec[ok_pull], bins=_bins, histtype='step', color='tomato',
                lw=1.5, label=r'$\mu_\delta$', density=True)
        _xr = np.linspace(-5, 5, 200)
        ax.plot(_xr, np.exp(-0.5 * _xr**2) / np.sqrt(2 * np.pi),
                'k--', lw=1.2, label='N(0,1)')
    ax.set_xlabel(r'$({\mu}_{\rm xmatch} - {\mu}_{\rm Gaia})\,/\,\sigma_{\rm Gaia}$')
    ax.set_ylabel('Density')
    ax.set_title('PM shift / Gaia σ')
    ax.legend(fontsize=9)
    _add_grid(ax)

    if title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Plot regeneration helper ──────────────────────────────────────────────────

def _regen_plots(field_dir: Path,
                 output_dir: Optional[Path] = None,
                 gaia_csv: Optional[Path] = None) -> None:
    """Reload master_combined.csv and regenerate all diagnostic figures."""
    field_dir = Path(field_dir)
    if output_dir is None:
        output_dir = field_dir / 'hst_xmatch'

    combined_path = output_dir / 'master_combined.csv'
    if not combined_path.exists():
        raise FileNotFoundError(f"No master_combined.csv found at {combined_path}. "
                                "Run without --plots_only first.")

    combined_df = pd.read_csv(combined_path, low_memory=False)
    print(f"Loaded {len(combined_df)} sources from {combined_path}")

    # Auto-detect Gaia CSV
    gaia_df: Optional[pd.DataFrame] = None
    if gaia_csv is None:
        gaia_candidates = [f for f in field_dir.glob('Gaia/*_gaia.csv')
                           if not f.name.startswith('._')]
        if gaia_candidates:
            gaia_csv = sorted(gaia_candidates)[-1]
    if gaia_csv is not None and Path(gaia_csv).exists():
        gaia_df = pd.read_csv(gaia_csv, low_memory=False)
        print(f"Loaded {len(gaia_df)} Gaia sources from {gaia_csv}")

    field_name = field_dir.name

    _ok_pm = (np.isfinite(combined_df['pmra_xmatch'].values)
              & np.isfinite(combined_df['pmdec_xmatch'].values)) \
             if 'pmra_xmatch' in combined_df.columns else \
             np.zeros(len(combined_df), bool)
    if _ok_pm.sum() > 0:
        _pmra_v  = combined_df['pmra_xmatch'].values
        _pmdec_v = combined_df['pmdec_xmatch'].values
        _ok_pm  &= (np.abs(_pmra_v) <= 50.) & (np.abs(_pmdec_v) <= 50.)
    good_pm_df = combined_df[_ok_pm].reset_index(drop=True)
    _pm_size = np.sqrt(good_pm_df['pmra_xmatch'].values**2
                       + good_pm_df['pmdec_xmatch'].values**2) \
               if len(good_pm_df) > 0 else np.empty(0)

    for name, fn, kwargs in [
        ('sky_distribution.png', _plot_sky,
         dict(pm_size=_pm_size, title=field_name)),
        ('vpd.png', _plot_vpd,
         dict(title=field_name)),
        ('cmds.png', _plot_cmds,
         dict(gaia_df=gaia_df, title=field_name, pm_size=_pm_size)),
    ]:
        try:
            if name == 'cmds.png':
                fn(good_pm_df, gaia_df, output_dir / name, title=field_name, pm_size=_pm_size)
            else:
                fn(good_pm_df, output_dir / name, **kwargs)
            print(f"  {name} written")
        except Exception as _e:
            print(f"  Warning: {name} failed: {_e}")
            import traceback; traceback.print_exc()

    if gaia_df is not None and 'gaia_source_id' in combined_df.columns:
        try:
            _plot_gaia_comparison(good_pm_df, gaia_df,
                                  output_dir / 'gaia_comparison.png',
                                  title=field_name)
            print("  gaia_comparison.png written")
        except Exception as _e:
            print(f"  Warning: gaia_comparison.png failed: {_e}")
            import traceback; traceback.print_exc()

    print(f"\nPlots regenerated in: {output_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Cross-match all HST sources between images using the BP3M alignment solution.',
    )
    parser.add_argument('--field_dir',   required=True,
                        help='Root directory of the processed field (must contain BP3M_results/)')
    parser.add_argument('--output_dir',  default=None,
                        help='Output directory (default: field_dir/hst_xmatch)')
    parser.add_argument('--gaia_csv',    default=None,
                        help='Gaia catalog CSV for Phase 3 recovery (auto-detected if not given)')
    parser.add_argument('--max_pm_masyr', type=float, default=100.,
                        help='Maximum proper motion allowed for within-filter matching (mas/yr)')
    parser.add_argument('--match_n_sigma', type=float, default=5.,
                        help='Match radius in units of combined positional sigma')
    parser.add_argument('--mag_n_sigma', type=float, default=3.0,
                        help='Magnitude match threshold in units of combined photometric sigma')
    parser.add_argument('--mag_floor', type=float, default=0.10,
                        help='Minimum magnitude tolerance regardless of photometric error (mag)')
    parser.add_argument('--min_detections', type=int, default=2,
                        help='Minimum detections for a source to appear in the master catalog')
    parser.add_argument('--cross_filter_radius_mas', type=float, default=200.,
                        help='Match radius for cross-filter association (mas)')
    parser.add_argument('--gaia_recovery_radius_mas', type=float, default=100.,
                        help='Match radius for Gaia recovery phase (mas)')
    parser.add_argument('--no_save_detections', action='store_true',
                        help='Skip saving per-filter detection catalogs (saves disk space)')
    parser.add_argument('--plots_only', action='store_true',
                        help='Skip all computation; reload master_combined.csv and regenerate plots only')
    args = parser.parse_args()

    if args.plots_only:
        _regen_plots(
            field_dir = Path(args.field_dir),
            output_dir = Path(args.output_dir) if args.output_dir else None,
            gaia_csv   = Path(args.gaia_csv)   if args.gaia_csv   else None,
        )
        return

    run_hst_crossmatch(
        field_dir                = Path(args.field_dir),
        output_dir               = Path(args.output_dir) if args.output_dir else None,
        gaia_csv                 = Path(args.gaia_csv)   if args.gaia_csv   else None,
        max_pm_masyr             = args.max_pm_masyr,
        match_n_sigma            = args.match_n_sigma,
        mag_n_sigma              = args.mag_n_sigma,
        mag_floor                = args.mag_floor,
        min_detections           = args.min_detections,
        cross_filter_radius_mas  = args.cross_filter_radius_mas,
        gaia_recovery_radius_mas = args.gaia_recovery_radius_mas,
        save_detections          = not args.no_save_detections,
    )


if __name__ == '__main__':
    main()
