"""
Step 3: PSF-fit each downloaded FLC image using pypass.

Calls pypass.io.run_photometry_fits for every image under
    {output_dir}/{field}/{telescope}/mastDownload/{telescope}/{obs_id}/
and writes
    {obs_id}_flc_catalog.fits
in the same directory.

Default pypass parameters are tuned for HST ACS/WFC and WFC3/UVIS FLC images.
All defaults can be overridden via keyword arguments to run_psf_fitting().

Extension note
--------------
JWST support will require:
  - A JWST STDPSF library in lib_dir
  - Possibly different default fmin / sat_threshold values
  - pypass updates to handle JWST FITS headers (different SCI/DQ extensions)
Passing telescope='JWST' currently raises NotImplementedError as a placeholder.
"""

from __future__ import annotations

import json
import sys
import os
import warnings
from pathlib import Path

import numpy as np

# float(masked_element) → nan is intentional here (concentration metrics for
# saturated/bad stars are legitimately masked and should become NaN).
warnings.filterwarnings(
    'ignore',
    message=r'.*converting a masked element to nan.*',
    category=UserWarning,
)


def _ensure_py1pass():
    pass  # pypass is installed as a package; no sys.path manipulation needed

# ── Default PSF-fitting parameters (user-confirmed for HST FLC images) ──────
_HST_DEFAULTS = dict(
    fmin_thresh=100.0,
    mag_st_max=28.0,
    hmin=4,
    n_passes=2,
    n_discovery_passes=1,
    sat_threshold=60000.0,
    max_iter_fit=100,
    half_width=3,
    sky_inner=4,
    sky_outer=8,
    tol=1e-3,
    sigma_clip=True,
    sigma_clip_sigma=4.0,
    conc_limit=0.9,
    n_jobs=-1,
    backend='auto',
)


class _FITSRecord:
    """Minimal duck-typed StarRecord built from a FITS catalog table row.

    Provides the subset of attributes needed by classify_stars(),
    inflate_chi2(), estimate_systematic_floor(), _apply_gdc_wcs(),
    plot_catalog_stats(), and plot_concentration_diagnostics() — without
    requiring a full PSF re-fit.

    No __slots__ so that _apply_gdc_wcs() can attach _x_gdc, _y_gdc,
    _cov_gdc, _ra, _dec, _cov_radec, _mc directly on the record.

    ``cov`` is the raw (pre-chi2-inflation, pre-floor) 4×4 covariance so
    that inflate_chi2() and estimate_systematic_floor() see the true
    photon-noise uncertainty.
    """

    def __init__(self, row, raw_cov_xx, raw_cov_yy, raw_cov_ff):
        self.mag   = float(row['mag'])
        self.qfit  = float(row['qfit'])
        self.chi2  = float(row['chi2'])
        self.flux  = float(row['flux'])
        self.sky   = float(row['sky'])
        self.sky_err = float(row['sky_err'])
        self.x     = float(row['x'])
        self.y     = float(row['y'])
        self.pass_number          = int(row['pass_number'])
        self.n_iter               = int(row['n_iter'])
        self.converged            = bool(row['converged'])
        self.delta_max            = float(row['delta_max'])
        self.chi2_scale           = float(row['chi2_scale'])
        self.eps_psf              = float(row['eps_psf'])
        self.peak                 = float(row['peak'])
        self.psf_frac             = float(row['psf_frac'])
        self.psf_peak             = float(row['psf_peak'])
        self.central_res          = float(row['central_res'])
        self.n_sat                = int(row['n_sat'])
        self.n_neighbors          = int(row['n_neighbors'])
        self.dist_nearest         = float(row['dist_nearest'])
        self.dist_nearest_brighter = float(row['dist_nearest_brighter'])
        self.concentration        = float(row['concentration'])
        self.concentration_2x2    = float(row['concentration_2x2'])
        self.concentration_3x3    = float(row['concentration_3x3'])
        self.n_conc_1x1           = int(row['n_conc_1x1']) if 'n_conc_1x1' in row.colnames else 1
        self.n_conc_2x2           = int(row['n_conc_2x2']) if 'n_conc_2x2' in row.colnames else 4
        self.n_conc_3x3           = int(row['n_conc_3x3']) if 'n_conc_3x3' in row.colnames else 9
        self.is_star_candidate    = bool(row['is_star_candidate'])
        self._chip_ext            = int(row['chip_ext'])
        self._x_offset            = 0.0
        self._y_offset            = 0.0

        # Reconstruct raw 4×4 covariance from per-element catalog columns.
        # Row indices: 0=flux, 1=x, 2=y, 3=sky.
        c = np.zeros((4, 4))
        c[0, 0] = raw_cov_ff
        c[1, 1] = raw_cov_xx
        c[2, 2] = raw_cov_yy
        c[3, 3] = float(row['cov_ss'])
        c[0, 1] = c[1, 0] = float(row['cov_fx'])
        c[0, 2] = c[2, 0] = float(row['cov_fy'])
        c[0, 3] = c[3, 0] = float(row['cov_fs'])
        c[1, 2] = c[2, 1] = float(row['cov_xy'])
        c[1, 3] = c[3, 1] = float(row['cov_xs'])
        c[2, 3] = c[3, 2] = float(row['cov_ys'])
        self.cov = c


def _records_from_fits_table(t):
    """Build _FITSRecord list from an astropy Table loaded from a catalog FITS file.

    Strips the stored floor (SIGMA_FLOOR_X/Y, EPS_FLUX) AND the chi2 inflation
    (chi2_scale stored per-row) so that ``r.cov`` holds the truly raw pre-inflation,
    pre-floor covariance.  This lets reclassify_psf_catalogs() re-run inflate_chi2()
    from scratch with the new star population.

    Returns (records, old_floor_x, old_floor_y, old_eps_flux, old_chi2_scales).
    old_chi2_scales is a float array of the chi2_scale values from the catalog, useful
    for approximating GDC covariance updates when lib_dir is unavailable.
    """
    old_floor_x   = float(t.meta.get('SIGMA_FLOOR_X', 0.0))
    old_floor_y   = float(t.meta.get('SIGMA_FLOOR_Y', 0.0))
    old_eps_flux  = float(t.meta.get('EPS_FLUX',      0.0))

    flux_arr      = np.asarray(t['flux'],    dtype=float)
    cov_ff_stored = np.asarray(t['cov_ff'],  dtype=float)
    cov_xx_stored = np.asarray(t['cov_xx'],  dtype=float)
    cov_yy_stored = np.asarray(t['cov_yy'],  dtype=float)

    # Strip floor from diagonal elements.  The stored value is
    #   chi2_scale² × raw_cov + floor²
    # so after removing the floor we still have chi2_scale² × raw_cov.
    floor_ff = (old_eps_flux * np.where(flux_arr > 0, flux_arr, 1.0)) ** 2
    raw_cov_ff = cov_ff_stored - floor_ff
    raw_cov_xx = cov_xx_stored - old_floor_x ** 2
    raw_cov_yy = cov_yy_stored - old_floor_y ** 2

    records = [
        _FITSRecord(t[i], raw_cov_xx[i], raw_cov_yy[i], raw_cov_ff[i])
        for i in range(len(t))
    ]

    # Capture old chi2_scale before un-inflating (needed for GDC approximation).
    old_chi2_scales = np.array([r.chi2_scale for r in records])

    # Un-inflate: divide the full 4×4 covariance by chi2_scale² so that
    # r.cov holds the raw photon-noise covariance.  inflate_chi2() can then
    # re-apply the correct scaling based on the new star classification.
    for r in records:
        s2 = max(r.chi2_scale ** 2, 1e-10)
        r.cov = r.cov / s2
        r.chi2_scale = 1.0

    return records, old_floor_x, old_floor_y, old_eps_flux, old_chi2_scales


def _apply_new_floor(t, records, new_floor_x, new_floor_y, new_eps_flux, new_floor_params,
                     old_chi2_scales=None):
    """Update all inflation- and floor-dependent columns in table ``t`` in-place.

    Called after inflate_chi2() has already been applied to *records*, so
    ``r.cov`` already contains the new chi2_scale² × raw_cov (no floor).
    ``r.chi2_scale`` holds the new per-star scale factor.

    Updates:
      chi2_scale, cov_ff/xx/yy + all off-diagonals (chi2-scaled, no floor),
      cov_xx_gdc, cov_yy_gdc, cov_xy_gdc (chi2-scaled + floor),
      flux_err, mag_err, mag_err_gdc, sigma_{x,y,f}_model, is_star_candidate.

    GDC columns:
      If ``r._cov_gdc`` is present on each record (set by _apply_gdc_wcs),
      those are used directly (correct: new chi2_scale baked in via Jacobian).
      Otherwise the existing table values are rescaled by (new_cs/old_cs)² using
      *old_chi2_scales* — an approximation valid when the GDC Jacobian ≈ identity.

    RA/Dec error columns are not updated (WCS Jacobian not stored in catalog).
    """

    flux_arr      = np.asarray(t['flux'], dtype=float)
    new_floor_ff  = (new_eps_flux * np.where(flux_arr > 0, flux_arr, 1.0)) ** 2
    new_floor_xx2 = new_floor_x ** 2
    new_floor_yy2 = new_floor_y ** 2

    # chi2_scale column
    t['chi2_scale'] = np.array([r.chi2_scale for r in records])

    # Covariance diagonal + floor
    t['cov_ff'] = np.array([r.cov[0, 0] for r in records]) + new_floor_ff
    t['cov_xx'] = np.array([r.cov[1, 1] for r in records]) + new_floor_xx2
    t['cov_yy'] = np.array([r.cov[2, 2] for r in records]) + new_floor_yy2

    # Off-diagonal covariance (no floor, but chi2_scale² baked in by inflate_chi2)
    t['cov_fx'] = np.array([r.cov[0, 1] for r in records])
    t['cov_fy'] = np.array([r.cov[0, 2] for r in records])
    t['cov_fs'] = np.array([r.cov[0, 3] for r in records])
    t['cov_xy'] = np.array([r.cov[1, 2] for r in records])
    t['cov_xs'] = np.array([r.cov[1, 3] for r in records])
    t['cov_ys'] = np.array([r.cov[2, 3] for r in records])

    # GDC covariance columns
    gdc_from_records = hasattr(records[0], '_cov_gdc') if records else False
    if gdc_from_records:
        # Fresh propagation: r._cov_gdc = J @ chi2_scaled_cov @ J.T
        gdc_xx = np.full(len(records), np.nan)
        gdc_yy = np.full(len(records), np.nan)
        gdc_xy = np.full(len(records), np.nan)
        for k, r in enumerate(records):
            cg = getattr(r, '_cov_gdc', None)
            if cg is not None and np.isfinite(cg).all():
                gdc_xx[k] = cg[0, 0]
                gdc_yy[k] = cg[1, 1]
                gdc_xy[k] = cg[0, 1]
        t['cov_xx_gdc'] = np.where(np.isfinite(gdc_xx), gdc_xx + new_floor_xx2, np.nan)
        t['cov_yy_gdc'] = np.where(np.isfinite(gdc_yy), gdc_yy + new_floor_yy2, np.nan)
        t['cov_xy_gdc'] = gdc_xy
    else:
        # Approximation: rescale existing GDC cov by (new_cs/old_cs)²
        old_cs = old_chi2_scales if old_chi2_scales is not None else np.ones(len(records))
        new_cs = np.array([r.chi2_scale for r in records])
        ratio2 = (new_cs / np.where(old_cs > 1e-6, old_cs, 1.0)) ** 2
        old_floor_xx2 = float(t.meta.get('SIGMA_FLOOR_X', 0.0)) ** 2
        old_floor_yy2 = float(t.meta.get('SIGMA_FLOOR_Y', 0.0)) ** 2
        raw_gdc_xx = np.asarray(t['cov_xx_gdc'], dtype=float) - old_floor_xx2
        raw_gdc_yy = np.asarray(t['cov_yy_gdc'], dtype=float) - old_floor_yy2
        raw_gdc_xy = np.asarray(t['cov_xy_gdc'], dtype=float)
        t['cov_xx_gdc'] = np.where(np.isfinite(raw_gdc_xx), raw_gdc_xx * ratio2 + new_floor_xx2, np.nan)
        t['cov_yy_gdc'] = np.where(np.isfinite(raw_gdc_yy), raw_gdc_yy * ratio2 + new_floor_yy2, np.nan)
        t['cov_xy_gdc'] = np.where(np.isfinite(raw_gdc_xy), raw_gdc_xy * ratio2, np.nan)

    # Flux / mag errors
    flux_err = np.sqrt(np.maximum(np.asarray(t['cov_ff'], dtype=float), 0.0))
    _log10e  = 2.5 / np.log(10.0)
    mag_err  = np.where(flux_arr > 0, _log10e * flux_err / flux_arr, np.nan)
    t['flux_err']    = flux_err
    t['mag_err']     = mag_err
    t['mag_err_gdc'] = mag_err

    # Noise model curves (sigma vs magnitude)
    if new_floor_params is not None and new_floor_params.get('fit_A_ok'):
        mag_arr = np.asarray(t['mag'], dtype=float)
        def _eval(mag, popt):
            if popt is None:
                return np.full(len(mag), np.nan)
            fl, lA, lC = popt
            return np.sqrt(
                (10.0 ** fl) ** 2
                + (10.0 ** lA * 10.0 ** (0.2 * mag)) ** 2
                + (10.0 ** lC * 10.0 ** (0.4 * mag)) ** 2
            )
        t['sigma_x_model'] = _eval(mag_arr, new_floor_params.get('popt_x'))
        t['sigma_y_model'] = _eval(mag_arr, new_floor_params.get('popt_y'))
        t['sigma_f_model'] = _eval(mag_arr, new_floor_params.get('popt_f'))

    # Classification flag
    t['is_star_candidate'] = np.array([r.is_star_candidate for r in records], dtype=bool)

    # Metadata
    t.meta['SIGMA_FLOOR_X'] = new_floor_x
    t.meta['SIGMA_FLOOR_Y'] = new_floor_y
    t.meta['EPS_FLUX']      = new_eps_flux


def reclassify_psf_catalogs(
    output_dir: Path,
    field_name: str,
    telescope: str = 'HST',
    im_type: str = '_flc',
    conc_limit: float | None = None,
    restrict_to_obsids: list[str] | None = None,
    psf_dir: Path | None = None,
    lib_dir: Path | None = None,
) -> list[Path]:
    """Re-run the full post-fit pipeline on existing PSF catalogs without re-fitting.

    Sequence per image:
      1. Load catalog, strip floor and chi2 inflation → raw photon-noise covariance
      2. Re-run classify_stars() with new conc_limit
      3. Re-run inflate_chi2() with the new star population
      4. Re-apply GDC Jacobian (requires lib_dir; approximation used if unavailable)
      5. Re-estimate systematic position floor
      6. Patch all dependent catalog columns and regenerate diagnostic figures

    The cross-match cache is invalidated for each updated image.

    Parameters
    ----------
    output_dir        : pipeline root directory
    field_name        : field subdirectory name
    telescope         : 'HST' (JWST not yet supported)
    im_type           : '_flc' or '_flt'
    conc_limit        : new concentration lower bound (default 0.9)
    restrict_to_obsids: if given, only reclassify these obs_ids
    psf_dir           : unused (pypass is installed as a package); kept for API compatibility
    lib_dir           : path to STDPSFs/STDGDCs library (for GDC re-propagation).
                        If None the GDC covariance is rescaled by the chi2_scale ratio
                        rather than re-propagated through the Jacobian.

    Returns
    -------
    List of updated catalog FITS paths
    """
    _ensure_py1pass()
    # psf_dir parameter retained for API compatibility but no longer needed;
    # pypass is installed as a package.

    from pypass.core import classify_stars, inflate_chi2
    from pypass.diagnostics import (estimate_systematic_floor,
                                     plot_catalog_stats,
                                     plot_concentration_diagnostics)
    from astropy.table import Table

    conc_lo = conc_limit if conc_limit is not None else _HST_DEFAULTS['conc_limit']

    from .download_hst import find_flc_images
    images = find_flc_images(output_dir, field_name, telescope=telescope,
                              im_type=im_type)
    if not images:
        print(f"[reclassify] No {im_type} images found under "
              f"{output_dir}/{field_name}/{telescope}/")
        return []

    if restrict_to_obsids is not None:
        keep = set(restrict_to_obsids)
        images = [p for p in images if p.parent.name in keep]

    gdc_note = "with GDC re-propagation" if lib_dir else "GDC approximated (no lib_dir)"
    print("\n" + "─"*50)
    print(f"Step 3b: Re-classifying stars ({len(images)} images, "
          f"conc_limit={conc_lo}, {gdc_note})")
    print("─"*50)

    # Pre-load py1pass GDC helpers if lib_dir given
    _gdc_helpers = None
    if lib_dir is not None:
        try:
            from pypass.io import (_apply_gdc_wcs, find_gdc, load_stdgdc,
                                    get_chip_config_from_fits, _DETECTOR_PREFIX)
            _gdc_helpers = dict(
                apply_gdc_wcs=_apply_gdc_wcs,
                find_gdc=find_gdc,
                load_stdgdc=load_stdgdc,
                get_chip_config_from_fits=get_chip_config_from_fits,
                DETECTOR_PREFIX=_DETECTOR_PREFIX,
            )
        except Exception as _e:
            print(f"  WARNING: could not load GDC helpers from py1pass: {_e}. "
                  "GDC covariance will be approximated.")

    updated = []
    for img in images:
        catalog = img.parent / f"{img.stem}_catalog.fits"
        if not catalog.exists():
            print(f"  {img.name}: no catalog found — run PSF fitting first")
            continue

        img_name = img.name
        img_dir  = img.parent

        try:
            t = Table.read(str(catalog))
        except Exception as e:
            print(f"  {img_name}: could not read catalog: {e}")
            continue

        records, old_fx, old_fy, old_eps, old_chi2_scales = _records_from_fits_table(t)

        # 1. Re-classify with new conc_limit
        n_old = int(sum(r.is_star_candidate for r in records))
        classify_stars(records, conc_lo=conc_lo)
        n_new = int(sum(r.is_star_candidate for r in records))

        # 2. Re-run chi2 inflation with the new star population
        inflate_chi2(records, zero_point=0.0)

        # 3. Re-apply GDC Jacobian if helpers available
        gdc_reapplied = False
        if _gdc_helpers is not None:
            try:
                from astropy.io import fits as _fits
                with _fits.open(str(img)) as hdul:
                    primary_hdr = hdul[0].header
                instrume = primary_hdr.get('INSTRUME', '').strip().upper()
                detector = primary_hdr.get('DETECTOR', '').strip().upper()
                det_prefix = _gdc_helpers['DETECTOR_PREFIX'].get((instrume, detector))
                if det_prefix:
                    gdc_dir = Path(lib_dir) / 'STDGDCs' / det_prefix
                    gdc_path = _gdc_helpers['find_gdc'](str(gdc_dir), primary_hdr) \
                               if gdc_dir.is_dir() else None
                    if gdc_path and os.path.exists(gdc_path):
                        gdc = _gdc_helpers['load_stdgdc'](gdc_path)
                        chips = _gdc_helpers['get_chip_config_from_fits'](
                            str(img), instrume, detector)
                        _gdc_helpers['apply_gdc_wcs'](
                            records, gdc, str(img), chips, instrume, detector)
                        gdc_reapplied = True
            except Exception as _e:
                print(f"  WARNING: {img_name}: GDC re-application failed: {_e}")

        # 4. Re-estimate systematic floor with new star classification + chi2_scale
        floor = estimate_systematic_floor(records)
        if floor is not None:
            new_fx  = floor.get('sigma_x_floor_A', old_fx) or old_fx
            new_fy  = floor.get('sigma_y_floor_A', old_fy) or old_fy
            new_eps = floor.get('eps_flux_A',       old_eps) or old_eps
        else:
            new_fx, new_fy, new_eps = old_fx, old_fy, old_eps

        # 5. Patch all dependent catalog columns
        _apply_new_floor(t, records, new_fx, new_fy, new_eps, floor,
                         old_chi2_scales=old_chi2_scales)

        # Sidecar safety: delete sidecar, write catalog, rewrite sidecar.
        params_path = img_dir / "psf_params.json"
        try:
            existing_params = json.loads(params_path.read_text()) if params_path.exists() else {}
        except Exception:
            existing_params = {}
        if params_path.exists():
            params_path.unlink()
        t.write(str(catalog), overwrite=True)
        existing_params['conc_limit'] = conc_lo
        params_path.write_text(json.dumps(existing_params, indent=2))

        # Invalidate downstream cross-match cache
        for _f in ('matched_gaia.csv', 'xmatch_params.json'):
            _p = img_dir / _f
            if _p.exists():
                _p.unlink()

        # Regenerate diagnostic figures
        try:
            plot_catalog_stats(
                records, floor_params=floor,
                output=str(img_dir / "psf_catalog_stats.png"),
                title=img_name,
            )
        except Exception as _e:
            print(f"  WARNING: psf_catalog_stats.png failed: {_e}")

        try:
            plot_concentration_diagnostics(
                records, conc_limit=conc_lo,
                output=str(img_dir / "psf_concentration.png"),
                title=img_name,
            )
        except Exception as _e:
            print(f"  WARNING: psf_concentration.png failed: {_e}")

        gdc_tag = " [GDC re-propagated]" if gdc_reapplied else " [GDC approx]"
        delta = n_new - n_old
        sign  = "+" if delta >= 0 else ""
        print(f"  {img_name}: {n_old}→{n_new} stars  ({sign}{delta})  "
              f"floor_x={new_fx:.4f} floor_y={new_fy:.4f}{gdc_tag}")
        updated.append(catalog)

    print(f"  Reclassification complete: {len(updated)}/{len(images)} catalogs updated.")
    return updated


def _save_psf_residuals(img_dir, pert, used_corrected_psf, pert_wing=None, hw_wing=12):
    """Save raw PSF perturbation accumulators to psf_residuals.npz.

    used_corrected_psf : bool — True when measurement used (stdpsf + existing δP).
    pert_wing : optional dict returned by a second measure_psf_perturbation call
        on isolated stars with hw_pert=hw_wing.  When provided, the wing arrays
        (sum_wv_combined_wing, etc.) are stored alongside the core arrays so that
        loaders can composite core (inner region) and wing (outer region) δP.
    hw_wing : int — hw_pert value used for the wing accumulation pass.
    """
    _zeros = np.zeros_like(pert['weight_map'])
    _n_chip = pert.get('n_stars_by_chip', {})
    save_kw = dict(
        sum_wv_combined          = pert.get('raw_sum_wv', _zeros),
        sum_w_combined           = pert.get('raw_sum_w',  _zeros),
        sum_wv_chip1             = pert.get('raw_sum_wv_by_chip', {}).get(1, _zeros),
        sum_w_chip1              = pert.get('raw_sum_w_by_chip',  {}).get(1, _zeros),
        sum_wv_chip4             = pert.get('raw_sum_wv_by_chip', {}).get(4, _zeros),
        sum_w_chip4              = pert.get('raw_sum_w_by_chip',  {}).get(4, _zeros),
        n_stars_used_combined    = np.array([pert['n_stars']]),
        n_stars_initial_combined = np.array([pert.get('n_stars_initial', pert['n_stars'])]),
        n_stars_chip1            = np.array([_n_chip.get(1, 0)]),
        n_stars_chip4            = np.array([_n_chip.get(4, 0)]),
        used_corrected_psf       = np.array([used_corrected_psf]),
    )
    if pert_wing is not None:
        _n_chip_w = pert_wing.get('n_stars_by_chip', {})
        save_kw.update(dict(
            sum_wv_combined_wing  = pert_wing.get('raw_sum_wv', _zeros),
            sum_w_combined_wing   = pert_wing.get('raw_sum_w',  _zeros),
            sum_wv_chip1_wing     = pert_wing.get('raw_sum_wv_by_chip', {}).get(1, _zeros),
            sum_w_chip1_wing      = pert_wing.get('raw_sum_w_by_chip',  {}).get(1, _zeros),
            sum_wv_chip4_wing     = pert_wing.get('raw_sum_wv_by_chip', {}).get(4, _zeros),
            sum_w_chip4_wing      = pert_wing.get('raw_sum_w_by_chip',  {}).get(4, _zeros),
            n_stars_wing_combined = np.array([pert_wing['n_stars']]),
            n_stars_wing_chip1    = np.array([_n_chip_w.get(1, 0)]),
            n_stars_wing_chip4    = np.array([_n_chip_w.get(4, 0)]),
            hw_wing               = np.array([hw_wing]),
        ))
    np.savez(str(img_dir / "psf_residuals.npz"), **save_kw)


def remeasure_psf_perturbation(
    output_dir: Path,
    field_name: str,
    lib_dir: Path,
    telescope: str = 'HST',
    im_type: str = '_flc',
    restrict_to_obsids: list[str] | None = None,
    psf_dir: Path | None = None,
    half_width: int | None = None,
    fmin_thresh: float | None = None,
    hw_wing: int = 12,
    wing_isolation_buffer: int = 2,
    verbose: bool = True,
) -> list[Path]:
    """Re-measure PSF perturbation on already-fitted images without re-fitting.

    Loads the existing PSF catalog for each image, reconstructs the final
    residual image by subtracting all accepted star models from a fresh copy of
    the image data, then calls measure_psf_perturbation to drizzle normalised
    leave-one-out residuals into the oversampled PSF grid.

    Overwrites psf_delta.npy and psf_perturbation.png in each image directory.

    Parameters
    ----------
    output_dir   : pipeline root directory
    field_name   : field subdirectory name
    lib_dir      : directory containing STDPSFs/ and STDGDCs/
    telescope    : 'HST' (JWST not yet supported)
    im_type      : '_flc' or '_flt'
    restrict_to_obsids : if given, only process these obs_ids
    psf_dir      : unused (pypass is installed as a package); kept for API compatibility
    half_width   : fitting half-width in detector pixels (default: _HST_DEFAULTS)
    fmin_thresh  : hard lower bound on detection flux threshold (default: _HST_DEFAULTS)
    hw_wing      : half-width for the wing accumulation pass (default 12, covering
        the full 101×101 PSF array at 4× oversampling).  Only stars isolated by at
        least hw_wing + wing_isolation_buffer detector pixels from any neighbour are
        used.  Set to 0 to skip the wing pass.
    wing_isolation_buffer : extra isolation margin beyond hw_wing (default 2 det px)
    verbose      : print per-image progress

    Returns
    -------
    List of image paths where perturbation was measured successfully.
    """
    _ensure_py1pass()
    # psf_dir parameter retained for API compatibility but no longer needed;
    # pypass is installed as a package.

    from pypass.io import (load_image, load_stdpsf, find_psf,
                            get_chip_config_from_fits, _DETECTOR_PREFIX)
    from pypass.diagnostics import measure_psf_perturbation, plot_psf_perturbation
    from pypass.multipass import subtract_stars
    from scipy.ndimage import spline_filter as _spline_filter
    from astropy.io import fits as _fits
    from astropy.table import Table

    _hw   = half_width if half_width is not None else _HST_DEFAULTS['half_width']
    _fmin = fmin_thresh if fmin_thresh is not None else _HST_DEFAULTS['fmin_thresh']

    from .download_hst import find_flc_images
    images = find_flc_images(output_dir, field_name, telescope=telescope,
                              im_type=im_type)
    if not images:
        print(f"[remeasure_psf_pert] No {im_type} images found.")
        return []

    if restrict_to_obsids is not None:
        keep = set(restrict_to_obsids)
        images = [p for p in images if p.parent.name in keep]

    lib_dir = Path(lib_dir)
    n_images = len(images)
    print("\n" + "─"*50)
    print(f"Step 3c: Re-measuring PSF perturbation ({n_images} images)")
    print("─"*50)

    done = []
    for img_i, img in enumerate(images, 1):
        catalog = img.parent / f"{img.stem}_catalog.fits"
        if not catalog.exists():
            print(f"  [{img_i}/{n_images}] {field_name}  {img.name}: no catalog — run PSF fitting first")
            continue

        img_name = img.name
        img_dir  = img.parent

        try:
            t = Table.read(str(catalog))
            records, _, _, _, _ = _records_from_fits_table(t)

            with _fits.open(str(img)) as hdul:
                primary_hdr = hdul[0].header
            instrume = primary_hdr.get('INSTRUME', '').strip().upper()
            detector = primary_hdr.get('DETECTOR', '').strip().upper()

            det_prefix = _DETECTOR_PREFIX.get((instrume, detector))
            if det_prefix is None:
                print(f"  {img_name}: unknown instrument {instrume}/{detector} — skipping")
                continue

            psf_dir_stdpsf = lib_dir / 'STDPSFs' / det_prefix
            psf_path = find_psf(str(psf_dir_stdpsf), primary_hdr)
            stdpsf_cube, xs, ys, psf_scale, _ = load_stdpsf(psf_path)

            # Load existing cumulative delta (if any) and apply it so that
            # subtraction and perturbation measurement use the same PSF model
            # that was used during the original fit.  The newly measured delta
            # is incremental w.r.t. (stdpsf + existing_delta).
            existing_delta = None
            _delta_path = img_dir / "psf_delta.npy"
            if _delta_path.exists():
                try:
                    existing_delta = np.load(str(_delta_path))
                except Exception as _de:
                    print(f"  WARNING: {img_name}: could not load psf_delta.npy: {_de}")

            if existing_delta is not None:
                peak = float(np.abs(existing_delta).max())
                print(f"  [{img_i}/{n_images}] {field_name}  [{img_name}] PSF: CORRECTED (stored δP, cumulative peak = {peak:+.5f})")
                psf_cube = stdpsf_cube + existing_delta[np.newaxis, :, :]
            else:
                print(f"  [{img_i}/{n_images}] {field_name}  [{img_name}] PSF: BARE stdpsf (no stored δP found)")
                psf_cube = stdpsf_cube

            psf_coeffs_cube = np.array([
                _spline_filter(p, order=3, output=np.float64) for p in psf_cube
            ])

            chips = get_chip_config_from_fits(str(img), instrume, detector)

            # Load residual images saved by the original py1pass fit.
            # These are the exact leave-one-out residuals computed during the
            # Newton iterations — preferred over a subtract_stars reconstruction
            # which subtracts all stars simultaneously and is not leave-one-out.
            # Fall back to reconstruction only if the file is absent (legacy run).
            _res_fits_path = img_dir / f"{img.stem}_residual.fits"
            residuals_by_chip = {}
            masks_by_chip     = {}

            if _res_fits_path.exists():
                try:
                    with _fits.open(str(_res_fits_path)) as _rh:
                        ext_names = [h.name for h in _rh]
                        for sci_ext, dq_ext, _y_off_chip in chips:
                            res_ext  = f'SCI{sci_ext}'
                            mask_ext = f'MASK{sci_ext}'
                            if res_ext in ext_names:
                                residuals_by_chip[sci_ext] = _rh[res_ext].data.astype(np.float64)
                            # Load mask from saved MASK extension; fall back to DQ re-read
                            if mask_ext in ext_names:
                                # Bitmask: bit0=DQ-valid, bit1=not-sigma-clipped.
                                # A pixel was used only if both bits are set (value==3).
                                # load_image convention: mask=True means BAD.
                                _m = _rh[mask_ext].data
                                masks_by_chip[sci_ext] = (_m != 3)  # True=bad
                            else:
                                _, _, _, _dq, _, _ = load_image(
                                    str(img), sci_ext=sci_ext, dq_ext=dq_ext)
                                if _dq is not None:
                                    masks_by_chip[sci_ext] = _dq
                    # Convert combined-frame x/y → chip-local for records
                    for sci_ext, dq_ext, _y_off_chip in chips:
                        _, _, _, _, x_off, y_off = load_image(
                            str(img), sci_ext=sci_ext, dq_ext=dq_ext)
                        for r in [r for r in records
                                  if getattr(r, '_chip_ext', sci_ext) == sci_ext]:
                            r.x = r.x - x_off
                            r.y = r.y - y_off
                            r._x_offset = x_off
                            r._y_offset = y_off
                    if residuals_by_chip:
                        print(f"    loaded saved residual FITS ({len(residuals_by_chip)} chip(s), "
                              f"{sum(1 for k in residuals_by_chip if k in masks_by_chip)} with mask)")
                    else:
                        raise ValueError("no SCI* extensions found")
                except Exception as _re:
                    print(f"    WARNING: could not load {_res_fits_path.name}: {_re}; "
                          f"falling back to subtract_stars reconstruction")
                    residuals_by_chip = {}

            if not residuals_by_chip:
                # Fall back: reconstruct by subtracting all accepted star models.
                # Note this is not leave-one-out; the saved residual FITS is preferred.
                print(f"    no saved residual FITS found — reconstructing via subtract_stars")
                for sci_ext, dq_ext, _y_off_chip in chips:
                    data, _gain, _rn, _mask, x_off, y_off = load_image(
                        str(img), sci_ext=sci_ext, dq_ext=dq_ext)
                    chip_records = [r for r in records
                                    if getattr(r, '_chip_ext', sci_ext) == sci_ext]
                    for r in chip_records:
                        r.x = r.x - x_off
                        r.y = r.y - y_off
                        r._x_offset = x_off
                        r._y_offset = y_off
                    residual = data.copy()
                    subtract_stars(residual, chip_records, psf_cube, xs, ys,
                                   psf_scale, _hw,
                                   x_offset=x_off, y_offset=y_off,
                                   psf_coeffs_cube=psf_coeffs_cube)
                    residuals_by_chip[sci_ext] = residual
                    if _mask is not None:
                        masks_by_chip[sci_ext] = _mask

            pert = measure_psf_perturbation(
                records=records,
                residuals_by_chip=residuals_by_chip,
                psf_cube=psf_cube, xs=xs, ys=ys,
                psf_scale=psf_scale, hw=_hw,
                fmin=_fmin,
                psf_coeffs_cube=psf_coeffs_cube,
                masks_by_chip=masks_by_chip or None,
                return_accumulators=True,
            )

            # Save cumulative delta relative to bare stdpsf.
            delta_new = pert['delta_psf']
            cumulative_delta = (existing_delta + delta_new) \
                               if existing_delta is not None else delta_new
            np.save(str(img_dir / "psf_delta.npy"), cumulative_delta)
            plot_psf_perturbation(
                psf_center=pert['psf_center'],
                delta_psf=cumulative_delta,
                weight_map=pert['weight_map'],
                output=str(img_dir / "psf_perturbation.png"),
                title=img_name,
            )

            # Wing accumulation pass: re-run on isolated stars with hw_pert=hw_wing.
            # At hw_wing=12 det px this covers the full 101×101 PSF array.
            # Isolation is safe at these distances — stars >hw_wing px away are
            # reliably detected in single-pass catalogs.
            pert_wing = None
            if hw_wing > 0:
                try:
                    from scipy.spatial import cKDTree as _KDTree
                    _min_sep = hw_wing + wing_isolation_buffer
                    # Compute per-chip nearest-neighbour distances in chip-local coords.
                    # Must be per-chip: chip-local y coords overlap between chips so
                    # cross-chip distances would be spuriously small.
                    _nn = {}
                    for _sci, _, _ in chips:
                        _crecs = [r for r in records
                                  if getattr(r, '_chip_ext', _sci) == _sci]
                        if len(_crecs) >= 2:
                            _pos = np.array([[r.x, r.y] for r in _crecs])
                            _d, _ = _KDTree(_pos).query(_pos, k=2)
                            for r, d in zip(_crecs, _d[:, 1]):
                                _nn[id(r)] = float(d)
                        else:
                            for r in _crecs:
                                _nn[id(r)] = np.inf
                    isolated = [r for r in records if _nn.get(id(r), np.inf) >= _min_sep]
                    if len(isolated) >= 5:
                        pert_wing = measure_psf_perturbation(
                            records=isolated,
                            residuals_by_chip=residuals_by_chip,
                            psf_cube=psf_cube, xs=xs, ys=ys,
                            psf_scale=psf_scale, hw=_hw,
                            hw_pert=hw_wing,
                            fmin=_fmin,
                            psf_coeffs_cube=psf_coeffs_cube,
                            masks_by_chip=masks_by_chip or None,
                            return_accumulators=True,
                        )
                        if verbose:
                            print(f"    wing pass: {len(isolated)} isolated stars "
                                  f"(sep ≥ {_min_sep} px), "
                                  f"{pert_wing['n_stars']} used after clipping")
                    else:
                        if verbose:
                            print(f"    wing pass skipped: only {len(isolated)} "
                                  f"isolated stars (need ≥ 5)")
                except Exception as _we:
                    print(f"  [{img_name}] WARNING: wing pass failed: {_we}")

            _save_psf_residuals(img_dir, pert,
                                used_corrected_psf=existing_delta is not None,
                                pert_wing=pert_wing, hw_wing=hw_wing)

            n_clipped = pert.get('n_outliers_clipped', 0)
            clip_str  = f", {n_clipped} outlier(s) σ-clipped" if n_clipped else ""
            print(f"  [{img_name}] perturbation: {pert['n_stars']} stars{clip_str}  "
                  f"incremental peak = {delta_new.max():+.4f}  "
                  f"cumulative peak = {cumulative_delta.max():+.4f}")
            if verbose:
                ca = pert['constraints_after']
                print(f"    sum after={ca['sum']:.2e}  "
                      f"mx={ca['mx']:.2e}  my={ca['my']:.2e}")
            done.append(img)

        except Exception as exc:
            import traceback
            print(f"  ERROR {img_name}: {exc}")
            if verbose:
                traceback.print_exc()

    print(f"  PSF perturbation re-measured: {len(done)}/{len(images)} images.")
    return done


def _get_image_header_info(img_path):
    """Read minimal FITS header info for the one-liner status print. Fast — primary header only."""
    try:
        from astropy.io import fits as _f
        with _f.open(str(img_path), memmap=False) as h:
            hdr = h[0].header
        instrume = hdr.get('INSTRUME', '?').strip()
        detector = hdr.get('DETECTOR', '').strip()
        instdet  = f"{instrume}/{detector}" if detector else instrume
        filt = hdr.get('FILTER2', hdr.get('FILTER1', hdr.get('FILTER', '?'))).strip()
        exptime  = float(hdr.get('EXPTIME', 0))
        return {'instdet': instdet, 'filter': filt, 'exptime': exptime}
    except Exception:
        return {'instdet': '?', 'filter': '?', 'exptime': 0}


# Global status queue set by pool initializer in parallel mode.
_status_queue = None

def _worker_pool_init(queue):
    global _status_queue
    _status_queue = queue


def _image_worker(args):
    """Parallel worker: fit all PSF iterations for one image, log to file.

    Sets OMP_NUM_THREADS=1 and jax_num_cpu_devices=1 so workers don't
    fight over cores.  All verbose output goes to psf_fitting_log.txt in the
    image's subfolder.  Status messages (start/finish/fail) are sent to the
    shared queue for the main process to print as one-liners.

    args: (img_path, catalog_path, lib_dir, params, params_meta_disk,
           n_img_iter, clean_psf, apply_psf_delta, force_refit)
    Returns: (success, str(img_path), n_found, n_converged, n_stars, elapsed, error_msg)
    """
    import os, sys, time, traceback
    global _status_queue

    (img_path, catalog_path, lib_dir, params, params_meta_disk,
     n_img_iter, clean_psf, apply_psf_delta, force_refit) = args

    img_path    = Path(img_path)
    catalog_path = Path(catalog_path)
    log_path    = img_path.parent / 'psf_fitting_log.txt'
    img_name    = img_path.name

    # Limit resource usage: one worker = one core.
    for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[_var] = '1'
    try:
        import jax as _jax
        _jax.config.update('jax_num_cpu_devices', 1)
    except Exception:
        pass

    t0 = time.perf_counter()

    if _status_queue is not None:
        _status_queue.put(('start', img_name))

    try:
        with open(log_path, 'w', buffering=1) as _log:
            _old_stdout = sys.stdout
            sys.stdout  = _log
            try:
                # Replicate the per-image PSF iteration loop from run_psf_fitting.
                use_clean = clean_psf or (not apply_psf_delta) or force_refit
                if use_clean:
                    initial_delta = None
                else:
                    delta_path = img_path.parent / 'psf_delta.npy'
                    try:
                        initial_delta = np.load(str(delta_path)) if delta_path.exists() else None
                    except Exception:
                        initial_delta = None

                current_delta = initial_delta
                succeeded = False

                for iter_i in range(n_img_iter):
                    label = f"iter {iter_i + 1}/{n_img_iter}"
                    if current_delta is not None:
                        print(f"   [{label}] PSF: CORRECTED (peak = {float(np.abs(current_delta).max()):+.5f})")
                    else:
                        print(f"   [{label}] PSF: BARE stdpsf")

                    w = (img_path, catalog_path, lib_dir, params,
                         params_meta_disk, True, current_delta)
                    path, n_stars, err = _fit_one_image(w)
                    if err:
                        print(f"   ERROR [{label}] {img_name}: {err}")
                        break
                    print(f"   [{label}] {img_name}: {n_stars} stars fitted")
                    succeeded = True

                    # Preserve intermediate PSF figures before next iteration.
                    if iter_i < n_img_iter - 1:
                        import shutil as _sh
                        for _fig in ("psf_catalog_stats.png", "psf_concentration.png",
                                     "psf_diagnostics.png", "psf_residual_map.png",
                                     "psf_perturbation.png"):
                            _src = img_path.parent / _fig
                            if _src.exists():
                                _sh.copy2(str(_src),
                                          str(img_path.parent / (_src.stem + f"_iter{iter_i+1}" + _src.suffix)))

                    # Load updated delta for next iteration.
                    if iter_i < n_img_iter - 1:
                        delta_path = img_path.parent / 'psf_delta.npy'
                        if delta_path.exists():
                            try:
                                current_delta = np.load(str(delta_path))
                            except Exception as _e:
                                print(f"   WARNING: could not load δP for next iter: {_e}")
                                break
                        else:
                            print(f"   WARNING: no psf_delta.npy after {label} — stopping")
                            break

            except Exception:
                print(traceback.format_exc())
                succeeded = False
            finally:
                sys.stdout = _old_stdout

        # Extract summary stats from the written catalog.
        n_found = n_converged = n_stars_cls = 0
        if succeeded and catalog_path.exists():
            try:
                from astropy.table import Table as _T
                _cat = _T.read(str(catalog_path))
                n_found = len(_cat)
                n_converged  = int(np.sum(_cat['converged'])) if 'converged' in _cat.colnames else n_found
                n_stars_cls  = int(np.sum(_cat['is_star']))   if 'is_star'   in _cat.colnames else n_found
            except Exception:
                n_found = n_converged = n_stars_cls = 0

        elapsed = time.perf_counter() - t0
        if _status_queue is not None:
            _status_queue.put(('done', img_name, n_found, n_converged, n_stars_cls, elapsed))
        return (succeeded, str(img_path), n_found, n_converged, n_stars_cls, elapsed, None)

    except Exception as e:
        elapsed = time.perf_counter() - t0
        tb = traceback.format_exc()
        try:
            with open(log_path, 'a') as _log:
                _log.write(f"\nFATAL ERROR: {e}\n{tb}")
        except Exception:
            pass
        if _status_queue is not None:
            _status_queue.put(('fail', img_name, str(e), elapsed))
        return (False, str(img_path), 0, 0, 0, elapsed, str(e))


def _fit_one_image(args):
    """Fit a single FLC image. Returns (path, n_stars, error).

    args is a 7-tuple: (image_path, out_catalog, lib_dir, params,
                        params_meta, verbose, psf_delta)
    psf_delta : None (use stdpsf as-is) or (psf_size, psf_size) ndarray
                to add to every PSF in the grid before fitting.
    """
    (image_path, out_catalog, lib_dir, params, params_meta, verbose, psf_delta) = args
    _ensure_py1pass()
    from pypass.io import (run_photometry_fits, catalog_to_table,
                            load_stdpsf, load_image, get_chip_config)
    from pypass.diagnostics import (estimate_systematic_floor,
                                     plot_catalog_stats, plot_diagnostics,
                                     plot_psf_residual_map,
                                     plot_concentration_diagnostics)
    from astropy.io import fits as _fits

    img_dir = Path(image_path).parent
    img_name = Path(image_path).name

    try:
        if psf_delta is not None:
            peak = float(np.abs(psf_delta).max())
            print(f"  [{img_name}] PSF: CORRECTED (cumulative δP peak = {peak:+.5f})")
        else:
            print(f"  [{img_name}] PSF: BARE stdpsf (no perturbation applied)")

        result = run_photometry_fits(
            image_path=str(image_path),
            psf_path=None,
            lib_dir=str(lib_dir) if lib_dir else None,
            return_residual=True,
            verbose=verbose,
            psf_delta=psf_delta,
            **params,
        )
        records, residuals, var_images, psf_path, gdc_path = result

        floor = estimate_systematic_floor(records)
        fx = floor['sigma_x_floor_A'] if floor else 0.0
        fy = floor['sigma_y_floor_A'] if floor else 0.0
        ff = floor['eps_flux_A']      if floor else 0.0

        table = catalog_to_table(records, params.get('zero_point', 0.0),
                                  sigma_floor_x=fx, sigma_floor_y=fy,
                                  eps_flux=ff, floor_params=floor)

        # Remove sidecar BEFORE writing the catalog so an interrupted write
        # leaves no stale sidecar that could mark a partial file as valid.
        params_path = Path(out_catalog).parent / "psf_params.json"
        if params_path.exists():
            params_path.unlink()

        table.write(str(out_catalog), overwrite=True)
        params_path.write_text(json.dumps(params_meta, indent=2))

        # Sky sanity check: warn if the median fitted sky is anomalously low.
        # A properly exposed science image always accumulates at least a few
        # counts of sky background per pixel.  Near-zero sky indicates a failed
        # observation (e.g. EXPFLAG=NORMAL but telescope off-target, or
        # background subtraction pathology).
        _sky_vals = np.array([r.sky for r in records if np.isfinite(r.sky)])
        if _sky_vals.size > 0:
            _med_sky = float(np.median(_sky_vals))
            _exptime = params_meta.get('exptime', None)
            if _exptime is None:
                try:
                    from astropy.io import fits as _fits_sk
                    _exptime = float(_fits_sk.getval(str(image_path), 'EXPTIME', ext=0))
                except Exception:
                    _exptime = None
            _sky_per_sec = _med_sky / _exptime if (_exptime and _exptime > 0) else None
            _sky_warn = (_sky_per_sec is not None and _sky_per_sec < 0.005) or \
                        (_sky_per_sec is None and _med_sky < 2.0)
            if _sky_warn:
                print(f"  WARNING: [{img_name}] anomalously low sky — "
                      f"median sky = {_med_sky:.2f} counts"
                      + (f" ({_sky_per_sec:.4f} counts/s/px)" if _sky_per_sec else "")
                      + " — image may have no real sky signal.")

        # Save per-chip residual, variance, and combined mask to a single FITS.
        # Extensions per chip (e.g. chip SCI extension = 1 or 4):
        #   SCI{ext}  — PSF-subtracted residual (float32, DN)
        #   VAR{ext}  — per-pixel noise variance used in fitting (float32, DN²)
        #   MASK{ext} — combined good-pixel map (uint8):
        #                 bit 0 (1) = DQ-valid (not flagged by detector)
        #                 bit 1 (2) = not sigma-clipped during Newton iterations
        #               value 3 = pixel was fully used; 0 = excluded entirely
        # This captures both the static DQ exclusions and the dynamic
        # sigma-clip outliers identified by py1pass during fitting.
        try:
            from astropy.io import fits as _fits_res
            _res_path = Path(out_catalog).parent / f"{Path(image_path).stem}_residual.fits"
            _hdus = [_fits_res.PrimaryHDU()]
            _sci_exts = sorted(residuals.keys())

            for _sci_ext in _sci_exts:
                _shape = residuals[_sci_ext].shape
                _ny, _nx = _shape

                # ── DQ mask ──────────────────────────────────────────────────
                _dq_ext = _sci_ext + 2
                try:
                    _, _, _, _dq_mask, _x_off, _y_off = load_image(
                        str(image_path), sci_ext=_sci_ext, dq_ext=_dq_ext)
                    _dq_good = (~_dq_mask).astype(np.uint8) if _dq_mask is not None \
                               else np.ones(_shape, dtype=np.uint8)
                except Exception:
                    _dq_good  = np.ones(_shape, dtype=np.uint8)
                    _x_off, _y_off = 0.0, 0.0

                # ── Sigma-clip mask: paint clipped_mask from each StarRecord ─
                # clipped_mask on a record is True for pixels that passed DQ
                # but were rejected by sigma-clipping during Newton iterations.
                # Shape matches the final fit window, clipped to image boundary.
                _sigma_good = np.ones(_shape, dtype=np.uint8)  # 1=not clipped
                _chip_records = [r for r in records
                                 if getattr(r, '_chip_ext', _sci_ext) == _sci_ext]
                _hw = params.get('half_width', _HST_DEFAULTS['half_width'])
                for _r in _chip_records:
                    _cm = getattr(_r, 'clipped_mask', None)
                    if _cm is None:
                        continue
                    try:
                        # Records have full-frame coords; convert to chip-local.
                        _xi = int(round(float(_r.x - _x_off)))
                        _yi = int(round(float(_r.y - _y_off)))
                        _y0 = max(0, _yi - _hw)
                        _y1 = min(_ny, _yi + _hw + 1)
                        _x0 = max(0, _xi - _hw)
                        _x1 = min(_nx, _xi + _hw + 1)
                        _cm_h = _y1 - _y0
                        _cm_w = _x1 - _x0
                        if _cm_h <= 0 or _cm_w <= 0:
                            continue
                        _cm_arr = np.asarray(_cm)
                        # JAX backend may store clipped_mask as 1-D; reshape if so.
                        if _cm_arr.ndim == 1:
                            if _cm_arr.size == _cm_h * _cm_w:
                                _cm_arr = _cm_arr.reshape(_cm_h, _cm_w)
                            else:
                                continue
                        # clipped_mask covers the actual (clipped) window directly.
                        _sigma_good[_y0:_y1, _x0:_x1] &= (~_cm_arr[:_cm_h, :_cm_w]).astype(np.uint8)
                    except Exception:
                        continue  # skip this record's sigma-clip contribution

                # Combined mask: bit 0=DQ-valid, bit 1=not-sigma-clipped
                _combined = _dq_good | (_sigma_good << 1)

                # ── Write extensions ─────────────────────────────────────────
                _hdus.append(_fits_res.ImageHDU(
                    data=residuals[_sci_ext].astype(np.float32),
                    name=f'SCI{_sci_ext}'))
                if _sci_ext in var_images:
                    _hdus.append(_fits_res.ImageHDU(
                        data=var_images[_sci_ext].astype(np.float32),
                        name=f'VAR{_sci_ext}'))
                _hdus.append(_fits_res.ImageHDU(
                    data=_combined, name=f'MASK{_sci_ext}'))

            _fits_res.HDUList(_hdus).writeto(str(_res_path), overwrite=True)
        except Exception as _e:
            print(f"  WARNING: [{img_name}] could not save residual FITS: {_e}")

        # Invalidate downstream cross-match cache so it reruns with the new catalog
        for _f in ('matched_gaia.csv', 'xmatch_params.json'):
            _p = Path(out_catalog).parent / _f
            if _p.exists():
                _p.unlink()

        # ── Diagnostic figures ────────────────────────────────────────────────
        # catalog_stats: all records, no image data needed
        try:
            plot_catalog_stats(
                records, floor_params=floor,
                output=str(img_dir / "psf_catalog_stats.png"),
                title=img_name,
            )
        except Exception as _e:
            print(f"  WARNING: psf_catalog_stats.png failed: {_e}")

        # concentration diagnostics: star/non-star classification vs magnitude
        try:
            plot_concentration_diagnostics(
                records,
                conc_limit=params.get('conc_limit', _HST_DEFAULTS['conc_limit']),
                output=str(img_dir / "psf_concentration.png"),
                title=img_name,
            )
        except Exception as _e:
            print(f"  WARNING: psf_concentration.png failed: {_e}")

        # diagnostics + residual map: use first chip only (matches CLI behaviour)
        try:
            with _fits.open(str(image_path)) as hdul:
                primary_hdr = hdul[0].header
            instrume = primary_hdr.get('INSTRUME', '').strip()
            detector = primary_hdr.get('DETECTOR', '').strip()
            chips    = get_chip_config(instrume, detector)
            sci_ext, dq_ext = chips[0][0], chips[0][1]

            data, gain, rn, chip_mask, x_off, y_off = load_image(
                str(image_path), sci_ext=sci_ext, dq_ext=dq_ext)
            psf_cube, xs, ys, psf_scale, _ = load_stdpsf(psf_path)
            # Apply PSF delta so diagnostics/residual-map use the same model
            # that was actually used for fitting.
            if psf_delta is not None:
                psf_cube = psf_cube + psf_delta[np.newaxis, :, :]
            hw = params.get('half_width', _HST_DEFAULTS['half_width'])

            chip_records = [r for r in records
                            if getattr(r, '_chip_ext', sci_ext) == sci_ext]
            if not chip_records:
                chip_records = records

            plot_diagnostics(
                records=chip_records, data=data,
                psf_cube=psf_cube, xs=xs, ys=ys, psf_scale=psf_scale,
                hw=hw, x_offset=x_off, y_offset=y_off,
                residual=residuals.get(sci_ext),
                mask=chip_mask, noise_map=var_images.get(sci_ext),
                output=str(img_dir / "psf_diagnostics.png"),
                title=img_name,
            )

            plot_psf_residual_map(
                records=chip_records, data=data,
                psf_cube=psf_cube, xs=xs, ys=ys, psf_scale=psf_scale,
                hw=hw, x_offset=x_off, y_offset=y_off,
                gain=gain, read_noise=rn, noise_map=var_images.get(sci_ext),
                output=str(img_dir / "psf_residual_map.png"),
                title=img_name,
            )
        except Exception as _e:
            print(f"  WARNING: psf_diagnostics/residual_map.png failed: {_e}")

        # ── PSF perturbation measurement ──────────────────────────────────────
        # psf_delta.npy stores the CUMULATIVE correction relative to stdpsf so
        # that applying it on any subsequent run gives the best-known model.
        # The residuals are computed w.r.t. (stdpsf + psf_delta), so delta_new
        # is the incremental residual on top of the existing correction.
        # Cumulative = psf_delta (prior, may be None/0) + delta_new.
        #
        # IMPORTANT: we use the on-disk catalog (written just above) instead of
        # the in-memory records from run_photometry_fits.  The in-memory records
        # include ~15% non-converged stars that subtract_stars skips, leaving
        # their full PSF-sized blobs of light in the residual image.  These
        # blobs contaminate the leave-one-out windows of nearby converged stars,
        # producing large spurious perturbations.  The on-disk catalog has
        # non-converged stars already removed, so every source in it passes
        # _should_subtract and the residuals are clean.
        try:
            from pypass.diagnostics import (measure_psf_perturbation,
                                             plot_psf_perturbation)
            from pypass.io import get_chip_config_from_fits, _DETECTOR_PREFIX
            from pypass.multipass import subtract_stars as _subtract_stars
            from scipy.ndimage import spline_filter as _spline_filter_pert
            from astropy.table import Table as _Table

            try:
                psf_cube  # already set by the diagnostics block above (corrected)
            except NameError:
                psf_cube, xs, ys, psf_scale, _ = load_stdpsf(psf_path)
                if psf_delta is not None:
                    psf_cube = psf_cube + psf_delta[np.newaxis, :, :]

            # Pre-compute spline-filter coefficients for subtract_stars.
            psf_coeffs_cube = np.array([
                _spline_filter_pert(p, order=3, output=np.float64) for p in psf_cube
            ])

            # Load on-disk catalog: non-converged stars already removed.
            _disk_table = _Table.read(str(out_catalog))
            _disk_records, _, _, _, _ = _records_from_fits_table(_disk_table)

            with _fits.open(str(image_path)) as _hdul:
                _phdr = _hdul[0].header
            _instrume_p = _phdr.get('INSTRUME', '').strip().upper()
            _detector_p = _phdr.get('DETECTOR', '').strip().upper()
            _all_chips = get_chip_config_from_fits(
                str(image_path), _instrume_p, _detector_p)

            _hw_pert = params.get('half_width', _HST_DEFAULTS['half_width'])
            fresh_residuals: dict = {}
            fresh_masks:     dict = {}
            for _sci_ext, _dq_ext, _ in _all_chips:
                _data, _, _, _dq_mask, _x_off, _y_off = load_image(
                    str(image_path), sci_ext=_sci_ext, dq_ext=_dq_ext)
                _chip_recs = [r for r in _disk_records
                              if getattr(r, '_chip_ext', _sci_ext) == _sci_ext]
                # Catalog stores full-frame coordinates; convert to chip-local
                # so residual stamps land correctly inside the chip image.
                for r in _chip_recs:
                    r.x = r.x - _x_off
                    r.y = r.y - _y_off
                    r._x_offset = _x_off
                    r._y_offset = _y_off
                _residual = _data.copy()
                _subtract_stars(
                    _residual, _chip_recs, psf_cube, xs, ys,
                    psf_scale, _hw_pert,
                    x_offset=_x_off, y_offset=_y_off,
                    psf_coeffs_cube=psf_coeffs_cube,
                )
                fresh_residuals[_sci_ext] = _residual
                if _dq_mask is not None:
                    fresh_masks[_sci_ext] = _dq_mask

            _fmin_pert = params.get('fmin_thresh', _HST_DEFAULTS['fmin_thresh'])
            pert = measure_psf_perturbation(
                records=_disk_records,
                residuals_by_chip=fresh_residuals,
                psf_cube=psf_cube, xs=xs, ys=ys,
                psf_scale=psf_scale, hw=_hw_pert,
                fmin=_fmin_pert,
                psf_coeffs_cube=psf_coeffs_cube,
                masks_by_chip=fresh_masks or None,
                return_accumulators=True,
            )
            # Save cumulative delta (psf_delta is the prior applied during fitting).
            delta_new = pert['delta_psf']
            cumulative_delta = (psf_delta + delta_new) if psf_delta is not None \
                               else delta_new
            np.save(str(img_dir / "psf_delta.npy"), cumulative_delta)
            plot_psf_perturbation(
                psf_center=pert['psf_center'],
                delta_psf=cumulative_delta,
                weight_map=pert['weight_map'],
                output=str(img_dir / "psf_perturbation.png"),
                title=img_name,
            )
            _save_psf_residuals(img_dir, pert, used_corrected_psf=psf_delta is not None)
            cb = pert['constraints_before']
            ca = pert['constraints_after']
            n_clipped = pert.get('n_outliers_clipped', 0)
            clip_str  = f", {n_clipped} outlier(s) σ-clipped" if n_clipped else ""
            print(f"  [{img_name}] perturbation: {pert['n_stars']} stars{clip_str}  "
                  f"incremental peak = {np.abs(delta_new).max():.5f}  "
                  f"cumulative peak = {np.abs(cumulative_delta).max():.5f}")
            if verbose:
                print(f"    sum before/after: {cb['sum']:.2e} / {ca['sum']:.2e}  "
                      f"mx: {cb['mx']:.2e} / {ca['mx']:.2e}  "
                      f"my: {cb['my']:.2e} / {ca['my']:.2e}")
        except Exception as _e:
            print(f"  WARNING: psf_perturbation.png failed: {_e}")

        return str(image_path), len(records), None

    except Exception as exc:
        return str(image_path), 0, str(exc)


def _params_cache_status(output_path: Path, params_path: Path,
                          current_params: dict) -> tuple[bool, list[str]]:
    """
    Return (cache_valid, diffs).

    cache_valid is True when output_path exists, params_path exists, and all
    parameter values match.  diffs lists human-readable mismatches.
    """
    if not output_path.exists():
        return False, []
    if not params_path.exists():
        return False, ["no params sidecar — cannot verify configuration match"]
    try:
        saved = json.loads(params_path.read_text())
    except Exception as e:
        return False, [f"could not read params sidecar: {e}"]
    diffs = [f"  {k}: saved={saved.get(k)!r}  current={v!r}"
             for k, v in current_params.items() if saved.get(k) != v]
    return len(diffs) == 0, diffs


def run_psf_fitting(
    output_dir: Path,
    field_name: str,
    lib_dir: Path,
    telescope: str = 'HST',
    im_type: str = '_flc',
    n_processes: int = -1,
    verbose: bool = True,
    force_refit: bool = False,
    clean_psf: bool = False,
    apply_psf_delta: bool = False,
    n_psf_iter: int | None = None,
    restrict_to_obsids: list[str] | None = None,
    psf_dir: Path | None = None,
    parallel: bool = True,
    # py1pass parameter overrides
    fmin: float | None = None,
    fmin_thresh: float | None = None,
    mag_st_max: float | None = None,
    hmin: int | None = None,
    n_passes: int | None = None,
    n_discovery_passes: int | None = None,
    sat_threshold: float | None = None,
    max_iter_fit: int | None = None,
    half_width: int | None = None,
    conc_limit: float | None = None,
) -> list[Path]:
    """
    Run PSF fitting on all downloaded FLC images for a field.

    Each image is processed serially so it has full access to all available
    cores via py1pass's internal joblib parallelism (n_jobs=n_processes).
    Cached catalogs are reused when the saved py1pass parameters match the
    current call.

    PSF iteration logic (per image):
      - Default: 1 iteration from the bare stdpsf (ignores any stored δP).
      - ``apply_psf_delta=True`` loads the stored ``psf_delta.npy`` (if present)
          and uses it as the starting PSF model for the first iteration.
      - ``clean_psf=True`` always uses the bare stdpsf, overriding
          ``apply_psf_delta``.
      - ``n_psf_iter=N`` explicitly sets the number of iterations.  Pass
          ``n_psf_iter=2`` to enable the iterative PSF correction (fit → measure
          δP → re-fit with corrected PSF).  WARNING: applying the measured δP
          in a second pass can introduce bilinear-interpolation aliasing that
          degrades the 2-D pixel-phase distribution for sparse fields.  Only
          use ``n_psf_iter >= 2`` when you have many bright stars (≳1000).

    Parameters
    ----------
    output_dir   : pipeline root directory
    field_name   : field subdirectory name
    lib_dir      : directory containing STDPSFs/ and STDGDCs/ subdirectories
    telescope    : 'HST' (JWST support coming)
    im_type      : '_flc' or '_flt'
    n_processes  : cores for py1pass internal parallelism (-1 = all, default)
    force_refit      : re-fit even if catalog and matching params already exist
    clean_psf        : ignore stored psf_delta.npy; start from bare stdpsf (overrides apply_psf_delta)
    apply_psf_delta  : load stored psf_delta.npy (if present) as starting PSF model
    n_psf_iter       : explicit number of PSF fitting iterations (overrides default)
    psf_dir      : unused (pypass is installed as a package); kept for API compatibility

    Returns
    -------
    List of output catalog FITS paths
    """
    if telescope.upper() != 'HST':
        raise NotImplementedError(
            "PSF fitting for non-HST telescopes is not yet implemented. "
            "JWST support is planned once py1pass is updated for JWST headers."
        )

    # psf_dir parameter retained for API compatibility but no longer needed;
    # pypass is installed as a package.

    from .download_hst import find_flc_images
    images = find_flc_images(output_dir, field_name, telescope=telescope,
                              im_type=im_type)
    if not images:
        print(f"[PSF] No {im_type} images found under "
              f"{output_dir}/{field_name}/{telescope}/")
        return []

    if restrict_to_obsids is not None:
        keep = set(restrict_to_obsids)
        images = [p for p in images if p.parent.name in keep]
        if not images:
            print(f"  No images match the provided selection of {len(keep)} obs_ids.")
            return []
        print(f"  Restricting to {len(images)} selected image(s).")

    print("\n" + "─"*50)
    print(f"Step 3: PSF fitting ({len(images)} images)")
    print("─"*50)

    # Suppress benign FITS standard-compliance warnings from WFC3/UVIS files and
    # py1pass catalog writes (long keyword names promoted to HIERARCH cards).
    # Set as persistent global filters so they are not undone by py1pass's own
    # internal warnings.catch_warnings() contexts.
    warnings.filterwarnings('ignore', message='.*not multiple of 2880.*')
    warnings.filterwarnings('ignore', message='.*greater than 8 characters.*')

    # Build parameter dict from defaults + any overrides.
    params = dict(_HST_DEFAULTS)
    if fmin is not None:
        # fmin directly sets the pypass flux threshold, overriding both
        # mag_st_max (set to 99 so fmin_from_mag ≈ 0) and fmin_thresh.
        params['fmin_thresh'] = fmin
        params['mag_st_max']  = 99.0
    else:
        if fmin_thresh is not None:
            params['fmin_thresh'] = fmin_thresh
        if mag_st_max is not None:
            params['mag_st_max'] = mag_st_max

    for key, val in [('hmin', hmin), ('n_passes', n_passes),
                      ('n_discovery_passes', n_discovery_passes),
                      ('sat_threshold', sat_threshold),
                      ('max_iter_fit', max_iter_fit),
                      ('half_width', half_width),
                      ('conc_limit', conc_limit)]:
        if val is not None:
            params[key] = val

    # n_processes controls py1pass's internal joblib parallelism for star fitting.
    # -1 means "use all available cores" (joblib convention).
    params['n_jobs'] = n_processes

    # psf_fit_params_meta is the cache key for the PSF *fitting* step.
    # Parameters that do NOT affect photometric results are excluded so that
    # changing them doesn't force a pointless re-fit:
    #   conc_limit  — triggers reclassification only, not re-fitting
    #   n_jobs      — parallelism only; same results regardless of core count
    #   backend     — JAX vs numpy produce identical results by design
    _FIT_CACHE_EXCLUDE = {'conc_limit', 'n_jobs', 'backend'}
    _fit_cache_keys = {k: v for k, v in params.items() if k not in _FIT_CACHE_EXCLUDE}
    params_meta = {'lib_dir': str(lib_dir), **_fit_cache_keys}
    # Full params_meta written to disk also records conc_limit for reference,
    # but the cache comparison uses only _fit_cache_keys.
    params_meta_disk = {'lib_dir': str(lib_dir), **params}

    work = []
    skipped = []
    for img in images:
        catalog     = img.parent / f"{img.stem}_catalog.fits"
        params_path = img.parent / "psf_params.json"

        if not force_refit:
            ok, diffs = _params_cache_status(catalog, params_path, params_meta)
            if ok:
                skipped.append(img.name)
                continue
            if catalog.exists():
                if diffs == ["no params sidecar — cannot verify configuration match"]:
                    print(f"  {img.name}: catalog exists but no params sidecar — re-fitting")
                else:
                    print(f"  {img.name}: params changed — re-fitting:")
                    for d in diffs:
                        print(d)

        work.append(img)

    # For cached images, check whether the stored conc_limit differs from the
    # requested one.  A changed conc_limit should not re-fit but must reclassify.
    current_conc = params['conc_limit']
    needs_reclassify = []
    for img in images:
        params_path = img.parent / "psf_params.json"
        catalog     = img.parent / f"{img.stem}_catalog.fits"
        if not catalog.exists() or not params_path.exists():
            continue
        if img in work:
            continue
        try:
            saved_conc = json.loads(params_path.read_text()).get('conc_limit')
        except Exception:
            saved_conc = None
        if saved_conc != current_conc:
            needs_reclassify.append(img.parent.name)

    if skipped:
        print(f"  {len(skipped)} image(s) already fitted with matching params — skipping.")
    if needs_reclassify:
        print(f"  {len(needs_reclassify)} cached image(s) have a different conc_limit "
              f"— triggering reclassification.")
    if not work and not needs_reclassify:
        print("  All catalogs up to date.")
        return [img.parent / f"{img.stem}_catalog.fits" for img in images]

    if fmin is not None:
        _mag_str    = f"--fmin {fmin}  (overrides mag_st_max and fmin_thresh)"
        _thresh_str = ""
    else:
        _mag_str    = f"--mag_st_max {params['mag_st_max']}"
        _thresh_str = f"--fmin_thresh {params['fmin_thresh']}"
    _cmd = (
        f"pypass --image <img> --lib_dir {lib_dir}"
        f" --n_passes {params['n_passes']}"
        f" --n_discovery_passes {params['n_discovery_passes']}"
        f" {_mag_str}  {_thresh_str}"
        f" --hmin {params['hmin']}"
        f" --half_width {params['half_width']}"
        f" --sky_inner {params['sky_inner']}"
        f" --sky_outer {params['sky_outer']}"
        f" --sat_threshold {params['sat_threshold']}"
        f" --max_iter {params['max_iter_fit']}"
        f" --tol {params['tol']}"
        f" --sigma_clip_sigma {params['sigma_clip_sigma']}"
    )
    print(f"  pypass command (per image):\n    {_cmd}")

    # ── Iterative PSF refinement ──────────────────────────────────────────────
    # Per-image: load the existing cumulative δP (if any, and if clean_psf is
    # False), determine how many iterations to run, then iterate:
    #   fit → measure δP_new → cumulative += δP_new → repeat.
    # psf_delta.npy always stores the cumulative correction vs bare stdpsf.

    catalogs = []
    n_work = len(work)
    n_img_iter = n_psf_iter if n_psf_iter is not None else 1

    # ── Parallel mode: N images simultaneously, each on 1 core ───────────────
    if parallel and n_work > 0:
        import multiprocessing as _mp
        import datetime as _dt

        def _ts():
            return _dt.datetime.now().strftime('%H:%M:%S')

        n_workers = n_processes if n_processes > 0 else _mp.cpu_count()
        print(f"  Parallel PSF fitting: {n_work} image(s), "
              f"{min(n_workers, n_work)} simultaneous workers. "
              f"Verbose output → psf_fitting_log.txt per image.\n")

        _mgr = _mp.Manager()
        _queue = _mgr.Queue()

        worker_args = []
        for img in work:
            catalog = img.parent / f"{img.stem}_catalog.fits"
            worker_args.append((
                img, catalog, lib_dir, params, params_meta_disk,
                n_img_iter, clean_psf, apply_psf_delta, force_refit,
            ))

        # Print header info for all images before the pool starts.
        _hdr_info = {img: _get_image_header_info(img) for img in work}

        _pool = _mp.Pool(
            processes=min(n_workers, n_work),
            initializer=_worker_pool_init,
            initargs=(_queue,),
        )
        _async_results = {
            _pool.apply_async(_image_worker, (wargs,)): img
            for wargs, img in zip(worker_args, work)
        }
        _pool.close()

        _pending = set(_async_results)
        _done    = 0

        while _pending:
            # Drain status messages from workers.
            while True:
                try:
                    msg = _queue.get_nowait()
                    kind = msg[0]
                    img_nm = msg[1]
                    if kind == 'start':
                        # Find the matching image path for header info.
                        _img_match = next(
                            (i for i in work if i.name == img_nm), None)
                        info = _hdr_info.get(_img_match, {})
                        _id  = info.get('instdet', '?')
                        _fi  = info.get('filter',  '?')
                        _et  = info.get('exptime',  0)
                        _ft  = params.get('fmin_thresh', '')
                        print(f"[{_ts()}] Starting  {img_nm} "
                              f"({_id} {_fi}, {_et:.0f}s, fmin_thresh={_ft}e-)")
                    elif kind == 'done':
                        _, img_nm, nf, nc, ns, elapsed = msg
                        _done += 1
                        print(f"[{_ts()}] Finished  {img_nm} in {elapsed:.0f}s — "
                              f"{nf} found, {nc} converged, {ns} stars "
                              f"[{_done}/{n_work}]")
                    elif kind == 'fail':
                        _, img_nm, errmsg, elapsed = msg
                        _done += 1
                        print(f"[{_ts()}] FAILED    {img_nm} after {elapsed:.0f}s — "
                              f"{errmsg} [{_done}/{n_work}]")
                except Exception:
                    break  # queue empty

            # Check for completed async results.
            finished = [ar for ar in _pending if ar.ready()]
            for ar in finished:
                _pending.discard(ar)
                img = _async_results[ar]
                catalog = img.parent / f"{img.stem}_catalog.fits"
                try:
                    success, _, nf, nc, ns, elapsed, err = ar.get()
                    if success:
                        catalogs.append(catalog)
                    elif err:
                        # Error already printed via queue; just log here.
                        pass
                except Exception as _e:
                    print(f"[{_ts()}] FAILED    {img.name} — {_e}")

            if _pending:
                import time as _time
                _time.sleep(0.2)

        # Drain any remaining queue messages after all workers finish.
        while True:
            try:
                msg = _queue.get_nowait()
                kind = msg[0]
                img_nm = msg[1]
                if kind == 'done':
                    _, img_nm, nf, nc, ns, elapsed = msg
                    _done += 1
                    print(f"[{_ts()}] Finished  {img_nm} in {elapsed:.0f}s — "
                          f"{nf} found, {nc} converged, {ns} stars "
                          f"[{_done}/{n_work}]")
                elif kind == 'fail':
                    _, img_nm, errmsg, elapsed = msg
                    _done += 1
                    print(f"[{_ts()}] FAILED    {img_nm} after {elapsed:.0f}s — "
                          f"{errmsg} [{_done}/{n_work}]")
            except Exception:
                break

        _pool.join()
        _mgr.shutdown()
        print()

    # ── Serial mode: one image at a time (current behaviour) ─────────────────
    else:
        for img_i, img in enumerate(work, 1):
            catalog = img.parent / f"{img.stem}_catalog.fits"

            use_clean = clean_psf or (not apply_psf_delta) or force_refit
            if use_clean:
                initial_delta = None
            else:
                delta_path = img.parent / "psf_delta.npy"
                try:
                    initial_delta = np.load(str(delta_path)) if delta_path.exists() else None
                except Exception:
                    initial_delta = None

            try:
                from astropy.io import fits as _fits_hdr
                _et = float(_fits_hdr.getval(str(img), 'EXPTIME', ext=0))
                _et_str = f"  EXPTIME={_et:.1f}s"
            except Exception:
                _et_str = ""
            print(f"\n── [{img_i}/{n_work}] {field_name}  {img.name}{_et_str} ──────────────────────────────────")
            if initial_delta is not None:
                peak = float(np.abs(initial_delta).max())
                print(f"   PSF model  : CORRECTED  (stored δP, cumulative peak = {peak:+.5f})")
            elif clean_psf:
                print(f"   PSF model  : BARE stdpsf  (--clean_psf)")
            elif force_refit:
                print(f"   PSF model  : BARE stdpsf  (--force_refit_psf)")
            elif apply_psf_delta:
                print(f"   PSF model  : BARE stdpsf  (--apply_psf_delta specified but no psf_delta.npy found)")
            else:
                print(f"   PSF model  : BARE stdpsf  (default)")
            print(f"   Iterations : {n_img_iter}")

            current_delta = initial_delta
            img_succeeded = False

            for iter_i in range(n_img_iter):
                label = f"iter {iter_i + 1}/{n_img_iter}"
                if current_delta is not None:
                    peak = float(np.abs(current_delta).max())
                    print(f"   [{label}] PSF: CORRECTED (peak = {peak:+.5f})")
                else:
                    print(f"   [{label}] PSF: BARE stdpsf")
                w = (img, catalog, lib_dir, params, params_meta_disk, verbose, current_delta)
                path, n_stars, err = _fit_one_image(w)
                name = Path(path).name
                if err:
                    print(f"   ERROR [{label}] {name}: {err}")
                    break
                print(f"   [{label}] {name}: {n_stars} stars fitted")
                img_succeeded = True

                if iter_i < n_img_iter - 1:
                    import shutil as _shutil
                    _suffix = f"_iter{iter_i + 1}"
                    for _fig in ("psf_catalog_stats.png", "psf_concentration.png",
                                 "psf_diagnostics.png", "psf_residual_map.png",
                                 "psf_perturbation.png"):
                        _src = img.parent / _fig
                        if _src.exists():
                            _dst = img.parent / (_src.stem + _suffix + _src.suffix)
                            _shutil.copy2(str(_src), str(_dst))

                if iter_i < n_img_iter - 1:
                    delta_path = img.parent / "psf_delta.npy"
                    if delta_path.exists():
                        try:
                            current_delta = np.load(str(delta_path))
                            peak = float(np.abs(current_delta).max())
                            print(f"   Loaded updated δP for next iter (cumulative peak = {peak:+.5f})")
                        except Exception as _e:
                            print(f"   WARNING: {name}: could not load δP for next iter: {_e}")
                            break
                    else:
                        print(f"   WARNING: {name}: no psf_delta.npy after {label} — stopping")
                        break

            if img_succeeded:
                catalogs.append(catalog)

    # Include previously-skipped catalogs in the return list.
    for img in images:
        cat = img.parent / f"{img.stem}_catalog.fits"
        if cat not in catalogs and cat.exists():
            catalogs.append(cat)

    # Reclassify any cached images whose stored conc_limit is stale.
    if needs_reclassify:
        reclassify_psf_catalogs(
            output_dir=output_dir,
            field_name=field_name,
            telescope=telescope,
            im_type=im_type,
            conc_limit=current_conc,
            restrict_to_obsids=needs_reclassify,
            lib_dir=lib_dir,
        )

    print(f"  PSF fitting complete: {len(catalogs)}/{len(images)} available.")
    return catalogs
