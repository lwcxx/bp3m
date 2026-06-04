#!/usr/bin/env python3
"""
bp3m — end-to-end pipeline combining Gaia and HST data to
measure improved proper motions.

Usage examples
--------------
# By target name (Simbad-resolved):
  bp3m --name "Sculptor dSph" --lib_dir ./lib

# By coordinates:
  bp3m --ra 15.039 --dec -33.709 --search_radius 0.3 --lib_dir ./lib

# Resume after cross-match (e.g. to re-run alignment with different params):
  bp3m --name "Sculptor dSph" --lib_dir ./lib \\
      --skip_download --skip_psf --skip_crossmatch \\
      --n_bp3m_iter 30 --bp3m_clip_sigma 3.5

Pipeline steps
--------------
  1  download_gaia    Download Gaia DR3 catalogue
  2  download_hst     Search MAST and download HST FLC images
  3  psf_fitting      PSF-fit each FLC image (py1pass)
  4  cross_match      Cross-match HST ↔ Gaia (fast_cross_match)
  5  alignment        Bayesian alignment + proper motions (BP3M)

Extension note
--------------
JWST support is planned. Pass --telescope JWST once py1pass and
fast_cross_match have been updated for JWST data.
"""

import argparse
import re
import sys
import os
from pathlib import Path
from multiprocessing import cpu_count

import numpy as np


def _config_lib_dir() -> str | None:
    """Read lib_dir from config.toml if it exists (written by bp3m-setup).

    Config location: $BP3M_HOME/config.toml, or ~/.bp3m/config.toml by default.
    """
    import os
    bp3m_home = Path(os.environ["BP3M_HOME"]) if "BP3M_HOME" in os.environ else Path.home() / ".bp3m"
    config = bp3m_home / "config.toml"
    if not config.exists():
        return None
    try:
        text = config.read_text()
        m = re.search(r'^lib_dir\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
        return m.group(1) if m else None
    except Exception:
        return None


def _parse_args():
    p = argparse.ArgumentParser(
        prog='bp3m',
        description='Measure proper motions by combining Gaia + HST data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Target ────────────────────────────────────────────────────────────────
    tgt = p.add_argument_group('Target (provide name OR ra+dec)')
    tgt.add_argument('--name', type=str, default=None,
                     help='Target name (resolved via Simbad). '
                          'Used as the field directory name.')
    tgt.add_argument('--ra',  type=float, default=None,
                     help='Centre R.A. (degrees)')
    tgt.add_argument('--dec', type=float, default=None,
                     help='Centre Dec. (degrees)')
    tgt.add_argument('--search_radius', type=float, default=None,
                     help='Circular search radius (degrees). '
                          'Converted to an equal-area box.')
    tgt.add_argument('--search_width',  type=float, default=None,
                     help='Search box width  (degrees, overrides --search_radius)')
    tgt.add_argument('--search_height', type=float, default=None,
                     help='Search box height (degrees, overrides --search_radius)')

    # ── Gaia ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group('Gaia options')
    g.add_argument('--min_gmag', type=float, default=16.0,
                   help='Brightest G magnitude (default 16.0)')
    g.add_argument('--max_gmag', type=float, default=None,
                   help='Faintest G magnitude (default: no limit)')
    g.add_argument('--source_table', type=str, default='gaiadr3.gaia_source',
                   help='Gaia TAP source table (default gaiadr3.gaia_source)')
    g.add_argument('--sigma_flux_excess', type=float, default=3.0,
                   help='Sigma for flux-excess-factor clipping (default 3.0)')
    g.add_argument('--only_5p', action='store_true',
                   help='Restrict to 5-parameter Gaia solutions only')

    # ── HST ───────────────────────────────────────────────────────────────────
    h = p.add_argument_group('HST / telescope options')
    h.add_argument('--telescope', type=str, default='HST',
                   help='Telescope (default HST; JWST planned)')
    h.add_argument('--hst_filters', type=str, nargs='+', default=None,
                   help='Required filters, e.g. F814W F606W F850LP '
                        '(default: all filters with PSF+GDC in lib_dir). '
                        'Use MAST filter names (e.g. F850LP not F850L).')
    h.add_argument('--hst_im_type', type=str, default='_flc',
                   help='Image type: _flc (default) or _flt')
    h.add_argument('--hst_exptime_min', type=float, default=2.0,
                   help='Minimum average exposure time per image set (s)')
    h.add_argument('--hst_exptime_max', type=float, default=np.inf,
                   help='Maximum average exposure time per image set (s)')
    h.add_argument('--time_baseline', type=float, default=None,
                   help='Minimum HST–Gaia time baseline in days (default: no limit)')
    h.add_argument('--obs_date_min', type=str, default=None,
                   help='Earliest HST observation date to include (ISO, e.g. 2005-01-01). '
                        'Default: no lower limit.')
    h.add_argument('--obs_date_max', type=str, default=None,
                   help='Latest HST observation date to include (ISO, e.g. 2020-12-31). '
                        'Default: no upper limit.')
    h.add_argument('--instruments', type=str, nargs='+', default=None,
                   help='HST instrument/detector combinations to include '
                        '(e.g. ACS/WFC WFC3/UVIS). Default: all supported instruments '
                        'that have PSF and GDC files in lib_dir.')
    h.add_argument('--field_ids', type=str, nargs='+', default=None,
                   help='Field IDs to download. Integers from the table '
                        '(space-separated), "y"/"all" to download everything, '
                        'or "n"/"0" to skip download. Default: interactive prompt.')

    # ── PSF fitting ───────────────────────────────────────────────────────────
    psf = p.add_argument_group('PSF fitting (py1pass)')
    _lib_default = _config_lib_dir() or str(Path.home() / 'GaiaHub-master' / 'lib')
    psf.add_argument('--lib_dir', type=str,
                     default=_lib_default,
                     help='Library directory containing STDPSFs/ and STDGDCs/ '
                          'subdirectories. Defaults to the path set by bp3m-setup '
                          f'(currently: {_lib_default})')
    psf.add_argument('--fmin_thresh', type=float, default=None,
                     help='Hard lower bound on the minimum source flux in electrons '
                          '(default 40). Acts as a floor: fmin will never go below '
                          'this value even when mag_st_max would imply a lower threshold.')
    psf.add_argument('--mag_st_max', type=float, default=None,
                     help='Faint ST-magnitude limit used to set the detection threshold '
                          '(default 28). Converted to a flux threshold per image using '
                          'PHOTFLAM and EXPTIME; floored at fmin_thresh.')
    psf.add_argument('--hmin', type=int, default=None,
                     help='NMS radius in pixels (default 4)')
    psf.add_argument('--n_passes', type=int, default=None,
                     help='Total PSF fit passes (default 2)')
    psf.add_argument('--n_discovery_passes', type=int, default=None,
                     help='How many of those passes include new-source detection '
                          '(default: n_passes-1, i.e. last pass is refit-only)')
    psf.add_argument('--psf_max_iter', type=int, default=None,
                     help='Max iterations for PSF fit convergence (default 100)')
    psf.add_argument('--conc_limit', type=float, default=None,
                     help='Concentration lower bound for star/non-star classification '
                          '(upper bound = 1/conc_limit, default 0.9)')
    psf.add_argument('--sat_threshold', type=float, default=None,
                     help='Saturation DN threshold (default 60000)')

    # ── Cross-matching ────────────────────────────────────────────────────────
    xm = p.add_argument_group('Cross-matching (fast_cross_match)')
    xm.add_argument('--cross_match_pix_floor', type=float, default=0.05,
                    help='HST positional uncertainty floor in pixels applied during cross-matching (default 0.05)')
    xm.add_argument('--min_matches', type=int, default=3,
                    help='Minimum seed matches for 4P discovery (default 3)')
    xm.add_argument('--max_mag_diff', type=float, default=3.0,
                    help='Maximum Gaia–HST magnitude difference (default 3.0)')
    xm.add_argument('--scale_sweep', action='store_true',
                    help='Enable pixel-scale sweep during 4P discovery (slower)')

    # ── Alignment (BP3M) ──────────────────────────────────────────────────────
    bp = p.add_argument_group('Bayesian alignment (BP3M)')
    bp.add_argument('--n_bp3m_iter', type=int, default=20,
                    help='Maximum BP3M outer iterations (default 20)')
    bp.add_argument('--n_samples', type=int, default=1000,
                    help='Posterior samples for uncertainty estimation (default 1000)')
    bp.add_argument('--bp3m_clip_sigma', type=float, default=4.5,
                    help='MAD sigma threshold for outlier rejection (default 4.5; '
                         '0 = disabled)')
    bp.add_argument('--poly_order', type=int, default=1,
                    help='Polynomial order for image transformation (default 1=linear)')
    bp.add_argument('--no_split_ccd', action='store_true',
                    help='Disable per-CCD splitting for ACS/WFC images (default: split enabled)')
    bp.add_argument('--min_stars_split_ccd', type=int, default=20,
                    help='Minimum stars required on each CCD half to allow splitting. '
                         'Images where either half has fewer than N stars are kept unsplit. '
                         'Only applies when --no_split_ccd is not set. (default: 20)')
    bp.add_argument('--no_inflate_hst_errors', action='store_true',
                    help='Disable per-image HST error inflation (default: inflation enabled)')
    bp.add_argument('--bp3m_pos_err_floor', type=float, default=5e-3,
                    help='Minimum HST position uncertainty floor in pixels before BP3M '
                         '(default 0.001 px; prevents numerically unstable residuals for '
                         'very bright stars with sub-pixel PSF uncertainties)')
    bp.add_argument('--no_influence_clip', action='store_true',
                    help='Disable test-4 Cook\'s D influence clipping (default: enabled; '
                         'targets moderate-outlier high-leverage detections missed by the sigma threshold)')
    bp.add_argument('--influence_d_thresh', type=float, default=1.0,
                    help="Cook's D threshold for influence clipping (default 1.0)")
    bp.add_argument('--influence_sigma_min', type=float, default=2.0,
                    help='Minimum sigma_resid for influence clipping (default 2.0; '
                         'prevents removing well-fit high-leverage anchors)')
    bp.add_argument('--two_tier', action='store_true',
                    help='Enable two-tier alignment/astrometry system: stars that fail '
                         'ok_gaia can still constrain their own astrometry at 3× the '
                         'alignment threshold (use_for_astrom independent of use_for_fit)')
    bp.add_argument('--sparse', action='store_true',
                    help='Use sparse solver (faster for large mosaics)')
    bp.add_argument('--bp3m_images', type=str, nargs='+', default=None,
                    help='Restrict BP3M to these image names')
    bp.add_argument('--bp3m_all_images', action='store_true',
                    help='Use all available images for BP3M, ignoring the '
                         'field_id selection from the HST download step')
    bp.add_argument('--bp3m_remove_images', type=str, nargs='+', default=None,
                    help='Exclude these images from BP3M')
    bp.add_argument('--restrict_filters', type=str, nargs='+', default=None,
                    help='Keep only images with these filters for BP3M')
    bp.add_argument('--restrict_instdet', type=str, nargs='+', default=None,
                    help='Keep only images from these instrument+detector combinations '
                         'for BP3M (e.g. ACSWFC WFC3UVIS)')

    # ── Synthetic tests ───────────────────────────────────────────────────────
    syn = p.add_argument_group('Synthetic tests (requires completed cross-match, Step 4)')
    syn.add_argument('--test_synthetic', action='store_true',
                     help='Run synthetic data test after cross-match. '
                          'Generates synthetic observations from real data, runs BP3M, '
                          'and compares recovered parameters to ground truth.')
    syn.add_argument('--synthetic_draw_from_prior', action='store_true',
                     help='Draw true stellar parameters from Gaia prior N(v_gaia, C_gaia) '
                          'instead of using MAP values as truth (default: MAP values).')
    syn.add_argument('--synthetic_zero_parallax', action='store_true',
                     help='Set true parallax = 0 for all stars.')
    syn.add_argument('--synthetic_true_gaia', action='store_true',
                     help='Feed true stellar parameters directly as the Gaia prior mean '
                          '(zero Gaia measurement noise). Useful for isolating HST noise.')
    syn.add_argument('--synthetic_jitter_sigma', type=float, default=0.0,
                     help='Std dev of Gaussian perturbation added to true transformation '
                          'parameters (default 0 = no perturbation).')
    syn.add_argument('--synthetic_seed', type=int, default=42,
                     help='Random seed for synthetic data generation (default 42).')
    syn.add_argument('--synthetic_only_5p', action='store_true',
                     help='Exclude 2-param Gaia stars (no measured PM/parallax) from '
                          'the synthetic test. Useful for isolating whether 2-param '
                          'stars affect image parameter estimation.')
    syn.add_argument('--synthetic_all_5p_gaia', action='store_true',
                     help='Give 2-param Gaia stars synthetic 5-param Gaia measurements '
                          '(PM+parallax drawn with median errors from real 5-param stars). '
                          'Tests whether BP3M handles all-5p Gaia data correctly; the '
                          'true PM is still drawn from N(0,10²).')
    syn.add_argument('--synthetic_true_pm', type=float, nargs=2,
                     metavar=('PMRA', 'PMDEC'), default=None,
                     help='Override ALL stars true PM: draw from N((PMRA,PMDEC), width²). '
                          'Generates self-consistent catalog = truth + Gaia noise. '
                          'Example: --synthetic_true_pm 5.0 -5.0')
    syn.add_argument('--synthetic_true_pm_width', type=float, default=0.1,
                     help='1σ width of the true PM distribution (mas/yr, default 0.1).')
    syn.add_argument('--synthetic_true_parallax', type=float, default=None,
                     help='Override ALL stars true parallax: draw from N(VAL, width²). '
                          'Use a positive value for physically meaningful parallaxes. '
                          'Example: --synthetic_true_parallax 5.0')
    syn.add_argument('--synthetic_true_parallax_width', type=float, default=0.1,
                     help='1σ width of the true parallax distribution (mas, default 0.1).')

    # ── Pipeline control ──────────────────────────────────────────────────────
    ctl = p.add_argument_group('Pipeline control')
    ctl.add_argument('--output_dir', type=str, default='.',
                     help='Root output directory (default: current directory)')
    ctl.add_argument('--n_processes', type=int, default=-1,
                     help='Number of cores to use (-1 = all available, default)')
    ctl.add_argument('--skip_download', action='store_true',
                     help='Skip Gaia and HST downloads (use existing files)')
    ctl.add_argument('--force_redownload_gaia', action='store_true',
                     help='Re-query Gaia archive even if local CSV already exists')
    ctl.add_argument('--force_redownload_hst', action='store_true',
                     help='Re-search MAST and re-download HST files even if cached')
    ctl.add_argument('--force_refit_psf', action='store_true',
                     help='Re-run PSF fitting even if catalogs already exist, starting '
                          'from the bare stdpsf (ignores any stored psf_delta.npy). '
                          'Pass --n_psf_iter 2 alongside this to explicitly apply a '
                          'stored delta in the second iteration.')
    ctl.add_argument('--clean_psf', action='store_true',
                     help='Start PSF fitting from the bare stdpsf, ignoring any stored '
                          'psf_delta.npy.')
    ctl.add_argument('--n_psf_iter', type=int, default=None,
                     help='Number of iterative PSF fitting passes (default: 1). Pass 2 to '
                          'enable the iterative PSF correction (fit → measure δP → re-fit '
                          'with corrected PSF). WARNING: applying δP in a second pass can '
                          'degrade the 2-D pixel-phase distribution for sparse fields; '
                          'only recommended when many bright stars (≳1000) are available.')
    ctl.add_argument('--reclassify_stars', action='store_true',
                     help='Re-run star classification on existing PSF catalogs using the '
                          'current --conc_limit, without re-fitting PSFs. Regenerates '
                          'concentration plots and invalidates the cross-match cache.')
    ctl.add_argument('--remeasure_psf_perturbation', action='store_true',
                     help='Re-measure PSF perturbation on existing catalogs without re-fitting. '
                          'Reconstructs residual images from catalog star models and regenerates '
                          'psf_delta.npy and psf_perturbation.png for each image.')
    ctl.add_argument('--force_rematch', action='store_true',
                     help='Re-run cross-matching even if matched_gaia.csv already exists')
    ctl.add_argument('--skip_psf', action='store_true',
                     help='Skip PSF fitting (use existing catalogs)')
    ctl.add_argument('--skip_crossmatch', action='store_true',
                     help='Skip cross-matching (use existing matched_gaia.csv)')
    ctl.add_argument('--skip_alignment', action='store_true',
                     help='Skip BP3M alignment (stop after cross-match)')
    ctl.add_argument('--no_plots', action='store_true',
                     help='Suppress all diagnostic plot generation')
    ctl.add_argument('--plot_residuals', action='store_true',
                     help='Generate per-image HST XY residual maps (slow for large fields; off by default)')
    ctl.add_argument('--plot_influence', action='store_true',
                     help='Generate Cook\'s D influence diagnostic plots (slow; off by default)')
    ctl.add_argument('--quiet', action='store_true',
                     help='Non-interactive mode; use defaults without prompts')
    ctl.add_argument('--checkpoint_dir', type=str, default=None,
                     help='Save BP3M checkpoint to this directory for later re-use')

    return p.parse_args()


_FIELD_IDS_ALL = 'all'   # sentinel: download all without prompting


def _parse_field_ids(raw: list[str] | None):
    """
    Convert raw --field_ids strings to what download_hst_images expects.

    None (not provided)  → None          (show interactive prompt)
    'y' / 'all' / 'yes'  → _FIELD_IDS_ALL  (download all without prompting)
    'n' / 'no'           → [0]           (skip download)
    integers             → list[int]
    """
    if raw is None:
        return None
    joined = ' '.join(raw).strip().lower()
    if joined in ('y', 'yes', 'all'):
        return _FIELD_IDS_ALL
    if joined in ('n', 'no', '0'):
        return [0]
    try:
        return [int(x) for x in raw]
    except ValueError:
        print(f"  WARNING: could not parse --field_ids {raw!r} — downloading all.")
        return _FIELD_IDS_ALL


def _resolve_target(args):
    """Resolve target name to RA/Dec and compute search box. Mutates args."""
    from bp3m.pipeline.download_gaia import resolve_target

    if args.ra is None or args.dec is None:
        if args.name is None:
            print("ERROR: provide --name or both --ra and --dec.", file=sys.stderr)
            sys.exit(1)
        print(f"Resolving '{args.name}' via Simbad...")
        try:
            ra, dec, auto_r = resolve_target(args.name)
            args.ra, args.dec = ra, dec
            if args.search_radius is None and auto_r is not None:
                args.search_radius = auto_r
                print(f"  Auto search radius from Simbad: {auto_r:.3f} deg")
        except Exception as exc:
            print(f"  Simbad lookup failed: {exc}")
            if not args.quiet:
                args.ra  = float(input('  Enter R.A. (degrees): '))
                args.dec = float(input('  Enter Dec. (degrees): '))
            else:
                print("ERROR: Could not resolve target coordinates.", file=sys.stderr)
                sys.exit(1)

    if args.search_radius is None:
        if not args.quiet:
            args.search_radius = float(
                input('Search radius not set. Enter value in degrees [0.25]: ')
                or 0.25)
        else:
            args.search_radius = 0.25

    if args.search_width is None:
        cos_dec = abs(np.cos(np.deg2rad(args.dec)))
        args.search_width = 2.0 * args.search_radius / max(cos_dec, 0.01)
    if args.search_height is None:
        args.search_height = 2.0 * args.search_radius

    if args.name is None:
        args.name = f"ra_{args.ra:.3f}_dec_{args.dec:.3f}"
    args.name = args.name.replace(' ', '_')

    print(f"\n  Field:  {args.name}")
    print(f"  Centre: ({args.ra:.5f}, {args.dec:.5f}) deg")
    print(f"  Box:    {args.search_width:.4f} × {args.search_height:.4f} deg")


def main():
    args = _parse_args()

    # Wire --n_processes to BLAS and JAX thread limits so all parallelism
    # is bounded by the same value. Env vars cover JAX (imported lazily)
    # and subprocesses; threadpoolctl covers already-loaded BLAS pools.
    if args.n_processes != -1:
        _n = str(args.n_processes)
        os.environ['OMP_NUM_THREADS']      = _n
        os.environ['OPENBLAS_NUM_THREADS'] = _n
        os.environ['MKL_NUM_THREADS']      = _n
        try:
            import threadpoolctl as _tpc
            _tpc.threadpool_limits(limits=args.n_processes)
        except ImportError:
            pass

    print("=" * 55)
    print("GaiaHub Improved")
    print("=" * 55)

    _resolve_target(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    field = args.name

    # ── Step 1: Download Gaia ─────────────────────────────────────────────────
    gaia_df = None
    if not args.skip_download:
        from bp3m.pipeline.download_gaia import download_gaia
        gaia_df = download_gaia(
            ra=args.ra, dec=args.dec,
            search_width=args.search_width, search_height=args.search_height,
            output_dir=output_dir, field_name=field,
            min_gmag=args.min_gmag, max_gmag=args.max_gmag,
            source_table=args.source_table,
            sigma_flux_excess=args.sigma_flux_excess,
            only_5p=args.only_5p,
            n_processes=args.n_processes,
            force_redownload=args.force_redownload_gaia,
            quiet=args.quiet,
        )

    # ── Step 2: Download HST ─────────────────────────────────────────────────
    if not args.skip_download:
        from bp3m.pipeline.download_hst import download_hst_images
        # Load Gaia catalog for footprint star counts if not already in memory
        if gaia_df is None:
            from bp3m.pipeline.explore_utils import load_gaia_catalog
            from bp3m.pipeline.download_gaia import _cache_stem
            _gaia_dir = output_dir / field / 'Gaia'
            _stem = _cache_stem(field, args.ra, args.dec,
                                args.search_width, args.search_height,
                                args.min_gmag, args.max_gmag)
            _gaia_csv = _gaia_dir / f"{_stem}.csv"
            if not _gaia_csv.exists():
                # Fall back to any CSV in the Gaia directory
                _candidates = list(_gaia_dir.glob("*.csv"))
                _gaia_csv = _candidates[0] if _candidates else _gaia_csv
            if _gaia_csv.exists():
                gaia_df = load_gaia_catalog(_gaia_csv)
        download_hst_images(
            ra=args.ra, dec=args.dec,
            search_width=args.search_width, search_height=args.search_height,
            output_dir=output_dir, field_name=field,
            hst_filters=args.hst_filters,
            t_exptime_min=args.hst_exptime_min,
            t_exptime_max=args.hst_exptime_max,
            time_baseline_days=args.time_baseline,
            obs_date_min=args.obs_date_min,
            obs_date_max=args.obs_date_max,
            im_type=args.hst_im_type,
            telescope=args.telescope,
            instruments=args.instruments,
            lib_dir=Path(args.lib_dir),
            gaia_df=gaia_df,
            field_ids=_parse_field_ids(args.field_ids),
            quiet=args.quiet,
            force_redownload=args.force_redownload_hst,
        )

    # Read manifest of selected obsids written by step 2 (persists across runs)
    import json as _json
    _hst_dir = output_dir / field / args.telescope.upper()
    _manifest = _hst_dir / f"{field}_selected_obsids.json"
    _failed_manifest = _hst_dir / f"{field}_failed_obsids.json"
    _selected_obsids: list[str] | None = None
    if _manifest.exists():
        try:
            _selected_obsids = _json.loads(_manifest.read_text())
        except Exception:
            pass
    # If the failed manifest doesn't exist yet (e.g. step 2 was skipped on this
    # run and the failed-obs check has never been written), scan on-disk FLC
    # files now so downstream steps never accidentally process bad images.
    if not _failed_manifest.exists() and _manifest.exists() and _selected_obsids:
        from bp3m.pipeline.download_hst import _check_exptime as _cet
        _mast_root = _hst_dir / "mastDownload" / args.telescope.upper()
        _scanned_failed: dict[str, str] = {}
        for _oid in list(_selected_obsids):
            _flc = _mast_root / _oid / f"{_oid}_{args.hst_im_type.lstrip('_')}.fits"
            if _flc.exists():
                _reason = _cet(_flc)
                if _reason:
                    _scanned_failed[_oid] = _reason
        if _scanned_failed:
            # Remove from selected list and write both manifests
            _selected_obsids = [o for o in _selected_obsids if o not in _scanned_failed]
            _manifest.write_text(_json.dumps(_selected_obsids, indent=2))
            _failed_manifest.write_text(_json.dumps(_scanned_failed, indent=2))

    if _failed_manifest.exists():
        try:
            _failed = _json.loads(_failed_manifest.read_text())
            if _failed:
                print(f"\nNOTE: {len(_failed)} image(s) are failed observations and will be "
                      f"skipped in all pipeline steps:")
                for _oid, _reason in sorted(_failed.items()):
                    print(f"  {_oid}: {_reason}")
        except Exception:
            pass

    # ── Check that we have images before continuing ───────────────────────────
    if _selected_obsids is not None and len(_selected_obsids) == 0:
        print("\nNo HST images available to process. Check your search "
              "parameters (filters, instruments, search radius, dates).")
        return

    # ── Resolve active image set for steps 3 onwards ─────────────────────────
    # --bp3m_images restricts ALL downstream steps (PSF, cross-match, BP3M),
    # not just the alignment step.  Resolve it now so every step uses the same
    # filtered list.  _selected_obsids is the full set from the download
    # manifest; _bp3m_images is the (possibly narrower) working set.
    _bp3m_images = args.bp3m_images
    if _bp3m_images is None and not args.bp3m_all_images and _selected_obsids is not None:
        _bp3m_images = _selected_obsids
    # _restrict is what we pass as restrict_to_obsids to every step from 3 on.
    _restrict = _bp3m_images  # may be None (→ process all on-disk images)

    # ── Filter by --restrict_instdet for steps 3 onwards ─────────────────────
    # BP3M does its own instdet filtering from image metadata, but steps 3/4
    # only know about obsids — so narrow _restrict here using the cached MAST
    # obs table (data_products CSV joined with obs CSV).  Falls back to reading
    # FITS headers if the CSVs aren't present.
    if args.restrict_instdet and _restrict is not None:
        import pandas as _pd
        _keep_id = {s.upper().replace('/', '') for s in args.restrict_instdet}

        def _instdet_key(name: str) -> str:
            return name.upper().replace('/', '')

        _obsid_to_instdet: dict[str, str] = {}
        _dp_csv = _hst_dir / f"{field}_data_products.csv"
        _obs_csv = _hst_dir / f"{field}_obs.csv"
        if _dp_csv.exists() and _obs_csv.exists():
            try:
                _dp  = _pd.read_csv(str(_dp_csv))
                _obs = _pd.read_csv(str(_obs_csv))
                _flc = _dp[_dp['productFilename'].str.endswith('_flc.fits', na=False)
                           | _dp['productFilename'].str.endswith('_flt.fits', na=False)]
                _merged = _flc.merge(
                    _obs[['obsid', 'instrument_name']],
                    left_on='parent_obsid', right_on='obsid', how='left')
                for _, _row in _merged.iterrows():
                    _oid = str(_row['obs_id'])
                    _inst = str(_row.get('instrument_name', ''))
                    if _inst and _inst != 'nan':
                        _obsid_to_instdet[_oid] = _instdet_key(_inst)
            except Exception as _e:
                print(f"  WARNING: could not read MAST CSVs for instdet filter: {_e}")
        if not _obsid_to_instdet:
            # Fallback: read FITS headers.
            _mast_root = _hst_dir / "mastDownload" / args.telescope.upper()
            _im_suffix = args.hst_im_type.lstrip('_') + '.fits'
            for _oid in _restrict:
                _flc_path = _mast_root / _oid / f"{_oid}_{_im_suffix}"
                if _flc_path.exists():
                    try:
                        from astropy.io import fits as _fits_hdr
                        with _fits_hdr.open(str(_flc_path)) as _h:
                            _inst = _h[0].header.get('INSTRUME', '').strip()
                            _det  = _h[0].header.get('DETECTOR', '').strip()
                        _obsid_to_instdet[_oid] = (_inst + _det).upper()
                    except Exception:
                        pass

        if _obsid_to_instdet:
            _before = len(_restrict)
            _restrict = [o for o in _restrict
                         if _obsid_to_instdet.get(o, '') in _keep_id]
            _bp3m_images = _restrict
            print(f"  --restrict_instdet {args.restrict_instdet}: "
                  f"{_before} → {len(_restrict)} images")
        else:
            print("  WARNING: --restrict_instdet specified but could not determine "
                  "instrument for any obsid — skipping filter for steps 3/4.")

    # ── Step 3: PSF fitting ───────────────────────────────────────────────────
    if not args.skip_psf:
        from bp3m.pipeline.psf_fitting import run_psf_fitting
        run_psf_fitting(
            output_dir=output_dir, field_name=field,
            lib_dir=Path(args.lib_dir),
            telescope=args.telescope,
            im_type=args.hst_im_type,
            n_processes=args.n_processes,
            verbose=not args.quiet,
            force_refit=args.force_refit_psf,
            clean_psf=args.clean_psf,
            n_psf_iter=args.n_psf_iter,
            fmin_thresh=args.fmin_thresh, mag_st_max=args.mag_st_max, hmin=args.hmin,
            n_passes=args.n_passes, n_discovery_passes=args.n_discovery_passes,
            max_iter_fit=args.psf_max_iter,
            sat_threshold=args.sat_threshold, conc_limit=args.conc_limit,
            restrict_to_obsids=_restrict,
        )

    if args.reclassify_stars:
        from bp3m.pipeline.psf_fitting import reclassify_psf_catalogs
        reclassify_psf_catalogs(
            output_dir=output_dir, field_name=field,
            telescope=args.telescope,
            im_type=args.hst_im_type,
            conc_limit=args.conc_limit,
            restrict_to_obsids=_restrict,
            lib_dir=Path(args.lib_dir) if args.lib_dir else None,
        )

    if args.remeasure_psf_perturbation:
        from bp3m.pipeline.psf_fitting import remeasure_psf_perturbation
        remeasure_psf_perturbation(
            output_dir=output_dir, field_name=field,
            lib_dir=Path(args.lib_dir),
            telescope=args.telescope,
            im_type=args.hst_im_type,
            restrict_to_obsids=_restrict,
            verbose=not args.quiet,
        )

    # ── Step 4: Cross-matching ────────────────────────────────────────────────
    if not args.skip_crossmatch:
        from bp3m.pipeline.cross_match import run_cross_match
        run_cross_match(
            output_dir=output_dir, field_name=field,
            telescope=args.telescope,
            im_type=args.hst_im_type,
            n_processes=args.n_processes,
            hst_pix_floor=args.cross_match_pix_floor,
            min_matches=args.min_matches,
            max_mag_diff=args.max_mag_diff,
            scale_sweep=args.scale_sweep,
            force_rematch=args.force_rematch,
            restrict_to_obsids=_restrict,
        )

    # ── Step 5a: Synthetic data generation (optional) ─────────────────────────

    if args.test_synthetic:
        from bp3m.pipeline.synthetic import generate_synthetic_data, compare_synthetic_results

        # Build a unique subdirectory name so different configurations don't
        # overwrite each other (e.g. 'synthetic_only5p_seed43').
        _syn_parts = ["synthetic"]
        if args.synthetic_only_5p:
            _syn_parts.append("only5p")
        if getattr(args, 'synthetic_all_5p_gaia', False):
            _syn_parts.append("all5pgaia")
        _true_pm = getattr(args, 'synthetic_true_pm', None)
        _true_plx = getattr(args, 'synthetic_true_parallax', None)
        if _true_pm is not None:
            _syn_parts.append(f"pm{_true_pm[0]:g}_{_true_pm[1]:g}"
                              .replace('-', 'm').replace('.', 'p'))
        if _true_plx is not None:
            _syn_parts.append(f"plx{_true_plx:g}".replace('-', 'm').replace('.', 'p'))
        if _bp3m_images is not None:
            _syn_parts.append(f"n{len(_bp3m_images)}")
        if args.synthetic_seed != 42:
            _syn_parts.append(f"seed{args.synthetic_seed}")
        syn_name = "_".join(_syn_parts)

        print("\n" + "=" * 55)
        print(f"Synthetic test — generating observations → {syn_name}/")
        print("=" * 55)
        generate_synthetic_data(
            output_dir=output_dir,
            field_name=field,
            telescope=args.telescope,
            im_type=args.hst_im_type,
            draw_from_prior=args.synthetic_draw_from_prior,
            zero_parallax=args.synthetic_zero_parallax,
            true_gaia=args.synthetic_true_gaia,
            jitter_sigma=args.synthetic_jitter_sigma,
            seed=args.synthetic_seed,
            only_5p=args.synthetic_only_5p,
            all_5p_gaia=getattr(args, 'synthetic_all_5p_gaia', False),
            true_pm_center=(tuple(_true_pm) if _true_pm is not None else None),
            true_pm_width=args.synthetic_true_pm_width,
            true_parallax_center=_true_plx,
            true_parallax_width=args.synthetic_true_parallax_width,
            images=_bp3m_images,
            syn_name=syn_name,
        )

    # ── Step 5: Bayesian alignment ────────────────────────────────────────────
    if not args.skip_alignment:
        from bp3m.pipeline.run_alignment import run_alignment

        if args.test_synthetic:
            # Run BP3M on the synthetic directory tree.
            # The synthetic data lives at {output_dir}/{field}/{syn_name}/,
            # so we pass output_dir={output_dir}/{field} and field_name=syn_name.
            print("\n" + "=" * 55)
            print(f"Synthetic test — running BP3M on {syn_name}/")
            print("=" * 55)
            run_alignment(
                output_dir=output_dir / field,
                field_name=syn_name,
                n_iter=args.n_bp3m_iter,
                n_samples=args.n_samples,
                clip_sigma=args.bp3m_clip_sigma,
                poly_order=args.poly_order,
                split_ccd=not args.no_split_ccd,
                min_stars_split_ccd=args.min_stars_split_ccd,
                inflate_hst_errors=not args.no_inflate_hst_errors,
                use_sparse=args.sparse,
                no_plots=args.no_plots,
                images=_bp3m_images,
                remove_images=args.bp3m_remove_images,
                restrict_filters=args.restrict_filters,
                restrict_instdet=args.restrict_instdet,
                checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
                use_influence_clip=not args.no_influence_clip,
                influence_d_thresh=args.influence_d_thresh,
                influence_sigma_min=args.influence_sigma_min,
                use_two_tier=args.two_tier,
                pos_err_floor=args.bp3m_pos_err_floor,
                plot_residuals=args.plot_residuals,
                plot_influence=args.plot_influence,
            )
            # ── Step 5b: Compare synthetic results to truth ────────────────────
            print("\n" + "=" * 55)
            print("Synthetic test — comparing results to truth")
            print("=" * 55)
            from bp3m.pipeline.synthetic import compare_synthetic_results, run_conditional_solve
            compare_synthetic_results(
                output_dir=output_dir,
                field_name=field,
                syn_name=syn_name,
            )
            # ── Step 5c: Conditional solve with r fixed at r_true ─────────────
            print("\n" + "=" * 55)
            print("Synthetic test — conditional solve (r = r_true)")
            print("=" * 55)
            run_conditional_solve(
                output_dir=output_dir,
                field_name=field,
                syn_name=syn_name,
                split_ccd=not args.no_split_ccd,
                min_stars_split_ccd=args.min_stars_split_ccd,
                poly_order=args.poly_order,
                inflate_hst_errors=not args.no_inflate_hst_errors,
            )
        else:
            run_alignment(
                output_dir=output_dir, field_name=field,
                n_iter=args.n_bp3m_iter,
                n_samples=args.n_samples,
                clip_sigma=args.bp3m_clip_sigma,
                poly_order=args.poly_order,
                split_ccd=not args.no_split_ccd,
                min_stars_split_ccd=args.min_stars_split_ccd,
                inflate_hst_errors=not args.no_inflate_hst_errors,
                use_sparse=args.sparse,
                no_plots=args.no_plots,
                images=_bp3m_images,
                remove_images=args.bp3m_remove_images,
                restrict_filters=args.restrict_filters,
                restrict_instdet=args.restrict_instdet,
                checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
                use_influence_clip=not args.no_influence_clip,
                influence_d_thresh=args.influence_d_thresh,
                influence_sigma_min=args.influence_sigma_min,
                use_two_tier=args.two_tier,
                pos_err_floor=args.bp3m_pos_err_floor,
                plot_residuals=args.plot_residuals,
                plot_influence=args.plot_influence,
            )

    # Save the command only on successful completion so interrupted runs
    # do not overwrite the record of the last successful invocation.
    import shlex as _shlex
    from datetime import datetime as _datetime
    _cmd_file = output_dir / field / 'bp3m_command.txt'
    _cmd_file.parent.mkdir(parents=True, exist_ok=True)
    _cmd_file.write_text(
        f"# {_datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        + ' '.join(_shlex.quote(a) for a in sys.argv) + '\n'
    )

    print("\n" + "=" * 55)
    print("Pipeline complete.")
    if args.test_synthetic:
        print(f"Synthetic results: "
              f"{output_dir / field / syn_name / 'BP3M_results' / 'synthetic_comparison.csv'}")
    else:
        print(f"Results: {output_dir / field / 'BP3M_results' / 'stellar_astrometry.csv'}")
    print("=" * 55)


if __name__ == '__main__':
    main()
