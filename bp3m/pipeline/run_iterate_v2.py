"""
bp3m-v2 — Iterative BP3M v2 + master crossmatch pipeline.

Steps
-----
1. Master crossmatch  (once, using BP3M_results/)
   → writes hst_xmatch/master_combined.csv + master_combined_v2.csv

For each refinement cycle (default 1):
  2. BP3M v2 alignment  (using master_combined_v2.csv)
     → writes BP3M_v2_results/
  3. Master crossmatch  (using BP3M_v2_results/)
     → overwrites hst_xmatch/master_combined.csv + master_combined_v2.csv

Usage
-----
    bp3m-v2 --name "Leo I"

    # Skip the initial crossmatch (already done):
    bp3m-v2 --name "Leo I" --skip_initial_crossmatch

    # Skip both the initial crossmatch and the first v2 BP3M (already done):
    bp3m-v2 --name "Leo I" --skip_initial_crossmatch --start_cycle 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog='bp3m-v2',
        description='Iterative BP3M v2 + master crossmatch pipeline.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Field ──────────────────────────────────────────────────────────────────
    parser.add_argument('--name', required=True,
                        help='Target name — must match the field directory created by bp3m '
                             '(spaces are replaced with underscores, e.g. "Leo I" → Leo_I)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Root output directory (default: current directory)')

    # ── Iteration control ──────────────────────────────────────────────────────
    parser.add_argument('--n_refine', type=int, default=1,
                        help='Number of (v2 BP3M → crossmatch) cycles to run')
    parser.add_argument('--skip_initial_crossmatch', action='store_true',
                        help='Skip step 1 (initial crossmatch); use existing master catalogs')
    parser.add_argument('--start_cycle', type=int, default=1,
                        help='Start at this refinement cycle number (1-based); '
                             'skips earlier cycles (useful for resuming)')

    # ── BP3M v2 parameters ─────────────────────────────────────────────────────
    parser.add_argument('--n_iter', type=int, default=20,
                        help='Maximum BP3M outer iterations per cycle')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Posterior samples for marginalisation')
    parser.add_argument('--clip_sigma', type=float, default=4.5,
                        help='MAD sigma for outlier rejection (0 = disabled)')
    parser.add_argument('--poly_order', type=int, default=None,
                        help='Polynomial order for image transformation')
    parser.add_argument('--hst_enable_iter', type=int, default=5,
                        help='Outer iteration at which HST-only sources are enabled')
    parser.add_argument('--hst_max_pm_unc', type=float, default=5.0,
                        help='Global PM uncertainty cut for HST-only eligibility (mas/yr)')
    parser.add_argument('--hst_max_per_image', type=int, default=1000,
                        help='Per-image cap on HST-only source count')
    parser.add_argument('--hst_pm_sigma_diffuse', type=float, default=100.0,
                        help='Diffuse PM prior sigma (mas/yr) for HST-only stars in v2 '
                             'alignment (default 100)')
    parser.add_argument('--det_chi2_threshold', type=float, default=None,
                        help='Exclude (star,image) pairs with Phase-4 per-detection '
                             'chi2 above this value (suggested: 9.0 = 3sigma)')
    parser.add_argument('--sparse', action='store_true',
                        help='Use sparse Schur-complement solver')
    parser.add_argument('--no_prefilter', action='store_true',
                        help='Skip BP3M Phase-0 pre-filter pass')
    parser.add_argument('--no_influence_clip', action='store_true',
                        help="Disable test-4 Cook's D influence clipping")
    parser.add_argument('--influence_d_thresh', type=float, default=1.0,
                        help="Cook's D threshold for test-4 influence clipping")
    parser.add_argument('--influence_sigma_min', type=float, default=2.0,
                        help='Minimum sigma_resid for test-4 (default 2.0)')
    parser.add_argument('--soft_weights', action='store_true',
                        help='Use Student-t IRLS soft weights instead of hard tests 1-4')
    parser.add_argument('--student_t_nu', type=float, default=50.0,
                        help='Student-t degrees of freedom for soft-weight IRLS')

    # ── Crossmatch parameters ──────────────────────────────────────────────────
    parser.add_argument('--match_n_sigma', type=float, default=5.0,
                        help='Match radius in units of combined positional sigma')
    parser.add_argument('--mag_n_sigma', type=float, default=3.0,
                        help='Magnitude match threshold in units of combined photometric sigma')
    parser.add_argument('--mag_floor', type=float, default=0.10,
                        help='Minimum magnitude tolerance regardless of photometric error (mag)')
    parser.add_argument('--min_detections', type=int, default=2,
                        help='Minimum detections for a source to appear in the master catalog')
    parser.add_argument('--cross_filter_radius_mas', type=float, default=200.,
                        help='Match radius for cross-filter association (mas)')
    parser.add_argument('--gaia_csv', default=None,
                        help='Gaia catalog CSV for Gaia recovery (auto-detected if not given)')
    parser.add_argument('--no_save_detections', action='store_true',
                        help='Skip saving per-filter detection catalogs')
    parser.add_argument('--phase4_outlier_sigma', type=float, default=3.5,
                        help='Per-detection chi2 sigma threshold for Phase 4 outlier rejection')
    parser.add_argument('--zp_max_corr', type=float, default=None,
                        help='Max ZP correction to apply (mag). Default: 0.0 when pre-measured ZP exists, 3.0 otherwise')

    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    field      = args.name.replace(' ', '_')
    field_dir  = output_dir / field

    if not field_dir.exists():
        print(f"Error: field directory not found: {field_dir}")
        print(f"  (looked for '{field}' — make sure --name matches what bp3m used)")
        sys.exit(1)

    from bp3m.pipeline.hst_catalog_crossmatch import run_hst_crossmatch
    from bp3m.pipeline.run_alignment_v2 import run_alignment_v2

    bp3m_v1_dir = field_dir / 'BP3M_results'
    bp3m_v2_dir = field_dir / 'BP3M_v2_results'
    xmatch_dir  = field_dir / 'hst_xmatch'

    # Default poly_order to whatever v1 used, if not explicitly specified
    if args.poly_order is None:
        _v1_cfg = bp3m_v1_dir / 'run_config.json'
        if _v1_cfg.exists():
            import json as _json
            _v1_poly = _json.load(open(_v1_cfg)).get('poly_order', 1)
            args.poly_order = _v1_poly
            print(f"  poly_order not specified — using v1 value: {args.poly_order}")
        else:
            args.poly_order = 1
            print(f"  poly_order not specified and no v1 run_config.json found — defaulting to 1")

    crossmatch_kwargs = dict(
        field_dir                = field_dir,
        output_dir               = xmatch_dir,
        gaia_csv                 = Path(args.gaia_csv) if args.gaia_csv else None,
        match_n_sigma            = args.match_n_sigma,
        mag_n_sigma              = args.mag_n_sigma,
        mag_floor                = args.mag_floor,
        min_detections           = args.min_detections,
        cross_filter_radius_mas  = args.cross_filter_radius_mas,
        save_detections          = not args.no_save_detections,
        phase4_outlier_sigma     = args.phase4_outlier_sigma,
        anchor_bp3m_dir          = bp3m_v1_dir if bp3m_v1_dir.exists() else None,
        zp_max_corr              = args.zp_max_corr,
    )

    bp3m_kwargs = dict(
        output_dir           = output_dir,
        field_name           = field,
        n_iter               = args.n_iter,
        n_samples            = args.n_samples,
        clip_sigma           = args.clip_sigma,
        poly_order           = args.poly_order,
        use_sparse           = args.sparse,
        no_prefilter         = args.no_prefilter,
        hst_enable_iter      = args.hst_enable_iter,
        hst_max_pm_unc       = args.hst_max_pm_unc,
        hst_max_per_image    = args.hst_max_per_image,
        hst_pm_sigma_diffuse = args.hst_pm_sigma_diffuse,
        det_chi2_threshold   = args.det_chi2_threshold,
        use_influence_clip   = not args.no_influence_clip,
        influence_d_thresh   = args.influence_d_thresh,
        influence_sigma_min  = args.influence_sigma_min,
        use_soft_weights     = args.soft_weights,
        student_t_nu         = args.student_t_nu,
    )

    # ── Step 1: initial crossmatch ─────────────────────────────────────────────
    if not args.skip_initial_crossmatch:
        print(f"\n{'#'*60}")
        print(f"# Step 1: initial master crossmatch  (using BP3M_results/)")
        print(f"{'#'*60}")
        if not bp3m_v1_dir.exists():
            print(f"Error: BP3M_results/ not found at {bp3m_v1_dir}")
            print("Run the initial bp3m alignment first, or use --skip_initial_crossmatch")
            sys.exit(1)
        run_hst_crossmatch(**crossmatch_kwargs, bp3m_results_dir=bp3m_v1_dir,
                           cycle_id=0)
    else:
        print(f"\nSkipping initial crossmatch (--skip_initial_crossmatch).")
        v2_csv = xmatch_dir / 'master_combined_v2.csv'
        if not v2_csv.exists():
            print(f"Error: expected {v2_csv} but it does not exist.")
            sys.exit(1)

    # ── Refinement cycles ──────────────────────────────────────────────────────
    for cycle in range(1, args.n_refine + 1):
        if cycle < args.start_cycle:
            print(f"\nSkipping cycle {cycle} (--start_cycle={args.start_cycle}).")
            continue

        print(f"\n{'#'*60}")
        print(f"# Refinement cycle {cycle}/{args.n_refine}")
        print(f"{'#'*60}")

        print(f"\n--- Step 2 (cycle {cycle}): BP3M v2 alignment ---")
        run_alignment_v2(**bp3m_kwargs)

        print(f"\n--- Step 3 (cycle {cycle}): master crossmatch (using BP3M_v2_results/) ---")
        run_hst_crossmatch(**crossmatch_kwargs, bp3m_results_dir=bp3m_v2_dir,
                           cycle_id=cycle)

    print(f"\n{'#'*60}")
    print(f"# Done.")
    print(f"#   Results: {bp3m_v2_dir}")
    print(f"#   Catalog: {xmatch_dir / 'master_combined_v2.csv'}")
    print(f"{'#'*60}")


if __name__ == '__main__':
    main()
