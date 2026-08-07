# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (development)
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/pypass/test_core.py

# Run a specific test
pytest tests/pypass/test_core.py::test_psf_normalisation

# Run the main pipeline (entry point)
bp3m --name "Leo I" --search_radius 0.1 --output_dir ./outputs

# Run v2 pipeline (adds HST-only sources via master cross-match)
bp3m-v2 --name "Leo I" --output_dir ./outputs

# Download PSF/GDC library files from STScI (one-time setup; HST by default)
bp3m-setup

# Also/instead download the JWST library (NIRCam/NIRISS/MIRI; NIRCam GDCs are several GB)
bp3m-setup --telescope JWST
bp3m-setup --telescope both

# Install notebooks into a target directory
bp3m-notebooks

# Resume pipeline after cross-match with new alignment parameters
bp3m --name "Leo I" --skip_download --skip_psf --skip_crossmatch \
     --n_bp3m_iter 30 --bp3m_clip_sigma 3.5

# Re-run star classification without re-fitting PSFs
bp3m --name "Leo I" --skip_download --skip_crossmatch \
     --reclassify_stars --conc_limit 0.85
```

## Architecture

The repo is a single Python package (`bp3m`) plus two bundled subpackages (`pypass`, `gaia_cross_match`) and a top-level entry-point script (`bp3m_run.py`).

### Pipeline flow (bp3m_run.py → bp3m/pipeline/)

As of August 2026, every HST/JWST module pair in the pipeline has been
consolidated into a single file that takes a `telescope='HST'|'JWST'` parameter
internally, and `bp3m_run.py` has been fully migrated to call these merged
modules unconditionally (no more if/else import branches at any call site).
The old `_jwst`/`_cal` files (`download_jwst.py`, `psf_fitting_cal.py`,
`cross_match_jwst.py` ×2, `validator_jwst.py`, `data_loader_cal.py`,
`run_alignment_jwst.py`, `synthetic_jwst.py`) are left on disk, untouched and
orphaned — nothing imports them anymore. See the per-module notes below for
merge details.

| Step | Module | What it does |
|------|--------|--------------|
| 1 | `download_gaia.py` | TAP query to Gaia DR3; caches as CSV (always shared) |
| 2 | `download_hst.py` | MAST search + download; `telescope` param picks HST/JWST behaviour |
| 3 | `psf_fitting.py` | Parallelises PSF fitting over images via multiprocessing; `telescope` param dispatches HST (pypass) vs JWST (jwst1pass_py_v2) internally |
| 4 | `cross_match.py` | Parallelises `gaia_cross_match` over images; `telescope` param picks HST/JWST behaviour |
| 5 | `run_alignment.py` | Calls `BP3MSolver` from `bp3m/solver.py`; `telescope` param passed through to the data loader |
| 5a | `synthetic.py` | Optional: generate synthetic observations + pull tests; `telescope` param picks HST/JWST behaviour |

Steps can be individually skipped with `--skip_download`, `--skip_psf`, `--skip_crossmatch`, `--skip_alignment`. An obsid manifest (`{field}_selected_obsids.json`) persists image selection across runs.

**Gaia TAP timeout behaviour** (`download_gaia.py:124-193`): each magnitude-bin query runs in a worker thread under a hard `_GAIA_QUERY_TIMEOUT = 60`s deadline (`future.result(timeout=...)`), retried up to `_GAIA_MAX_RETRIES = 3` times before raising:
```
TimeoutError: Gaia TAP timed out after 60s — check connectivity to gea.esac.esa.int
```
Observed in practice: the TAP job can actually finish server-side (visible as `INFO: Query finished.` / `INFO: Removed jobs: [...]` in the astroquery log) *after* the local 60s deadline has already fired and the thread was abandoned — i.e. this isn't necessarily "archive unreachable," it can just be "round-trip took longer than 60s" (slow network path, VPN, or the archive under load). Options if this happens:
1. Just retry — a later run may finish within the deadline if it was transient congestion.
2. Raise `_GAIA_QUERY_TIMEOUT` in `download_gaia.py` if this happens consistently on a given network.
3. Narrow the query (`--search_radius`, `--min_gmag`/`--max_gmag`) so each TAP job returns faster.

### Telescope dispatch

Every step (2 through 5a) now takes `telescope='HST'|'JWST'` directly on its
merged module's entry point, and `bp3m_run.py` imports each module
unconditionally — no if/else branches at any call site. `_im_type` is resolved
once at the start of `main()` (`args.jwst_im_type` for JWST, `args.hst_im_type`
for HST) and threaded through everywhere.

- Step 2: `download_hst_images()` in `download_hst.py`. `--hst_im_type` (default `_flc`) or `--jwst_im_type` (default `_cal`) passed through as `im_type`. JWST supports NIRCam, NIRISS, MIRI.
- Step 3: `run_psf_fitting()` / `reclassify_psf_catalogs()` / `remeasure_psf_perturbation()` in `psf_fitting.py`. For JWST, `_fit_one_image` dispatches via `params['_telescope']` to `_fit_one_image_jwst`, which calls `jwst1pass_py_v2` and produces pypass-schema catalogs.
- Step 4: `run_cross_match()` in `pipeline/cross_match.py`, and `process_single_image()` in `gaia_cross_match/cross_match.py`.
- Step 5: `run_alignment()` in `run_alignment.py`; `telescope` is passed through to `data_loader_flc.load_image_data_flc`.
- Step 5a: `generate_synthetic_data()` / `compare_synthetic_results()` / `run_conditional_solve()` in `synthetic.py`.

HST uses `_flc` images; JWST uses `_cal` images. `split_ccd` defaults to `False` for JWST throughout (ACS/WFC-chip-specific, not applicable to JWST detectors). `inflate_hst_errors` (per-image empirical covariance inflation; the "hst" in the name is legacy — it operates on generic per-image position covariance) now defaults to `True` for JWST as well, same as HST, controlled by `--no_inflate_hst_errors`. The `NotImplementedError` guard for non-HST telescopes has been removed from every step.

### Additional pipeline modules

- `pipeline/psf_fitting.py` (merged HST+JWST, August 2026) — `run_psf_fitting`, `reclassify_psf_catalogs`, and `remeasure_psf_perturbation` all take `telescope='HST'|'JWST'` and this is what `bp3m_run.py` imports for both telescopes. For JWST, `_fit_one_image` dispatches (via `params['_telescope']`) to `_fit_one_image_jwst`, and `remeasure_psf_perturbation` dispatches to `_remeasure_psf_perturbation_jwst` — both full copies of the logic that used to live only in `psf_fitting_cal.py`. `_fit_one_image_jwst` computes `zero_point` per-image from `PIXAR_SR` (`ZP_AB = -2.5 * log10(PIXAR_SR × 1e6 / 3631)`), calls `jwst1pass_py_v2.jwst1pass.io.run_photometry_fits`, and writes a pypass-schema catalog via `_build_jwst_catalog_table`. New helpers `_ensure_jwst1pass()` and `_build_jwst_catalog_table()`, and a `_JWST_DEFAULTS` dict (`fmin_thresh=5.0`, `hmin=5`, `half_width=5`, `mag_limit=28.0`) alongside the existing `_HST_DEFAULTS` (`fmin_thresh=100.0`, `hmin=4`, `half_width=3`, `mag_st_max=28.0`). Fixed one latent bug found during the merge: `_effective_fmin()` used to raise `KeyError` on every JWST status-print call (looked up `_JWST_DEFAULTS['mag_st_max']`, a key that only exists in `_HST_DEFAULTS`) — silently swallowed by a broad `except Exception: break` at the call site, which just truncated the parallel-fitting status messages; now correctly reads `mag_limit` for JWST.
- `pipeline/psf_fitting_cal.py` — **orphaned** (August 2026). No longer imported anywhere; superseded by `psf_fitting.py`'s `telescope='JWST'` branch.
- `pipeline/hst_catalog_crossmatch.py` — Cross-match ALL HST sources between images (not just Gaia-matched ones). Three-phase: (1) within-filter, (2) cross-filter, (3) Gaia recovery. Outputs go to `hst_xmatch/`. Used by the v2 pipeline to build the master catalog for BP3M v2.
- `pipeline/run_alignment.py` (JWST support, August 2026) — `run_alignment()` takes `telescope='HST'|'JWST'`, passed through to `data_loader_flc.load_image_data_flc`. `split_ccd` default unchanged at `True` (HST); `bp3m_run.py` always passes an explicit `split_ccd=False if telescope=='JWST' else ...` at the call site regardless of the function default.
- `pipeline/run_alignment_jwst.py` — **orphaned** (August 2026). No longer imported anywhere; superseded by `run_alignment.py`'s `telescope='JWST'` branch.
- `pipeline/run_alignment_v2.py` — BP3M v2 alignment using `master_combined_v2.csv`. Adds HST-only sources (no Gaia prior) with a phased-inclusion callback (`V2AlignmentCallback`) that enables them after iteration `hst_enable_iter`. Writes to `BP3M_v2_results/`.
- `pipeline/run_iterate_v2.py` — Entry point for `bp3m-v2`. Orchestrates: (1) initial master cross-match → (2) BP3M v2 alignment → (3) updated master cross-match; repeated `--n_refine` times.
- `pipeline/data_loader_master.py` — Loads `master_combined_v2.csv` for BP3M v2. HST-only sources get synthetic negative Gaia IDs, flat position priors, and Michalik+100 mas/yr PM prior (treated as `gaia_2p`).
- `pipeline/catalog_utils.py` — Gaia covariance construction, quality filtering, error inflation. `GAIA_REQUIRED_COLS` lists the 33 columns expected from a Gaia CSV.
- `pipeline/explore_utils.py` — `load_gaia_catalog()`, `load_bp3m_results()` and other notebook helpers.
- `pipeline/output.py` — `print_field_summary()`, `write_ds9_region_file()`.
- `pipeline/synthetic.py` (merged HST+JWST, August 2026) — `generate_synthetic_data()`, `compare_synthetic_results()`, and `run_conditional_solve()` all take `telescope='HST'|'JWST'` (added to `run_conditional_solve`, which had no such parameter in either original file — JWST worked before only because `bp3m_run.py` imported the whole separate `synthetic_jwst.py` module). `_mjd_from_flc()` and `_sky_to_pixel()` also take `telescope`. `im_type` default changed from a hardcoded `'_flc'` to `None`-resolved-by-telescope. Matched-Gaia index column branches `hst_index`/`jwst_index`. **Known pre-existing bug, deliberately left unfixed during this merge**: `_sky_to_pixel()`'s HST branch does `from miracle_match import rd2x, rd2y` — missing the `gaia_cross_match.` package prefix (no top-level `miracle_match` module exists; verified `ModuleNotFoundError` on import). `_ensure_fcm()` is a no-op and does not rescue this via sys.path. `_sky_to_pixel()` is called unconditionally inside `generate_synthetic_data()`'s per-image loop with no surrounding try/except, so **every HST synthetic-test run (`--test_synthetic` with `--telescope HST`, the default) should currently crash** at that point. The JWST branch (`from gaia_cross_match.miracle_match import rd2x, rd2y`) is correct. This bug predates the merge and was already present in `synthetic.py` before this session — it is now documented inline in the code with a comment, not fixed.
- `pipeline/synthetic_jwst.py` — **orphaned** (August 2026). No longer imported anywhere; superseded by `synthetic.py`'s `telescope='JWST'` branch. Its `_sky_to_pixel()` has the correct import that `synthetic.py`'s HST branch is missing (see bug note above).
- `bp3m/data_loader_flc.py` (merged HST+JWST, August 2026) — `_read_image_meta`, `_build_stars_df`, and the public `load_image_data_flc(data_root, field_name, pos_err_floor=..., telescope='HST')` all take `telescope`, and this is what `run_alignment.py`/`synthetic.py` import for both telescopes now. For JWST: image suffix `_cal.fits`/`_cal_catalog.fits`, `EXPSTART`/`EXPEND` fall back to `MJD-BEG`/`MJD-END` (HST requires them present, direct dict access), matched-Gaia index column is `jwst_index` (not `hst_index`), directory root `JWST/mastDownload/JWST/`. Function name `load_image_data_flc` is unchanged (both files already exported a same-named function pre-merge, just from different modules).
- `bp3m/data_loader_cal.py` — **orphaned** (August 2026). No longer imported anywhere; superseded by `data_loader_flc.py`'s `telescope='JWST'` branch.
- `bp3m/checkpointing.py` — Save/restore solver inputs and posterior arrays. Layout: `metadata.json`, `gaia_catalog.csv`, `hst_sources/<img>.csv`, `results/{r_hat,C_r,v_hat,v_mean,v_cov,...}.npy`.

### Core solver (`bp3m/solver.py`)

`BP3MSolver` implements the closed-form Gaussian posterior from McKinnon et al. (in prep) using Schur complement / information-form marginalisation. The joint model:

```
x_survey_ij = X_ij @ r_j - JU_ij @ v_T,i
```

- `r_j` — image transformation: 8D linear (`a,b,c,d,w,z,Δα₀,Δδ₀`) for `poly_order=1`; expands to 14D/22D for order 2/3
- `v_T,i` — 5D astrometry per star: `(Δα*, Δδ, μα*, μδ, ϖ)`

**EM loop** (per outer iteration):
1. Compute `C_s = R @ C_hst @ R^T` (rotate HST covariance to survey frame)
2. Schur complement → `r_hat`, `C_r` (image transformation posterior)
3. Conditional solve → `v_hat`, `C_vT` (per-star astrometry, given `r_hat`)
4. Update rotation matrices `R_j` from new `(a,b,c,d)` in `r_hat`

**Outlier rejection**: MAD sigma clipping (`--bp3m_clip_sigma`, default 4.5) + Cook's D influence clipping (`--influence_d_thresh`, default 1.0; disabled with `--no_influence_clip`).

**Gaia source types** (set in `_cache_gaia`):
- `gaia_5p` — 5-parameter solution (full PM + parallax); uses Gaia covariance directly (×1.05 inflation)
- `gaia_6p` — 6-parameter (pseudocolour); treated like 5p (×1.22 inflation)
- `gaia_2p` — 2-parameter (position only); gets diffuse PM prior (100 mas/yr) + Michalik parallax prior

A sparse variant lives in `solver_sparse.py` (activated by `--sparse`); `astro_utils.py` owns all coordinate/matrix helpers.

### PSF fitting subpackage (`pypass/`)

**`StarRecord` dataclass** (all fields, in declaration order):
```
x, y, flux, flux_err, sky, sky_err, mag, mag_err
qfit          # Σ|res|/Σ|data-sky|, 0=perfect; Fortran 'q'
chi2          # sqrt(Σr²/var / (n_good-4)); ~1=good; Fortran 'c'
central_res   # (data_cen - sky - flux·P_cen)/flux; Fortran 'C'
n_sat         # pixels in fit window above sat_threshold; Fortran 'n'
psf_frac      # PSF value at fitted position; Fortran 'f'
psf_peak      # PSF value at perfect center; Fortran 'F'
peak          # raw peak pixel value
cov           # 4×4 ndarray in (flux, x, y, sky) order
pass_number
n_neighbors, dist_nearest, dist_nearest_brighter  # filled by compute_neighbor_stats
n_iter        # Newton iterations taken
converged     # False if hit max_iter
delta_max     # max(|δx|,|δy|) at final Newton step
clipped_mask  # sigma-clipped pixel mask
chi2_scale    # max(chi2_individual, chi2_global_floor)
eps_psf       # chi2 / sqrt(flux * psf_frac * gain); implied PSF model error
concentration      # 1×1 peak-pixel / (flux * PSF_model_peak)
concentration_2x2  # 2×2 sum / (flux * PSF_model_2x2_sum)
concentration_3x3  # 3×3 sum / (flux * PSF_model_3x3_sum)
n_conc_1x1, n_conc_2x2, n_conc_3x3  # unmasked pixel counts for each metric
is_star_candidate   # bool — set by classify_stars()
dq_1x1, dq_2x2, dq_3x3  # bitwise OR of raw DQ integer values at 1×1/2×2/3×3
```

**`run_photometry_fits()`** (`pypass/io.py`) key parameters:
```
image_path         FITS file path
psf_path / lib_dir PSF lookup; lib_dir/STDPSFs/{det_prefix}/
gdc_path           GDC lookup; lib_dir/STDGDCs/{det_prefix}/
fmin_thresh        hard floor in electrons (default 40)
mag_st_max         faint ST-mag limit → fmin = max(fmin_from_mag(mag_st_max), fmin_thresh)
hmin               NMS suppression radius in pixels (default 4)
n_passes           total fitting passes (default 1)
half_width         PSF fit half-window in pixels (default 3)
backend            'auto'|'numpy'|'jax' (see _backend.py)
conc_limit         concentration lower bound for classify_stars (default 0.9)
psf_delta          additive PSF correction array (from psf_delta.npy)
```

**`classify_stars()`** (`pypass/core.py:1329`) — sets `is_star_candidate` on every record in-place:
- Pass 1: concentration in `[conc_lo, 1/conc_lo]` (all three metrics that are finite)
- Pass 2: adaptive per-magnitude window: `median ± conc_width_factor × half_68%_spread`
- qfit below `qfit_global_max=1.5` AND below `p25 × 4.0` within magnitude bin

**Backend dispatch** (`pypass/_backend.py`):
- `'numpy'` — always use NumPy/scipy
- `'jax'` — force JAX (raises `ImportError` if not installed)
- `'auto'` — use JAX when n_stars ≥ threshold (2000 CPU-only, 500 GPU/TPU); reads `PYPASS_BACKEND` env var; threshold overridable via `PYPASS_JAX_THRESHOLD`

**JAX kernel** (`pypass/_jax_kernel.py`) — pre-computes batched fixed-shape arrays (`prepare_jax_inputs`) consumed by `jax.vmap + jax.jit` Newton-loop kernel.

**Multi-pass loop** (`pypass/multipass.py`):
- `run_photometry()` — main entry; per-pass: detect → fit → subtract → re-fit
- `subtract_stars()` — removes PSF models of converged stars from residual image
- `refit_stars()` — leave-one-out re-fitting in discovery passes
- `build_variance_image()` — accumulates neighbour Poisson noise into variance map
- `deduplicate_records()` — removes sources within hmin of a brighter source (KDTree O(N log N))
- PSF interpolation cache: `psf_cache={}` dict keyed by `(int(x)//5, int(y)//5)`, max 2048 entries

**`catalog_to_table()`** (`pypass/io.py:1133`) — converts records to astropy Table. Complete column list:
```
# Core photometry (from StarRecord)
x, y, flux, flux_err, sky, sky_err, mag, mag_err
qfit, chi2, central_res, n_sat, psf_frac, psf_peak, peak, pass_number
n_neighbors, dist_nearest, dist_nearest_brighter

# Covariance (4×4 flattened, chi2-scaled + floor applied)
cov_ff, cov_xx, cov_yy, cov_ss
cov_fx, cov_fy, cov_fs, cov_xy, cov_xs, cov_ys

# Convergence / fit quality
n_iter, converged, delta_max, chi2_scale, eps_psf

# Concentration and classification
concentration, concentration_2x2, concentration_3x3
n_conc_1x1, n_conc_2x2, n_conc_3x3
is_star_candidate

# Noise model
sigma_x_model, sigma_y_model, sigma_f_model
chip_ext

# GDC-corrected astrometry (NaN when no GDC applied)
x_gdc, y_gdc, mag_gdc, mag_err_gdc
cov_xx_gdc, cov_yy_gdc, cov_xy_gdc

# WCS sky coordinates (NaN when no WCS)
ra, dec, ra_err, dec_err
cov_ra_ra, cov_dec_dec, cov_ra_dec

# Calibrated magnitudes (NaN when photometric keywords absent)
mag_st, mag_ab, mag_st_gdc
```

Table metadata keys: `SIGMA_FLOOR_X`, `SIGMA_FLOOR_Y`, `EPS_FLUX`, `ZP_AB` (when computed from `PIXAR_SR`).

### Cross-matching subpackage (`gaia_cross_match/`)

- `cross_match.py` — merged HST+JWST implementation (August 2026; was two separate files). `process_single_image(img, ..., telescope='HST')`: 4P offset discovery (2D histogram peak) → 6P affine refinement with per-iteration empirical residual covariance floor → final match, shared verbatim between telescopes. Telescope-specific pieces:
  - `get_hst_params(image_file, catalog_file=None, telescope='HST')` — for HST reads `ORIENTAT` + ACS/WFC3 pixel scales from FLC headers; for JWST (early-return branch) reads `PA_APER` (then `ORIENTAT`, then `PA_V3` as fallback) from the SCI extension, pixel scales from `_JWST_PIXEL_SCALE`/per-detector literals, `initial_scale` varies by detector (≈0.99–1.01) rather than the fixed `1.0` the old `cross_match_jwst.py` used
  - `find_hst_image_folders(target, data_dir, telescope='HST')` — HST branch walks `HST/`, matches `*_flc_catalog.fits`; JWST branch walks `JWST/`, matches `*_cal_catalog.fits`; dict key is `flc` or `cal` accordingly
  - Output columns: HST writes `hst_*` + `hst_is_star`; JWST writes `jwst_*` + plain `is_star` (not `jwst_is_star` — asymmetric, preserved intentionally since `validator.py`'s JWST branch expects `is_star`)
  - `_CHIP_CONFIG`/`get_chip_config` (HST-only two-chip detector config) kept, used only on the HST branch
- `cross_match_jwst.py` — **orphaned** (August 2026). No longer imported anywhere; superseded by `cross_match.py`'s `telescope='JWST'` branch. Left on disk untouched. One bug was found and fixed during the merge and does **not** carry over: its standalone `main()` imported `.validator` (HST) instead of `.validator_jwst`, so a JWST cross-image validation run via that CLI entry point would silently no-op (see `validator.py` note below).
- `catalog_matcher.py` — nearest-neighbour matching with magnitude constraint
- `miracle_match.py` — fallback robust geometric matching via V/VMAX + SNS + progressive sigma tightening
- `diagnostics.py` — 8-panel per-image diagnostic plots
- `validator.py` — merged HST+JWST cross-image validation (August 2026; was two separate files, `validator.py` + `validator_jwst.py`). `validate_target(target, data_dir, ..., telescope='HST')` and every helper it calls (`_science_filter`, `load_image_data`, `find_processed_images`, `has_valid_stmag`, `compute_pairwise_zps`, `validate_filter_group`, `write_solo_quality`, `build_global_catalog`) now take `telescope`. JWST output column asymmetry preserved: `is_star`/`jwst_index_list` (not `jwst_is_star`) vs HST's `hst_is_star`/`hst_index_list`.
- `validator_jwst.py` — **orphaned** (August 2026), no longer imported anywhere; superseded by `validator.py`'s `telescope='JWST'` branch.

**Hard catalog column requirements** (both `cross_match.py` telescope branches raise or skip without these):

| Column | Behaviour if missing |
|--------|---------------------|
| `is_star_candidate` | Image SKIPPED with WARNING |
| `x_gdc`, `y_gdc` | Rows dropped if NaN/inf; crash if column absent |
| `mag_st_gdc` | `ValueError` ("stale jwst1pass/py1pass output") |
| `cov_xx_gdc`, `cov_yy_gdc`, `cov_xy_gdc` | Used to build 2×2 positional covariance |
| `qfit`, `chi2` | Used for 4P discovery quality tiers |

**Optional catalog columns** (used when present, NaN-safe otherwise):
`mag_gdc`, `mag_err_gdc`, `mag_ab`

### PSF fitting defaults

Both dicts below now live in `psf_fitting.py`, which is the live path for both
telescopes (the orphaned `psf_fitting_cal.py` has an identical, unused copy of
`_JWST_DEFAULTS` — see "Additional pipeline modules").

`_HST_DEFAULTS` (tuned for HST FLC images):
```python
fmin_thresh=100.0, mag_st_max=28.0, hmin=4, n_passes=2, n_discovery_passes=1,
sat_threshold=60000.0, max_iter_fit=100, half_width=3,
sky_inner=4, sky_outer=8, tol=1e-3,
sigma_clip=True, sigma_clip_sigma=4.0,
conc_limit=0.9, n_jobs=-1, backend='auto'
```

`_JWST_DEFAULTS` (tuned for JWST CAL images):
```python
fmin_thresh=5.0, hmin=5, half_width=5, mag_limit=28.0,
n_passes=2, sky_inner=4, sky_outer=8, conc_limit=0.9, n_jobs=-1
```

### Output layout

```
{output_dir}/{field}/
  Gaia/
    {stem}.csv                          ← Gaia DR3 catalog cache
  HST/mastDownload/HST/{obsid}/         ← HST images (--telescope HST)
    {obsid}_flc.fits
    {obsid}_flc_catalog.fits            ← pypass output
    psf_params.json                     ← saved fitting parameters
    psf_delta.npy                       ← PSF perturbation (if measured)
    matched_gaia.csv                    ← cross-match output
    transformation.csv                  ← affine transformation params
    diagnostic_plots.png
    offset_histogram.png
    psf_catalog_stats.png
    psf_concentration.png
  JWST/mastDownload/JWST/{obsid}/       ← JWST images (--telescope JWST)
    {obsid}_cal.fits
    {obsid}_cal_catalog.fits            ← jwst1pass_py_v2 output (pypass schema)
    psf_params.json
    psf_delta.npy                       ← PSF perturbation (reference; no iteration yet)
    {obsid}_cal_residual.fits           ← per-chip SCI/VAR/MASK residual
    psf_catalog_stats.png
    psf_concentration.png
    psf_diagnostics.png
    psf_residual_map.png
    psf_perturbation.png
    matched_gaia.csv                    ← cross-match output (Step 4)
  hst_xmatch/                           ← v2 pipeline only
    detections_{filter}.csv
    master_{filter}.csv
    master_combined.csv
    master_combined_v2.csv
    gaia_recovered.csv
  BP3M_results/
    stellar_astrometry.csv              ← primary science output
    image_transformations.csv
    v_cov_marginalised.npy              ← (N, 5, 5) full covariance
    plots/
  BP3M_v2_results/                      ← v2 pipeline
    stellar_astrometry.csv
    image_transformations.csv
    v_cov_marginalised.npy
  bp3m_command.txt                      ← saved CLI invocation
  {field}_selected_obsids.json          ← obsid manifest from step 2
  {field}_failed_obsids.json            ← failed obsid reasons
```

### Two output sets in stellar_astrometry.csv

- `pmra_bp3m` / `sigma_pmra_bp3m` — **marginalised** over alignment uncertainty; stars are correlated; use for Gaia comparisons
- `pmra_bp3m_cond` / `sigma_pmra_bp3m_cond` — **conditional** (MAP alignment fixed); stars uncorrelated; use for per-star membership analyses

### Gaia star classification

Stars are typed by which Gaia solution they have, checked in `solver.py::_cache_gaia`:
- `gaia_5p` — 5-parameter solution (full PM + parallax); uses Gaia covariance directly
- `gaia_6p` — 6-parameter (pseudocolour); treated like 5p
- `gaia_2p` — 2-parameter (position only); gets diffuse PM prior (100 mas/yr) + Michalik parallax prior

### JWST status

As of August 2026, JWST is fully wired through every pipeline step (2–5a), with
`bp3m_run.py` calling the same merged modules for both telescopes. See "Pipeline
flow" and "Telescope dispatch" above for the step-by-step summary; this section
covers implementation depth and known gaps.

**What is implemented:**
- `download_hst.py` (merged HST+JWST, formerly also `download_jwst.py`) — full MAST search + download for NIRCam, NIRISS, MIRI `_cal.fits` via `telescope='JWST'`; reads available STDPSFs/STDGDCs from lib_dir to filter to supported filters; writes the same obsid manifest as the HST path; `_check_exptime()` validates exposure times (branches on telescope: HST checks `EXPTIME`/`EXPFLAG`/HDRLET presence, JWST checks `EFFEXPTM`/`ENG_QUAL`/`DATAPROB`/`VISITSTA`).
- `jwst1pass_py_v2` (in `GaiaWebb-master/jwst1pass_py_v2/`) — JWST PSF-fitting engine, BP3M-compatible:
  - `StarRecord` carries `concentration`, `concentration_2x2`, `concentration_3x3`, `n_conc_1×1/2×2/3×3`, and `is_star_candidate` fields, matching pypass schema
  - `classify_stars()` and `_conc_adaptive_bounds()` are verbatim copies of the pypass implementations
  - `qfit` convention matches pypass: genuine quality metric (`small ≈ good`); old `qfit=0.0` sentinel removed
  - `conc_limit` threaded through `run_photometry_fits` → `_run_nircam_meta` → `run_photometry` → `classify_stars`
  - Validated on Draco NIRISS F200W: 81/515 star candidates correctly identified
- **Step 3 PSF fitting** — merged into `psf_fitting.py` (see "Additional pipeline modules" for the `_effective_fmin` bug fixed during the merge). `_fit_one_image_jwst()` reads `PIXAR_SR` from the FITS primary header to compute `zero_point` (`ZP_AB = -2.5 * log10(PIXAR_SR × 1e6 / 3631)`), calls `jwst1pass_py_v2`, writes catalog, params sidecar, residual FITS (per-chip SCI/VAR/MASK with DQ + sigma-clip masks), all diagnostic plots, and PSF perturbation `psf_delta.npy`.

- **Step 4 cross-matching** — merged into `gaia_cross_match/cross_match.py`. `get_hst_params(flc_file, catalog_file=None, telescope='JWST')` (param still named `flc_file` even for JWST calls) reads `PA_APER`/`ORIENTAT` from SCI ext, pixel scale from `_JWST_PIXEL_SCALE`, `EXPSTART`/`MJD-BEG` from primary header; `initial_scale` is now per-detector (≈0.99–1.01, derived from the WCS-measured pixel scales below) rather than the old fixed `1.0`. `find_hst_image_folders(telescope='JWST')` walks `JWST/` and matches `*_cal_catalog.fits` + `*_cal.fits`. `process_single_image(img, ..., telescope='JWST')` outputs `jwst_index`, `jwst_x_gdc`, `jwst_y_gdc`, `jwst_mag_gdc`, `jwst_mag_st_gdc`, `jwst_mag_ab`, `is_star`. `gaia_cross_match/__init__.py` no longer exports the old `_jwst`-suffixed symbols (`process_single_image_jwst`, `find_jwst_image_folders`, `get_jwst_params`, etc.) since the merged `cross_match.py` symbols cover both telescopes.

- **Step 4 cross-image validation** — merged into `gaia_cross_match/validator.py`. `_science_filter(h0, telescope='JWST')`: reads `FILTER`/`PUPIL` from primary header; for NIRISS returns `PUPIL` when `FILTER == CLEAR` (matching `jwst1pass/io.py::_extract_filter`). `load_image_data(telescope='JWST')`: reads `*_cal.fits`; `INSTRUME`/`DETECTOR`/`FILTER` from primary header, `CRVAL1`/`CRVAL2` from SCI ext (`h[1]`). `find_processed_images(telescope='JWST')`: walks `JWST/`, matches `*_cal_catalog.fits`. Column references branch correctly: `jwst_mag_st_gdc`, `jwst_mag_err_gdc`, `jwst_index`, `jwst_index_list`, `is_star` (not `hst_is_star`) for JWST vs `hst_*`/`hst_is_star`/`hst_index_list` for HST. Note: the orphaned `cross_match_jwst.py`'s standalone `main()` used to import the wrong validator (`.validator` instead of `.validator_jwst`) — fixed as a drive-by during the merge even though that file is no longer live, in case it's ever resurrected.

- **Step 5 alignment / Step 5a synthetic** — merged into `run_alignment.py` and `synthetic.py`; see "Additional pipeline modules" for details, including the pre-existing `_sky_to_pixel()` import bug in `synthetic.py`'s HST branch (deliberately left unfixed).

- **`bp3m_run.py` JWST wiring** — fully migrated (August 2026). No step has an `if args.telescope.upper() == 'JWST': import X_jwst else: import X` branch anymore; every step imports its merged module unconditionally and passes `telescope=args.telescope` (or `.upper()`) explicitly:
  - `_im_type` resolved once at start of `main()`: `args.jwst_im_type` for JWST, `args.hst_im_type` for HST; used in Steps 3, 4, and synthetic test
  - Step 2: `download_hst.download_hst_images`; failed-obsid rescan block (after Step 2) imports `_check_exptime` from the same module
  - Step 3: `psf_fitting.{run_psf_fitting, reclassify_psf_catalogs, remeasure_psf_perturbation}`
  - Step 4: `cross_match.run_cross_match`
  - Step 5: `run_alignment.run_alignment`; `split_ccd` forced to `False` for JWST in all call sites (`inflate_hst_errors` follows `--no_inflate_hst_errors` for both telescopes as of 2026-08-01)
  - Step 5a/5b/5c: `synthetic.{generate_synthetic_data, compare_synthetic_results, run_conditional_solve}`; `_split_ccd_syn` and `_inflate_errors_syn` computed once and reused across synthetic sub-steps

- **`_JWST_PIXEL_SCALE` in `cross_match.py`** (moved from the now-orphaned `cross_match_jwst.py`) — nominal pixel scales (arcsec/px) used for initial Gaia projection; verified against real `_cal.fits` WCS CD matrices (July 2026):

  | Key | Hardcoded | WCS measured | Notes |
  |-----|-----------|--------------|-------|
  | `NIRCAM_SW` | 0.031 | 0.0308–0.0313 | varies by detector (A3, A4, B3) |
  | `NIRCAM_LW` | 0.063 | 0.0628 | NRCALONG |
  | `NIRISS` | 0.066 | 0.0653 | NIS |
  | `MIRI` | 0.111 | 0.1099–0.1105 | varies by dither pointing |

  All within ~1% of WCS values. `get_hst_params(..., telescope='JWST')` now encodes each detector's measured ratio directly as `initial_scale` (e.g. NRCA1=1.0073, NIRISS=0.9934, MIRI=0.9992) rather than leaving the full ~1% correction to the 6P affine refinement.

**What is not yet implemented:**
- PSF iteration (`n_psf_iter >= 2`) for JWST — `psf_delta` is not passed into `_fit_one_image_jwst`; `psf_delta.npy` is written for reference only
- HST synthetic testing (`--test_synthetic` with `--telescope HST`, the default) is currently broken by the pre-existing `_sky_to_pixel()` import bug in `synthetic.py` — see "Additional pipeline modules". JWST synthetic testing is unaffected (its branch has the correct import).

### Library setup (`bp3m-setup` / `bp3m/setup.py`)

`bp3m-setup` downloads the STDPSF/STDGDC reference library from Jay Anderson's STScI pages. As of August 2026 it supports both telescopes:

- `--telescope {HST,JWST,both}` — default `HST` (unchanged, for backward compatibility). JWST is opt-in since NIRCam GDCs alone total several GB.
- `--instruments` — HST instrument list (unchanged: `ACSWFC ACSHRC WFC3UV WFC3IR`).
- `--jwst-instruments` — JWST instrument list, default all of `NIRCam NIRISS MIRI`. Source: `https://www.stsci.edu/~jayander/JWST1PASS/LIB`.

**Filenames are discovered by scraping the server's directory listing (`_list_fits`/`_list_dirs`), never guessed from a fixed template.** This matters because STScI is inconsistent about NIRCam STDPSF/STDGDC filename token order: most filters publish `STD{X}_{detector}_{filter}.fits`, but a few (confirmed: GDCs for `F210M`, `F070W` under `NIRCam/SWC`) are published the other way round, `STD{X}_{filter}_{detector}.fits`. A downloader that assumes one fixed order 404s on those filters — and a prior, non-bp3m download of this library did exactly that, silently saving the resulting HTML 404 page to disk with a `.fits` extension (17 corrupted files found in `GaiaWebb-master/lib/STDGDCs/NIRCam/`, all in `F210M`/`F070W`), which later crashed `jwst1pass_py_v2.io.load_stdgdc()` deep inside PSF fitting with `OSError: No SIMPLE card found`.

Two fixes in `setup.py` address this:
1. `_canonical_nircam_name(kind, basename)` parses the detector (`NRCA1`–`NRCB4`, `NRCAL`/`NRCBL`) and filter tokens out of whatever name the server actually used, and always **saves** the file locally as `STD{kind}_{detector}_{filter}.fits` — the fixed order `find_gdc()`/`find_psf()` in `jwst1pass_py_v2/jwst1pass/io.py` construct when looking a file up. So the on-disk name is always canonical even when the upstream name isn't.
2. `_download()` now checks the downloaded content actually starts with the FITS `SIMPLE` magic bytes before committing it to `dest`; anything else (error page, truncated transfer) is treated as a failed download and the temp file is discarded, rather than being saved as if it were real data.

`_download_jwst_group()` handles NIRCam's two possible layouts generically — SWC is split into per-filter subdirectories, LWC is flat — by trying to list filter subdirectories first and falling back to listing `.fits` files directly if none are found, so it doesn't hardcode which channel has which layout.

### Notebooks

Jupyter notebooks in `bp3m/notebooks/` (and mirrored in `notebooks/`) cover field overview, proper motions, astrometric quality, cross-match diagnostics, v2 results, and alignment posterior sampling. Install with `bp3m-notebooks`.
