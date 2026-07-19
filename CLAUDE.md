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

# Download PSF/GDC library files from STScI (one-time setup)
bp3m-setup

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

| Step | Module | What it does |
|------|--------|--------------|
| 1 | `download_gaia.py` | TAP query to Gaia DR3; caches as CSV |
| 2 | `download_hst.py` / `download_jwst.py` | MAST search + download; dispatched by `--telescope` |
| 3 | `psf_fitting.py` | Parallelises `pypass` over images via multiprocessing |
| 4 | `cross_match.py` | Parallelises `gaia_cross_match` over images |
| 5 | `run_alignment.py` | Calls `BP3MSolver` from `bp3m/solver.py` |

Steps can be individually skipped with `--skip_download`, `--skip_psf`, `--skip_crossmatch`, `--skip_alignment`. An obsid manifest (`{field}_selected_obsids.json`) persists image selection across runs.

### Telescope dispatch

Step 2 dispatches on `--telescope`:
- `HST` (default) → `download_hst_images()` using `--hst_im_type` (default `_flc`)
- `JWST` → `download_jwst_images()` using `--jwst_im_type` (default `_cal`); supports NIRCam, NIRISS, MIRI

Step 3 (`run_psf_fitting`) in `psf_fitting_cal.py` now supports JWST via `jwst1pass_py_v2` (see JWST status below). `psf_fitting.py` (HST-only) still raises `NotImplementedError` for non-HST. Step 4 (`run_cross_match`) uses the same code path for both telescopes but needs JWST pixel-scale verification.

**JWST PSF fitting**: `psf_fitting_cal.py` is the JWST entry point. `run_psf_fitting(telescope='JWST')` now calls `jwst1pass_py_v2` and produces pypass-schema catalogs; the `NotImplementedError` guard has been removed.

### Additional pipeline modules

- `pipeline/psf_fitting_cal.py` — JWST-only CAL entry point (HST uses `psf_fitting.py`). `run_psf_fitting` calls `_fit_one_image`, which is a thin wrapper around `_fit_one_image_jwst`. `_fit_one_image_jwst` computes `zero_point` per-image from `PIXAR_SR` (`ZP_AB = -2.5 * log10(PIXAR_SR × 1e6 / 3631)`), then calls `jwst1pass_py_v2.jwst1pass.io.run_photometry_fits` and writes a pypass-schema catalog via `_build_jwst_catalog_table`. Key helpers: `_ensure_jwst1pass()` — sys.path injection for jwst1pass_py_v2; `_build_jwst_catalog_table()` — column adapter (renames `q→qfit`, adds `mag_st_gdc`, `is_star_candidate`, `chip_ext`, `eps_psf`, `sigma_*_model`, floor metadata). `_JWST_DEFAULTS` are tuned for JWST CAL: `fmin_thresh=5.0`, `hmin=5`, `half_width=5`, `mag_limit=28.0` (vs. `psf_fitting.py`'s `_HST_DEFAULTS`: `fmin_thresh=100.0`, `hmin=4`, `half_width=3` for HST FLC).
- `pipeline/hst_catalog_crossmatch.py` — Cross-match ALL HST sources between images (not just Gaia-matched ones). Three-phase: (1) within-filter, (2) cross-filter, (3) Gaia recovery. Outputs go to `hst_xmatch/`. Used by the v2 pipeline to build the master catalog for BP3M v2.
- `pipeline/run_alignment_v2.py` — BP3M v2 alignment using `master_combined_v2.csv`. Adds HST-only sources (no Gaia prior) with a phased-inclusion callback (`V2AlignmentCallback`) that enables them after iteration `hst_enable_iter`. Writes to `BP3M_v2_results/`.
- `pipeline/run_iterate_v2.py` — Entry point for `bp3m-v2`. Orchestrates: (1) initial master cross-match → (2) BP3M v2 alignment → (3) updated master cross-match; repeated `--n_refine` times.
- `pipeline/data_loader_master.py` — Loads `master_combined_v2.csv` for BP3M v2. HST-only sources get synthetic negative Gaia IDs, flat position priors, and Michalik+100 mas/yr PM prior (treated as `gaia_2p`).
- `pipeline/catalog_utils.py` — Gaia covariance construction, quality filtering, error inflation. `GAIA_REQUIRED_COLS` lists the 33 columns expected from a Gaia CSV.
- `pipeline/explore_utils.py` — `load_gaia_catalog()`, `load_bp3m_results()` and other notebook helpers.
- `pipeline/output.py` — `print_field_summary()`, `write_ds9_region_file()`.
- `pipeline/synthetic.py` — Generates synthetic HST observations from real cross-match data; `compare_synthetic_results()` checks recovered vs. truth PMs.
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

- `cross_match.py` — HST implementation. `process_single_image()`: 4P offset discovery (2D histogram peak) → 6P affine refinement with per-iteration empirical residual covariance floor → final match. Uses `get_hst_params()` (reads `ORIENTAT`, ACS/WFC3 pixel scales from FLC headers). Outputs `hst_*` columns.
- `cross_match_jwst.py` — JWST implementation. Same matching algorithm as `cross_match.py`. Key differences:
  - `get_jwst_params()` — reads `PA_APER` (then `ORIENTAT`, then `PA_V3` as fallback) from the SCI extension; pixel scales from `_JWST_PIXEL_SCALE` dict (NIRCam SW 0.031″, LW 0.063″, NIRISS 0.066″, MIRI 0.111″); `initial_scale=1.0` (no GDC plate-scale correction needed)
  - `find_hst_image_folders()` (modified) — walks `JWST/`, matches `{name}_cal_catalog.fits` + `*_cal.fits`; returns dicts with `cal` key instead of `flc`
  - `process_single_image(img, ...)` — takes `img` dict with keys `root`, `catalog`, `cal`; outputs `jwst_*` columns and `is_star` flag
  - HST-specific dead code removed: `_CHIP_CONFIG`, `get_chip_config`, `get_hst_params`
- `catalog_matcher.py` — nearest-neighbour matching with magnitude constraint
- `miracle_match.py` — fallback robust geometric matching via V/VMAX + SNS + progressive sigma tightening
- `diagnostics.py` — 8-panel per-image diagnostic plots

**Hard catalog column requirements** (both cross_match.py and cross_match_jwst.py raise or skip without these):

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

`psf_fitting.py` `_HST_DEFAULTS` (tuned for HST FLC images):
```python
fmin_thresh=100.0, mag_st_max=28.0, hmin=4, n_passes=2, n_discovery_passes=1,
sat_threshold=60000.0, max_iter_fit=100, half_width=3,
sky_inner=4, sky_outer=8, tol=1e-3,
sigma_clip=True, sigma_clip_sigma=4.0,
conc_limit=0.9, n_jobs=-1, backend='auto'
```

`psf_fitting_cal.py` `_JWST_DEFAULTS` (tuned for JWST CAL images):
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

**What is implemented:**
- `download_jwst.py` — full MAST search + download for NIRCam, NIRISS, MIRI `_cal.fits`; reads available STDPSFs/STDGDCs from lib_dir to filter to supported filters; writes the same obsid manifest as the HST downloader; `_check_exptime()` validates exposure times
- `jwst1pass_py_v2` (in `GaiaWebb-master/jwst1pass_py_v2/`) — JWST PSF-fitting engine, BP3M-compatible:
  - `StarRecord` carries `concentration`, `concentration_2x2`, `concentration_3x3`, `n_conc_1×1/2×2/3×3`, and `is_star_candidate` fields, matching pypass schema
  - `classify_stars()` and `_conc_adaptive_bounds()` are verbatim copies of the pypass implementations
  - `qfit` convention matches pypass: genuine quality metric (`small ≈ good`); old `qfit=0.0` sentinel removed
  - `conc_limit` threaded through `run_photometry_fits` → `_run_nircam_meta` → `run_photometry` → `classify_stars`
  - Validated on Draco NIRISS F200W: 81/515 star candidates correctly identified
- **Step 3 PSF fitting (`psf_fitting_cal.py`)** — JWST-only module (July 2026):
  - `_ensure_jwst1pass()` adds `GaiaWebb-master/jwst1pass_py_v2` to `sys.path` (respects `JWST1PASS_DIR` env var)
  - `_build_jwst_catalog_table(records, zero_point, sigma_floor_x, sigma_floor_y, eps_flux, floor_params)` — column adapter from jwst1pass schema to pypass schema: `q→qfit`, `mag_gdc→mag_st_gdc`, adds `is_star_candidate`, `chip_ext`, `eps_psf=NaN`, `sigma_*_model=NaN`, floor metadata
  - `_fit_one_image` — thin wrapper that delegates to `_fit_one_image_jwst`; accepts the same 7-tuple args as `_image_worker` for API compatibility
  - `_fit_one_image_jwst()` — JWST engine: reads `PIXAR_SR` from the FITS primary header to compute `zero_point` (`ZP_AB = -2.5 * log10(PIXAR_SR × 1e6 / 3631)`), calls `jwst1pass_py_v2`, writes catalog, params sidecar, residual FITS (per-chip SCI/VAR/MASK with DQ + sigma-clip masks), all diagnostic plots, and PSF perturbation `psf_delta.npy`
  - `estimate_systematic_floor` is called on jwst1pass records (uses covariance matrix entries, which jwst1pass computes); `fx/fy/ff` propagated to catalog and `plot_catalog_stats`
  - Image discovery uses `download_jwst.find_flc_images`; `_JWST_DEFAULTS` provides all JWST-specific parameter defaults including `mag_limit=28.0`

- **Step 4 cross-matching (`gaia_cross_match/cross_match_jwst.py`)** — JWST implementation complete (July 2026):
  - `get_jwst_params(cal_file)` reads `PA_APER`/`ORIENTAT` from SCI ext, pixel scale from `_JWST_PIXEL_SCALE`, `EXPSTART`/`MJD-BEG` from primary header; `initial_scale=1.0`
  - `find_hst_image_folders()` walks `JWST/` and matches `*_cal_catalog.fits` + `*_cal.fits` (name kept for API compatibility with pipeline's `process_single_image` call)
  - `process_single_image(img, ...)` — JWST-adapted: `img['cal']` instead of `hst['flc']`; outputs `jwst_index`, `jwst_x_gdc`, `jwst_y_gdc`, `jwst_mag_gdc`, `jwst_mag_st_gdc`, `jwst_mag_ab`, `is_star`
  - `pipeline/cross_match_jwst.py` still raises `NotImplementedError` for non-HST and imports from `gaia_cross_match.cross_match` (HST) — needs to be updated to import from `gaia_cross_match.cross_match_jwst`

**What is not yet implemented:**
- `psf_fitting.py` (HST-only) — still raises `NotImplementedError` for non-HST; use `psf_fitting_cal.py` for JWST
- `pipeline/cross_match_jwst.py` wiring — still imports from `gaia_cross_match.cross_match` and has `raise NotImplementedError` for non-HST; needs updating to use `gaia_cross_match.cross_match_jwst.process_single_image`
- `gaia_cross_match/validator.py` — HST-only; walks `HST/`, matches `*_flc.fits`/`*_flc_catalog.fits`, and expects `hst_mag_st_gdc`, `hst_index`, `hst_is_star`, `hst_mag_err_gdc` columns. JWST `matched_gaia.csv` uses `jwst_*` column names and `is_star`. Needs a JWST variant or a `telescope` parameter before `validate_target` can be called after JWST cross-matching.
- `gaia_cross_match/__init__.py` — exports `process_single_image`, `find_hst_image_folders`, `get_hst_params`, `propagate_gaia_with_cov` only from `cross_match` (HST); nothing is exported from `cross_match_jwst`
- PSF iteration (`n_psf_iter >= 2`) for JWST — `psf_delta` is not passed into `_fit_one_image_jwst`; `psf_delta.npy` is written for reference only

### Notebooks

Jupyter notebooks in `bp3m/notebooks/` (and mirrored in `notebooks/`) cover field overview, proper motions, astrometric quality, cross-match diagnostics, v2 results, and alignment posterior sampling. Install with `bp3m-notebooks`.
