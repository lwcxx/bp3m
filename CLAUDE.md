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

# Run the pipeline (entry point)
bp3m --name "Leo I" --search_radius 0.1 --output_dir ./outputs

# Download PSF/GDC library files from STScI
bp3m-setup

# Run v2 pipeline (HST-only sources)
bp3m-v2 --name "Leo I" --output_dir ./outputs
```

## Architecture

The repo is a single Python package (`bp3m`) plus two bundled subpackages (`pypass`, `gaia_cross_match`) and a top-level entry-point script (`bp3m_run.py`).

### Pipeline flow (bp3m_run.py → bp3m/pipeline/)

Each of the 5 steps maps to a module in `bp3m/pipeline/`:

| Step | Module | What it does |
|------|--------|--------------|
| 1 | `download_gaia.py` | TAP query to Gaia DR3; caches as CSV |
| 2 | `download_hst.py` | MAST search + download of FLC/FLT FITS files |
| 3 | `psf_fitting.py` | Parallelises `pypass` over images via joblib |
| 4 | `cross_match.py` | Parallelises `gaia_cross_match` over images |
| 5 | `run_alignment.py` | Calls `BP3MSolver` from `bp3m/solver.py` |

Steps can be individually skipped with `--skip_download`, `--skip_psf`, `--skip_crossmatch`, `--skip_alignment`. An obsid manifest (`{field}_selected_obsids.json`) persists image selection across runs.

### Core solver (`bp3m/solver.py`)

`BP3MSolver` is the mathematical heart. It implements the closed-form Gaussian posterior from McKinnon et al. (in prep) using Schur complement / information-form marginalisation. The joint model is:

```
x_survey_ij = X_ij @ r_j - JU_ij @ v_T,i
```

- `r_j` — 8D image transformation per image (a, b, c, d, w, z, Δα₀, Δδ₀); expands to 14/22D for `poly_order=2/3`
- `v_T,i` — 5D astrometry update per star (Δα\*, Δδ, μα\*, μδ, ϖ)

The EM loop iterates: compute residual covariances → Schur complement solve for `r_hat` → conditional solve for `v_hat` → update rotation matrices. Outlier rejection uses both MAD sigma clipping and Cook's D influence clipping.

A sparse variant lives in `solver_sparse.py` (activated by `--sparse`); `astro_utils.py` owns all coordinate/matrix helpers.

### PSF fitting subpackage (`pypass/`)

- `core.py` — PSF evaluation, sky estimation, source finding, Newton fitting
- `_jax_kernel.py` — JAX-accelerated batch fitting (`jax.vmap` + `jax.jit`); falls back to NumPy when JAX unavailable or `PYPASS_BACKEND=numpy`
- `_backend.py` — backend dispatch logic
- `multipass.py` — iterative multi-pass detection loop
- `io.py` — FITS I/O, `run_photometry_fits()` top-level API

### Cross-matching subpackage (`gaia_cross_match/`)

- `cross_match.py` — 4P (4-parameter) offset discovery via histogram peak + affine refinement with per-iteration empirical residual covariance floor
- `catalog_matcher.py` — nearest-neighbour matching with magnitude constraint
- `miracle_match.py` — fallback for difficult fields
- `diagnostics.py` — per-image diagnostic plots

### Output layout

```
{output_dir}/{field}/
  Gaia/                        ← Gaia CSV cache
  HST/mastDownload/HST/{obsid}/
    {obsid}_flc.fits            ← raw image
    {obsid}_flc_catalog.fits    ← pypass output
    matched_gaia.csv            ← cross-match output
  BP3M_results/
    stellar_astrometry.csv      ← primary science output
    image_transformations.csv
    v_cov_marginalised.npy      ← (N, 5, 5) full covariance
    plots/
```

### Two output sets in stellar_astrometry.csv

- `pmra_bp3m` / `sigma_pmra_bp3m` — **marginalised** over alignment uncertainty; stars are correlated; use for Gaia comparisons
- `pmra_bp3m_cond` / `sigma_pmra_bp3m_cond` — **conditional** (MAP alignment fixed); stars uncorrelated; use for per-star membership analyses

### Gaia star classification

Stars are typed by which Gaia solution they have, checked in `solver.py::_cache_gaia`:
- `gaia_5p` — 5-parameter solution (full PM + parallax); uses Gaia covariance directly
- `gaia_6p` — 6-parameter (pseudocolour); treated like 5p
- `gaia_2p` — 2-parameter (position only); gets a diffuse PM prior (100 mas/yr) + Michalik parallax prior

### Notebooks

Jupyter notebooks in `bp3m/notebooks/` (and mirrored in `notebooks/`) cover field overview, proper motions, astrometric quality, cross-match diagnostics, v2 results, and alignment posterior sampling. Run with `bp3m-notebooks` to install them into a target directory.
