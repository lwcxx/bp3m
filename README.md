# bp3m

bp3m is a Python pipeline for measuring improved proper motions of stars by combining multi-epoch HST imaging with Gaia DR3 astrometry. It takes a sky position or target name, automatically downloads and processes all relevant archival HST data from MAST, and simultaneously solves for the per-image HST transformations and per-star proper motions and parallaxes using a closed-form Bayesian algorithm. The result is a catalogue of stellar astrometry where every star with HST detections has significantly tighter proper motion uncertainties than Gaia alone, with the improvement scaling with the number of HST epochs and the HST-Gaia time baseline.

bp3m implements and extends the Bayesian proper motion method of McKinnon et al. (2024, ApJ 972 150), replacing the original MCMC posterior with a closed-form Gaussian solution that is analytically exact and fast enough to simultaneously fit thousands of stars across >100 HST images. The pipeline follows the science workflow of GaiaHub (del Pino et al. 2022, ApJ 933 76) and uses pypass, a Python implementation of the hst1pass photometry algorithm (Anderson 2022, WFC ISR 2022-05).

> **This is the actively developed version of bp3m and should be used in place of the original code.** The original MCMC-based implementation is archived at https://github.com/KevinMcK95/BayesianPMs. The closed-form Gaussian posterior in this version is not only faster but analytically superior — it does not suffer from MCMC convergence issues and scales to datasets that were impractical with the original code.

## Installation

```bash
pip install bp3m
```

For the full environment including PyMC (required for the Bayesian solver):

```bash
conda env create -f environment.yml
conda activate bp3m
pip install bp3m
```

bp3m bundles [pypass](pypass/README.md) (PSF-fitting photometry) and [gaia_cross_match](gaia_cross_match/README.md) (Gaia cross-matching) as internal packages — no separate installs are needed.

## Setup

After installation, run the setup command to download the required HST PSF and geometric distortion correction (GDC) library files from STScI:

```bash
bp3m-setup
```

By default the library files are stored in `~/.bp3m/lib`. To store them elsewhere (e.g. on a large-storage server), set the `BP3M_HOME` environment variable before running setup:

```bash
export BP3M_HOME=/path/to/storage/.bp3m
bp3m-setup --lib-dir /path/to/storage/bp3m_lib
```

## Quick start

```bash
bp3m --name "Leo I" --search_radius 0.1 --output_dir ./outputs
```

## Pipeline steps

1. **Download Gaia** — query Gaia DR3 via TAP and cache the result
2. **Download HST** — search MAST and download FLC/FLT images
3. **PSF fitting** — run iterative PSF photometry on each image (pypass)
4. **Cross-match** — match each HST catalog to Gaia with an affine transformation (gaia_cross_match)
5. **Bayesian alignment** — simultaneously solve for image transformations and stellar proper motions/parallaxes using the closed-form BP3M algorithm

## Key features

- Closed-form Gaussian posterior (not MCMC) — exact and scales to thousands of stars across >100 images
- Full Python pipeline from HST download through proper motion measurement
- Iterative multi-pass PSF photometry with JAX acceleration (via pypass)
- Robust Gaia cross-matching with affine transformation (via gaia_cross_match)
- Magnitude-dependent chi2 uncertainty calibration
- Diagnostic plots at every pipeline stage

## Primary output: `stellar_astrometry.csv`

The main science output is `{output_dir}/{field}/BP3M_results/stellar_astrometry.csv`. It contains one row per star and is designed as a near-drop-in replacement for the Gaia astrometric solution — the BP3M columns follow Gaia's naming convention and carry the same physical meaning, but with substantially reduced proper motion uncertainties for stars with multiple HST epochs.

**Key columns:**

| Column | Description |
|--------|-------------|
| `pmra_bp3m` | Proper motion in RA×cos(Dec) [mas/yr] — marginalised posterior mean |
| `pmdec_bp3m` | Proper motion in Dec [mas/yr] — marginalised posterior mean |
| `parallax_bp3m` | Parallax [mas] — marginalised posterior mean |
| `sigma_pmra_bp3m` | Uncertainty on pmra_bp3m [mas/yr] |
| `sigma_pmdec_bp3m` | Uncertainty on pmdec_bp3m [mas/yr] |
| `sigma_parallax_bp3m` | Uncertainty on parallax_bp3m [mas] |
| `corr_pmra_pmdec` | Correlation between pmra and pmdec |
| `corr_pmra_plx` | Correlation between pmra and parallax |
| `corr_pmdec_plx` | Correlation between pmdec and parallax |
| `delta_racosdec_bp3m` | BP3M position offset from Gaia in RA×cos(Dec) [mas] |
| `delta_dec_bp3m` | BP3M position offset from Gaia in Dec [mas] |
| `n_hst_used` | Total HST detections used (alignment + astrometry) |
| `chi2_hst_red` | Reduced HST chi2 — should be ~1 for well-fit stars |

The `_cond` variants (e.g. `pmra_bp3m_cond`) are the MAP conditional posteriors with the image transformations held fixed. These are tighter but do not account for transformation uncertainty; use the marginalised columns for science.

To use the BP3M results as a drop-in replacement for Gaia proper motions, substitute `pmra_bp3m` → `pmra`, `sigma_pmra_bp3m` → `pmra_error`, and the `corr_*` columns → the corresponding Gaia correlation columns. The full 5×5 posterior covariance is also saved as `v_cov_marginalised.npy` for downstream use.

## v2 pipeline: extending to HST-only sources

After running the standard `bp3m` pipeline, you can optionally run `bp3m-v2` to extend the analysis to HST-detected sources that have no Gaia counterpart. This is most useful for fields with deep, multi-epoch HST imaging where many faint sources are detected by HST but fall below the Gaia detection limit.

`bp3m-v2` runs a two-step post-processing pipeline:

1. **Master cross-match** — uses the BP3M transformation solution to project every PSF-fit HST source to RA/Dec with full uncertainty propagation, then cross-matches sources across images of the same filter to build a master HST catalogue (`hst_xmatch/master_combined_v2.csv`)
2. **v2 BP3M alignment** — re-runs the Bayesian alignment including the HST-only sources, using the Gaia-constrained transformation as initialisation and phasing in HST-only sources after the transformation has converged

```bash
# Step 1: run the standard pipeline
bp3m --name "Leo I" --search_radius 0.1 --output_dir ./outputs

# Step 2: run v2 post-processing
bp3m-v2 --name "Leo I" --output_dir ./outputs
```

**v2 outputs** are written to `{output_dir}/{field}/`:
- `BP3M_v2_results/stellar_astrometry.csv` — astrometry for all sources (Gaia-matched + HST-only), same column format as the standard output
- `hst_xmatch/master_combined_v2.csv` — the master HST cross-match catalogue used as input to v2 BP3M
- `hst_xmatch/master_combined.csv` — cross-filter merged HST source catalogue

## A note on systematics

Combining astrometry between two telescopes (HST and Gaia) with different passbands, pixel scales, and epochs can introduce complicated systematic errors that affect the final proper motion catalogue. Common sources of systematics include:

- **Colour-dependent PSF effects** — differential chromatic refraction or filter-dependent PSF structure can introduce position offsets that vary with stellar colour
- **Geometric distortion residuals** — imperfect GDC corrections leave small systematic position errors that vary across the detector
- **Epoch-dependent effects** — charge transfer inefficiency (CTI), focus drift, or guide star jitter can introduce time-dependent systematics

**We strongly recommend that users:**

- Examine the diagnostic plots generated in `BP3M_results/plots/` — particularly `pm_vector_diagram_detector_pos.png`, which shows whether the BP3M proper motions show unexpected trends as a function of position on the HST detector. Any coherent pattern is a warning sign of unmodelled systematics.
- Check `chi2_hst_distributions.png` to verify that per-image chi2 distributions are consistent with the expected chi2(2) distribution. Images with large median chi2 or alpha > 2 may have problematic data.
- Be cautious when interpreting small systematic PM offsets between populations (e.g. cluster vs. field), as these can be of similar magnitude to unmodelled systematics in shallow or single-epoch datasets.

Where possible, mitigate systematics by using only images with long HST-Gaia time baselines (which reduce the impact of positional errors on the PM solution), restricting to a single instrument/detector, or applying per-filter or per-epoch quality cuts using the `chi2_hst_red` and `n_hst_used` columns.

## Status and feedback

bp3m has been tested on a range of stellar fields across multiple HST instruments and epochs, but as with any research software there may be edge cases and bugs that haven't been caught yet. If you run into unexpected behaviour or incorrect results, please open a GitHub issue — all feedback is welcome.

## Development notes

Code optimization, the Python translation of supporting routines, and pipeline development were assisted by [Claude Code](https://claude.ai/code) (Anthropic).

## Citation

If you use bp3m in your research, please cite the original BP3M paper at a minimum:

> McKinnon et al. 2024, ApJ 972 150.
> https://ui.adsabs.harvard.edu/abs/2024ApJ...972..150M/abstract

A new paper describing the updates and extensions in this version of bp3m is in preparation.

If you use the full pipeline (PSF fitting and/or Gaia cross-matching), please also cite the works that these components are based on:

> Anderson, J. 2022, "One-Pass HST Photometry with hst1pass", Space Telescope WFC Instrument Science Report 2022-05.
> https://ui.adsabs.harvard.edu/abs/2022wfc..rept....5A/abstract

> del Pino, A., et al. 2022, "GaiaHub: A Method for Combining HST and Gaia to Obtain Improved Proper Motions for HST Observations", ApJ 933 76.
> https://doi.org/10.3847/1538-4357/ac71ae
