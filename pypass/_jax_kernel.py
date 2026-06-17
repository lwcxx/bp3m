"""Pre-computation helpers for the JAX fitting backend.

All functions in this module use NumPy/scipy and are called *before* any JAX
code runs.  They transform the per-star image data and spatially-blended PSF
arrays into the batched fixed-shape arrays consumed by the JAX vmap kernel.

Public API (NumPy side):
    tile_radius          — supersampled half-width of a PSF tile
    tile_side            — full width of a PSF tile
    extract_psf_tile     — slice the central tile from a blended PSF
    eval_psf_on_tile     — NumPy mirror of the in-loop JAX PSF evaluation
    extract_pixel_window — pixel values, variance, mask for one star
    flux_sky_init        — 2-parameter LS initialisation (mirrors fit_star)
    prepare_jax_inputs   — batch all stars into arrays ready for jit/vmap
"""

import numpy as np
from scipy.ndimage import map_coordinates, spline_filter


# ---------------------------------------------------------------------------
# Tile geometry  (pure functions, depend only on hw and psf_scale)
# ---------------------------------------------------------------------------

def tile_radius(hw: int, psf_scale: int) -> int:
    """Supersampled half-width of the PSF tile for a given fit window.

    Covers ±hw detector pixels (the fit window), plus psf_scale//2 margin
    for sub-pixel position updates during Newton iteration (star can drift
    up to ±0.5 detector px from the initial estimate), plus 1 extra pixel
    for central-difference derivative steps.
    """
    return hw * psf_scale + psf_scale // 2 + 1


def tile_side(hw: int, psf_scale: int) -> int:
    """Full side length of the PSF tile in supersampled pixels."""
    return 2 * tile_radius(hw, psf_scale) + 1


# ---------------------------------------------------------------------------
# PSF tile extraction
# ---------------------------------------------------------------------------

def extract_psf_tile(psf_blended: np.ndarray,
                     hw: int,
                     psf_scale: int) -> np.ndarray:
    """Extract the central tile from a spatially-blended PSF array.

    The tile is always centred on the PSF centre pixel ``(half_ss, half_ss)``,
    which is the *same* reference for every star.  Inside the JAX Newton loop,
    the tile-local coordinate for window pixel ``(dix, diy)`` at sub-pixel
    offset ``(dx, dy)`` is::

        x_tile = tile_radius(hw, psf_scale) + (dx - dix) * psf_scale
        y_tile = tile_radius(hw, psf_scale) + (dy - diy) * psf_scale

    No per-star origin bookkeeping is required.

    Parameters
    ----------
    psf_blended : (ny_psf, nx_psf) float array
        Spatially-blended (raw, unprefiltered) PSF for this star's detector
        position, returned by ``interpolate_psf`` on the *raw* PSF cube.
    hw : int
        Fit-window half-width in detector pixels.
    psf_scale : int
        PSF supersampling factor.

    Returns
    -------
    tile : (tile_side, tile_side) float32 array
    """
    ny_psf, nx_psf = psf_blended.shape
    half_ss_y = (ny_psf - 1) // 2
    half_ss_x = (nx_psf - 1) // 2
    r = tile_radius(hw, psf_scale)

    y0 = half_ss_y - r;  y1 = half_ss_y + r + 1
    x0 = half_ss_x - r;  x1 = half_ss_x + r + 1

    # Edge-pad when PSF is smaller than the required tile (non-standard models)
    if y0 < 0 or x0 < 0 or y1 > ny_psf or x1 > nx_psf:
        pad_y = (max(0, -y0), max(0, y1 - ny_psf))
        pad_x = (max(0, -x0), max(0, x1 - nx_psf))
        psf_blended = np.pad(psf_blended, [pad_y, pad_x], mode='edge')
        y0 += pad_y[0];  x0 += pad_x[0]
        y1 = y0 + 2 * r + 1;  x1 = x0 + 2 * r + 1

    return np.asarray(psf_blended[y0:y1, x0:x1], dtype=np.float32)


# ---------------------------------------------------------------------------
# Cubic B-spline helpers (NumPy) — shared by eval_psf_on_tile and the JAX kernel
# ---------------------------------------------------------------------------

def _bspline3_weights(t: np.ndarray) -> np.ndarray:
    """Cubic B-spline basis weights at fractional position t ∈ [0,1). Shape (..., 4)."""
    t2 = t * t;  t3 = t2 * t
    return np.stack([
        (1.0 - t)**3 / 6.0,
        (4.0 - 6.0*t2 + 3.0*t3) / 6.0,
        (1.0 + 3.0*t + 3.0*t2 - 3.0*t3) / 6.0,
        t3 / 6.0,
    ], axis=-1)


def _bspline3_dweights(t: np.ndarray) -> np.ndarray:
    """Derivatives of cubic B-spline basis weights w.r.t. t. Shape (..., 4)."""
    return np.stack([
        -(1.0 - t)**2 / 2.0,
        t * (3.0*t - 4.0) / 2.0,
        (1.0 + 2.0*t - 3.0*t**2) / 2.0,
        t**2 / 2.0,
    ], axis=-1)


# ---------------------------------------------------------------------------
# NumPy mirror of the JAX in-loop PSF evaluation
# ---------------------------------------------------------------------------

def eval_psf_on_tile(
    coeff_tile: np.ndarray,
    dx: float,
    dy: float,
    hw: int,
    psf_scale: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate PSF + x/y gradients for all window pixels from a B-spline coefficient tile.

    This is the NumPy reference implementation of what the JAX kernel
    executes inside each Newton iteration.  Uses cubic B-spline interpolation
    (separable 4×4 stencil with analytical gradient weights) on the
    pre-filtered coefficient tile — the exact same algorithm as the JAX kernel,
    giving < 1e-10 px agreement.

    Sign convention matches ``core.py``:
    the design-matrix column for δx is ``flux * dPdx``, where::

        d(x_tile)/d(x_star) = -psf_scale  →  dPdx = dP/d(x_tile) * (-psf_scale)

    Tile coordinate: ``x_tile = tile_radius + (dix - dx) * psf_scale``, so
    tile[tile_radius] == PSF[half_ss] (PSF centre).  Pixel dix is at +dix
    detector pixels from the star's reference pixel xi, and the PSF there is
    sampled at +dix − dx supersampled steps from the PSF centre.

    Parameters
    ----------
    coeff_tile : (tile_side, tile_side) float64 array — B-spline prefiltered tile
                 (``spline_filter(psf_blended, order=3)`` cropped to tile region)
    dx, dy : float — sub-pixel offset of star from its integer reference pixel
    hw : int — fit-window half-width in detector pixels
    psf_scale : int — PSF supersampling factor

    Returns
    -------
    P    : (n_pix,) — PSF values   [n_pix = (2*hw+1)²]
    dPdx : (n_pix,) — ∂P/∂x_det
    dPdy : (n_pix,) — ∂P/∂y_det
    """
    r = tile_radius(hw, psf_scale)
    ts = coeff_tile.shape[0]

    diy_grid, dix_grid = np.mgrid[-hw:hw + 1, -hw:hw + 1]
    dix = dix_grid.ravel().astype(np.float64)
    diy = diy_grid.ravel().astype(np.float64)

    # Tile-local supersampled coordinates (same formula as JAX kernel).
    x_t = r + (dix - dx) * psf_scale   # (n_pix,)
    y_t = r + (diy - dy) * psf_scale   # (n_pix,)

    ix = np.floor(x_t).astype(np.intp)
    iy = np.floor(y_t).astype(np.intp)
    tx = x_t - np.floor(x_t)           # fractional parts in [0, 1)
    ty = y_t - np.floor(y_t)

    kv = np.array([-1, 0, 1, 2], dtype=np.intp)

    # Gather 4×4 coefficient neighborhood per pixel: (n_pix, 4, 4)
    ix_g = np.clip(ix[:, None] + kv[None, :], 0, ts - 1)   # (n_pix, 4)
    iy_g = np.clip(iy[:, None] + kv[None, :], 0, ts - 1)
    C = coeff_tile[iy_g[:, :, None], ix_g[:, None, :]]      # (n_pix, 4, 4)

    wx  = _bspline3_weights(tx)    # (n_pix, 4)
    dwx = _bspline3_dweights(tx)
    wy  = _bspline3_weights(ty)
    dwy = _bspline3_dweights(ty)

    P      = np.einsum('pi,pij,pj->p', wy, C, wx)
    dP_dxt = np.einsum('pi,pij,pj->p', wy, C, dwx)   # dP/d(x_tile)
    dP_dyt = np.einsum('pi,pij,pj->p', dwy, C, wx)

    # Chain rule: d(x_tile)/d(x_star) = -psf_scale
    dPdx = dP_dxt * (-float(psf_scale))
    dPdy = dP_dyt * (-float(psf_scale))

    return P, dPdx, dPdy


# ---------------------------------------------------------------------------
# Pixel window extraction (single star)
# ---------------------------------------------------------------------------

def extract_pixel_window(
    data: np.ndarray,
    x0: float,
    y0: float,
    hw: int,
    mask,          # (ny, nx) bool or None
    noise_map,     # (ny, nx) float or None
    gain: float,
    read_noise: float,
    sky: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int, int]:
    """Extract fixed-shape pixel data for one star.

    The window is anchored at the *initial* rounded position ``(xi, yi)``
    and does not shift if the Newton loop updates x0 by more than 0.5 px
    (acceptable since convergence is typically within ±0.3 px).

    Parameters
    ----------
    data       : (ny, nx) science image
    x0, y0     : initial star position (0-indexed, sub-pixel)
    hw         : fit-window half-width in detector pixels
    mask       : bad-pixel mask (True = bad) or None
    noise_map  : per-pixel variance array or None
    gain       : detector gain (e-/DN); used for Poisson noise when noise_map is None
    read_noise : read noise in electrons
    sky        : sky background per pixel (used to fill out-of-bounds pixels)

    Returns
    -------
    pixel_vals    : (n_pix,) float64 — image values in window (sky-filled at edges)
    pixel_var_rn  : (n_pix,) float64 — read-noise² component of variance
                    (or full noise_map values when noise_map is provided)
    valid_mask    : (n_pix,) bool    — True = pixel is usable (in-bounds, not masked)
    dx0           : float — initial fractional x offset = x0 - xi
    dy0           : float — initial fractional y offset = y0 - yi
    xi            : int — reference integer x pixel
    yi            : int — reference integer y pixel
    """
    ny, nx = data.shape
    xi = int(round(x0));  yi = int(round(y0))
    dx0 = x0 - xi;         dy0 = y0 - yi

    # Pixel offsets in window (row-major, matching core.py _window_offsets)
    diy_grid, dix_grid = np.mgrid[-hw:hw + 1, -hw:hw + 1]
    dix = dix_grid.ravel();  diy = diy_grid.ravel()
    px = xi + dix;            py = yi + diy

    in_bounds = (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
    px_c = np.clip(px, 0, nx - 1);  py_c = np.clip(py, 0, ny - 1)

    pixel_vals = np.where(in_bounds, data[py_c, px_c], sky).astype(np.float64)

    if noise_map is not None:
        pixel_var_rn = np.where(in_bounds, noise_map[py_c, px_c], 1e6).astype(np.float64)
    else:
        rn_per_dn = read_noise / gain
        pixel_var_rn = np.full(len(dix), rn_per_dn ** 2, dtype=np.float64)

    valid = in_bounds.copy()
    if mask is not None:
        bad = mask[py_c, px_c]
        valid = valid & ~bad

    return pixel_vals, pixel_var_rn, valid, dx0, dy0, xi, yi


# ---------------------------------------------------------------------------
# Flux + sky initialisation (mirrors fit_star's 2-parameter LS pre-solve)
# ---------------------------------------------------------------------------

def flux_sky_init(
    tile: np.ndarray,
    pixel_vals: np.ndarray,
    valid_mask: np.ndarray,
    dx0: float,
    dy0: float,
    hw: int,
    psf_scale: int,
    sky_annulus: float,
) -> tuple[float, float]:
    """Initialise flux and sky via 2-parameter least-squares at fixed position.

    Mirrors the pre-solve in ``fit_star`` (core.py) that fits [flux, sky]
    with position held at ``(dx0, dy0)``.  Returns ``(flux, sky)`` with
    ``flux`` clamped to ≥ 1.0.

    Falls back to ``(max(peak-sky, 1.0), sky_annulus)`` when fewer than 2
    valid pixels are available.
    """
    P, _, _ = eval_psf_on_tile(tile, dx0, dy0, hw, psf_scale)
    g = valid_mask
    n_good = int(g.sum())

    if n_good >= 2:
        A2 = np.column_stack([P[g], np.ones(n_good)])
        fs, *_ = np.linalg.lstsq(A2, pixel_vals[g], rcond=None)
        flux = float(max(fs[0], 1.0))
        sky  = float(fs[1])
    else:
        n_pix = (2 * hw + 1) ** 2
        center_idx = n_pix // 2
        peak = float(pixel_vals[center_idx]) - sky_annulus
        flux = float(max(peak, 1.0))
        sky  = sky_annulus

    return flux, sky


# ---------------------------------------------------------------------------
# Batch preparation (all stars → arrays for jit/vmap)
# ---------------------------------------------------------------------------

def prepare_jax_inputs(
    data: np.ndarray,
    xs_stars: np.ndarray,
    ys_stars: np.ndarray,
    sky_estimates: np.ndarray,
    psf_cube_raw: np.ndarray,
    psf_xs: np.ndarray,
    psf_ys: np.ndarray,
    psf_scale: int,
    hw: int,
    mask,
    noise_map,
    gain: float,
    read_noise: float,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    psf_coeffs_cube: np.ndarray | None = None,
    restore_fluxes: np.ndarray | None = None,
    n_jobs: int = 1,
) -> dict:
    """Batch all stars into fixed-shape NumPy arrays for the JAX kernel.

    Performs spatial PSF interpolation, tile extraction, pixel window
    extraction, and flux/sky initialisation for every star.  The returned
    arrays are padded to a uniform shape so JAX can jit-compile once per
    unique ``(n_stars, n_pix, tile_side)`` combination.

    Parameters
    ----------
    data            : (ny, nx) science image
    xs_stars        : (n_stars,) x positions (0-indexed, sub-pixel)
    ys_stars        : (n_stars,) y positions
    sky_estimates   : (n_stars,) sky per pixel from annulus estimation
    psf_cube_raw    : (n_psf, ny_psf, nx_psf) *raw* (unprefiltered) PSF cube
    psf_xs, psf_ys  : PSF grid positions (for spatial interpolation)
    psf_scale       : PSF supersampling factor
    hw              : fit-window half-width in detector pixels
    mask            : bad-pixel mask (True=bad) or None
    noise_map       : per-pixel variance array or None
    gain            : detector gain
    read_noise      : read noise (electrons)
    psf_coeffs_cube : optional (n_psf, ny_psf, nx_psf) float64 array of
                      B-spline prefiltered PSF coefficients
                      (``np.array([spline_filter(p) for p in psf_cube_raw])``).
                      When provided, per-star ``spline_filter`` calls are skipped;
                      the blended coefficient tile is obtained by a second
                      ``interpolate_psf`` call on this cube, which is valid
                      because ``spline_filter`` is linear.
    x_offset      : added to image-x for PSF grid lookup (chip offset)
    y_offset      : added to image-y for PSF grid lookup
    restore_fluxes: optional (n_stars,) array of flux values to add back
                    to each star's pixel window before fitting.  Used by
                    ``refit_stars_jax``: the residual image has all stars
                    subtracted, so each star's own contribution must be
                    restored before fitting.  Zero entries are skipped.

    Returns
    -------
    dict with keys:
        psf_tiles      : (n_stars, ts, ts) float32 — ts = tile_side(hw, psf_scale)
        psf_coeff_tiles: (n_stars, ts, ts) float64 — B-spline prefiltered tiles
                         (``spline_filter`` applied to full blended PSF before crop,
                         so boundary conditions are exact — no IIR edge artifacts)
        pixel_vals    : (n_stars, n_pix) float64
        pixel_var_rn  : (n_stars, n_pix) float64
        valid_masks   : (n_stars, n_pix) bool
        dx0           : (n_stars,) float64 — initial fractional x offset
        dy0           : (n_stars,) float64 — initial fractional y offset
        flux0         : (n_stars,) float64 — initial flux from 2-param LS
        sky0          : (n_stars,) float64 — initial sky from 2-param LS
        xi            : (n_stars,) int32   — reference integer x pixel
        yi            : (n_stars,) int32   — reference integer y pixel
        hw            : int
        psf_scale     : int
        tile_radius   : int
    """
    from .core import interpolate_psf

    n_stars = len(xs_stars)
    n_pix   = (2 * hw + 1) ** 2
    ts      = tile_side(hw, psf_scale)
    tr      = tile_radius(hw, psf_scale)

    psf_tiles_out       = np.empty((n_stars, ts, ts),   dtype=np.float32)
    psf_coeff_tiles_out = np.empty((n_stars, ts, ts),   dtype=np.float64)
    pixel_vals_out      = np.empty((n_stars, n_pix),    dtype=np.float64)
    pixel_var_out       = np.empty((n_stars, n_pix),    dtype=np.float64)
    valid_out           = np.empty((n_stars, n_pix),    dtype=bool)
    dx0_out             = np.empty(n_stars,             dtype=np.float64)
    dy0_out             = np.empty(n_stars,             dtype=np.float64)
    flux0_out           = np.empty(n_stars,             dtype=np.float64)
    sky0_out            = np.empty(n_stars,             dtype=np.float64)
    xi_out              = np.empty(n_stars,             dtype=np.int32)
    yi_out              = np.empty(n_stars,             dtype=np.int32)

    def _process_star(i):
        x0  = float(xs_stars[i])
        y0  = float(ys_stars[i])
        sky = float(sky_estimates[i])

        psf_blended = interpolate_psf(
            psf_cube_raw, psf_xs, psf_ys,
            x0 + x_offset, y0 + y_offset,
        )

        tile = extract_psf_tile(psf_blended, hw, psf_scale)

        # B-spline coefficient tile.
        # Fast path: spatial interpolation on pre-filtered cube avoids per-star
        # spline_filter.  Valid because spline_filter is linear:
        #   filter(Σ wᵢ·PSFᵢ) = Σ wᵢ·filter(PSFᵢ).
        # Slow path: filter the full blended PSF directly, then crop.  Applied
        # to the full PSF (not the already-cropped tile) so the IIR filter uses
        # the correct boundary values — no edge artifacts.
        if psf_coeffs_cube is not None:
            coeffs_blended = interpolate_psf(
                psf_coeffs_cube, psf_xs, psf_ys,
                x0 + x_offset, y0 + y_offset,
            )
            _ny, _nx = coeffs_blended.shape
            _hy, _hx = (_ny - 1) // 2, (_nx - 1) // 2
            coeff_tile = np.asarray(
                coeffs_blended[_hy - tr : _hy + tr + 1, _hx - tr : _hx + tr + 1],
                dtype=np.float64,
            )
        else:
            _f64 = psf_blended.astype(np.float64)
            _coeffs = spline_filter(_f64, order=3, output=np.float64)
            _ny, _nx = _coeffs.shape
            _hy, _hx = (_ny - 1) // 2, (_nx - 1) // 2
            coeff_tile = _coeffs[_hy - tr : _hy + tr + 1,
                                  _hx - tr : _hx + tr + 1]   # (ts, ts) float64

        pv, pvar, valid, dx0, dy0, xi, yi = extract_pixel_window(
            data, x0, y0, hw, mask, noise_map, gain, read_noise, sky
        )

        # Restore this star's flux into its pixel window (refit mode only).
        # residual image has all stars subtracted; adding back flux_k lets
        # the solver see an isolated star instead of a star-shaped hole.
        rf = restore_fluxes[i] if restore_fluxes is not None else 0.0
        if rf != 0.0:
            P_restore, _, _ = eval_psf_on_tile(coeff_tile, dx0, dy0, hw, psf_scale)
            pv = pv.copy()
            pv[valid] += rf * P_restore[valid]

        flux, sky_fit = flux_sky_init(coeff_tile, pv, valid, dx0, dy0, hw, psf_scale, sky)

        return tile, coeff_tile, pv, pvar, valid, dx0, dy0, flux, sky_fit, xi, yi

    # Threading does not help here: Python-level GIL overhead in interpolate_psf
    # serializes all threads regardless of n_jobs. Run serially.
    results = [_process_star(i) for i in range(n_stars)]

    for i, (tile, coeff_tile, pv, pvar, valid, dx0, dy0, flux, sky_fit, xi, yi) in enumerate(results):
        psf_tiles_out[i]       = tile
        psf_coeff_tiles_out[i] = coeff_tile
        pixel_vals_out[i]      = pv
        pixel_var_out[i]       = pvar
        valid_out[i]           = valid
        dx0_out[i]             = dx0
        dy0_out[i]             = dy0
        flux0_out[i]           = flux
        sky0_out[i]            = sky_fit
        xi_out[i]              = xi
        yi_out[i]              = yi

    return dict(
        psf_tiles       = psf_tiles_out,
        psf_coeff_tiles = psf_coeff_tiles_out,
        pixel_vals      = pixel_vals_out,
        pixel_var_rn    = pixel_var_out,
        valid_masks     = valid_out,
        dx0             = dx0_out,
        dy0             = dy0_out,
        flux0           = flux0_out,
        sky0            = sky0_out,
        xi              = xi_out,
        yi              = yi_out,
        hw              = hw,
        psf_scale       = psf_scale,
        tile_radius     = tr,
        has_noise_map   = noise_map is not None,
    )


# ===========================================================================
# Post-JAX sigma clipping  (pure NumPy — mirrors fit_star's clipping block)
# ===========================================================================


def _sigma_clip_jax_results(
    jax_res: dict,
    inputs_dict: dict,
    gain: float,
    sigma_clip_sigma: float,
    sigma_clip_iter: int,
) -> dict:
    """Apply post-convergence sigma clipping to JAX fit results.

    Mirrors the sigma-clipping block in ``fit_star`` (core.py).  Runs in
    NumPy after ``fit_batch_jax`` returns, using the already-extracted pixel
    data from ``inputs_dict`` — no image reads required.

    For each star:
      1. Evaluate PSF + variance at the converged (dx, dy).
      2. Flag pixels with |residual|/σ > sigma_clip_sigma as outliers.
      3. Re-solve the 4-parameter system without the outliers.
      4. Reject if the clipping would shift the position by > 0.5 px.
      5. Repeat for sigma_clip_iter rounds.
      6. Recompute cov, qfit, chi2, psf_frac, central_res at the final position.

    Adds a ``clipped_masks`` key to the returned dict: ``(n_stars, n_pix)``
    bool array where True means the pixel passed the valid_mask check but was
    removed by sigma clipping.

    Parameters
    ----------
    jax_res      : dict from ``fit_batch_jax``
    inputs_dict  : dict from ``prepare_jax_inputs``
    gain         : detector gain (e-/DN)
    sigma_clip_sigma : clipping threshold in units of σ
    sigma_clip_iter  : maximum clipping rounds per star
    """
    has_noise_map = inputs_dict.get('has_noise_map', False)
    hw        = inputs_dict['hw']
    psf_scale = inputs_dict['psf_scale']
    n_pix     = (2 * hw + 1) ** 2
    center    = n_pix // 2
    n_stars   = len(jax_res['flux'])

    flux_arr        = jax_res['flux'].copy()
    dx_arr          = jax_res['dx'].copy()
    dy_arr          = jax_res['dy'].copy()
    sky_arr         = jax_res['sky'].copy()
    cov_arr         = jax_res['cov'].copy()
    qfit_arr        = jax_res['qfit'].copy()
    chi2_arr        = jax_res['chi2'].copy()
    psf_frac_arr    = jax_res['psf_frac'].copy()
    central_res_arr = jax_res['central_res'].copy()
    clipped_masks   = np.zeros((n_stars, n_pix), dtype=bool)

    for i in range(n_stars):
        tile    = inputs_dict['psf_coeff_tiles'][i]   # float64, already filtered
        pv      = inputs_dict['pixel_vals'][i]
        pvar_rn = inputs_dict['pixel_var_rn'][i]
        valid   = inputs_dict['valid_masks'][i].copy()

        flux = float(flux_arr[i])
        dx   = float(dx_arr[i])
        dy   = float(dy_arr[i])
        sky  = float(sky_arr[i])

        for _ in range(sigma_clip_iter):
            P, dPdx, dPdy = eval_psf_on_tile(tile, dx, dy, hw, psf_scale)

            if has_noise_map:
                var = np.maximum(pvar_rn, 1e-10)
            else:
                var = np.maximum(
                    (np.maximum(flux * P, 0.0) + np.maximum(sky, 0.0)) / gain
                    + pvar_rn,
                    1e-10,
                )

            r       = pv - sky - flux * P
            outlier = np.abs(r) / np.sqrt(var) > sigma_clip_sigma
            new_valid = valid & ~outlier

            if new_valid.sum() < 5 or not outlier[valid].any():
                break

            n_g = int(new_valid.sum())
            g   = new_valid
            w_g = 1.0 / var[g]
            A   = np.column_stack([P[g], flux * dPdx[g], flux * dPdy[g],
                                   np.ones(n_g)])
            AtWA = A.T @ (w_g[:, None] * A) + 1e-10 * np.eye(4)
            AtWr = A.T @ (w_g * r[g])

            try:
                delta = np.linalg.solve(AtWA, AtWr)
            except np.linalg.LinAlgError:
                valid = new_valid
                break

            # Reject position shift > 0.5 px — the outlier is likely a real
            # neighbour rather than a cosmic ray; keep the clipped mask but
            # don't move the centroid.
            if abs(delta[1]) > 0.5 or abs(delta[2]) > 0.5:
                valid = new_valid
                break

            flux = max(flux + delta[0], 1.0)
            dx  += delta[1]
            dy  += delta[2]
            sky += delta[3]
            valid = new_valid

        # --- Final evaluation at the post-clipping position ---
        P, dPdx, dPdy = eval_psf_on_tile(tile, dx, dy, hw, psf_scale)
        if has_noise_map:
            var = np.maximum(pvar_rn, 1e-10)
        else:
            var = np.maximum(
                (np.maximum(flux * P, 0.0) + np.maximum(sky, 0.0)) / gain
                + pvar_rn,
                1e-10,
            )

        g   = valid
        n_g = int(g.sum())
        w_g = 1.0 / var[g]
        r   = pv - sky - flux * P

        if n_g >= 4:
            A    = np.column_stack([P[g], flux * dPdx[g], flux * dPdy[g],
                                    np.ones(n_g)])
            AtWA = A.T @ (w_g[:, None] * A) + 1e-6 * np.eye(4)
            cov  = np.linalg.inv(AtWA)
        else:
            cov = np.eye(4) * 1e6

        sum_abs_res  = float(np.sum(np.abs(r[g])))
        sum_abs_data = float(np.sum(np.abs(pv[g] - sky)))
        qfit = sum_abs_res / max(sum_abs_data, 1e-10)
        dof  = max(n_g - 4, 1)
        chi2 = float(np.sqrt(np.sum(r[g] ** 2 / var[g]) / dof))

        psf_frac    = float(P[center])
        central_res = float(np.clip(
            (pv[center] - sky - flux * psf_frac) / max(flux, 1e-10),
            -0.999, 0.999,
        ))

        clipped_masks[i]   = inputs_dict['valid_masks'][i] & ~valid
        flux_arr[i]        = flux
        dx_arr[i]          = dx
        dy_arr[i]          = dy
        sky_arr[i]         = sky
        cov_arr[i]         = cov
        qfit_arr[i]        = qfit
        chi2_arr[i]        = chi2
        psf_frac_arr[i]    = psf_frac
        central_res_arr[i] = central_res

    return dict(
        flux          = flux_arr,
        dx            = dx_arr,
        dy            = dy_arr,
        sky           = sky_arr,
        cov           = cov_arr,
        n_iter        = jax_res['n_iter'],
        converged     = jax_res['converged'],
        delta_max     = jax_res['delta_max'],
        qfit          = qfit_arr,
        chi2          = chi2_arr,
        psf_frac      = psf_frac_arr,
        central_res   = central_res_arr,
        clipped_masks = clipped_masks,
    )


# ===========================================================================
# JAX Newton kernel  (requires jax; lazy-imported at call time)
# ===========================================================================
#
# Public entry point: fit_batch_jax(inputs_dict, gain, tol, max_iter)
#   inputs_dict  — dict from prepare_jax_inputs
#   Returns dict of result arrays, all converted to NumPy on return.
#
# Internal structure:
#   _build_jax_kernel(hw, psf_scale, has_noise_map)
#     → JIT'd vmap over _fit_one (Newton loop for a single star)
#   Results are cached in _JAX_KERNEL_CACHE keyed by (hw, psf_scale, has_noise_map).
# ===========================================================================

_JAX_KERNEL_CACHE: dict = {}


def _build_jax_kernel(hw: int, psf_scale: int, has_noise_map: bool):
    """Build and return a JIT+vmap'd Newton solver specialised for (hw, psf_scale).

    The returned function has signature::

        fn(tiles, pixel_vals, pixel_var_rn, valid_masks_f,
           dx0, dy0, flux0, sky0, gain, tol, max_iter)
        → (flux, dx, dy, sky, cov, n_iter, converged, qfit, chi2, psf_frac, central_res)

    All array args are (n_stars, ...) JAX arrays.  Scalar args (gain, tol,
    max_iter) are not batched.  The function is compiled once per unique
    (hw, psf_scale, has_noise_map) triple.
    """
    import jax
    import jax.numpy as jnp
    import jax.lax as lax

    jax.config.update("jax_enable_x64", True)

    tr_val       = tile_radius(hw, psf_scale)
    n_pix        = (2 * hw + 1) ** 2
    psf_scale_f  = float(psf_scale)
    center_idx   = n_pix // 2   # flat index of dix=0, diy=0 pixel

    # Pixel-offset grids for the fit window — constant for this hw.
    # Closed over so they become compile-time constants in the traced function.
    _diy_g, _dix_g = np.mgrid[-hw:hw + 1, -hw:hw + 1]
    _dix = jnp.array(_dix_g.ravel(), dtype=jnp.float64)
    _diy = jnp.array(_diy_g.ravel(), dtype=jnp.float64)

    # Neighbor offsets for the 4×4 B-spline stencil — compile-time constant.
    _kv = jnp.array([-1, 0, 1, 2], dtype=jnp.int32)

    def _bs3w(t):
        """Cubic B-spline weights at fractional position t. Shape (..., 4)."""
        t2 = t * t;  t3 = t2 * t
        return jnp.stack([
            (1.0 - t)**3 / 6.0,
            (4.0 - 6.0*t2 + 3.0*t3) / 6.0,
            (1.0 + 3.0*t + 3.0*t2 - 3.0*t3) / 6.0,
            t3 / 6.0,
        ], axis=-1)

    def _bs3dw(t):
        """Derivatives of cubic B-spline weights w.r.t. t. Shape (..., 4)."""
        return jnp.stack([
            -(1.0 - t)**2 / 2.0,
            t * (3.0*t - 4.0) / 2.0,
            (1.0 + 2.0*t - 3.0*t**2) / 2.0,
            t**2 / 2.0,
        ], axis=-1)

    def _eval_psf(coeff_tile, dx, dy):
        """Cubic B-spline PSF + analytical x/y detector-space gradients.

        Uses the same 4×4 separable stencil as the NumPy eval_psf_on_tile,
        giving < 1e-10 px agreement with the NumPy reference.
        """
        x_t = tr_val + (_dix - dx) * psf_scale_f   # (n_pix,)
        y_t = tr_val + (_diy - dy) * psf_scale_f

        ix = jnp.floor(x_t).astype(jnp.int32)      # (n_pix,)
        iy = jnp.floor(y_t).astype(jnp.int32)
        tx = x_t - jnp.floor(x_t)                  # fractional parts in [0, 1)
        ty = y_t - jnp.floor(y_t)

        ts = coeff_tile.shape[0]

        # Gather 4×4 coefficient neighborhood for all pixels: (n_pix, 4, 4)
        ix_g = jnp.clip(ix[:, None] + _kv[None, :], 0, ts - 1)  # (n_pix, 4)
        iy_g = jnp.clip(iy[:, None] + _kv[None, :], 0, ts - 1)
        C = coeff_tile[iy_g[:, :, None], ix_g[:, None, :]]       # (n_pix, 4, 4)

        wx  = _bs3w(tx);   dwx = _bs3dw(tx)   # (n_pix, 4)
        wy  = _bs3w(ty);   dwy = _bs3dw(ty)

        # Separable evaluation
        P      = jnp.einsum('pi,pij,pj->p', wy,  C, wx)
        dP_dxt = jnp.einsum('pi,pij,pj->p', wy,  C, dwx)
        dP_dyt = jnp.einsum('pi,pij,pj->p', dwy, C, wx)

        # Chain rule: d(x_tile)/d(x_star) = -psf_scale
        dPdx = dP_dxt * (-psf_scale_f)
        dPdy = dP_dyt * (-psf_scale_f)
        return P, dPdx, dPdy

    def _atwa_and_atwr(A0, A1, A2, A3, w, r):
        """Build 4×4 weighted normal-equation matrix and 4-vector RHS.

        A = stack([A0,A1,A2,A3])  shape (4, n_pix)
        AtWA = A @ diag(w) @ A.T = (A*w) @ A.T   — one (4,4) matmul
        AtWr = A @ diag(w) @ r   = (A*w) @ r     — one (4,) matvec
        Replaces 20 scalar dot products with 2 BLAS calls.
        """
        A = jnp.stack([A0, A1, A2, A3], axis=0)  # (4, n_pix)
        Aw = A * w                                 # (4, n_pix)
        AtWA = Aw @ A.T                            # (4, 4)
        AtWr = Aw @ r                              # (4,)
        return AtWA, AtWr

    def _fit_one(coeff_tile, pixel_vals, pixel_var_rn, valid_f,
                 dx0, dy0, flux0, sky0, gain, tol_val, max_iter_int):
        """Single-star Newton solver. Used inside vmap — no Python control flow."""

        def _pixel_var(flux, P, sky):
            if has_noise_map:
                return jnp.maximum(pixel_var_rn, 1e-10)
            return jnp.maximum(
                (jnp.maximum(flux * P, 0.0) + jnp.maximum(sky, 0.0)) / gain
                + pixel_var_rn,
                1e-10,
            )

        # ----- Newton loop -----
        def cond_fn(state):
            _, _, _, _, n_iter, converged = state[:6]
            return (n_iter < max_iter_int) & ~converged

        def body_fn(state):
            flux, dx, dy, sky, n_iter, converged, _dm = state

            P, dPdx, dPdy = _eval_psf(coeff_tile, dx, dy)
            var  = _pixel_var(flux, P, sky)
            w    = jnp.where(valid_f > 0.5, 1.0 / var, 0.0)
            r    = pixel_vals - sky - flux * P

            A0 = P
            A1 = flux * dPdx
            A2 = flux * dPdy
            A3 = jnp.ones(n_pix, dtype=jnp.float64)
            # Ridge regularisation prevents singular AtWA when pixels are all
            # masked (w=0) or the fit window is degenerate.
            AtWA, AtWr = _atwa_and_atwr(A0, A1, A2, A3, w, r)
            AtWA = AtWA + 1e-6 * jnp.eye(4)
            delta = jnp.linalg.solve(AtWA, AtWr)

            # Flux-step clamp: prevent flux from dropping > 75 % in one step.
            flux_floor = jnp.maximum(0.25 * flux, 1.0)
            delta = delta.at[0].set(jnp.maximum(delta[0], flux_floor - flux))

            # Position-divergence guard: zero the step if |δ| > 2·hw px.
            pos_bad = (jnp.abs(delta[1]) > 2.0 * hw) | (jnp.abs(delta[2]) > 2.0 * hw)
            delta   = jnp.where(pos_bad, jnp.zeros(4, dtype=jnp.float64), delta)

            flux_new = jnp.maximum(flux + delta[0], 1.0)
            dx_new   = dx  + delta[1]
            dy_new   = dy  + delta[2]
            sky_new  = sky + delta[3]

            dm   = jnp.maximum(jnp.abs(delta[1]), jnp.abs(delta[2]))
            conv = dm < tol_val
            return (flux_new, dx_new, dy_new, sky_new,
                    n_iter + jnp.int32(1), conv, dm)

        init_state = (
            jnp.float64(flux0), jnp.float64(dx0),
            jnp.float64(dy0),   jnp.float64(sky0),
            jnp.int32(0),       jnp.bool_(False),
            jnp.float64(1e9),
        )
        flux, dx, dy, sky, n_iter, converged, _dm = lax.while_loop(
            cond_fn, body_fn, init_state
        )

        # ----- Final evaluation: covariance, qfit, chi² -----
        P, dPdx, dPdy = _eval_psf(coeff_tile, dx, dy)
        var = _pixel_var(flux, P, sky)
        w   = jnp.where(valid_f > 0.5, 1.0 / var, 0.0)
        r   = pixel_vals - sky - flux * P

        A0 = P
        A1 = flux * dPdx
        A2 = flux * dPdy
        A3 = jnp.ones(n_pix, dtype=jnp.float64)
        AtWA, _ = _atwa_and_atwr(A0, A1, A2, A3, w, r)
        AtWA    = AtWA + 1e-6 * jnp.eye(4)
        cov     = jnp.linalg.inv(AtWA)

        n_good       = jnp.sum(valid_f)
        valid_bool   = valid_f > 0.5
        sum_abs_res  = jnp.sum(jnp.abs(r)             * valid_bool)
        sum_abs_data = jnp.sum(jnp.abs(pixel_vals - sky) * valid_bool)
        qfit  = sum_abs_res / jnp.maximum(sum_abs_data, 1e-10)
        dof   = jnp.maximum(n_good - 4.0, 1.0)
        chi2  = jnp.sqrt(jnp.sum(r ** 2 / var * valid_bool) / dof)

        psf_frac    = P[center_idx]
        central_res = jnp.clip(
            (pixel_vals[center_idx] - sky - flux * psf_frac)
            / jnp.maximum(flux, 1e-10),
            -0.999, 0.999,
        )

        return (flux, dx, dy, sky, cov, n_iter, converged, _dm,
                qfit, chi2, psf_frac, central_res)

    n_devices = len(jax.devices())
    _vmapped = jax.vmap(_fit_one, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, None, None))
    if n_devices > 1:
        # pmap distributes shards across virtual CPU devices (each device gets its
        # own OS thread and a portion of the XLA thread pool), then vmap maps over
        # stars within each shard. Inputs must be pre-shaped (n_devices, n_per_device, ...).
        _batched = jax.pmap(_vmapped, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, None, None))
    else:
        _batched = jax.jit(_vmapped)
    return _batched


def fit_batch_jax(
    inputs_dict: dict,
    gain: float,
    tol: float,
    max_iter: int,
) -> dict:
    """Run the JAX Newton solver on all stars in *inputs_dict*.

    Compiles the kernel on the first call for each unique
    ``(hw, psf_scale, has_noise_map)`` configuration; subsequent calls with the
    same configuration reuse the cached compiled function.

    Parameters
    ----------
    inputs_dict  : dict from ``prepare_jax_inputs``
    gain         : detector gain (e-/DN)
    tol          : position convergence tolerance (detector pixels)
    max_iter     : maximum Newton iterations per star

    Returns
    -------
    dict with NumPy arrays:
        flux, dx, dy, sky  : (n_stars,) float64 — fitted parameters
            ``x_final = xi + dx``, ``y_final = yi + dy``
        cov                : (n_stars, 4, 4) float64 — (flux,x,y,sky) covariance
        n_iter             : (n_stars,) int32
        converged          : (n_stars,) bool
        delta_max          : (n_stars,) float64 — max(|δx|,|δy|) at final step
        qfit               : (n_stars,) float64 — Σ|res|/Σ|data−sky|
        chi2               : (n_stars,) float64 — sqrt(Σ(r²/σ²)/DOF)
        psf_frac           : (n_stars,) float64 — PSF value at center pixel
        central_res        : (n_stars,) float64 — normalised central residual
    """
    import jax
    import jax.numpy as jnp

    hw        = inputs_dict['hw']
    psf_scale = inputs_dict['psf_scale']
    has_nm    = inputs_dict.get('has_noise_map', False)
    n_stars   = len(inputs_dict['dx0'])
    n_devices = len(jax.devices())

    cache_key = (hw, psf_scale, has_nm, n_devices)
    if cache_key not in _JAX_KERNEL_CACHE:
        _JAX_KERNEL_CACHE[cache_key] = _build_jax_kernel(hw, psf_scale, has_nm)
    _fn = _JAX_KERNEL_CACHE[cache_key]

    if n_devices > 1:
        # Pad n_stars to the next multiple of n_devices, then reshape to
        # (n_devices, n_per_device, ...) for pmap.
        pad = (-n_stars) % n_devices
        n_padded = n_stars + pad
        n_per = n_padded // n_devices

        def _prep(arr, dtype, fill=0.0):
            a = jnp.array(arr, dtype=dtype)
            if pad:
                a = jnp.concatenate(
                    [a, jnp.full((pad,) + a.shape[1:], fill, dtype=dtype)]
                )
            return a.reshape((n_devices, n_per) + a.shape[1:])

        tiles = _prep(inputs_dict['psf_coeff_tiles'], jnp.float64)
        pvals = _prep(inputs_dict['pixel_vals'],      jnp.float64)
        pvar  = _prep(inputs_dict['pixel_var_rn'],    jnp.float64)
        vmask = _prep(inputs_dict['valid_masks'],     jnp.float64)
        dx0   = _prep(inputs_dict['dx0'],             jnp.float64)
        dy0   = _prep(inputs_dict['dy0'],             jnp.float64)
        flux0 = _prep(inputs_dict['flux0'],           jnp.float64)
        sky0  = _prep(inputs_dict['sky0'],            jnp.float64)

        result = _fn(tiles, pvals, pvar, vmask, dx0, dy0, flux0, sky0,
                     float(gain), float(tol), int(max_iter))

        def _trim(a):
            return np.asarray(a.reshape((-1,) + a.shape[2:])[:n_stars])
    else:
        tiles  = jnp.array(inputs_dict['psf_coeff_tiles'], dtype=jnp.float64)
        pvals  = jnp.array(inputs_dict['pixel_vals'],      dtype=jnp.float64)
        pvar   = jnp.array(inputs_dict['pixel_var_rn'],    dtype=jnp.float64)
        vmask  = jnp.array(inputs_dict['valid_masks'],     dtype=jnp.float64)
        dx0    = jnp.array(inputs_dict['dx0'],             dtype=jnp.float64)
        dy0    = jnp.array(inputs_dict['dy0'],             dtype=jnp.float64)
        flux0  = jnp.array(inputs_dict['flux0'],           dtype=jnp.float64)
        sky0   = jnp.array(inputs_dict['sky0'],            dtype=jnp.float64)

        result = _fn(tiles, pvals, pvar, vmask, dx0, dy0, flux0, sky0,
                     float(gain), float(tol), int(max_iter))

        def _trim(a):
            return np.asarray(a)

    (flux, dx, dy, sky, cov, n_iter, converged, delta_max,
     qfit, chi2, psf_frac, central_res) = result

    return dict(
        flux        = _trim(flux),
        dx          = _trim(dx),
        dy          = _trim(dy),
        sky         = _trim(sky),
        cov         = _trim(cov),
        n_iter      = _trim(n_iter),
        converged   = _trim(converged),
        delta_max   = _trim(delta_max),
        qfit        = _trim(qfit),
        chi2        = _trim(chi2),
        psf_frac    = _trim(psf_frac),
        central_res = _trim(central_res),
    )
