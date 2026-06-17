"""Core PSF evaluation, sky estimation, source finding, and linear PSF fitter."""

import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter, spline_filter
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Numba bicubic B-spline kernel (optional fast path)
# ---------------------------------------------------------------------------
# Replaces 5 separate scipy.ndimage.map_coordinates calls (each ~250 µs on a
# 25-point window) with a single compiled loop (~2 µs for all 5 evaluations).
# With cache=True the JIT result is stored on disk — subsequent runs load it
# in <100 ms with no recompilation.
# Falls back to scipy with prefilter=False (~22 µs) when numba is absent.

try:
    import numba as nb
    _NUMBA = True

    @nb.njit(cache=True, fastmath=True)
    def _bspline3(t):
        """Cubic B-spline basis function."""
        t = abs(t)
        if t < 1.0:
            return 2.0/3.0 + (0.5*t - 1.0)*t*t
        elif t < 2.0:
            s = 2.0 - t
            return s*s*s / 6.0
        return 0.0

    @nb.njit(cache=True, fastmath=True)
    def _bicubic_nearest(c, y, x):
        """Evaluate a 2-D cubic B-spline at (y, x), nearest boundary."""
        ny, nx = c.shape
        iy = int(np.floor(y)); ix = int(np.floor(x))
        v = 0.0
        for j in range(iy - 1, iy + 3):
            jj = 0 if j < 0 else (ny - 1 if j >= ny else j)
            wy = _bspline3(y - j)
            s = 0.0
            for i in range(ix - 1, ix + 3):
                ii = 0 if i < 0 else (nx - 1 if i >= nx else i)
                s += c[jj, ii] * _bspline3(x - i)
            v += wy * s
        return v

    @nb.njit(cache=True, fastmath=True)
    def _nb_eval_psf_grad(c, ys, xs, scale, P, dPdx, dPdy):
        """Evaluate PSF value + x/y gradients at all pixel positions in one pass."""
        n = len(ys)
        for k in range(n):
            y, x = ys[k], xs[k]
            P[k]    = _bicubic_nearest(c, y,       x)
            Pxp     = _bicubic_nearest(c, y,       x + 1.0)
            Pxm     = _bicubic_nearest(c, y,       x - 1.0)
            Pyp     = _bicubic_nearest(c, y + 1.0, x)
            Pym     = _bicubic_nearest(c, y - 1.0, x)
            dPdx[k] = (Pxm - Pxp) * 0.5 * scale
            dPdy[k] = (Pym - Pyp) * 0.5 * scale

except ImportError:
    _NUMBA = False


def _eval_psf_grad_fast(coeffs, dx, dy, dix, diy, psf_scale):
    """Evaluate PSF + gradients from pre-filtered B-spline coefficients.

    Chooses the Numba path when available, otherwise falls back to
    scipy.ndimage.map_coordinates with prefilter=False.
    """
    size = coeffs.shape[0]
    half_ss = size // 2

    dix_bc, diy_bc = np.broadcast_arrays(np.asarray(dix), np.asarray(diy))
    shape = dix_bc.shape
    xs_flat = (half_ss + (dix_bc.ravel() - dx) * psf_scale).astype(np.float64)
    ys_flat = (half_ss + (diy_bc.ravel() - dy) * psf_scale).astype(np.float64)

    if _NUMBA:
        n = len(xs_flat)
        P_f    = np.empty(n, dtype=np.float64)
        dPdx_f = np.empty(n, dtype=np.float64)
        dPdy_f = np.empty(n, dtype=np.float64)
        _nb_eval_psf_grad(coeffs, ys_flat, xs_flat, float(psf_scale),
                          P_f, dPdx_f, dPdy_f)
        return P_f.reshape(shape), dPdx_f.reshape(shape), dPdy_f.reshape(shape)
    else:
        kw = dict(order=3, mode='nearest', prefilter=False)
        P    = map_coordinates(coeffs, [ys_flat, xs_flat    ], **kw).reshape(shape)
        Pxp  = map_coordinates(coeffs, [ys_flat, xs_flat + 1], **kw).reshape(shape)
        Pxm  = map_coordinates(coeffs, [ys_flat, xs_flat - 1], **kw).reshape(shape)
        Pyp  = map_coordinates(coeffs, [ys_flat + 1, xs_flat], **kw).reshape(shape)
        Pym  = map_coordinates(coeffs, [ys_flat - 1, xs_flat], **kw).reshape(shape)
        return P, (Pxm - Pxp) * 0.5 * psf_scale, (Pym - Pyp) * 0.5 * psf_scale


@dataclass
class StarRecord:
    x: float
    y: float
    flux: float
    flux_err: float
    sky: float
    sky_err: float
    mag: float
    mag_err: float
    qfit: float         # Σ|res|/Σ|data-sky|  (0=perfect, >0.5=poor; Fortran 'q')
    chi2: float         # sqrt(Σ(r²/var)/(n_good-4)); ~1 for good fit (Fortran 'c')
    central_res: float  # (data_cen - sky - flux·P_cen)/flux   (Fortran 'C')
    n_sat: int          # pixels in fit window above sat_threshold (Fortran 'n')
    psf_frac: float     # PSF value at star's fitted position (Fortran 'f')
    psf_peak: float     # PSF value at perfect center (Fortran 'F')
    peak: float
    cov: np.ndarray       # 4×4 in (flux, x, y, sky) order
    pass_number: int
    # Neighbor statistics (filled by compute_neighbor_stats after all passes)
    n_neighbors: int          # other detected stars with dist < hw px
    dist_nearest: float       # px to nearest other detected star
    dist_nearest_brighter: float  # px to nearest detected star with higher flux
    # Convergence diagnostics (filled by fit_star)
    n_iter: int = field(default=0)         # Newton iterations taken
    converged: bool = field(default=True)  # False if hit max_iter without convergence
    delta_max: float = field(default=0.0)  # max(|δx|,|δy|) at final Newton step
    # Sigma-clipping mask (True = pixel was clipped); None if clipping disabled.
    # Shape matches the final fit window (2hw+1)×(2hw+1), clipped to image boundary.
    clipped_mask: object = field(default=None, compare=False, repr=False)
    # Chi²-scaling applied to covariance (post-fit, in run_photometry).
    # chi2_scale = max(chi2_individual, chi2_global_floor).
    # cov, flux_err, sky_err, mag_err are all already multiplied by chi2_scale (or chi2_scale²).
    chi2_scale: float = field(default=1.0)
    # Per-star implied fractional PSF model error:
    #   eps_psf = chi2 / sqrt(flux * psf_frac * gain)
    # Bright stars with chi2 >> 1 have large eps_psf; ideal chi2 ≈ 1 gives small eps_psf.
    eps_psf: float = field(default=0.0)
    # Concentration parameters: observed pixel sum / (flux * PSF model sum) for boxes
    # of increasing size centred on the fitted position.  Stars ≈ 1.0 for all three;
    # cosmic rays > 1 (too sharp); extended sources (galaxies) deviate from 1.
    # NaN for failed records or when psf_frac ≤ 0.
    #   concentration    — 1×1: single peak pixel
    #   concentration_2x2 — 2×2: four pixels bracketing fitted (x,y)
    #   concentration_3x3 — 3×3: nine pixels centred on rounded position
    concentration: float = field(default=np.nan)
    concentration_2x2: float = field(default=np.nan)
    concentration_3x3: float = field(default=np.nan)
    # Number of unmasked (not DQ-flagged, not sigma-clipped) pixels used in each
    # concentration calculation.  0 means the metric is NaN (all pixels masked).
    n_conc_1x1: int = field(default=0)
    n_conc_2x2: int = field(default=0)
    n_conc_3x3: int = field(default=0)
    # Star/galaxy classification flag set by classify_stars() after all passes.
    # True = morphology and fit quality consistent with a point source (star).
    is_star_candidate: bool = field(default=True)
    # DQ flag summary (bitwise OR of raw DQ integer values) computed post-fit.
    # dq_1x1  — single pixel at the fitted (x,y) position
    # dq_2x2  — 2×2 region centred on fitted position
    # dq_3x3  — 3×3 region centred on fitted position
    # 0 means no DQ flags; non-zero values encode which flag types were present.
    dq_1x1: int = field(default=0)
    dq_2x2: int = field(default=0)
    dq_3x3: int = field(default=0)


# ---------------------------------------------------------------------------
# Sky estimation — single star (used during fitting)
# ---------------------------------------------------------------------------

def estimate_sky(data, ix, iy, sky_inner, sky_outer, mask=None):
    """Sigma-clipped median sky from an annulus around integer pixel (ix, iy)."""
    ny, nx = data.shape
    y0 = max(0, iy - sky_outer)
    y1 = min(ny, iy + sky_outer + 1)
    x0 = max(0, ix - sky_outer)
    x1 = min(nx, ix + sky_outer + 1)

    sub = data[y0:y1, x0:x1]
    dy = np.arange(y0 - iy, y1 - iy)
    dx = np.arange(x0 - ix, x1 - ix)
    DY, DX = np.meshgrid(dy, dx, indexing='ij')
    r2 = DY**2 + DX**2

    in_ann = (r2 >= sky_inner**2) & (r2 <= sky_outer**2)
    if mask is not None:
        in_ann &= ~mask[y0:y1, x0:x1]

    pixels = sub[in_ann].astype(np.float64)
    if len(pixels) < 3:
        return float(np.median(sub)), 1.0

    for _ in range(3):
        med = np.median(pixels)
        mad = np.median(np.abs(pixels - med))
        sigma = mad * 1.4826
        if sigma <= 0.0:
            break
        keep = np.abs(pixels - med) <= 3.0 * sigma
        if keep.sum() < 3:
            break
        pixels = pixels[keep]

    sky = float(np.median(pixels))
    sky_sigma = float(np.std(pixels)) if len(pixels) > 1 else 1.0
    return sky, max(sky_sigma, 1e-10)


# ---------------------------------------------------------------------------
# Sky estimation — vectorised over many candidates (used in find_sources)
# ---------------------------------------------------------------------------

def _build_annulus_offsets(sky_inner, sky_outer):
    """Return (iy_off, ix_off) 1D arrays for pixels in the sky annulus."""
    hw = sky_outer
    dy = np.arange(-hw, hw + 1)
    dx = np.arange(-hw, hw + 1)
    DY, DX = np.meshgrid(dy, dx, indexing='ij')
    r2 = DY**2 + DX**2
    ann = (r2 >= sky_inner**2) & (r2 <= sky_outer**2)
    iy_off, ix_off = np.where(ann)
    return (iy_off - hw).astype(np.int32), (ix_off - hw).astype(np.int32)


def estimate_sky_batch(data, ix_arr, iy_arr, sky_inner, sky_outer, mask=None):
    """Vectorised sigma-clipped median sky for many candidates simultaneously.

    Extracts the sky annulus pixels for all candidates in one numpy fancy-index
    operation, then sigma-clips row-wise (no Python loop over stars).

    Parameters
    ----------
    data    : 2D float64 image
    ix_arr  : integer 1D array of candidate x positions
    iy_arr  : integer 1D array of candidate y positions

    Returns
    -------
    sky_vals : 1D float64 array, sky per candidate
    sky_sigs : 1D float64 array, sky sigma per candidate
    """
    ny, nx = data.shape
    iy_off, ix_off = _build_annulus_offsets(sky_inner, sky_outer)

    # (n_cand, n_ann) index arrays — clipped to image boundaries
    y_pix = np.clip(iy_arr[:, None] + iy_off[None, :], 0, ny - 1)  # (n, n_ann)
    x_pix = np.clip(ix_arr[:, None] + ix_off[None, :], 0, nx - 1)

    # Extract all annulus pixels at once
    pixels = data[y_pix, x_pix].astype(np.float64)     # (n_cand, n_ann)

    if mask is not None:
        pixels[mask[y_pix, x_pix]] = np.nan

    # Vectorised sigma-clipping (3 rounds, no Python loop over candidates)
    for _ in range(3):
        med = np.nanmedian(pixels, axis=1)              # (n_cand,)
        dev = np.abs(pixels - med[:, None])
        mad = np.nanmedian(dev, axis=1)
        sigma = mad * 1.4826
        pixels[dev > 3.0 * sigma[:, None]] = np.nan

    sky_vals = np.nanmedian(pixels, axis=1)
    sky_sigs = np.nanstd(pixels, axis=1, ddof=0)

    # Fallback for candidates where all annulus pixels were masked
    global_sky = float(np.nanmedian(data))
    global_sig = float(np.nanstd(data) * 0.05)  # very rough
    bad = ~np.isfinite(sky_vals)
    sky_vals[bad] = global_sky
    sky_sigs[bad] = global_sig

    return sky_vals, np.maximum(sky_sigs, 1e-10)


# ---------------------------------------------------------------------------
# PSF interpolation — bicubic Catmull-Rom
# ---------------------------------------------------------------------------

def _catmull_rom_weights(t):
    """Catmull-Rom cubic spline weights for fractional position t in [0, 1].

    Returns weights [w_{i-1}, w_i, w_{i+1}, w_{i+2}] summing to 1.
    Gives C1-continuous interpolation with zero second derivative at knots.
    """
    t2, t3 = t * t, t * t * t
    return np.array([
        -0.5*t3 + t2       - 0.5*t,
         1.5*t3 - 2.5*t2            + 1.0,
        -1.5*t3 + 2.0*t2   + 0.5*t,
         0.5*t3 - 0.5*t2,
    ])


def interpolate_psf(psf_cube, xs, ys, x_det, y_det, _cache=None):
    """Bicubic (Catmull-Rom) interpolation among PSF grid models.

    xs, ys : 1D float64 arrays of PSF grid detector coordinates.
    PSF cube index k = iy_g * nx_g + ix_g.

    Uses np.interp to map (x_det, y_det) to fractional grid indices —
    this handles non-uniform grids correctly for the bilinear term; the
    cubic cross-terms assume locally uniform spacing, which is an
    excellent approximation for STDPSF grids (< 1% spacing variation).

    Falls back to bilinear for 1×n or n×1 grids.

    _cache : dict or None
        Optional per-run cache dict.  Keys are (x//5, y//5); PSF variation
        over a 5 px cell is < 0.5 % on typical HST grids — negligible for
        photometry.  Pass the same dict for every fit_star call in a single
        run_photometry to avoid redundant bicubic contractions in crowded
        fields.  Limited to 2 048 entries so memory stays bounded.
    """
    n_psf = psf_cube.shape[0]
    if n_psf == 1:
        return psf_cube[0]

    if _cache is not None:
        key = (int(x_det) // 5, int(y_det) // 5)
        hit = _cache.get(key)
        if hit is not None:
            return hit

    nx_g = len(xs)
    ny_g = len(ys)

    if nx_g < 2 or ny_g < 2:
        return psf_cube[0]

    # Fractional grid coordinates — np.interp clamps and handles non-uniform
    gx = np.arange(nx_g, dtype=np.float64)
    gy = np.arange(ny_g, dtype=np.float64)
    tx_full = float(np.interp(x_det, xs, gx))
    ty_full = float(np.interp(y_det, ys, gy))

    # Integer cell index, clamped so ix+1 and iy+1 are always in-bounds
    ix = int(min(np.floor(tx_full), nx_g - 2))
    iy = int(min(np.floor(ty_full), ny_g - 2))
    tx = tx_full - ix
    ty = ty_full - iy

    wx = _catmull_rom_weights(tx)
    wy = _catmull_rom_weights(ty)

    # Gather the 4×4 stencil of PSF arrays (clamped at boundaries)
    iy_idx = np.clip(np.arange(iy - 1, iy + 3), 0, ny_g - 1)
    ix_idx = np.clip(np.arange(ix - 1, ix + 3), 0, nx_g - 1)
    k_idx = (iy_idx[:, np.newaxis] * nx_g + ix_idx[np.newaxis, :]).ravel()  # 16

    # psf_cube[k_idx] shape: (16, ny_psf, nx_psf) → (4, 4, ny_psf, nx_psf)
    psf_block = psf_cube[k_idx].reshape(4, 4, *psf_cube.shape[1:])
    weights = np.outer(wy, wx)  # (4, 4)

    # Weighted sum over the 4×4 stencil
    result = np.einsum('ij,ij...->...', weights, psf_block)

    if _cache is not None and len(_cache) < 2048:
        _cache[key] = result
    return result


# ---------------------------------------------------------------------------
# PSF + gradient evaluation
# ---------------------------------------------------------------------------

def eval_psf_and_grad(psf, dx, dy, dix, diy, psf_scale):
    """Evaluate PSF and x/y detector-pixel gradients over a pixel window.

    psf       : 2D supersampled PSF array, shape (size, size)
    dx, dy    : fractional offset of star from its integer pixel (xi, yi)
    dix, diy  : integer pixel offsets from (xi, yi), broadcast-compatible arrays
    psf_scale : supersampling factor

    Returns P, dPdx, dPdy each with shape np.broadcast(dix, diy).shape.
    Pre-filters *psf* to cubic B-spline coefficients then delegates to the
    fast Numba or scipy path in ``_eval_psf_grad_fast``.
    """
    coeffs = spline_filter(psf, order=3, output=np.float64)
    return _eval_psf_grad_fast(coeffs, dx, dy, dix, diy, psf_scale)


# ---------------------------------------------------------------------------
# Single-star fitter
# ---------------------------------------------------------------------------

def _failed_record(x0, y0, flux, sky, peak, pass_number, zero_point):
    from .utils import mag_from_flux
    mag, mag_err = mag_from_flux(max(flux, 1e-10), np.inf, zero_point)
    return StarRecord(
        x=float(x0), y=float(y0), flux=float(flux), flux_err=np.inf,
        sky=float(sky), sky_err=np.inf,
        mag=float(mag), mag_err=np.inf,
        qfit=9.99, chi2=np.inf, central_res=0.0,
        n_sat=0, psf_frac=0.0, psf_peak=0.0,
        peak=float(peak), cov=np.eye(4) * 1e6, pass_number=pass_number,
        n_neighbors=0, dist_nearest=np.inf, dist_nearest_brighter=np.inf,
        n_iter=0, converged=False,
        delta_max=0.0, clipped_mask=None, chi2_scale=1.0,
        concentration=np.nan, concentration_2x2=np.nan, concentration_3x3=np.nan,
        n_conc_1x1=0, n_conc_2x2=0, n_conc_3x3=0,
        is_star_candidate=False,
    )


def _jax_results_to_records(
    jax_res: dict,
    inputs_dict: dict,
    pass_number: int,
    gain: float,
    zero_point: float,
    sat_threshold: float,
) -> list:
    """Convert fit_batch_jax output + prepare_jax_inputs dict into StarRecords.

    All per-star fields that the NumPy path derives inside fit_star are
    computed here from the JAX result arrays and the pre-extracted pixel data.
    Fields filled later by the run_photometry driver (chi2_scale via
    inflate_chi2, neighbour stats via compute_neighbor_stats) are set to
    their default values here.
    """
    from .utils import mag_from_flux

    hw        = inputs_dict['hw']
    tr        = inputs_dict['tile_radius']
    n_pix     = (2 * hw + 1) ** 2
    center    = n_pix // 2

    flux_arr   = jax_res['flux']
    sky_arr    = jax_res['sky']
    pixel_vals = inputs_dict['pixel_vals']   # (n_stars, n_pix)

    # psf_peak: PSF value at perfect centre (dx=0, dy=0).
    # tile[tr, tr] is the exact integer grid point — no interpolation needed.
    psf_peaks = inputs_dict['psf_tiles'][:, tr, tr].astype(np.float64)

    # peak: central pixel above sky (uses pre-fit pixel values — same as NumPy path)
    peaks = pixel_vals[:, center] - sky_arr

    # n_sat: pixels above sat_threshold in the fit window
    n_sat_arr = np.sum(pixel_vals > sat_threshold, axis=1)

    from scipy.ndimage import map_coordinates as _mc

    # Pre-extract PSF coefficient tiles for multi-box concentration evaluation.
    _psf_coeff_tiles = inputs_dict.get('psf_coeff_tiles', None)
    _nx_win  = 2 * hw + 1
    psf_scale = int(inputs_dict.get('psf_scale', 4))

    records = []
    for i in range(len(flux_arr)):
        flux  = float(flux_arr[i])
        sky   = float(sky_arr[i])
        cov   = jax_res['cov'][i].copy()
        chi2  = float(jax_res['chi2'][i])
        psf_frac = float(jax_res['psf_frac'][i])

        dx_i = float(jax_res['dx'][i])
        dy_i = float(jax_res['dy'][i])
        x = float(inputs_dict['xi'][i]) + dx_i
        y = float(inputs_dict['yi'][i]) + dy_i

        flux_err = float(np.sqrt(max(cov[0, 0], 0.0)))
        sky_err  = float(np.sqrt(max(cov[3, 3], 0.0)))
        mag, mag_err = mag_from_flux(max(flux, 1e-10), flux_err, zero_point)

        eps_psf = chi2 / np.sqrt(max(flux * max(psf_frac, 1e-6) * gain, 1.0))

        # Combined good-pixel mask: DQ-valid AND not sigma-clipped
        _vm_i = inputs_dict['valid_masks'][i]
        _cm_i = (jax_res['clipped_masks'][i] if 'clipped_masks' in jax_res
                 else np.zeros(n_pix, dtype=bool))
        _good_flat = _vm_i & ~_cm_i   # (n_pix,) bool

        def _good_at(dix_list, diy_list):
            idx = np.clip(
                (np.array(diy_list) + hw) * _nx_win + (np.array(dix_list) + hw),
                0, n_pix - 1
            )
            return _good_flat[idx]

        _peak_val = float(peaks[i])
        _conc_denom = flux * max(psf_frac, 1e-10)
        _cen_good = bool(_good_flat[center])
        concentration = (float(_peak_val / _conc_denom)
                         if _conc_denom > 0 and _cen_good else np.nan)
        n_conc_1x1 = int(_cen_good)

        # Multi-box concentrations using the PSF coefficient tile and pixel_vals,
        # summing only over good (not DQ-flagged, not sigma-clipped) pixels in
        # both numerator and denominator.
        concentration_2x2 = concentration_3x3 = np.nan
        n_conc_2x2 = n_conc_3x3 = 0
        if _psf_coeff_tiles is not None and flux > 0:
            _tile = _psf_coeff_tiles[i]

            def _pv_at(dix_arr, diy_arr):
                """Evaluate PSF tile at detector offsets (dix, diy) from (xi, yi)."""
                tile_xs = tr + (dix_arr.ravel() - dx_i) * psf_scale
                tile_ys = tr + (diy_arr.ravel() - dy_i) * psf_scale
                return _mc(_tile, [tile_ys, tile_xs],
                           order=3, mode='nearest', prefilter=False)

            def _data_at(dix_list, diy_list):
                """Pixel values (above sky) from pixel_vals window for given offsets."""
                idx = np.clip(
                    (np.array(diy_list) + hw) * _nx_win + (np.array(dix_list) + hw),
                    0, n_pix - 1
                )
                return pixel_vals[i, idx].astype(np.float64) - sky

            # 2×2 bracketing the fitted (dx_i, dy_i) position
            ix_lo_off = 0 if dx_i >= 0 else -1
            iy_lo_off = 0 if dy_i >= 0 else -1
            _dix_2 = np.array([[ix_lo_off, ix_lo_off+1]] * 2)
            _diy_2 = np.array([[iy_lo_off] * 2, [iy_lo_off+1] * 2])
            _good_2 = _good_at(_dix_2.ravel().tolist(), _diy_2.ravel().tolist())
            _P2 = _pv_at(_dix_2, _diy_2)
            _d2 = _data_at(_dix_2.ravel().tolist(), _diy_2.ravel().tolist())
            _P2_sum = float(_P2[_good_2].sum())
            n_conc_2x2 = int(_good_2.sum())
            if n_conc_2x2 >= 2 and _P2_sum > 0:
                concentration_2x2 = float(_d2[_good_2].sum() / (flux * _P2_sum))

            # 3×3 centred at integer position
            _off = np.array([-1, 0, 1])
            _DIX_3, _DIY_3 = np.meshgrid(_off, _off)
            _good_3 = _good_at(_DIX_3.ravel().tolist(), _DIY_3.ravel().tolist())
            _P3 = _pv_at(_DIX_3, _DIY_3)
            _d3 = _data_at(_DIX_3.ravel().tolist(), _DIY_3.ravel().tolist())
            _P3_sum = float(_P3[_good_3].sum())
            n_conc_3x3 = int(_good_3.sum())
            if n_conc_3x3 >= 5 and _P3_sum > 0:
                concentration_3x3 = float(_d3[_good_3].sum() / (flux * _P3_sum))

        cm = jax_res['clipped_masks'][i] if 'clipped_masks' in jax_res else None

        records.append(StarRecord(
            x=x, y=y,
            flux=flux, flux_err=flux_err,
            sky=sky, sky_err=sky_err,
            mag=float(mag), mag_err=float(mag_err),
            qfit=float(jax_res['qfit'][i]),
            chi2=chi2,
            central_res=float(jax_res['central_res'][i]),
            n_sat=int(n_sat_arr[i]),
            psf_frac=psf_frac,
            psf_peak=float(psf_peaks[i]),
            peak=float(peaks[i]),
            cov=cov,
            pass_number=pass_number,
            n_neighbors=0,
            dist_nearest=np.inf,
            dist_nearest_brighter=np.inf,
            n_iter=int(jax_res['n_iter'][i]),
            converged=bool(jax_res['converged'][i]),
            delta_max=float(jax_res['delta_max'][i]),
            clipped_mask=cm,
            chi2_scale=1.0,
            eps_psf=float(eps_psf),
            concentration=concentration,
            concentration_2x2=concentration_2x2,
            concentration_3x3=concentration_3x3,
            n_conc_1x1=n_conc_1x1,
            n_conc_2x2=n_conc_2x2,
            n_conc_3x3=n_conc_3x3,
        ))

    return records


def _window_offsets(xi, yi, hw, ny, nx):
    y_lo = max(0, yi - hw)
    y_hi = min(ny, yi + hw + 1)
    x_lo = max(0, xi - hw)
    x_hi = min(nx, xi + hw + 1)
    diy = (np.arange(y_lo, y_hi) - yi)[:, np.newaxis]
    dix = (np.arange(x_lo, x_hi) - xi)[np.newaxis, :]
    return y_lo, y_hi, x_lo, x_hi, diy, dix


def fit_star(data, x0, y0, psf_cube, xs, ys, psf_scale, hw,
             sky, gain, read_noise, max_iter, tol,
             noise_map, mask, x_offset, y_offset, zero_point, pass_number,
             psf_coeffs_cube=None, sat_threshold=np.inf,
             sigma_clip=True, sigma_clip_sigma=4.0, sigma_clip_iter=2,
             psf_cache=None, _fail_counter=None,
             eps_psf_star: float = 0.0):
    """Fit a single point source by joint linear Newton iteration.

    sky is fitted simultaneously as a 4th free parameter, initialized from
    the *sky* argument (annulus estimate) and updated each Newton step.

    eps_psf_star : float
        Fractional PSF model error for this star (dimensionless, typically
        0.01–0.05).  Adds a per-pixel variance term ``(eps_psf_star·max(flux·P,0))²``
        to account for PSF model mismatch.  Zero (default) disables the term.
        Set from the previous pass's ``StarRecord.eps_psf`` during refit.

    psf_coeffs_cube : prefiltered PSF cube from run_photometry (optional).
        When provided, interpolation runs on the prefiltered cube and no
        per-star spline_filter call is needed.  When None (e.g. in tests),
        the raw psf_cube is used and the interpolated PSF is prefiltered once
        per star before the Newton loop.
    sat_threshold : pixel value above which a pixel is counted as saturated
        (stored in StarRecord.n_sat).  Default np.inf disables counting.
    Returns StarRecord.
    """
    from .utils import mag_from_flux

    ny, nx = data.shape
    xi0c = int(np.clip(round(x0), 0, nx - 1))
    yi0c = int(np.clip(round(y0), 0, ny - 1))
    peak0 = float(data[yi0c, xi0c]) - sky  # used only in _failed_record calls

    # Interpolate and prefilter the PSF once at the initial position.
    # The PSF grid is coarse (~500 px spacing) so changes across one star's
    # Newton steps are negligible.
    _cube = psf_coeffs_cube if psf_coeffs_cube is not None else psf_cube
    local_psf = interpolate_psf(_cube, xs, ys, x0 + x_offset, y0 + y_offset,
                                _cache=psf_cache)
    if psf_coeffs_cube is None:
        psf_coeffs = spline_filter(local_psf, order=3, output=np.float64)
    else:
        psf_coeffs = local_psf  # already prefiltered coefficients

    # Pre-solve: 2-parameter least-squares fit for (flux, sky) with position
    # held fixed at x0, y0.  Evaluating the PSF over the whole window and
    # solving the 2-column normal equations gives a robust flux+sky estimate
    # before any position update runs.
    #
    # Single-pixel initialisers (peak0 / P_center) fail when:
    #   • the rounded starting pixel is off the true peak (random sub-pixel offset)
    #   • sky is overestimated in a crowded field  → peak0 < 0 → flux = 1
    # Either case leaves flux ≈ 1 while the true flux is e.g. 5000, making the
    # position columns (flux·∂P/∂x) 5000× too small → AᵀWA near-singular →
    # Newton step explodes on the very first iteration.
    _dx0 = x0 - xi0c;  _dy0 = y0 - yi0c
    _y0b, _y1b, _x0b, _x1b, _diy0, _dix0 = _window_offsets(xi0c, yi0c, hw, ny, nx)
    _d0  = data[_y0b:_y1b, _x0b:_x1b].astype(np.float64)
    _g0  = (~mask[_y0b:_y1b, _x0b:_x1b].ravel()) if mask is not None \
           else np.ones(_d0.size, dtype=bool)
    _P0, _, _ = _eval_psf_grad_fast(psf_coeffs, _dx0, _dy0, _dix0, _diy0, psf_scale)
    _n0 = int(_g0.sum())
    if _n0 >= 2:
        _A2 = np.column_stack([_P0.ravel()[_g0], np.ones(_n0)])
        _fs, *_ = np.linalg.lstsq(_A2, _d0.ravel()[_g0], rcond=None)
        flux = max(float(_fs[0]), 1.0)
        sky  = float(_fs[1])
    else:
        flux = max(peak0, 1.0)  # degenerate window; caught by good.sum()<5 below

    n_iter = max_iter
    converged = False
    delta_max = 0.0
    sky_init = sky  # for sky-drift diagnostics at failure
    for _it in range(max_iter):
        xi = int(round(x0))
        yi = int(round(y0))
        dx = x0 - xi
        dy = y0 - yi

        y_lo, y_hi, x_lo, x_hi, diy, dix = _window_offsets(xi, yi, hw, ny, nx)
        d = data[y_lo:y_hi, x_lo:x_hi].astype(np.float64)

        good = (~mask[y_lo:y_hi, x_lo:x_hi]) if mask is not None \
               else np.ones(d.shape, dtype=bool)

        if good.sum() < 5:   # need ≥ 5 good pixels for 4 parameters
            if _fail_counter is not None:
                _fail_counter['few_good_px'] = _fail_counter.get('few_good_px', 0) + 1
                _fail_counter['few_good_px_n_good'] = _fail_counter.get('few_good_px_n_good', [])
                _fail_counter['few_good_px_n_good'].append(int(good.sum()))
            return _failed_record(x0, y0, flux, sky, peak0, pass_number, zero_point)

        P, dPdx, dPdy = _eval_psf_grad_fast(psf_coeffs, dx, dy, dix, diy, psf_scale)

        if P.shape != d.shape:
            if _fail_counter is not None:
                _fail_counter['shape_mismatch'] = _fail_counter.get('shape_mismatch', 0) + 1
            return _failed_record(x0, y0, flux, sky, peak0, pass_number, zero_point)

        r = d - sky - flux * P

        if noise_map is not None:
            var = noise_map[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
        else:
            _model = np.maximum(flux * P, 0.0) + max(sky, 0.0)
            var = np.maximum(d, _model) / gain + (read_noise / gain) ** 2
        if eps_psf_star > 0.0:
            var = var + (eps_psf_star * np.maximum(flux * P, 0.0)) ** 2
        var = np.maximum(var, 1e-10)
        w = np.where(good, 1.0 / var, 0.0)

        g = good.ravel()
        w_g = w.ravel()[g]
        n_g = g.sum()
        # Design matrix: [δf column, δx column, δy column, δsky column]
        A = np.column_stack([P.ravel()[g],
                             (flux * dPdx).ravel()[g],
                             (flux * dPdy).ravel()[g],
                             np.ones(n_g)])
        AtWA = A.T @ (w_g[:, None] * A)
        AtWr = A.T @ (w_g * r.ravel()[g])

        try:
            delta = np.linalg.solve(AtWA, AtWr)
        except np.linalg.LinAlgError:
            if _fail_counter is not None:
                _fail_counter['linalg_error'] = _fail_counter.get('linalg_error', 0) + 1
            return _failed_record(x0, y0, flux, sky, peak0, pass_number, zero_point)

        # Clamp position step: a shift larger than hw pixels per iteration is
        # a sign of a diverging Newton step (e.g. nearly-singular system on a
        # poorly-sampled or bad-pixel window).  Flag and bail out.
        if abs(delta[1]) > 2 * hw or abs(delta[2]) > 2 * hw:
            if _fail_counter is not None:
                _fail_counter['pos_clamp'] = _fail_counter.get('pos_clamp', 0) + 1
                _fail_counter.setdefault('pos_clamp_steps', []).append(
                    (float(abs(delta[1])), float(abs(delta[2]))))
                _fail_counter.setdefault('pos_clamp_iter',  []).append(_it)
                _fail_counter.setdefault('pos_clamp_flux',  []).append(float(flux))
                _fail_counter.setdefault('pos_clamp_sky_drift', []).append(
                    float(sky - sky_init))
                _cond = float(np.linalg.cond(AtWA))
                _fail_counter.setdefault('pos_clamp_cond',  []).append(_cond)
            return _failed_record(x0, y0, flux, sky, peak0, pass_number, zero_point)

        # Flux step: prevent flux from crashing more than 75 % in one iteration.
        # A large negative δf collapses flux to the 1.0 floor and makes the next
        # iteration's position columns (flux·∂P/∂x) vanish → near-singular AᵀWA.
        # The relative limit gives the position estimate one or two steps to
        # self-correct before flux is allowed to approach zero.
        _flux_floor = max(0.25 * flux, 1.0)
        if flux + delta[0] < _flux_floor:
            delta[0] = _flux_floor - flux
        flux += delta[0]
        x0   += delta[1]
        y0   += delta[2]
        sky  += delta[3]
        flux  = max(flux, 1.0)

        # Bail if Newton has wandered the position outside the image entirely.
        if x0 < 0 or x0 >= nx or y0 < 0 or y0 >= ny:
            if _fail_counter is not None:
                _fail_counter['out_of_bounds'] = _fail_counter.get('out_of_bounds', 0) + 1
                _oob = []
                if x0 < 0: _oob.append(f'x={x0:.1f}<0')
                elif x0 >= nx: _oob.append(f'x={x0:.1f}>={nx}')
                if y0 < 0: _oob.append(f'y={y0:.1f}<0')
                elif y0 >= ny: _oob.append(f'y={y0:.1f}>={ny}')
                _fail_counter.setdefault('out_of_bounds_detail', []).append(', '.join(_oob))
            return _failed_record(x0, y0, flux, sky, peak0, pass_number, zero_point)

        delta_max = max(abs(delta[1]), abs(delta[2]))
        if delta_max < tol:
            n_iter = _it + 1
            converged = True
            break

    # --- Final evaluation for covariance and qfit ---
    xi = int(round(x0));  yi = int(round(y0))
    dx = x0 - xi;         dy = y0 - yi

    y_lo, y_hi, x_lo, x_hi, diy, dix = _window_offsets(xi, yi, hw, ny, nx)
    d = data[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
    good = (~mask[y_lo:y_hi, x_lo:x_hi]) if mask is not None \
           else np.ones(d.shape, dtype=bool)

    P, dPdx, dPdy = _eval_psf_grad_fast(psf_coeffs, dx, dy, dix, diy, psf_scale)

    if noise_map is not None:
        var = noise_map[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
    else:
        _model = np.maximum(flux * P, 0.0) + max(sky, 0.0)
        var = np.maximum(d, _model) / gain + (read_noise / gain) ** 2
    if eps_psf_star > 0.0:
        var = var + (eps_psf_star * np.maximum(flux * P, 0.0)) ** 2
    var = np.maximum(var, 1e-10)

    # --- Post-convergence sigma clipping ---
    # Iteratively mask pixels with large normalised residuals and re-solve.
    # Handles cosmic rays and warm pixels that survived the DQ mask.
    if sigma_clip and sigma_clip_iter > 0:
        for _sc in range(sigma_clip_iter):
            r_sc = d - sky - flux * P
            outlier = np.abs(r_sc) / np.sqrt(var) > sigma_clip_sigma
            new_good = good & ~outlier
            if new_good.sum() < 5 or not outlier[good].any():
                break  # nothing to clip, or too few pixels left

            g_sc = new_good.ravel()
            w_sc = 1.0 / var.ravel()[g_sc]
            n_sc = int(g_sc.sum())
            A_sc = np.column_stack([P.ravel()[g_sc],
                                    (flux * dPdx).ravel()[g_sc],
                                    (flux * dPdy).ravel()[g_sc],
                                    np.ones(n_sc)])
            try:
                delta_sc = np.linalg.solve(
                    A_sc.T @ (w_sc[:, None] * A_sc),
                    A_sc.T @ (w_sc * r_sc.ravel()[g_sc])
                )
            except np.linalg.LinAlgError:
                good = new_good
                break

            # Reject if the outlier was pulling position by an implausible amount
            if abs(delta_sc[1]) > 0.5 or abs(delta_sc[2]) > 0.5:
                good = new_good
                break

            if flux + delta_sc[0] < 1.0:
                delta_sc[0] = 1.0 - flux
            flux += delta_sc[0]; x0 += delta_sc[1]; y0 += delta_sc[2]; sky += delta_sc[3]
            flux = max(flux, 1.0)
            good = new_good

            # Re-evaluate at updated position for the next clipping round
            xi = int(round(x0)); yi = int(round(y0))
            dx = x0 - xi; dy = y0 - yi
            y_lo_n, y_hi_n, x_lo_n, x_hi_n, diy_n, dix_n = \
                _window_offsets(xi, yi, hw, ny, nx)
            if y_lo_n != y_lo or x_lo_n != x_lo:
                y_lo, y_hi, x_lo, x_hi = y_lo_n, y_hi_n, x_lo_n, x_hi_n
                diy, dix = diy_n, dix_n
                d = data[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
                good = (~mask[y_lo:y_hi, x_lo:x_hi]) if mask is not None \
                       else np.ones(d.shape, dtype=bool)
            P, dPdx, dPdy = _eval_psf_grad_fast(psf_coeffs, dx, dy, dix, diy, psf_scale)
            if noise_map is not None:
                var = noise_map[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
            else:
                _model = np.maximum(flux * P, 0.0) + max(sky, 0.0)
                var = np.maximum(d, _model) / gain + (read_noise / gain) ** 2
            if eps_psf_star > 0.0:
                var = var + (eps_psf_star * np.maximum(flux * P, 0.0)) ** 2
            var = np.maximum(var, 1e-10)

    # Record which pixels were sigma-clipped (good at DQ level but rejected by clipping)
    good_dq = (~mask[y_lo:y_hi, x_lo:x_hi]) if mask is not None \
              else np.ones(d.shape, dtype=bool)
    clipped_mask = good_dq & ~good  # True = passed DQ but clipped by sigma-clip

    r = d - sky - flux * P
    w = np.where(good, 1.0 / var, 0.0)

    g = good.ravel()
    w_g = w.ravel()[g]
    n_g = g.sum()
    A = np.column_stack([P.ravel()[g],
                         (flux * dPdx).ravel()[g],
                         (flux * dPdy).ravel()[g],
                         np.ones(n_g)])
    AtWA = A.T @ (w_g[:, None] * A)
    try:
        cov = np.linalg.inv(AtWA)   # 4×4 in (flux, x, y, sky) order
    except np.linalg.LinAlgError:
        cov = np.eye(4) * 1e6

    data_m_sky = (d - sky)[good]
    qfit = float(np.sum(np.abs(r[good])) / max(np.sum(np.abs(data_m_sky)), 1e-10))

    # Scaled reduced chi-squared RMS: sqrt(Σ(r²/var) / DOF)  (Fortran 'c')
    dof = max(n_g - 4, 1)
    chi2 = float(np.sqrt(np.sum(r[good] ** 2 / var[good]) / dof))

    # Saturated pixel count in fit window (Fortran 'n') — uses raw data, not residual
    n_sat = int(np.sum(d >= sat_threshold))

    # PSF value at dix=0, diy=0 — the central pixel of the fit window
    P_cen, _, _ = _eval_psf_grad_fast(psf_coeffs, dx, dy,
                                       np.array([[0]]), np.array([[0]]), psf_scale)
    psf_frac = float(P_cen.flat[0])   # PSF at fitted position (Fortran 'f')

    # PSF peak for a perfectly centered star (Fortran 'F')
    P_peak, _, _ = _eval_psf_grad_fast(psf_coeffs, 0.0, 0.0,
                                        np.array([[0]]), np.array([[0]]), psf_scale)
    psf_peak = float(P_peak.flat[0])

    # Normalized central pixel residual (Fortran 'C' = cc)
    xi_f = int(np.clip(round(x0), 0, nx - 1))
    yi_f = int(np.clip(round(y0), 0, ny - 1))
    d_cen = float(data[yi_f, xi_f])
    central_res = float((d_cen - sky - flux * psf_frac) / max(flux, 1e-10))
    central_res = float(np.clip(central_res, -0.999, 0.999))

    flux_err = float(np.sqrt(max(cov[0, 0], 0.0)))
    sky_err  = float(np.sqrt(max(cov[3, 3], 0.0)))
    mag, mag_err = mag_from_flux(max(flux, 1e-10), flux_err, zero_point)

    peak = d_cen - sky

    # Concentration: observed pixel sum / (flux * PSF model sum) for three box sizes,
    # restricted to pixels that are good (not DQ-flagged, not sigma-clipped).
    # Summing only over good pixels in both numerator and denominator avoids biasing
    # concentration low for dead/masked pixels or high for uncorrected hot pixels.
    # Stars ≈ 1.0; CRs > 1 (too sharp); extended sources < 1 (too broad).
    _good_h, _good_w = good.shape

    def _pixel_good(abs_y, abs_x):
        ly = abs_y - y_lo
        lx = abs_x - x_lo
        return (0 <= ly < _good_h) and (0 <= lx < _good_w) and bool(good[ly, lx])

    # 1×1: central pixel only — nan if that pixel is masked or clipped
    _conc_denom = float(flux) * max(psf_frac, 1e-10)
    _cen_good = _pixel_good(yi_f, xi_f)
    concentration = (float(peak / _conc_denom)
                     if _conc_denom > 0 and _cen_good else np.nan)
    n_conc_1x1 = int(_cen_good)

    # 2×2 concentration — 4 pixels bracketing the fitted sub-pixel position.
    _ix_lo = xi_f + (0 if (x0 - xi_f) >= 0 else -1)
    _iy_lo = yi_f + (0 if (y0 - yi_f) >= 0 else -1)
    _dix_2 = np.array([[_ix_lo - xi_f, _ix_lo + 1 - xi_f],
                        [_ix_lo - xi_f, _ix_lo + 1 - xi_f]])
    _diy_2 = np.array([[_iy_lo - yi_f, _iy_lo - yi_f],
                        [_iy_lo + 1 - yi_f, _iy_lo + 1 - yi_f]])
    _y2_abs = np.array([_iy_lo, _iy_lo, _iy_lo + 1, _iy_lo + 1])
    _x2_abs = np.array([_ix_lo, _ix_lo + 1, _ix_lo, _ix_lo + 1])
    _good_2 = np.array([_pixel_good(int(_y2_abs[k]), int(_x2_abs[k])) for k in range(4)])
    _y2 = np.clip(_y2_abs, 0, ny - 1)
    _x2 = np.clip(_x2_abs, 0, nx - 1)
    _d2 = data[_y2, _x2].astype(np.float64) - sky
    _P2, _, _ = _eval_psf_grad_fast(psf_coeffs, x0 - xi_f, y0 - yi_f,
                                     _dix_2, _diy_2, psf_scale)
    _P2_sum = float(_P2.ravel()[_good_2].sum())
    n_conc_2x2 = int(_good_2.sum())
    concentration_2x2 = (float(_d2[_good_2].sum() / (flux * _P2_sum))
                         if n_conc_2x2 >= 2 and _P2_sum > 0 and flux > 0 else np.nan)

    # 3×3 concentration — 9 pixels centred at the rounded position.
    _off = np.array([-1, 0, 1])
    _DIX_3, _DIY_3 = np.meshgrid(_off, _off)
    _y3_abs = yi_f + _DIY_3.ravel()
    _x3_abs = xi_f + _DIX_3.ravel()
    _good_3 = np.array([_pixel_good(int(_y3_abs[k]), int(_x3_abs[k])) for k in range(9)])
    _y3 = np.clip(_y3_abs, 0, ny - 1)
    _x3 = np.clip(_x3_abs, 0, nx - 1)
    _d3 = data[_y3, _x3].astype(np.float64) - sky
    _P3, _, _ = _eval_psf_grad_fast(psf_coeffs, x0 - xi_f, y0 - yi_f,
                                     _DIX_3, _DIY_3, psf_scale)
    _P3_sum = float(_P3.ravel()[_good_3].sum())
    n_conc_3x3 = int(_good_3.sum())
    concentration_3x3 = (float(_d3[_good_3].sum() / (flux * _P3_sum))
                         if n_conc_3x3 >= 5 and _P3_sum > 0 and flux > 0 else np.nan)

    # Per-star implied fractional PSF model error.  For a Poisson-limited bright star
    # chi2 ≈ 1 regardless of flux, so eps_psf → 1/sqrt(flux·psf_frac·gain) → 0.
    # When PSF model errors dominate, chi2 scales as sqrt(ε·flux·psf_frac·gain),
    # making eps_psf ≈ ε · constant that is independent of flux at high S/N.
    eps_psf = float(chi2 / np.sqrt(max(float(flux) * max(psf_frac, 1e-6) * gain, 1.0)))

    return StarRecord(
        x=float(x0), y=float(y0), flux=float(flux), flux_err=flux_err,
        sky=float(sky), sky_err=sky_err,
        mag=float(mag), mag_err=float(mag_err),
        qfit=qfit, chi2=chi2, central_res=central_res,
        n_sat=n_sat, psf_frac=psf_frac, psf_peak=psf_peak,
        peak=peak, cov=cov, pass_number=pass_number,
        n_neighbors=0, dist_nearest=np.inf, dist_nearest_brighter=np.inf,
        n_iter=n_iter, converged=converged,
        delta_max=delta_max, clipped_mask=clipped_mask,
        eps_psf=eps_psf,
        concentration=concentration,
        concentration_2x2=concentration_2x2,
        concentration_3x3=concentration_3x3,
        n_conc_1x1=n_conc_1x1,
        n_conc_2x2=n_conc_2x2,
        n_conc_3x3=n_conc_3x3,
    )


# ---------------------------------------------------------------------------
# Source finding — optimised
# ---------------------------------------------------------------------------

def find_sources(data, sky_inner, sky_outer, hmin, fmin, mask=None,
                 suppress_radius=None, peak_mask=None, verbose=False,
                 psf_peak_val=1.0, psf_core_3x3=None):
    """Find point-source candidates as local maxima above the flux threshold.

    Pipeline:
      1. Local maxima after excluding peak_mask-flagged pixels (or mask if
         peak_mask is None).
      2. NMS (non-maximum suppression): keep only the dominant peak within
         each *suppress_radius*-pixel box; peak_mask pixels cannot dominate.
      3. fmin threshold applied to flux estimate = (peak - sky) / psf_peak_val.
      4. Return surviving candidates.

    The DQ *mask* (True = bad pixel) is applied before peak detection
    (via peak_mask, which may exclude mild flags like warm pixels) and
    also used for sky estimation.

    Parameters
    ----------
    mask            : 2D bool (True = DQ-flagged), used for sky estimation
    peak_mask       : 2D bool (True = exclude from peak detection), used for
                      local-maxima exclusion and NMS.  If None, falls back to
                      mask.  Pass a stricter subset of mask to allow mild flags
                      (e.g. warm pixels, bit 4) through peak detection while
                      still excluding them from sky and fitting.
    psf_peak_val    : float, the peak pixel value of the PSF (fraction of total
                      flux in the central pixel).  Used as fallback when
                      psf_core_3x3 is None or fewer than 2 unmasked core pixels
                      remain.  Default 1.0 (no conversion).
    psf_core_3x3    : (3, 3) float64 array of PSF values at the 9 detector-pixel
                      positions surrounding the star centre (dx=dy=0, central PSF
                      model).  When provided, the fmin flux estimate uses a
                      noise-weighted matched-filter over all unmasked core pixels
                      instead of the single peak pixel.  Masked pixels are
                      excluded; falls back to single-pixel estimate if fewer than
                      2 valid pixels remain.

    Returns
    -------
    xs, ys, peaks, skys, sigs : tuple of 1D float64 arrays, sorted by peak
        flux descending.  Empty arrays if no candidates found.
    """
    border = sky_outer + 1
    ny, nx = data.shape

    _pmask = peak_mask if peak_mask is not None else mask

    # --- Step 1: 8-neighbour local maxima, excluding peak_mask pixels ---
    c = data[1:-1, 1:-1]
    is_max = np.zeros((ny, nx), dtype=bool)
    is_max[1:-1, 1:-1] = (
        (c > data[:-2, :-2]) & (c > data[:-2, 1:-1]) & (c > data[:-2, 2:]) &
        (c > data[1:-1, :-2]) & (c > data[1:-1, 2:]) &
        (c > data[2:, :-2])  & (c > data[2:, 1:-1]) & (c > data[2:, 2:])
    )
    if _pmask is not None:
        is_max[_pmask] = False

    # --- Step 2: Border exclusion ---
    is_max[:border, :]  = False;  is_max[-border:, :] = False
    is_max[:, :border]  = False;  is_max[:, -border:] = False

    iy_arr, ix_arr = np.where(is_max)
    n_initial = len(iy_arr)
    if verbose:
        print(f"      {n_initial} initial local maxima")
    if n_initial == 0:
        return (np.empty(0), np.empty(0), np.empty(0),
                np.empty(0), np.empty(0))

    # --- Step 3: NMS (before sky estimation to minimise sky-estimation work) ---
    # peak_mask pixels are set to -inf so they cannot dominate their
    # neighbourhood and suppress a real nearby source.
    # np.where builds the masked array in a single pass (no copy + index).
    if suppress_radius and suppress_radius > 1:
        if _pmask is not None and _pmask.any():
            _nms_data = np.where(_pmask, -np.inf, data.astype(float))
        else:
            _nms_data = data.astype(float)
        dominant = maximum_filter(_nms_data, size=2 * suppress_radius + 1,
                                  mode='constant', cval=-np.inf)
        is_dom = data[iy_arr, ix_arr] >= dominant[iy_arr, ix_arr]
        n_nms_drop = int((~is_dom).sum())
        iy_arr = iy_arr[is_dom]
        ix_arr = ix_arr[is_dom]
        if verbose:
            print(f"      {len(iy_arr)} after NMS (r={suppress_radius}), "
                  f"{n_nms_drop} suppressed by brighter neighbours")

    if len(iy_arr) == 0:
        return (np.empty(0), np.empty(0), np.empty(0),
                np.empty(0), np.empty(0))

    # --- Step 4: Sky estimation (only on NMS survivors) ---
    sky_vals, sky_sigs = estimate_sky_batch(
        data, ix_arr, iy_arr, sky_inner, sky_outer, mask)

    # --- Step 5: fmin threshold ---
    # flux_est converts peak counts above sky into estimated total source flux.
    # With psf_core_3x3: noise-weighted matched filter over the 3×3 unmasked
    # core pixels → more robust to sub-pixel offsets and masked neighbours.
    # Without (or <2 valid pixels): single-pixel fallback using psf_peak_val.
    peaks = data[iy_arr, ix_arr] - sky_vals

    if psf_core_3x3 is not None:
        _p9  = psf_core_3x3.ravel().astype(np.float64)          # (9,)
        _dj9 = np.array([-1,-1,-1, 0, 0, 0, 1, 1, 1], dtype=int)
        _di9 = np.array([-1, 0, 1,-1, 0, 1,-1, 0, 1], dtype=int)

        # Absolute pixel positions for every candidate × every offset: (N, 9)
        _y9 = iy_arr[:, None] + _dj9[None, :]
        _x9 = ix_arr[:, None] + _di9[None, :]

        # In-bounds flag; clip for safe indexing
        _inb  = (_y9 >= 0) & (_y9 < ny) & (_x9 >= 0) & (_x9 < nx)
        _yc   = np.clip(_y9, 0, ny - 1)
        _xc   = np.clip(_x9, 0, nx - 1)

        # Raw pixel values and sky-subtracted values: (N, 9)
        _raw9 = data[_yc, _xc].astype(np.float64)
        _d9   = _raw9 - sky_vals[:, None]

        # Valid mask: in bounds and not DQ-flagged
        _valid = _inb & (~mask[_yc, _xc] if mask is not None
                         else np.ones(_y9.shape, dtype=bool))

        # Noise-weighted matched filter.
        # Variance: Poisson signal + sky noise floor = max(raw_pixel, sky_sig²)
        _var9  = np.maximum(_raw9, (sky_sigs[:, None]) ** 2)
        _w9    = np.where(_valid, 1.0 / np.maximum(_var9, 1e-10), 0.0)

        _num   = np.sum(_d9   * _p9[None, :] * _w9, axis=1)   # (N,)
        _den   = np.sum(_p9[None, :]**2       * _w9, axis=1)   # (N,)
        _nvalid = _valid.sum(axis=1)

        # Use 3×3 estimate where ≥2 valid pixels; single-pixel fallback otherwise
        _flux3 = np.where(_den > 1e-10,
                          _num / np.maximum(_den, 1e-10),
                          peaks / max(psf_peak_val, 1e-30))
        flux_est = np.where(_nvalid >= 2, _flux3,
                            peaks / max(psf_peak_val, 1e-30))
    else:
        flux_est = peaks / max(psf_peak_val, 1e-30)

    good = (flux_est >= fmin) & (sky_sigs > 0)

    ix_ok    = ix_arr[good].astype(np.float64)
    iy_ok    = iy_arr[good].astype(np.float64)
    peaks_ok = peaks[good]
    sky_ok   = sky_vals[good]
    sigs_ok  = sky_sigs[good]

    if verbose:
        n_fmin_drop = int((~good).sum())
        print(f"      {len(ix_ok)} above fmin={fmin:.0f} e- "
              f"(psf_peak={psf_peak_val:.3f}), "
              f"{n_fmin_drop} below threshold → {len(ix_ok)} candidates for fitting")

    if len(ix_ok) == 0:
        return (np.empty(0), np.empty(0), np.empty(0),
                np.empty(0), np.empty(0))

    # --- Step 7: Sort by peak flux descending (bright stars fitted first) ---
    order = np.argsort(-peaks_ok)
    return (ix_ok[order], iy_ok[order], peaks_ok[order],
            sky_ok[order], sigs_ok[order])


# ---------------------------------------------------------------------------
# DQ statistics (post-fit, applied to the full catalogue)
# ---------------------------------------------------------------------------

def compute_dq_stats(records, dq_array, x_offset=0.0, y_offset=0.0):
    """Compute per-star DQ flag summaries and store them on each StarRecord.

    For each record computes the bitwise OR of the raw DQ integer values
    in the 1×1, 2×2 and 3×3 windows centred on the fitted (x, y) position.

    Parameters
    ----------
    records   : list of StarRecord
    dq_array  : 2D int32 array of raw DQ values for the chip
    x_offset  : chip x-coordinate offset (records use chip-local coords)
    y_offset  : chip y-coordinate offset
    """
    if dq_array is None or len(records) == 0:
        return

    ny, nx = dq_array.shape

    for rec in records:
        xi = int(round(float(rec.x)))
        yi = int(round(float(rec.y)))

        # 1×1: single pixel
        if 0 <= yi < ny and 0 <= xi < nx:
            rec.dq_1x1 = int(dq_array[yi, xi])
        else:
            rec.dq_1x1 = 0

        # 2×2: four pixels at (xi,yi),(xi+1,yi),(xi,yi+1),(xi+1,yi+1)
        dq_2 = 0
        for dy2, dx2 in [(0,0),(0,1),(1,0),(1,1)]:
            yw = yi + dy2;  xw = xi + dx2
            if 0 <= yw < ny and 0 <= xw < nx:
                dq_2 |= int(dq_array[yw, xw])
        rec.dq_2x2 = dq_2

        # 3×3: all nine pixels centred on (xi, yi)
        dq_3 = 0
        for dy3 in range(-1, 2):
            for dx3 in range(-1, 2):
                yw = yi + dy3;  xw = xi + dx3
                if 0 <= yw < ny and 0 <= xw < nx:
                    dq_3 |= int(dq_array[yw, xw])
        rec.dq_3x3 = dq_3


# ---------------------------------------------------------------------------
# Neighbour statistics (post-fit, applied to the full catalogue)
# ---------------------------------------------------------------------------

def compute_neighbor_stats(records, hw):
    """Compute per-star neighbour metrics and store them on each StarRecord.

    Must be called after all fitting passes so that the full catalogue is
    available.  Modifies *records* in-place.

    Parameters
    ----------
    records : list of StarRecord
    hw      : int — fit window half-width; stars within this radius are
              counted as "in the fit window" for n_neighbors.

    Sets on each record
    -------------------
    n_neighbors          : int   — other detected stars with dist < hw px
    dist_nearest         : float — px to nearest other detected star
    dist_nearest_brighter: float — px to nearest other detected star that
                           has strictly higher flux; np.inf if none exists
                           in the catalogue (star is the brightest nearby, or
                           the only star).

    Algorithm
    ---------
    Uses a scipy cKDTree for O(n log n) distance queries.  For
    dist_nearest_brighter, the K=min(n-1, 64) nearest neighbours are checked;
    if no brighter star appears among them the value is set to np.inf.  For
    typical HST/JWST star densities (< 5 stars / 1000 px²) K=64 captures
    every star within ~100 px, which is far beyond any meaningful crowding
    radius.
    """
    n = len(records)
    if n == 0:
        return

    from scipy.spatial import cKDTree

    xy = np.array([[r.x, r.y] for r in records], dtype=np.float64)
    flux = np.array([r.flux for r in records], dtype=np.float64)

    if n == 1:
        records[0].n_neighbors = 0
        records[0].dist_nearest = np.inf
        records[0].dist_nearest_brighter = np.inf
        return

    tree = cKDTree(xy)

    # --- n_neighbors: count catalogue stars within hw pixels (excluding self) ---
    # return_length=True (scipy ≥ 1.9) avoids allocating the full index lists.
    try:
        counts = tree.query_ball_point(xy, r=hw, return_length=True)
    except TypeError:
        counts = np.array([len(lst) for lst in tree.query_ball_point(xy, r=hw)])

    # --- K nearest neighbours (k=1 is self with dist=0) ---
    K = min(n, 65)   # 64 neighbours + self
    dists, idxs = tree.query(xy, k=K)

    for i, rec in enumerate(records):
        rec.n_neighbors = int(counts[i]) - 1          # subtract self

        rec.dist_nearest = float(dists[i, 1]) if n >= 2 else np.inf

        # Among the K nearest (excluding self at column 0), find the nearest
        # one that has strictly higher flux.
        neigh_flux = flux[idxs[i, 1:]]               # shape (K-1,)
        neigh_dist = dists[i, 1:]
        brighter = neigh_flux > flux[i]
        rec.dist_nearest_brighter = (
            float(neigh_dist[brighter].min()) if brighter.any() else np.inf
        )


# ---------------------------------------------------------------------------
# Star/galaxy classification
# ---------------------------------------------------------------------------

def _conc_adaptive_bounds(conc_arr, mag_arr, star_mask,
                          fixed_lo, fixed_hi,
                          mag_bin_width, min_bin_stars,
                          conc_width_factor, conc_min_width,
                          conc_hard_lo=0.5, conc_hard_hi=2.0):
    """Compute per-star adaptive concentration bounds from the stellar locus.

    Uses the binned median ± conc_width_factor × half-width of the 68% region
    (with a floor of conc_min_width on the half-width) among sources in
    *star_mask*.  Falls back to the fixed [fixed_lo, fixed_hi] bounds where
    there are too few stars in a bin or overall.

    Returns
    -------
    lo_per_star, hi_per_star : ndarray, same length as conc_arr
    """
    n = len(conc_arr)
    lo_per_star = np.full(n, fixed_lo)
    hi_per_star = np.full(n, fixed_hi)

    avail = np.isfinite(conc_arr) & np.isfinite(mag_arr)
    sm = star_mask & avail & (conc_arr >= conc_hard_lo) & (conc_arr <= conc_hard_hi)
    if sm.sum() < min_bin_stars:
        return lo_per_star, hi_per_star

    mag_s  = mag_arr[sm]
    conc_s = conc_arr[sm]
    bin_edges   = np.arange(mag_s.min(), mag_s.max() + mag_bin_width, mag_bin_width)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bmed = np.full(len(bin_centres), np.nan)
    blo  = np.full(len(bin_centres), np.nan)
    bhi  = np.full(len(bin_centres), np.nan)
    for k, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        m = (mag_s >= lo) & (mag_s < hi)
        if m.sum() >= min_bin_stars:
            v  = conc_s[m]
            med = float(np.median(v))
            hw  = max(float((np.percentile(v, 84) - np.percentile(v, 16)) / 2),
                      conc_min_width)
            bmed[k] = med
            blo[k]  = med - conc_width_factor * hw
            bhi[k]  = med + conc_width_factor * hw

    ok = np.isfinite(bmed)
    if ok.sum() < 2:
        return lo_per_star, hi_per_star

    bc_ok = bin_centres[ok]
    lo_per_star = np.interp(mag_arr, bc_ok, blo[ok])
    hi_per_star = np.interp(mag_arr, bc_ok, bhi[ok])
    return lo_per_star, hi_per_star


def classify_stars(records,
                   conc_lo=0.75, conc_hi=None,
                   qfit_global_max=1.5,
                   qfit_percentile=25, qfit_multiplier=4.0,
                   mag_bin_width=0.5, min_bin_stars=5,
                   conc_width_factor=4.0, conc_min_width=0.01,
                   conc_hard_lo=0.5, conc_hard_hi=2.0,
                   max_conc_iter=10):
    """Classify each record as a likely star or likely non-star.

    Sets ``is_star_candidate`` on each record in-place based on two criteria:

    1. **Concentration** (two-pass adaptive):
       Pass 1 seeds the stellar locus using the fixed [conc_lo, 1/conc_lo]
       window.  Pass 2 replaces the fixed window with per-magnitude bounds of
       ``median ± conc_width_factor × half_width`` (where half_width is half
       the 68% spread, floored at conc_min_width).  All three metrics that have
       finite values must pass; NaN metrics are skipped.

    2. **qfit**: below both a global cap (*qfit_global_max*) and a
       magnitude-adaptive threshold derived from the p<qfit_percentile> of
       qfit within each magnitude bin, multiplied by *qfit_multiplier*.

    Parameters
    ----------
    conc_lo          : float — initial lower bound; upper = 1/conc_lo
    conc_width_factor: float — adaptive half-width multiplier (default 4)
    conc_min_width   : float — floor on adaptive half-width (default 0.05)
    qfit_global_max  : float — hard upper limit regardless of adaptive threshold
    qfit_percentile  : int   — percentile of qfit used as the locus anchor
    qfit_multiplier  : float — multiplier above the percentile → threshold
    mag_bin_width    : float — magnitude bin width for adaptive thresholds
    min_bin_stars    : int   — minimum stars per bin to compute local threshold

    Returns
    -------
    n_candidates : int — number of records flagged as likely stars
    """
    if conc_hi is None:
        conc_hi = 1.0 / conc_lo

    if not records:
        return 0

    mag    = np.array([r.mag   for r in records])
    qfit   = np.array([r.qfit  for r in records])
    conc1  = np.array([getattr(r, 'concentration',      np.nan) for r in records])
    conc2  = np.array([getattr(r, 'concentration_2x2',  np.nan) for r in records])
    conc3  = np.array([getattr(r, 'concentration_3x3',  np.nan) for r in records])

    _a1 = np.isfinite(conc1)
    _a2 = np.isfinite(conc2)
    _a3 = np.isfinite(conc3)
    _kw = dict(mag_bin_width=mag_bin_width, min_bin_stars=min_bin_stars,
               conc_width_factor=conc_width_factor, conc_min_width=conc_min_width,
               conc_hard_lo=conc_hard_lo, conc_hard_hi=conc_hard_hi)

    def _conc_check(lo1, hi1, lo2, hi2, lo3, hi3):
        p1 = ~_a1 | ((conc1 >= lo1) & (conc1 <= hi1))
        p2 = ~_a2 | ((conc2 >= lo2) & (conc2 <= hi2))
        p3 = ~_a3 | ((conc3 >= lo3) & (conc3 <= hi3))
        return p1 & p2 & p3

    # Seed — 2×2 only with fixed [conc_lo, conc_hi].
    current_star = ~_a2 | ((conc2 >= conc_lo) & (conc2 <= conc_hi))

    # Global qfit cap.
    qfit_ok = np.isfinite(qfit) & (qfit < qfit_global_max)

    # Adaptive magnitude-bin qfit threshold: trace the stellar qfit locus.
    finite_mask = np.isfinite(mag) & np.isfinite(qfit)
    if finite_mask.sum() >= min_bin_stars:
        mag_f   = mag[finite_mask]
        qfit_f  = qfit[finite_mask]
        mag_min = float(np.nanmin(mag_f))
        mag_max = float(np.nanmax(mag_f))
        bin_edges = np.arange(mag_min, mag_max + mag_bin_width, mag_bin_width)

        bin_centers    = []
        bin_thresholds = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            m = (mag_f >= lo) & (mag_f < hi)
            if m.sum() >= min_bin_stars:
                p_qfit = float(np.percentile(qfit_f[m], qfit_percentile))
                thresh = min(p_qfit * qfit_multiplier, qfit_global_max)
                bin_centers.append(float((lo + hi) / 2.0))
                bin_thresholds.append(thresh)

        if len(bin_centers) >= 2:
            bc = np.array(bin_centers)
            bt = np.array(bin_thresholds)
            adaptive_thresh = np.interp(mag, bc, bt, left=bt[0], right=bt[-1])
            qfit_ok &= (qfit < adaptive_thresh)

    current_star = current_star & qfit_ok

    # Iterate: fit adaptive bounds from current_star (all 3 metrics), re-select,
    # repeat until convergence.  At convergence the final star set is exactly
    # the set of sources within the bounds derived from itself — self-consistent
    # by construction, so the plot can use is_star_candidate directly.
    for _ in range(max_conc_iter):
        _lo1, _hi1 = _conc_adaptive_bounds(conc1, mag, current_star, conc_lo, conc_hi, **_kw)
        _lo2, _hi2 = _conc_adaptive_bounds(conc2, mag, current_star, conc_lo, conc_hi, **_kw)
        _lo3, _hi3 = _conc_adaptive_bounds(conc3, mag, current_star, conc_lo, conc_hi, **_kw)
        new_star = _conc_check(_lo1, _hi1, _lo2, _hi2, _lo3, _hi3) & qfit_ok
        if np.array_equal(new_star, current_star):
            break
        current_star = new_star

    n_candidates = 0
    for i, r in enumerate(records):
        r.is_star_candidate = bool(current_star[i])
        if r.is_star_candidate:
            n_candidates += 1

    return n_candidates


# ---------------------------------------------------------------------------
# Chi²-inflation (standalone so run_photometry_fits can apply it globally)
# ---------------------------------------------------------------------------

def _build_chi2_mag_bins(mags, chi2s, bin_width=0.5, min_bin_width=0.1,
                         min_stars=50, max_stars=200, smooth_bandwidth=1.5):
    """Bin chi2 by adaptive magnitude bins, merge small bins, then smooth.

    Steps:
    1. Adaptive bins: a new bin starts each time the running span exceeds
       bin_width (0.5 mag) OR the count reaches max_stars (100), subject to a
       minimum span of min_bin_width (0.1 mag) — so dense regions get finer
       bins but never narrower than 0.1 mag.
    2. Forward-merge bins with < min_stars stars into the next bin; any
       leftover tail is merged backward into the last accepted bin.
    3. Compute median chi2 and standard error of the median (MAD-based) per bin.
    4. Uncertainty-aware Gaussian kernel smoothing: weight = 1/sigma^2, kernel
       in magnitude; edge-padded with nearest value so the endpoints are not
       pulled toward the global mean.

    Returns
    -------
    (bin_mags, bin_smooth, bin_raw, bin_unc) — all 1-D ndarrays sorted by
    ascending magnitude — or None if fewer than 2 bins result.
    """
    try:
        from scipy.stats import median_abs_deviation as _madfn
    except ImportError:
        def _madfn(x, scale='normal'):
            med = np.median(x)
            m = np.median(np.abs(x - med))
            return m * (1.4826 if scale == 'normal' else 1.0)

    order   = np.argsort(mags)
    mags_s  = mags[order]
    chi2s_s = chi2s[order]

    # Adaptive binning: close the current bin when (span >= bin_width OR
    # count >= max_stars) AND span >= min_bin_width.  The min_bin_width guard
    # prevents very dense regions from producing sub-0.1-mag slivers.
    init_bins = []
    bin_start = 0
    for i in range(len(mags_s)):
        n_in_bin = i - bin_start
        span     = mags_s[i] - mags_s[bin_start] if n_in_bin > 0 else 0.0
        hit_max  = n_in_bin >= max_stars or span >= bin_width
        if n_in_bin > 0 and hit_max and span >= min_bin_width:
            init_bins.append((mags_s[bin_start:i], chi2s_s[bin_start:i]))
            bin_start = i
    if bin_start < len(mags_s):
        init_bins.append((mags_s[bin_start:], chi2s_s[bin_start:]))

    if not init_bins:
        return None

    carry_m = carry_c = np.array([])
    merged  = []
    for m, c in init_bins:
        cm = np.concatenate([carry_m, m])
        cc = np.concatenate([carry_c, c])
        if len(cm) >= min_stars:
            merged.append((cm, cc))
            carry_m = carry_c = np.array([])
        else:
            carry_m, carry_c = cm, cc
    if carry_m.size:
        if merged:
            mm, mc = merged[-1]
            merged[-1] = (np.concatenate([mm, carry_m]),
                          np.concatenate([mc, carry_c]))
        else:
            merged.append((carry_m, carry_c))

    if len(merged) < 2:
        return None

    bin_mags = np.array([float(np.median(m)) for m, _ in merged])
    bin_raw  = np.clip([float(np.median(c)) for _, c in merged], 0.1, 20.0)
    bin_raw  = np.asarray(bin_raw, dtype=float)
    bin_unc  = []
    for _, c in merged:
        n = len(c)
        if n >= 3:
            mad = float(_madfn(c, scale='normal'))
            sigma_med = 1.2533 * mad / np.sqrt(n)
        elif n == 2:
            sigma_med = float(np.abs(c[1] - c[0])) / np.sqrt(2)
        else:
            sigma_med = 0.5
        bin_unc.append(max(sigma_med, 0.01))
    bin_unc = np.array(bin_unc)
    n_bins  = len(bin_raw)

    # Uncertainty-weighted linear fit through the first 3 bins at the bright
    # edge (index 0 = brightest).  Used for both the Gaussian smoother's left
    # padding and for explicit extrapolated points prepended to the PCHIP.
    n_fit_l = min(3, n_bins)
    if n_fit_l >= 2:
        _c_bright = np.polyfit(np.arange(n_fit_l, dtype=float), bin_raw[:n_fit_l], 1,
                               w=1.0 / bin_unc[:n_fit_l])
    else:
        _c_bright = None

    # Bright-end extrapolated anchor points: spaced by the median bin spacing of
    # the first 3 bins, covering from the brightest bin back to the actual data
    # minimum (i.e. the brightest star in the catalog).
    if _c_bright is not None and n_fit_l >= 2:
        dM = float(np.median(np.diff(bin_mags[:n_fit_l])))
        if dM > 0:
            gap = bin_mags[0] - float(mags_s[0])
            n_extrap = max(2, int(np.ceil(gap / dM)) + 1)
            extrap_idx  = np.arange(-n_extrap, 0, dtype=float)     # negative bin indices
            extrap_mags = bin_mags[0] + extrap_idx * dM            # ascending magnitude
            extrap_chi2 = np.maximum(np.polyval(_c_bright, extrap_idx), 1.0)
        else:
            extrap_mags = extrap_chi2 = np.array([])
    else:
        extrap_mags = extrap_chi2 = np.array([])

    # Gaussian kernel smoothing weighted by 1/sigma^2.
    # Distance is measured in bin-index units so smoothing is uniform across
    # the full magnitude range regardless of adaptive bin spacing.
    weights = 1.0 / (bin_unc ** 2)
    n_pad   = max(5, int(np.ceil(3.0 * smooth_bandwidth)))

    if _c_bright is not None:
        pad_left = np.maximum(np.polyval(_c_bright, np.arange(-n_pad, 0, dtype=float)), 1.0)
    else:
        pad_left = np.full(n_pad, max(float(bin_raw[0]), 1.0))
    pad_right = np.full(n_pad, float(bin_raw[-1]))

    idx_full = np.concatenate([
        np.arange(-n_pad, 0),
        np.arange(n_bins),
        np.arange(n_bins, n_bins + n_pad),
    ], dtype=float)
    pv = np.concatenate([pad_left,                        bin_raw,  pad_right])
    pw = np.concatenate([np.full(n_pad, weights[0]),      weights,  np.full(n_pad, weights[-1])])

    smoothed = np.empty_like(bin_raw)
    for i in range(n_bins):
        d           = (idx_full - i) / smooth_bandwidth
        kw          = pw * np.exp(-0.5 * d * d)
        smoothed[i] = np.sum(kw * pv) / np.sum(kw)

    smoothed = np.clip(smoothed, 0.1, 20.0)
    return bin_mags, smoothed, bin_raw, bin_unc, extrap_mags, extrap_chi2

def inflate_chi2(records, zero_point, verbose=False):
    """Apply magnitude-dependent chi²-scaling to covariances across *records* in-place.

    Bins useful stars into equal-count flux bins, computes the median chi² in
    each bin, then interpolates (linearly in log-flux) to assign every star a
    chi2_scale = chi2_eff(flux).  The scaling is applied regardless of whether
    chi2_eff > 1 or < 1, so overestimated noise is corrected downward too.

    Returns a dict with the correction curve for diagnostic plotting:
        {'flux_bins': 1D array, 'chi2_medians': 1D array, 'n_bins': int}

    Call this once on the full multi-chip catalogue (from run_photometry_fits)
    so the correction curve is derived from all chips together.
    """
    from .utils import mag_from_flux as _mff

    useful = [(r.flux, r.chi2) for r in records
              if np.isfinite(r.chi2) and r.chi2 < 10.0 and r.flux > 1.1
              and getattr(r, 'is_star_candidate', r.qfit < 2.0)]

    correction_info = {'flux_bins': np.array([]), 'chi2_medians': np.array([]), 'n_bins': 0}

    if len(useful) < 10:
        chi2_global = float(np.median([c for _, c in useful])) if useful else 1.0
        bin_lf   = np.array([0.0, 10.0])
        bin_chi2 = np.array([chi2_global, chi2_global])
        bin_chi2_raw = bin_chi2.copy()
        bin_chi2_unc = np.array([0.5, 0.5])
        if verbose:
            print(f"  chi²_global (too few stars, single value): {chi2_global:.3f}")
    else:
        fluxes = np.array([f for f, _ in useful])
        chi2s  = np.array([c for _, c in useful])
        mags   = zero_point - 2.5 * np.log10(np.maximum(fluxes, 1.0))

        result = _build_chi2_mag_bins(mags, chi2s)
        if result is None:
            chi2_global = float(np.median(chi2s))
            bin_lf   = np.array([0.0, 10.0])
            bin_chi2 = np.array([chi2_global, chi2_global])
            bin_chi2_raw = bin_chi2.copy()
            bin_chi2_unc = np.array([0.5, 0.5])
        else:
            _bin_mags, _bin_smooth, _bin_raw, _bin_unc, _extrap_mags, _extrap_chi2 = result
            # Prepend extrapolated bright-end points so the PCHIP covers stars
            # brighter than the first measured bin without flat-clamping.
            if _extrap_mags.size:
                _all_mags  = np.concatenate([_extrap_mags, _bin_mags])
                _all_chi2  = np.concatenate([_extrap_chi2, _bin_smooth])
            else:
                _all_mags  = _bin_mags
                _all_chi2  = _bin_smooth
            # Convert mag → log-flux; sort ascending lf (faint→bright)
            _bin_lf = (zero_point - _all_mags) / 2.5
            sort_lf = np.argsort(_bin_lf)
            bin_lf       = _bin_lf[sort_lf]
            bin_chi2     = _all_chi2[sort_lf]
            bin_chi2_raw = np.concatenate([np.full(len(_extrap_mags), np.nan), _bin_raw])[sort_lf]
            bin_chi2_unc = np.concatenate([np.full(len(_extrap_mags), np.nan), _bin_unc])[sort_lf]

        correction_info = {
            'flux_bins':    10 ** bin_lf,
            'chi2_medians': bin_chi2,
            'chi2_raw':     bin_chi2_raw,
            'chi2_unc':     bin_chi2_unc,
            'n_bins':       len(bin_lf),
        }
        if verbose:
            print(f"  chi²_scale: {len(bin_lf)} mag bins, "
                  f"range [{bin_chi2.min():.3f}, {bin_chi2.max():.3f}]")

    # Smooth interpolator: PchipInterpolator gives a C¹ shape-preserving curve.
    # Extrapolation is clamped to the endpoint values.
    if len(bin_lf) >= 2:
        from scipy.interpolate import PchipInterpolator as _Pchip
        _pchip = _Pchip(bin_lf, bin_chi2)

        def _eval_chi2_eff(lf_val):
            if lf_val <= bin_lf[0]:
                return float(bin_chi2[0])
            if lf_val >= bin_lf[-1]:
                return float(bin_chi2[-1])
            return float(np.clip(_pchip(lf_val), 0.1, 20.0))
    else:
        def _eval_chi2_eff(lf_val):
            return float(bin_chi2[0])

    for r in records:
        if not np.isfinite(r.chi2) or r.qfit >= 9.0:
            r.chi2_scale = 1.0
            continue
        lf = float(np.log10(max(r.flux, 1.0)))
        chi2_eff = max(_eval_chi2_eff(lf), 0.1)
        chi2_residual = r.chi2 / chi2_eff
        total_scale = r.chi2 if chi2_residual > 1.5 else chi2_eff
        r.cov      = r.cov * (total_scale ** 2)
        r.flux_err = float(np.sqrt(max(r.cov[0, 0], 0.0)))
        r.sky_err  = float(np.sqrt(max(r.cov[3, 3], 0.0)))
        _, r.mag_err = _mff(max(r.flux, 1e-10), r.flux_err, zero_point)
        r.chi2_scale = total_scale

    return correction_info


# ---------------------------------------------------------------------------
# Top-level photometry driver
# ---------------------------------------------------------------------------

def run_photometry(
    data,
    psf_models,
    psf_positions=None,
    psf_scale=4,
    half_width=3,
    sky_inner=4,
    sky_outer=8,
    hmin=4,
    fmin=0.0,
    max_iter_fit=15,
    tol=1e-3,
    n_passes=1,
    n_discovery_passes=None,
    gain=1.0,
    read_noise=5.0,
    zero_point=0.0,
    mask=None,
    peak_mask=None,
    noise_map=None,
    verbose=False,
    x_offset=0.0,
    y_offset=0.0,
    suppress_radius=None,
    n_jobs=-1,
    sat_threshold=np.inf,
    sigma_clip=True,
    sigma_clip_sigma=4.0,
    sigma_clip_iter=2,
    return_residual=False,
    _apply_chi2_inflation=True,
    _classify=True,
    backend='auto',
    conc_limit=0.9,
):
    """Find and measure point sources in *data* by PSF fitting.

    Parameters
    ----------
    n_discovery_passes : int or None
        How many passes include new-source detection.  Subsequent passes only
        re-fit already-found stars via leave-one-out on the residual image.
        None (default) → n_passes - 1 for n_passes > 1, else n_passes
        (i.e. the last pass is always refit-only when n_passes > 1).
        Set to n_passes to run discovery on every pass; 0 to skip discovery
        entirely after pass 1 (not recommended; pass 1 always discovers).
    hmin : int
        Non-maximum suppression radius (pixels) — same convention as hst1pass.
        Only the brightest local maximum within a (2·hmin+1)² box is kept on
        each discovery pass.  Larger values reduce spurious detections in
        crowded fields at the cost of blended-source completeness.
    suppress_radius : int or None
        Overrides *hmin* for pass-1 suppression when not None.  Rarely needed;
        prefer setting *hmin* directly.
    n_jobs : int
        Number of parallel worker threads for star fitting.
        -1 (default) uses all available CPU cores.
        1 disables parallelism (useful for debugging or profiling).
        Threads share the image and PSF arrays with no serialization overhead.

    All other parameters: see module docstring / CLAUDE.md.
    """
    from .multipass import subtract_stars, restore_stars, refit_stars, build_variance_image, deduplicate_records
    from ._backend import resolve_backend

    # Validate backend early and fail fast if JAX is explicitly requested but
    # unavailable.  The actual per-pass dispatch happens after find_sources so
    # that the resolved star count drives the auto threshold.
    _backend_req = backend
    if _backend_req == 'jax':
        resolve_backend('jax', 0)  # raises ImportError if JAX not installed

    if psf_models.ndim == 2:
        psf_cube = psf_models[np.newaxis].astype(np.float64)
    else:
        psf_cube = np.asarray(psf_models, dtype=np.float64)

    if psf_positions is None:
        xs = np.array([0.0]);  ys = np.array([0.0])
    else:
        xs, ys = psf_positions
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)

    data = np.asarray(data, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
    if noise_map is not None:
        noise_map = np.asarray(noise_map, dtype=np.float64)

    # Prefilter the entire PSF cube once.  spline_filter is linear, so
    # bilinear combinations of prefiltered arrays equal the prefiltered
    # bilinear combination — interpolate_psf on the coefficient cube gives
    # the correct coefficients for any detector position without per-star
    # recomputation.
    psf_coeffs_cube = np.array([
        spline_filter(p, order=3, output=np.float64) for p in psf_cube
    ])

    # PSF peak value and 3×3 core patch (central PSF model, dx=dy=0).
    # Used in find_sources for the fmin pre-filter: the 3×3 patch enables a
    # noise-weighted matched-filter flux estimate over the core rather than a
    # single-pixel estimate, which is more robust to sub-pixel centering errors
    # and correctly handles masked pixels in the core region.
    _psf_peak_val = float(psf_cube.max())
    _cen_psf  = psf_cube[len(psf_cube) // 2]
    _half_psf = _cen_psf.shape[0] // 2
    _sc       = int(psf_scale)
    _off3     = np.array([-1, 0, 1], dtype=int)
    _DIJ, _DII = np.meshgrid(_off3 * _sc, _off3 * _sc, indexing='ij')
    _psf_core_3x3 = _cen_psf[_half_psf + _DIJ, _half_psf + _DII].astype(np.float64)

    # Trigger Numba JIT in the main thread before parallel workers start so
    # all threads share the cached compiled code rather than racing to compile.
    if _NUMBA:
        _dum_y = np.array([50.0], dtype=np.float64)
        _dum_x = np.array([50.0], dtype=np.float64)
        _P = np.empty(1); _G = np.empty(1)
        _nb_eval_psf_grad(psf_coeffs_cube[0], _dum_y, _dum_x,
                          float(psf_scale), _P, _G, _G.copy())

    # Default suppression: use sky_inner for pass 1 (keeps only brightest
    # peak within the sky-annulus footprint), 1 for later passes.
    sr_pass1 = sky_inner if suppress_radius is None else suppress_radius

    # Resolve n_discovery_passes: pass 1 always discovers; default is to skip
    # discovery on the final pass (all stars already found, just refit).
    if n_discovery_passes is None:
        n_discovery_passes = n_passes - 1 if n_passes > 1 else n_passes

    # Import parallel utilities once; fall back gracefully if joblib absent.
    try:
        from joblib import Parallel, delayed as _delayed
        _have_joblib = True
    except ImportError:
        _have_joblib = False

    try:
        from tqdm import tqdm as _tqdm
        _have_tqdm = True
    except ImportError:
        _have_tqdm = False

    if verbose:
        _nm = 'external noise_map' if noise_map is not None else \
              f'Poisson model  (σ² = max(data, model+sky)/gain + (RN/gain)²)'
        print(f"  Noise model   : {_nm}")
        if n_passes > 1:
            print(f"  PSF model err : per-star eps_psf inflates refit noise (pass 2+)")
        print(f"  Gain (noise)  : {gain:.4f}  "
              f"{'[= 1 because image is in electrons]' if gain == 1.0 else '[e-/DN]'}")
        print(f"  Read noise    : {read_noise:.2f} e-")
        print(f"  Image size    : {data.shape[1]} × {data.shape[0]} px")
        print(f"  PSF grid      : {psf_cube.shape[0]} PSFs  "
              f"(scale={psf_scale}×,  array={psf_cube.shape[-1]}×{psf_cube.shape[-2]} px)")
        print(f"  Fit window    : {2*half_width+1}×{2*half_width+1} px  (hw={half_width})")
        print(f"  Sigma clip    : {'on' if sigma_clip else 'off'}"
              f"  (σ={sigma_clip_sigma}, iter={sigma_clip_iter})")

    all_records = []
    residual = data.copy()

    # Per-run PSF interpolation cache.  Keyed by (x_det//5, y_det//5); stores
    # the bicubic-interpolated PSF array so stars in the same 5-px cell share
    # the result without recomputing the einsum contraction.
    _psf_cache: dict = {}

    # Failure diagnostic counter — populated by fit_star when _fail_counter is passed.
    _fail_counter: dict = {}

    # Tracks the best current variance estimate for fitting.  Starts as the
    # user-supplied noise_map (or None → local Poisson model) and is replaced
    # with a full star-aware variance image after each discovery pass / refit,
    # so every subsequent fitting step benefits from neighbour Poisson noise.
    _current_noise_map = noise_map

    # Shared kwargs for fit_star — noise_map is updated in-place below.
    _fit_kw = dict(
        psf_cube=psf_cube, xs=xs, ys=ys,
        psf_scale=psf_scale, hw=half_width,
        gain=gain, read_noise=read_noise,
        max_iter=max_iter_fit, tol=tol,
        noise_map=_current_noise_map, mask=mask,
        x_offset=x_offset, y_offset=y_offset,
        zero_point=zero_point,
        psf_coeffs_cube=psf_coeffs_cube,
        sat_threshold=sat_threshold,
        sigma_clip=sigma_clip,
        sigma_clip_sigma=sigma_clip_sigma,
        sigma_clip_iter=sigma_clip_iter,
        psf_cache=_psf_cache,
        _fail_counter=_fail_counter,
    )

    def _rebuild_var(label=""):
        """Rebuild the star-aware variance image and update _fit_kw."""
        nonlocal _current_noise_map
        _current_noise_map = build_variance_image(
            all_records, psf_cube, xs, ys, psf_scale, data.shape,
            gain, read_noise, x_offset, y_offset,
            noise_map=noise_map,
            psf_coeffs_cube=psf_coeffs_cube, psf_cache=_psf_cache)
        _fit_kw['noise_map'] = _current_noise_map
        if verbose and label:
            n_sub = sum(1 for r in all_records
                        if r.qfit < 2.0 and r.chi2 < 5.0
                        and getattr(r, 'converged', True))
            print(f"  Variance image {label} ({n_sub} stars contribute)")

    for pass_num in range(1, n_passes + 1):
        do_discovery = pass_num <= n_discovery_passes

        # --- Pass 2+: leave-one-out re-fitting of all known stars first ---
        if pass_num > 1 and all_records:
            _refit_backend = resolve_backend(_backend_req, len(all_records))
            if _refit_backend == 'jax':
                from .multipass import refit_stars_jax
                if verbose:
                    print(f"Pass {pass_num}: re-fitting {len(all_records)} "
                          f"stars (JAX batch)...")
                refit_stars_jax(
                    residual, all_records, psf_cube, xs, ys, psf_scale,
                    half_width, gain, read_noise, mask, _current_noise_map,
                    x_offset, y_offset, zero_point, max_iter_fit, tol,
                    psf_coeffs_cube=psf_coeffs_cube,
                    sat_threshold=sat_threshold,
                    verbose=verbose,
                    sigma_clip=sigma_clip,
                    sigma_clip_sigma=sigma_clip_sigma,
                    sigma_clip_iter=sigma_clip_iter,
                    psf_cache=_psf_cache,
                    n_jobs=n_jobs,
                )
            else:
                refit_stars(
                    residual, all_records, psf_cube, xs, ys, psf_scale,
                    half_width, gain, read_noise, mask, _current_noise_map,
                    x_offset, y_offset, zero_point, max_iter_fit, tol,
                    psf_coeffs_cube=psf_coeffs_cube,
                    sat_threshold=sat_threshold,
                    verbose=verbose, desc=f"Pass {pass_num} re-fitting",
                    sigma_clip=sigma_clip,
                    sigma_clip_sigma=sigma_clip_sigma,
                    sigma_clip_iter=sigma_clip_iter,
                    psf_cache=_psf_cache,
                )
            # Dedup after refit: positions may drift during leave-one-out
            # fitting, bringing two records that were originally distinct into
            # the same pixel.  Removed duplicates were already subtracted by
            # refit_stars, so restore their flux to keep the residual clean.
            kept_records, n_dupes_refit = deduplicate_records(all_records, 2.0)
            if n_dupes_refit:
                _kept_ids = {id(r) for r in kept_records}
                removed = [r for r in all_records if id(r) not in _kept_ids]
                restore_stars(residual, removed, psf_cube, xs, ys,
                              psf_scale, half_width, x_offset, y_offset,
                              psf_coeffs_cube=psf_coeffs_cube,
                              psf_cache=_psf_cache)
                all_records[:] = kept_records

            if verbose:
                n_conv = sum(1 for r in all_records if getattr(r, 'converged', True))
                n_fail = len(all_records) - n_conv
                _conv_str = f"{n_conv} converged"
                if n_fail:
                    _conv_str += f", {n_fail} did not"
                _dedup_str = (f", {n_dupes_refit} duplicate{'s' if n_dupes_refit != 1 else ''} removed"
                              if n_dupes_refit else "")
                print(f"Pass {pass_num}: re-fit {len(all_records)} stars "
                      f"({_conv_str}{_dedup_str})")

            # Rebuild after refit: fluxes have changed, so the Poisson noise
            # estimate for subsequent discovery in this same pass is improved.
            _rebuild_var("rebuilt after re-fit")

        # --- Discovery: find and fit new sources in the current residual ---
        if do_discovery:
            # sr = sr_pass1 if pass_num == 1 else 1
            # sr = sr_pass1 if pass_num == 1 else sr_pass1
            sr = hmin

            if verbose:
                print(f"Pass {pass_num}: finding sources "
                      f"(suppress_radius={sr})...")

            xs_c, ys_c, peaks_c, skys_c, _ = find_sources(
                residual, sky_inner, sky_outer, hmin, fmin, mask,
                suppress_radius=sr, peak_mask=peak_mask, verbose=verbose,
                psf_peak_val=_psf_peak_val, psf_core_3x3=_psf_core_3x3)

            n_cand = len(xs_c)

            # Break pixel-phase bias before dispatch so both backends see the
            # same jittered starting positions.  Seed is deterministic so runs
            # are reproducible regardless of which backend is active.
            if n_cand > 0:
                _rng = np.random.default_rng(seed=pass_num)
                xs_c = xs_c.astype(float) + _rng.uniform(-0.5, 0.5, n_cand)
                ys_c = ys_c.astype(float) + _rng.uniform(-0.5, 0.5, n_cand)

            # Resolve backend now that we know the star count.
            _active_backend = resolve_backend(_backend_req, n_cand)

            if _active_backend == 'jax':
                from ._jax_kernel import (
                    prepare_jax_inputs, fit_batch_jax,
                    _sigma_clip_jax_results,
                )

                if verbose:
                    print(f"Pass {pass_num}: {n_cand} candidates, fitting (JAX)...")

                _jax_inputs = prepare_jax_inputs(
                    residual, xs_c, ys_c, skys_c,
                    psf_cube, xs, ys,
                    psf_scale, half_width,
                    mask=mask, noise_map=_current_noise_map,
                    gain=gain, read_noise=read_noise,
                    x_offset=x_offset, y_offset=y_offset,
                    psf_coeffs_cube=psf_coeffs_cube,
                    n_jobs=n_jobs,
                )
                _jax_res = fit_batch_jax(
                    _jax_inputs, gain=gain, tol=tol, max_iter=max_iter_fit,
                )
                if sigma_clip and sigma_clip_iter > 0 and n_cand > 0:
                    _jax_res = _sigma_clip_jax_results(
                        _jax_res, _jax_inputs,
                        gain=gain,
                        sigma_clip_sigma=sigma_clip_sigma,
                        sigma_clip_iter=sigma_clip_iter,
                    )
                pass_records = _jax_results_to_records(
                    _jax_res, _jax_inputs,
                    pass_number=pass_num,
                    gain=gain, zero_point=zero_point,
                    sat_threshold=sat_threshold,
                )

            else:
                if verbose:
                    print(f"Pass {pass_num}: {n_cand} candidates, fitting...")

                use_parallel = _have_joblib and n_jobs != 1 and n_cand > 0

                if use_parallel:
                    # Threads share the read-only residual, PSF cube, and mask
                    # arrays with no serialization cost; NumPy releases the GIL
                    # during its internal C loops for genuine parallel throughput.
                    _fit = _delayed(fit_star)

                    if _have_tqdm and verbose:
                        from contextlib import contextmanager
                        import joblib as _jl

                        @contextmanager
                        def _tqdm_joblib(bar):
                            class _CB(_jl.parallel.BatchCompletionCallBack):
                                def __call__(self, *a, **kw):
                                    bar.update(n=self.batch_size)
                                    return super().__call__(*a, **kw)
                            _orig = _jl.parallel.BatchCompletionCallBack
                            _jl.parallel.BatchCompletionCallBack = _CB
                            try:
                                yield bar
                            finally:
                                _jl.parallel.BatchCompletionCallBack = _orig
                                bar.close()

                        bar = _tqdm(total=n_cand,
                                    desc=f"Pass {pass_num} fitting",
                                    unit="star")
                        with _tqdm_joblib(bar):
                            pass_records = Parallel(
                                n_jobs=n_jobs, backend="threading"
                            )(
                                _fit(data=residual,
                                     x0=float(xs_c[i]), y0=float(ys_c[i]),
                                     sky=float(skys_c[i]), pass_number=pass_num,
                                     **_fit_kw)
                                for i in range(n_cand)
                            )
                    else:
                        pass_records = Parallel(
                            n_jobs=n_jobs, backend="threading"
                        )(
                            _fit(data=residual,
                                 x0=float(xs_c[i]), y0=float(ys_c[i]),
                                 sky=float(skys_c[i]), pass_number=pass_num,
                                 **_fit_kw)
                            for i in range(n_cand)
                        )
                else:
                    _iterator = (
                        _tqdm(range(n_cand), desc=f"Pass {pass_num} fitting",
                              unit="star")
                        if (_have_tqdm and verbose) else range(n_cand)
                    )
                    pass_records = []
                    for i in _iterator:
                        rec = fit_star(
                            data=residual,
                            x0=float(xs_c[i]), y0=float(ys_c[i]),
                            sky=float(skys_c[i]), pass_number=pass_num,
                            **_fit_kw,
                        )
                        pass_records.append(rec)

            # De-duplicate: sources initialised in the wings of a masked star
            # can converge to the same position.  Keeping all copies would
            # subtract that star multiple times from the residual.
            # Remove any new record within 2 px of a brighter record.
            # Using 2 px (not hmin) because hmin governs peak-finding suppression;
            # fitted positions that converged to the same star land within ~1 px
            # of each other, whereas genuine nearby pairs are typically 2+ px apart.
            pass_records, n_dupes = deduplicate_records(
                pass_records, 2.0, existing_records=all_records)

            if verbose:
                n_conv = sum(1 for r in pass_records if getattr(r, 'converged', True))
                n_fail = len(pass_records) - n_conv
                _conv_str = f"{n_conv} converged"
                if n_fail:
                    _conv_str += f", {n_fail} did not"
                _dedup_str = (f", {n_dupes} duplicate{'s' if n_dupes != 1 else ''} removed"
                              if n_dupes else "")
                print(f"Pass {pass_num}: {n_cand} candidates → "
                      f"{len(pass_records)} kept "
                      f"({_conv_str}{_dedup_str})  "
                      f"total: {len(all_records) + len(pass_records)}")

            all_records.extend(pass_records)

            # Subtract newly found stars from the residual so the next pass
            # (whether discovery or refit-only) sees a clean image.
            subtract_stars(residual, pass_records, psf_cube, xs, ys,
                           psf_scale, half_width, x_offset, y_offset,
                           psf_coeffs_cube=psf_coeffs_cube,
                           psf_cache=_psf_cache)
            # Build star-aware variance image for the next pass.
            _rebuild_var("built after discovery")

        else:
            # Refit-only pass: refit_stars already ran above and printed stats.
            pass

    # Build the final star-aware variance image.  Used for chi2 reporting in
    # diagnostics and saved alongside the residual in residual.fits.
    _final_var_image = build_variance_image(
        all_records, psf_cube, xs, ys, psf_scale, data.shape,
        gain, read_noise, x_offset, y_offset,
        noise_map=noise_map,
        psf_coeffs_cube=psf_coeffs_cube, psf_cache=_psf_cache)

    # Classify each source as likely star or non-star before chi²-inflation so
    # the inflation curve is built from the stellar locus only.
    # When called from run_photometry_fits(), _classify=False defers this step
    # so that classification uses the full multi-chip catalogue instead of each
    # chip individually.
    if _classify:
        n_star_cand = classify_stars(all_records, conc_lo=conc_limit)
        if verbose:
            n_tot = len(all_records)
            print(f"  Star classification: {n_star_cand}/{n_tot} sources classified "
                  f"as likely stars  ({100.0*n_star_cand/max(n_tot,1):.1f}%)")

    if _apply_chi2_inflation:
        inflate_chi2(all_records, zero_point, verbose=verbose)

    # Compute neighbour statistics on the final catalogue.  Uses the fully
    # converged positions from all passes so the distances are as accurate
    # as possible.
    compute_neighbor_stats(all_records, half_width)

    n_total = len(all_records)
    n_not_converged = sum(1 for r in all_records if not r.converged)
    if verbose and n_not_converged:
        nc_deltas = np.array([r.delta_max for r in all_records
                              if not r.converged and np.isfinite(r.delta_max)])
        delta_info = (f"  final |δ|: median={np.median(nc_deltas):.4f}px, "
                      f"max={nc_deltas.max():.4f}px"
                      if nc_deltas.size else "")
        frac = 100.0 * n_not_converged / max(n_total, 1)
        hint = ""
        if nc_deltas.size and np.median(nc_deltas) < 10 * tol:
            hint = " → close to convergence; try --max_iter or tighter --tol"
        elif nc_deltas.size and np.median(nc_deltas) > 0.1:
            hint = " → large residual motion; likely crowded or poor PSF match"
        print(f"  Non-converged stars (hit max_iter={max_iter_fit}): "
              f"{n_not_converged} / {n_total}  ({frac:.1f}%){delta_info}{hint}")

    if verbose and _fail_counter:
        total_fail = sum(v if isinstance(v, int) else 0
                         for v in _fail_counter.values())
        print(f"\n  Fit failure breakdown ({total_fail} stars discarded before "
              f"measurement, {100*total_fail/max(n_total+total_fail,1):.1f}% of candidates):")
        if _fail_counter.get('few_good_px'):
            n_good_list = _fail_counter.get('few_good_px_n_good', [])
            rng = (f"  (had {min(n_good_list)}–{max(n_good_list)} unmasked pix, need ≥5)"
                   if n_good_list else "")
            print(f"    {_fail_counter['few_good_px']:5d}  Not enough unmasked pixels{rng}"
                  f"\n           → star on/near DQ-masked region or image edge")
        if _fail_counter.get('shape_mismatch'):
            print(f"    {_fail_counter['shape_mismatch']:5d}  PSF/window shape mismatch"
                  f"\n           → star clipped to image boundary mid-iteration")
        if _fail_counter.get('linalg_error'):
            print(f"    {_fail_counter['linalg_error']:5d}  Singular normal equations "
                  f"(AᵀWA not invertible)"
                  f"\n           → all pixels masked, zero-flux, or degenerate window")
        if _fail_counter.get('pos_clamp'):
            steps = _fail_counter.get('pos_clamp_steps', [])
            iters = np.array(_fail_counter.get('pos_clamp_iter', []), dtype=int)
            fluxs = np.array(_fail_counter.get('pos_clamp_flux', []))
            conds = np.array(_fail_counter.get('pos_clamp_cond', []))
            n_pc  = _fail_counter['pos_clamp']
            if steps:
                sa    = np.array([max(dx, dy) for dx, dy in steps])
                x_dom = sum(1 for dx, dy in steps if dx >= dy)
                y_dom = n_pc - x_dom
                # Break down by when the failure occurs
                n_iter0  = int(np.sum(iters == 0))
                n_iter1p = n_pc - n_iter0
                # Flux distribution at time of failure
                n_flux_lo = int(np.sum(fluxs < 10))   # effectively at 1.0 floor
                n_flux_hi = n_pc - n_flux_lo
                # Condition number
                cond_med = float(np.median(conds)) if conds.size else float('nan')
                print(f"    {n_pc:5d}  Newton step exploded (|δ|>{2*half_width}px threshold)")
                print(f"           step sizes  : median={np.median(sa):.1f}px, "
                      f"max={sa.max():.1f}px  (x-dominant: {x_dom}, y-dominant: {y_dom})")
                sky_drifts = np.array(_fail_counter.get('pos_clamp_sky_drift', []))
                print(f"           when failed : iter=0 (1st step): {n_iter0}  "
                      f"({100*n_iter0/n_pc:.0f}%)   |   iter≥1: {n_iter1p}  "
                      f"({100*n_iter1p/n_pc:.0f}%)")
                print(f"           flux at fail: flux<10 (near-floor): {n_flux_lo}  "
                      f"({100*n_flux_lo/n_pc:.0f}%)   |   flux≥10: {n_flux_hi}  "
                      f"({100*n_flux_hi/n_pc:.0f}%)")
                if sky_drifts.size:
                    print(f"           sky drift   : median={np.median(sky_drifts):+.1f}  "
                          f"p10={np.percentile(sky_drifts, 10):+.1f}  "
                          f"p90={np.percentile(sky_drifts, 90):+.1f}  "
                          f"(sky_final − sky_init; large +ve → sky runaway drove flux to floor)")
                print(f"           AᵀWA cond # : median={cond_med:.2e}")
                sky_drift_large = (sky_drifts.size > 0 and
                                   float(np.median(np.abs(sky_drifts))) > 50)
                if n_iter0 > 0.9 * n_pc and n_flux_lo > 0.5 * n_pc:
                    print(f"           DIAGNOSIS: bad initialisation — "
                          f"flux at floor before first Newton step\n"
                          f"           check sky overestimation or masked peak pixel")
                elif n_iter0 > 0.9 * n_pc and n_flux_lo < 0.1 * n_pc:
                    print(f"           DIAGNOSIS: near-singular AᵀWA on first step "
                          f"despite correct flux\n"
                          f"           check PSF derivatives or systematic bad-pixel pattern")
                elif n_flux_lo > 0.5 * n_pc and sky_drift_large:
                    print(f"           DIAGNOSIS: sky runaway — sky estimate drifted "
                          f"during Newton, driving flux to floor\n"
                          f"           consider tighter sky annulus or bad-pixel masking")
                elif n_flux_lo > 0.5 * n_pc and not sky_drift_large:
                    print(f"           DIAGNOSIS: PSF mismatch — solver drives flux→0 "
                          f"because a flat model fits better than any PSF\n"
                          f"           sky barely moved ({float(np.median(sky_drifts)):+.1f} "
                          f"median drift); these are likely galaxies / extended sources\n"
                          f"           PSF fitting correctly rejects non-point-source detections")
                else:
                    print(f"           DIAGNOSIS: mixed — some extended sources, "
                          f"some blends, some edge cases")
            else:
                print(f"    {n_pc:5d}  Newton step exploded (|δ|>{2*half_width}px)")
        if _fail_counter.get('out_of_bounds'):
            details = _fail_counter.get('out_of_bounds_detail', [])
            example = f"  e.g. {details[0]}" if details else ""
            print(f"    {_fail_counter['out_of_bounds']:5d}  Position drifted outside "
                  f"image boundary{example}"
                  f"\n           → Newton diverging on a near-empty window")

    if return_residual:
        return all_records, residual, _final_var_image
    return all_records
