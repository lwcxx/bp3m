"""Multi-pass star subtraction and leave-one-out re-fitting."""

import numpy as np
from scipy.ndimage import spline_filter
from scipy.spatial import cKDTree

from .core import interpolate_psf, _eval_psf_grad_fast, _window_offsets, fit_star


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sub_hw(psf_cube, psf_scale):
    """Half-width in detector pixels covering the full PSF array for subtraction."""
    psf_size = psf_cube.shape[-1]
    return (psf_size - 1) // (2 * int(psf_scale))


def _should_subtract(rec):
    """True if this star's PSF model should be removed from the residual image.

    More permissive than the old qfit < 0.5 cutoff: bright stars with a
    slightly-blended or noisy fit often have qfit in 0.5–2 but a valid
    converged position.  Excluding them leaves their flux in the residual
    and biases all nearby neighbours.

      qfit < 2  — rejects only unmistakable artefacts / extended sources
      chi2 < 5  — rejects catastrophically failed Newton solutions
                  (rec.chi2 == inf for _failed_record, so inf < 5 is False)
      converged — position must have passed the convergence test
    """
    return (rec.qfit < 2.0 and
            rec.chi2 < 5.0 and
            getattr(rec, 'converged', True) and
            rec.flux > 0)


def _psf_window(rec, cube, xs, ys, psf_scale, hw, ny, nx,
                x_offset, y_offset, prefiltered, psf_cache=None,
                hw_override=None):
    """Return (P, y_lo, y_hi, x_lo, x_hi) for the PSF footprint of *rec*."""
    xi = int(round(rec.x)); yi = int(round(rec.y))
    _hw = hw_override if hw_override is not None else hw
    y_lo, y_hi, x_lo, x_hi, diy, dix = _window_offsets(xi, yi, _hw, ny, nx)
    dx = rec.x - xi; dy = rec.y - yi
    local_psf = interpolate_psf(cube, xs, ys, rec.x + x_offset, rec.y + y_offset,
                                _cache=psf_cache)
    coeffs = local_psf if prefiltered else \
             spline_filter(local_psf, order=3, output=np.float64)
    P, _, _ = _eval_psf_grad_fast(coeffs, dx, dy, dix, diy, psf_scale)
    return P, y_lo, y_hi, x_lo, x_hi


# ---------------------------------------------------------------------------
# Duplicate removal
# ---------------------------------------------------------------------------

def deduplicate_records(new_records, hmin, existing_records=None):
    """Remove sources within *hmin* pixels of a brighter source.

    After PSF fitting, sources that were initialised in the wings of a bright
    star (because its core was masked) can all converge to the same position.
    Building a residual image from all of them would subtract that star
    multiple times.  This function removes the fainter copies.

    Parameters
    ----------
    new_records      : list of StarRecord — just-fitted sources to filter
    hmin             : float — proximity radius in detector pixels
    existing_records : list of StarRecord or None — already-accepted stars
                       from earlier passes.  Any new record within *hmin* of
                       an existing record is removed regardless of brightness.

    Returns
    -------
    kept     : list of StarRecord — de-duplicated subset of *new_records*
    n_removed: int — number of duplicates removed
    """
    if len(new_records) == 0:
        return new_records, 0

    combined = list(new_records) + (list(existing_records) if existing_records else [])
    positions = np.array([[r.x, r.y] for r in combined], dtype=float)
    fluxes    = np.array([r.flux     for r in combined], dtype=float)
    n_new     = len(new_records)

    keep = np.ones(len(combined), dtype=bool)

    if len(combined) > 1:
        tree = cKDTree(positions)
        # Process brightest-first so the brightest of any clump survives
        order = np.argsort(-fluxes)
        for idx in order:
            if not keep[idx]:
                continue
            neighbors = tree.query_ball_point(positions[idx], hmin)
            for nb in neighbors:
                if nb != idx:
                    keep[nb] = False

    n_removed = int((~keep[:n_new]).sum())
    kept = [new_records[i] for i in range(n_new) if keep[i]]
    return kept, n_removed


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def build_variance_image(records, psf_cube, xs, ys, psf_scale, shape,
                         gain, read_noise, x_offset=0.0, y_offset=0.0,
                         noise_map=None, psf_coeffs_cube=None, psf_cache=None):
    """Build a pixel-wise variance image accumulating Poisson noise from all star models.

    var[y,x] = base_var[y,x] + Σ_i max(flux_i · P_i[y,x], 0) / gain

    The base variance is *noise_map* if provided (user-supplied external variance),
    otherwise a uniform floor of median_sky/gain + (RN/gain)^2.  Only stars
    passing _should_subtract are included (same criterion as subtract_stars).

    Call this after each fitting pass and pass the result as *noise_map* to
    subsequent fit_star / refit_stars calls so each pass benefits from improved
    neighbour Poisson noise estimates.

    Parameters
    ----------
    records   : list of StarRecord — current full catalogue
    shape     : (ny, nx) — image shape
    noise_map : 2D base variance array (user-supplied); star Poisson contributions
                are added on top.  None → uniform sky+RN floor.

    Returns
    -------
    var_image : 2D float64 array, same shape as the image
    """
    ny, nx = shape
    _cube = psf_coeffs_cube if psf_coeffs_cube is not None else psf_cube
    prefiltered = psf_coeffs_cube is not None
    shw = _sub_hw(_cube, psf_scale)

    if noise_map is not None:
        var_image = noise_map.copy()
    else:
        good_sky = [r.sky for r in records
                    if _should_subtract(r) and np.isfinite(r.sky)]
        median_sky = float(np.median(good_sky)) if good_sky else 0.0
        var_base = max(median_sky, 0.0) / gain + (read_noise / gain) ** 2
        var_image = np.full((ny, nx), var_base, dtype=np.float64)

    for rec in records:
        if not _should_subtract(rec):
            continue
        P, y_lo, y_hi, x_lo, x_hi = _psf_window(
            rec, _cube, xs, ys, psf_scale, hw=0, ny=ny, nx=nx,
            x_offset=x_offset, y_offset=y_offset, prefiltered=prefiltered,
            psf_cache=psf_cache, hw_override=shw)
        if y_lo >= y_hi or x_lo >= x_hi:
            continue
        var_image[y_lo:y_hi, x_lo:x_hi] += np.maximum(rec.flux * P, 0.0) / gain

    return np.maximum(var_image, 1e-10)


def subtract_stars(residual, records, psf_cube, xs, ys, psf_scale, hw,
                   x_offset=0.0, y_offset=0.0,
                   psf_coeffs_cube=None, psf_cache=None):
    """Subtract PSF models of well-fit stars from *residual* in-place.

    Stars not passing _should_subtract (qfit≥2, chi2≥5, or not converged) are
    skipped to avoid injecting artefacts from cosmic rays or extended sources.
    Pass psf_coeffs_cube (prefiltered B-spline coefficients) to skip the
    per-call spline_filter overhead.
    """
    ny, nx = residual.shape
    _cube = psf_coeffs_cube if psf_coeffs_cube is not None else psf_cube
    prefiltered = psf_coeffs_cube is not None

    shw = _sub_hw(_cube, psf_scale)  # subtraction half-width = full PSF extent
    for rec in records:
        if not _should_subtract(rec):
            continue
        P, y_lo, y_hi, x_lo, x_hi = _psf_window(
            rec, _cube, xs, ys, psf_scale, hw, ny, nx,
            x_offset, y_offset, prefiltered=prefiltered, psf_cache=psf_cache,
            hw_override=shw)
        residual[y_lo:y_hi, x_lo:x_hi] -= rec.flux * P


def restore_stars(residual, records, psf_cube, xs, ys, psf_scale, hw,
                  x_offset=0.0, y_offset=0.0,
                  psf_coeffs_cube=None, psf_cache=None):
    """Add PSF models back to *residual* in-place (inverse of subtract_stars).

    Used after deduplication on a refit pass: removed duplicate records were
    already subtracted by refit_stars' leave-one-out loop, so their flux must
    be restored to keep the residual consistent.
    """
    ny, nx = residual.shape
    _cube = psf_coeffs_cube if psf_coeffs_cube is not None else psf_cube
    prefiltered = psf_coeffs_cube is not None

    shw = _sub_hw(_cube, psf_scale)
    for rec in records:
        if not _should_subtract(rec):
            continue
        P, y_lo, y_hi, x_lo, x_hi = _psf_window(
            rec, _cube, xs, ys, psf_scale, hw, ny, nx,
            x_offset, y_offset, prefiltered=prefiltered, psf_cache=psf_cache,
            hw_override=shw)
        residual[y_lo:y_hi, x_lo:x_hi] += rec.flux * P


def refit_stars(residual, records, psf_cube, xs, ys, psf_scale, hw,
                gain, read_noise, mask, noise_map, x_offset, y_offset,
                zero_point, max_iter, tol, psf_coeffs_cube=None,
                sat_threshold=float('inf'), verbose=False, desc="Re-fitting",
                sigma_clip=True, sigma_clip_sigma=4.0, sigma_clip_iter=2,
                psf_cache=None):
    """Re-fit all stars in *records* via leave-one-out on *residual*.

    *residual* must equal the original image with ALL currently known star
    models already subtracted (the standard state between passes).

    For each star, sequentially (Gauss-Seidel order so each re-fit sees the
    most recently updated neighbours):

      1. Restore its current PSF model into *residual* — this star is now
         isolated with every other star already removed.
      2. Re-fit at its current position using fit_star.
      3. Subtract the updated model back into *residual*.
      4. Replace the StarRecord in-place with the improved measurement.

    Every star is re-tried regardless of its initial quality.  After the re-fit,
    the updated model is subtracted only if _should_subtract(new_rec) passes,
    maintaining a consistent residual state for subsequent stars.

    *records* is modified in-place.  Returns *records* for convenience.
    """
    ny, nx = residual.shape
    _cube = psf_coeffs_cube if psf_coeffs_cube is not None else psf_cube
    prefiltered = psf_coeffs_cube is not None

    try:
        from tqdm import tqdm as _tqdm
        _have_tqdm = True
    except ImportError:
        _have_tqdm = False

    iterator = (
        _tqdm(range(len(records)), desc=desc, unit="star")
        if (_have_tqdm and verbose) else range(len(records))
    )

    shw = _sub_hw(_cube, psf_scale)

    for i in iterator:
        rec = records[i]

        # Step 1 — restore this star's contribution, but only if it was
        # actually subtracted from the residual (i.e. it passed _should_subtract
        # on the previous round).  Restoring a star that was never subtracted
        # would add phantom flux to the residual and bias the refit.
        was_subtracted = _should_subtract(rec)
        P, y_lo, y_hi, x_lo, x_hi = _psf_window(
            rec, _cube, xs, ys, psf_scale, hw, ny, nx,
            x_offset, y_offset, prefiltered=prefiltered, psf_cache=psf_cache,
            hw_override=shw)
        if y_lo >= y_hi or x_lo >= x_hi:
            continue
        if was_subtracted:
            residual[y_lo:y_hi, x_lo:x_hi] += rec.flux * P

        # Step 2 — re-fit with all neighbours already subtracted.
        new_rec = fit_star(
            data=residual, x0=rec.x, y0=rec.y,
            psf_cube=psf_cube, xs=xs, ys=ys, psf_scale=psf_scale,
            hw=hw, sky=rec.sky, gain=gain, read_noise=read_noise,
            max_iter=max_iter, tol=tol, noise_map=noise_map, mask=mask,
            x_offset=x_offset, y_offset=y_offset,
            zero_point=zero_point, pass_number=rec.pass_number,
            psf_coeffs_cube=psf_coeffs_cube,
            sat_threshold=sat_threshold,
            sigma_clip=sigma_clip,
            sigma_clip_sigma=sigma_clip_sigma,
            sigma_clip_iter=sigma_clip_iter,
            psf_cache=psf_cache,
            eps_psf_star=getattr(rec, 'eps_psf', 0.0),
        )

        # Preserve n_sat from the original record: the residual image has had
        # PSF models subtracted, so pixel values may no longer exceed the
        # saturation threshold even for a genuinely saturated star.
        new_rec.n_sat = rec.n_sat

        # Step 3 — subtract the updated model, but only if the new fit passes
        # the quality threshold.  This keeps the residual state consistent:
        # stars that pass _should_subtract are always subtracted; those that
        # don't pass (extended sources, artefacts) stay in the residual.
        P2, y_lo2, y_hi2, x_lo2, x_hi2 = _psf_window(
            new_rec, _cube, xs, ys, psf_scale, hw, ny, nx,
            x_offset, y_offset, prefiltered=prefiltered, psf_cache=psf_cache,
            hw_override=shw)
        if y_lo2 < y_hi2 and x_lo2 < x_hi2 and _should_subtract(new_rec):
            residual[y_lo2:y_hi2, x_lo2:x_hi2] -= new_rec.flux * P2

        records[i] = new_rec

    return records


def refit_stars_jax(residual, records, psf_cube, xs, ys, psf_scale, hw,
                    gain, read_noise, mask, noise_map, x_offset, y_offset,
                    zero_point, max_iter, tol, psf_coeffs_cube=None,
                    sat_threshold=float('inf'), verbose=False,
                    sigma_clip=True, sigma_clip_sigma=4.0, sigma_clip_iter=2,
                    psf_cache=None, n_jobs=1):
    """Re-fit all stars via leave-one-out batch fitting using JAX.

    Jacobi (parallel) equivalent of ``refit_stars``: each star's pixel window
    is extracted from ``residual + restored_star_k`` simultaneously, then all
    stars are fit in one JAX batch.  The residual is updated in-place after
    the batch.  Accuracy difference vs Gauss-Seidel is negligible when PSF
    footprints don't strongly overlap.

    *records* is modified in-place.  Returns *records* for convenience.
    """
    from ._jax_kernel import (
        prepare_jax_inputs, fit_batch_jax, _sigma_clip_jax_results,
    )
    from .core import _jax_results_to_records

    n_stars = len(records)
    if n_stars == 0:
        return records

    was_subtracted = np.array([_should_subtract(r) for r in records], dtype=bool)
    restore_fluxes = np.array(
        [r.flux if was_subtracted[i] else 0.0 for i, r in enumerate(records)],
        dtype=np.float64,
    )

    xs_rec  = np.array([r.x   for r in records])
    ys_rec  = np.array([r.y   for r in records])
    sky_rec = np.array([r.sky for r in records])

    inputs = prepare_jax_inputs(
        residual, xs_rec, ys_rec, sky_rec,
        psf_cube, xs, ys,
        psf_scale, hw,
        mask=mask, noise_map=noise_map,
        gain=gain, read_noise=read_noise,
        x_offset=x_offset, y_offset=y_offset,
        psf_coeffs_cube=psf_coeffs_cube,
        restore_fluxes=restore_fluxes,
        n_jobs=n_jobs,
    )
    jax_res = fit_batch_jax(inputs, gain=gain, tol=tol, max_iter=max_iter)

    if sigma_clip and sigma_clip_iter > 0:
        jax_res = _sigma_clip_jax_results(
            jax_res, inputs, gain=gain,
            sigma_clip_sigma=sigma_clip_sigma,
            sigma_clip_iter=sigma_clip_iter,
        )

    # Use pass_number=0 as a placeholder; overridden per-star below.
    new_records = _jax_results_to_records(
        jax_res, inputs,
        pass_number=0,
        gain=gain, zero_point=zero_point,
        sat_threshold=sat_threshold,
    )

    ny, nx = residual.shape
    _cube      = psf_coeffs_cube if psf_coeffs_cube is not None else psf_cube
    prefiltered = psf_coeffs_cube is not None
    shw = _sub_hw(_cube, psf_scale)

    for i, (old_rec, new_rec) in enumerate(zip(records, new_records)):
        # Preserve fields that the batch path can't recover.
        new_rec.pass_number          = old_rec.pass_number
        new_rec.n_sat                = old_rec.n_sat
        new_rec.n_neighbors          = old_rec.n_neighbors
        new_rec.dist_nearest         = old_rec.dist_nearest
        new_rec.dist_nearest_brighter = old_rec.dist_nearest_brighter

        # Update residual using the full PSF footprint (shw).
        P_old, y_lo, y_hi, x_lo, x_hi = _psf_window(
            old_rec, _cube, xs, ys, psf_scale, hw, ny, nx,
            x_offset, y_offset, prefiltered=prefiltered, psf_cache=psf_cache,
            hw_override=shw)
        if was_subtracted[i] and y_lo < y_hi and x_lo < x_hi:
            residual[y_lo:y_hi, x_lo:x_hi] += old_rec.flux * P_old

        P_new, y_lo2, y_hi2, x_lo2, x_hi2 = _psf_window(
            new_rec, _cube, xs, ys, psf_scale, hw, ny, nx,
            x_offset, y_offset, prefiltered=prefiltered, psf_cache=psf_cache,
            hw_override=shw)
        if _should_subtract(new_rec) and y_lo2 < y_hi2 and x_lo2 < x_hi2:
            residual[y_lo2:y_hi2, x_lo2:x_hi2] -= new_rec.flux * P_new

        records[i] = new_rec

    return records
