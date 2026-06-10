"""
Step 5: Bayesian astrometric alignment and proper motion measurement (BP3M).

Calls bp3m directly via its Python API using the FLC pipeline data layout.
Results are written to:
    {output_dir}/{field}/BP3M_results/
        stellar_astrometry.csv      — per-star posterior PMs + positions
        image_transformations.csv   — per-image alignment parameters
        v_cov_marginalised.npy      — (N, 5, 5) full posterior covariance
        plots/                      — diagnostic figures

The ``stellar_astrometry.csv`` produced here is the primary science output
of the pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

def _ensure_bp3m(bp3m_dir=None):
    pass  # bp3m is installed as a package; no sys.path manipulation needed


def run_alignment(
    output_dir: Path,
    field_name: str,
    n_iter: int = 20,
    n_samples: int = 1000,
    clip_sigma: float = 4.5,
    poly_order: int = 1,
    split_ccd: bool = True,
    min_stars_split_ccd: int = 20,
    use_sparse: bool = False,
    inflate_hst_errors: bool = True,
    no_prefilter: bool = False,
    no_plots: bool = False,
    images: list[str] | None = None,
    remove_images: list[str] | None = None,
    restrict_filters: list[str] | None = None,
    restrict_instdet: list[str] | None = None,
    bp3m_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    use_influence_clip: bool = True,
    influence_d_thresh: float = 1.0,
    influence_sigma_min: float = 2.0,
    use_two_tier: bool = False,
    pos_err_floor: float = 5e-3,
    plot_residuals: bool = False,
    plot_influence: bool = False,
) -> Path:
    """
    Run BP3M Bayesian alignment on a field.

    Parameters
    ----------
    output_dir       : pipeline root directory
    field_name       : field subdirectory name
    n_iter           : maximum EM outer iterations
    n_samples        : posterior samples for marginalisation
    clip_sigma       : MAD sigma for outlier rejection (0 = disabled)
    poly_order       : polynomial order for image transformation (1 = linear)
    split_ccd        : split ACS/WFC images into independent CCD halves
    min_stars_split_ccd : minimum stars per CCD half to allow splitting (default 20)
    use_sparse       : use sparse Schur-complement solver (faster for mosaics)
    inflate_hst_errors: enable per-image HST error inflation
    no_prefilter     : skip Phase-0 pre-filter pass
    no_plots         : skip diagnostic plot generation
    images           : restrict to these image names (None = all)
    remove_images    : exclude these image names
    restrict_filters : keep only images with these HST filters
    restrict_instdet : keep only images from these instrument+detector combos
    bp3m_dir         : override default bp3m location
    checkpoint_dir   : save/load fitting checkpoint here
    use_influence_clip  : enable test-4 Cook's D influence clipping
    influence_d_thresh  : Cook's D threshold (default 1.0)
    influence_sigma_min : minimum sigma_resid for influence flagging (default 2.0)

    Returns
    -------
    Path to output directory ({output_dir}/{field}/BP3M_results/)
    """
    _ensure_bp3m(bp3m_dir)

    from bp3m.data_loader_flc import load_image_data_flc
    from bp3m.data_loader import build_index_maps
    from bp3m.solver import BP3MSolver
    from bp3m.solver_sparse import BP3MSolverSparse
    from bp3m.checkpointing import save_results

    import time
    import pandas as pd

    data_root   = Path(output_dir)
    output_bp3m = data_root / field_name / "BP3M_results"
    output_bp3m.mkdir(parents=True, exist_ok=True)

    print("\n" + "─"*50)
    print("Step 5: Bayesian alignment (BP3M)")
    print("─"*50)
    print(f"  n_iter={n_iter}  n_samples={n_samples}  "
          f"clip_sigma={clip_sigma}  poly_order={poly_order}")
    _cmd = (
        f"run_bp3m.py {field_name} --data-root {data_root}"
        f" --flc-pipeline"
        f" --n-iter {n_iter}"
        f" --n-samples {n_samples}"
        f" --clip-sigma {clip_sigma}"
        f" --poly-order {poly_order}"
        + (" --split-ccd"          if split_ccd else "")
        + (f" --min-stars-split-ccd {min_stars_split_ccd}" if split_ccd and min_stars_split_ccd != 20 else "")
        + (" --inflate-hst-errors" if inflate_hst_errors else "")
        + (" --sparse"             if use_sparse else "")
        + (" --no-prefilter"       if no_prefilter else "")
        + (" --no-plots"           if no_plots else "")
        + (f" --images {' '.join(images)}"          if images else "")
        + (f" --remove-images {' '.join(remove_images)}" if remove_images else "")
        + (f" --restrict-to-hst-filters {' '.join(restrict_filters)}" if restrict_filters else "")
        + (f" --checkpoint {checkpoint_dir}"        if checkpoint_dir else "")
    )
    print(f"  run_bp3m command:\n    {_cmd}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n  Loading FLC pipeline data for '{field_name}'...")
    imgs, stars_per_image, gaia_catalog = load_image_data_flc(
        data_root, field_name, pos_err_floor=pos_err_floor)
    if imgs is None or len(imgs) == 0:
        raise RuntimeError(
            f"No usable images found for '{field_name}'. "
            "Check that cross-matching completed successfully."
        )

    star_id_to_idx, image_names, star_in_image = build_index_maps(
        stars_per_image, gaia_catalog)

    # ── Image filtering ───────────────────────────────────────────────────────
    if images is not None:
        requested = set(images)
        image_names = [n for n in image_names if n in requested]
    if remove_images is not None:
        drop = set(remove_images)
        image_names = [n for n in image_names if n not in drop]
    if restrict_filters is not None:
        keep_filters = {f.upper() for f in restrict_filters}
        image_names = [n for n in image_names
                       if imgs[n].get('filter', '').upper() in keep_filters]
    if restrict_instdet is not None:
        keep_id = {s.upper() for s in restrict_instdet}
        image_names = [
            n for n in image_names
            if (imgs[n].get('instrument', '') + imgs[n].get('detector', '')).upper()
               in keep_id
        ]

    if not image_names:
        raise RuntimeError("No images remain after filtering.")
    print(f"  Images: {len(image_names)}")

    # Rebuild index maps after filtering
    filtered_spi = {n: stars_per_image[n] for n in image_names}
    star_id_to_idx, image_names, star_in_image = build_index_maps(
        filtered_spi, gaia_catalog)

    # Filter gaia_catalog to observed stars
    observed_ids = set()
    for spi in filtered_spi.values():
        observed_ids.update(spi['Gaia_id'].values)
    gaia_catalog = (gaia_catalog[gaia_catalog['Gaia_id'].isin(observed_ids)]
                    .reset_index(drop=True))
    star_id_to_idx = {gid: i for i, gid in enumerate(gaia_catalog['Gaia_id'])}

    # Keep imgs in sync with filtered_spi (e.g. after --restrict_instdet)
    imgs = {n: imgs[n] for n in image_names}

    # ── Split CCD if requested ────────────────────────────────────────────────
    if split_ccd:
        from bp3m.data_loader import split_images_by_ccd
        imgs, filtered_spi = split_images_by_ccd(
            imgs, filtered_spi, min_stars_per_ccd=min_stars_split_ccd)
        image_names = sorted(filtered_spi.keys())
        star_id_to_idx, image_names, star_in_image = build_index_maps(
            filtered_spi, gaia_catalog)

    # ── Initialise solver ─────────────────────────────────────────────────────
    SolverClass = BP3MSolverSparse if use_sparse else BP3MSolver
    solver = SolverClass(imgs, filtered_spi, gaia_catalog,
                          star_id_to_idx, image_names, star_in_image,
                          poly_order=poly_order)

    print(f"  Stars: {solver.n_stars}   Images: {solver.n_images}")

    # ── Fit ───────────────────────────────────────────────────────────────────
    clip = clip_sigma if clip_sigma > 0 else None
    t0 = time.time()
    r_hat, C_r, v_hat, C_vT, a_arr, K_img, _ = solver.fit(
        n_iter=n_iter,
        clip_sigma=clip,
        inflate_hst_errors=inflate_hst_errors,
        prefilter=not no_prefilter,
        use_influence_clip=use_influence_clip,
        influence_d_thresh=influence_d_thresh,
        influence_sigma_min=influence_sigma_min,
        use_two_tier=use_two_tier,
    )
    print(f"  Fit completed in {time.time()-t0:.1f}s")

    # ── Sample posteriors ─────────────────────────────────────────────────────
    print(f"  Drawing {n_samples} posterior samples...")
    r_samp, v_mean, v_cov = solver.sample_posteriors(
        r_hat, C_r, a_arr, K_img, C_vT, n_samples=n_samples)

    # ── Save results ──────────────────────────────────────────────────────────
    _save_results(
        output_bp3m, solver, imgs, gaia_catalog, image_names,
        r_hat, C_r, v_hat, C_vT, v_mean, v_cov, K_img, a_arr,
        run_config={
            'n_iter':       n_iter,
            'n_samples':    n_samples,
            'clip_sigma':   clip_sigma,
            'split_ccd':    split_ccd,
            'inflate_hst_errors': inflate_hst_errors,
            'poly_order':   poly_order,
        },
    )

    # ── Star influence ────────────────────────────────────────────────────────
    print("  Computing star influence metrics...")
    try:
        import pandas as _pd
        influence_df = solver.compute_star_influence(r_hat, C_r, a_arr)
        influence_df.to_csv(output_bp3m / "star_influence.csv", index=False)
        print(f"  Saved: star_influence.csv  ({len(influence_df)} star-image pairs)")

        if not no_plots and plot_influence:
            from bp3m.plot_influence import plot_influence_diagnostics
            plot_dir = output_bp3m / "plots"
            plot_dir.mkdir(exist_ok=True)
            plot_influence_diagnostics(influence_df, plot_dir)
    except Exception as _exc:
        print(f"  WARNING: star influence computation failed — {_exc}")
        import traceback; traceback.print_exc()

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    if not no_plots:
        try:
            from bp3m.pipeline.plot_results import make_plots
            print("  Generating diagnostic plots...")
            make_plots(solver, imgs, gaia_catalog,
                       r_hat, v_hat, v_mean, v_cov, C_vT, C_r,
                       output_dir=output_bp3m,
                       plot_residuals=plot_residuals)
        except Exception as exc:
            print(f"  WARNING: plots failed — {exc}")

    if checkpoint_dir is not None:
        from bp3m.checkpointing import save_inputs, save_results as _save_ckpt
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        save_inputs(solver, checkpoint_dir)
        _save_ckpt(r_hat, C_r, v_hat, v_mean, v_cov, C_vT,
                   K_img, solver, checkpoint_dir)

    print(f"\n  Results written to: {output_bp3m}")
    return output_bp3m


# ── Internal: write result CSVs and npy ─────────────────────────────────────

def compute_chi2_per_star(solver, r_hat, v_hat, image_names, use_key='use_for_astrom'):
    """Per-star chi2 using given transformation r_hat and astrometry v_hat.

    chi2_i = sum_{j: use[j]} resid_j @ C_hst_j_inv @ resid_j
    where resid_j = x_j^obs - (X_j @ r_k - JU_j @ v_hat_i)

    Returns
    -------
    chi2 : (n_stars,) float  — summed chi2 per star
    n_det : (n_stars,) int   — number of detections included
    """
    n_stars = solver.n_stars
    chi2 = np.zeros(n_stars)
    n_det = np.zeros(n_stars, dtype=int)

    for j, img in enumerate(image_names):
        d = solver._img_data.get(img)
        if d is None:
            continue
        use = d.get(use_key, d['use_for_fit'])
        if not use.any():
            continue

        r_j    = r_hat[j * solver.N_R:(j + 1) * solver.N_R]
        sidx   = d['sidx'][use]
        xys    = d['xys'][use]
        X_mat  = d['X_mat'][use]
        JU     = d['JU'][use]
        C_hst  = d['C_hst'][use]           # (n, 2, 2)
        C_inv  = np.linalg.inv(C_hst)      # (n, 2, 2)

        v_star = v_hat[sidx]               # (n, 5)
        motion = np.einsum('nij,nj->ni', JU, v_star)   # (n, 2)
        x_pred = np.einsum('nkl,l->nk', X_mat, r_j) - motion  # (n, 2)
        resid  = xys - x_pred              # (n, 2)

        chi2_det = np.einsum('ni,nij,nj->n', resid, C_inv, resid)
        np.add.at(chi2, sidx, chi2_det)
        np.add.at(n_det, sidx, 1)

    return chi2, n_det


def _save_results(output_dir, solver, images, gaia_catalog, image_names,
                  r_hat, C_r, v_hat, C_vT, v_mean, v_cov, K_img, a_arr,
                  run_config: dict | None = None):
    import pandas as pd

    v_cov_full = v_cov + C_vT

    # 1. Image transformation parameters
    rows = []
    for j, img in enumerate(image_names):
        cs   = j * solver.N_R
        r_j  = r_hat[cs: cs + solver.N_R]
        C_j  = C_r[cs: cs + solver.N_R, cs: cs + solver.N_R]
        d_img = solver._img_data[img]
        n_align  = int(np.sum(d_img['use_for_fit']))
        use_ast  = d_img.get('use_for_astrom', d_img['use_for_fit'])
        n_astrom = int(np.sum(use_ast & ~d_img['use_for_fit']))
        a, b, c, d = r_j[:4]
        alpha_applied = float(d_img.get('alpha_applied', 1.0))
        rows.append(dict(
            image_name=img,
            n_stars_alignment=n_align,
            n_stars_astrometry_only=n_astrom,
            a=a, b=b, c=c, d=d,
            w=r_j[4], z=r_j[5],
            delta_ra0_mas=r_j[6]*1000,
            delta_dec0_mas=r_j[7]*1000,
            pixel_scale_mas=np.sqrt(a*d - b*c) * images[img].get('orig_pixel_scale', 50.0),
            rotation_deg=np.degrees(np.arctan2(b - c, a + d)),
            on_skew=(a - d) / 2,
            off_skew=(b + c) / 2,
            sigma_a=np.sqrt(C_j[0,0]), sigma_b=np.sqrt(C_j[1,1]),
            sigma_c=np.sqrt(C_j[2,2]), sigma_d=np.sqrt(C_j[3,3]),
            sigma_w=np.sqrt(C_j[4,4]), sigma_z=np.sqrt(C_j[5,5]),
            sigma_dra0_mas=np.sqrt(C_j[6,6])*1000,
            sigma_ddec0_mas=np.sqrt(C_j[7,7])*1000,
            alpha=alpha_applied,
            **{f'r_{k}': float(r_j[k]) for k in range(8, solver.N_R)},
        ))
    pd.DataFrame(rows).to_csv(output_dir / "image_transformations.csv", index=False)

    # 2. Stellar astrometry
    g = gaia_catalog.copy()
    g['n_hst_used'] = solver.gaia_n_hst_used  # detections used for alignment OR astrometry

    # Per-star alignment detection count
    n_align = np.zeros(solver.n_stars, dtype=int)
    for img in image_names:
        d_img = solver._img_data.get(img)
        if d_img is not None:
            np.add.at(n_align, d_img['sidx'][d_img['use_for_fit']], 1)
    g['n_hst_alignment'] = n_align

    # Per-star chi2 using best-fit (r_hat, v_hat) for use_for_astrom detections
    chi2_hst, n_chi2 = compute_chi2_per_star(
        solver, r_hat, v_hat, image_names, use_key='use_for_astrom'
    )
    g['chi2_hst']   = chi2_hst
    g['n_det_chi2'] = n_chi2
    # Reduced chi2 (chi2 per 2-dof detection): 0 when no detections
    with np.errstate(invalid='ignore', divide='ignore'):
        g['chi2_hst_red'] = np.where(n_chi2 > 0, chi2_hst / (2 * n_chi2), np.nan)

    g['delta_racosdec_bp3m'] = v_mean[:, 0]
    g['delta_dec_bp3m']      = v_mean[:, 1]
    g['pmra_bp3m']           = v_mean[:, 2]
    g['pmdec_bp3m']          = v_mean[:, 3]
    g['parallax_bp3m']       = v_mean[:, 4]

    g['sigma_delta_racosdec'] = np.sqrt(v_cov_full[:, 0, 0])
    g['sigma_delta_dec']      = np.sqrt(v_cov_full[:, 1, 1])
    g['sigma_pmra_bp3m']      = np.sqrt(v_cov_full[:, 2, 2])
    g['sigma_pmdec_bp3m']     = np.sqrt(v_cov_full[:, 3, 3])
    g['sigma_parallax_bp3m']  = np.sqrt(v_cov_full[:, 4, 4])

    _sig = np.sqrt(np.diagonal(v_cov_full, axis1=1, axis2=2))
    for col, i, j in [
        ('corr_dra_ddec', 0, 1), ('corr_dra_pmra', 0, 2),
        ('corr_dra_pmdec', 0, 3), ('corr_dra_plx', 0, 4),
        ('corr_ddec_pmra', 1, 2), ('corr_ddec_pmdec', 1, 3),
        ('corr_ddec_plx', 1, 4), ('corr_pmra_pmdec', 2, 3),
        ('corr_pmra_plx', 2, 4), ('corr_pmdec_plx', 3, 4),
    ]:
        denom = _sig[:, i] * _sig[:, j]
        g[col] = np.where(denom > 0, v_cov_full[:, i, j] / denom, np.nan)

    # Conditional (MAP alignment fixed)
    g['pmra_bp3m_cond']           = v_hat[:, 2]
    g['pmdec_bp3m_cond']          = v_hat[:, 3]
    g['parallax_bp3m_cond']       = v_hat[:, 4]
    g['sigma_pmra_bp3m_cond']     = np.sqrt(C_vT[:, 2, 2])
    g['sigma_pmdec_bp3m_cond']    = np.sqrt(C_vT[:, 3, 3])
    g['sigma_parallax_bp3m_cond'] = np.sqrt(C_vT[:, 4, 4])

    g.to_csv(output_dir / "stellar_astrometry.csv", index=False)

    # 3. Full covariance arrays
    np.save(output_dir / "v_cov_marginalised.npy", v_cov)
    np.save(output_dir / "C_vT.npy", C_vT)
    np.save(output_dir / "C_r.npy", C_r)

    # 4. Per-detection use flags (for reproducibility and hierarchical modelling)
    # use_for_fit[img]   : (n,) bool — detection used for ALIGNMENT (constrains r_hat)
    # use_for_astrom[img]: (n,) bool — detection used for ASTROMETRY (constrains v_hat)
    # star_indices[img]  : (n,) int  — indices into stellar_astrometry.csv rows
    _fit_data    = {}
    _astrom_data = {}
    _idx_data    = {}
    for img in image_names:
        d_img = solver._img_data.get(img)
        if d_img is None:
            continue
        _fit_data[img]    = d_img['use_for_fit']
        _astrom_data[img] = d_img.get('use_for_astrom', d_img['use_for_fit'])
        _idx_data[img]    = d_img['sidx']
    np.savez(output_dir / "use_for_fit.npz",    **_fit_data)
    np.savez(output_dir / "use_for_astrom.npz", **_astrom_data)
    np.savez(output_dir / "star_indices.npz",   **_idx_data)

    print(f"  Saved: stellar_astrometry.csv  "
          f"({len(g)} stars, {g['n_hst_used'].sum()} HST detections)")

    # 5. Machine-readable run configuration for downstream tools (e.g. hst_catalog_crossmatch)
    import json as _json
    config = {
        'poly_order':   solver.poly_order,
        'n_r_per_image': solver.N_R,
        'n_images':     len(image_names),
        'n_stars':      solver.n_stars,
        'image_names':  image_names,   # ordered to match C_r blocks
    }
    if run_config:
        config.update(run_config)
    with open(output_dir / 'run_config.json', 'w') as _f:
        _json.dump(config, _f, indent=2)
