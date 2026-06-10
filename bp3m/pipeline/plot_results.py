"""
Diagnostic plots comparing BP3M astrometry to Gaia.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from collections import defaultdict
from pathlib import Path


PLOT_DIR_NAME = "plots"
RESID_PLOT_DIR_NAME = "residuals"


# ── Public entry point ────────────────────────────────────────────────────────

def make_plots(solver, images, gaia_catalog,
               r_hat, v_hat, v_mean, v_cov, C_vT, C_r,
               output_dir,
               plot_residuals: bool = False):
    """
    Generate all diagnostic plots.

    Parameters
    ----------
    solver        : BP3MSolver  (fitted, for compute_residuals)
    images        : dict        (image metadata from data_loader)
    gaia_catalog  : pd.DataFrame
    r_hat         : (n_r,) MAP image transformation vector
    v_hat         : (n_stars, 5) MAP stellar astrometry
    v_mean        : (n_stars, 5) posterior mean from sample_posteriors
    v_cov         : (n_stars, 5, 5) posterior covariance from sample_posteriors
                    (r-propagation only — C_vT is added internally for full marginal)
    C_vT          : (n_stars, 5, 5) conditional covariance
    C_r           : (n_r, n_r) posterior covariance of image transformations
    output_dir    : Path-like, parent results directory
    plot_residuals : generate per-image XY residual maps (slow for large fields)
    """
    plot_dir = Path(output_dir) / PLOT_DIR_NAME
    plot_dir.mkdir(parents=True, exist_ok=True)
    resid_plot_dir = Path(plot_dir) / RESID_PLOT_DIR_NAME
    resid_plot_dir.mkdir(parents=True, exist_ok=True)

    # Full marginal covariance = r-propagation + conditional
    v_cov_full = v_cov + C_vT   # (n_stars, 5, 5)

    pmra_bp3m   = v_mean[:, 2]
    pmdec_bp3m  = v_mean[:, 3]

    pmra_gaia   = solver.v_survey[:,2]
    pmdec_gaia  = solver.v_survey[:,3]
    gmag        = solver.gaia_cat["gmag"].to_numpy(float)

    has_gaia = solver.full_gaia_astrometry   # bool (n_stars,)
    C_pm_gaia = solver.C_survey[:,2:4,2:4]
    sig_pmra_gaia = np.sqrt(C_pm_gaia[:,0,0])
    sig_pmdec_gaia = np.sqrt(C_pm_gaia[:,1,1])
    rho_gaia = C_pm_gaia[:,0,1]/(sig_pmra_gaia*sig_pmdec_gaia)

    sig_pm_gaia = _pm_geom_unc(sig_pmra_gaia, sig_pmdec_gaia, rho_gaia)

    C_pm_bp3m   = v_cov_full[:, 2:4, 2:4]
    det_bp3m    = np.linalg.det(C_pm_bp3m)
    sig_pm_bp3m = np.where(det_bp3m > 0, det_bp3m ** 0.25, np.nan)
    bp3m_converged = (sig_pm_bp3m < 90)

    sig_pmra_bp3m = np.sqrt(C_pm_bp3m[:,0,0])
    sig_pmdec_bp3m = np.sqrt(C_pm_bp3m[:,1,1])
    rho_bp3m = C_pm_bp3m[:,0,1]/(sig_pmra_bp3m*sig_pmdec_bp3m)

    # ── Figure 1: 1:1 PM comparison (top) + PM uncertainty vs mag (bottom) ───
    print("  Plotting 1:1 PM comparison + uncertainty vs magnitude...")

    fig = plt.figure(figsize=(13, 11/2*3), layout="constrained")
    gs  = fig.add_gridspec(3, 2)
    ax_pmra  = fig.add_subplot(gs[0, 0])
    ax_pmdec = fig.add_subplot(gs[0, 1])
    ax_unc   = fig.add_subplot(gs[1, :])
    ax_unc_improve   = fig.add_subplot(gs[2, :])

    _gc = solver.gaia_cat
    _gaia_ids = _gc["Gaia_id"].to_numpy(dtype=np.int64, na_value=0)
    hst_only  = ~has_gaia
    _n_gc = len(_gc)
    if "pmra_xmatch" in _gc.columns:
        _pmra_xmatch  = np.array(
            [float(v) if v is not None and str(v) not in ('nan', 'None', '') else np.nan
             for v in _gc["pmra_xmatch"].values], dtype=float)
        _pmdec_xmatch = np.array(
            [float(v) if v is not None and str(v) not in ('nan', 'None', '') else np.nan
             for v in _gc["pmdec_xmatch"].values], dtype=float)
    else:
        _pmra_xmatch  = np.full(_n_gc, np.nan)
        _pmdec_xmatch = np.full(_n_gc, np.nan)
    hst_has_xmatch = hst_only & np.isfinite(_pmra_xmatch) & np.isfinite(_pmdec_xmatch)

    for ax, gaia_pm, bp3m_pm_g, sig_g, sig_b_g, comp in zip(
            [ax_pmra, ax_pmdec],
            [pmra_gaia[has_gaia],   pmdec_gaia[has_gaia]],
            [pmra_bp3m[has_gaia],   pmdec_bp3m[has_gaia]],
            [sig_pmra_gaia[has_gaia], sig_pmdec_gaia[has_gaia]],
            [sig_pmra_bp3m[has_gaia], sig_pmdec_bp3m[has_gaia]],
            [r"$\mu_{\alpha*}$",    r"$\mu_\delta$"]):
        ax.errorbar(gaia_pm, bp3m_pm_g, xerr=sig_g, yerr=sig_b_g,
                    fmt='o', ms=3, lw=0.5, alpha=0.5, color='steelblue',
                    label='Gaia-matched', zorder=2)
        lim = _padded_lim(gaia_pm, bp3m_pm_g)
        ax.plot(lim, lim, 'k--', lw=1, zorder=4)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"{comp} Gaia [mas/yr]")
        ax.set_ylabel(f"{comp} BP3M [mas/yr]")
        ax.set_title(f"{comp}: BP3M vs Gaia (Gaia-matched only)")
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc='upper left')
        _style_ax(ax)

    gm = gmag[has_gaia]
    ax_unc.scatter(gm, sig_pm_gaia[has_gaia],
                   s=6, alpha=0.5, color='#aaaaaa', label='Gaia 5p', zorder=2)
    _bp3m_gaia_conv = bp3m_converged & has_gaia
    _bp3m_hst_conv  = bp3m_converged & hst_only
    ax_unc.scatter(gmag[_bp3m_gaia_conv], sig_pm_bp3m[_bp3m_gaia_conv],
                   s=6, alpha=0.7, color='steelblue', label='BP3M Gaia 5p', zorder=3)
    if _bp3m_hst_conv.any():
        ax_unc.scatter(gmag[_bp3m_hst_conv], sig_pm_bp3m[_bp3m_hst_conv],
                       s=10, alpha=0.8, color='darkorange', marker='^',
                       label='BP3M Gaia 2p + HST', zorder=4)
    ax_unc.set_xlabel("G [mag]")
    ax_unc.set_ylabel(r"$(\det\,C_{\mu})^{1/4}$ [mas/yr]")
    ax_unc.set_title(r"Geometric-mean PM uncertainty $(\det\,C_{\mu})^{1/4}$ vs magnitude")
    ax_unc.legend()
    ax_unc.set_yscale("log")
    xlim = ax_unc.get_xlim()
    _style_ax(ax_unc)

    ax_unc_improve.scatter(gm, sig_pm_gaia[has_gaia]/sig_pm_bp3m[has_gaia],
                   s=6, alpha=0.6, color='steelblue', zorder=2)
    ax_unc_improve.set_xlabel("Gaia G [mag]")
    ax_unc_improve.set_ylabel(r"PM Improvement Factor")
    ax_unc_improve.set_title(r"PM uncertainty Improvement vs magnitude compared to Gaia-alone")
    ax_unc_improve.set_xlim(xlim)
    ax_unc_improve.axhline(1.0,c='k',lw=2,ls='--',zorder=-1e10)
    _style_ax(ax_unc_improve)

    fig.suptitle("Proper motion comparison", fontsize=13)
    _save(fig, plot_dir / "pm_one_to_one.png")

    # ── Figure 2: PM vector diagrams coloured by geometric-mean uncertainty ───
    print("  Plotting PM vector diagrams...")

    gaia_pmra_h  = pmra_gaia[has_gaia]
    gaia_pmdec_h = pmdec_gaia[has_gaia]
    bp3m_pmra_h  = pmra_bp3m[bp3m_converged]
    bp3m_pmdec_h = pmdec_bp3m[bp3m_converged]

    full_xlim = _padded_lim(gaia_pmra_h)
    full_ylim = _padded_lim(gaia_pmdec_h)

    zoom_xcen = np.nanmedian(gaia_pmra_h)
    zoom_ycen = np.nanmedian(gaia_pmdec_h)
    zoom_xhw  = max(np.abs(np.nanpercentile(gaia_pmra_h,  [16, 84]) - zoom_xcen))
    zoom_yhw  = max(np.abs(np.nanpercentile(gaia_pmdec_h, [16, 84]) - zoom_ycen))
    zoom_hw   = max(zoom_xhw, zoom_yhw) * 1.15
    zoom_xlim = (zoom_xcen - zoom_hw, zoom_xcen + zoom_hw)
    zoom_ylim = (zoom_ycen - zoom_hw, zoom_ycen + zoom_hw)

    c_gaia = sig_pm_gaia[has_gaia]
    c_bp3m = sig_pm_bp3m[bp3m_converged]
    all_unc = np.concatenate([c_gaia[np.isfinite(c_gaia)],
                              c_bp3m[np.isfinite(c_bp3m)]])
    vmin = np.nanpercentile(all_unc, 2)
    vmax = np.nanpercentile(all_unc, 98)
    norm = mcolors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)
    cmap = "plasma"

    fig, axes = plt.subplots(2, 2, figsize=(13, 12), layout="constrained")

    sc_last = None
    for col, pmra, pmdec, c_vals, label in zip(
            [0, 1],
            [gaia_pmra_h,  bp3m_pmra_h],
            [gaia_pmdec_h, bp3m_pmdec_h],
            [c_gaia,       c_bp3m],
            ["Gaia",       "BP3M"]):

        for row, xlim, ylim, suffix in zip(
                [0, 1],
                [full_xlim, zoom_xlim],
                [full_ylim, zoom_ylim],
                ["full range", "zoom (68% CI)"]):

            ax = axes[row, col]
            sc = ax.scatter(pmra, pmdec, c=c_vals, s=6, alpha=0.8,
                            cmap=cmap, norm=norm, zorder=2)
            ax.axhline(0,c='k',lw=1,ls='--',zorder=1e10)
            ax.axvline(0,c='k',lw=1,ls='--',zorder=1e10)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xlabel(r"$\mu_{\alpha*}$ [mas/yr]")
            ax.set_ylabel(r"$\mu_\delta$ [mas/yr]")
            ax.set_title(f"{label} — {suffix}")
            ax.set_aspect("equal")
            _style_ax(ax)
            sc_last = sc

    cbar = fig.colorbar(sc_last, ax=axes, shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label(r"$(\det\,C_{\mu})^{1/4}$ [mas/yr]")
    fig.suptitle("PM vector diagrams coloured by geometric-mean uncertainty", fontsize=13)
    _save(fig, plot_dir / "pm_vector_diagram.png")

    # ── Figure 2b: PM vector diagrams with covariance error bars ─────────────
    print("  Plotting PM vector diagrams with error bars...")

    C_pm_gaia_h = solver.C_survey[has_gaia, 2:4, 2:4]
    C_pm_bp3m_h = C_pm_bp3m[bp3m_converged]

    fig, axes = plt.subplots(2, 2, figsize=(13, 12), layout="constrained")

    sc_last = None
    for col, pmra, pmdec, c_vals, C_pm_col, label in zip(
            [0, 1],
            [gaia_pmra_h,  bp3m_pmra_h],
            [gaia_pmdec_h, bp3m_pmdec_h],
            [c_gaia,       c_bp3m],
            [C_pm_gaia_h,  C_pm_bp3m_h],
            ["Gaia",       "BP3M"]):

        for row, xlim, ylim, suffix in zip(
                [0, 1],
                [full_xlim, zoom_xlim],
                [full_ylim, zoom_ylim],
                ["full range", "zoom (68% CI)"]):

            ax = axes[row, col]
            _pm_error_bars(ax, pmra, pmdec, C_pm_col)
            sc = ax.scatter(pmra, pmdec, c=c_vals, s=6, alpha=0.8,
                            cmap=cmap, norm=norm, zorder=2)
            ax.axhline(0, c='k', lw=1, ls='--', zorder=1e10)
            ax.axvline(0, c='k', lw=1, ls='--', zorder=1e10)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xlabel(r"$\mu_{\alpha*}$ [mas/yr]")
            ax.set_ylabel(r"$\mu_\delta$ [mas/yr]")
            ax.set_title(f"{label} — {suffix}")
            ax.set_aspect("equal")
            _style_ax(ax)
            sc_last = sc

    cbar = fig.colorbar(sc_last, ax=axes, shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label(r"$(\det\,C_{\mu})^{1/4}$ [mas/yr]")
    fig.suptitle(
        "PM vector diagrams with 1σ principal-axis error bars\n"
        r"(coloured by $(\det\,C_{\mu})^{1/4}$)",
        fontsize=13)
    _save(fig, plot_dir / "pm_vector_diagram_errorbars.png")

    # ── Figure 2c: BP3M PM coloured by detector position ─────────────────────
    print("  Plotting BP3M PM vector diagram coloured by detector position...")

    n_stars_global = len(solver.gaia_cat)
    _xo_sum = np.zeros(n_stars_global)
    _yo_sum = np.zeros(n_stars_global)
    _det_cnt = np.zeros(n_stars_global)
    for _img in solver.image_names:
        _df = solver.stars_per_image[_img]
        _gids = _df["Gaia_id"].to_numpy()
        _sidx_img = np.array([solver.star_id_to_idx[int(g)]
                               for g in _gids
                               if int(g) in solver.star_id_to_idx])
        _valid = np.array([int(g) in solver.star_id_to_idx for g in _gids])
        _xcol = "X_orig" if "X_orig" in _df.columns else "X"
        _ycol = "Y_orig" if "Y_orig" in _df.columns else "Y"
        _xo_sum[_sidx_img] += _df[_xcol].to_numpy(float)[_valid]
        _yo_sum[_sidx_img] += _df[_ycol].to_numpy(float)[_valid]
        _det_cnt[_sidx_img] += 1

    _obs = _det_cnt > 0
    x_orig_star = np.where(_obs, _xo_sum / np.maximum(_det_cnt, 1), np.nan)[bp3m_converged]
    y_orig_star = np.where(_obs, _yo_sum / np.maximum(_det_cnt, 1), np.nan)[bp3m_converged]

    bp3m_full_xlim = _padded_lim(bp3m_pmra_h)
    bp3m_full_ylim = _padded_lim(bp3m_pmdec_h)

    _bx_cen = np.nanmedian(bp3m_pmra_h)
    _by_cen = np.nanmedian(bp3m_pmdec_h)
    _bx_hw  = max(np.abs(np.nanpercentile(bp3m_pmra_h,  [16, 84]) - _bx_cen))
    _by_hw  = max(np.abs(np.nanpercentile(bp3m_pmdec_h, [16, 84]) - _by_cen))
    _b_hw   = max(_bx_hw, _by_hw) * 1.15
    bp3m_zoom_xlim = (_bx_cen - _b_hw, _bx_cen + _b_hw)
    bp3m_zoom_ylim = (_by_cen - _b_hw, _by_cen + _b_hw)

    def _lin_norm(vals):
        fin = vals[np.isfinite(vals)]
        vlo, vhi = np.nanpercentile(fin, [2, 98])
        return mcolors.Normalize(vmin=vlo, vmax=vhi)

    norm_xo = _lin_norm(x_orig_star)
    norm_yo = _lin_norm(y_orig_star)

    fig, axes = plt.subplots(2, 2, figsize=(13, 12), layout="constrained")

    sc_xo = sc_yo = None
    for row, xlim, ylim, row_label in zip(
            [0, 1],
            [bp3m_full_xlim, bp3m_zoom_xlim],
            [bp3m_full_ylim, bp3m_zoom_ylim],
            ["full range", "zoom (68% CI)"]):

        for col, c_vals, norm_c, cmap_c, coord_label in zip(
                [0, 1],
                [x_orig_star, y_orig_star],
                [norm_xo,     norm_yo],
                ["plasma",    "plasma"],
                ["X_orig",    "Y_orig"]):

            ax = axes[row, col]
            sc = ax.scatter(bp3m_pmra_h, bp3m_pmdec_h,
                            c=c_vals, s=6, alpha=0.8,
                            cmap=cmap_c, norm=norm_c, zorder=2)
            ax.axhline(0, c='k', lw=1, ls='--', zorder=1e10)
            ax.axvline(0, c='k', lw=1, ls='--', zorder=1e10)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xlabel(r"$\mu_{\alpha*}$ [mas/yr]")
            ax.set_ylabel(r"$\mu_\delta$ [mas/yr]")
            ax.set_title(f"BP3M — {row_label}  (colour: {coord_label})")
            ax.set_aspect("equal")
            _style_ax(ax)

            if row == 0 and col == 0:
                sc_xo = sc
            if row == 0 and col == 1:
                sc_yo = sc

    cbar_xo = fig.colorbar(sc_xo, ax=axes[:, 0], shrink=0.6, pad=0.02, aspect=30)
    cbar_xo.set_label("X_orig [pixels]")
    cbar_yo = fig.colorbar(sc_yo, ax=axes[:, 1], shrink=0.6, pad=0.02, aspect=30)
    cbar_yo.set_label("Y_orig [pixels]")
    fig.suptitle("BP3M proper motions coloured by HST detector position", fontsize=13)
    _save(fig, plot_dir / "pm_vector_diagram_detector_pos.png")

    # ── Figure: HST chi2 distributions ───────────────────────────────────────
    print("  Plotting HST chi2 distributions...")
    _plot_chi2_distributions(solver, r_hat, v_hat, plot_dir)

    # ── Figure: HST XY residuals + BP3M proper motions on detector ───────────
    if not plot_residuals:
        print(f"  All plots saved to {plot_dir}/")
        return
    print("  Plotting detector residuals and proper motion maps...")
    resid_dict = solver.compute_residuals(r_hat, v_hat, C_r=C_r, C_vT=C_vT)

    _AMP_SUFFIXES = ('_llo', '_rlo', '_lhi', '_rhi')
    _CCD_SUFFIXES = ('_lo', '_hi')

    img_groups = defaultdict(list)
    for img in resid_dict:
        if img.endswith(_AMP_SUFFIXES):
            base = img[:-4]
        elif img.endswith(_CCD_SUFFIXES):
            base = img[:-3]
        else:
            base = img
        img_groups[base].append(img)

    for base_name, img_list in sorted(img_groups.items()):
        img_list = sorted(img_list)

        def _cat(key):
            return np.concatenate([resid_dict[img][key] for img in img_list])

        X_c   = _cat("X_c")
        Y_c   = _cat("Y_c")
        res_x = _cat("resid_x")
        res_y = _cat("resid_y")
        sr_x  = _cat("sigma_resid_x")
        sr_y  = _cat("sigma_resid_y")
        use   = _cat("use")
        sidx  = _cat("sidx")

        if np.sum(use) == 0:
            print(f'  SKIPPING {base_name}: no usable stars')
            continue

        use &= bp3m_converged[sidx]

        pscale    = images[img_list[0]]["orig_pixel_scale"]
        res_x_mas = res_x * pscale
        res_y_mas = res_y * pscale

        pmra_img  = pmra_bp3m[sidx]
        pmdec_img = pmdec_bp3m[sidx]

        n_split = len(img_list)
        if n_split == 4:
            split_note = "  (4 amp quadrants combined)"
        elif n_split > 1:
            split_note = f"  ({n_split} CCD halves combined)"
        else:
            split_note = ""

        fig, axes = plt.subplots(3, 2, figsize=(13, 15), layout="constrained")

        for ax, res_mas, comp in zip(axes[0], [res_x_mas, res_y_mas], ["x", "y"]):
            vmax = np.nanpercentile(np.abs(res_mas[use]), 95)
            sc = ax.scatter(
                X_c[use], Y_c[use], c=res_mas[use],
                s=10, cmap="RdYlBu_r", vmin=-vmax, vmax=vmax, alpha=0.8, zorder=2)
            fig.colorbar(sc, ax=ax, label=f"residual {comp} [mas]")
            ax.set_xlabel("X − Xo [pixels]")
            ax.set_ylabel("Y − Yo [pixels]")
            ax.set_title(f"{base_name}  residual {comp}  (n={use.sum()}){split_note}")
            ax.set_aspect("equal")
            _style_ax(ax)

        for ax, sr, comp in zip(axes[1], [sr_x, sr_y], ["x", "y"]):
            finite = np.isfinite(sr[use])
            vmax_s = np.nanpercentile(np.abs(sr[use][finite]), 95) if finite.any() else 3.
            sc = ax.scatter(
                X_c[use], Y_c[use], c=sr[use],
                s=10, cmap="RdYlBu_r", vmin=-vmax_s, vmax=vmax_s, alpha=0.8, zorder=2)
            fig.colorbar(sc, ax=ax, label=f"residual {comp} / σ_HST  [σ]")
            ax.set_xlabel("X − Xo [pixels]")
            ax.set_ylabel("Y − Yo [pixels]")
            ax.set_title(f"{base_name}  σ-residual {comp}  "
                         f"(RMS = {np.nanstd(sr[use]):.2f} σ){split_note}")
            ax.set_aspect("equal")
            _style_ax(ax)

        for ax, pm_vals, comp_tex in zip(
                axes[2],
                [pmra_img,  pmdec_img],
                [r"$\mu_{\alpha*}$", r"$\mu_\delta$"]):
            pm_use = pm_vals[use]
            p16, p84 = np.nanpercentile(pm_use, [16, 84])
            sc = ax.scatter(
                X_c[use], Y_c[use], c=pm_use,
                s=10, cmap="viridis", vmin=p16, vmax=p84, alpha=0.8, zorder=2)
            fig.colorbar(sc, ax=ax, label=f"{comp_tex} BP3M [mas/yr]")
            ax.set_xlabel("X − Xo [pixels]")
            ax.set_ylabel("Y − Yo [pixels]")
            ax.set_title(f"{base_name}  {comp_tex} BP3M  "
                         f"(clim: [{p16:.2f}, {p84:.2f}] mas/yr){split_note}")
            ax.set_aspect("equal")
            _style_ax(ax)

        fig.suptitle(f"HST detector residuals & proper motions — {base_name}{split_note}",
                     fontsize=12)
        _save(fig, resid_plot_dir / f"residuals_{base_name}.png")

    print(f"  All plots saved to {plot_dir}/")


def _plot_chi2_distributions(solver, r_hat, v_hat, plot_dir):
    """Three-panel diagnostic for the HST-only chi2 per star per image."""
    from scipy.stats import chi2 as chi2_dist

    resid_hst = solver.compute_residuals(r_hat, v_hat)

    per_img_chi2   = {}
    per_img_all    = {}
    per_img_alpha  = {}
    _MEDIAN_CHI2_2 = 2.0 * np.log(2.0)

    for img, rd in resid_hst.items():
        use    = solver._img_data[img]["use_for_fit"]
        chi2_v = rd["sigma_resid"] ** 2

        per_img_all[img]   = chi2_v
        per_img_chi2[img]  = chi2_v[use]

        med = np.median(chi2_v[use]) if use.sum() >= 2 else np.nan
        per_img_alpha[img] = float(max(1.0, np.sqrt(med / _MEDIAN_CHI2_2)))

    all_accepted = np.concatenate(list(per_img_chi2.values()))
    all_vals     = np.concatenate(list(per_img_all.values()))

    thresholds = {
        "0.99":   chi2_dist.ppf(0.99,   df=2),
        "0.999":  chi2_dist.ppf(0.999,  df=2),
        "0.9999": chi2_dist.ppf(0.9999, df=2),
    }
    thresh_colors = {"0.99": "royalblue", "0.999": "darkorange", "0.9999": "crimson"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("HST-only chi2 distributions at convergence", fontsize=13)

    ax = axes[0]
    clip = np.percentile(all_accepted, 99.5)
    bins = np.linspace(0, max(clip, 20), 80)
    ax.hist(all_accepted, bins=bins, density=True, color="steelblue",
            alpha=0.6, label="accepted stars")
    xx = np.linspace(0.01, bins[-1], 400)
    ax.plot(xx, chi2_dist.pdf(xx, df=2), "k-", lw=1.5, label="χ²(2) theory")
    for label, thr in thresholds.items():
        ax.axvline(thr, color=thresh_colors[label], lw=1.2, ls="--",
                   label=f"q={label}  ({thr:.1f})")
    ax.set_xlabel("σ_resid² (HST-only chi2)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution (accepted stars)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, bins[-1])

    ax = axes[1]
    sorted_chi2 = np.sort(all_accepted)
    cdf = np.arange(1, len(sorted_chi2) + 1) / len(sorted_chi2)
    ax.plot(sorted_chi2, cdf, color="steelblue", lw=1.5)
    ax.plot(np.sort(all_vals), np.arange(1, len(all_vals) + 1) / len(all_vals),
            color="gray", lw=1, ls=":", alpha=0.7, label="all (incl. excluded)")
    for label, thr in thresholds.items():
        frac_survive = float((all_accepted < thr).mean())
        ax.axvline(thr, color=thresh_colors[label], lw=1.2, ls="--",
                   label=f"q={label}: {100*frac_survive:.1f}% survive")
    ax.set_xlabel("σ_resid² threshold")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("CDF — surviving fraction vs. threshold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, max(thresholds["0.9999"] * 1.3, np.percentile(all_accepted, 98)))
    ax.set_ylim(0, 1)

    ax = axes[2]
    img_names = list(per_img_chi2.keys())
    medians   = [np.median(per_img_chi2[im]) for im in img_names]
    alphas    = [per_img_alpha[im] for im in img_names]

    order   = np.argsort(medians)[::-1]
    names_s = [img_names[i] for i in order]
    meds_s  = [medians[i]   for i in order]
    alps_s  = [alphas[i]    for i in order]

    y = np.arange(len(names_s))
    bar_h = max(0.3, min(0.8, 12.0 / max(len(names_s), 1)))

    bars = ax.barh(y, meds_s, height=bar_h, color="steelblue", alpha=0.7,
                   label="median chi2")
    for bar, alp in zip(bars, alps_s):
        bar.set_facecolor("tomato" if alp > 2 else "steelblue")

    for i, (med, alp) in enumerate(zip(meds_s, alps_s)):
        ax.text(med + 0.05, i, f"α={alp:.2f}", va="center", fontsize=6)

    for label, thr in thresholds.items():
        ax.axvline(thr, color=thresh_colors[label], lw=1.0, ls="--", alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(names_s, fontsize=max(5, min(8, 200 // max(len(names_s), 1))))
    ax.set_xlabel("Median σ_resid² (HST-only chi2, accepted stars)")
    ax.set_title("Per-image: median chi2 & alpha\n(red = α > 2)")
    finite_meds = [m for m in meds_s if np.isfinite(m)]
    xlim_right = max(max(finite_meds) * 1.25 if finite_meds else 0,
                     thresholds["0.999"] * 1.1)
    ax.set_xlim(0, xlim_right)
    ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, plot_dir / "chi2_hst_distributions.png")
    plt.close(fig)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pm_error_bars(ax, pmra, pmdec, C_pm, color="gray", alpha=0.18, lw=0.6):
    """Draw 1σ principal-axis error bars for each point in a PM vector diagram."""
    eigvals, eigvecs = np.linalg.eigh(C_pm)
    half_axes = np.sqrt(np.maximum(eigvals, 0.))

    centers = np.stack([pmra, pmdec], axis=1)
    delta = eigvecs * half_axes[:, np.newaxis, :]

    segs = np.concatenate([
        np.stack([centers - delta[:, :, 0], centers + delta[:, :, 0]], axis=1),
        np.stack([centers - delta[:, :, 1], centers + delta[:, :, 1]], axis=1),
    ], axis=0)

    lc = LineCollection(segs, colors=color, alpha=alpha, linewidths=lw, zorder=1)
    ax.add_collection(lc)


def _pm_geom_unc(sig_pmra, sig_pmdec, rho):
    """Geometric-mean PM uncertainty: (det C_pm)^(1/4)."""
    rho  = np.clip(np.nan_to_num(rho), -0.9999, 0.9999)
    det  = sig_pmra**2 * sig_pmdec**2 * (1.0 - rho**2)
    return np.where((sig_pmra > 0) & (sig_pmdec > 0), det**0.25, np.nan)


def _padded_lim(*arrays, pad=0.04):
    """Return (lo, hi) spanning all values in *arrays with a fractional pad."""
    lo = min(np.nanmin(a) for a in arrays)
    hi = max(np.nanmax(a) for a in arrays)
    margin = (hi - lo) * pad
    return lo - margin, hi + margin


def _style_ax(ax):
    """Apply consistent grid + minor-tick style to an axis."""
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="-",  linewidth=0.5, alpha=0.6)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.4)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")
