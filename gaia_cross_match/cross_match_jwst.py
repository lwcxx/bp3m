import os
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
from scipy.spatial import KDTree
from concurrent.futures import ProcessPoolExecutor, as_completed
from .miracle_match import miracle_match, rd2x, rd2y
from .catalog_matcher import fit_affine_weighted, fit_4p_weighted, apply_affine, compute_mahalanobis, compute_logprob_cost, find_offset, find_scale_and_offset

def load_gaia_data(target, data_dir):
    gaia_path = os.path.join(data_dir, target, "Gaia", "*.csv")
    gaia_files = glob.glob(gaia_path)
    if not gaia_files:
        print(f"No Gaia CSV files found in {gaia_path}"); return None
    print(f"Reading {len(gaia_files)} Gaia CSV files...")
    # Normalise SOURCE_ID → source_id per-file BEFORE concat so that pandas
    # never fills the source_id column with NaN (which would promote int64→float64
    # and silently corrupt Gaia source IDs through floating-point rounding).
    df_list = []
    for f in gaia_files:
        dfi = pd.read_csv(f, dtype={'source_id': np.int64, 'SOURCE_ID': np.int64})
        if 'SOURCE_ID' in dfi.columns and 'source_id' not in dfi.columns:
            dfi = dfi.rename(columns={'SOURCE_ID': 'source_id'})
        df_list.append(dfi)
    df = pd.concat(df_list, ignore_index=True)
    df = df.drop_duplicates(subset=["source_id"])
    if 'bp_rp' not in df.columns and 'phot_bp_mean_mag' in df.columns and 'phot_rp_mean_mag' in df.columns:
        df['bp_rp'] = df['phot_bp_mean_mag'] - df['phot_rp_mean_mag']
    mask = np.isfinite(df['ra']) & np.isfinite(df['dec']) & np.isfinite(df['gmag'])
    if 'bp_rp' in df.columns: mask &= np.isfinite(df['bp_rp'])
    return df[mask]

def find_hst_image_folders(target, data_dir):
    jwst_root = os.path.join(data_dir, target, "JWST")
    folders = []
    for root, dirs, files in os.walk(jwst_root):
        image_name = root.split('/')[-1]
        cat_fname = f"{image_name}_cal_catalog.fits"
        if cat_fname in files:
            cal_files = glob.glob(os.path.join(root, "*_cal.fits"))
            if cal_files:
                folders.append({"root": root, "catalog": os.path.join(root, cat_fname), "cal": cal_files[0]})
    return folders


# Nominal pixel scales (arcsec/px), refined by per-detector empirical
# ratios measured against WCS CD-matrix scales (see bp3m/CLAUDE.md).
_JWST_PIXEL_SCALE = {
    'NRCA1':     0.031087,
    'NRCA2':     0.031050,
    'NRCA3':     0.031086,
    'NRCA4':     0.031044,
    'NRCB1':     0.031048,
    'NRCB2':     0.031091,
    'NRCB3':     0.031046,
    'NRCB4':     0.031088,
    'NRCALONG':  0.063151,
    'NRCBLONG':  0.063146,
    'NIRCAM_SW': 0.031,   # fallback: unrecognised SW detector
    'NIRCAM_LW': 0.063,   # fallback: unrecognised LW detector
    'NIRISS':    0.066238,
    'MIRI':      0.110979,
}


def get_jwst_params(cal_file):
    """Extract astrometric parameters from a JWST _cal.fits file.

    Returns a dict with the same keys as get_jwst_params so the rest of the
    cross-matching pipeline can use it unchanged.
    """
    with fits.open(cal_file) as hdul:
        hdr0     = hdul[0].header
        instrume = hdr0.get('INSTRUME', '').upper()
        detector = hdr0.get('DETECTOR', '').upper()

        sci_hdr = hdul['SCI'].header
        naxis1  = sci_hdr.get('NAXIS1', 2048)
        naxis2  = sci_hdr.get('NAXIS2', 2048)
        ra_cen  = sci_hdr.get('CRVAL1', 0.0)
        dec_cen = sci_hdr.get('CRVAL2', 0.0)
        x_cen   = sci_hdr.get('CRPIX1', naxis1 / 2.0)
        y_cen   = sci_hdr.get('CRPIX2', naxis2 / 2.0)

        # PA_APER: position angle of the aperture y-axis East of North.
        # ORIENTAT is equivalent and lives in the same SCI extension.
        # Falls back to PA_V3 (telescope roll, approximate) as last resort.
        pa_aper = sci_hdr.get('PA_APER',
                  sci_hdr.get('ORIENTAT',
                  hdr0.get('PA_V3', 0.0)))

        if instrume == 'NIRCAM':
            if detector in _JWST_PIXEL_SCALE:
                pixel_scale = _JWST_PIXEL_SCALE[detector]
            elif any(s in detector for s in ('LONG', 'AL')):
                pixel_scale = _JWST_PIXEL_SCALE['NIRCAM_LW']
            else:
                pixel_scale = _JWST_PIXEL_SCALE['NIRCAM_SW']
        elif instrume == 'NIRISS':
            pixel_scale = _JWST_PIXEL_SCALE['NIRISS']
        elif instrume == 'MIRI':
            pixel_scale = _JWST_PIXEL_SCALE['MIRI']
        else:
            pixel_scale = 0.066

        exp_start = hdr0.get('EXPSTART', hdr0.get('MJD-BEG', 51544.0))
        exp_end   = hdr0.get('EXPEND',   hdr0.get('MJD-END', exp_start))
        expstart  = 0.5 * (exp_start + exp_end)

    return {
        'ra_cen':        ra_cen,
        'dec_cen':       dec_cen,
        'x_cen':         x_cen,
        'y_cen':         y_cen,
        'pixel_scale':   pixel_scale,
        'initial_scale': 1.0,
        'obs_epoch_mjd': expstart,
        'orientat':      pa_aper,
        'naxis1':        naxis1,
        'naxis2':        naxis2,
        'instrument':    instrume,
        'detector':      detector,
        'chip_dims':     {1: (naxis1, naxis2)},
    }


def construct_gaia_cov(df, zero_pm=False):
    n = len(df)
    errors = np.zeros((n, 5))
    errors[:, 0], errors[:, 1] = df['ra_error'].values, df['dec_error'].values

    if zero_pm:
        errors[:, 2] = 20.0 # 20 mas
        errors[:, 3], errors[:, 4] = 100.0, 100.0 # 100 mas/yr
    else:
        errors[:, 2] = df['parallax_error'].fillna(20.0).values
        errors[:, 3], errors[:, 4] = df['pmra_error'].fillna(100.0).values, df['pmdec_error'].fillna(100.0).values

    corrs = {(0, 1): 'ra_dec_corr', (0, 2): 'ra_parallax_corr', (0, 3): 'ra_pmra_corr', (0, 4): 'ra_pmdec_corr',
             (1, 2): 'dec_parallax_corr', (1, 3): 'dec_pmra_corr', (1, 4): 'dec_pmdec_corr',
             (2, 3): 'parallax_pmra_corr', (2, 4): 'parallax_pmdec_corr', (3, 4): 'pmra_pmdec_corr'}
    covs = np.zeros((n, 5, 5))
    for i in range(5): covs[:, i, i] = errors[:, i]**2

    if not zero_pm:
        for (i, j), col in corrs.items():
            if col in df.columns:
                val = df[col].fillna(0.0).values
                c = val * errors[:, i] * errors[:, j]
                covs[:, i, j] = c; covs[:, j, i] = c

    gaia_6p = np.isfinite(df['pseudocolour'])
    gaia_5p = np.isfinite(df['pmra']) & ~gaia_6p
    gaia_2p = np.isfinite(df['ra']) & ~gaia_5p

    #inflate Gaia covs according to literature
    covs[gaia_6p] *= 1.22**2
    covs[gaia_5p] *= 1.05**2
    covs[gaia_2p] *= 1.00**2

    #add Gaia systematics according to literature
    parallax_sys_err = 0.011 #mas, from E. Vasiliev and H. Baumgardt 2021, MNRAS 505, 5978–6002
    pm_sys_err = 0.026 #mas/yr, from E. Vasiliev and H. Baumgardt 2021, MNRAS 505, 5978–6002
    covs += np.diag(np.array([0,0,parallax_sys_err,pm_sys_err,pm_sys_err])**2)

    return covs

def propagate_gaia_with_cov(df, target_mjd, zero_pm=False):
    ref_epoch = df['ref_epoch'].iloc[0] if 'ref_epoch' in df.columns else 2016.0
    t_hst = Time(target_mjd, format='mjd')
    dt = (t_hst.jyear - ref_epoch)
    n = len(df)
    ra, dec = np.radians(df['ra'].values), np.radians(df['dec'].values)

    if zero_pm:
        plx = np.zeros(n)
        pmra, pmdec = np.zeros(n), np.zeros(n)
    else:
        plx = df['parallax'].fillna(0.0).values
        pmra, pmdec = df['pmra'].fillna(0.0).values, df['pmdec'].fillna(0.0).values

    with solar_system_ephemeris.set('builtin'): earth_pos = get_body_barycentric('earth', t_hst)
    X, Y, Z = earth_pos.x.to_value('au'), earth_pos.y.to_value('au'), earth_pos.z.to_value('au')
    p_ra_cosdec = X * np.sin(ra) - Y * np.cos(ra)
    p_dec = X * np.cos(ra) * np.sin(dec) + Y * np.sin(ra) * np.sin(dec) - Z * np.cos(dec)
    ra_off_mas, dec_off_mas = (pmra * dt + plx * p_ra_cosdec), (pmdec * dt + plx * p_dec)
    ra_prop = df['ra'].values + (ra_off_mas / 3600000.0) / np.cos(dec)
    dec_prop = df['dec'].values + (dec_off_mas / 3600000.0)
    C0 = construct_gaia_cov(df, zero_pm=zero_pm)
    J = np.zeros((n, 2, 5))
    J[:, 0, 0], J[:, 0, 2], J[:, 0, 3] = 1.0, p_ra_cosdec, dt
    J[:, 1, 1], J[:, 1, 2], J[:, 1, 4] = 1.0, p_dec, dt
    Ct = np.einsum('nij,njk,nlk->nil', J, C0, J)
    return ra_prop, dec_prop, Ct

def project_gaia_cov_to_pixel(Ct, ra, dec, params):
    """Projects sky-frame covariance (mas²) into 2x2 instrument pixel-frame covariance."""
    n = len(ra)
    mas_to_px = 1.0 / (params['pixel_scale'] * 1000.0)
    theta_init = np.radians(-params['orientat'])

    J_proj = np.zeros((n, 2, 2))
    J_proj[:, 0, 0] =  np.cos(theta_init) * mas_to_px
    J_proj[:, 0, 1] = -np.sin(theta_init) * mas_to_px
    J_proj[:, 1, 0] =  np.sin(theta_init) * mas_to_px
    J_proj[:, 1, 1] =  np.cos(theta_init) * mas_to_px

    C_pix = np.einsum('nij,njk,nlk->nil', J_proj, Ct, J_proj)
    return C_pix

def save_diagnostic_plots(out_dir, image_name, matched_df, rejected_df):
    """Generates diagnostic plots.

    Colour scheme:
      blue   — matched star candidates (hst_is_star == True)
      orange — matched non-star sources (hst_is_star == False)
      red    — rejected / unmatched
    """
    fig, axes = plt.subplots(5, 2, figsize=(14, 24/4*5))
    fig.suptitle(f"Match Diagnostics: {image_name}", fontsize=18)
    all_df = pd.concat([matched_df, rejected_df])

    has_star_col = 'is_star' in matched_df.columns
    if has_star_col:
        m_stars    = matched_df[matched_df['is_star'].astype(bool)]
        m_nonstars = matched_df[~matched_df['is_star'].astype(bool)]
    else:
        m_stars, m_nonstars = matched_df, matched_df.iloc[0:0]

    def _scatter_matched(ax, col_x, col_y, **kwargs):
        if len(m_nonstars) > 0:
            ax.scatter(m_nonstars[col_x], m_nonstars[col_y],
                       c='orange', alpha=0.6, s=12, label='Matched non-star', **kwargs)
        if len(m_stars) > 0:
            ax.scatter(m_stars[col_x], m_stars[col_y],
                       c='blue', alpha=0.6, s=10, label='Matched star', **kwargs)

    # 1. Pixel Positions
    ax = axes[0, 0]
    lines_px = [[(r.x, r.y), (r.hx, r.hy)] for r in all_df.itertuples()]
    ax.add_collection(LineCollection(lines_px, colors='grey', alpha=0.1, linewidths=0.5, zorder=1))
    ax.scatter(all_df['hx'], all_df['hy'], c='grey', s=2, alpha=0.3, zorder=2)
    if len(rejected_df) > 0:
        ax.scatter(rejected_df['x'], rejected_df['y'], c='red', alpha=0.3, s=5, label='Rejected Gaia', zorder=3)
    if len(m_nonstars) > 0:
        ax.scatter(m_nonstars['x'], m_nonstars['y'], c='orange', alpha=0.6, s=12, label='Matched non-star', zorder=4)
    if len(m_stars) > 0:
        ax.scatter(m_stars['x'], m_stars['y'], c='blue', alpha=0.6, s=10, label='Matched star', zorder=5)
    if len(matched_df) > 0:
        x_m, y_m = matched_df['x'], matched_df['y']
        px_p, py_p = (x_m.max()-x_m.min())*0.05, (y_m.max()-y_m.min())*0.05
        ax.set_xlim(x_m.min()-px_p, x_m.max()+px_p); ax.set_ylim(y_m.min()-py_p, y_m.max()+py_p)
    ax.set_xlabel("X_Gaia (pixels)"); ax.set_ylabel("Y_Gaia (pixels)"); ax.set_title("Field Map (Pixels)"); ax.legend(fontsize=7)

    # 2. Gaia CMD (G vs BP-RP)
    ax = axes[1, 0]
    if 'color' in matched_df.columns:
        if len(rejected_df) > 0:
            ax.scatter(rejected_df['color'], rejected_df['mag'], c='red', alpha=0.15, s=5, label='Rejected')
        _scatter_matched(ax, 'color', 'mag')
        ax.invert_yaxis(); ax.set_xlabel("BP - RP (mag)"); ax.set_ylabel("Gaia G (mag)"); ax.set_title("Gaia Color-Magnitude Diagram"); ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "BP-RP not available", ha='center', va='center'); ax.set_title("CMD Placeholder")

    # 3. Gaia vs HST CMD (G vs G-HST_mag)
    ax = axes[1, 1]
    if len(rejected_df) > 0:
        ax.scatter(rejected_df['color_jwst'], rejected_df['mag'], c='red', alpha=0.15, s=5, label='Rejected')
    _scatter_matched(ax, 'color_jwst', 'mag')
    ax.invert_yaxis(); ax.set_xlabel("Gaia G - JWST (mag)"); ax.set_ylabel("Gaia G (mag)"); ax.set_title("Gaia G - JWST Color-Magnitude"); ax.legend(fontsize=7)

    # 4. XY Residual Scatter
    ax = axes[2, 0]
    if len(rejected_df) > 0:
        ax.scatter(rejected_df['dx'], rejected_df['dy'], c='red', alpha=0.2, s=8)
    _scatter_matched(ax, 'dx', 'dy')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5); ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    if len(matched_df) > 0:
        lim = max(matched_df['dx'].abs().max(), matched_df['dy'].abs().max()) * 2.5
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("dX (pixels)"); ax.set_ylabel("dY (pixels)"); ax.set_title("XY Residuals")

    # 5. Normalized Residuals
    ax = axes[2, 1]
    if len(rejected_df) > 0:
        rsx, rsy = np.sqrt(rejected_df['cxx']), np.sqrt(rejected_df['cyy'])
        ax.scatter(rejected_df['dx']/rsx, rejected_df['dy']/rsy, c='red', alpha=0.15, s=8)
    for sub_df, col in [(m_nonstars, 'orange'), (m_stars, 'blue')]:
        if len(sub_df) > 0:
            sx, sy = np.sqrt(sub_df['cxx']), np.sqrt(sub_df['cyy'])
            ax.scatter(sub_df['dx']/sx, sub_df['dy']/sy, c=col, alpha=0.5, s=15)
    ax.add_artist(plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', alpha=0.5))
    ax.add_artist(plt.Circle((0, 0), 5, color='red', fill=False, linestyle=':', alpha=0.5))
    ax.set_xlim(-8, 8); ax.set_ylim(-8, 8); ax.set_xlabel("dX / sigma_x"); ax.set_ylabel("dY / sigma_y"); ax.set_title("Normalized Residuals")

    # Calculate shared magnitude limits for Panels 6 and 8
    if len(all_df) > 0:
        mag_min, mag_max = all_df['mag'].min(), all_df['mag'].max()
        mag_pad = (mag_max - mag_min) * 0.05
        mag_lims = (mag_min - mag_pad, mag_max + mag_pad)
    else:
        mag_lims = None

    # 6. Combined Residual vs Gaia Magnitude (Log-Scaled Y)
    ax = axes[3, 0]
    if len(rejected_df) > 0:
        res_r = np.sqrt(rejected_df['dx']**2 + rejected_df['dy']**2)
        ax.scatter(rejected_df['mag'], res_r, c='red', alpha=0.15, s=5, label='Rejected')
    for sub_df, col, lbl in [(m_nonstars, 'orange', 'Non-star'), (m_stars, 'blue', 'Star')]:
        if len(sub_df) > 0:
            res = np.sqrt(sub_df['dx']**2 + sub_df['dy']**2)
            ax.scatter(sub_df['mag'], res, c=col, alpha=0.5, s=10, label=lbl)
    ax.set_yscale('log'); ax.set_xlabel("Gaia G Magnitude"); ax.set_ylabel("Residual Size (pixels)"); ax.set_title("Residual Magnitude vs Gaia Mag"); ax.legend(fontsize=7)
    if mag_lims: ax.set_xlim(mag_lims)

    # 7. Sigma Histogram
    ax = axes[3, 1]
    bins = np.linspace(0, 10, 50)
    if len(m_nonstars) > 0:
        ax.hist(m_nonstars['sigma'], bins=bins, color='orange', alpha=0.5, label='Non-star')
    if len(m_stars) > 0:
        ax.hist(m_stars['sigma'], bins=bins, color='blue', alpha=0.6, label='Star')
    if len(rejected_df) > 0:
        rej_near = rejected_df[rejected_df['sigma'] < 10.0]
        ax.hist(rej_near['sigma'], bins=bins, color='red', alpha=0.3, label='Rejected (<10s)')
    ax.axvline(5, color='red', linestyle='--'); ax.set_yscale('log'); ax.set_xlabel("Sigma"); ax.set_ylabel("Count (Log)"); ax.set_title("Sigma Distribution"); ax.legend(fontsize=7)

    # 8. Residual Sigma vs Gaia Magnitude
    ax = axes[4, 0]
    if len(rejected_df) > 0:
        rej_near = rejected_df[rejected_df['sigma'] < 15.0]
        ax.scatter(rej_near['mag'], rej_near['sigma'], c='red', alpha=0.15, s=5, label='Rejected (<15s)')
    for sub_df, col, lbl in [(m_nonstars, 'orange', 'Non-star'), (m_stars, 'blue', 'Star')]:
        if len(sub_df) > 0:
            ax.scatter(sub_df['mag'], sub_df['sigma'], c=col, alpha=0.5, s=10, label=lbl)
    ax.axhline(5, color='red', linestyle='--', label='Threshold (5s)')
    ax.set_xlabel("Gaia G Magnitude"); ax.set_ylabel("Residual Sigma"); ax.set_title("Sigma vs Gaia Magnitude"); ax.legend(fontsize=7)
    if mag_lims: ax.set_xlim(mag_lims)

    # 9. Color vs Color
    ax = axes[4, 1]
    if 'color' in matched_df.columns:
        if len(rejected_df) > 0:
            ax.scatter(rejected_df['color'], rejected_df['color_jwst'], c='red', alpha=0.15, s=5, label='Rejected')
        _scatter_matched(ax, 'color', 'color_jwst')
        ax.invert_yaxis(); ax.set_xlabel("BP - RP (mag)"); ax.set_ylabel("G - JWST (mag)"); ax.set_title("Color-Color Diagram"); ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "BP-RP not available", ha='center', va='center'); ax.set_title("CMD Placeholder")

    # 10. Gaia Proper Motions (PMRA vs PMDec)
    ax = axes[0, 1]
    if len(rejected_df) > 0:
        ax.scatter(rejected_df['pmra'], rejected_df['pmdec'], c='red', alpha=0.15, s=5, label='Rejected')
    _scatter_matched(ax, 'pmra', 'pmdec')
    ax.set_xlabel("PMRA (mas/yr)"); ax.set_ylabel("PMDec (mas/yr)"); ax.set_title("Gaia Proper Motions"); ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(out_dir, "diagnostic_plots.png"), dpi=150); plt.close()

class FileLogger(object):
    def __init__(self, filename): self.log = open(filename, "w")
    def write(self, message): self.log.write(message)
    def flush(self): self.log.flush()

# ---------------------------------------------------------------------------
# Discovery and refinement helpers
# ---------------------------------------------------------------------------

def _plot_offset_histogram(hist, xed, yed, peaks, title, filepath):
    fig, ax = plt.subplots(figsize=(6, 5))
    with np.errstate(divide='ignore', invalid='ignore'):
        # log_hist = np.log10(hist.T + 1e-30)
        log_hist = np.log10(hist.T)
        log_hist = hist.T
        log_hist[log_hist == 0] = np.nan
        # finite = log_hist[np.isfinite(log_hist)]
        # log_hist[~np.isfinite(log_hist)] = finite.min() if len(finite) else 0
    im = ax.imshow(log_hist, origin='lower', aspect='equal',
                   extent=[xed[0], xed[-1], yed[0], yed[-1]], cmap='viridis')
    plt.colorbar(im, ax=ax, label='log10(weighted density)')
    for dx, dy, _ in peaks:
        ax.axvline(dx, color='red', lw=0.8, ls='--', alpha=0.7)
        ax.axhline(dy, color='red', lw=0.8, ls='--', alpha=0.7)
    ax.set_xlabel('dx  (JWST − Gaia guess, pixels)')
    ax.set_ylabel('dy  (JWST − Gaia guess, pixels)')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filepath, dpi=120)
    plt.close(fig)


def _save_offset_histogram(best, image_name, out_dir):
    if best.get('offset_hist') is None:
        return
    ds_str = f"ds={best.get('best_ds', 0.0):+.4f}"
    title = f'{image_name}  |  best tier q<{best["q"]} m<{best["m"]:.1f}  {ds_str}'
    _plot_offset_histogram(best['offset_hist'], best['offset_xed'], best['offset_yed'],
                           best.get('offset_peaks', []), title,
                           os.path.join(out_dir, 'offset_histogram.png'))


def _run_4p_discovery(hst_d, gaia_f, params, max_mag_diff, scale_sweep=False, discovery_max_offset=50):
    """
    Tier-walks qfit x mag limits to find a physically plausible 4P similarity seed.

    hst_d  keys: x, y, mag, C, qfit, chi2
    gaia_f keys: x, y, C, mag, err, has_pms, xguess, yguess

    Returns best-tier dict or None.
    """
    seed_margin = 2000
    near = (np.abs(gaia_f['x'] - params['x_cen']) <= seed_margin) & \
           (np.abs(gaia_f['y'] - params['y_cen']) <= seed_margin)
    field_idx = np.where(near)[0]
    n_subset = 1000
    if len(field_idx) > n_subset:
        seed_idx = field_idx[np.argsort(gaia_f['err'][near])[:n_subset]]
    else:
        seed_idx = field_idx

    xg_s = gaia_f['xguess'][seed_idx]
    yg_s = gaia_f['yguess'][seed_idx]
    Cg_s = gaia_f['C'][seed_idx]

    tree_hst = KDTree(np.column_stack([hst_d['x'], hst_d['y']]))

    qfit_limits = [0.1, 0.2, 0.3, 0.5, 1.0, np.inf]
    mag_limits = np.arange(hst_d['mag'].min() + 1.0, hst_d['mag'].max() + 0.5, 1.0)
    if len(mag_limits) == 0:
        mag_limits = [hst_d['mag'].max()]
    mag_limits[-1] = hst_d['mag'].max()

    discovered = []
    print(f"  Walking over {len(qfit_limits)*len(mag_limits)} tiers for 4P discovery...")

    for qlim in qfit_limits:
        for mlim in mag_limits:
            curr_q_mag_lims = (hst_d['qfit'] <= qlim) & (hst_d['mag'] <= mlim)
            h_mask = curr_q_mag_lims & (hst_d['qfit'] > 0.0) & (hst_d['chi2'] < 5.0)
            if np.sum(h_mask) < 3:
                h_mask = curr_q_mag_lims & (hst_d['qfit'] >= 0.0) & (hst_d['chi2'] < 5.0)
            if np.sum(h_mask) < 3:
                h_mask = curr_q_mag_lims & (hst_d['qfit'] >= 0.0) & (hst_d['chi2'] < 10.0)
            if np.sum(h_mask) < 3:
                continue
            h_idx_tier = np.where(h_mask)[0]

            curr_keep = np.ones(len(xg_s)).astype(bool)
            if np.sum(gaia_f['has_pms'][seed_idx]) >= 3:
                curr_keep = gaia_f['has_pms'][seed_idx].copy()

            best_ds, offset_peaks, tier_hist, tier_xed, tier_yed = find_scale_and_offset(
                xg_s[curr_keep], yg_s[curr_keep], gaia_f['err'][seed_idx][curr_keep],
                hst_d['x'][h_idx_tier], hst_d['y'][h_idx_tier], hst_d['mag'][h_idx_tier],
                cov1=Cg_s[curr_keep], cov2=hst_d['C'][h_idx_tier],
                x_cen=params['x_cen'], y_cen=params['y_cen'],
                max_offset=discovery_max_offset, bin_size=1, top_n=3,
                ds_range=(-0.02, 0.02) if scale_sweep else (0.0, 0.0), n_scales=41 if scale_sweep else 1,
                return_histogram=True
            )
            dx_off, dy_off, _ = offset_peaks[0]
            xg_tier = xg_s + best_ds * (xg_s - params['x_cen']) + dx_off
            yg_tier = yg_s + best_ds * (yg_s - params['y_cen']) + dy_off

            # Try 40px (tight prior) then 100px (header fallback)
            valid_seed = False
            for cur_rad in [40.0, 100.0]:
                tree_h_tier = KDTree(np.column_stack([hst_d['x'][h_idx_tier], hst_d['y'][h_idx_tier]]))
                dists, h_idxs_tier = tree_h_tier.query(np.column_stack([xg_tier, yg_tier]), k=1, distance_upper_bound=cur_rad)
                valid_idx = dists < cur_rad
                if np.sum(valid_idx) >= 3:
                    valid_seed = True
                    break
            if not valid_seed:
                continue

            # Greedy 1-to-1 cleanup using log-probability cost
            h_v_full = h_idx_tier[h_idxs_tier[valid_idx]]
            g_v_seed = np.where(valid_idx)[0]
            dx = hst_d['x'][h_v_full] - xg_tier[g_v_seed]
            dy = hst_d['y'][h_v_full] - yg_tier[g_v_seed]
            C_tot_seed = Cg_s[g_v_seed] + hst_d['C'][h_v_full]
            costs_seed = compute_logprob_cost(dx, dy, C_tot_seed)
            mdf_seed = pd.DataFrame({'g_seed': g_v_seed, 'h_full': h_v_full, 'c': costs_seed})\
                         .sort_values('c').drop_duplicates('h_full').drop_duplicates('g_seed')
            if len(mdf_seed) < 3:
                continue

            h_b_idx = mdf_seed['h_full'].values
            g_b_full_idx = seed_idx[mdf_seed['g_seed'].values]
            xh_b, yh_b = hst_d['x'][h_b_idx], hst_d['y'][h_b_idx]
            xg_b, yg_b = gaia_f['x'][g_b_full_idx], gaia_f['y'][g_b_full_idx]
            mag_diffs = gaia_f['mag'][g_b_full_idx] - hst_d['mag'][h_b_idx]
            curr_keep = np.ones(len(xh_b), dtype=bool)
            if np.sum(gaia_f['has_pms'][g_b_full_idx]) >= 3:
                curr_keep = gaia_f['has_pms'][g_b_full_idx].copy()
            zp = np.median(mag_diffs[curr_keep])
            cur_M = np.eye(2)

            # Iterative 4P fit with sigma rejection
            for _ in range(5):
                res_4p, _, C_params, _ = fit_4p_weighted(
                    xh_b[curr_keep], yh_b[curr_keep],
                    xg_b[curr_keep], yg_b[curr_keep],
                    hst_d['C'][h_b_idx[curr_keep]],
                    gaia_f['C'][g_b_full_idx[curr_keep]],
                    initial_M=cur_M)
                A, B, C, D, xs_o, ys_o, xt_o, yt_o = res_4p
                cur_M = np.array([[A, B], [C, D]])

                xh_p, yh_p = apply_affine(xh_b, yh_b, A, B, C, D, xs_o, ys_o, xt_o, yt_o)
                dx_v, dy_v = xg_b - xh_p, yg_b - yh_p
                C_proj = np.einsum('ij,njk,lk->nil', cur_M, hst_d['C'][h_b_idx], cur_M)
                dxh_v, dyh_v = xh_b - xs_o, yh_b - ys_o
                J = np.zeros((len(dxh_v), 2, 4))
                J[:, 0, 0], J[:, 0, 1], J[:, 0, 2] = dxh_v, -dyh_v, 1.0
                J[:, 1, 0], J[:, 1, 1], J[:, 1, 3] = dyh_v, dxh_v, 1.0
                C_model = np.einsum('nij,jk,nlk->nil', J, C_params, J)
                C_tot_v = gaia_f['C'][g_b_full_idx] + C_proj + C_model
                sigs_v = compute_mahalanobis(dx_v, dy_v, C_tot_v)
                costs_v = compute_logprob_cost(dx_v, dy_v, C_tot_v)
                chi2 = np.sum(sigs_v[curr_keep])
                cost = np.sum(costs_v[curr_keep])

                ds_4p = np.sqrt(dx_v**2 + dy_v**2)
                finite_dists = np.isfinite(ds_4p)
                p16, p50 = np.nanpercentile(ds_4p[finite_dists & curr_keep], [16, 50])
                thresh = min(max(p50 + 3*(p50-p16), 1), cur_rad)
                if not np.isfinite(thresh):
                    thresh = cur_rad

                good_v = (ds_4p < thresh) & (np.abs(mag_diffs - zp) < max_mag_diff) & (sigs_v < 5.0)
                if np.sum(good_v) < 3:
                    break
                if np.all(curr_keep == good_v):
                    break
                curr_keep[:] = good_v
                zp = np.median(mag_diffs[curr_keep])

            g_b_full_idx = g_b_full_idx[curr_keep]
            h_b_idx = h_b_idx[curr_keep]
            if np.sum(good_v) < 3:
                continue

            scale_fit = np.sqrt(A*D - B*C)
            rot_fit = np.degrees(np.arctan2(B - C, A + D))
            if (0.98*params['initial_scale'] <= scale_fit <= 1.02*params['initial_scale']) and (abs(rot_fit) < 0.2):
                red_chi2 = chi2 / (2*len(h_b_idx) - 4)
                red_cost = cost - np.log(2*len(h_b_idx) - 4)
                zp_tier = np.median(gaia_f['mag'][g_b_full_idx] - hst_d['mag'][h_b_idx])
                discovered.append({
                    'A': A, 'B': B, 'C': C, 'D': D,
                    'xs_o': xs_o, 'ys_o': ys_o, 'xt_o': xt_o, 'yt_o': yt_o,
                    'n_match': len(h_b_idx), 'red_chi2': red_chi2, 'red_cost': red_cost,
                    'q': qlim, 'm': mlim, 'zp': zp_tier,
                    'h_v': h_b_idx, 'g_v': g_b_full_idx,
                    'best_ds': best_ds,
                    'offset_peaks': offset_peaks,
                    'offset_hist': tier_hist, 'offset_xed': tier_xed, 'offset_yed': tier_yed,
                })
                peaks_str = "  |  ".join(f"dx={dx:.1f},dy={dy:.1f}(s={s:.2f})" for dx, dy, s in offset_peaks)
                print(f"    q<{qlim} m<{mlim:.1f}: {len(h_b_idx)} stars, red_chi2={red_chi2:.3f}, "
                      f"red_cost={red_cost:.2f}, zp={zp_tier:.3f}, scale={scale_fit:.6f}, rot={rot_fit:.4f} | "
                      f"offsets(ds={best_ds:+.4f}): {peaks_str}")

    if not discovered:
        return None
    return min(discovered, key=lambda x: x['red_cost'])


def _run_affine_refinement(best_4p, hst_d, gaia_f, tree_gaia, max_mag_diff, use_resid_floor=True):
    """
    Upgrades 4P seeds to a 6P affine transform and iterates until convergence.

    hst_d  keys: x, y, mag, C
    gaia_f keys: x, y, C, mag

    Returns (A, B, C, D, xs_o, ys_o, xt_o, yt_o, C_params, resid_cov, zp, h_f, g_f).
    """
    A, B, C, D = best_4p['A'], best_4p['B'], best_4p['C'], best_4p['D']
    xs_o, ys_o = best_4p['xs_o'], best_4p['ys_o']
    xt_o, yt_o = best_4p['xt_o'], best_4p['yt_o']
    zp = best_4p['zp']
    h_idx_b, g_idx_b = best_4p['h_v'], best_4p['g_v']
    M = np.array([[A, B], [C, D]])

    # Upgrade 4P seeds to initial 6P affine fit
    xh_b, yh_b = hst_d['x'][h_idx_b], hst_d['y'][h_idx_b]
    xg_b, yg_b = gaia_f['x'][g_idx_b], gaia_f['y'][g_idx_b]
    fit_res, _, C_params, _ = fit_affine_weighted(
        xh_b, yh_b, xg_b, yg_b, hst_d['C'][h_idx_b], gaia_f['C'][g_idx_b], initial_M=M)
    A, B, C, D, xs_o, ys_o, xt_o, yt_o = fit_res
    M = np.array([[A, B], [C, D]])

    xh_in_g, yh_in_g = apply_affine(xh_b, yh_b, A, B, C, D, xs_o, ys_o, xt_o, yt_o)
    dx_v, dy_v = xg_b - xh_in_g, yg_b - yh_in_g
    resid_sigma_x = 0.5 * np.diff(np.nanpercentile(dx_v, [16, 84]))[0]
    resid_sigma_y = 0.5 * np.diff(np.nanpercentile(dy_v, [16, 84]))[0]
    resid_cov = np.diag(np.array([resid_sigma_x, resid_sigma_y])**2) if use_resid_floor else np.zeros((2, 2))

    ratio, rot = np.sqrt(A*D-B*C), np.degrees(np.arctan2(B-C, A+D))
    print(f"  Init 6P: {len(xh_b)} seeds, scale={ratio:.6f}, rot={rot:.4f}deg, "
          f"on_skew={0.5*(A-D):.2e}, off_skew={0.5*(B+C):.2e}, "
          f"resid=[{resid_sigma_x:.4f},{resid_sigma_y:.4f}]px, zp={zp:.3f}")

    h_f, g_f = h_idx_b, g_idx_b
    for it in range(10):
        xh_in_g, yh_in_g = apply_affine(hst_d['x'], hst_d['y'], A, B, C, D, xs_o, ys_o, xt_o, yt_o)
        ds, g_idxs = tree_gaia.query(np.column_stack([xh_in_g, yh_in_g]), k=5, distance_upper_bound=100)
        h_idx_all = np.repeat(np.arange(len(hst_d['x'])), 5)
        valid = ds.flatten() < 100
        h_v, g_v = h_idx_all[valid], g_idxs.flatten()[valid]
        if len(h_v) < 3:
            print(f"  Iter {it}: only {len(h_v)} candidates within 100px. Breaking.")
            break

        dx_v, dy_v = gaia_f['x'][g_v] - xh_in_g[h_v], gaia_f['y'][g_v] - yh_in_g[h_v]
        C_proj = np.einsum('ij,njk,lk->nil', M, hst_d['C'][h_v], M)
        dxh_v, dyh_v = hst_d['x'][h_v] - xs_o, hst_d['y'][h_v] - ys_o
        J = np.zeros((len(h_v), 2, 6))
        J[:, 0, 0], J[:, 0, 1], J[:, 0, 2] = dxh_v, dyh_v, 1.0
        J[:, 1, 3], J[:, 1, 4], J[:, 1, 5] = dxh_v, dyh_v, 1.0
        C_model = np.einsum('nij,jk,nlk->nil', J, C_params, J)
        C_total = gaia_f['C'][g_v] + C_proj + C_model + resid_cov

        sigs_v = compute_mahalanobis(dx_v, dy_v, C_total)
        costs_v = compute_logprob_cost(dx_v, dy_v, C_total)
        mag_diffs = gaia_f['mag'][g_v] - hst_d['mag'][h_v]
        costs_v += ((mag_diffs - zp) / 1.0)**2
        costs_v[np.abs(mag_diffs - zp) > max_mag_diff] = np.inf

        mdf = pd.DataFrame({'h': h_v, 'g': g_v, 's': sigs_v, 'c': costs_v,
                             'dx': dx_v, 'dy': dy_v})\
                .sort_values('c').drop_duplicates('g').drop_duplicates('h')
        mdf['mag_diff'] = gaia_f['mag'][mdf['g'].values] - hst_d['mag'][mdf['h'].values]
        good = mdf[(mdf['s'] < 5.0) & (np.abs(mdf['mag_diff'] - zp) < max_mag_diff)]
        if len(good) < 3:
            print(f"  Iter {it}: only {len(good)} stars passed sigma<5 filter. Breaking.")
            break

        h_f, g_f = good['h'].values, good['g'].values
        fit_res_new, _, C_params, _ = fit_affine_weighted(
            hst_d['x'][h_f], hst_d['y'][h_f], gaia_f['x'][g_f], gaia_f['y'][g_f],
            hst_d['C'][h_f], gaia_f['C'][g_f], initial_M=M)
        change = np.abs(fit_res_new[0] - A) + np.abs(fit_res_new[1] - B)
        A, B, C, D, xs_o, ys_o, xt_o, yt_o = fit_res_new
        M = np.array([[A, B], [C, D]])

        resid_sigma_x = 0.5 * np.diff(np.nanpercentile(good['dx'], [16, 84]))[0]
        resid_sigma_y = 0.5 * np.diff(np.nanpercentile(good['dy'], [16, 84]))[0]
        resid_cov = np.diag(np.array([resid_sigma_x, resid_sigma_y])**2) if use_resid_floor else np.zeros((2, 2))
        zp = np.median(good['mag_diff'])

        ratio, rot = np.sqrt(A*D-B*C), np.degrees(np.arctan2(B-C, A+D))
        print(f"  Iter {it}: {len(h_f)} matches, scale={ratio:.6f}, rot={rot:.4f}deg, "
              f"resid=[{resid_sigma_x:.4f},{resid_sigma_y:.4f}]px, zp={zp:.3f}")
        if it > 5 and change < 1e-11:
            break

    return A, B, C, D, xs_o, ys_o, xt_o, yt_o, C_params, resid_cov, zp, h_f, g_f


# ---------------------------------------------------------------------------
# Main per-image processor
# ---------------------------------------------------------------------------

def process_single_image(img, gaia_df, hst_pix_floor=0.01, min_matches=3, zero_pm=False, max_mag_diff=3.0, scale_sweep=False, discovery_max_offset=50, use_resid_floor=True):
    start_time = time.time()
    image_name = os.path.basename(img['cal']).replace("_cal.fits", "")
    log_file, original_stdout = os.path.join(img['root'], "processing_log.txt"), sys.stdout
    sys.stdout = FileLogger(log_file)
    print(f"Starting {image_name}...", file=original_stdout)
    try:
        print(f"--- Processing JWST image: {image_name} ---")
        params = get_jwst_params(img['cal'])
        if params is None:
            print(f"Finished {image_name}: Failed to load parameters.", file=original_stdout)
            return
        params['min_matches'] = min_matches

        # --- Propagate Gaia to JWST epoch and project to pixel frame ---
        ra_prop, dec_prop, Ct = propagate_gaia_with_cov(gaia_df, params['obs_epoch_mjd'], zero_pm=zero_pm)
        dx_deg_full = rd2x(ra_prop, dec_prop, params['ra_cen'], params['dec_cen'])
        dy_deg_full = rd2y(ra_prop, dec_prop, params['ra_cen'], params['dec_cen'])
        scale_deg = params['pixel_scale'] / 3600.0
        mas_to_px = 1.0 / (params['pixel_scale'] * 1000.0)

        # Convert sky covariance (mas²) to pixel frame, accounting for +X = -RA
        C_pix_gaia = Ct * mas_to_px**2
        C_pix_gaia[:, 0, :] *= -1
        C_pix_gaia[:, :, 0] *= -1
        x_gaia_proj = params['x_cen'] - dx_deg_full / scale_deg
        y_gaia_proj = params['y_cen'] + dy_deg_full / scale_deg

        # Rotate Gaia positions into the approximate JWST detector frame using PA_APER
        theta_init = np.radians(-params['orientat'])
        init_rot_mat    = np.array([[ np.cos(theta_init), np.sin(theta_init)],
                                    [-np.sin(theta_init), np.cos(theta_init)]])
        init_inv_rot_mat = np.linalg.inv(init_rot_mat)
        xy_gaia_proj = np.einsum('ij,nj->ni',
                                  init_inv_rot_mat,
                                  np.column_stack([x_gaia_proj, y_gaia_proj])
                                  - np.array([params['x_cen'], params['y_cen']]))\
                       + np.array([params['x_cen'], params['y_cen']])
        x_gaia_proj, y_gaia_proj = xy_gaia_proj[:, 0], xy_gaia_proj[:, 1]
        C_pix_gaia = np.einsum('ij,njk,lk->nil', init_inv_rot_mat, C_pix_gaia, init_inv_rot_mat)
        gaia_err_total = np.power(np.linalg.det(C_pix_gaia), 0.25)
        has_gaia_pms = np.isfinite(gaia_df['pmra'].to_numpy())

        # --- Filter to stars near the JWST field ---
        margin = 3000
        in_field = (np.abs(x_gaia_proj - params['x_cen']) <= margin) & \
                   (np.abs(y_gaia_proj - params['y_cen']) <= margin)
        if not np.any(in_field):
            print(f"Finished {image_name}: No stars in field.", file=original_stdout)
            return

        x_g_in    = x_gaia_proj[in_field]
        y_g_in    = y_gaia_proj[in_field]
        C_g_in    = C_pix_gaia[in_field]
        g_err_in  = gaia_err_total[in_field]
        g_mag_in  = gaia_df['gmag'].values[in_field]
        g_color_in = gaia_df['bp_rp'].values[in_field] if 'bp_rp' in gaia_df.columns else None
        g_pmra_in  = gaia_df['pmra'].values[in_field]
        g_pmdec_in = gaia_df['pmdec'].values[in_field]
        ra_in, dec_in = ra_prop[in_field], dec_prop[in_field]
        in_has_pms = has_gaia_pms[in_field]

        # --- Load JWST catalog ---
        jwst_cat = fits.getdata(img['catalog'])

        # Require is_star_candidate; skip image if absent
        if 'is_star_candidate' not in jwst_cat.dtype.names:
            msg = (f'\n  WARNING: {image_name} catalog is missing the '
                   f'is_star_candidate column — image SKIPPED.\n'
                   f'  Re-run PSF fitting to produce updated catalogs before cross-matching.')
            print(msg)
            print(f'Finished {image_name}: SKIPPED — no is_star_candidate column.', file=original_stdout)
            return

        _orig_row_idx = np.arange(len(jwst_cat))
        _valid = (np.isfinite(jwst_cat['x_gdc'].astype(float)) &
                  np.isfinite(jwst_cat['y_gdc'].astype(float)))
        if not _valid.all():
            print(f'  Dropping {(~_valid).sum()} NaN/inf rows from JWST catalog')
            jwst_cat = jwst_cat[_valid]
            _orig_row_idx = _orig_row_idx[_valid]
        x_hst       = jwst_cat['x_gdc'].astype(float)
        y_hst       = jwst_cat['y_gdc'].astype(float)
        mag_hst_gdc = jwst_cat['mag_gdc'].astype(float)
        is_star     = jwst_cat['is_star_candidate'].astype(bool)
        mag_err_hst = jwst_cat['mag_err_gdc'].astype(float) if 'mag_err_gdc' in jwst_cat.dtype.names else None
        if 'mag_st_gdc' not in jwst_cat.dtype.names:
            raise ValueError(
                f"Catalog missing 'mag_st_gdc' column — stale jwst1pass output. "
                f"Delete the catalog so jwst1pass will re-run."
            )
        mag_hst    = jwst_cat['mag_st_gdc'].astype(float)
        mag_st_hst = mag_hst
        mag_ab_hst = jwst_cat['mag_ab'].astype(float) if 'mag_ab' in jwst_cat.dtype.names else None
        C_pix_hst = np.zeros((len(x_hst), 2, 2))
        C_pix_hst[:, 0, 0] = jwst_cat['cov_xx_gdc'].astype(float) + hst_pix_floor**2
        C_pix_hst[:, 1, 1] = jwst_cat['cov_yy_gdc'].astype(float) + hst_pix_floor**2
        C_pix_hst[:, 0, 1] = jwst_cat['cov_xy_gdc'].astype(float)
        C_pix_hst[:, 1, 0] = C_pix_hst[:, 0, 1]

        n_stars = is_star.sum()
        print(f'  JWST catalog: {len(x_hst)} sources, {n_stars} star candidates ({100*n_stars/len(x_hst):.1f}%)')

        # Scale-adjusted Gaia guess positions for seeding.
        # initial_scale is the detector→Gaia scale from the 4P fit, so the inverse
        # (Gaia→detector) is 1/initial_scale.
        inv_initial_scale = 1.0 / params['initial_scale']
        xg_guess_in = params['x_cen'] + (x_g_in - params['x_cen']) * inv_initial_scale
        yg_guess_in = params['y_cen'] + (y_g_in - params['y_cen']) * inv_initial_scale

        gaia_field = {
            'x': x_g_in, 'y': y_g_in, 'C': C_g_in, 'mag': g_mag_in,
            'err': g_err_in, 'has_pms': in_has_pms,
            'xguess': xg_guess_in, 'yguess': yg_guess_in,
        }
        # 4P discovery uses high-confidence star candidates only (tight qfit/chi2
        # tiers make the geometric matching more reliable).  The 6P affine
        # refinement and final pass use all sources: non-stars contribute
        # additional positional constraints once a good transform seed exists.
        star_indices = np.where(is_star)[0]
        hst_data = {
            'x': x_hst[is_star], 'y': y_hst[is_star], 'mag': mag_hst[is_star],
            'C': C_pix_hst[is_star],
            'qfit': jwst_cat['qfit'].astype(float)[is_star],
            'chi2': jwst_cat['chi2'].astype(float)[is_star],
        }
        # All sources (stars + non-stars) for 6P refinement and final pass.
        hst_data_all = {
            'x': x_hst, 'y': y_hst, 'mag': mag_hst,
            'C': C_pix_hst,
        }
        tree_gaia_all = KDTree(np.column_stack([x_g_in, y_g_in]))

        # --- 4P Discovery ---
        best = _run_4p_discovery(hst_data, gaia_field, params, max_mag_diff, scale_sweep=scale_sweep, discovery_max_offset=discovery_max_offset)
        if best is None:
            print(f"Finished {image_name}: 4P Discovery failed to find physically plausible matches.", file=original_stdout)
            return
        print(f"  4P Discovery Succeeded: Best Tier Q<{best['q']}, Mag<{best['m']:.1f} "
              f"({best['n_match']} stars, red_chi2={best['red_chi2']:.2f}, red_cost={best['red_cost']:.2f})")

        # --- Save offset histogram plot for the best discovery tier ---
        _save_offset_histogram(best, image_name, img['root'])

        # --- Affine Refinement (all sources) ---
        # Seed indices from 4P discovery are into hst_data (star-only); remap
        # them to full-array indices so _run_affine_refinement can work with
        # hst_data_all which contains every source.
        best_all = {**best, 'h_v': star_indices[best['h_v']]}
        A, B, C, D, xs_o, ys_o, xt_o, yt_o, C_params, resid_cov, zp, h_f, g_f = \
            _run_affine_refinement(best_all, hst_data_all, gaia_field, tree_gaia_all, max_mag_diff, use_resid_floor=use_resid_floor)
        M = np.array([[A, B], [C, D]])

        # --- Final pass: gather all candidates with the converged transform ---
        xh_in_g, yh_in_g = apply_affine(x_hst, y_hst, A, B, C, D, xs_o, ys_o, xt_o, yt_o)
        ds, g_idxs = tree_gaia_all.query(np.column_stack([xh_in_g, yh_in_g]), k=5, distance_upper_bound=100)
        h_idx_all = np.repeat(np.arange(len(x_hst)), 5)
        valid = ds.flatten() < 100
        h_v, g_v = h_idx_all[valid], g_idxs.flatten()[valid]

        dx_v, dy_v = x_g_in[g_v] - xh_in_g[h_v], y_g_in[g_v] - yh_in_g[h_v]
        C_proj = np.einsum('ij,njk,lk->nil', M, C_pix_hst[h_v], M)
        dxh_v, dyh_v = x_hst[h_v] - xs_o, y_hst[h_v] - ys_o
        J = np.zeros((len(h_v), 2, 6))
        J[:, 0, 0], J[:, 0, 1], J[:, 0, 2] = dxh_v, dyh_v, 1.0
        J[:, 1, 3], J[:, 1, 4], J[:, 1, 5] = dxh_v, dyh_v, 1.0
        C_model = np.einsum('nij,jk,nlk->nil', J, C_params, J)
        C_total = C_g_in[g_v] + C_proj + C_model + resid_cov

        sigs_v = compute_mahalanobis(dx_v, dy_v, C_total)
        costs_v = compute_logprob_cost(dx_v, dy_v, C_total)
        mag_diffs = g_mag_in[g_v] - mag_hst[h_v]
        costs_v += ((mag_diffs - zp) / 1.0)**2
        costs_v[np.abs(mag_diffs - zp) > max_mag_diff] = np.inf

        final_mdf = pd.DataFrame({
            'h': h_v, 'g': g_v, 's': sigs_v, 'c': costs_v,
            'dx': dx_v, 'dy': dy_v, 'mag_diff': mag_diffs,
            'cxx': C_total[:, 0, 0], 'cyy': C_total[:, 1, 1],
        }).sort_values('c')
        all_mdf   = final_mdf.drop_duplicates('g')
        final_mdf = final_mdf.drop_duplicates('g').drop_duplicates('h')
        final_mdf = final_mdf[(final_mdf['s'] < 5.0) & (np.abs(final_mdf['mag_diff'] - zp) < max_mag_diff)]

        h_final, g_final = final_mdf['h'].values, final_mdf['g'].values
        print(f"  Final matches found: {len(h_final)}")
        if len(h_final) == 0:
            print(f"Finished {image_name}: Final match filtering removed all stars.", file=original_stdout)
            return

        # --- Build diagnostic dataframe ---
        diag_df = pd.DataFrame({
            'h_idx': all_mdf['h'], 'g_idx': all_mdf['g'],
            'x': x_g_in[all_mdf['g']], 'y': y_g_in[all_mdf['g']],
            'hx': xh_in_g[all_mdf['h']], 'hy': yh_in_g[all_mdf['h']],
            'ra': ra_in[all_mdf['g']], 'dec': dec_in[all_mdf['g']],
            'dx': all_mdf['dx'], 'dy': all_mdf['dy'],
            'sigma': all_mdf['s'], 'cxx': all_mdf['cxx'], 'cyy': all_mdf['cyy'],
            'mag': g_mag_in[all_mdf['g']], 'mag_hst': mag_hst[all_mdf['h']],
            'pmra': g_pmra_in[all_mdf['g']], 'pmdec': g_pmdec_in[all_mdf['g']],
            'is_star': is_star[all_mdf['h'].values],
        })
        if g_color_in is not None:
            diag_df['color'] = g_color_in[diag_df['g_idx']]
        diag_df['color_jwst'] = diag_df['mag'] - diag_df['mag_hst'] - zp

        final_match_keys = set(zip(h_final, g_final))
        is_m = diag_df.apply(lambda r: (int(r.h_idx), int(r.g_idx)) in final_match_keys, axis=1)
        if not np.any(is_m):
            print(f"Finished {image_name}: Final match filtering removed all stars.", file=original_stdout)
            return

        # --- Save outputs ---
        save_diagnostic_plots(img['root'], image_name, diag_df[is_m], diag_df[~is_m])
        final_matches = diag_df[is_m].copy()

        output = Table()
        output['jwst_index']      = _orig_row_idx[final_matches['h_idx'].values]
        output['jwst_x_gdc']      = x_hst[final_matches['h_idx'].values]
        output['jwst_y_gdc']      = y_hst[final_matches['h_idx'].values]
        output['jwst_mag_gdc']    = mag_hst_gdc[final_matches['h_idx'].values]
        if mag_err_hst is not None:
            output['jwst_mag_err_gdc'] = mag_err_hst[final_matches['h_idx'].values]
        if mag_st_hst is not None:
            output['jwst_mag_st_gdc']  = mag_st_hst[final_matches['h_idx'].values]
        if mag_ab_hst is not None:
            output['jwst_mag_ab']      = mag_ab_hst[final_matches['h_idx'].values]
        output['gaia_source_id'] = gaia_df.iloc[in_field]['source_id'].to_numpy(dtype=np.int64)[final_matches['g_idx'].values]
        output['has_gaia_pms']   = has_gaia_pms[in_field][final_matches['g_idx'].values]
        output['gaia_ra_prop']   = ra_in[final_matches['g_idx'].values]
        output['gaia_dec_prop']  = dec_in[final_matches['g_idx'].values]
        output['gaia_gmag']      = g_mag_in[final_matches['g_idx'].values]
        # residual_mag uses the same calibrated magnitude that was used for ZP
        # estimation (mag_hst = mag_st_gdc when available, else mag_gdc).
        mag_hst_for_resid = mag_hst[final_matches['h_idx'].values]
        output['residual_mag']   = output['gaia_gmag'] - (mag_hst_for_resid + zp)
        output['residual_x']     = final_matches['dx'].values
        output['residual_y']     = final_matches['dy'].values
        output['residual_sigma'] = final_matches['sigma'].values
        output['is_star']        = is_star[final_matches['h_idx'].values]
        output.write(os.path.join(img['root'], "matched_gaia.csv"), format='ascii.csv', overwrite=True)

        ratio, rot = np.sqrt(A*D - B*C), np.degrees(np.arctan2(B-C, A+D))
        on_skew, off_skew = 0.5*(A-D), 0.5*(B+C)
        trans_out = Table()
        trans_out['parameter'] = ['A','B','C','D','xs_o','ys_o','xt_o','yt_o',
                                   'ratio','rot_deg','on_skew','off_skew','zp',
                                   'ra_cen','dec_cen','x_cen','y_cen','pixel_scale','orientat']
        trans_out['value'] = [A, B, C, D, xs_o, ys_o, xt_o, yt_o,
                               ratio, rot, on_skew, off_skew, zp,
                               params['ra_cen'], params['dec_cen'],
                               params['x_cen'], params['y_cen'],
                               params['pixel_scale'], params['orientat']]
        trans_out.write(os.path.join(img['root'], "transformation.csv"), format='ascii.csv', overwrite=True)
        print(f"Finished {image_name}: Found {len(final_matches)} matches in {time.time()-start_time:.2f}s.", file=original_stdout)

    except Exception as e:
        print(f"Finished {image_name}: Error - {e}", file=original_stdout)
    finally:
        sys.stdout = original_stdout

def main():
    parser = argparse.ArgumentParser(description="Parallel Gaia-HST catalog cross-matcher with covariance weighting and magnitude rejection.")
    parser.add_argument("--target", required=True,
                        help="Target name (e.g. Fornax_dSph). Expects data in [data_dir]/[target]/Gaia and [data_dir]/[target]/HST")
    parser.add_argument("--data-dir", default="./data",
                        help="Root directory containing target data folders. Default: ./data")
    parser.add_argument("--threads", type=int, default=os.cpu_count(),
                        help="Number of parallel processing threads. Default: All available cores")
    parser.add_argument("--hst-pix-floor", type=float, default=0.01,
                        help="Minimum HST positional uncertainty (pixels) added in quadrature to reported errors. Default: 0.01")
    parser.add_argument("--min-matches", type=int, default=3,
                        help="Minimum number of seeds required for initial match. Default: 3")
    parser.add_argument("--zero-gaia-pm", action="store_true",
                        help="Set all Gaia PMs and Parallaxes to 0 with large default uncertainties. Useful for debugging.")
    parser.add_argument("--scale-sweep", action="store_true",
                        help="Enable simultaneous scale+offset sweep during 4P discovery (slower but more robust when pixel scale is uncertain).")
    parser.add_argument("--image", type=str, default=None,
                        help="Process only this image (by observation ID). Useful for debugging.")

    args = parser.parse_args()
    gaia_df = load_gaia_data(args.target, args.data_dir)
    if gaia_df is None: return

    img_folders = find_hst_image_folders(args.target, args.data_dir)
    if args.image:
        img_folders = [i for i in img_folders if i['root'].split('/')[-1] == args.image]
        if not img_folders:
            print(f"No image folder found for '{args.image}'"); return
    print(f"Found {len(img_folders)} images. Processing with {args.threads} threads...")

    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(process_single_image, img, gaia_df, args.hst_pix_floor, args.min_matches, args.zero_gaia_pm, scale_sweep=args.scale_sweep): img for img in img_folders}
        for f in as_completed(futures):
            f.result()
    print("All tasks completed.")

    from .validator import validate_target
    print("\n--- Running cross-image validation ---")
    validate_target(args.target, args.data_dir)

if __name__ == "__main__": main()
