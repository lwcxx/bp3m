"""
BP3M Linear Algebra Solver.

Implements the closed-form posterior distributions from Equations 9-11
of McKinnon et al. (in prep), using the Schur complement / information-form
marginalization for efficiency at scale.

Coordinate conventions:
  - HST positions: x_c = X - Xo, y_c = Y - Yo  (centered on image pivot)
  - Gaia pseudo-image: xs = plane_project(ra, dec, ra0, dec0, pscale)
    which gives ~(X_G - Wo) in the GaiaHub frame
  - Both are in units of HST pixels (same pixel scale)

The transformation model:
  x_survey_i,j = X_i,j @ r_j - JU_i,j @ v_T,i   (Eq. 8)

where:
  r_j = (a, b, c, d, w, z, Δα0, Δδ0)^T    image transformation (8-dim)
  v_T,i = (Δα*, Δδ, μα*, μδ, ϖ)^T         astrometry update (5-dim)
  X_i,j uses centered HST positions (x_c, y_c)
  JU_i,j = J_i,j @ U_i,j (Jacobian x time-evolution)

The iterative algorithm:
  1. Init R_j from image header rotation/scale
  2. Compute C_s,i,j = R_j @ C_hst_i,j @ R_j^T
  3. Solve for r_hat, C_r (Schur complement of joint Gaussian)
  4. Solve for v_hat_i, C_vT,i (conditional on r_hat)
  5. Update R_j from new (a,b,c,d) in r_hat; repeat from 2
"""
from __future__ import annotations # to account for the Python version
import numpy as np
from scipy import linalg
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord

from .astro_utils import (
    plane_project, plane_project_jacobian, plane_project_tangent_derivs,
    get_parallax_factors, build_U_matrix, build_X_matrix,
    hst_position_cov, rotation_matrix_from_abcd, gaia_cov_to_survey_cov,
    abcd_from_rotation_pixscale_skew, n_r_from_poly_order, compute_poly_jacobian,
    get_tele_position, michalik_sigma_plx_prior, RAD2MAS, DEG2RAD, GAIA_SYS_DICT
)

N_R = 8     # r_j dimensions for poly_order=1 (backward-compat constant)
N_V = 5     # v_T,i dimensions: (Δα*, Δδ, μα*, μδ, ϖ)

# ── Global astrometry prior ────────────────────────────────────────────────────
# Gaia 5p/6p sources: NO diffuse prior — their Gaia covariance is the sole prior.
# Gaia 2p sources: diffuse PM prior (100 mas/yr) + Michalik et al. (2015)
#   magnitude/direction-dependent parallax prior (10 * sigma_F90).
# HST-only sources (future v2 loader): same as 2p.
# Per-star precision arrays are built in _load_star_data as self._C_VG_inv_per_star.
_SIGMA_POS = 1e6   # effectively flat prior on Δα*, Δδ  (all sources)
_SIGMA_PM  = 100.0  # mas/yr  (2p / HST-only only)

# Image transformation prior uncertainties (1-sigma).
# Empirically calibrated on Leo I, Fornax, and NGC 55 (110 half-images):
#   Δrot rms ≈ 0.015°  p99 ≈ 0.063°  → 0.1° gives 1.6× margin on p99
#   Δscale rms ≈ 0.5%  p99 ≈ 0.6%   → 1.5% gives 2.5× margin on p99
#   skew rms ≈ 0.05–0.1%            → 0.5% gives 5× margin on p99
# These are ~5–30× tighter than the previous values, which improves EM
# convergence and reduces seed-specific bias in marginalised posteriors.
_SIGMA_ROT_DEG  = 0.1    # degrees
_SIGMA_SCALE    = 1.5e-2  # fractional pixel scale ratio
_SIGMA_SKEW     = 5e-3   # on- and off-axis skew terms
_SIGMA_POINTING = 1e-6  # RA0,Dec0 (ARCSEC) — effectively fixed at 0; tangent-point
                        # update is disabled in _update_geometry (w/z degeneracy), so
                        # these must be pinned or the [w,Δα0] block is 3e8-ill-conditioned
_SIGMA_CENTER = 2048   # WZ (global pixels)

# Initial residual filter applied in _precompute_geometry.
# Stars whose corrected 2D residual (after removing bulk w,z offset)
# exceeds this threshold are excluded from use_for_fit before the first
# solve pass.  Fornax real matches: p99 ≈ 37 px.  Leo_I false matches:
# p50 ≈ 200 px.  100 px is a clean cut with comfortable margin on both sides.
_INIT_RESID_CLIP_PX = 100.0



def _make_image_prior(meta, poly_order=1):
    """
    Return (r_prior_j, C_r_prior_inv_j) for image j.

    r_j = (a, b, c, d, w, z, Δα0, Δδ0 [, poly terms...])
    Prior:
      (a,b,c,d) — from header rotation/scale (strong prior)
      (w, z)    — uninformative (flat); mean initialised to median residual later
      (Δα0,Δδ0) — sigma = _SIGMA_POINTING arcsec
      poly terms — zero mean, flat prior (determined entirely by data)
    """
    n_r = n_r_from_poly_order(poly_order)

    rot_rad = meta["orig_rot_deg"] * DEG2RAD
    s = 1.0
    a = np.cos(rot_rad)
    b = np.sin(rot_rad)
    c = -np.sin(rot_rad)
    d = np.cos(rot_rad)

    r_prior = np.zeros(n_r)
    r_prior[:4] = [a, b, c, d]
    # r_prior[4:] = 0  (w, z, Δα0, Δδ0 and all poly terms start at zero)

    # Jacobian ∂(a,b,c,d)/∂(rot_rad, scale_ratio, on_skew, off_skew)
    cr, sr = np.cos(rot_rad), np.sin(rot_rad)
    J = np.array([
        [-s*sr,  cr,  1,  0],
        [ s*cr,  sr,  0,  1],
        [-s*cr, -sr,  0,  1],
        [-s*sr,  cr, -1,  0],
    ])
    sigma = np.array([_SIGMA_ROT_DEG * DEG2RAD, _SIGMA_SCALE, _SIGMA_SKEW, _SIGMA_SKEW])
    C_abcd = J @ np.diag(sigma**2) @ J.T   # (4, 4)

    # Full n_r × n_r prior precision matrix
    C_r_prior_inv = np.zeros((n_r, n_r))
    try:
        C_r_prior_inv[:4, :4] = np.linalg.inv(C_abcd)
    except np.linalg.LinAlgError:
        C_r_prior_inv[:4, :4] = np.diag(1.0 / np.diag(C_abcd + 1e-30 * np.eye(4)))

    C_r_prior_inv[4, 4] = _SIGMA_CENTER ** -2
    C_r_prior_inv[5, 5] = _SIGMA_CENTER ** -2
    C_r_prior_inv[6, 6] = _SIGMA_POINTING ** -2
    C_r_prior_inv[7, 7] = _SIGMA_POINTING ** -2

    # Indices 4, 5 (w, z) and 8+ (poly terms) remain zero — flat prior.

    return r_prior, C_r_prior_inv


class BP3MSolver:
    """
    Simultaneous HST-Gaia astrometric alignment and stellar PM/parallax update.

    Parameters
    ----------
    images : dict  {image_name: dict of metadata}
    stars_per_image : dict  {image_name: pd.DataFrame}
    gaia_catalog : pd.DataFrame  (one row per unique Gaia source)
    star_id_to_idx : dict  {Gaia_id: int index}
    image_names : list[str]
    star_in_image : dict  (unused; we rebuild this from data)
    """

    def __init__(self, images, stars_per_image, gaia_catalog,
                 star_id_to_idx, image_names, star_in_image,
                 poly_order=1):
        """
        Parameters
        ----------
        poly_order : int, optional
            Polynomial order for the image transformation.
              1 → linear (a,b,c,d,w,z,Δα0,Δδ0) — 8 parameters per image (default)
              2 → adds degree-2 terms — 14 parameters per image
              3 → adds degree-2 and degree-3 terms — 22 parameters per image
            N_R(p) = 2 + (p+1)*(p+2)
        """
        if poly_order < 1:
            raise ValueError(f"poly_order must be ≥ 1, got {poly_order}")
        self.poly_order = poly_order
        self.N_R = n_r_from_poly_order(poly_order)

        self.images = images
        self.stars_per_image = stars_per_image
        self.gaia_cat = gaia_catalog.reset_index(drop=True)
        self.star_id_to_idx = star_id_to_idx
        self.image_names = list(image_names)
        self.n_stars = len(gaia_catalog)
        self.n_images = len(image_names)

        self._cache_gaia()
        self._precompute_geometry()
        self._init_transforms()

    # ── Gaia data caching ──────────────────────────────────────────────────────

    def _cache_gaia(self):
        g = self.gaia_cat
        ruwe = g["ruwe"].to_numpy(float)
        # NaN RUWE = 2-param stars (no 5D Gaia solution); treat as trustworthy
        # since NaN means "not applicable", not "poor astrometry".
        self.gaia_trustworthy = np.isnan(ruwe) | (ruwe <= 1.4)
        self.gaia_g  = g["gmag"].to_numpy(float)
        self.gaia_n_hst_used = np.zeros(len(self.gaia_g)).astype(int)
        self.gaia_ra  = g["ra"].to_numpy(float)
        self.gaia_dec = g["dec"].to_numpy(float)
        self.gaia_time  = Time(g["Gaia_time"].fillna(2016.0).to_numpy(float),format='jyear',scale='tcb')
        self.gaia_yr  = self.gaia_time.jyear
        self.sigma_from_gaia_prior  = np.zeros(len(self.gaia_g)).astype(float)

        # Survey astrometry vector: v_s,i = (0, 0, pmra, pmdec, parallax)
        # (Δα*, Δδ) = 0 since Gaia position IS the reference; updates captured in v_T,i
        self.gaia_6p = np.isfinite(g['pseudocolour'])
        self.gaia_5p = np.isfinite(g['pmra']) & ~self.gaia_6p
        # 6p sources have full 5D Gaia astrometry (pmra/pmdec/parallax) like 5p;
        # they are NOT 2p just because they also have a pseudocolour measurement.
        self.gaia_2p = np.isfinite(self.gaia_ra) & ~self.gaia_5p & ~self.gaia_6p
        self.full_gaia_astrometry = np.isfinite(g['pmra']) & np.isfinite(g['pmdec']) & np.isfinite(g['parallax'])

        self.v_survey = np.zeros((self.n_stars, N_V))
        for col, idx in [("pmra", 2), ("pmdec", 3), ("parallax", 4)]:
            if col in g.columns:
                self.v_survey[:, idx] = g[col].fillna(0.0).to_numpy(float)

        # Build 5×5 Gaia covariance matrices (units: mas, mas/yr)
        def _get(col, default=0.0):
            return g[col].fillna(default).to_numpy(float) if col in g.columns else \
                   np.full(self.n_stars, default)

        ra_e   = _get("ra_error",    1e6)   # mas
        dec_e  = _get("dec_error",   1e6)
        pmra_e = _get("pmra_error",  1e3)
        pmdec_e= _get("pmdec_error", 1e3)
        plx_e  = _get("parallax_error", 1e3)

        # Correlations
        corr_ra_dec = _get("ra_dec_corr")
        corr_ra_plx = _get("ra_parallax_corr")
        corr_ra_pmra= _get("ra_pmra_corr")
        corr_ra_pmdec= _get("ra_pmdec_corr")
        corr_dec_plx= _get("dec_parallax_corr")
        corr_dec_pmra= _get("dec_pmra_corr")
        corr_dec_pmdec= _get("dec_pmdec_corr")
        corr_plx_pmra= _get("parallax_pmra_corr")
        corr_plx_pmdec= _get("parallax_pmdec_corr")
        corr_pmra_pmdec= _get("pmra_pmdec_corr")

        # Build (n_stars, 5, 5) covariance matrices
        sigmas = np.stack([ra_e, dec_e, pmra_e, pmdec_e, plx_e], axis=1)  # (n, 5)

        corr_mat = np.zeros((self.n_stars, N_V, N_V))
        for i in range(N_V):
            corr_mat[:, i, i] = 1.0
        # Fill off-diagonals (order: Δα*, Δδ, μα*, μδ, ϖ → indices 0,1,2,3,4)
        pairs = [
            (0,1,corr_ra_dec), (0,2,corr_ra_pmra), (0,3,corr_ra_pmdec),
            (0,4,corr_ra_plx), (1,2,corr_dec_pmra), (1,3,corr_dec_pmdec),
            (1,4,corr_dec_plx),(2,3,corr_pmra_pmdec),(2,4,corr_plx_pmra),
            (3,4,corr_plx_pmdec),
        ]
        for i, j, arr in pairs:
            corr_mat[:, i, j] = arr
            corr_mat[:, j, i] = arr

        # C_survey = diag(sigma) @ corr @ diag(sigma)
        self.C_survey = sigmas[:, :, None] * corr_mat * sigmas[:, None, :]

        #account for systematics in Gaia data
        #amount to inflate uncertainties by
        #might want to change to function of magnitude in the future
        self.C_survey[self.gaia_6p] *= GAIA_SYS_DICT['mult_6p']
        self.C_survey[self.gaia_5p] *= GAIA_SYS_DICT['mult_5p']
        self.C_survey[self.gaia_2p] *= GAIA_SYS_DICT['mult_2p']
        self.C_survey += np.diag(np.array([0,0,
                            GAIA_SYS_DICT['pm_sys_err'],GAIA_SYS_DICT['pm_sys_err'],
                            GAIA_SYS_DICT['parallax_sys_err']])**2)

        # Flag stars with full Gaia astrometry
        self.has_full_astro = (
            (g["pmra_error"].notna() if "pmra_error" in g.columns else
             np.zeros(self.n_stars, bool)) &
            (g["parallax_error"].notna() if "parallax_error" in g.columns else
             np.zeros(self.n_stars, bool))
        ).to_numpy(bool)

        # Invert C_survey
        self.C_survey_inv = np.zeros_like(self.C_survey)
        self.C_survey_inv[self.full_gaia_astrometry] = np.linalg.inv(self.C_survey[self.full_gaia_astrometry])
        self.C_survey_inv[~self.full_gaia_astrometry,:2,:2] = np.linalg.inv(self.C_survey[~self.full_gaia_astrometry,:2,:2])

        self.C_survey_inv_dot_v = np.einsum('nij,nj->ni',self.C_survey_inv,self.v_survey)

        # for i in range(self.n_stars):
        #     try:
        #         self.C_survey_inv[i] = np.linalg.inv(self.C_survey[i])
        #     except np.linalg.LinAlgError:
        #         self.C_survey_inv[i] = np.diag(1.0 / np.diag(self.C_survey[i] + 1e-30))

        # ── Per-star astrometry prior (Michalik et al. 2015) ──────────────────
        # Gaia 5p/6p: zero precision (no diffuse prior — Gaia covariance suffices).
        # Gaia 2p / HST-only: flat position prior + 100 mas/yr PM prior +
        #   magnitude/direction-dependent parallax prior (10 * sigma_F90).
        needs_diffuse = self.gaia_2p  # 2p; HST-only rows added by v2 loader extend this
        sigma_plx_prior = np.full(self.n_stars, np.inf)
        if needs_diffuse.any():
            sigma_plx_prior[needs_diffuse] = michalik_sigma_plx_prior(
                self.gaia_ra[needs_diffuse],
                self.gaia_dec[needs_diffuse],
                self.gaia_g[needs_diffuse],
            )

        # _C_VG_inv_per_star : (n_stars, 5) diagonal precision additions.
        # param order: (Δα*, Δδ, μα*, μδ, ϖ)
        self._C_VG_inv_per_star = np.zeros((self.n_stars, N_V), dtype=float)
        self._C_VG_inv_per_star[needs_diffuse, 0] = _SIGMA_POS**-2
        self._C_VG_inv_per_star[needs_diffuse, 1] = _SIGMA_POS**-2
        self._C_VG_inv_per_star[needs_diffuse, 2] = _SIGMA_PM**-2
        self._C_VG_inv_per_star[needs_diffuse, 3] = _SIGMA_PM**-2
        finite_plx = needs_diffuse & np.isfinite(sigma_plx_prior)
        self._C_VG_inv_per_star[finite_plx, 4] = sigma_plx_prior[finite_plx]**-2

        # _sigma_diff_per_star : (n_stars, 5) used in the diffuse-prior chi2
        # outlier test.  5p/6p get very large sigmas so the test never triggers.
        self._sigma_diff_per_star = np.full((self.n_stars, N_V), 1e9, dtype=float)
        self._sigma_diff_per_star[needs_diffuse, 0] = 1e4
        self._sigma_diff_per_star[needs_diffuse, 1] = 1e4
        self._sigma_diff_per_star[needs_diffuse, 2] = _SIGMA_PM
        self._sigma_diff_per_star[needs_diffuse, 3] = _SIGMA_PM
        self._sigma_diff_per_star[finite_plx,    4] = sigma_plx_prior[finite_plx]

    # ── Geometry precomputation ────────────────────────────────────────────────

    def _precompute_geometry(self):
        """
        For each image, precompute all star-level geometric quantities that
        don't depend on the current estimate of R_j (i.e., everything except C_s,i,j).
        """
        print("Precomputing geometry...")
        self._img_data = {}

        self.gaia_n_hst_used[:] = 0

        for img in self.image_names:
            meta  = self.images[img]
            df    = self.stars_per_image[img]
            mask  = df["Gaia_id"].isin(self.star_id_to_idx)
            df    = df[mask].copy().reset_index(drop=True)
            if len(df) == 0:
                self._img_data[img] = None
                continue

            n = len(df)
            sidx = np.array([self.star_id_to_idx[gid] for gid in df["Gaia_id"]])

            ra0, dec0 = meta["ra0"], meta["dec0"]
            # pscale    = meta["pixel_scale"]   # mas/pixel
            pscale    = meta["orig_pixel_scale"]   # mas/pixel
            # Xo, Yo    = meta["Xo"], meta["Yo"]
            Xo, Yo    = 2048.0, 2048.0
            meta['Xo'] = Xo
            meta['Yo'] = Yo
            hst_time  = Time(meta["hst_time_mjd"],format='mjd')
            hst_mjd   = hst_time.mjd
            hst_yr    = hst_time.jyear

            # Gaia positions for these stars
            ra_g  = self.gaia_ra[sidx]
            dec_g = self.gaia_dec[sidx]
            t_g= self.gaia_time[sidx]
            dt_yr = (hst_time - t_g).to(u.year).value   # time offset: negative = Gaia is after HST

            # Gaia pseudo-image positions (x_data): plane project relative to image center
            xs, ys = plane_project(ra_g, dec_g, ra0, dec0, pscale)  # (n,) pixels

            # Jacobian J_i,j: (n, 2, 2) in pix/mas
            J = plane_project_jacobian(ra_g, dec_g, ra0, dec0, pscale)

            # Tangent-point derivatives for Δα0, Δδ0 columns of X_mat
            #change to Δα0, Δδ0 to be in ARCSEC (not mas) for numerical stability
            #by using pscale/1000
            dxs_dra0, dxs_ddec0, dys_dra0, dys_ddec0 = plane_project_tangent_derivs(
                ra_g, dec_g, ra0, dec0, pscale/1000)
            # dxs_dra0, dxs_ddec0, dys_dra0, dys_ddec0 = plane_project_tangent_derivs(
            #     ra_g, dec_g, ra0, dec0, pscale)

            # Parallax factors: difference between HST epoch and Gaia epoch
            #Gaia has already removed the parallax, so no need to subtract plx at J2016
            tele_xyz = get_tele_position(hst_time,curr_id='earth')
            meta['tele_XYZ'] = tele_xyz
            d_plx_ra,  d_plx_dec  = get_parallax_factors(ra_g, dec_g, tele_xyz)

            # U matrix for each star: (n, 2, 5)
            U_arr = np.zeros((n, 2, N_V))
            for k in range(n):
                U_arr[k] = build_U_matrix(dt_yr[k], d_plx_ra[k], d_plx_dec[k])

            # JU = J @ U: (n, 2, 5)
            JU = np.einsum('nij,njk->nik', J, U_arr)

            # Centered HST pixel positions: x_c = X - Xo, y_c = Y - Yo
            X_c = df["X"].to_numpy(float) - Xo
            Y_c = df["Y"].to_numpy(float) - Yo
            good_for_fitting = df['use_for_alignment'].to_numpy(bool).copy()
            # Phase-6 outlier flag: Gaia detections that start inactive but are
            # real matches that can be re-enabled by the EM loop if residuals
            # improve.  Separate from use_for_align_init so they don't inflate
            # the adaptive threshold (which uses init_trusted = use_for_align_init).
            _phase6_outlier = (
                (~df['use_for_alignment'].to_numpy(bool))
                & (df['use_for_align_init_flag'].to_numpy(bool))
                if 'use_for_align_init_flag' in df.columns
                else np.zeros(len(df), dtype=bool)
            )

            # #try removing saturated stars for first iteration
            q_hst_ok = df['q_hst'].to_numpy() > 0
            if np.sum(good_for_fitting & q_hst_ok) > 0:
                good_for_fitting &= q_hst_ok
            if np.sum(good_for_fitting & self.gaia_trustworthy[sidx]) > 0:
                good_for_fitting &= self.gaia_trustworthy[sidx]

            # X matrix: (n, 2, N_R)
            X_mat = np.zeros((n, 2, self.N_R))
            for k in range(n):
                X_mat[k] = build_X_matrix(
                    X_c[k], Y_c[k],
                    dxs_dra0[k], dxs_ddec0[k],
                    dys_dra0[k], dys_ddec0[k],
                    poly_order=self.poly_order)

            # HST position covariance C_hst: (n, 2, 2)
            x_err  = df["x_hst_err"].to_numpy(float)
            y_err  = df["y_hst_err"].to_numpy(float)
            xy_cor = df["xy_hst_corr"].fillna(0.).to_numpy(float)
            C_hst = np.zeros((n, 2, 2))
            for k in range(n):
                C_hst[k] = hst_position_cov(x_err[k], y_err[k], xy_cor[k])

            r_prior, C_r_prior_inv = _make_image_prior(meta, poly_order=self.poly_order)

            # ── Build r_init (initial iterate) ───────────────────────────────
            # When transformation.csv provides (a,b,c,d,w,z) from fast_cross_match,
            # use those as the starting point.  The prior (r_prior, C_r_prior_inv)
            # is computed solely from the WCS header and is never modified here.
            # r_init is a copy: changing it never changes the prior.
            fcm_abcdwz = meta.get("fcm_abcdwz")
            r_init = r_prior.copy()
            if fcm_abcdwz is not None:
                r_init[:6] = fcm_abcdwz   # a, b, c, d, w, z from cross-match

            # ── Initial residual screening ────────────────────────────────────
            # Used to permanently block implausible cross-matches (> 100 px after
            # accounting for the bulk offset).  r_init provides a better prediction
            # than r_prior because a,b,c,d,w,z are already well-constrained, so
            # the per-star residuals are much smaller and the screening is cleaner.
            # Use only PM and parallax (cols 2-4): position offset Δα,Δδ = 0 at
            # reference epoch. v_survey[:, 0:2] stores absolute ra/dec in degrees,
            # which would produce spuriously large offsets for non-Gaia stars.
            _v_pm_plx = np.zeros_like(self.v_survey[sidx])
            _v_pm_plx[:, 2:] = self.v_survey[sidx, 2:]
            ave_motion_offset = np.einsum('nij,nj->ni', JU, _v_pm_plx)
            xys = np.stack([xs, ys], axis=1)
            x_pred_init = np.einsum('nkl,l->nk', X_mat, r_init) - ave_motion_offset
            x_resid_init = xys - x_pred_init

            if fcm_abcdwz is not None:
                # w,z already encoded in r_init — residuals are centred near zero.
                # No bulk-offset subtraction needed for the ok_init screen.
                resid_mag = np.hypot(x_resid_init[:, 0], x_resid_init[:, 1])
            else:
                # r_init = r_prior (w=z=0): subtract median bulk offset as before.
                med_wz_screen = (np.nanmedian(x_resid_init[good_for_fitting], axis=0)
                                 if good_for_fitting.any() else np.zeros(2))
                x_resid_corr = x_resid_init - med_wz_screen
                resid_mag = np.hypot(x_resid_corr[:, 0], x_resid_corr[:, 1])

            ok_init = resid_mag <= _INIT_RESID_CLIP_PX  # (n,) hard ceiling mask

            if good_for_fitting.any():
                n_before = int(good_for_fitting.sum())
                good_for_fitting = good_for_fitting & ok_init
                n_rej = n_before - int(good_for_fitting.sum())
                if n_rej > 0:
                    print(f"    {img}: rejected {n_rej}/{n_before} stars with "
                          f"initial residual > {_INIT_RESID_CLIP_PX:.0f} px")

            # ── Set prior mean from cross-match solution ──────────────────────
            # When transformation.csv provides (a,b,c,d,w,z), override the
            # WCS-only prior mean for all 6 parameters.  This ensures that when
            # no data stars contribute (good_for_fitting all-False), the solve
            # returns r_hat = r_prior = cross-match solution rather than the
            # WCS-only estimate, so Phase-0 residuals are computed at the correct
            # transformation and stars can be re-admitted.  Falls back to residual
            # median for w,z (or zero) only when no cross-match solution exists.
            if fcm_abcdwz is not None:
                r_prior[:6] = fcm_abcdwz[:6]
            elif good_for_fitting.any():
                r_prior[[4, 5]] = np.nanmedian(x_resid_init[good_for_fitting], axis=0)

            self.gaia_n_hst_used[sidx[good_for_fitting]] += 1

            self._img_data[img] = {
                "sidx"           : sidx,              # (n,) global star indices
                "n"              : n,
                "xys"            : xys,               # (n, 2) Gaia pseudo-image xy
                "JU"             : JU,                # (n, 2, 5)
                "X_mat"          : X_mat,             # (n, 2, N_R)
                "C_hst"          : C_hst,             # (n, 2, 2) — may be inflated
                "C_hst_orig"     : C_hst.copy(),      # (n, 2, 2) — original, never modified
                "X_c"            : X_c,               # (n,) centered HST x (needed for poly Jacobian)
                "Y_c"            : Y_c,               # (n,) centered HST y
                "r_prior"        : r_prior,           # (N_R,) prior mean — from WCS header only
                "r_init"         : r_init,            # (N_R,) initial iterate — from transformation.csv when available
                "C_r_prior_inv"  : C_r_prior_inv,     # (N_R, N_R)
                "use_for_fit"    : good_for_fitting,  # (n,) boolean — used for alignment (and astrometry)
                "use_for_astrom" : good_for_fitting.copy(),  # (n,) boolean — used for astrometry only
                "use_for_fit_max": ok_init.copy(),    # hard ceiling: only blocks 100px+ outliers
                # Frozen snapshot of initially-trusted stars (use_for_alignment=True,
                # q_hst>0, gaia_trustworthy, initial residual ≤ 100px).  Used as the
                # reference population for the test-3 adaptive threshold so that
                # sources initially excluded (e.g. non-stars with inflated PSF
                # covariances) cannot bias the threshold even if they are later
                # re-evaluated by the EM loop.
                "use_for_align_init": good_for_fitting.copy(),  # False for Phase-6 outliers (don't inflate threshold)
                "phase6_outlier":     _phase6_outlier.copy(),   # True for Phase-6 Gaia outliers (can re-enter)
            }

        n_total = sum(d["n"] for d in self._img_data.values() if d)
        print(f"  Done: {n_total} star-image pairs across {self.n_images} images.")

        # Diagnostic: track 5p/6p/2p stars through the three admission gates.
        # Gate 1: star appears in at least one image's sidx (has an HST match).
        # Gate 2: star has use_for_fit=True in at least one image after the
        #         ruwe/q_hst/ok_init filters (gaia_n_hst_used > 0).
        # Gate 3: (future) EM loop re-admission.
        in_images = np.zeros(self.n_stars, bool)
        for d in self._img_data.values():
            if d is not None:
                in_images[d["sidx"]] = True
        admitted = self.gaia_n_hst_used > 0

        pop_labels = [('5p', self.gaia_5p), ('6p', self.gaia_6p),
                      ('2p', self.gaia_2p)]
        rows = []
        for label, mask in pop_labels:
            n_cat   = int(mask.sum())
            n_img   = int((mask & in_images).sum())
            n_admit = int((mask & admitted).sum())
            n_excl  = n_img - n_admit
            rows.append((label, n_cat, n_img, n_admit, n_excl))
        hdr = f"  {'pop':>3}  {'catalog':>7}  {'in_images':>9}  {'admitted':>8}  {'excluded':>8}"
        print(hdr)
        for label, n_cat, n_img, n_admit, n_excl in rows:
            excl_str = f"  ← {n_excl} lost to ruwe/q_hst/resid filter" if n_excl > 0 else ""
            print(f"  {label:>3}  {n_cat:>7}  {n_img:>9}  {n_admit:>8}  {n_excl:>8}{excl_str}")

    def _init_transforms(self):
        """Initialise R_j (2×2 rotation matrix) from header info."""
        self.R = {}
        for img in self.image_names:
            meta = self.images[img]
            # a, b, c, d = abcd_from_rotation_pixscale_skew(
            #     meta["rotation_deg"], meta["pixel_scale_ratio"],
            #     meta["on_skew"], meta["off_skew"])
            # a, b, c, d = meta['AG'], meta['BG'], meta['CG'], meta['DG']
            rot_rad = meta["orig_rot_deg"] * DEG2RAD
            s = 1.0
            a = np.cos(rot_rad)
            b = np.sin(rot_rad)
            c = -np.sin(rot_rad)
            d = np.cos(rot_rad)

            self.R[img] = rotation_matrix_from_abcd(a, b, c, d)

    # ── Geometry update (called every fit iteration) ──────────────────────────

    def _update_geometry(self, r_hat, v_hat):
        """
        Recompute per-image geometry (xys, JU, X_mat, X_c, Y_c) using the
        current best estimates of stellar positions and the tangent-point shifts.

        After convergence of a few iterations, the updated stellar positions
        (from v_hat) and the tangent-point corrections (Δα0, Δδ0 in r_hat)
        can shift xys, the Jacobians J, and the parallax factors enough to
        matter.  This method recomputes all position-dependent quantities in
        _img_data except C_hst (which depends only on the HST measurement and
        the transformation Jacobian, the latter handled by _compute_Cs).

        Parameters
        ----------
        r_hat : (n_r,)        current image transformation vector
        v_hat : (n_stars, 5)  current stellar astrometry estimate
                               v_hat[:,0:2] = (Δα*, Δδ) offsets from Gaia [mas]
        """
        import astropy.units as u
        from astropy.time import Time

        self.gaia_n_hst_used[:] = 0
        nr = self.N_R
        for j_idx, img in enumerate(self.image_names):
            d = self._img_data.get(img)
            if d is None:
                continue

            meta    = self.images[img]
            r_j     = r_hat[j_idx * nr:(j_idx + 1) * nr]
            sidx    = d["sidx"]
            use_align  = d["use_for_fit"]
            use_astrom = d.get("use_for_astrom", use_align)
            n       = d["n"]

            self.gaia_n_hst_used[sidx[use_align | use_astrom]] += 1

            # ── Updated tangent point (ra0 + Δα0, dec0 + Δδ0) ──────────────
            # Δα0, Δδ0 are in ARCSEC (pscale/1000 scaling used in X_mat).
            # r_j[6] = Δα0 arcsec, r_j[7] = Δδ0 arcsec.
            # ra0_up  = meta["ra0"]  + r_j[6] / 3600.0   # degrees
            # dec0_up = meta["dec0"] + r_j[7] / 3600.0   # degrees

            #it is currently unstable to update RA0,Dec0 (because of correlation with WZ)
            #so don't update for now. Maybe future versions will have better priors on WZ
            #and RA0,Dec0 (including correlations between images, e.g., where we have a 
            #good estimate of their offsets from each other)
            ra0_up  = meta["ra0"]   # degrees
            dec0_up = meta["dec0"]  # degrees

            pscale   = meta["orig_pixel_scale"]
            hst_time = Time(meta["hst_time_mjd"], format="mjd")

            # ── Updated stellar RA/Dec from v_hat[:,0:2] ─────────────────────
            # v_hat[:,0] = Δα* [mas], v_hat[:,1] = Δδ [mas]
            ra_g_orig  = self.gaia_ra[sidx]
            dec_g_orig = self.gaia_dec[sidx]

            # Δα* = Δα · cos(δ), so Δα = Δα* / cos(δ)
            cos_dec = np.cos(dec_g_orig * DEG2RAD)
            ra_g_up  = ra_g_orig  + v_hat[sidx, 0] / (cos_dec * RAD2MAS)  # degrees
            dec_g_up = dec_g_orig + v_hat[sidx, 1] / RAD2MAS              # degrees

            t_g   = self.gaia_time[sidx]
            dt_yr = (hst_time - t_g).to(u.year).value

            # ── Recompute projected Gaia positions ────────────────────────────
            #DO NOT USE THE UPDATED GAIA COORDINATES HERE! Just the RA0,Dec0 updates
            xs, ys = plane_project(ra_g_orig, dec_g_orig, ra0_up, dec0_up, pscale)
            xys    = np.stack([xs, ys], axis=1)

            # ── Recompute Jacobian J and tangent-point derivatives ────────────
            J = plane_project_jacobian(ra_g_orig, dec_g_orig, ra0_up, dec0_up, pscale)
            dxs_dra0, dxs_ddec0, dys_dra0, dys_ddec0 = \
                plane_project_tangent_derivs(ra_g_orig, dec_g_orig, ra0_up, dec0_up,
                                             pscale / 1000)

            # ── Recompute parallax factors ────────────────────────────────────
            #DO use the new best fit RA,Dec positions here
            tele_xyz = meta['tele_XYZ'] 
            d_plx_ra, d_plx_dec = get_parallax_factors(ra_g_up, dec_g_up, tele_xyz)

            # ── Rebuild U and JU ─────────────────────────────────────────────
            U_arr = np.zeros((n, 2, N_V))
            for k in range(n):
                U_arr[k] = build_U_matrix(dt_yr[k], d_plx_ra[k], d_plx_dec[k])
            JU = np.einsum('nij,njk->nik', J, U_arr)

            # ── Rebuild X_mat ────────────────────────────────────────────────
            X_c = d["X_c"]   # unchanged — detector positions don't move
            Y_c = d["Y_c"]
            X_mat = np.zeros((n, 2, self.N_R))
            for k in range(n):
                X_mat[k] = build_X_matrix(
                    X_c[k], Y_c[k],
                    dxs_dra0[k], dxs_ddec0[k],
                    dys_dra0[k], dys_ddec0[k],
                    poly_order=self.poly_order)

            d["xys"]  = xys
            d["JU"]   = JU
            d["X_mat"] = X_mat

    # ── Core solver ────────────────────────────────────────────────────────────

    def _compute_Cs(self, img, r_j=None):
        """
        Transformed HST covariance: C_s,k = J_k @ C_hst,k @ J_k^T.

        For poly_order=1, J_k = R_j = [[a,b],[c,d]] (constant across stars).
        For poly_order>1, J_k is the full position-dependent Jacobian of the
        transformation evaluated at each star's (X_c, Y_c) position.

        Parameters
        ----------
        img  : str   image name
        r_j  : (N_R,) array or None
            Current r_j for this image. Required for poly_order > 1.
            If None (or poly_order==1), falls back to the cached R matrix.

        Returns
        -------
        C_s : (n, 2, 2) ndarray
        """
        d = self._img_data[img]
        C_hst = d["C_hst"]   # (n, 2, 2)

        if self.poly_order == 1 or r_j is None:
            R = self.R[img]
            return R @ C_hst @ R.T   # broadcasts over n

        # Higher-order: per-star Jacobian
        J = compute_poly_jacobian(r_j, d["X_c"], d["Y_c"], self.poly_order)
        # J: (n, 2, 2),  C_hst: (n, 2, 2)
        return np.einsum('nij,njk,nlk->nil', J, C_hst, J)

    def _solve_one_pass(self, r_current, z_weights=None):
        """
        Single pass of the linear solver, working in RESIDUAL coordinates to
        avoid catastrophic cancellation.

        We solve for Δr = r - r_current and v_T,i given the residuals
            x_resid_{i,j} = x_data_{i,j} - X_{i,j} @ r_j_current

        which are small (~few pixels) even though absolute coordinates are large
        (~2000 pixels).

        Parameters
        ----------
        z_weights : dict {img: (n,) float} or None
            When provided, soft-weight IRLS mode.  Each entry replaces the
            hard use_for_fit/use_for_astrom flags with use_for_fit_max (Phase-0
            hard floor), and scales Cs_inv by z for each detection.

        Returns
        -------
        r_hat  : (n_r,)          absolute r (= r_current + Δr)
        C_r    : (n_r, n_r)      posterior covariance of r
        a_arr  : (n_stars, 5)    astrometry mean when Δr=0 (i.e., at r=r_current)
        K_img  : dict{img->(n,5,8)}
        C_vT   : (n_stars, 5, 5) astrometry posterior covariance conditional on r
        """
        nr = self.N_R
        n_r = nr * self.n_images

        # ── Precision matrices and information vectors ─────────────────────────
        H_vv = self.C_survey_inv.copy()
        H_vv[:, np.arange(N_V), np.arange(N_V)] += self._C_VG_inv_per_star

        # h_align: prior + alignment contributions only.
        #   Used in the Schur complement rhs so that image-calibration parameters
        #   are driven only by alignment detections.  Prevents slow convergence
        #   caused by astrometry-only residuals (which depend on r_j of other
        #   images) creating indirect cross-image coupling.
        # h_all: prior + alignment + astrometry-only contributions.
        #   Used to compute the returned stellar posteriors so that astrometry-only
        #   detections constrain each star's own v_hat.
        h_align = self.C_survey_inv_dot_v.copy()
        h_all   = self.C_survey_inv_dot_v.copy()

        H_rr = np.zeros((n_r, n_r))

        K_img = {}
        XCs_xresid = {}

        for j_idx, img in enumerate(self.image_names):
            d = self._img_data[img]
            if d is None:
                K_img[img] = None
                continue

            if z_weights is not None:
                # Soft-weight two-tier mode: Gaia-matched (align_init) drive the
                # transformation; all Phase-0-surviving detections (including
                # HST-only) constrain stellar astrometry.  This mirrors the hard-EM
                # two-tier split that keeps HST-only out of the transformation
                # estimate, preventing the instability from Bug 9.
                z          = z_weights[img]   # (n,) float, 0 for excluded detections
                # Mirror the hard-EM two-tier exactly: same Gaia population for
                # the transformation (post-Phase-0 use_for_fit), same astrometry
                # population (use_for_fit | use_for_astrom, i.e. Gaia + callback-
                # enabled HST-only).  Using use_for_align_init instead of
                # use_for_fit would include Phase-0-rejected Gaia detections which
                # shift the transformation away from the hard-EM fixed point.
                use_align  = d["use_for_fit"]      # post-Phase-0 Gaia → H_rr
                use_astrom = (d["use_for_fit"]
                              | d.get("use_for_astrom",
                                      d["use_for_fit"]))  # Gaia + HST-only → H_vv
            else:
                use_align  = d["use_for_fit"]
                if getattr(self, '_use_two_tier', False):
                    use_astrom = d.get("use_for_astrom", use_align)
                else:
                    use_astrom = use_align
            use_any    = use_align | use_astrom   # for H_vv/h_all (stellar precision)
            sidx_any   = d["sidx"][use_any]
            sidx_align = d["sidx"][use_align]
            JU   = d["JU"]       # (n, 2, 5)
            X    = d["X_mat"]    # (n, 2, N_R)
            xys  = d["xys"]      # (n, 2)

            # Extract r_j first so we can pass it to _compute_Cs (poly Jacobian)
            cs  = j_idx * nr
            r_j = r_current[cs:cs + nr]

            Cs     = self._compute_Cs(img, r_j)   # (n, 2, 2)
            Cs_inv = np.linalg.inv(Cs)

            if z_weights is not None:
                # Scale precision by soft weight: (n,2,2) * (n,1,1)
                Cs_inv = Cs_inv * z[:, None, None]

            x_pred  = np.einsum('nkl,l->nk', X, r_j)
            x_resid = xys - x_pred

            JUT_Cs = np.einsum('nki,nkl->nil', JU, Cs_inv)

            # H_vv: all stars used for either alignment or astrometry
            np.add.at(H_vv, sidx_any, np.einsum('nik,nkj->nij', JUT_Cs[use_any], JU[use_any]))

            # h_all: residual information from all used detections
            np.subtract.at(h_all, sidx_any, np.einsum('nik,nk->ni', JUT_Cs[use_any], x_resid[use_any]))

            # h_align: residual information from alignment detections only
            # (used in Schur complement rhs to avoid cross-image coupling)
            np.subtract.at(h_align, sidx_align, np.einsum('nik,nk->ni', JUT_Cs[use_align], x_resid[use_align]))

            K = np.einsum('nik,nkl->nil', JUT_Cs, X)   # (n, 5, N_R)
            K_img[img] = K

            # H_rr/XCs_xresid: alignment stars only (calibrate image transform)
            XCsX = np.einsum('nki,nkl,nlj->ij', X[use_align], Cs_inv[use_align], X[use_align])
            H_rr[cs:cs+nr, cs:cs+nr] += XCsX
            XCs_xresid[img] = np.einsum('nki,nkl,nl->ni', X[use_align], Cs_inv[use_align], x_resid[use_align])

            H_rr[cs:cs+nr, cs:cs+nr] += self._img_data[img]["C_r_prior_inv"]

        # ── Invert H_vv → C_vT ────────────────────────────────────────────────
        C_vT    = np.linalg.inv(H_vv)
        a_align = np.einsum('nij,nj->ni', C_vT, h_align)  # for Schur complement rhs
        a       = np.einsum('nij,nj->ni', C_vT, h_all)    # returned stellar posteriors

        # ── Schur complement for Δr ────────────────────────────────────────────
        Cr_inv = H_rr.copy()
        rhs    = np.zeros(n_r)

        for j_idx, img in enumerate(self.image_names):
            r_prior_j      = self._img_data[img]["r_prior"]
            Cr_prior_inv_j = self._img_data[img]["C_r_prior_inv"]
            cs = j_idx * nr
            rhs[cs:cs+nr] += Cr_prior_inv_j @ (r_prior_j - r_current[cs:cs+nr])

            d = self._img_data[img]
            if d is None or K_img[img] is None:
                continue
            use  = d["use_for_fit"]
            sidx = d["sidx"][use]
            K    = K_img[img][use]

            rhs[cs:cs+nr] += XCs_xresid[img].sum(axis=0)

            CvT_K    = np.einsum('nij,njk->nik', C_vT[sidx], K)
            KT_CvT_K = np.einsum('nji,njk->ik',  K, CvT_K)
            Cr_inv[cs:cs+nr, cs:cs+nr] -= KT_CvT_K

            rhs[cs:cs+nr] += np.einsum('nji,nj->i', K, a_align[sidx])

            for j2_idx, img2 in enumerate(self.image_names):
                if j2_idx <= j_idx:
                    continue
                d2 = self._img_data[img2]
                if d2 is None or K_img[img2] is None:
                    continue
                use2 = d2["use_for_fit"]
                sidx2 = d2["sidx"][use2]
                K2    = K_img[img2][use2]

                common, idx1, idx2 = np.intersect1d(sidx, sidx2,
                                                     return_indices=True)
                if len(common) == 0:
                    continue

                CvT_c  = C_vT[common]
                CvT_K2 = np.einsum('nij,njk->nik', CvT_c, K2[idx2])
                block  = np.einsum('nji,njk->ik', K[idx1], CvT_K2)

                cs2 = j2_idx * nr
                Cr_inv[cs:cs+nr,   cs2:cs2+nr] -= block
                Cr_inv[cs2:cs2+nr, cs:cs+nr]   -= block.T

        # ── Solve for Δr, then r_hat = r_current + Δr ─────────────────────────
        # Diagonal preconditioning: the (a,b,c,d) columns have scale ~2048 px
        # while (w,z) columns have scale ~1, giving a ~4e6 condition ratio.
        # Scaling Cr_inv by D^{-1} on both sides (D = sqrt(diag)) reduces the
        # effective condition number to ~1 before inversion.
        # Math: D^{-1} Cr_inv D^{-1} @ D delta_r_tilde = D^{-1} rhs
        #  → C_r = D^{-1} inv(Cr_inv_sc) D^{-1};  delta_r = C_r @ rhs
        d_diag     = np.sqrt(np.maximum(np.abs(np.diag(Cr_inv)), 1e-30))
        d_inv      = 1.0 / d_diag
        Cr_inv_sc  = d_inv[:, None] * Cr_inv * d_inv[None, :]
        try:
            C_r_sc = np.linalg.inv(Cr_inv_sc)
        except np.linalg.LinAlgError:
            C_r_sc = np.linalg.pinv(Cr_inv_sc)
        C_r     = d_inv[:, None] * C_r_sc * d_inv[None, :]
        delta_r = C_r @ rhs
        r_hat   = r_current + delta_r

        # for j_idx, img in enumerate(self.image_names):
        #     meta = self.images[img]
        #     cs = j_idx * N_R
        #     ag,bg,cg,dg = r_hat[cs:cs+N_R][:4]
        #     on_skew = (ag-dg)/2
        #     off_skew = (bg+cg)/2
        #     ratio = np.sqrt(ag*dg-bg*cg)
        #     rot = np.arctan2((bg-cg),(ag+dg))/DEG2RAD
        #     print(img)
        #     print(delta_r[cs:cs+N_R])
        #     print(r_hat[cs:cs+N_R][4:])
        #     print(rot,ratio,on_skew,off_skew)
        #     print(rot-meta['orig_rot_deg'])
        #     print()
        # print()

        return r_hat, C_r, a, K_img, C_vT

    def _r_to_dict(self, r_hat):
        nr = self.N_R
        return {img: r_hat[j*nr:(j+1)*nr]
                for j, img in enumerate(self.image_names)}

    def _update_R(self, r_hat):
        nr = self.N_R
        self._r_hat_current = r_hat.copy()
        for j_idx, img in enumerate(self.image_names):
            r_j = r_hat[j_idx * nr:(j_idx + 1) * nr]
            self.R[img] = rotation_matrix_from_abcd(*r_j[:4])

    # ── Public fit interface ───────────────────────────────────────────────────

    def fit(self, n_iter=20, tol=1e-6, clip_sigma=4.5, inflate_hst_errors=False,
            inflate_from_iter=3, min_outer_iters=None,
            hst_fit_sigma_mult=0.5,
            prefilter=True, chi2_threshold=None, alpha_scale_chi2=False,
            use_influence_clip=True, influence_d_thresh=1.0, influence_sigma_min=2.0,
            use_two_tier=False, per_iter_callback=None,
            use_soft_weights: bool = False,
            student_t_nu: float = 50.0,
            z_tol: float = 1.0,
            z_init: dict | None = None):
        """
        Iterative BP3M fit with outlier rejection.

        Structure
        ---------
        Phase 1 — initial convergence (full sample, no outlier updates):
          Iterate _solve_one_pass until max|Δr| < tol.

        Phase 2 — EM-style outer/inner loops (when clip_sigma is not None):
          Outer loop (up to n_iter iterations):
            1. Update outliers (_update_use_for_fit) based on current solution.
            2. If use_for_fit did not change, stop — solution is fully converged
               for the accepted star set.
            3. Inner loop: iterate _solve_one_pass until max|Δr| < tol with
               the new (frozen) use_for_fit.  This ensures each outlier update
               starts from the exact MAP for the current accepted set, not a
               partially-converged approximation.
          Stop when the outlier set is stable (no use_for_fit changes).

        This guarantees that the returned r_hat is the exact MAP for the final
        accepted star set — no separate frozen-outlier phase needed.

        Parameters
        ----------
        n_iter     : int,   maximum number of outer (outlier-update) iterations
        tol        : float, inner-loop convergence threshold on max|Δr|
        clip_sigma : float or None
            Kept for API compatibility; actual rejection uses chi2 thresholds.
            Set to None to skip outlier rejection entirely.
        inflate_hst_errors : bool, default False
            If True, apply per-image C_hst inflation (alpha adjustment) starting
            at outer iteration inflate_from_iter.  Can cause oscillation; leave
            False unless you have a specific reason to enable it.
        inflate_from_iter : int, default 3
            First outer iteration (0-based) at which alpha inflation updates fire.
            Default 3 gives outlier rejection time to stabilise before alpha is
            adjusted.  Pass 0 when starting from a pre-validated alpha (e.g. the
            v1 BP3M result) so alpha can decrease from the v1 starting value on
            the very first EM iteration.  The update formula is always
            ``max(1.0, alpha_prev * alpha_raw)`` so alpha never drops below 1
            relative to C_hst_orig, regardless of this setting.
        min_outer_iters : int or None, default None
            Minimum number of outer EM iterations before early stopping is
            allowed.  None → 4 if inflate_hst_errors else 2.  Set explicitly
            when HST-only sources are enabled mid-run (e.g.
            ``max(hst_enable_iter + 3, 4)``) so the EM has time to converge
            after the new sources are added.
        hst_fit_sigma_mult : float, default 0.5
            Multiplicative factor applied to the per-image residual threshold
            for detections from stars that were NOT initially in alignment
            (e.g. HST-only sources admitted by V2AlignmentCallback).  Must
            be <= 1.  A value of 0.5 means HST-only detections must have
            sigma_resid < 0.5 × thresh_Gaia to stay in use_for_fit.  Stricter
            than Gaia-matched because HST-only stars can have biased PM priors
            (field-star contamination) that would otherwise pull the
            transformation away from the Gaia-constrained solution.
        prefilter : bool, default True
            Before Phase 1, run one solve pass and apply _update_use_for_fit
            (identical logic to Phase 2) to establish a clean initial star set.
            The updated r_hat from this pass is used as the starting point for
            Phase 1.  Pass False to skip (starts Phase 1 with all stars that
            passed use_for_alignment).
        chi2_threshold : float or None, default None
            If given, replaces the adaptive p50+k*(p84-p50) threshold in test 3
            with a fixed chi2 cut (e.g. 9.21 = chi2(2).ppf(0.99)).
            Expulsion threshold is scaled to chi2_threshold*(1+delta/k) to
            preserve hysteresis.  None → use the standard adaptive threshold.
        alpha_scale_chi2 : bool, default False
            If True, divide each star's HST chi2 by alpha² before applying the
            test-3 threshold, starting at outer iteration 3.  Alpha is estimated
            from the previous iteration's accepted star set for that image, so it
            is available without a chicken-and-egg dependency.  This makes the
            threshold image-independent in units of "sigma given image noise"
            rather than raw chi2, preventing over-rejection of images whose
            formal errors are slightly underestimated.
        use_influence_clip : bool, default True
            If True, apply test-4 influence-based clipping after the EM
            converges (tests 1-3).  Flags detections where Cook's D >
            influence_d_thresh AND sigma_resid > influence_sigma_min.
            Cook's D captures moderate-residual, high-leverage detections that
            pass the sigma threshold but disproportionately pull r_hat.
            The flagging is reset at each outer iteration so that stars are not
            permanently excluded based on a possibly-wrong r_hat.
        influence_d_thresh : float, default 1.0
            Cook's D threshold for test-4.  D > 1 means removing this detection
            shifts r_hat by more than one sqrt(C_r) length.
        influence_sigma_min : float, default 2.0
            Minimum sigma_resid for a detection to be flagged by test-4.
            Prevents removing well-fit high-leverage stars (low residual, high
            leverage is NOT a problem — it means the star is a good anchor).

        per_iter_callback : callable or None, default None
            If provided, called as ``per_iter_callback(solver, it_outer)``
            at the end of each Phase-2 outer iteration (after _inner_converge).
            ``it_outer`` is the 1-based outer iteration number.  The callback
            may read ``solver._r_hat_current`` and ``solver._img_data`` and
            modify ``solver._img_data[img]["use_for_fit"]`` /
            ``solver._img_data[img]["use_for_astrom"]`` in-place (e.g. to
            enable HST-only sources in the v2 phased-inclusion scheme).

        Returns
        -------
        r_hat       : (n_r,)
        C_r         : (n_r, n_r)
        v_hat       : (n_stars, 5)  final astrometry posteriors
        C_vT        : (n_stars, 5, 5)  conditional covariance (given r_hat)
        a_arr       : (n_stars, 5)  astrometry at Δr=0 (= r_hat from last iter)
        K_img       : dict  K matrices per image (for sampling)
        z_weights_out : dict or None  soft weights if use_soft_weights=True, else None
        """
        # Store as instance variable so _solve_one_pass and _update_use_for_fit
        # can access it without signature changes.
        self._use_two_tier = use_two_tier

        r_hat = np.concatenate([self._img_data[img]["r_init"]
                                 for img in self.image_names])
        self._update_R(r_hat)
        nr  = self.N_R
        C_r = None

        # Parameter names for diagnostic output (indices 6,7 are always zeroed)
        _pnames = ['a', 'b', 'c', 'd', 'w', 'z', 'Δα0', 'Δδ0']
        if nr > 8:
            _pnames += [f'poly{i}' for i in range(nr - 8)]
        _n_imgs = len(self.image_names)

        def _delta_summary(diff):
            """Return formatted strings: (max_location_str, per_param_stats_str)."""
            imax      = int(np.argmax(diff))
            img_idx   = imax // nr
            param_idx = imax % nr
            max_str   = (f"{diff[imax]:.3e}"
                         f"  [{self.image_names[img_idx]} / {_pnames[param_idx]}]")

            # Per-parameter median and 68% width across images (skip pinned params 6,7)
            parts = []
            for p in range(nr):
                if p in (6, 7):
                    continue
                vals = diff[p::nr]
                med  = float(np.median(vals))
                if _n_imgs > 1:
                    w68 = float(np.percentile(vals, 84) - np.percentile(vals, 16))
                    parts.append(f"{_pnames[p]}: {med:.2e} [{w68:.2e}]")
                else:
                    parts.append(f"{_pnames[p]}: {med:.2e}")
            return max_str, '  '.join(parts)

        def _inner_converge(r_hat, label, z_weights=None):
            """Iterate _solve_one_pass until max|Δr| < tol. Returns updated r_hat etc."""
            for it_i in range(500):
                r_new, C_r_i, a_i, K_i, CvT_i = self._solve_one_pass(r_hat, z_weights=z_weights)
                diff = np.abs(r_new - r_hat)
                diff[6::nr] = 0
                diff[7::nr] = 0
                delta = np.max(diff)
                r_hat = r_new
                self._update_R(r_hat)
                self._update_geometry(r_hat, a_i)
                if it_i % 10 == 0:
                    max_str, stats_str = _delta_summary(diff)
                    print(f"  {label}: step {it_i+1:3d},  max|Δr| = {max_str}")
                    print(f"    params: {stats_str}")
                if delta < tol:
                    max_str, stats_str = _delta_summary(diff)
                    print(f"  {label}: converged in {it_i+1} inner steps "
                          f"(max|Δr| = {max_str})")
                    print(f"    params: {stats_str}")
                    return r_hat, C_r_i, a_i, K_i, CvT_i
            max_str, stats_str = _delta_summary(diff)
            print(f"  {label}: WARNING — did not converge (max|Δr| = {max_str})")
            print(f"    params: {stats_str}")
            return r_hat, C_r_i, a_i, K_i, CvT_i

        # ── Phase 0: pre-filter using one solve + same outlier rejection as Phase 2
        if prefilter and clip_sigma is not None:
            print(' Phase 0: pre-filter (one solve pass + outlier rejection)')
            r_hat, C_r, a_arr, K_img, C_vT = self._solve_one_pass(r_hat)
            self._update_R(r_hat)
            self._update_geometry(r_hat, a_arr)
            clip_info, _, _ = self._update_use_for_fit(
                r_hat, a_arr, C_r, C_vT, clip_sigma,
                ok_star_prev=None, inflate_errors=False,
                skip_star_tests=True,
                chi2_threshold=chi2_threshold,
                alpha_scale_chi2=False)   # no alpha scaling in pre-filter
            n_in  = sum(n_use for _, n_use, _, _, _, _ in clip_info)
            n_tot = sum(n_t   for _, _,    n_t, _, _, _ in clip_info)
            print(f"  Pre-filter: {n_in}/{n_tot} stars accepted across all images\n")

        # ── Phase 1: initial convergence with filtered sample ─────────────────
        if n_iter == 0:
            print(' Phase 1: skipped (n_iter=0, transformation held fixed at r_init)')
            _, C_r, a_arr, K_img, C_vT = self._solve_one_pass(r_hat)
            self._update_geometry(r_hat, a_arr)
        else:
            print(' Phase 1: convergence with pre-filtered sample')
            r_hat, C_r, a_arr, K_img, C_vT = _inner_converge(r_hat, 'init')

        # ── Phase 2: EM-style outlier rejection ───────────────────────────────
        # Tests 1-3 (chi² / sigma) and test-4 (Cook's D influence) run
        # together in the same outer loop.  Test-4 uses a ratchet: newly
        # flagged detections are added to _img_data[img]["influence_excl"],
        # which _update_use_for_fit treats as a permanent ceiling so they
        # can never be re-admitted by tests 1-3.  The ratchet guarantees
        # monotonic convergence: n_inf_new decreases toward 0.
        _default_min = 4 if inflate_hst_errors else 2
        min_outer = int(min_outer_iters) if min_outer_iters is not None else _default_min

        z_weights_out = None   # set below if use_soft_weights=True

        if use_soft_weights and clip_sigma is not None:
            print(f'\n Phase 2 (soft-weight IRLS): Student-t downweighting  (ν={student_t_nu})')

            # Seed PM estimates for all stars (including HST-only) before IRLS
            # starts.  Calling the callback at hst_enable_iter triggers the PM
            # seeding step that sets v_survey for HST-only sources from the
            # xmatch catalogue.  Without this, HST-only PMs stay at the diffuse
            # prior (0 mas/yr), giving huge residuals → z≈0 → PMs never improve.
            if per_iter_callback is not None:
                _seed_iter = getattr(per_iter_callback, 'hst_enable_iter', None) or 0
                per_iter_callback(self, _seed_iter)
                # Re-solve once so a_arr reflects the seeded PM estimates before
                # we compute the first set of IRLS weights.
                r_hat, C_r, a_arr, K_img, C_vT = _inner_converge(
                    r_hat, 'soft-w seed', z_weights=None)

            # Initialise weights.  If Phase-6 chi2 values were pre-computed
            # during catalogue building and passed in as z_init, use them
            # directly — they provide a better warm start than the seed-solve
            # residuals for detections the catalogue already flagged as poor.
            # Fall back to computing from the seed-solve residuals otherwise.
            if z_init is not None:
                # Phase-6 chi2 warm start.  Any images missing from z_init fall
                # back to seed-solve residuals (single shared call for efficiency).
                if any(z_init.get(img) is None for img in self.image_names):
                    _z_fb, _, _ = self._update_soft_weights(r_hat, a_arr, student_t_nu)
                    z_weights = {img: (z_init[img] if z_init.get(img) is not None
                                       else _z_fb.get(img))
                                 for img in self.image_names}
                else:
                    z_weights = z_init
            else:
                z_weights, _, _ = self._update_soft_weights(r_hat, a_arr, student_t_nu)

            _n_consec_z_stable = 0
            for it_outer in range(n_iter):
                z_new, n_det, n_eff = self._update_soft_weights(r_hat, a_arr, student_t_nu)

                delta_z = sum(float(np.abs(z_new[img] - z_weights[img]).sum())
                              for img in z_new if z_new[img] is not None)
                z_weights = z_new

                if delta_z < z_tol:
                    _n_consec_z_stable += 1
                else:
                    _n_consec_z_stable = 0

                print(f"\n  Soft-weight iter {it_outer+1}: "
                      f"N_eff={n_eff:.1f}/{n_det} ({100*n_eff/max(n_det,1):.1f}%),  "
                      f"Δz={delta_z:.3f}")

                if _n_consec_z_stable >= 2 and it_outer >= min_outer:
                    print(f"  Weights converged (Δz < {z_tol} for 2 consecutive iters) — stopping.")
                    break

                r_hat, C_r, a_arr, K_img, C_vT = _inner_converge(
                    r_hat, f'soft-w {it_outer+1}', z_weights=z_weights)
            else:
                print(f"  Stopped after {n_iter} IRLS iterations (weights did not converge)")

            z_weights_out = z_weights

        elif clip_sigma is not None:
            print('\n Phase 2: EM-style outlier rejection')

            ok_star_prev = np.ones(self.n_stars, dtype=bool)
            _n_consec_stable = 0  # consecutive iters with 0 tests-1/2/3 changes

            for it_outer in range(n_iter):
                clip_info, ok_star_new, n_use_changed = self._update_use_for_fit(
                    r_hat, a_arr, C_r, C_vT, clip_sigma, iteration=it_outer,
                    ok_star_prev=ok_star_prev, inflate_errors=inflate_hst_errors,
                    inflate_from_iter=inflate_from_iter,
                    hst_fit_sigma_mult=hst_fit_sigma_mult,
                    chi2_threshold=chi2_threshold, alpha_scale_chi2=alpha_scale_chi2)

                n_global_changed = int(np.sum(ok_star_prev != ok_star_new))
                n_total_changed  = n_global_changed + n_use_changed

                # Track consecutive stability of tests 1-3 (before test-4).
                if n_global_changed == 0 and n_use_changed == 0:
                    _n_consec_stable += 1
                else:
                    _n_consec_stable = 0

                # Test-4: influence clipping.
                # Only runs after min_outer iters so chi² tests remove gross
                # outliers first.  Also suppressed once the EM has been stable
                # for ≥2 consecutive iterations: firing Cook's D on a fully
                # converged solution can perturb sparse fields (few Gaia stars)
                # and trigger a cascade.  The C_r ratio pre-scales the threshold
                # to match V1's physical shift magnitude, but the timing guard
                # is the primary stability mechanism.
                n_inf_new = 0
                if use_influence_clip and it_outer >= min_outer and _n_consec_stable < 2:
                    n_inf_new = self._apply_influence_clip(
                        r_hat, C_r, a_arr,
                        cooks_d_thresh=influence_d_thresh,
                        sigma_min=influence_sigma_min)
                    n_total_changed += n_inf_new

                print(f"\n  Outer iter {it_outer+1}: "
                      f"{n_global_changed} test-1/2 changes, "
                      f"{n_use_changed} test-3 changes"
                      + (f", {n_inf_new} test-4 changes" if use_influence_clip else "")
                      + f"  ({n_total_changed} total)")
                for img, n_use, n_tot, alpha_applied, alpha_raw, n_astrom_only in clip_info:
                    tags = []
                    if inflate_hst_errors and it_outer >= 3:
                        tags.append("α-inflated")
                    if alpha_scale_chi2 and it_outer >= 3:
                        tags.append("α-scaled-chi2")
                    tag_str = f"  [{', '.join(tags)}]" if tags else ""
                    if inflate_hst_errors and it_outer >= 3:
                        alpha_str = (f"α_applied={alpha_applied:.3f}  "
                                     f"α_raw={alpha_raw:.3f}")
                    else:
                        alpha_str = f"α={alpha_applied:.3f}"
                    astrom_str = f" (+{n_astrom_only} astrom-only)" if n_astrom_only > 0 else ""
                    print(f"    {img}: {n_use}/{n_tot} align{astrom_str},  {alpha_str}{tag_str}")

                ok_star_prev = ok_star_new.copy()

                # Convergence: tests 1-2 stable AND test-4 found nothing new.
                # The ratchet guarantees n_inf_new → 0, so this always terminates.
                if (n_global_changed == 0 and n_inf_new == 0
                        and it_outer >= min_outer):
                    print(f"  Tests 1-4 stable — stopping.")
                    break

                r_hat, C_r, a_arr, K_img, C_vT = _inner_converge(
                    r_hat, f'outer {it_outer+1}')

                if per_iter_callback is not None:
                    per_iter_callback(self, it_outer + 1)
            else:
                print(f"  Stopped after {n_iter} outer iterations "
                      f"(star set did not fully stabilise)")

        # Final v_hat = a_arr (Δr = 0 at the last converged r_hat)
        v_hat = a_arr.copy()

        return r_hat, C_r, v_hat, C_vT, a_arr, K_img, z_weights_out

    def _update_soft_weights(self, r_hat, a_arr, nu=5.0):
        """
        Compute per-detection Student-t IRLS weights from current residuals.

        For each detection k in image j:
            chi2_k = res_k^T Cs_inv_k res_k
            z_k    = min(1, (nu+2) / (nu+chi2_k))
        Phase-0 hard rejections (use_for_fit_max=False) receive z=0.

        Returns
        -------
        z_dict      : {img: (n,) float}
        n_det_total : int   total number of Phase-0-surviving detections
        n_eff_total : float sum of all z values (effective sample size)
        """
        nr = self.N_R
        z_dict = {}
        n_det_total = 0
        n_eff_total = 0.0

        for j_idx, img in enumerate(self.image_names):
            d = self._img_data.get(img)
            if d is None:
                z_dict[img] = None
                continue

            cs   = j_idx * nr
            r_j  = r_hat[cs:cs + nr]
            sidx = d["sidx"]
            n    = d["n"]

            JU   = d["JU"]      # (n, 2, 5)
            X    = d["X_mat"]   # (n, 2, N_R)
            xys  = d["xys"]     # (n, 2)

            Cs     = self._compute_Cs(img, r_j)
            Cs_inv = np.linalg.inv(Cs)   # (n, 2, 2)

            # Model: xys = X r - JU a + noise  (JU carries the sign; see
            # build_X_matrix / sample_posteriors for the same convention).
            # Residual = xys - (X r - JU a) = xys - X r + JU a.
            pred  = (np.einsum('nij,j->ni',  X,  r_j)
                     - np.einsum('nij,nj->ni', JU, a_arr[sidx]))   # (n, 2)
            res   = xys - pred                                        # (n, 2)

            chi2  = np.einsum('ni,nij,nj->n', res, Cs_inv, res)     # (n,)
            z     = np.minimum(1.0, (nu + 2.0) / (nu + chi2))

            # Hard floor: Phase-0-rejected detections always get z=0.
            # All surviving detections (Gaia-matched and HST-only alike) get
            # soft weights from their residuals.  HST-only PMs are seeded from
            # the xmatch catalogue before the first weight computation, so their
            # residuals start small for good detections.
            # Same population as the hard-EM astrometry tier: post-Phase-0 Gaia
            # (use_for_fit) plus callback-enabled HST-only (use_for_astrom).
            # Excludes Phase-0-rejected detections which would otherwise get
            # z > 0 and shift the transformation relative to the hard-EM solution.
            _astrom_mask = d["use_for_fit"] | d.get("use_for_astrom", d["use_for_fit"])
            mask  = _astrom_mask.astype(float)
            z    *= mask

            z_dict[img] = z
            n_det_total += int(mask.sum())
            n_eff_total += float(z.sum())

        return z_dict, n_det_total, n_eff_total

    def _update_use_for_fit(self, r_hat, v_hat, C_r, C_vT, clip_sigma,
                            chi2_pval=0.95, iteration=0,
                            adaptive_k=5.0, adaptive_delta=0.1,
                            sigma_pm_diffuse=100.0, sigma_plx_diffuse=20.0,
                            ok_star_prev=None, inflate_errors=False,
                            inflate_from_iter=3,
                            hst_fit_sigma_mult=0.5,
                            skip_star_tests=False,
                            chi2_threshold=None, alpha_scale_chi2=False):
        """
        Update use_for_fit via two star-level chi2 tests plus per-image
        residual clipping.

        All chi2 thresholds use a data-driven adaptive form p50 + k*(p84-p50)
        computed from the empirical chi2 distribution of currently-observed stars.
        This is more robust than a fixed chi2.ppf threshold.

        Hysteresis (adaptive_delta > 0): currently-included stars require chi2 >
        p50 + (k+delta)*(p84-p50) to be expelled; currently-excluded stars must
        clear p50 + k*(p84-p50) to be re-admitted.  This dead-band prevents
        borderline stars from oscillating and stabilises EM convergence.

        Star-level tests (applied globally):

          1. Gaia prior:
               chi2_gaia = (ã_i - v_s_i)^T (C_vT_i + C_survey_i)^{-1} (ã_i - v_s_i)
             Adaptive threshold computed separately for df=5 and df=2 populations.
             Hysteresis applied when ok_star_prev is provided.

          2. Diffuse prior — catches stars with physically extreme astrometry:
               chi2_diff = sum((v_hat / sigma_diffuse)^2)
             Fixed threshold chi2.ppf(0.9999, df=5) ≈ 21.7 (data-independent).

        Per-image test (stars may be excluded from one image but kept in others):
          3. Position residuals: sigma_resid^2 < adaptive threshold from
             globally-accepted stars in this image.

        Returns (info, ok_star) where:
          info    : list of (img_name, n_used, n_total, alpha) for logging
          ok_star : (n_stars,) bool — stars passing tests 1 and 2
        """
        from scipy.stats import chi2 as chi2_dist

        observed = self.gaia_n_hst_used > 0   # stars used in previous iteration

        # Theoretical chi2 floors: adaptive thresholds may not drop below the
        # q=0.99 expected value for the relevant distribution.  This prevents
        # runaway exclusion when the empirical chi2 distribution narrows
        # (e.g. in single-epoch runs where few stars constrain the image).
        floor_5 = float(chi2_dist.ppf(0.99, df=5))  # ≈ 15.1
        floor_2 = float(chi2_dist.ppf(0.99, df=2))  # ≈ 9.2

        def _adapt_thresh(values, k, fallback, floor=0.0):
            """p50 + k*(p50-p16), floored at `floor`; fallback when few points.
            Returns (threshold, p16, p50, p84)."""
            if len(values) < 10:
                return float(max(fallback, floor)), float('nan'), float('nan'), float('nan')
            p16 = float(np.percentile(values, 16))
            p50 = float(np.median(values))
            p84 = float(np.percentile(values, 84))
            return float(max(p50 + k * max(p50 - p16, 1e-6), floor)), p16, p50, p84

        # ── 1. Gaia prior chi2 test ───────────────────────────────────────────
        # chi2 = (ã - v_s)^T (C_vT + C_survey)^{-1} (ã - v_s)
        delta_gaia = v_hat - self.v_survey             # (n_stars, 5)
        C_comb     = C_vT + self.C_survey              # (n_stars, 5, 5)
        C_comb_inv = np.linalg.inv(C_comb)
        chi2_gaia  = np.einsum('ni,nij,nj->n', delta_gaia, C_comb_inv, delta_gaia)

        # Adaptive thresholds: separate df=5 (5/6-param) and df=2 (2-param).
        # 2p Gaia solutions have near-infinite C_survey for pm/plx, so chi2_gaia
        # is effectively df=2 (only position components constrained by Gaia).
        obs_5p = observed & ~self.gaia_2p
        obs_2p = observed & self.gaia_2p
        thresh_gaia_5, p16_5, p50_5, p84_5 = _adapt_thresh(
            chi2_gaia[obs_5p], adaptive_k, chi2_dist.ppf(chi2_pval, df=5), floor=floor_5)
        thresh_gaia_2, p16_2, p50_2, p84_2 = _adapt_thresh(
            chi2_gaia[obs_2p], adaptive_k, chi2_dist.ppf(chi2_pval, df=2), floor=floor_2)
        # Admission: a star must clear thresh_gaia to be (re-)included.
        ok_gaia_admit = np.where(self.gaia_2p,
                                 chi2_gaia < thresh_gaia_2,
                                 chi2_gaia < thresh_gaia_5)

        # Hysteresis: currently-included stars use a higher (looser) expulsion
        # threshold thresh_out = p50 + (k+delta)*(p84-p50).  The dead-band
        # between admission and expulsion prevents borderline stars from
        # oscillating as the adaptive threshold shifts slightly between iterations.
        if ok_star_prev is not None and adaptive_delta > 0:
            thresh_out_5, _, _, _ = _adapt_thresh(chi2_gaia[obs_5p],
                                         adaptive_k + adaptive_delta,
                                         chi2_dist.ppf(chi2_pval, df=5), floor=floor_5)
            thresh_out_2, _, _, _ = _adapt_thresh(chi2_gaia[obs_2p],
                                         adaptive_k + adaptive_delta,
                                         chi2_dist.ppf(chi2_pval, df=2), floor=floor_2)
            ok_gaia_retain = np.where(self.gaia_2p,
                                      chi2_gaia < thresh_out_2,
                                      chi2_gaia < thresh_out_5)
            # Currently in: keep unless chi2 > thresh_out.
            # Currently out: admit only if chi2 < thresh_gaia.
            ok_gaia = np.where(ok_star_prev, ok_gaia_retain, ok_gaia_admit)
        else:
            thresh_out_5 = thresh_gaia_5   # symmetric (no hysteresis)
            ok_gaia = ok_gaia_admit

        # ── 2. Diffuse prior test (fixed, data-independent) ──────────────────
        # Excludes stars with physically extreme astrometry regardless of what
        # the rest of the sample is doing.  Uses per-star sigma_diff so that
        # Gaia 5p/6p stars (which have no diffuse prior) are never expelled here —
        # their Gaia chi2 test (test 1) is the sole outlier criterion.
        chi2_diff  = np.sum((v_hat / self._sigma_diff_per_star)**2, axis=1)  # (n_stars,)
        # 2-sigma equivalent: quantile of chi2(5) matching 2σ in 1D (Φ(2)≈0.9545)
        thresh_diff = float(chi2_dist.ppf(chi2_dist.cdf(4.0, df=1), df=5))  # ≈ 11.1
        ok_diffuse  = chi2_diff < thresh_diff

        ok_star = ok_gaia & ok_diffuse

        # Store chi2_gaia for diagnostics
        self.sigma_from_gaia_prior[:] = np.sqrt(chi2_gaia)

        # When called from Phase 0 pre-filter, v_hat is not yet reliable (only
        # one un-converged pass).  Skip tests 1+2 and filter on position only.
        if skip_star_tests:
            ok_star    = np.ones(self.n_stars, dtype=bool)
            ok_diffuse = np.ones(self.n_stars, dtype=bool)  # diffuse test unreliable before Phase 1
        else:
            # Logging (skipped in pre-filter: diffuse-prior chi2 is meaningless
            # before Phase 1 convergence and would just be confusing)
            n_obs          = int(observed.sum())
            n_fail_gaia    = int((~ok_gaia   & observed).sum())
            n_fail_diffuse = int((~ok_diffuse & ok_gaia & observed).sum())
            hyst_str = (f"→{thresh_out_5:.2f}" if ok_star_prev is not None
                        and adaptive_delta > 0 else "")
            n_obs_5p = int(obs_5p.sum())
            n_obs_2p = int(obs_2p.sum())
            def _pct_str(p16, p50, p84, n):
                if np.isnan(p16):
                    return f"(n={n}, <10)"
                return f"[{p16:.1f},{p50:.1f},{p84:.1f}] (n={n})"
            print(f"    thresh  5p+6p:{thresh_gaia_5:.2f}{hyst_str} {_pct_str(p16_5,p50_5,p84_5,n_obs_5p)}  "
                  f"df=2:{thresh_gaia_2:.2f} {_pct_str(p16_2,p50_2,p84_2,n_obs_2p)}  "
                  f"diffuse:{thresh_diff:.1f}")

            # ── Per-population breakdown ──────────────────────────────────────
            # test-1 (Gaia chi2) and test-2 (diffuse) failures, split by population.
            # For 5p/6p stars print individual chi2 values (few enough to be useful).
            # For 2p just show counts.
            fail_gaia_5p  = ~ok_gaia   & obs_5p
            fail_gaia_2p  = ~ok_gaia   & obs_2p
            fail_diff_5p  = ~ok_diffuse & ok_gaia & obs_5p
            fail_diff_2p  = ~ok_diffuse & ok_gaia & obs_2p

            # new rejections vs new admissions (test-1/2 only)
            if ok_star_prev is not None:
                newly_rej = ok_star_prev & ~ok_star & observed
                newly_adm = ~ok_star_prev & ok_star & observed
                chg_str = (f"  ({int(newly_rej.sum())} newly rejected, "
                           f"{int(newly_adm.sum())} newly admitted)")
            else:
                chg_str = ""

            n_fail_gaia    = int((~ok_gaia   & observed).sum())
            n_fail_diffuse = int((~ok_diffuse & ok_gaia & observed).sum())
            print(f"    chi2 outliers (of {n_obs} observed): "
                  f"{n_fail_gaia} Gaia-incompatible "
                  f"({int(fail_gaia_5p.sum())} 5p+6p, {int(fail_gaia_2p.sum())} 2p), "
                  f"{n_fail_diffuse} diffuse "
                  f"({int(fail_diff_5p.sum())} 5p+6p, {int(fail_diff_2p.sum())} 2p)"
                  f"{chg_str}")


        # ── 3. Per-image position chi2 test + alpha estimation + flag update ───
        resid_full     = self.compute_residuals(r_hat, v_hat, C_r, C_vT)
        resid_hst      = self.compute_residuals(r_hat, v_hat)
        _MEDIAN_CHI2_2 = 2.0 * np.log(2.0)

        self.gaia_n_hst_used[:] = 0
        info = []
        n_use_changed = 0

        for img, rd in resid_full.items():
            sidx    = rd["sidx"]

            # Use HST-only chi2 (not C_total) for the per-image position test.
            # C_total inflates sigma_resid when the transformation is poorly
            # constrained (large C_r), masking genuinely bad HST positions.
            # HST-only chi2 is purely about position quality in the detector frame
            # and is stable regardless of transformation uncertainty.
            rd_hst  = resid_hst[img]
            sig_sq  = rd_hst["sigma_resid"]**2   # (n,) HST-noise-only chi2

            # Alpha-scale chi2: divide by alpha² from previous iteration so the
            # threshold is uniform across images in units of "sigma given image
            # noise".  Alpha is estimated from the previous use_for_fit to avoid
            # a chicken-and-egg dependency.  Only applied at iteration >= 3 so
            # early iterations exclude obvious outliers before alpha is reliable.
            prev_use = np.asarray(self._img_data[img]["use_for_fit"])
            if alpha_scale_chi2 and iteration >= 3 and prev_use.sum() >= 4:
                chi2_prev = sig_sq[prev_use]
                alpha_prev = float(max(1.0, np.sqrt(
                    np.median(chi2_prev) / _MEDIAN_CHI2_2)))
                sig_sq_eff = sig_sq / alpha_prev**2
            else:
                sig_sq_eff = sig_sq

            # Per-image threshold from globally-accepted stars.
            ok_glob_here = ok_star[sidx]

            # Threshold reference: restrict to initially-trusted stars so that
            # sources that began with use_for_alignment=False (e.g. non-stars
            # whose inflated PSF-fit covariances produce artificially small
            # sigma_resid) cannot pull the adaptive threshold down.
            init_trusted  = self._img_data[img]["use_for_align_init"]
            ok_thresh_ref = ok_glob_here & init_trusted
            if ok_thresh_ref.sum() < 10:
                ok_thresh_ref = ok_glob_here   # fall back if reference set too small

            if chi2_threshold is not None:
                # Fixed threshold; scale expulsion threshold by same ratio as
                # adaptive_k → adaptive_k+delta to preserve hysteresis width.
                thresh_admit = float(chi2_threshold)
                thresh_expel = thresh_admit * (1.0 + adaptive_delta / adaptive_k)
            else:
                thresh_admit, _, _, _ = _adapt_thresh(sig_sq_eff[ok_thresh_ref],
                                             adaptive_k,
                                             chi2_dist.ppf(chi2_pval, df=2),
                                             floor=floor_2)
                thresh_expel, _, _, _ = _adapt_thresh(sig_sq_eff[ok_thresh_ref],
                                             adaptive_k + adaptive_delta,
                                             chi2_dist.ppf(chi2_pval, df=2),
                                             floor=floor_2)

            ok_resid_admit = sig_sq_eff < thresh_admit

            # Hysteresis: currently-included stars use the looser expulsion threshold.
            if adaptive_delta > 0:
                ok_resid = np.where(prev_use, sig_sq_eff < thresh_expel, ok_resid_admit)
            else:
                ok_resid = ok_resid_admit

            # Stricter residual threshold for non-initially-aligned stars (HST-only
            # admitted via callback).  These stars may have biased PM priors and
            # generally larger positional scatter; requiring a smaller sigma_resid
            # limits their influence on the transformation without excluding all of
            # them.  hst_fit_sigma_mult < 1 means they must be hst_fit_sigma_mult ×
            # tighter than Gaia-matched stars; 0.5 ≈ 0.71σ stricter.
            if hst_fit_sigma_mult < 1.0:
                _align_init_k = np.asarray(self._img_data[img]["use_for_align_init"], dtype=bool)
                _hst_in_fit   = (~_align_init_k) & np.asarray(self._img_data[img]["use_for_fit"], dtype=bool)
                if _hst_in_fit.any():
                    ok_resid = ok_resid & (
                        _align_init_k | (sig_sq_eff < thresh_admit * hst_fit_sigma_mult))

            new_use = ok_resid & ok_glob_here

            new_use = np.asarray(new_use, dtype=bool)
            # Hard ceilings: never re-admit initial-filter rejects or
            # test-4 influence-flagged detections (ratchet).
            new_use = new_use & self._img_data[img]["use_for_fit_max"]
            infl_excl = self._img_data[img].get("influence_excl")
            if infl_excl is not None:
                new_use = new_use & ~infl_excl
            # Guard against automatic re-admission of stars that were never in
            # alignment or that were removed from it.  A star can participate in
            # use_for_fit only if it started in alignment (align_init=True) OR if
            # it is CURRENTLY in use_for_fit (explicitly admitted, e.g. by
            # V2AlignmentCallback).  This prevents HST-only stars from flooding
            # the alignment tier through test-3 re-admission: once a star is
            # removed from use_for_fit it cannot re-enter via residual tests.
            align_init    = np.asarray(self._img_data[img]["use_for_align_init"], dtype=bool)
            current_fit   = np.asarray(self._img_data[img]["use_for_fit"],        dtype=bool)
            # Phase-6 outliers (real Gaia detections flagged inactive by the
            # crossmatch astrometry fit) can also re-enter alignment if their
            # residuals are acceptable at the current transformation.
            phase6_out    = np.asarray(self._img_data[img].get("phase6_outlier",
                                       np.zeros(len(current_fit), bool)), dtype=bool)
            can_enter_fit = align_init | current_fit | phase6_out
            new_use = new_use & can_enter_fit

            n_use_changed += int(np.sum(current_fit != new_use))

            # Astrometry mask: match use_for_fit for initially-aligned stars.
            # HST-only stars (align_init=False) keep their use_for_astrom unchanged —
            # it is managed externally by V2AlignmentCallback.
            new_use_astrom = np.asarray(self._img_data[img]["use_for_astrom"], dtype=bool).copy()
            new_use_astrom[align_init] = new_use[align_init]
            self._img_data[img]["use_for_astrom"] = new_use_astrom

            if new_use.sum() >= 4:
                chi2_hst = rd_hst["sigma_resid"][new_use]**2
                alpha_raw = float(np.sqrt(np.median(chi2_hst) / _MEDIAN_CHI2_2))
            else:
                alpha_raw = 1.0

            if inflate_errors and iteration >= inflate_from_iter:
                # alpha_raw is measured against the already-inflated C_hst, so it
                # equals alpha_true / alpha_prev.  The cumulative inflation needed
                # relative to C_hst_orig is therefore alpha_prev * alpha_raw.
                # Clamping to 1.0 prevents ever deflating below no-inflation
                # (C_hst is never made smaller than C_hst_orig).
                # alpha_prev is the starting alpha from the previous iteration
                # (or the v1 BP3M starting alpha loaded in run_alignment_v2.py),
                # so a decrease from alpha=2.0 to 2.0*alpha_raw is fully supported
                # as long as the result stays >= 1.0.
                alpha_prev = self._img_data[img].get("alpha_applied", 1.0)
                alpha_j    = float(max(1.0, alpha_prev * alpha_raw))
                self._img_data[img]["alpha_applied"] = alpha_j
                self._img_data[img]["C_hst"] = (
                    alpha_j**2 * self._img_data[img]["C_hst_orig"])
            else:
                # Alpha not yet updated: report alpha_raw for diagnostics but keep
                # alpha_applied (and C_hst) unchanged.
                alpha_j = self._img_data[img].get("alpha_applied", 1.0)

            self._img_data[img]["use_for_fit"] = np.asarray(new_use)
            use_any = new_use | new_use_astrom
            self.gaia_n_hst_used[sidx[use_any]] += 1
            n_astrom_only = int((use_any & ~new_use).sum())
            info.append((img, int(new_use.sum()), len(new_use), alpha_j, alpha_raw, n_astrom_only))

        return info, ok_star, n_use_changed

    def _apply_influence_clip(self, r_hat, C_r, a_arr,
                               cooks_d_thresh=1.0, sigma_min=2.0):
        """
        Test-4: influence-based clipping with ratchet semantics.

        Flags star-image pairs where Cook's D > cooks_d_thresh AND
        sigma_resid > sigma_min that are not already influence-excluded.
        Newly flagged pairs are added to ``_img_data[img]["influence_excl"]``,
        a persistent boolean mask that _update_use_for_fit respects as a hard
        ceiling — flagged pairs are never re-admitted by tests 1-3.

        The ratchet guarantees monotonic convergence: the influence_excl set
        only grows, so the count of *new* flags must eventually reach zero.

        Cook's D_k = (X_k^T Cs_k^{-1} resid_k)^T C_r_j (X_k^T Cs_k^{-1} resid_k) / N_R

        Under the null, E[D_k] = leverage_k / N_R, so D_k > 1 means the
        one-step Newton shift in r_hat from removing this detection exceeds
        one sqrt(C_r) length.  Combined with sigma_resid > sigma_min, this
        targets moderate-outlier / high-leverage detections that pass the
        sigma threshold but disproportionately pull r_hat.

        Returns
        -------
        n_new : int  number of detections *newly* added to influence_excl
        """
        nr = self.N_R
        n_new = 0

        for j_idx, img in enumerate(self.image_names):
            d = self._img_data.get(img)
            if d is None:
                continue
            use = np.asarray(d["use_for_fit"], dtype=bool)
            n   = len(use)
            if use.sum() < 4:
                continue

            # Initialise influence_excl on first call
            if "influence_excl" not in d:
                d["influence_excl"] = np.zeros(n, dtype=bool)
            already_excl = d["influence_excl"]

            cs    = j_idx * nr
            r_j   = r_hat[cs:cs + nr]
            C_r_j = C_r[cs:cs + nr, cs:cs + nr]

            sidx  = d["sidx"]
            X_mat = d["X_mat"]
            JU    = d["JU"]
            xys   = d["xys"]

            Cs     = self._compute_Cs(img, r_j)
            Cs_inv = np.linalg.inv(Cs)

            pred  = (np.einsum('nij,j->ni', X_mat, r_j)
                     - np.einsum('nij,nj->ni', JU, a_arr[sidx]))
            resid = xys - pred
            mah2  = np.einsum('ni,nij,nj->n', resid, Cs_inv, resid)
            sigma_resid = np.sqrt(np.maximum(mah2, 0.))

            CsR   = np.einsum('nij,nj->ni', Cs_inv, resid)
            XtCsR = np.einsum('nij,ni->nj', X_mat, CsR)
            delta_r = XtCsR @ C_r_j
            cooks_d = np.sum(XtCsR * delta_r, axis=1) / nr

            # Only consider pairs currently in use and not already excluded
            new_flag = (use
                        & ~already_excl
                        & (cooks_d > cooks_d_thresh)
                        & (sigma_resid > sigma_min))

            if new_flag.any():
                # Guard: never drop below 4 stars per image
                if (use & ~new_flag).sum() >= 4:
                    d["influence_excl"] = already_excl | new_flag
                    d["use_for_fit"]    = use & ~new_flag
                    n_new += int(new_flag.sum())

        return n_new

    def compute_analytic_posteriors(self, r_hat, C_r, a_arr, K_img, C_vT):
        """Compute exactly marginalised per-star posteriors analytically.

        Analytic counterpart to sample_posteriors.  The conditional stellar mean
        is a linear function of r:

            v_i(r) = a_arr_i + C_vT_i  Σ_j K_{ij}  (r_j − r_hat_j)

        Marginalising over r ~ N(r_hat, C_r) gives:

            v_mean_i  = a_arr_i                    (mean unchanged)
            C_extra_i = (C_vT_i K_i) C_r (C_vT_i K_i)^T

        where K_i = Σ_{detections of star i across all images} K_{ij}
        is the (5, n_r_tot) linear sensitivity of v_i to r.

        Returns
        -------
        v_mean : (n_stars, 5)      same as a_arr (no change in mean)
        v_cov  : (n_stars, 5, 5)  C_extra = C_u − C_vT  (add C_vT to get full C_u)
        """
        nr      = self.N_R
        n_r_tot = nr * self.n_images
        n_stars = self.n_stars

        # Build K_all[i, :, cs:cs+nr] = Σ_{detections of star i in img j} K_{ij}
        K_all = np.zeros((n_stars, 5, n_r_tot))

        for j_idx, img in enumerate(self.image_names):
            if K_img.get(img) is None:
                continue
            d = self._img_data[img]
            use_fit    = d['use_for_fit']
            use_astrom = d.get('use_for_astrom', use_fit)
            use_any    = use_fit | use_astrom
            if not use_any.any():
                continue
            sidx = d['sidx'][use_any]
            K    = K_img[img][use_any]   # (n, 5, nr)
            cs   = j_idx * nr
            np.add.at(K_all[:, :, cs:cs + nr], sidx, K)

        # C_extra[i] = (C_vT[i] @ K_all[i]) @ C_r @ (C_vT[i] @ K_all[i]).T
        CvT_K   = np.einsum('nij,njk->nik', C_vT, K_all)           # (n_stars, 5, n_r_tot)
        C_extra = CvT_K @ C_r @ np.swapaxes(CvT_K, -1, -2)        # (n_stars, 5, 5)

        return a_arr.copy(), C_extra

    def sample_posteriors(self, r_hat, C_r, a_arr, K_img, C_vT,
                          n_samples=1000, seed=42):
        """
        Draw posterior samples of r, propagate to v_T,i samples.
        Returns v_mean (n_stars, 5) and v_cov (n_stars, 5, 5) marginalised over r.
        """
        rng = np.random.default_rng(seed)
        n_r = r_hat.shape[0]

        # Sample r from N(r_hat, C_r)
        try:
            L = np.linalg.cholesky(C_r + 1e-12*np.eye(n_r))
            r_samp = r_hat + (L @ rng.standard_normal((n_r, n_samples))).T
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(C_r)
            vals = np.maximum(vals, 0)
            # (n_r, n_r) @ (n_r, n_r) → scale columns → (n_r, n_samples)
            r_samp = r_hat[:, None] + (vecs * np.sqrt(vals)[None, :]) @ rng.standard_normal((n_r, n_samples))
            r_samp = r_samp.T  # (n_samples, n_r)

        # v_hat(r) = a + B r  where B_{i,j} = -C_vT_i K_{i,j}
        # v_samp[s, i] = a_i - Σ_j (C_vT_i K_{i,j}) r_j^(s)
        v_samp = np.tile(a_arr[None, :, :], (n_samples, 1, 1))  # (n_samp, n_stars, 5)

        nr = self.N_R
        for j_idx, img in enumerate(self.image_names):
            if K_img[img] is None:
                continue
            d    = self._img_data[img]
            use_align  = d["use_for_fit"]
            use_astrom = d.get("use_for_astrom", use_align)
            use_any    = use_align | use_astrom
            sidx = d["sidx"][use_any]

            K     = K_img[img][use_any]
            CvT_K = np.einsum('nij,njk->nik', C_vT[sidx], K)
            cs    = j_idx * nr
            r_j_delta = r_samp[:, cs:cs+nr] - r_hat[cs:cs+nr]
            corr      = np.einsum('sk,njk->snj', r_j_delta, CvT_K)
            v_samp[:, sidx, :] += corr

        v_mean = v_samp.mean(axis=0)                         # (n_stars, 5)
        v_cov  = np.array([np.cov(v_samp[:, i, :].T)
                           for i in range(self.n_stars)])     # (n_stars, 5, 5)

        return r_samp, v_mean, v_cov

    def compute_residuals(self, r_hat, v_hat, C_r=None, C_vT=None):
        """
        Compute per-star, per-image fit residuals and sigma-normalised residuals.

        Parameters
        ----------
        r_hat : (n_r,) MAP image transformation vector
        v_hat : (n_stars, 5) MAP stellar astrometry
        C_r   : (n_r, n_r) posterior covariance of r, optional.
        C_vT  : (n_stars, 5, 5) conditional stellar astrometry covariance, optional.
            When both C_r and C_vT are provided the total uncertainty used for
            sigma-normalisation is the sum of three independent contributions:

                C_total = C_s  +  JU C_vT JU^T  +  X C_r_j X^T

            where C_s is the HST measurement noise, JU C_vT JU^T propagates the
            conditional uncertainty in the fitted stellar astrometry, and
            X C_r_j X^T propagates the image-transformation uncertainty.
            When omitted (default), only C_s is used (HST noise only).

        Returns a dict keyed by image name, each value a dict with:
            'X_c'           : (n,) centered HST x pixel positions (X - Xo)
            'Y_c'           : (n,) centered HST y pixel positions (Y - Yo)
            'resid_x'       : (n,) residual in Gaia pseudo-image x [pixels]
            'resid_y'       : (n,) residual in Gaia pseudo-image y [pixels]
            'sigma_x'       : (n,) 1-σ total noise in pseudo-image x [pixels]
            'sigma_y'       : (n,) 1-σ total noise in pseudo-image y [pixels]
            'sigma_resid_x' : (n,) resid_x / sigma_x  [dimensionless σ]
            'sigma_resid_y' : (n,) resid_y / sigma_y  [dimensionless σ]
            'sigma_resid'   : (n,) 2D Mahalanobis distance sqrt(r^T C_total^{-1} r)
                              (chi distribution with 2 dof under the total noise model)
            'sidx'          : (n,) global star indices
            'use'           : (n,) boolean mask (True = used in fit)

        Residual defined as:
            resid = x_obs - (X r_hat - JU v_hat)
                  = xys - X_mat @ r_hat_j + JU @ v_hat_i
        """
        result = {}
        nr = self.N_R
        for j_idx, img in enumerate(self.image_names):
            meta = self.images[img]

            d = self._img_data.get(img)
            if d is None:
                continue
            cs   = j_idx * nr
            r_j  = r_hat[cs:cs + nr]
            sidx = d["sidx"]
            use  = d["use_for_fit"]

            X_mat = d["X_mat"]   # (n, 2, N_R)
            JU    = d["JU"]      # (n, 2, 5)
            xys   = d["xys"]     # (n, 2)

            # Model prediction: X r_j - JU v_hat_i
            pred = (np.einsum('nij,j->ni', X_mat, r_j)
                    - np.einsum('nij,nj->ni', JU, v_hat[sidx]))   # (n, 2)
            resid = xys - pred  # (n, 2)

            # ── Total uncertainty in pseudo-image space ───────────────────────
            # C_s = J_poly @ C_hst @ J_poly^T  (n, 2, 2) — HST measurement noise
            C_total = self._compute_Cs(img, r_j)   # (n, 2, 2)

            if C_vT is not None:
                # JU @ C_vT_i @ JU^T  — uncertainty from fitted stellar astrometry
                C_total = C_total + np.einsum(
                    'nik,nkl,njl->nij', JU, C_vT[sidx], JU)   # (n, 2, 2)

            if C_r is not None:
                # X @ C_r_j @ X^T  — uncertainty from image transformation
                C_r_j = C_r[cs:cs + nr, cs:cs + nr]            # (N_R, N_R)
                C_total = C_total + np.einsum(
                    'nik,kl,njl->nij', X_mat, C_r_j, X_mat)    # (n, 2, 2)

            sigma_x = np.sqrt(np.maximum(C_total[:, 0, 0], 0.))   # (n,) pix
            sigma_y = np.sqrt(np.maximum(C_total[:, 1, 1], 0.))   # (n,) pix

            sigma_resid_x = np.where(sigma_x > 0, resid[:, 0] / sigma_x, np.nan)
            sigma_resid_y = np.where(sigma_y > 0, resid[:, 1] / sigma_y, np.nan)

            # 2D Mahalanobis distance: sqrt(resid^T C_total^{-1} resid) per star
            C_total_inv = np.linalg.inv(C_total)   # (n, 2, 2)
            mah2        = np.einsum('ni,nij,nj->n', resid, C_total_inv, resid)
            sigma_resid = np.sqrt(np.maximum(mah2, 0.))

            # Recover centered detector positions from cached X_mat
            # X_mat[:,0,0] = X_c (col 0) and X_mat[:,0,1] = Y_c (col 1) by build_X_matrix
            # Row 0: [x, y, 0, 0, 1, 0, ...], Row 1: [0, 0, x, y, 0, 1, ...]
            X_c = X_mat[:, 0, 0]
            Y_c = X_mat[:, 0, 1]

            result[img] = {
                "X_c":           X_c,
                "Y_c":           Y_c,
                "resid_x":       resid[:, 0],
                "resid_y":       resid[:, 1],
                "sigma_x":       sigma_x,
                "sigma_y":       sigma_y,
                "sigma_resid_x": sigma_resid_x,
                "sigma_resid_y": sigma_resid_y,
                "sigma_resid":   sigma_resid,
                "sidx":          sidx,
                "use":           use,
            }
        return result

    def compute_gdc_residuals(self, r_hat, v_hat, C_r=None, C_vT=None):
        """
        Compute per-detection residuals and full covariance in each image's
        local GDC pixel frame.

        The pseudo-image residual (xys - pred) is back-projected through J⁻¹
        to the GDC-corrected HST pixel frame.  The full covariance propagates
        three contributions back to the same frame:

            C_gdc_total = J⁻¹ @ C_total_pseudo @ J⁻¹ᵀ

        where in pseudo-image space:
            C_total_pseudo = C_s  +  JU C_vT JUᵀ  +  X C_r_j Xᵀ

        C_hst (measurement-only, already in GDC frame) is also saved separately.

        Parameters
        ----------
        r_hat  : (n_r,)          MAP image transformation vector
        v_hat  : (n_stars, 5)    MAP stellar astrometry
        C_r    : (n_r, n_r) or None   alignment parameter covariance
        C_vT   : (n_stars, 5, 5) or None   conditional stellar astrometry cov

        Returns a dict keyed by image name.  Each value is a dict with:
            'X_c'           : (n,) centered GDC pixel x  (= X - Xo)
            'Y_c'           : (n,) centered GDC pixel y  (= Y - Yo)
            'dx_gdc'        : (n,) x residual in GDC frame [pixels]
            'dy_gdc'        : (n,) y residual in GDC frame [pixels]
            'C_hst'         : (n, 2, 2) measurement-only covariance in GDC frame
            'C_gdc_total'   : (n, 2, 2) full covariance in GDC frame
                              (= C_hst when C_r and C_vT are both None)
            'sidx'          : (n,) indices into stellar_astrometry rows
            'use_for_fit'   : (n,) bool — used for transformation fitting
            'use_for_astrom': (n,) bool — used for stellar astrometry
        """
        result = {}
        nr = self.N_R
        for j_idx, img in enumerate(self.image_names):
            d = self._img_data.get(img)
            if d is None:
                continue
            cs    = j_idx * nr
            r_j   = r_hat[cs:cs + nr]
            sidx  = d["sidx"]
            X_mat = d["X_mat"]   # (n, 2, N_R)
            JU    = d["JU"]      # (n, 2, 5)
            xys   = d["xys"]     # (n, 2) — Gaia pseudo-image positions

            # Pseudo-image residual: xys - (X r_j - JU v_hat_i)
            pred         = (np.einsum('nij,j->ni', X_mat, r_j)
                            - np.einsum('nij,nj->ni', JU, v_hat[sidx]))
            resid_pseudo = xys - pred    # (n, 2)

            # Total covariance in pseudo-image frame
            C_total = self._compute_Cs(img, r_j)             # (n, 2, 2) = J C_hst Jᵀ
            if C_vT is not None:
                C_total = C_total + np.einsum(
                    'nik,nkl,njl->nij', JU, C_vT[sidx], JU)
            if C_r is not None:
                C_r_j   = C_r[cs:cs + nr, cs:cs + nr]
                C_total = C_total + np.einsum(
                    'nik,kl,njl->nij', X_mat, C_r_j, X_mat)

            # Jacobian and inverse; back-project residual and covariance to GDC frame
            if self.poly_order == 1:
                J     = self.R[img]                          # (2, 2) constant
                J_inv = np.linalg.inv(J)                     # (2, 2)
                dxy   = resid_pseudo @ J_inv.T               # (n, 2)
                # J⁻¹ C_total J⁻¹ᵀ  broadcast over n
                C_gdc = np.einsum('ij,njk,lk->nil',
                                  J_inv, C_total, J_inv)     # (n, 2, 2)
            else:
                J     = compute_poly_jacobian(               # (n, 2, 2)
                    r_j, d["X_c"], d["Y_c"], self.poly_order)
                J_inv = np.linalg.inv(J)                     # (n, 2, 2)
                dxy   = np.einsum('nij,nj->ni', J_inv, resid_pseudo)  # (n, 2)
                C_gdc = np.einsum('nij,njk,nlk->nil',
                                  J_inv, C_total, J_inv)     # (n, 2, 2)

            result[img] = {
                "X_c":            d["X_c"],
                "Y_c":            d["Y_c"],
                "dx_gdc":         dxy[:, 0],
                "dy_gdc":         dxy[:, 1],
                "C_hst":          d["C_hst"],
                "C_gdc_total":    C_gdc,
                "sidx":           sidx,
                "use_for_fit":    np.asarray(d["use_for_fit"],  dtype=bool),
                "use_for_astrom": np.asarray(
                    d.get("use_for_astrom", d["use_for_fit"]), dtype=bool),
            }
        return result

    def compute_star_influence(self, r_hat, C_r, a_arr):
        """
        Compute per-star, per-image leverage, influence, and Cook's distance.

        Uses the one-step Newton approximation: if star k were removed from
        image j, r_hat_j would shift by approximately δr = C_r_j @ X_k^T @ Cs_inv_k @ resid_k.

        Parameters
        ----------
        r_hat : (n_r,) converged image parameter vector
        C_r   : (n_r, n_r) posterior covariance of r
        a_arr : (n_stars, 5) converged stellar astrometry (= v_hat)

        Returns
        -------
        pd.DataFrame with one row per (star, image) detection, columns:
            Gaia_id, image_name,
            X_c, Y_c          — centred detector pixel coordinates
            mag               — HST magnitude
            resid_x, resid_y  — pixel residuals
            sigma_resid       — 2D Mahalanobis distance (resid/noise)
            leverage          — hat-matrix trace (0–2; >1 is high leverage)
            infl_a … infl_z   — influence on each image parameter (pixels)
            cooks_d           — Cook's distance analog
            use_for_fit       — was this detection included in the fit?
        """
        import pandas as pd

        nr = self.N_R
        param_names = ['a', 'b', 'c', 'd', 'w', 'z',
                       'da0', 'dd0'][:nr]

        rows = []
        for j_idx, img in enumerate(self.image_names):
            d = self._img_data.get(img)
            if d is None:
                continue

            cs    = j_idx * nr
            r_j   = r_hat[cs:cs + nr]
            C_r_j = C_r[cs:cs + nr, cs:cs + nr]

            sidx  = d["sidx"]          # (n,) global star indices
            X_mat = d["X_mat"]         # (n, 2, N_R)
            JU    = d["JU"]            # (n, 2, 5)
            xys   = d["xys"]           # (n, 2)
            use        = d["use_for_fit"]                         # (n,) bool — alignment
            use_astrom = d.get("use_for_astrom", use)           # (n,) bool — astrometry

            # HST measurement noise covariance and precision
            Cs     = self._compute_Cs(img, r_j)   # (n, 2, 2)
            Cs_inv = np.linalg.inv(Cs)            # (n, 2, 2)

            # Residual using same sign convention as compute_residuals:
            #   pred = X r_j - JU v_hat
            #   resid = xys - pred = xys - X r_j + JU a_arr
            pred  = (np.einsum('nij,j->ni', X_mat, r_j)
                     - np.einsum('nij,nj->ni', JU, a_arr[sidx]))   # (n, 2)
            resid = xys - pred   # (n, 2)

            # Mahalanobis distance (HST noise only)
            mah2      = np.einsum('ni,nij,nj->n', resid, Cs_inv, resid)
            sigma_res = np.sqrt(np.maximum(mah2, 0.))

            # ── Influence quantities ─────────────────────────────────────────
            # XtCsR_k = X_k^T Cs_inv_k resid_k  (N_R,) per star
            CsR   = np.einsum('nij,nj->ni', Cs_inv, resid)   # (n, 2)
            XtCsR = np.einsum('nij,ni->nj', X_mat, CsR)      # (n, N_R)

            # delta_r_k = C_r_j @ XtCsR_k  (N_R,) per star
            # C_r_j is symmetric so C_r_j.T = C_r_j
            delta_r = XtCsR @ C_r_j   # (n, N_R)

            # Cook's distance: XtCsR_k . delta_r_k / N_R
            #   = delta_r^T C_r_j^{-1} delta_r / N_R  (since delta_r = C_r_j XtCsR)
            cooks_d = np.sum(XtCsR * delta_r, axis=1) / nr   # (n,)

            # Leverage: tr(Cs_inv_k @ X_k @ C_r_j @ X_k^T)
            XCrX    = np.einsum('nik,kl,njl->nij', X_mat, C_r_j, X_mat)   # (n, 2, 2)
            leverage = np.einsum('nij,nji->n', Cs_inv, XCrX)              # (n,)

            # Detector coordinates from cached X_mat
            X_c = X_mat[:, 0, 0]
            Y_c = X_mat[:, 0, 1]

            # Magnitude from stars_per_image
            spi = self.stars_per_image.get(img)
            if spi is not None and "mag" in spi.columns:
                mag_vals = spi["mag"].values
            else:
                mag_vals = np.full(len(sidx), np.nan)

            gaia_ids = self.gaia_cat["Gaia_id"].iloc[sidx].values

            for n in range(len(sidx)):
                row = dict(
                    Gaia_id    = int(gaia_ids[n]),
                    image_name = img,
                    X_c        = float(X_c[n]),
                    Y_c        = float(Y_c[n]),
                    mag        = float(mag_vals[n]) if n < len(mag_vals) else np.nan,
                    resid_x    = float(resid[n, 0]),
                    resid_y    = float(resid[n, 1]),
                    sigma_resid= float(sigma_res[n]),
                    leverage   = float(leverage[n]),
                    cooks_d    = float(cooks_d[n]),
                    use_for_fit   = bool(use[n]),
                    use_for_astrom= bool(use_astrom[n]),
                )
                for p_idx, pname in enumerate(param_names):
                    row[f"infl_{pname}"] = float(delta_r[n, p_idx])
                rows.append(row)

        return pd.DataFrame(rows)
