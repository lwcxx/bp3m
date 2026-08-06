"""
cross_match_validator.py - Cross-image validation of Gaia-HST/JWST cross-matches.

For each target, groups processed images by filter/camera, then:

  1. Writes per-image source_quality.csv annotating each matched source with:
       mag_normalized     = mag_st_gdc + cross_image_zp  (comparable across images;
                             hst_mag_st_gdc for HST, jwst_mag_st_gdc for JWST)
       n_same_filter      = number of same-filter/camera images also matching this Gaia source
       mag_norm_mad       = MAD of mag_normalized across those images
       mag_residual       = deviation from cross-image median
       is_mag_consistent  = mag_norm_mad < threshold
       expected/observed inter-image magnitude delta vs reference image
       wcs_offset_px      = pointing change from WCS headers (image-level constant)
       is_trustworthy     = combined flag

  2. Writes a per-target cross_match_catalog.csv with one row per
     (gaia_source_id, filter_camera) pair:
       gaia_source_id, filter_camera, n_images, image_list, hst_index_list/jwst_index_list,
       mag_norm_mean, mag_norm_std, mag_norm_mad, is_consistent

Usage:
    conda activate pymc_new
    python cross_match_validator.py --target Fornax_dSph --data-dir ./data --telescope HST
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
from collections import defaultdict


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _science_filter(h0, telescope='HST'):
    if telescope == 'JWST':
        instrume = h0.get('INSTRUME', '').strip().upper()
        filt  = h0.get('FILTER', '').strip().upper()
        pupil = h0.get('PUPIL',  '').strip().upper()
        if instrume == 'NIRISS' and filt == 'CLEAR':
            return pupil
        return filt

    # WFC3 (and other single-wheel instruments) use a single FILTER keyword.
    single = h0.get('FILTER', '').strip()
    if single:
        return single
    # ACS uses two filter wheels (FILTER1/FILTER2); return the non-CLEAR one.
    f1 = h0.get('FILTER1', 'CLEAR').strip()
    f2 = h0.get('FILTER2', 'CLEAR').strip()
    return f2 if 'CLEAR' in f1 else f1


def load_image_data(image_dir, image_name, telescope='HST'):
    matched_path   = os.path.join(image_dir, 'matched_gaia.csv')
    transform_path = os.path.join(image_dir, 'transformation.csv')
    if telescope == 'JWST':
        image_paths = glob.glob(os.path.join(image_dir, '*_cal.fits'))
        stmag_col   = 'jwst_mag_st_gdc'
    else:
        image_paths = glob.glob(os.path.join(image_dir, '*_flc.fits'))
        stmag_col   = 'hst_mag_st_gdc'
    if not (os.path.exists(matched_path) and
            os.path.exists(transform_path) and image_paths):
        return None

    matched   = pd.read_csv(matched_path)
    transform = pd.read_csv(transform_path, index_col='parameter')['value']

    with fits.open(image_paths[0]) as h:
        h0 = h[0].header
        h1 = h[1].header
        exptime  = float(h0.get('EXPTIME', 1.0))
        filt     = _science_filter(h0, telescope=telescope)
        instrume = h0.get('INSTRUME', '').strip()
        detector = h0.get('DETECTOR', '').strip()
        crval1   = float(h1.get('CRVAL1', 0.0))
        crval2   = float(h1.get('CRVAL2', 0.0))

    has_stmag = (stmag_col in matched.columns and
                 matched[stmag_col].notna().any())

    return {
        'image_name':    image_name,
        'image_dir':     image_dir,
        'matched':       matched,
        'transform':     transform,
        'exptime':       exptime,
        'filter':        filt,
        'instrume':      instrume,
        'detector':      detector,
        'camera':        f'{instrume}/{detector}',
        'filter_camera': f'{filt}/{instrume}/{detector}',
        'has_stmag':     has_stmag,
        'zp':            float(transform['zp']),
        'crval1':        crval1,
        'crval2':        crval2,
        'pixel_scale':   float(transform.get('pixel_scale', 0.05)),
        'dec_cen':       float(transform['dec_cen']),
    }


def find_processed_images(target, data_dir, telescope='HST'):
    if telescope == 'JWST':
        image_root, suffix = os.path.join(data_dir, target, 'JWST'), '_cal'
    else:
        image_root, suffix = os.path.join(data_dir, target, 'HST'), '_flc'
    images = {}
    for root, dirs, files in os.walk(image_root):
        name = os.path.basename(root)
        if f'{name}{suffix}_catalog.fits' in files and 'matched_gaia.csv' in files:
            data = load_image_data(root, name, telescope=telescope)
            if data is not None:
                images[name] = data
    return images


# ---------------------------------------------------------------------------
# Magnitude helpers
# ---------------------------------------------------------------------------

def has_valid_stmag(matched_df, telescope='HST'):
    stmag_col = 'jwst_mag_st_gdc' if telescope == 'JWST' else 'hst_mag_st_gdc'
    return (stmag_col in matched_df.columns and
            matched_df[stmag_col].notna().any())


def compute_pairwise_zps(group, telescope='HST'):
    """
    For every pair (a, b) of images in `group` that share ≥ 3 sources, compute:

        ZP(a→b) = median( mag_st_gdc_a_j − mag_st_gdc_b_j )
                  for Gaia sources j detected in both images.

    The ZP is always computed from direct per-star differences — never from
    population medians and never via Gaia magnitudes as an intermediary.
    Both directions are stored so BFS traversal is straightforward.

    Returns dict {(name_a, name_b): (zp, n_shared)} for pairs with n_shared ≥ 3,
    with both (a,b) and (b,a) entries (ZPs negated for the reverse direction).
    """
    stmag_col = 'jwst_mag_st_gdc' if telescope == 'JWST' else 'hst_mag_st_gdc'
    names = list(group.keys())
    result = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            df_a = (group[a]['matched']
                    [['gaia_source_id', stmag_col]]
                    .rename(columns={stmag_col: 'mag_a'}))
            df_b = (group[b]['matched']
                    [['gaia_source_id', stmag_col]]
                    .rename(columns={stmag_col: 'mag_b'}))
            shared = df_a.merge(df_b, on='gaia_source_id')
            ok = (np.isfinite(shared['mag_a'].values) &
                  np.isfinite(shared['mag_b'].values))
            n_ok = int(ok.sum())
            if n_ok >= 3:
                zp = float(np.median(
                    shared['mag_a'].values[ok] - shared['mag_b'].values[ok]))
                result[(a, b)] = ( zp, n_ok)
                result[(b, a)] = (-zp, n_ok)
    return result


def find_overlap_components(names, pairwise_zps):
    """
    Find connected components of the image overlap graph.
    Two images are connected if they share ≥ 3 sources (i.e. have an entry in
    pairwise_zps).  Images with no overlap with anyone are their own component
    (solo).

    Returns a list of sets of image names.
    """
    adj = defaultdict(set)
    for (a, b) in pairwise_zps:
        adj[a].add(b)

    visited = set()
    components = []
    for name in names:
        if name not in visited:
            component = set()
            queue = [name]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                queue.extend(adj[node] - visited)
            components.append(component)
    return components


def propagate_zps_bfs(component, pairwise_zps, ref_name):
    """
    BFS from ref_name through the overlap graph, accumulating ZPs:

        ZP_neighbor = ZP_current + pairwise_zps[(current, neighbor)][0]

    so that mag_norm_i = mag_st_gdc_i + ZP_i ≈ mag_st_gdc_ref for all i.
    ref_name always has ZP = 0.0.

    Returns dict {name: zp} for every image reachable from ref_name.
    """
    zps = {ref_name: 0.0}
    queue = [ref_name]
    while queue:
        current = queue.pop(0)
        for neighbor in component:
            if neighbor not in zps and (current, neighbor) in pairwise_zps:
                zp_edge, _ = pairwise_zps[(current, neighbor)]
                zps[neighbor] = zps[current] + zp_edge
                queue.append(neighbor)
    return zps


def mad(x):
    return float(np.median(np.abs(x - np.median(x))))


def weighted_mean_and_err(mags, errs):
    """Inverse-variance weighted mean and its uncertainty."""
    w = 1.0 / errs**2
    mu  = np.sum(w * mags) / np.sum(w)
    sig = 1.0 / np.sqrt(np.sum(w))
    return mu, sig


# ---------------------------------------------------------------------------
# Pointing offset from WCS headers
# ---------------------------------------------------------------------------

def wcs_offset_px(d_i, d_ref):
    """
    First-order pixel offset of image i's pointing relative to reference,
    from CRVAL differences.  Sign convention: +RA → −X (East-Left).
    Returns scalar distance in pixels.
    """
    cos_dec = np.cos(np.radians(d_ref['dec_cen']))
    pix     = d_ref['pixel_scale']
    dx = -(d_i['crval1'] - d_ref['crval1']) * 3600.0 * cos_dec / pix
    dy =  (d_i['crval2'] - d_ref['crval2']) * 3600.0 / pix
    return float(np.hypot(dx, dy))


# ---------------------------------------------------------------------------
# Per-group validation
# ---------------------------------------------------------------------------

def validate_filter_group(group, ref_name, zp_dict, mag_scatter_thr, offset_tol_mag,
                          offset_tol_px, cross_camera_extra_tol=0.05, z_outlier=3.0,
                          telescope='HST'):
    """
    Validate a set of images forming one overlap-connected component.

    zp_dict: {name: zp} pre-computed via propagate_zps_bfs so that
             mag_norm_i = mag_st_gdc_i + zp_i ≈ mag_st_gdc_ref.
             ref_name has zp = 0.0 by construction.

    All images in `group` must have a valid stmag column (caller guarantees
    this). ZPs are never assumed to be 0; they are always derived from direct
    per-star differences along a spanning tree of the overlap graph.
    """
    if telescope == 'JWST':
        stmag_col, mag_err_col = 'jwst_mag_st_gdc', 'jwst_mag_err_gdc'
    else:
        stmag_col, mag_err_col = 'hst_mag_st_gdc', 'hst_mag_err_gdc'

    ref = group[ref_name]

    for name, d in group.items():
        zp = zp_dict[name]
        d['mag_norm']   = d['matched'][stmag_col].values + zp
        d['cross_zp']   = zp
        d['used_stmag'] = True

    # --- Cross-image source magnitude table ---
    # Error for mag_st_gdc equals mag_err_gdc (STMAG is a linear flux
    # scaling, so fractional errors are identical).  Add a 0.01 mag floor so
    # that near-perfect formal errors don't make the pull statistic too
    # aggressive.
    has_err = all(mag_err_col in d['matched'].columns for d in group.values())
    rows = []
    for name, d in group.items():
        tmp = d['matched'][['gaia_source_id']].copy()
        tmp['mag_norm'] = d['mag_norm']
        tmp['image']    = name
        if has_err:
            tmp['mag_err'] = d['matched'][mag_err_col].values
        rows.append(tmp)
    combined = pd.concat(rows, ignore_index=True)

    def per_source_stats(g):
        mags = g['mag_norm'].values
        imgs = g['image'].values
        if has_err:
            errs = np.clip(g['mag_err'].values, 1e-4, None) + 0.01
            mu_w, sig_w = weighted_mean_and_err(mags, errs)
            pulls = (mags - mu_w) / errs
        else:
            mu_w  = np.median(mags)
            sig_w = mad(mags)
            pulls = (mags - mu_w) / (sig_w if sig_w > 0 else 1.0)

        outlier_mask = np.abs(pulls) > z_outlier
        n_consistent = int((~outlier_mask).sum())
        outlier_imgs = ','.join(sorted(imgs[outlier_mask]))
        return pd.Series({
            'n_same_filter':   len(mags),
            'mag_norm_median': float(np.median(mags)),
            'mag_norm_mad':    mad(mags),
            'mag_norm_wmean':  float(mu_w),
            'mag_norm_werr':   float(sig_w),
            'n_consistent':    n_consistent,
            'outlier_images':  outlier_imgs,
        })

    source_stats = (combined
        .groupby('gaia_source_id')
        .apply(per_source_stats)
        .reset_index())

    # --- Per-image stats vs reference ---
    image_stats = {}
    for name, d in group.items():
        zp = zp_dict[name]
        same_camera = (d['camera'] == ref['camera'])
        tol = offset_tol_mag if same_camera else offset_tol_mag + cross_camera_extra_tol

        ref_df  = ref['matched'][['gaia_source_id']].assign(mag_ref=ref['mag_norm'])
        this_df = d['matched'][['gaia_source_id']].assign(mag_this=d['mag_norm'])
        shared  = ref_df.merge(this_df, on='gaia_source_id')
        ok      = (np.isfinite(shared['mag_ref'].values) &
                   np.isfinite(shared['mag_this'].values))
        n_shared = int(ok.sum())
        if n_shared >= 3:
            residual_after_zp = float(np.median(
                shared['mag_ref'].values[ok] - shared['mag_this'].values[ok]))
            offset_mag_ok = abs(residual_after_zp) < tol
        else:
            residual_after_zp = np.nan
            offset_mag_ok     = True

        image_stats[name] = {
            'ref_image':          ref_name,
            'mag_scale':          'mag_st_gdc+CrossZP',
            'same_camera_as_ref': same_camera,
            'used_stmag':         True,
            'n_shared_with_ref':  n_shared,
            'cross_image_zp':     zp,
            'residual_after_zp':  residual_after_zp,
            'offset_mag_ok':      offset_mag_ok,
            'wcs_offset_px':      wcs_offset_px(d, ref),
        }

    return source_stats, image_stats


# ---------------------------------------------------------------------------
# Per-image output
# ---------------------------------------------------------------------------

def write_source_quality(data, source_stats, image_stats, mag_scatter_thr):
    df = data['matched'].copy()
    df['mag_normalized'] = data['mag_norm']

    df = df.merge(source_stats, on='gaia_source_id', how='left')
    df['mag_residual_from_wmean'] = df['mag_normalized'] - df['mag_norm_wmean']

    # This image is consistent if it is not in the outlier list for this source
    name = data['image_name']
    df['is_mag_consistent'] = ~df['outlier_images'].fillna('').str.contains(
        name, regex=False)
    df.loc[df['n_same_filter'] == 1, 'is_mag_consistent'] = True  # solo: can't assess

    s = image_stats[data['image_name']]
    for col, val in s.items():
        df[col] = val

    # pointing_ok is informational only; WCS offset alone doesn't make a
    # cross-match untrustworthy — dithered images can be 50+ px apart.
    df['is_trustworthy'] = df['is_mag_consistent'] & df['offset_mag_ok']

    out = os.path.join(data['image_dir'], 'source_quality.csv')
    df.to_csv(out, index=False)
    return out


def write_solo_quality(data, telescope='HST'):
    if telescope == 'JWST':
        stmag_col, legacy_tool = 'jwst_mag_st_gdc', 'jwst1pass'
    else:
        stmag_col, legacy_tool = 'hst_mag_st_gdc', 'py1pass'

    df = data['matched'].copy()
    if not has_valid_stmag(data['matched'], telescope=telescope):
        print(f"  WARNING: {data['image_name']} missing {stmag_col} "
              f"(stale {legacy_tool} output?) — solo image skipped")
        return None
    mag_norm = data['matched'][stmag_col].values.copy()
    df['mag_normalized']    = mag_norm
    df['n_same_filter']     = 1
    df['mag_norm_median']   = df['mag_normalized']
    df['mag_norm_mad']      = np.nan
    df['is_mag_consistent'] = True
    df['ref_image']         = data['image_name']
    df['mag_scale']         = 'STMAG'
    df['n_shared_with_ref'] = len(df)
    df['cross_image_zp']    = 0.0
    df['residual_after_zp'] = 0.0
    df['wcs_offset_px']     = 0.0
    df['same_camera_as_ref'] = True
    df['used_stmag']        = True
    df['offset_mag_ok']     = True
    df['is_trustworthy']    = True

    out = os.path.join(data['image_dir'], 'source_quality.csv')
    df.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Global target-level catalog
# ---------------------------------------------------------------------------

def build_global_catalog(images, target, data_dir, telescope='HST'):
    """
    Aggregate all source_quality.csv files into a single cross_match_catalog.csv
    at the target level.  One row per (gaia_source_id, filter_camera).
    """
    if telescope == 'JWST':
        index_col, star_col = 'jwst_index', 'is_star'
    else:
        index_col, star_col = 'hst_index', 'hst_is_star'
    index_list_key = f'{index_col}_list'

    rows = []
    for name, d in images.items():
        sq_path = os.path.join(d['image_dir'], 'source_quality.csv')
        if not os.path.exists(sq_path):
            continue
        sq = pd.read_csv(sq_path)
        sq['image_name']    = name
        sq['filter_camera'] = d['filter_camera']
        cols = ['gaia_source_id', index_col, 'image_name', 'filter_camera',
                'mag_normalized', 'mag_norm_mad', 'mag_norm_wmean', 'mag_norm_werr',
                'n_consistent', 'outlier_images', 'is_trustworthy',
                'has_gaia_pms', 'gaia_gmag', 'residual_sigma', star_col]
        rows.append(sq[[c for c in cols if c in sq.columns]])

    if not rows:
        return

    all_data = pd.concat(rows, ignore_index=True)

    # Per (gaia_source_id, filter_camera) aggregation
    def agg_group(g):
        mags   = g['mag_normalized'].values
        trusts = g['is_trustworthy'].values
        # Weighted mean/err from the per-image source_quality files (same value
        # repeated per image, so just take the first non-null entry)
        wmean = g['mag_norm_wmean'].dropna().iloc[0] if 'mag_norm_wmean' in g and g['mag_norm_wmean'].notna().any() else float(np.mean(mags))
        werr  = g['mag_norm_werr'].dropna().iloc[0]  if 'mag_norm_werr'  in g and g['mag_norm_werr'].notna().any()  else np.nan
        n_con = int(g['n_consistent'].dropna().iloc[0]) if 'n_consistent' in g and g['n_consistent'].notna().any() else len(g)
        out_imgs = g['outlier_images'].dropna().iloc[0] if 'outlier_images' in g and g['outlier_images'].notna().any() else ''

        # is_star aggregation across images
        if star_col in g.columns:
            star_vals = g[star_col].astype(bool)
            is_star_all = bool(star_vals.all())
            is_star_any = bool(star_vals.any())
            non_star_imgs = ','.join(sorted(g.loc[~star_vals, 'image_name'].tolist()))
        else:
            is_star_all = np.nan
            is_star_any = np.nan
            non_star_imgs = ''

        return pd.Series({
            'n_images':              len(g),
            'image_list':            ','.join(g['image_name'].tolist()),
            index_list_key:          ','.join(g[index_col].astype(str).tolist()),
            'mag_norm_wmean':        float(wmean),
            'mag_norm_werr':         float(werr) if not np.isnan(werr) else np.nan,
            'mag_norm_mad':          float(g['mag_norm_mad'].median()),
            'n_consistent':          n_con,
            'outlier_images':        out_imgs,
            'n_trustworthy':         int(trusts.sum()),
            'all_trustworthy':       bool(trusts.all()),
            'any_trustworthy':       bool(trusts.any()),
            'has_gaia_pms':          bool(g['has_gaia_pms'].any()),
            'gaia_gmag':             float(g['gaia_gmag'].iloc[0]),
            'median_residual_sigma': float(g['residual_sigma'].median()),
            'is_star_all_images':    is_star_all,
            'is_star_any_image':     is_star_any,
            'non_star_images':       non_star_imgs,
        })

    catalog = (all_data
               .groupby(['gaia_source_id', 'filter_camera'])
               .apply(agg_group)
               .reset_index())

    out = os.path.join(data_dir, target, 'cross_match_catalog.csv')
    catalog.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_target(target, data_dir, mag_scatter_thr=0.1,
                    offset_tol_mag=0.05, offset_tol_px=10.0, telescope='HST'):
    if telescope == 'JWST':
        stmag_col, legacy_tool = 'jwst_mag_st_gdc', 'jwst1pass'
    else:
        stmag_col, legacy_tool = 'hst_mag_st_gdc', 'py1pass'

    images = find_processed_images(target, data_dir, telescope=telescope)
    if not images:
        print(f'No processed images found for {target}'); return

    print(f'{target}: {len(images)} processed images')

    # Group by filter only — overlap-connectivity within each filter is
    # determined from the pairwise ZP graph, not assumed from the camera/detector.
    # Two images with the same filter but different sky positions will end up in
    # separate connected components and be validated independently.
    by_filter = defaultdict(list)
    for name, d in images.items():
        by_filter[d['filter']].append(name)

    all_image_stats = {}

    for filt, names in sorted(by_filter.items()):
        # Drop images without a valid stmag column before building the graph.
        missing = [n for n in names if not has_valid_stmag(images[n]['matched'], telescope=telescope)]
        if missing:
            print(f'\n  [{filt}] WARNING: {len(missing)} image(s) missing '
                  f'{stmag_col} (stale {legacy_tool}?) — skipped: {missing}')
        valid = [n for n in names if n not in missing]
        if not valid:
            continue

        # Pairwise ZPs between all images in this filter that share ≥ 3 sources.
        group_all = {n: images[n] for n in valid}
        pairwise  = compute_pairwise_zps(group_all, telescope=telescope)

        # Connected components: each is an independently calibratable set of images.
        components = find_overlap_components(valid, pairwise)
        cameras_all = sorted({images[n]['camera'] for n in valid})
        print(f'\n  [{filt}]  {len(valid)} image(s)  cameras: {", ".join(cameras_all)}'
              f'  →  {len(components)} overlap component(s)')

        for comp in components:
            comp_names = sorted(comp)

            if len(comp) == 1:
                name = comp_names[0]
                out  = write_solo_quality(images[name], telescope=telescope)
                if out is None:
                    continue
                n_total = len(images[name]['matched'])
                all_image_stats[name] = {
                    'filter':            filt,
                    'ref_image':         name,
                    'mag_scale':         'STMAG',
                    'cross_image_zp':    0.0,
                    'n_shared_with_ref': n_total,
                    'residual_after_zp': 0.0,
                    'wcs_offset_px':     0.0,
                    'n_trustworthy':     n_total,
                    'n_total':           n_total,
                }
                print(f'    {name}: solo → {out}')
                continue

            ref_name = max(comp, key=lambda n: len(images[n]['matched']))
            zp_dict  = propagate_zps_bfs(comp, pairwise, ref_name)
            group    = {n: images[n] for n in comp}

            source_stats, image_stats = validate_filter_group(
                group, ref_name, zp_dict, mag_scatter_thr, offset_tol_mag, offset_tol_px,
                telescope=telescope)

            for name in comp_names:
                out = write_source_quality(images[name], source_stats, image_stats,
                                           mag_scatter_thr)
                sq = pd.read_csv(out)
                n_trust = int(sq['is_trustworthy'].sum())
                n_total = len(sq)
                s = image_stats[name]
                cam_tag = '' if s['same_camera_as_ref'] else ' [x-cam]'
                zp_str  = f'{s["cross_image_zp"]:+.3f}'
                res_str = (f'{s["residual_after_zp"]:+.3f}'
                           if not np.isnan(s['residual_after_zp']) else 'N/A')
                print(f'    {name}{cam_tag} [{s["mag_scale"]}]: '
                      f'{n_trust}/{n_total} trustworthy | '
                      f'ZP={zp_str} residual={res_str} | '
                      f'WCS={s["wcs_offset_px"]:.1f}px')
                all_image_stats[name] = {
                    'filter': filt,
                    **s,
                    'n_trustworthy': n_trust,
                    'n_total':       n_total,
                }

    # Write ZP offset table: one row per image, showing which reference was used
    # and what ZP offset was applied.  Rows where cross_image_zp == 0.0 and
    # ref_image == image_name are the per-filter photometric anchors.
    if all_image_stats:
        zp_rows = []
        for name, s in all_image_stats.items():
            zp_rows.append({
                'image':               name,
                'filter':              s.get('filter', ''),
                'ref_image':           s.get('ref_image', ''),
                'mag_scale':           s.get('mag_scale', ''),
                'cross_image_zp': s.get('cross_image_zp', np.nan),
                'n_shared_with_ref':   s.get('n_shared_with_ref', 0),
                'residual_after_zp':   s.get('residual_after_zp', np.nan),
                'wcs_offset_px':       s.get('wcs_offset_px', np.nan),
                'n_trustworthy':       s.get('n_trustworthy', 0),
                'n_total':             s.get('n_total', 0),
            })
        zp_df = pd.DataFrame(zp_rows).sort_values(['filter', 'ref_image', 'image'])
        zp_path = os.path.join(data_dir, target, 'magnitude_zp_offsets.csv')
        zp_df.to_csv(zp_path, index=False)
        print(f'\n  ZP offsets: {len(zp_df)} images → {zp_path}')

    # Global catalog
    out = build_global_catalog(images, target, data_dir, telescope=telescope)
    if out:
        cat = pd.read_csv(out)
        n_sources = len(cat)
        n_multi   = int((cat['n_images'] > 1).sum())
        print(f'  Global catalog: {n_sources} source/filter entries '
              f'({n_multi} seen in >1 image) → {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cross-image validation of Gaia-HST/JWST cross-matches.')
    parser.add_argument('--target', required=True)
    parser.add_argument('--telescope', choices=['HST', 'JWST'], default='HST',
                        help='Telescope whose cross-matches to validate. Default: HST')
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--mag-scatter-threshold', type=float, default=0.1,
                        help='MAD threshold for magnitude consistency flag. Default: 0.1 mag')
    parser.add_argument('--offset-tolerance-mag', type=float, default=0.05,
                        help='Tolerance for expected vs observed inter-image ZP offset. Default: 0.05 mag')
    parser.add_argument('--offset-tolerance-px', type=float, default=10.0,
                        help='WCS pointing offset tolerance (small dithers only). Default: 10 px')
    args = parser.parse_args()

    validate_target(args.target, args.data_dir,
                    mag_scatter_thr=args.mag_scatter_threshold,
                    offset_tol_mag=args.offset_tolerance_mag,
                    offset_tol_px=args.offset_tolerance_px,
                    telescope=args.telescope)
