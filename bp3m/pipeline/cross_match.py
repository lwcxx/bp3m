"""
Step 4: Cross-match HST PSF catalogs against Gaia using gaia_cross_match.

For each image directory that contains {obs_id}_flc_catalog.fits, calls
gaia_cross_match.cross_match.process_single_image and writes:
    matched_gaia.csv        — HST↔Gaia matched pairs
    transformation.csv      — affine transformation parameters
    diagnostic_plots.png    — 8-panel diagnostic figure
    offset_histogram.png    — 2D offset histogram from discovery step

Gaia data is loaded from {output_dir}/{field}/Gaia/*.csv.

Extension note
--------------
JWST cross-matching will use the same gaia_cross_match code once it handles
JWST-specific pixel scales and FITS header conventions. Pass telescope='JWST'
once that support is in place.
"""

from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd


def _find_image_folders(output_dir: Path, field_name: str,
                         telescope: str = 'HST', im_type: str = '_flc') -> list[dict]:
    """
    Return list of dicts with keys {root, catalog, flc} for each image that
    has both an FLC FITS file and a matching _catalog.fits.
    """
    root = (Path(output_dir) / field_name / telescope.upper()
            / "mastDownload" / telescope.upper())
    folders = []
    if not root.exists():
        return folders
    for obs_dir in sorted(root.iterdir()):
        if not obs_dir.is_dir():
            continue
        name   = obs_dir.name
        suffix = f"{im_type}.fits"
        flc    = obs_dir / f"{name}{suffix}"
        cat    = obs_dir / f"{name}{suffix.replace('.fits', '_catalog.fits')}"
        if flc.exists() and cat.exists():
            folders.append({'root': str(obs_dir), 'catalog': str(cat), 'flc': str(flc)})
    return folders


def _match_one(args):
    """Worker: cross-match one image. Returns (image_name, n_matched, error)."""
    hst_dict, gaia_df, kwargs = args
    from gaia_cross_match.cross_match import process_single_image

    name = Path(hst_dict['root']).name
    params_meta = kwargs.get('params_meta', {})
    params_path = Path(hst_dict['root']) / "xmatch_params.json"
    try:
        # Remove sidecar BEFORE running so an interrupted match leaves no stale
        # sidecar that could mark incomplete results as valid on the next run.
        if params_path.exists():
            params_path.unlink()

        process_single_image(
            hst_dict, gaia_df,
            hst_pix_floor=kwargs.get('hst_pix_floor', 0.05),
            min_matches=kwargs.get('min_matches', 3),
            zero_pm=kwargs.get('zero_pm', False),
            max_mag_diff=kwargs.get('max_mag_diff', 3.0),
            scale_sweep=kwargs.get('scale_sweep', False),
            discovery_max_offset=kwargs.get('discovery_max_offset', 50),
            use_resid_floor=kwargs.get('use_resid_floor', True),
        )
        out = Path(hst_dict['root']) / "matched_gaia.csv"
        n = len(pd.read_csv(out)) if out.exists() else 0
        if params_meta and out.exists():
            params_path.write_text(json.dumps(params_meta, indent=2))
        return name, n, None
    except Exception as exc:
        return name, 0, str(exc)


def _params_cache_status(output_path: Path, params_path: Path,
                          current_params: dict) -> tuple[bool, list[str]]:
    """Return (cache_valid, diffs) by comparing saved params sidecar to current."""
    if not output_path.exists():
        return False, []
    if not params_path.exists():
        return False, ["no params sidecar — cannot verify configuration match"]
    try:
        saved = json.loads(params_path.read_text())
    except Exception as e:
        return False, [f"could not read params sidecar: {e}"]
    diffs = [f"  {k}: saved={saved.get(k)!r}  current={v!r}"
             for k, v in current_params.items() if saved.get(k) != v]
    return len(diffs) == 0, diffs


def run_cross_match(
    output_dir: Path,
    field_name: str,
    telescope: str = 'HST',
    im_type: str = '_flc',
    n_processes: int = 4,
    hst_pix_floor: float = 0.05,
    min_matches: int = 3,
    zero_pm: bool = False,
    max_mag_diff: float = 3.0,
    scale_sweep: bool = False,
    discovery_max_offset: int = 50,
    use_resid_floor: bool = True,
    force_rematch: bool = False,
    image_id: str | None = None,
    restrict_to_obsids: list[str] | None = None,
) -> list[Path]:
    """
    Cross-match all PSF-fit HST catalogs in a field against Gaia.

    Cached results are always reused when the saved cross-match parameters match
    the current call.  Pass ``force_rematch=True`` to re-match regardless.

    Parameters
    ----------
    output_dir           : pipeline root directory
    field_name           : field subdirectory name
    telescope            : 'HST' (JWST planned)
    n_processes          : parallel workers
    hst_pix_floor        : minimum positional uncertainty floor (pixels)
    min_matches          : minimum seed matches required for 4P discovery
    zero_pm              : set all Gaia PMs to 0 (debugging)
    max_mag_diff         : maximum allowed magnitude difference for matching
    scale_sweep          : sweep over pixel scale during 4P discovery (slower)
    discovery_max_offset : half-width of the offset histogram search during 4P
                           discovery (pixels); default 50
    use_resid_floor      : if False, zero out the per-iteration empirical residual
                           covariance added to C_total during affine refinement
    force_rematch        : re-match even if matched_gaia.csv and matching params exist
    image_id             : process only this single observation ID

    Returns
    -------
    List of matched_gaia.csv paths
    """
    if telescope.upper() != 'HST':
        raise NotImplementedError(
            "Cross-matching for non-HST telescopes is not yet implemented. "
            "JWST support is planned once gaia_cross_match handles JWST headers."
        )

    from gaia_cross_match.cross_match import load_gaia_data

    print("\n" + "─"*50)
    print("Step 4: Cross-matching HST ↔ Gaia")
    print("─"*50)

    gaia_df = load_gaia_data(field_name, str(Path(output_dir)))
    if gaia_df is None:
        print("  ERROR: could not load Gaia catalogue.")
        return []

    folders = _find_image_folders(output_dir, field_name,
                                   telescope=telescope, im_type=im_type)
    if image_id:
        folders = [f for f in folders if Path(f['root']).name == image_id]

    if restrict_to_obsids is not None:
        keep = set(restrict_to_obsids)
        folders = [f for f in folders if Path(f['root']).name in keep]
        if not folders:
            print(f"  No image folders match the provided selection of {len(keep)} obs_ids.")
            return []

    if not folders:
        print("  No image catalogs found to cross-match.")
        return []

    params_meta = {
        'hst_pix_floor':        hst_pix_floor,
        'min_matches':          min_matches,
        'zero_pm':              zero_pm,
        'max_mag_diff':         max_mag_diff,
        'scale_sweep':          scale_sweep,
        'discovery_max_offset': discovery_max_offset,
        'use_resid_floor':      use_resid_floor,
    }

    work = []
    skipped = []
    for hst in folders:
        out         = Path(hst['root']) / "matched_gaia.csv"
        params_path = Path(hst['root']) / "xmatch_params.json"

        if not force_rematch:
            ok, diffs = _params_cache_status(out, params_path, params_meta)
            if ok:
                skipped.append(Path(hst['root']).name)
                continue
            if out.exists():
                if diffs == ["no params sidecar — cannot verify configuration match"]:
                    print(f"  {Path(hst['root']).name}: results exist but no params sidecar — re-matching")
                else:
                    print(f"  {Path(hst['root']).name}: params changed — re-matching:")
                    for d in diffs:
                        print(d)

        work.append((hst, gaia_df, {
            'hst_pix_floor':        hst_pix_floor,
            'min_matches':          min_matches,
            'zero_pm':              zero_pm,
            'max_mag_diff':         max_mag_diff,
            'scale_sweep':          scale_sweep,
            'discovery_max_offset': discovery_max_offset,
            'use_resid_floor':      use_resid_floor,
            'params_meta':          params_meta,
        }))

    if skipped:
        print(f"  {len(skipped)} image(s) already matched with matching params — skipping.")
    if not work:
        print("  All cross-matches up to date.")
        existing = [Path(f['root']) / "matched_gaia.csv" for f in folders]
        return [p for p in existing if p.exists()]

    print(f"  Gaia stars:  {len(gaia_df)}")
    print(f"  Images to process: {len(work)}")
    _cmd = (
        f"gaia_cross_match --target {field_name} --data-dir {output_dir}"
        f" --hst-pix-floor {hst_pix_floor}"
        f" --min-matches {min_matches}"
        f" --max-mag-diff {max_mag_diff}"
        + (" --zero-gaia-pm" if zero_pm else "")
        + (" --scale-sweep"  if scale_sweep else "")
    )
    print(f"  gaia_cross_match command (per image):\n    {_cmd}")

    results = []
    if n_processes > 1 and len(work) > 1:
        with ProcessPoolExecutor(max_workers=n_processes) as ex:
            futures = {ex.submit(_match_one, w): w for w in work}
            for fut in as_completed(futures):
                name, n, err = fut.result()
                if err:
                    print(f"  ERROR {name}: {err}")
                else:
                    print(f"  {name}: {n} matches")
                    results.append(Path(next(
                        f['root'] for f in folders
                        if Path(f['root']).name == name
                    )) / "matched_gaia.csv")
    else:
        for w in work:
            name, n, err = _match_one(w)
            if err:
                print(f"  ERROR {name}: {err}")
            else:
                print(f"  {name}: {n} matches")
                results.append(Path(next(
                    f['root'] for f in folders
                    if Path(f['root']).name == name
                )) / "matched_gaia.csv")

    # Run cross-image validation
    try:
        from gaia_cross_match.validator import validate_target
        print("\n  Running cross-image validation...")
        validate_target(field_name, str(Path(output_dir)))
    except Exception as _e:
        print(f"  WARNING: cross-image validation failed — {_e}")

    # Include previously-skipped results in the return list
    for hst in folders:
        p = Path(hst['root']) / "matched_gaia.csv"
        if p not in results and p.exists():
            results.append(p)

    print(f"  Cross-match complete: {len(results)}/{len(folders)} available.")
    return results
