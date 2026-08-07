"""bp3m-setup: Download HST and JWST PSF/GDC library files from STScI."""

import argparse
import re
import sys
from pathlib import Path
from urllib.request import urlopen, urlretrieve
from urllib.error import URLError

HST_BASE_URL = "https://www.stsci.edu/~jayander/HST1PASS/LIB"
JWST_BASE_URL = "https://www.stsci.edu/~jayander/JWST1PASS/LIB"

def _bp3m_home() -> Path:
    """Base directory for bp3m config and default lib. Override with BP3M_HOME."""
    import os
    return Path(os.environ["BP3M_HOME"]) if "BP3M_HOME" in os.environ else Path.home() / ".bp3m"

CONFIG_FILE = _bp3m_home() / "config.toml"
DEFAULT_LIB_DIR = _bp3m_home() / "lib"

PSF_INSTRUMENTS = ["ACSWFC", "ACSHRC", "WFC3UV"]
GDC_INSTRUMENTS = ["ACSWFC", "ACSHRC", "WFC3UV"]
# WFC3IR has PSFs on the server but no GDCs; not yet supported by pypass.
# Users can request it explicitly with --instruments WFC3IR.
_OPTIONAL_PSF_ONLY = {"WFC3IR"}

# JWST: PSFs and GDCs are published for the same three instruments.
JWST_INSTRUMENTS = ["NIRCam", "NIRISS", "MIRI"]
# NIRCam is split into short-wave (SWC) and long-wave (LWC) channels. On the
# server, SWC filters each live in their own subdirectory; LWC files sit
# flat directly under LWC/. Both cases are handled generically (see
# _download_jwst_group): if a channel directory has no filter
# subdirectories, its .fits files are listed directly instead.
_NIRCAM_CHANNELS = ["SWC", "LWC"]

# STScI is NOT consistent about filename token order for NIRCam STDPSF/STDGDC
# files: most filters use STD{X}_{detector}_{filter}.fits, but a few (observed:
# F210M, F070W under GDCs/SWC) are published the other way round, as
# STD{X}_{filter}_{detector}.fits. jwst1pass_py_v2's find_psf()/find_gdc()
# only ever construct the detector-first form when looking a file up locally,
# so files must be *saved* under the canonical detector-first name regardless
# of which order the server happened to use. See _canonical_nircam_name().
_NRC_DET_RE = re.compile(r"(NRC[AB](?:L|[1-4]))", re.IGNORECASE)
_FILT_RE = re.compile(r"(F\d{2,4}[A-Z]{1,2})", re.IGNORECASE)


def _canonical_nircam_name(kind: str, basename: str) -> str:
    """Reorder a scraped NIRCam filename into STD{kind}_{detector}_{filter}.fits.

    kind is 'PSF' or 'GDC'. Falls back to the original basename if the
    detector/filter tokens can't both be parsed out of it.
    """
    det_m = _NRC_DET_RE.search(basename)
    filt_m = _FILT_RE.search(basename)
    if not det_m or not filt_m:
        return basename
    return f"STD{kind}_{det_m.group(1).upper()}_{filt_m.group(1).upper()}.fits"


def _list_fits(url: str) -> list:
    """Return list of full .fits file URLs by scraping the STScI directory listing."""
    try:
        with urlopen(url, timeout=30) as r:
            html = r.read().decode("utf-8", errors="replace")
        names = re.findall(r'href="([^"]+\.fits)"', html, re.IGNORECASE)
        base = url.rstrip("/")
        return [f"{base}/{n}" for n in names]
    except URLError as e:
        print(f"  WARNING: could not list {url}: {e}")
        return []


def _list_dirs(url: str) -> list:
    """Return subdirectory names linked from an STScI directory listing.

    Excludes the "Parent Directory" link (absolute path, starts with "/")
    and the column-sort query links (e.g. "?C=N;O=D", no trailing "/").
    """
    try:
        with urlopen(url, timeout=30) as r:
            html = r.read().decode("utf-8", errors="replace")
        return re.findall(r'href="([^"?/][^"]*)/"', html)
    except URLError as e:
        print(f"  WARNING: could not list {url}: {e}")
        return []


def _download(url: str, dest: Path) -> bool:
    """Download url to dest. Returns True on success.

    Validates that the downloaded content actually starts with a FITS
    "SIMPLE" card before committing it to dest. urlretrieve already raises
    on a non-2xx HTTP status, but this is a second line of defense against
    any download path (e.g. a mirroring tool, a proxy that returns 200 for
    an error page) that would otherwise silently save a bad response body
    with a .fits extension.
    """
    tmp = dest.with_suffix(".tmp")
    try:
        urlretrieve(url, str(tmp))
        with open(tmp, "rb") as f:
            magic = f.read(6)
        if magic != b"SIMPLE":
            raise ValueError(f"downloaded content is not a FITS file (starts with {magic!r})")
        tmp.rename(dest)
        return True
    except Exception as e:
        print(f"  ERROR downloading {url}: {e}")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False


def _download_hst_group(lib_dir: Path, kind: str, instruments: list, force: bool) -> tuple:
    """kind: 'PSF' or 'GDC'. Downloads flat per-instrument HST libraries."""
    label = "STDPSFs" if kind == "PSF" else "STDGDCs"
    top = "PSFs" if kind == "PSF" else "GDCs"
    n_ok = n_skip = n_err = 0
    for inst in instruments:
        # ACSWFC GDCs live in a VINTAGE_2005 subdirectory
        if kind == "GDC" and inst == "ACSWFC":
            url = f"{HST_BASE_URL}/{top}/{label}/{inst}/VINTAGE_2005"
        else:
            url = f"{HST_BASE_URL}/{top}/{label}/{inst}"
        files = _list_fits(url)
        if not files:
            print(f"  {inst}: no .fits files found at {url}")
            continue
        dest_dir = lib_dir / label / inst
        dest_dir.mkdir(parents=True, exist_ok=True)
        for file_url in files:
            fname = file_url.rsplit("/", 1)[-1]
            dest = dest_dir / fname
            if dest.exists() and not force:
                n_skip += 1
                continue
            print(f"  {inst}/{fname}")
            if _download(file_url, dest):
                n_ok += 1
            else:
                n_err += 1
    return n_ok, n_skip, n_err


def _download_jwst_group(lib_dir: Path, kind: str, instruments: list, force: bool) -> tuple:
    """kind: 'PSF' or 'GDC'. Downloads JWST libraries.

    NIRISS/MIRI are flat: STD{kind}_{INST}_{filter}.fits directly under the
    instrument directory. NIRCam is split into SWC/LWC channels; each
    channel is either flat (LWC) or split further into per-filter
    subdirectories (SWC) -- both are handled by first trying to list filter
    subdirectories and falling back to listing .fits files directly.
    """
    label = "STDPSFs" if kind == "PSF" else "STDGDCs"
    top = "PSFs" if kind == "PSF" else "GDCs"
    n_ok = n_skip = n_err = 0
    for inst in instruments:
        if inst == "NIRCam":
            for channel in _NIRCAM_CHANNELS:
                chan_url = f"{JWST_BASE_URL}/{top}/{label}/NIRCam/{channel}"
                filters = _list_dirs(chan_url)
                # (filter_subdir_or_None, listing_url) pairs to walk
                groups = [(f, f"{chan_url}/{f}") for f in filters] if filters else [(None, chan_url)]
                for filt, list_url in groups:
                    files = _list_fits(list_url)
                    if not files:
                        continue
                    dest_dir = lib_dir / label / "NIRCam" / channel / filt if filt \
                        else lib_dir / label / "NIRCam" / channel
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    for file_url in files:
                        raw_fname = file_url.rsplit("/", 1)[-1]
                        fname = _canonical_nircam_name(kind, raw_fname)
                        dest = dest_dir / fname
                        if dest.exists() and not force:
                            n_skip += 1
                            continue
                        label_str = f"NIRCam/{channel}/{filt}/{fname}" if filt else f"NIRCam/{channel}/{fname}"
                        print(f"  {label_str}")
                        if _download(file_url, dest):
                            n_ok += 1
                        else:
                            n_err += 1
        else:
            url = f"{JWST_BASE_URL}/{top}/{label}/{inst}"
            files = _list_fits(url)
            if not files:
                print(f"  {inst}: no .fits files found at {url}")
                continue
            dest_dir = lib_dir / label / inst
            dest_dir.mkdir(parents=True, exist_ok=True)
            for file_url in files:
                fname = file_url.rsplit("/", 1)[-1]
                dest = dest_dir / fname
                if dest.exists() and not force:
                    n_skip += 1
                    continue
                print(f"  {inst}/{fname}")
                if _download(file_url, dest):
                    n_ok += 1
                else:
                    n_err += 1
    return n_ok, n_skip, n_err


def main():
    p = argparse.ArgumentParser(
        description=(
            "Download PSF and geometric distortion correction (GDC) library "
            "files for bp3m from STScI (Jay Anderson's HST1PASS/JWST1PASS LIB "
            "pages). Saves the lib_dir path to config.toml so --lib_dir is "
            "optional when running bp3m. Config location defaults to ~/.bp3m/ "
            "but can be overridden by setting the BP3M_HOME environment variable."
        )
    )
    p.add_argument(
        "--lib-dir",
        default=None,
        help=f"Directory to store PSF/GDC files (default: {DEFAULT_LIB_DIR})",
    )
    p.add_argument(
        "--no-config",
        action="store_true",
        help="Skip writing lib_dir to config.toml",
    )
    p.add_argument(
        "--telescope",
        choices=["HST", "JWST", "both"],
        default="HST",
        help=(
            "Which telescope's library to download (default: HST, for "
            "backward compatibility). JWST NIRCam GDCs alone are several GB "
            "-- pass --telescope JWST or --telescope both to opt in."
        ),
    )
    p.add_argument(
        "--instruments",
        nargs="+",
        default=None,
        metavar="INST",
        help=(
            "HST instruments to download PSFs/GDCs for (default: all). "
            "PSF choices: ACSWFC ACSHRC WFC3UV WFC3IR. "
            "GDC choices: ACSWFC ACSHRC WFC3UV."
        ),
    )
    p.add_argument(
        "--jwst-instruments",
        nargs="+",
        default=None,
        metavar="INST",
        help="JWST instruments to download PSFs/GDCs for (default: all). Choices: NIRCam NIRISS MIRI.",
    )
    p.add_argument(
        "--no-gdcs",
        action="store_true",
        help="Skip downloading GDC files",
    )
    p.add_argument(
        "--no-psfs",
        action="store_true",
        help="Skip downloading PSF files",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download files that already exist locally",
    )
    args = p.parse_args()

    lib_dir = Path(args.lib_dir) if args.lib_dir else DEFAULT_LIB_DIR

    if args.instruments:
        requested = {i.upper() for i in args.instruments}
        all_psf = PSF_INSTRUMENTS + list(_OPTIONAL_PSF_ONLY)
        hst_psf_insts = [i for i in all_psf if i in requested]
        hst_gdc_insts = [i for i in GDC_INSTRUMENTS if i in requested]
    else:
        hst_psf_insts = PSF_INSTRUMENTS
        hst_gdc_insts = GDC_INSTRUMENTS

    if args.jwst_instruments:
        requested = {i.upper() for i in args.jwst_instruments}
        jwst_insts = [i for i in JWST_INSTRUMENTS if i.upper() in requested]
    else:
        jwst_insts = JWST_INSTRUMENTS

    do_hst = args.telescope in ("HST", "both")
    do_jwst = args.telescope in ("JWST", "both")

    print("bp3m library setup")
    print(f"  lib_dir   : {lib_dir}")
    if do_hst:
        print(f"  HST  PSF insts : {', '.join(hst_psf_insts)}")
        print(f"  HST  GDC insts : {', '.join(hst_gdc_insts)}")
    if do_jwst:
        print(f"  JWST insts     : {', '.join(jwst_insts)} (PSFs + GDCs; NIRCam GDCs are several GB)")
    print()

    n_ok = n_skip = n_err = 0

    if do_hst and not args.no_psfs:
        print("Downloading HST PSF files...")
        ok, skip, err = _download_hst_group(lib_dir, "PSF", hst_psf_insts, args.force)
        n_ok += ok; n_skip += skip; n_err += err
        print()

    if do_hst and not args.no_gdcs:
        print("Downloading HST GDC files...")
        ok, skip, err = _download_hst_group(lib_dir, "GDC", hst_gdc_insts, args.force)
        n_ok += ok; n_skip += skip; n_err += err
        print()

    if do_jwst and not args.no_psfs:
        print("Downloading JWST PSF files...")
        ok, skip, err = _download_jwst_group(lib_dir, "PSF", jwst_insts, args.force)
        n_ok += ok; n_skip += skip; n_err += err
        print()

    if do_jwst and not args.no_gdcs:
        print("Downloading JWST GDC files...")
        ok, skip, err = _download_jwst_group(lib_dir, "GDC", jwst_insts, args.force)
        n_ok += ok; n_skip += skip; n_err += err
        print()

    print(f"Done: {n_ok} downloaded, {n_skip} already present, {n_err} errors.")

    # ── Write config ──────────────────────────────────────────────────────────
    if not args.no_config:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(f'lib_dir = "{lib_dir}"\n')
        print(f"Config written to {CONFIG_FILE}")
        print(f"bp3m will use lib_dir={lib_dir} by default (override with --lib_dir).")

    if n_err > 0:
        sys.exit(1)
