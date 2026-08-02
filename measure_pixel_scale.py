#!/usr/bin/env python3
"""Measure per-detector JWST pixel scale directly from _cal.fits WCS CD matrices.

Usage:
    python measure_pixel_scale.py /path/to/outputs/<field>/JWST
    python measure_pixel_scale.py /path/to/outputs/<field>/JWST --output pixel_scale_report.txt
"""
import argparse
import datetime
import glob
import os
from collections import defaultdict

import numpy as np
from astropy.io import fits


def cd_pixel_scale_arcsec(sci_hdr):
    """Geometric-mean pixel scale (arcsec/px) from the SCI-extension CD matrix."""
    cd11 = sci_hdr.get('CD1_1')
    cd12 = sci_hdr.get('CD1_2', 0.0)
    cd21 = sci_hdr.get('CD2_1', 0.0)
    cd22 = sci_hdr.get('CD2_2')
    if cd11 is None or cd22 is None:
        # Fall back to PC + CDELT convention if CD keywords aren't present.
        pc11 = sci_hdr.get('PC1_1', 1.0)
        pc12 = sci_hdr.get('PC1_2', 0.0)
        pc21 = sci_hdr.get('PC2_1', 0.0)
        pc22 = sci_hdr.get('PC2_2', 1.0)
        cdelt1 = sci_hdr.get('CDELT1', 1.0)
        cdelt2 = sci_hdr.get('CDELT2', 1.0)
        cd11, cd12 = pc11 * cdelt1, pc12 * cdelt1
        cd21, cd22 = pc21 * cdelt2, pc22 * cdelt2
    det = abs(cd11 * cd22 - cd12 * cd21)
    return np.sqrt(det) * 3600.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', help='Directory to search recursively for *_cal.fits')
    parser.add_argument('--output', default='pixel_scale_report.txt',
                         help='Text file to write per-image and per-detector results to '
                              '(default: pixel_scale_report.txt, written under --root)')
    args = parser.parse_args()

    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(args.root, out_path)

    lines = []

    def log(msg=''):
        print(msg)
        lines.append(msg)

    files = sorted(glob.glob(os.path.join(args.root, '**', '*_cal.fits'), recursive=True))
    if not files:
        log(f"No *_cal.fits files found under {args.root}")
        with open(out_path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        return

    log(f"Pixel scale report — generated {datetime.datetime.now().isoformat(timespec='seconds')}")
    log(f"Root: {args.root}")
    log(f"Files scanned: {len(files)}")
    log('')
    log("=== Per-image pixel scale ===")

    per_detector = defaultdict(list)
    skipped = []
    for f in files:
        try:
            with fits.open(f) as hdul:
                hdr0 = hdul[0].header
                sci_hdr = hdul['SCI'].header
                instrume = hdr0.get('INSTRUME', '?').upper()
                detector = hdr0.get('DETECTOR', '?').upper()
                scale = cd_pixel_scale_arcsec(sci_hdr)
        except Exception as e:
            skipped.append(f"  SKIP {f}: {e}")
            log(skipped[-1])
            continue
        per_detector[(instrume, detector)].append(scale)
        log(f"  {os.path.basename(f):40s} {instrume:8s} {detector:10s} {scale:.6f} arcsec/px")

    log("\n=== Per-detector average pixel scale ===")
    for (instrume, detector), scales in sorted(per_detector.items()):
        scales = np.array(scales)
        log(f"{instrume:8s} {detector:10s} n={len(scales):3d}  "
            f"mean={scales.mean():.6f}  median={np.median(scales):.6f}  std={scales.std():.6f}  "
            f"min={scales.min():.6f}  max={scales.max():.6f}")

    if skipped:
        log(f"\n{len(skipped)} file(s) skipped due to errors (see above).")

    with open(out_path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"\nReport written to {out_path}")


if __name__ == '__main__':
    main()
