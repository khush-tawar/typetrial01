#!/usr/bin/env python3
"""Convert raster PNG glyph images to SVG outlines using potrace.

potrace only accepts PBM/PGM/PPM/BMP input, so each PNG is first converted
to a BMP (1‑bit black-and-white) before being traced.

Usage:
    python vectorizer/raster_to_svg.py --input-dir dataset/images --output-dir dataset/svgs
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_potrace() -> str:
    """Return the path to potrace, or exit with a helpful message."""
    potrace = shutil.which("potrace")
    if potrace is None:
        print(
            "[ERROR] potrace not found on PATH.\n"
            "Install it with:  sudo apt install potrace  (Debian/Ubuntu)\n"
            "              or:  brew install potrace      (macOS)",
            file=sys.stderr,
        )
        sys.exit(1)
    return potrace


def png_to_bmp(png_path: Path, bmp_path: Path, threshold: int = 128) -> None:
    """Convert a PNG to a 1-bit BMP suitable for potrace.

    The glyph images are black text on white background.  potrace traces
    *black* regions, so we threshold and keep that polarity.
    """
    img = Image.open(png_path).convert("L")                 # grayscale
    img = img.point(lambda px: 0 if px < threshold else 255, mode="1")  # binarise
    img.save(bmp_path, format="BMP")


def trace_to_svg(
    potrace_bin: str,
    bmp_path: Path,
    svg_path: Path,
    turdsize: int = 2,
    alphamax: float = 1.0,
    opttolerance: float = 0.2,
) -> None:
    """Call potrace to convert a BMP into an SVG."""
    cmd: List[str] = [
        potrace_bin,
        str(bmp_path),
        "-s",                              # SVG output
        "-o", str(svg_path),
        "-t", str(turdsize),               # suppress speckles smaller than this
        "-a", str(alphamax),               # corner threshold
        "-O", str(opttolerance),           # curve optimisation tolerance
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[WARN] potrace failed for {bmp_path.name}: {result.stderr.strip()}",
              file=sys.stderr)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def convert_folder(
    input_dir: Path,
    output_dir: Path,
    threshold: int = 128,
    turdsize: int = 2,
    alphamax: float = 1.0,
    opttolerance: float = 0.2,
) -> None:
    potrace_bin = check_potrace()

    png_files = sorted(input_dir.rglob("*.png"))
    if not png_files:
        print(f"[ERROR] No PNG files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tracing {len(png_files)} PNGs from {input_dir} → {output_dir} ...")

    with tempfile.TemporaryDirectory(prefix="raster2svg_") as tmpdir:
        tmp = Path(tmpdir)
        for png_path in tqdm(png_files, unit="file"):
            # Mirror sub-directory structure
            relative = png_path.relative_to(input_dir)
            svg_path = (output_dir / relative).with_suffix(".svg")
            svg_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert PNG → BMP → SVG
            bmp_path = tmp / f"{png_path.stem}.bmp"
            png_to_bmp(png_path, bmp_path, threshold=threshold)
            trace_to_svg(
                potrace_bin, bmp_path, svg_path,
                turdsize=turdsize,
                alphamax=alphamax,
                opttolerance=opttolerance,
            )

    print(f"Done!  SVGs written to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace PNG glyph images into SVG outlines using potrace",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dataset/images"),
        help="Folder containing PNG glyph images (default: dataset/images/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/svgs"),
        help="Output folder for SVG files (default: dataset/svgs/)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Binarisation threshold 0-255 (default: 128)",
    )
    parser.add_argument(
        "--turdsize",
        type=int,
        default=2,
        help="potrace -t: suppress speckles up to this size (default: 2)",
    )
    parser.add_argument(
        "--alphamax",
        type=float,
        default=1.0,
        help="potrace -a: corner threshold (default: 1.0)",
    )
    parser.add_argument(
        "--opttolerance",
        type=float,
        default=0.2,
        help="potrace -O: curve optimisation tolerance (default: 0.2)",
    )
    args = parser.parse_args()

    convert_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        turdsize=args.turdsize,
        alphamax=args.alphamax,
        opttolerance=args.opttolerance,
    )


if __name__ == "__main__":
    main()
