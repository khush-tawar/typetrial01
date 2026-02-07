#!/usr/bin/env python3
"""Generate a Devanagari glyph dataset from .ttf fonts.

For every font in assets/fonts/ and every character in the top-50 Devanagari
list, renders a 512x512 black-on-white PNG into dataset/images/ and writes a
HuggingFace-compatible metadata.jsonl alongside the images.

Usage:
    python data_prep/generate_hindi_dataset.py
    python data_prep/generate_hindi_dataset.py --fonts-dir path/to/fonts
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Top 50 most common Devanagari characters
# (vowels, consonants, anusvara, visarga, chandrabindu, nukta, virama)
# Each entry: (romanised name, Unicode codepoint, display label for prompt)
# ---------------------------------------------------------------------------
DEVANAGARI_CHARS: List[Tuple[str, int, str]] = [
    # -- Vowels (स्वर) --
    ("a",            0x0905, "Vowel A"),
    ("aa",           0x0906, "Vowel Aa"),
    ("i",            0x0907, "Vowel I"),
    ("ii",           0x0908, "Vowel Ii"),
    ("u",            0x0909, "Vowel U"),
    ("uu",           0x090A, "Vowel Uu"),
    ("ri",           0x090B, "Vowel Ri"),
    ("e",            0x090F, "Vowel E"),
    ("ai",           0x0910, "Vowel Ai"),
    ("o",            0x0913, "Vowel O"),
    ("au",           0x0914, "Vowel Au"),
    # -- Consonants (व्यंजन) --
    ("ka",           0x0915, "Consonant Ka"),
    ("kha",          0x0916, "Consonant Kha"),
    ("ga",           0x0917, "Consonant Ga"),
    ("gha",          0x0918, "Consonant Gha"),
    ("nga",          0x0919, "Consonant Nga"),
    ("cha",          0x091A, "Consonant Cha"),
    ("chha",         0x091B, "Consonant Chha"),
    ("ja",           0x091C, "Consonant Ja"),
    ("jha",          0x091D, "Consonant Jha"),
    ("nya",          0x091E, "Consonant Nya"),
    ("tta",          0x091F, "Consonant Tta"),
    ("ttha",         0x0920, "Consonant Ttha"),
    ("dda",          0x0921, "Consonant Dda"),
    ("ddha",         0x0922, "Consonant Ddha"),
    ("nna",          0x0923, "Consonant Nna"),
    ("ta",           0x0924, "Consonant Ta"),
    ("tha",          0x0925, "Consonant Tha"),
    ("da",           0x0926, "Consonant Da"),
    ("dha",          0x0927, "Consonant Dha"),
    ("na",           0x0928, "Consonant Na"),
    ("pa",           0x092A, "Consonant Pa"),
    ("pha",          0x092B, "Consonant Pha"),
    ("ba",           0x092C, "Consonant Ba"),
    ("bha",          0x092D, "Consonant Bha"),
    ("ma",           0x092E, "Consonant Ma"),
    ("ya",           0x092F, "Consonant Ya"),
    ("ra",           0x0930, "Consonant Ra"),
    ("la",           0x0932, "Consonant La"),
    ("va",           0x0935, "Consonant Va"),
    ("sha",          0x0936, "Consonant Sha"),
    ("shha",         0x0937, "Consonant Shha"),
    ("sa",           0x0938, "Consonant Sa"),
    ("ha",           0x0939, "Consonant Ha"),
    # -- Special marks --
    ("chandrabindu", 0x0901, "Chandrabindu"),
    ("anusvara",     0x0902, "Anusvara"),
    ("visarga",      0x0903, "Visarga"),
    ("nukta",        0x093C, "Nukta"),
    ("virama",       0x094D, "Virama (Halant)"),
    ("om",           0x0950, "Om"),
]

# Defaults
DEFAULT_FONTS_DIR = Path("assets/fonts")
DEFAULT_OUTPUT_DIR = Path("dataset/images")
DEFAULT_METADATA_PATH = Path("dataset/metadata.jsonl")
IMAGE_SIZE = 512
FONT_SIZE = 360


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_character(
    char: str,
    font: ImageFont.FreeTypeFont,
    size: int = IMAGE_SIZE,
) -> Image.Image:
    """Render a single character centred on a white 512x512 canvas."""
    img = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Measure the glyph bounding box so we can centre it
    bbox = draw.textbbox((0, 0), char, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (size - text_w) // 2 - bbox[0]
    y = (size - text_h) // 2 - bbox[1]

    draw.text((x, y), char, font=font, fill=(0, 0, 0))
    return img


def load_fonts(fonts_dir: Path) -> List[Tuple[str, ImageFont.FreeTypeFont]]:
    """Load every .ttf file found in the fonts directory."""
    fonts: List[Tuple[str, ImageFont.FreeTypeFont]] = []
    if not fonts_dir.exists():
        print(f"[ERROR] Fonts directory not found: {fonts_dir}", file=sys.stderr)
        sys.exit(1)

    for path in sorted(fonts_dir.rglob("*")):
        if path.suffix.lower() in (".ttf", ".otf"):
            try:
                pil_font = ImageFont.truetype(str(path), FONT_SIZE)
                fonts.append((path.stem, pil_font))
            except Exception as exc:
                print(f"[WARN] Skipping {path.name}: {exc}", file=sys.stderr)
    return fonts


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def generate_dataset(
    fonts_dir: Path = DEFAULT_FONTS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> None:
    fonts = load_fonts(fonts_dir)
    if not fonts:
        print(f"[ERROR] No .ttf/.otf fonts found in {fonts_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(fonts) * len(DEVANAGARI_CHARS)
    records: List[dict] = []

    print(f"Generating {total} images from {len(fonts)} font(s) "
          f"x {len(DEVANAGARI_CHARS)} characters ...")

    with tqdm(total=total, unit="img") as pbar:
        for font_name, pil_font in fonts:
            for char_name, codepoint, label in DEVANAGARI_CHARS:
                char = chr(codepoint)
                filename = f"{font_name}_{char_name}_{codepoint:04X}.png"
                out_path = output_dir / filename

                img = render_character(char, pil_font)
                img.save(out_path, format="PNG")

                # HuggingFace datasets expects file_name relative to the
                # directory that contains metadata.jsonl.  We store images
                # in a sibling "images/" folder, so the relative path is
                # "images/<filename>".
                relative_path = os.path.relpath(out_path, metadata_path.parent)
                records.append({
                    "file_name": relative_path,
                    "text": f"Devanagari {label}, {font_name} font style",
                    "char_name": char_name,
                    "codepoint": f"U+{codepoint:04X}",
                    "font": font_name,
                })

                pbar.update(1)

    # Write metadata.jsonl (HuggingFace ImageFolder format)
    with metadata_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone!  {len(records)} images saved to {output_dir}/")
    print(f"Metadata written to {metadata_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a Devanagari glyph dataset from .ttf fonts",
    )
    parser.add_argument(
        "--fonts-dir",
        type=Path,
        default=DEFAULT_FONTS_DIR,
        help="Directory containing .ttf/.otf font files (default: assets/fonts/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for rendered PNGs (default: dataset/images/)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to metadata.jsonl (default: dataset/metadata.jsonl)",
    )
    args = parser.parse_args()

    generate_dataset(
        fonts_dir=args.fonts_dir,
        output_dir=args.output_dir,
        metadata_path=args.metadata,
    )


if __name__ == "__main__":
    main()
