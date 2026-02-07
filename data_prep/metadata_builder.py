#!/usr/bin/env python3
"""Build metadata.jsonl for generated Devanagari glyph images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

GLYPH_LABELS: Dict[str, str] = {
    "a": "A",
    "aa": "Aa",
    "i": "I",
    "ii": "Ii",
    "u": "U",
    "uu": "Uu",
    "e": "E",
    "ai": "Ai",
    "o": "O",
    "au": "Au",
    "ka": "Ka",
    "kha": "Kha",
    "ga": "Ga",
    "gha": "Gha",
    "nga": "Nga",
    "ca": "Ca",
    "cha": "Cha",
    "ja": "Ja",
    "jha": "Jha",
    "nya": "Nya",
    "tta": "Tta",
    "ttha": "Ttha",
    "dda": "Dda",
    "ddha": "Ddha",
    "nna": "Nna",
    "ta": "Ta",
    "tha": "Tha",
    "da": "Da",
    "dha": "Dha",
    "na": "Na",
    "pa": "Pa",
    "pha": "Pha",
    "ba": "Ba",
    "bha": "Bha",
    "ma": "Ma",
    "ya": "Ya",
    "ra": "Ra",
    "la": "La",
    "va": "Va",
    "sha": "Sha",
    "ssa": "Ssa",
    "sa": "Sa",
    "ha": "Ha",
    "anusvara": "Anusvara",
    "visarga": "Visarga",
    "candrabindu": "Candrabindu",
}


def infer_glyph_name(file_stem: str) -> str:
    parts = file_stem.split("_")
    return parts[0]


def build_prompt(glyph_name: str, style: str) -> str:
    label = GLYPH_LABELS.get(glyph_name, glyph_name.title())
    return f"Devanagari letter {label}, {style}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build metadata.jsonl")
    parser.add_argument("--images-dir", required=True, help="Folder with PNG images")
    parser.add_argument("--output-file", required=True, help="Path to metadata.jsonl")
    parser.add_argument(
        "--style",
        default="calligraphy style",
        help="Prompt style suffix",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as handle:
        for image_path in sorted(images_dir.rglob("*.png")):
            glyph_name = infer_glyph_name(image_path.stem)
            record = {
                "file_name": str(image_path.relative_to(images_dir)),
                "text": build_prompt(glyph_name, args.style),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
