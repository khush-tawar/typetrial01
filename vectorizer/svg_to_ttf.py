#!/usr/bin/env fontforge
"""Import SVG glyphs and export a TTF font.

Invoke with: fontforge -script vectorizer/svg_to_ttf.py --help
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import fontforge

DEFAULT_GLYPHS: Dict[str, int] = {
    "a": 0x0905,
    "aa": 0x0906,
    "i": 0x0907,
    "ii": 0x0908,
    "u": 0x0909,
    "uu": 0x090A,
    "e": 0x090F,
    "ai": 0x0910,
    "o": 0x0913,
    "au": 0x0914,
    "ka": 0x0915,
    "kha": 0x0916,
    "ga": 0x0917,
    "gha": 0x0918,
    "nga": 0x0919,
    "ca": 0x091A,
    "cha": 0x091B,
    "ja": 0x091C,
    "jha": 0x091D,
    "nya": 0x091E,
    "tta": 0x091F,
    "ttha": 0x0920,
    "dda": 0x0921,
    "ddha": 0x0922,
    "nna": 0x0923,
    "ta": 0x0924,
    "tha": 0x0925,
    "da": 0x0926,
    "dha": 0x0927,
    "na": 0x0928,
    "pa": 0x092A,
    "pha": 0x092B,
    "ba": 0x092C,
    "bha": 0x092D,
    "ma": 0x092E,
    "ya": 0x092F,
    "ra": 0x0930,
    "la": 0x0932,
    "va": 0x0935,
    "sha": 0x0936,
    "ssa": 0x0937,
    "sa": 0x0938,
    "ha": 0x0939,
    "anusvara": 0x0902,
    "visarga": 0x0903,
    "candrabindu": 0x0901,
}


def load_mapping(mapping_file: Path | None) -> Dict[str, int]:
    if mapping_file is None:
        return DEFAULT_GLYPHS

    mapping: Dict[str, int] = {}
    with mapping_file.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row["name"].strip()
            codepoint = int(row["codepoint"], 16)
            mapping[name] = codepoint
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SVG glyphs into TTF")
    parser.add_argument("--svg-dir", required=True, help="Folder containing SVG files")
    parser.add_argument("--output-ttf", required=True, help="Output TTF file")
    parser.add_argument("--mapping-file", help="CSV mapping file with name,codepoint")
    parser.add_argument("--font-name", default="IndicFontAI", help="Internal font name")
    parser.add_argument("--family-name", default="Indic Font AI", help="Family name")
    parser.add_argument("--style-name", default="Regular", help="Style name")
    parser.add_argument("--em-size", type=int, default=1000, help="Units per em")
    parser.add_argument("--ascender", type=int, default=800, help="Ascender height")
    parser.add_argument("--descender", type=int, default=200, help="Descender height")
    args = parser.parse_args()

    svg_dir = Path(args.svg_dir)
    mapping = load_mapping(Path(args.mapping_file)) if args.mapping_file else load_mapping(None)

    font = fontforge.font()
    font.encoding = "UnicodeFull"
    font.em = args.em_size
    font.ascent = args.ascender
    font.descent = args.descender
    font.fontname = args.font_name
    font.familyname = args.family_name
    font.fullname = f"{args.family_name} {args.style_name}".strip()

    for name, codepoint in mapping.items():
        svg_path = svg_dir / f"{name}.svg"
        if not svg_path.exists():
            continue
        glyph = font.createChar(codepoint, name)
        glyph.importOutlines(str(svg_path))
        glyph.width = args.em_size

    output_path = Path(args.output_ttf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font.generate(str(output_path))


if __name__ == "__main__":
    main()
