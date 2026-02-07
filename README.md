# Indic Font AI (Hindi/Devanagari Focused)

## Overview
This project is an AI-powered typeface generator specialized in Devanagari (Hindi) scripts. It uses a diffusion-based architecture (FontDiffuser) to perform style transfer (for example, "Bollywood style") onto Devanagari glyphs. The pipeline includes data preparation, fine-tuning, and vectorization (converting generated images to TTF).

## Goals
1. Overcome Chinese bias in the base model by fine-tuning on high-quality Hindi fonts.
2. Vectorize AI-generated raster images into valid `.ttf` files using `potrace` and `fontforge`.
3. (Future) Support variable font interpolation (weight and slant).

## Architecture & File Structure
```
data_prep/
	generate_hindi_dataset.py
	metadata_builder.py
training/
	config.yaml
	train_lora.py
vectorizer/
	raster_to_svg.py
	svg_to_ttf.py
assets/
	fonts/
dataset/
	train/
```

## Dependencies
- Python 3.10+
- AI: `torch`, `torchvision`, `diffusers`, `transformers`, `accelerate`, `pyyaml`
- Image/Font: `Pillow`, `fonttools`, `uharfbuzz`
- System Tools: `potrace`, `fontforge`

## Usage Workflow
1. Place reference Hindi fonts in `assets/fonts/`.
2. Run `data_prep/generate_hindi_dataset.py` to create `dataset/train/`.
3. (On GPU) Run `training/train_lora.py` pointing to `dataset/train/`.
4. Run your inference script with the new weights to generate glyph images.
5. Run `vectorizer/raster_to_svg.py` to trace PNGs to SVGs.
6. Run `vectorizer/svg_to_ttf.py` to build the final `.ttf` file.

## Notes
- Devanagari shaping depends on HarfBuzz. `generate_hindi_dataset.py` uses Pillow's libraqm layout engine when available.
- The training script is designed to run on a GPU node (for example, Colab or a workstation with CUDA).
- `svg_to_ttf.py` is intended to be invoked via `fontforge -script`.