#!/usr/bin/env python3
"""LoRA fine-tuning script for Devanagari glyph datasets.

This script targets GPU training and expects a metadata.jsonl created by
metadata_builder.py.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcsLayers, LoRAAttnProcessor
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

import yaml


@dataclass
class TrainConfig:
    model_name_or_path: str
    train_data_dir: str
    metadata_file: str
    output_dir: str
    resolution: int
    train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_train_steps: int
    lora_rank: int
    lora_alpha: int
    seed: int
    mixed_precision: str


class GlyphDataset(Dataset):
    def __init__(self, data_dir: Path, metadata_file: Path, resolution: int) -> None:
        self.data_dir = data_dir
        self.resolution = resolution
        self.records: List[Dict[str, str]] = []
        with metadata_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                self.records.append(json.loads(line))

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        record = self.records[index]
        image_path = self.data_dir / record["file_name"]
        image = Image.open(image_path).convert("L").convert("RGB")
        return {"pixel_values": self.image_transform(image), "text": record["text"]}


def load_config(path: Path) -> TrainConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TrainConfig(**data)


def apply_lora(unet: UNet2DConditionModel, rank: int, alpha: int) -> AttnProcsLayers:
    lora_attn_procs = {}
    for name, attn_processor in unet.attn_processors.items():
        if hasattr(attn_processor, "cross_attention_dim"):
            cross_attention_dim = attn_processor.cross_attention_dim
            hidden_size = attn_processor.hidden_size
        else:
            cross_attention_dim = None
            hidden_size = attn_processor.hidden_size
        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
            lora_alpha=alpha,
        )
    unet.set_attn_processor(lora_attn_procs)
    return AttnProcsLayers(unet.attn_processors)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Devanagari glyphs")
    parser.add_argument("--config", default="training/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    torch.manual_seed(config.seed)

    accelerator = Accelerator(mixed_precision=config.mixed_precision)

    tokenizer = CLIPTokenizer.from_pretrained(config.model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config.model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config.model_name_or_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(config.model_name_or_path, subfolder="scheduler")

    lora_layers = apply_lora(unet, config.lora_rank, config.lora_alpha)

    dataset = GlyphDataset(
        data_dir=Path(config.train_data_dir),
        metadata_file=Path(config.metadata_file),
        resolution=config.resolution,
    )
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=2)

    optimizer = torch.optim.AdamW(lora_layers.parameters(), lr=config.learning_rate)

    unet, text_encoder, vae, optimizer, dataloader = accelerator.prepare(
        unet, text_encoder, vae, optimizer, dataloader
    )

    weight_dtype = torch.float16 if config.mixed_precision == "fp16" else torch.float32
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    global_step = 0
    loss_fn = nn.MSELoss()

    unet.train()
    for epoch in range(10_000):
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                text_inputs = tokenizer(
                    batch["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                encoder_hidden_states = text_encoder(text_inputs.input_ids.to(accelerator.device))[0]

                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = loss_fn(noise_pred.float(), noise.float())

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.is_main_process:
                if global_step % 50 == 0:
                    accelerator.print(f"step={global_step} loss={loss.item():.4f}")

            global_step += 1
            if global_step >= config.max_train_steps:
                break
        if global_step >= config.max_train_steps:
            break

    if accelerator.is_main_process:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        unet.save_attn_procs(output_dir)


if __name__ == "__main__":
    main()
