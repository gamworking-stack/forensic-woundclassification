#!/usr/bin/env python3
"""T34: generate wound crops of chosen classes from a trained StyleGAN2-ADA or Stable Diffusion LoRA.

StyleGAN2-ADA: ``--checkpoint`` is a ``network-snapshot-*.pkl``; its generator (G_ema) is conditioned on
the one-hot class in the classifiers' class order (``dataset_info.json`` of the generator dataset).
Stable Diffusion: ``--checkpoint`` is a LoRA folder (``pytorch_lora_weights.safetensors``) on the base
model ``--base``; the prompt is the class caption the LoRA was trained with. The pipeline's NSFW filter
is off: it blanks wound photographs as unsafe.

Every image gets its own seed (``--seed`` + its index), so a sample can be regenerated. Writes
``<class>/<generator>_<n>.png`` and ``samples.csv`` (file, class, generator, checkpoint, seed).
Generated crops come from patient photographs: they stay on Raptor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.export_generator_crops import PROMPT, REPO_ROOT  # noqa: E402

GENERATORS = ("stylegan2", "sd15_lora")
SMALL_CLASSES = ("cut wound", "scratch", "penetration wound")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate wound crops from a trained generator.")
    parser.add_argument("--generator", choices=GENERATORS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="StyleGAN .pkl or LoRA folder.")
    parser.add_argument("--dataset-info", type=Path, required=True, help="dataset_info.json of the generator dataset.")
    parser.add_argument("--classes", nargs="+", default=list(SMALL_CLASSES))
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, required=True, help="Outside the repository.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stylegan-code", type=Path, default=None, help="NVlabs/stylegan3 checkout (StyleGAN).")
    parser.add_argument("--truncation", type=float, default=1.0, help="StyleGAN psi; 1 keeps the full diversity.")
    parser.add_argument("--base", type=Path, default=None, help="Stable Diffusion base model folder (diffusion).")
    parser.add_argument("--steps", type=int, default=30, help="Diffusion sampling steps.")
    parser.add_argument("--guidance", type=float, default=5.0, help="Classifier-free guidance scale.")
    return parser.parse_args(argv)


def stylegan_sampler(args: argparse.Namespace, classes: List[str], device: torch.device) -> Callable:
    """Batch of uint8 HWC images for one class index and a list of seeds."""
    sys.path.insert(0, str(args.stylegan_code))
    import dnnlib  # noqa: F401  (needed to unpickle)
    import legacy

    with open(args.checkpoint, "rb") as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device).eval()
    if G.c_dim != len(classes):
        raise ValueError(f"{args.checkpoint} has {G.c_dim} classes, dataset_info.json {len(classes)}")

    @torch.no_grad()
    def sample(label: int, seeds: List[int]) -> np.ndarray:
        z = torch.from_numpy(np.stack([np.random.RandomState(s).randn(G.z_dim) for s in seeds])).float().to(device)
        c = torch.zeros(len(seeds), G.c_dim, device=device)
        c[:, label] = 1
        images = G(z, c, truncation_psi=args.truncation, noise_mode="const")
        return (images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

    return sample


def diffusion_sampler(args: argparse.Namespace, classes: List[str], device: torch.device) -> Callable:
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(args.base, torch_dtype=torch.bfloat16, safety_checker=None,
                                                   requires_safety_checker=False).to(device)
    # Named explicitly: offline, diffusers will not look the file up itself
    pipe.load_lora_weights(args.checkpoint, weight_name="pytorch_lora_weights.safetensors")
    pipe.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def sample(label: int, seeds: List[int]) -> np.ndarray:
        generators = [torch.Generator(device=device).manual_seed(s) for s in seeds]
        images = pipe([PROMPT.format(classes[label])] * len(seeds), num_inference_steps=args.steps,
                      guidance_scale=args.guidance, generator=generators, output_type="np").images
        return (images * 255).round().clip(0, 255).astype(np.uint8)

    return sample


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = args.output_dir.resolve()
    if out == REPO_ROOT or REPO_ROOT in out.parents:
        raise ValueError(f"--output-dir {out} is inside the repository; generated crops stay outside it")
    classes = json.loads(args.dataset_info.read_text(encoding="utf-8"))["classes"]
    unknown = sorted(set(args.classes) - set(classes))
    if unknown:
        raise ValueError(f"Unknown classes {unknown}; the generator knows {classes}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = (stylegan_sampler if args.generator == "stylegan2" else diffusion_sampler)(args, classes, device)

    rows = []
    for k, name in enumerate(args.classes):
        (out / name).mkdir(parents=True, exist_ok=True)
        # Seeds differ between classes, so no two classes share a latent
        seeds = [args.seed + k * args.per_class + i for i in range(args.per_class)]
        for start in range(0, len(seeds), args.batch_size):
            batch = seeds[start:start + args.batch_size]
            for seed, pixels in zip(batch, sampler(classes.index(name), batch)):
                file = f"{name}/{args.generator}_{seed:06d}.png"
                Image.fromarray(pixels).save(out / file)
                rows.append({"file": file, "class": name, "generator": args.generator,
                             "checkpoint": str(args.checkpoint), "seed": seed})
        print(f"{name}: {args.per_class} crops", flush=True)
    pd.DataFrame(rows).to_csv(out / "samples.csv", index=False)
    print(f"Wrote {len(rows)} crops to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
