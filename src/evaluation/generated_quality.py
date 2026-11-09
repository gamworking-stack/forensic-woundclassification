#!/usr/bin/env python3
"""T34: realism and memorisation of generated wound crops, per generator and class.

Features are the 2,048-d pool features of the Inception network StyleGAN's metrics use
(``--detector``, a TorchScript pickle), so the KIDs here are on the same scale as StyleGAN's own.

- KID of the generated crops against the real training crops and against the real validation crops
  of the same class. Reference: the KID of the real validation crops against the real training crops,
  i.e. the distance of genuinely new wounds.
- Memorisation: each crop's nearest training crop (cosine similarity of the features, any class).
  A generated crop is flagged when it is closer to a training crop than ``--quantile`` of the real
  validation crops are; a validation crop is a new wound, so closer than that suggests a copy.
- Panels: generated crops beside real training crops, and the closest generated-training pairs.

Real crops are padded to a square as the classifiers see them. Writes ``quality.csv``,
``nearest.csv`` and ``panels/``. Patient-derived images: run on Raptor, outputs stay there.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.export_generator_crops import REPO_ROOT  # noqa: E402
from src.data.paths import DATA_DIR  # noqa: E402
from src.training.data import load_manifest, prepare_rows  # noqa: E402
from src.training.transforms import PadToSquare  # noqa: E402

PANEL_TILE = 128


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KID and memorisation of generated wound crops.")
    parser.add_argument("--samples", type=Path, nargs="+", required=True, help="Folders written by generate_crops.py.")
    parser.add_argument("--train-dir", type=Path, required=True, help="Generator dataset (crops.csv) the generators saw.")
    parser.add_argument("--manifest", type=Path, default=DATA_DIR / "roi_box/roi_manifest.csv")
    parser.add_argument("--image-root", type=Path, required=True, help="Per-box crop folder, for the validation crops.")
    parser.add_argument("--detector", type=Path, required=True, help="StyleGAN's Inception TorchScript pickle.")
    parser.add_argument("--stylegan-code", type=Path, default=None,
                        help="NVlabs/stylegan3 checkout; its detector pickle needs the repo's torch_utils.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Outside the repository.")
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--size", type=int, default=256, help="Side the images are read at before the detector.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def kid(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, subsets: int = 100, max_size: int = 1000) -> float:
    """Kernel Inception Distance (unbiased MMD², cubic polynomial kernel), as StyleGAN computes it."""
    n = x.shape[1]
    m = min(len(x), len(y), max_size)
    if m < 2:
        return float("nan")
    t = 0.0
    for _ in range(subsets):
        a_, b_ = x[rng.choice(len(x), m, replace=False)], y[rng.choice(len(y), m, replace=False)]
        a = (a_ @ a_.T / n + 1) ** 3 + (b_ @ b_.T / n + 1) ** 3
        b = (a_ @ b_.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    return float(t / subsets / m)


def nearest(x: np.ndarray, reference: np.ndarray) -> tuple:
    """Index of and cosine similarity to each row's nearest reference row."""
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    rn = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    sim = xn @ rn.T
    index = sim.argmax(axis=1)
    return index, sim[np.arange(len(x)), index]


def load_square(path: Path, size: int, pad: PadToSquare) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(pad(image.convert("RGB")).resize((size, size), Image.LANCZOS))


def features(paths: Sequence[Path], detector, size: int, device: torch.device, batch: int = 64) -> np.ndarray:
    pad, out = PadToSquare(), []
    for start in range(0, len(paths), batch):
        pixels = np.stack([load_square(p, size, pad) for p in paths[start:start + batch]])
        x = torch.from_numpy(pixels).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            out.append(detector(x, return_features=True).cpu().numpy().astype(np.float64))
    return np.concatenate(out) if out else np.zeros((0, 2048))


def tile_row(paths: Sequence[Path], size: int = PANEL_TILE) -> Image.Image:
    pad = PadToSquare()
    row = Image.new("RGB", (size * max(len(paths), 1), size), (255, 255, 255))
    for i, p in enumerate(paths):
        row.paste(Image.fromarray(load_square(p, size, pad)), (i * size, 0))
    return row


def stack(rows: Sequence[Image.Image]) -> Image.Image:
    width = max(r.width for r in rows)
    canvas = Image.new("RGB", (width, sum(r.height for r in rows) + 4 * (len(rows) - 1)), (255, 255, 255))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + 4
    return canvas


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = args.output_dir.resolve()
    if out == REPO_ROOT or REPO_ROOT in out.parents:
        raise ValueError(f"--output-dir {out} is inside the repository; patient-derived images stay outside it")
    (out / "panels").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stylegan_code:
        sys.path.insert(0, str(args.stylegan_code))
    with open(args.detector, "rb") as f:
        detector = pickle.load(f).to(device).eval()
    rng = np.random.default_rng(args.seed)

    # Real training crops (what the generators saw) and real validation crops (new wounds)
    train = pd.read_csv(args.train_dir / "crops.csv")
    train_paths = [args.train_dir / f for f in train["file"]]
    df, _ = prepare_rows(load_manifest(args.manifest, "crop"), "crop", args.image_root)
    val = df[df["split"] == "validation"].reset_index(drop=True)
    f_train = features(train_paths, detector, args.size, device)
    f_val = features(val["image_path"].tolist(), detector, args.size, device)
    _, val_sim = nearest(f_val, f_train)
    threshold = float(np.quantile(val_sim, args.quantile))
    print(f"Validation-to-training nearest similarity: median {np.median(val_sim):.3f}, "
          f"{args.quantile:.0%} quantile {threshold:.3f}", flush=True)

    rows, near = [], []
    for folder in args.samples:
        samples = pd.read_csv(folder / "samples.csv")
        for (generator, name), group in samples.groupby(["generator", "class"], sort=False):
            paths = [folder / f for f in group["file"]]
            f_gen = features(paths, detector, args.size, device)
            real_train = f_train[(train["class"] == name).to_numpy()]
            real_val = f_val[(val["canonical_wound_label"] == name).to_numpy()]
            index, sim = nearest(f_gen, f_train)
            flagged = sim > threshold
            rows.append({"generator": generator, "class": name, "generated": len(paths), "real_train": len(real_train),
                         "real_validation": len(real_val), "kid_vs_train": kid(f_gen, real_train, rng),
                         "kid_vs_validation": kid(f_gen, real_val, rng),
                         "kid_real_validation_vs_train": kid(real_val, real_train, rng),
                         "nearest_similarity_median": float(np.median(sim)), "flagged_share": float(flagged.mean()),
                         "threshold": threshold})
            near += [{"generator": generator, "class": name, "file": str(p), "nearest_train": train["file"][i],
                      "nearest_train_class": train["class"][i], "similarity": s, "flagged": bool(fl)}
                     for p, i, s, fl in zip(paths, index, sim, flagged)]
            print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rows[-1].items()), flush=True)

            # Panels: 16 generated over 16 real training crops; the 8 closest generated-training pairs
            slug = f"{generator}_{name.replace(' ', '_')}"
            real_paths = [train_paths[i] for i in np.flatnonzero((train["class"] == name).to_numpy())]
            picks = rng.choice(len(real_paths), min(16, len(real_paths)), replace=False)
            stack([tile_row(paths[:8]), tile_row(paths[8:16]), tile_row([real_paths[i] for i in picks[:8]]),
                   tile_row([real_paths[i] for i in picks[8:16]])]).save(out / "panels" / f"{slug}_samples.png")
            closest = np.argsort(-sim)[:8]
            stack([tile_row([paths[i] for i in closest]), tile_row([train_paths[index[i]] for i in closest])]).save(
                out / "panels" / f"{slug}_closest.png")

    pd.DataFrame(rows).to_csv(out / "quality.csv", index=False)
    pd.DataFrame(near).to_csv(out / "nearest.csv", index=False)
    print(f"Wrote quality.csv, nearest.csv and panels to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
