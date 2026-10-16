#!/usr/bin/env python3
"""Detector arm 4Z: zero-shot Grounding DINO and OWLv2 with the four frozen prompts (T19 §2.2).

16 configurations = 2 models x 4 prompts x tiling off/on, named ``<model>_<prompt>[_tiled]`` and
listed by ``config_order`` from simplest to most complex (fewer prompt words, then no tiling, then
Grounding DINO before OWLv2), which is the tie order ``select.py`` applies.

Development run (no test photo is opened)::

    zero_shot.py --model gdino --splits train validation --output-dir <arm>/detections

writes ``<config>.csv`` for the model's 8 configurations. After ``select.py`` has frozen the arm,
``--splits test --config <chosen> --selection <selection.json>`` runs the test photos.

Scores are the per-box maximum over text tokens (Grounding DINO) or text queries (OWLv2), as the
``transformers`` post-processors compute them, with no threshold; each pass keeps its
``RAW_PER_PASS`` highest-scoring boxes. Tiling adds 2 x 2 overlapping tiles (each 60% of the photo's
width and height, at the four corners) to the full-photo pass; ``candidate_boxes`` later merges
them with NMS. Weights load offline (``HF_HUB_OFFLINE=1``, ``HF_HOME`` set; safetensors only).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.detection.common import (  # noqa: E402
    PHOTO,
    detection_rows,
    load_photo,
    photo_table,
    test_allowed,
    write_detections,
)


MODELS = {"gdino": "IDEA-Research/grounding-dino-base", "owlv2": "google/owlv2-large-patch14-ensemble"}
# Frozen 2026-10-16 (T19 §2.2); P4 follows the proposal's definitions (DiMaio 2021, GLOSSARY.md).
PROMPTS: Dict[str, List[str]] = {
    "P1": ["wound"],
    "P2": ["injury on skin"],
    "P3": ["cut wound", "graze wound", "gunshot wound", "contusion", "laceration", "scratch", "penetration wound"],
    "P4": [
        "a straight incised cut with clean sharp edges on skin",
        "a superficial scraped abrasion on skin",
        "a narrow linear scrape abrasion on skin",
        "a bruise with bleeding under intact skin",
        "an irregular torn wound with abraded bruised edges and tissue bridges",
        "a puncture or stab wound made by a pointed object",
        "a round bullet wound on skin",
    ],
}
TILE_FRACTION = 0.6
RAW_PER_PASS = 300

Detect = Callable[[object, List[str]], Tuple[np.ndarray, np.ndarray]]  # (PIL image, phrases) -> boxes, scores


def grounding_text(phrases: Sequence[str]) -> str:
    """Grounding DINO's prompt: lowercase phrases, each ending with ' .'."""
    return " ".join(f"{p.lower()} ." for p in phrases)


def config_name(model: str, prompt: str, tiled: bool) -> str:
    return f"{model}_{prompt}" + ("_tiled" if tiled else "")


def config_order() -> List[str]:
    """All 16 configurations, simplest first: prompt words, then no tiling, then gdino before owlv2."""
    return [config_name(m, p, t) for p in PROMPTS for t in (False, True) for m in MODELS]


def tiles(width: int, height: int, fraction: float = TILE_FRACTION) -> List[Tuple[int, int, int, int]]:
    """The four corner tiles ``(x0, y0, x1, y1)``, each ``fraction`` of the width and height."""
    tw, th = int(np.ceil(fraction * width)), int(np.ceil(fraction * height))
    return [(x0, y0, x0 + tw, y0 + th) for y0 in (0, height - th) for x0 in (0, width - tw)]


def top(boxes: np.ndarray, scores: np.ndarray, k: int = RAW_PER_PASS) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="stable")[:k]
    return boxes[order], scores[order]


def detect_photo(detect: Detect, image, phrases: List[str], tiled: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Raw boxes in photo pixels: the full-photo pass plus, when ``tiled``, the four tile passes."""
    boxes, scores = top(*detect(image, phrases))
    parts_b, parts_s = [boxes], [scores]
    if tiled:
        for x0, y0, x1, y1 in tiles(*image.size):
            b, s = top(*detect(image.crop((x0, y0, x1, y1)), phrases))
            parts_b.append(b + np.array([x0, y0, x0, y0], dtype=float))
            parts_s.append(s)
    return np.concatenate(parts_b).reshape(-1, 4), np.concatenate(parts_s)


# ---------- the two models (need transformers and the cached weights) ----------

def load_detector(model: str, device: str) -> Detect:
    import torch
    from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, Owlv2ForObjectDetection,
                              Owlv2Processor)

    name = MODELS[model]
    if model == "gdino":
        processor = AutoProcessor.from_pretrained(name)
        net = AutoModelForZeroShotObjectDetection.from_pretrained(name, use_safetensors=True).to(device).eval()

        @torch.no_grad()
        def detect(image, phrases):
            inputs = processor(images=image, text=grounding_text(phrases), return_tensors="pt").to(device)
            out = net(**inputs)
            scores = out.logits[0].sigmoid().max(dim=-1).values  # max over text tokens, as the post-processor
            cx, cy, w, h = out.pred_boxes[0].unbind(-1)  # normalised to the photo
            width, height = image.size
            boxes = torch.stack([(cx - w / 2) * width, (cy - h / 2) * height,
                                 (cx + w / 2) * width, (cy + h / 2) * height], dim=-1)
            return boxes.float().cpu().numpy(), scores.float().cpu().numpy()
        return detect

    processor = Owlv2Processor.from_pretrained(name)
    net = Owlv2ForObjectDetection.from_pretrained(name, use_safetensors=True).to(device).eval()

    @torch.no_grad()
    def detect(image, phrases):
        inputs = processor(text=[list(phrases)], images=image, return_tensors="pt").to(device)
        out = net(**inputs)
        scores = out.logits[0].sigmoid().max(dim=-1).values  # max over text queries
        cx, cy, w, h = out.pred_boxes[0].unbind(-1)  # normalised to the photo padded to a square
        side = float(max(image.size))
        boxes = torch.stack([(cx - w / 2) * side, (cy - h / 2) * side,
                             (cx + w / 2) * side, (cy + h / 2) * side], dim=-1)
        return boxes.float().cpu().numpy(), scores.float().cpu().numpy()
    return detect


# ---------- script ----------

def run(detect: Detect, model: str, configs: List[str], photos, image_root: Path, output_dir: Path,
        log: Callable[[str], None] = print) -> None:
    """Write ``<config>.csv`` for each of ``configs`` (all of one ``model``) over ``photos``."""
    rows: Dict[str, List[Dict]] = {c: [] for c in configs}
    for n, row in enumerate(photos.to_dict("records"), start=1):
        image = load_photo(image_root, row)
        for config in configs:
            _, prompt, *tiled = config.split("_")
            boxes, scores = detect_photo(detect, image, PROMPTS[prompt], bool(tiled))
            rows[config] += detection_rows(row[PHOTO], boxes, scores, *image.size)
        if n % 50 == 0 or n == len(photos):
            log(f"{model}: {n}/{len(photos)} photos")
    for config in configs:
        write_detections(rows[config], output_dir / f"{config}.csv")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detector arm 4Z: zero-shot detections (T19).")
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", default=None, help="One configuration only (required for test).")
    parser.add_argument("--selection", type=Path, default=None, help="The arm's selection.json (test only).")
    parser.add_argument("--box-manifest", type=Path, default=Path("data/roi_box/roi_manifest.csv"))
    parser.add_argument("--image-root", type=Path, required=True, help="Raw photo tree (wound_ml_7classes).")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if "test" in args.splits:
        if args.config is None:
            parser.error("test needs --config (the chosen one) and --selection")
        test_allowed(args.selection, args.config)
    if args.config is not None and not args.config.startswith(f"{args.model}_"):
        parser.error(f"--config {args.config} is not a {args.model} configuration")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configs = [args.config] if args.config else [c for c in config_order() if c.startswith(f"{args.model}_")]
    photos = photo_table(args.box_manifest, args.splits)
    detect = load_detector(args.model, args.device)
    run(detect, args.model, configs, photos, args.image_root, args.output_dir)
    print(f"Wrote {len(configs)} detections files for {len(photos)} photos ({'+'.join(args.splits)}) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
