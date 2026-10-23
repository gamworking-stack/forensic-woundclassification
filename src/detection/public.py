#!/usr/bin/env python3
"""Detector arm 4P: the public Hanterdsith forensic YOLOv8 models (T19 §2.2).

Three steps, each its own subcommand:

``prepare`` (a throwaway job whose only writable bind is a scratch folder): take the three
checkpoints at the pinned revision (downloaded beforehand), scan each pickle with ``picklescan``, load it once with
``ultralytics`` and re-save the weights as ``safetensors`` plus a JSON with the architecture and
class names. ``weights.json`` records the SHA-256 of every original and re-saved file. Every later
step loads only the re-saved files: no pickle is opened again.

``detect``: 6 configurations = 3 sizes x colour / grayscale input, named ``<size>_<colour|grey>``
and listed by ``config_order`` smallest model first, colour before grey (the tie order of
``select.py``). Each photo is predicted at 640 px (the training size) with confidence 0.001,
class-agnostic NMS at IoU 0.7 and up to 300 boxes; every one of the 8 source classes counts as
"wound", and the class is kept in ``source_class`` for the appendix count.

``willie-smoke``: the single load test of WILLIE MINI from its public code (T19 §2.2). It writes
``willie_smoke.json``; if WILLIE loads, its inference adapter is still to be written before it can
join the selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
from src.data.paths import DATA_DIR  # noqa: E402


REPO = "trapezius60/forensic_wound_detection"
REVISION = "0c61ba140e2c800b25e562e092b777082c7a10fe"
CHECKPOINTS = {"v8n": "best.pt", "v8m": "best_yolov8m_roboflow_v7.pt", "v8x": "best_yolov8x_robov9.pt"}
INPUTS = ("colour", "grey")
IMAGE_SIZE = 640
CONFIDENCE = 0.001
YOLO_NMS_IOU = 0.7
MAX_DETECTIONS = 300

Detect = Callable[[object], Tuple[np.ndarray, np.ndarray, List[str]]]  # PIL image -> boxes, scores, classes


def config_order() -> List[str]:
    # Smallest model first, colour before grey: the tie order of select.py
    return [f"{size}_{colour}" for size in CHECKPOINTS for colour in INPUTS]


def as_input(image, colour: str):
    """Grayscale input is the photo's luminance repeated in three channels."""
    return image.convert("L").convert("RGB") if colour == "grey" else image


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------- prepare: scan, load once, re-save ----------

def prepare(weights_dir: Path, snapshot_dir: Path) -> Dict:
    """``snapshot_dir`` holds the three files at the pinned revision (downloaded beforehand, on the login node)."""
    from picklescan.scanner import scan_file_path
    from safetensors.torch import save_file
    from ultralytics import YOLO

    weights_dir.mkdir(parents=True, exist_ok=True)
    # weights.json records where the checkpoints came from and their hashes
    record: Dict = {"repo": REPO, "revision": REVISION, "checkpoints": {},
                    "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for size, filename in CHECKPOINTS.items():
        # Scan the pickle for dangerous code before it is ever opened
        original = snapshot_dir / filename
        scan = scan_file_path(str(original))
        if scan.scan_err or scan.infected_files or scan.issues_count:
            raise SystemExit(f"picklescan flagged {original}: errors={scan.scan_err}, "
                             f"infected={scan.infected_files}, issues={scan.issues_count}")
        # Open it once and copy its weights as plain tensors
        model = YOLO(str(original)).model.float()  # the only time a pickle is opened
        state = {k: v.contiguous().clone() for k, v in model.state_dict().items()}
        # Save the weights as safetensors (no code inside) and the architecture and class names as JSON
        saved = weights_dir / f"{size}.safetensors"
        save_file(state, str(saved))
        meta = {"yaml": model.yaml, "names": {int(k): v for k, v in model.names.items()}}
        (weights_dir / f"{size}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        # Record both files' hashes so the verification can check them
        record["checkpoints"][size] = {"file": filename, "original_sha256": sha256(original),
                                       "resaved_sha256": sha256(saved), "picklescan_issues": int(scan.issues_count),
                                       "classes": meta["names"]}
        print(f"{size}: {filename} scanned clean, re-saved to {saved}", flush=True)
    (weights_dir / "weights.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


# ---------- detect ----------

def load_detector(weights_dir: Path, size: str, colour: str, device: str) -> Detect:
    import yaml
    from safetensors.torch import load_file
    from ultralytics import YOLO

    # Rebuild the YOLO architecture from the saved JSON, then load the safetensors weights
    meta = json.loads((weights_dir / f"{size}.json").read_text(encoding="utf-8"))
    names = {int(k): v for k, v in meta["names"].items()}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(meta["yaml"], f)
    yolo = YOLO(f.name, task="detect")  # architecture only, no weights
    yolo.model.load_state_dict(load_file(str(weights_dir / f"{size}.safetensors")), strict=True)
    yolo.model.names = names
    yolo.model.eval()

    def detect(image):
        # Predict at 640 px with a very low confidence floor, keeping every class (class-agnostic NMS)
        result = yolo.predict(source=as_input(image, colour), imgsz=IMAGE_SIZE, conf=CONFIDENCE, iou=YOLO_NMS_IOU,
                              agnostic_nms=True, max_det=MAX_DETECTIONS, device=device, verbose=False)[0]
        boxes = result.boxes.xyxy.float().cpu().numpy()  # pixels of the input photo
        scores = result.boxes.conf.float().cpu().numpy()
        classes = [names[int(c)] for c in result.boxes.cls.cpu().numpy()]
        return boxes, scores, classes
    return detect


def run(detectors: Dict[str, Detect], photos, image_root: Path, output_dir: Path,
        log: Callable[[str], None] = print) -> None:
    # Open each photo once and run every configuration on it
    rows: Dict[str, List[Dict]] = {c: [] for c in detectors}
    for n, row in enumerate(photos.to_dict("records"), start=1):
        image = load_photo(image_root, row)
        for config, detect in detectors.items():
            boxes, scores, classes = detect(image)
            rows[config] += detection_rows(row[PHOTO], boxes, scores, *image.size, extra={"source_class": classes})
        # Progress line every 50 photos
        if n % 50 == 0 or n == len(photos):
            log(f"4P: {n}/{len(photos)} photos")
    # One detections file per configuration, with the YOLO class of each box
    for config in detectors:
        write_detections(rows[config], output_dir / f"{config}.csv", extra_columns=("source_class",))


# ---------- WILLIE smoke test ----------

def willie_smoke(repo: Path, weights: Path, output: Path) -> Dict:
    """One try to build WILLIE MINI from ``ablations/mini_ablation_FINAL.py`` and load a fold's weights."""
    import importlib.util
    import torch

    # The single try: import WILLIE's public model code, build it, load fold 0's weights
    result: Dict = {"repo": str(repo), "weights": str(weights), "loaded": False,
                    "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        spec = importlib.util.spec_from_file_location("willie_mini", repo / "ablations" / "mini_ablation_FINAL.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.WILLIEModel()
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)  # trusted MIT source (T19 §2.2)
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        missing, unexpected = model.load_state_dict(state, strict=False)
        result.update({"loaded": not missing, "missing_keys": len(missing), "unexpected_keys": len(unexpected),
                       "weights_sha256": sha256(weights)})
    except Exception as error:  # the single try: any failure leaves WILLIE out, with the reason
        result["error"] = f"{type(error).__name__}: {error}"
    # Record the outcome (loaded or the error) either way
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


# ---------- script ----------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detector arm 4P: public forensic YOLOv8 (T19).")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Scan, load once and re-save the three checkpoints.")
    p.add_argument("--weights-dir", type=Path, required=True)
    p.add_argument("--snapshot-dir", type=Path, required=True,
                   help="$HF_HOME/hub/models--trapezius60--forensic_wound_detection/snapshots/<revision>")
    d = sub.add_parser("detect", help="Detections files for the 4P configurations.")
    d.add_argument("--weights-dir", type=Path, required=True)
    d.add_argument("--splits", nargs="+", choices=("train", "validation", "test"), required=True)
    d.add_argument("--output-dir", type=Path, required=True)
    d.add_argument("--config", default=None, help="One configuration only (required for test).")
    d.add_argument("--selection", type=Path, default=None)
    d.add_argument("--box-manifest", type=Path, default=DATA_DIR / "roi_box/roi_manifest.csv")
    d.add_argument("--image-root", type=Path, required=True)
    d.add_argument("--device", default="0")
    w = sub.add_parser("willie-smoke", help="The one WILLIE MINI load test.")
    w.add_argument("--repo", type=Path, required=True, help="A clone of github.com/Qian-Group-HRI/Willie.")
    w.add_argument("--weights", type=Path, required=True, help="mini/willie_mini_fold0_best.pt")
    w.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    # Detecting test photos needs the chosen configuration and a passing selection.json
    if args.command == "detect":
        if "test" in args.splits:
            if args.config is None:
                parser.error("test needs --config (the chosen one) and --selection")
            test_allowed(args.selection, args.config)
        if args.config is not None and args.config not in config_order():
            parser.error(f"--config must be one of {config_order()}")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    # Run the requested subcommand
    if args.command == "prepare":
        prepare(args.weights_dir, args.snapshot_dir)
        return 0
    if args.command == "willie-smoke":
        result = willie_smoke(args.repo, args.weights, args.output)
        print(f"WILLIE MINI {'loaded' if result['loaded'] else 'did not load'}: {result.get('error', '')} -> {args.output}")
        return 0
    # detect: one configuration (test) or all six (development)
    configs = [args.config] if args.config else config_order()
    detectors = {c: load_detector(args.weights_dir, *c.split("_"), args.device) for c in configs}
    photos = photo_table(args.box_manifest, args.splits)
    run(detectors, photos, args.image_root, args.output_dir)
    print(f"Wrote {len(configs)} detections files for {len(photos)} photos ({'+'.join(args.splits)}) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
