"""Tests for the blinded real-or-generated rating set (T34)."""

import shutil
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from PIL import Image

from src.data.export_generator_crops import main as export_main
from src.evaluation.rating_set import main as rating_main
from tests.synthetic_data import build_synthetic_box_dataset, build_synthetic_dataset


def fake_samples(folder: Path, generator: str, source: Path, n: int = 4) -> list:
    # n "generated" scratch crops (copies of one training crop) and their samples.csv
    (folder / "scratch").mkdir(parents=True)
    rows = []
    for i in range(n):
        shutil.copy(source, folder / "scratch" / f"{generator}_{i}.png")
        rows.append({"file": f"scratch/{generator}_{i}.png", "class": "scratch", "generator": generator,
                     "checkpoint": "-", "seed": i})
    pd.DataFrame(rows).to_csv(folder / "samples.csv", index=False)
    return [str(folder / r["file"]) for r in rows]


def test_rating_set_is_blind_and_scored(tmp_path):
    root = tmp_path / "wounds"
    boxes = build_synthetic_box_dataset(root, build_synthetic_dataset(root))
    train_dir = tmp_path / "train"
    export_main(["--manifest", str(boxes["box_manifest"]), "--image-root", str(boxes["box_root"]),
                 "--output-dir", str(train_dir), "--size", "32"])
    crop = train_dir / pd.read_csv(train_dir / "crops.csv").query("`class` == 'scratch'")["file"].iloc[0]
    gan = fake_samples(tmp_path / "gan", "stylegan2", crop)
    sd = fake_samples(tmp_path / "sd", "sd15_lora", crop)
    # One GAN crop is flagged as a near-copy and must never be shown
    nearest = pd.DataFrame({"file": gan + sd, "flagged": [True] + [False] * 7})
    nearest.to_csv(tmp_path / "nearest.csv", index=False)

    out, keys = tmp_path / "rating", tmp_path / "key"
    rating_main(["make", "--samples", str(tmp_path / "gan"), str(tmp_path / "sd"), "--nearest", str(tmp_path / "nearest.csv"),
                 "--train-dir", str(train_dir), "--classes", "scratch", "--per-source", "2",
                 "--output-dir", str(out), "--key-dir", str(keys)])
    sheet = pd.read_csv(out / "rating_sheet.csv")
    key = pd.read_csv(keys / "answer_key.csv")
    assert not (out / "answer_key.csv").exists() and list(sheet.columns) == ["item", "class", "rating"]
    assert key["source"].value_counts().to_dict() == {"real": 2, "stylegan2": 2, "sd15_lora": 2}
    assert gan[0] not in set(key["file"])
    for item in sheet["item"]:
        with Image.open(out / "images" / item) as image:
            assert image.size == (256, 256)

    # The rater calls everything real except the diffusion crops
    sheet["rating"] = ["generated" if s == "sd15_lora" else "real" for s in key.set_index("item").loc[sheet["item"], "source"]]
    sheet.to_csv(out / "rating_sheet.csv", index=False)
    assert rating_main(["score", "--sheet", str(out / "rating_sheet.csv"), "--key", str(keys / "answer_key.csv")]) == 0
