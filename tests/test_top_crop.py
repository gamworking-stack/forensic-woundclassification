"""T22 top crop: one crop per photo, the most confident, scored next to the full image."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.evaluation.evaluate import FAMILY_ORDER, SPLITS, evaluate_scenario
from src.evaluation.top_crop import main as top_crop_main
from src.training.photo import PHOTO_COLUMN, photo_top_crop_frame


CLASSES = ["contusion", "cut wound"]
# (case, photo, class, [(crop, p_cut)]): photo B's confident crop is wrong, its unsure one right
PHOTOS = [("1", "A", "contusion", [("A1", 0.1)]),
          ("1", "B", "contusion", [("B1", 0.4), ("B2", 0.9)]),
          ("2", "C", "cut wound", [("C1", 0.8)]),
          ("3", "D", "cut wound", [("D1", 0.7), ("D2", 0.55), ("D3", 0.35)]),
          ("4", "E", "contusion", [("E1", 0.3)]),
          ("5", "F", "cut wound", [("F1", 0.95)])]


def row(relative_path, case, split, label, p_cut, **extra):
    # One prediction row with the two class probabilities
    return {"relative_path": relative_path, **extra, "pseudonimized_id": case, "split": split, "true_class": label,
            "predicted_class": CLASSES[int(p_cut > 0.5)], "prob_contusion": 1 - p_cut, "prob_cut_wound": p_cut}


@pytest.fixture()
def runs(tmp_path):
    # Per-box manifest, per-crop predictions and scored full-image predictions, same photos in both splits
    manifest, crops, full = [], {s: [] for s in SPLITS}, {s: [] for s in SPLITS}
    for split in SPLITS:
        for case, photo, label, boxes in PHOTOS:
            case_id = f"{split[0]}{case}"  # validation and test cases differ
            for crop, p in boxes:
                crop_rel = f"{split}/{label}/{photo}_{crop}.jpg"
                manifest.append({"relative_path": f"{label}/{split}_{photo}", "roi_relative_path": crop_rel, "split": split})
                crops[split].append(row(crop_rel, case_id, split, label, p))
            # The full image is right for every photo but D
            full[split].append(row(f"raw/{split}_{photo}.JPG", case_id, split, label,
                                   0.3 if (label == "contusion") != (photo == "D") else 0.7))
    pd.DataFrame(manifest).to_csv(tmp_path / "box_manifest.csv", index=False)
    for family in FAMILY_ORDER:
        crop_dir = tmp_path / "crop_run" / "crop" / family
        full_dir = tmp_path / "main" / "full" / family
        for folder, data in ((crop_dir, crops), (full_dir, full)):
            folder.mkdir(parents=True)
            for split in SPLITS:
                pd.DataFrame(data[split]).to_csv(folder / f"{split}_predictions.csv", index=False)
        evaluate_scenario(full_dir, f"full {family}", 20, 42)
    args = ["--crop-root", str(tmp_path / "crop_run"), "--main-root", str(tmp_path / "main"),
            "--box-manifest", str(tmp_path / "box_manifest.csv"), "--bootstrap", "20"]
    return tmp_path, args


def test_top_crop_keeps_the_most_confident_crop():
    crops = pd.DataFrame([row(f"{c}.jpg", case, "test", label, p, **{PHOTO_COLUMN: photo})
                          for case, photo, label, boxes in PHOTOS for c, p in boxes])
    frame = photo_top_crop_frame(crops).set_index("relative_path")
    # One row per photo; B keeps its wrong but confident crop, D its most confident crop (top class 0.7 > 0.65 > 0.55)
    assert list(frame.index) == ["A", "B", "C", "D", "E", "F"]
    assert frame.loc["B", "top_crop"] == "B2.jpg" and frame.loc["B", "predicted_class"] == "cut wound"
    assert frame.loc["D", "top_crop"] == "D1.jpg" and frame.loc["D", "n_crops"] == 3
    assert frame.loc["D", "prob_cut_wound"] == pytest.approx(0.7)


def test_top_crop_table_lists_full_and_top_crop(runs):
    tmp_path, args = runs
    assert top_crop_main(args) == 0
    table = pd.read_csv(tmp_path / "crop_run" / "summary" / "top_crop_table.csv")
    # Both inputs per family, every one on the same 6 test photos
    assert len(table) == 2 * len(FAMILY_ORDER)
    assert set(table["input"]) == {"full image", "top crop"} and set(table["test_photos"]) == {6}
    assert set(table.loc[table["input"] == "top crop", "scenario_id"]) == {f"S{i}" for i in range(25, 33)}
    # Full image misses D only; top crop misses B (its confident crop is wrong)
    assert table.loc[table["input"] == "full image", "accuracy"].round(4).eq(round(5 / 6, 4)).all()
    assert table.loc[table["input"] == "top crop", "accuracy"].round(4).eq(round(5 / 6, 4)).all()
    per_class = pd.read_csv(tmp_path / "crop_run" / "summary" / "top_crop_per_class.csv")
    assert set(per_class["class"]) == set(CLASSES) and len(per_class) == 2 * 2 * len(FAMILY_ORDER)
    top = pd.read_csv(tmp_path / "crop_run" / "crop" / FAMILY_ORDER[0] / "top_crop" / "test_predictions.csv")
    assert len(top) == 6 and (tmp_path / "crop_run" / "crop" / FAMILY_ORDER[0] / "top_crop" / "test_metrics.json").is_file()


def test_crop_missing_from_manifest_is_an_error(runs):
    tmp_path, args = runs
    manifest = pd.read_csv(tmp_path / "box_manifest.csv")
    manifest[manifest["roi_relative_path"] != "test/cut wound/F_F1.jpg"].to_csv(tmp_path / "box_manifest.csv", index=False)
    with pytest.raises(ValueError, match="not in the per-box manifest"):
        top_crop_main(args)
