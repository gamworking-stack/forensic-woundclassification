"""T21 single-box analysis: full image vs one crop on the test photos with exactly one box."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.evaluation.evaluate import FAMILY_ORDER
from src.evaluation.single_box import main as single_box_main, photo_stem, single_box_photos


CLASSES = ["contusion", "cut wound"]
# (case, file name without extension, class, boxes); 'B' has two boxes and must be left out
PHOTOS = [("1", "A", "contusion", 1), ("1", "B", "contusion", 2), ("2", "C", "cut wound", 1),
          ("3", "D", "cut wound", 1), ("3", "E", "contusion", 1), ("4", "F", "cut wound", 1),
          ("5", "G", "contusion", 1)]
RAW_FOLDER = {"contusion": "contusion", "cut wound": "cut"}


def prediction_rows(rows, wrong=()):
    # Prediction-file rows for (relative_path, case, class); names in ``wrong`` get the other class
    out = []
    for relative_path, case, label in rows:
        p = 0.2 if (label == CLASSES[0]) != (relative_path in wrong) else 0.8
        out.append({"relative_path": relative_path, "pseudonimized_id": case, "split": "test",
                    "true_class": label, "predicted_class": CLASSES[int(p > 0.5)],
                    "prob_contusion": 1 - p, "prob_cut_wound": p})
    return pd.DataFrame(out)


def write_scenario(root: Path, input_path: str, frame: pd.DataFrame) -> None:
    # The same test file for every family; summarize only needs test_metrics.json to exist
    for family in FAMILY_ORDER:
        folder = root / input_path / family
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_csv(folder / "test_predictions.csv", index=False)
        (folder / "test_metrics.json").write_text(json.dumps({}), encoding="utf-8")


@pytest.fixture()
def runs(tmp_path):
    # Per-box and union manifests, plus full, union and per-crop test predictions
    box, union, full, roi, crop = [], [], [], [], []
    for case, name, label, n in PHOTOS:
        photo = f"{label}/{name}"
        common = {"pseudonimized_id": case, "filename": name, "canonical_wound_label": label,
                  "relative_path": photo, "split": "test"}
        union.append({**common, "roi_relative_path": f"test/{label}/{name}_roi.jpg"})
        full.append((f"{RAW_FOLDER[label]}/{name}.JPG", case, label))
        roi.append((f"test/{label}/{name}_roi.jpg", case, label))
        for k in range(n):
            box.append({**common, "roi_relative_path": f"test/{label}/{name}_bbox{k + 1:02d}.jpg"})
            crop.append((f"test/{label}/{name}_bbox{k + 1:02d}.jpg", case, label))
    # A train photo with one box must not enter the test subset
    box.append({"pseudonimized_id": "9", "filename": "Z", "canonical_wound_label": "contusion",
                "relative_path": "contusion/Z", "split": "train", "roi_relative_path": "train/contusion/Z_bbox01.jpg"})
    pd.DataFrame(box).to_csv(tmp_path / "box_manifest.csv", index=False)
    pd.DataFrame(union).to_csv(tmp_path / "roi_manifest.csv", index=False)
    # Full images miss C and D; both crop paths classify everything correctly
    write_scenario(tmp_path / "main", "full", prediction_rows(full, wrong={"cut/C.JPG", "cut/D.JPG"}))
    write_scenario(tmp_path / "main", "roi", prediction_rows(roi))
    write_scenario(tmp_path / "crop_run", "crop", prediction_rows(crop))
    args = ["--main-root", str(tmp_path / "main"), "--crop-root", str(tmp_path / "crop_run"),
            "--box-manifest", str(tmp_path / "box_manifest.csv"), "--roi-manifest", str(tmp_path / "roi_manifest.csv"),
            "--bootstrap", "50"]
    return tmp_path, args


def test_photo_stem_drops_only_image_extensions():
    assert photo_stem("cut/IMG_0235.JPG") == "IMG_0235"
    assert photo_stem("IMG_0235") == "IMG_0235"
    assert photo_stem("P1.2") == "P1.2"


def test_single_box_photos_keeps_test_photos_with_one_box(runs):
    tmp_path, _ = runs
    photos = single_box_photos(tmp_path / "box_manifest.csv")
    assert sorted(photos["filename"]) == ["A", "C", "D", "E", "F", "G"]


def test_full_vs_crop_on_single_box_photos(runs):
    tmp_path, args = runs
    assert single_box_main(args) == 0
    out = tmp_path / "crop_run" / "summary"
    # Every scenario is scored on the same 6 photos and 5 cases
    metrics = pd.read_csv(out / "single_box_metrics.csv")
    assert len(metrics) == 3 * len(FAMILY_ORDER)
    assert set(metrics["test_photos"]) == {6} and set(metrics["test_cases"]) == {5}
    assert set(metrics.loc[metrics["input_path"] == "full", "scenario_id"]) == {f"S{i:02d}" for i in range(1, 16, 2)}
    assert set(metrics.loc[metrics["input_path"] == "crop", "scenario_id"]) == {f"S{i}" for i in range(25, 33)}
    # Full misses 2 of 6, both crop paths none
    full = metrics[metrics["input_path"] == "full"]["accuracy"]
    assert full.round(4).eq(round(4 / 6, 4)).all()
    assert metrics[metrics["input_path"] != "full"]["accuracy"].eq(1.0).all()
    for name in ("crop", "roi"):
        table = pd.read_csv(out / f"single_box_{name}_vs_full.csv")
        assert (table.loc[table["metric"] == "macro_f1", "difference"] > 0).all()
        assert (out / f"single_box_{name}_vs_full_macro_f1.png").is_file()
    # The two-box photo's crops are not in the per-crop subset
    kept = pd.read_csv(tmp_path / "crop_run" / "single_box" / "crop" / FAMILY_ORDER[0] / "test_predictions.csv")
    assert not kept["relative_path"].str.contains("/B_").any()
    counts = pd.read_csv(out / "single_box_counts.csv").set_index("class")
    assert counts.loc["all", "photos"] == 6 and counts.loc["contusion", "photos"] == 3


def test_missing_single_box_photo_is_an_error(runs):
    tmp_path, args = runs
    # Drop photo F from one full-image file
    path = tmp_path / "main" / "full" / FAMILY_ORDER[0] / "test_predictions.csv"
    frame = pd.read_csv(path, dtype=str)
    frame[frame["relative_path"] != "cut/F.JPG"].to_csv(path, index=False)
    with pytest.raises(ValueError, match="1 single-box photos missing"):
        single_box_main(args)
