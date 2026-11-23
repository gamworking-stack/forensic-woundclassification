"""T24 subgroup analysis: test performance by age, race/ethnicity, skin tone, setting, body region and tattoo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluate import FAMILY_ORDER
from src.evaluation.metrics import summary_metrics
from src.evaluation.subgroup import (AGE_GROUPS, SKIN_TONE_GROUPS, SUBGROUP_METRICS, UNKNOWN, binned,
                                     fast_metrics, main as subgroup_main)
from src.evaluation.summarize import POOLED


CLASSES = ["contusion", "cut wound"]
RAW_FOLDER = {"contusion": "contusion", "cut wound": "cut"}
# (group ID, sheet case ID, file name, class, boxes, age, race, skin tone); groups 4 and 5 join two sheet cases
PHOTOS = [("1", "1", "A", "contusion", 1, "25", "Asian", "4"), ("1", "1", "B", "cut wound", 2, "25", "Asian", "4"),
          ("2", "2", "C", "cut wound", 1, "30", "Asian", "3"), ("3", "3", "D", "contusion", 1, "45", "Asian", "4"),
          ("3", "3", "E", "cut wound", 1, "45", "Asian", "5"), ("4", "4", "F", "contusion", 1, "50", "Asian", "4"),
          ("4", "6", "G", "cut wound", 1, "0", "0", ""), ("5", "5", "H", "contusion", 1, "-", "European", "6")]
# Per file name: (autopsy, body site, tattoo); C and D are of living patients, blank and N/A sites are unknown
EXTRA = {"A": ("1", "face", "0"), "B": ("1", "ear", "0"), "C": ("0", "leg", "0"), "D": ("0", "Hand", "0"),
         "E": ("1", "arm", "1"), "F": ("1", "chest", "0"), "G": ("", "", ""), "H": ("1", "N/A", "0")}


def prediction_rows(rows, wrong=()):
    # Prediction-file rows for (relative_path, case, class); names in ``wrong`` get the other class
    out = []
    for relative_path, case, label in rows:
        p = 0.2 if (label == CLASSES[0]) != (relative_path in wrong) else 0.8
        out.append({"relative_path": relative_path, "pseudonimized_id": case, "split": "test",
                    "true_class": label, "predicted_class": CLASSES[int(p > 0.5)],
                    "prob_contusion": 1 - p, "prob_cut_wound": p})
    return pd.DataFrame(out)


def write_scenario(root: Path, folder: str, frame: pd.DataFrame) -> None:
    # The same test file for every family
    for family in FAMILY_ORDER:
        path = root / folder.format(family=family)
        path.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path / "test_predictions.csv", index=False)


@pytest.fixture()
def runs(tmp_path):
    # Sheet (with Excel padding rows), split / union / per-box manifests, and test predictions of 4 paths
    sheet, split, union, box, full, roi, photo, crop = [], [], [], [], [], [], [], []
    for group, case, name, label, n, age, race, tone in PHOTOS:
        sheet.append({"pseudonimized_id": case, "patient_id": f"P{case}", "filename": f"{name}.JPG",
                      "skin_tone": tone, "wound_label": RAW_FOLDER[label], "race_ethnicity": race, "age": age,
                      "autopsy": EXTRA[name][0], "body_site": EXTRA[name][1], "tattoo": EXTRA[name][2],
                      "Inclusion": "TRUE"})
        common = {"pseudonimized_id": group, "filename": f"{name}.JPG", "canonical_wound_label": label,
                  "relative_path": f"{label}/{name}.JPG", "source_pseudonimized_id": case, "split": "test"}
        split.append(common)
        union.append({**common, "roi_relative_path": f"test/{label}/{name}_roi.jpg"})
        full.append((f"{RAW_FOLDER[label]}/{name}.JPG", group, label))
        roi.append((f"test/{label}/{name}_roi.jpg", group, label))
        photo.append((f"{label}/{name}.JPG", group, label))
        for k in range(n):
            box.append({**common, "roi_relative_path": f"test/{label}/{name}_bbox{k + 1:02d}.jpg"})
            crop.append((f"test/{label}/{name}_bbox{k + 1:02d}.jpg", group, label))
    # An excluded duplicate of photo A under another label, and a train photo, must both be ignored
    sheet.append({**sheet[0], "wound_label": "cut", "Inclusion": "FALSE"})
    split.append({"pseudonimized_id": "9", "filename": "Z.JPG", "canonical_wound_label": "contusion",
                  "relative_path": "contusion/Z.JPG", "source_pseudonimized_id": "9", "split": "train"})
    sheet = pd.DataFrame(sheet)
    padding = pd.DataFrame({"pseudonimized_id": ["99", "100"]}).reindex(columns=sheet.columns).fillna("")
    pd.concat([sheet, padding]).to_csv(tmp_path / "sheet.csv", index=False)
    pd.DataFrame(split).to_csv(tmp_path / "split.csv", index=False)
    pd.DataFrame(union).to_csv(tmp_path / "roi_manifest.csv", index=False)
    pd.DataFrame(box).to_csv(tmp_path / "box_manifest.csv", index=False)
    # Full images miss C (age 18-39, MST 1-3); every ROI path is right everywhere
    write_scenario(tmp_path / "main", "full/{family}", prediction_rows(full, wrong={"cut/C.JPG"}))
    write_scenario(tmp_path / "main", "roi/{family}", prediction_rows(roi))
    write_scenario(tmp_path / "box_run", "box/{family}/photo", prediction_rows(photo))
    write_scenario(tmp_path / "crop_run", "crop/{family}", prediction_rows(crop))
    args = ["--sheet", str(tmp_path / "sheet.csv"), "--main-root", str(tmp_path / "main"),
            "--box-root", str(tmp_path / "box_run"), "--crop-root", str(tmp_path / "crop_run"),
            "--output-dir", str(tmp_path / "out"), "--split-manifest", str(tmp_path / "split.csv"),
            "--roi-manifest", str(tmp_path / "roi_manifest.csv"), "--box-manifest", str(tmp_path / "box_manifest.csv"),
            "--min-photos", "2", "--bootstrap", "50"]
    return tmp_path, args


def test_binned_groups_and_unknowns():
    assert binned("25", AGE_GROUPS) == "18-39" and binned("60", AGE_GROUPS) == "60+"
    assert binned("17", AGE_GROUPS) == "<18"
    for missing in ("0", "-", "", "ไม่พบ ID"):
        assert binned(missing, AGE_GROUPS) == UNKNOWN
    assert binned("3", SKIN_TONE_GROUPS) == "MST 1-3" and binned("7", SKIN_TONE_GROUPS) == "MST 5-10"
    assert binned("", SKIN_TONE_GROUPS) == UNKNOWN


def test_fast_metrics_match_summary_metrics():
    # Random labels over 5 classes with one class never true and one never predicted
    rng = np.random.default_rng(0)
    for _ in range(20):
        y_true = rng.choice([0, 1, 2, 4], size=30)
        y_pred = rng.choice([0, 1, 3, 4], size=30)
        probs = np.full((30, 5), 0.2)
        expected = summary_metrics(y_true, y_pred, probs)
        assert np.allclose(fast_metrics(y_true, y_pred, 5), [expected[m] for m in SUBGROUP_METRICS])
    assert np.isnan(fast_metrics(np.array([], dtype=int), np.array([], dtype=int), 5)).all()


def test_subgroup_tables(runs):
    tmp_path, args = runs
    assert subgroup_main(args) == 0
    out = tmp_path / "out"
    counts = pd.read_csv(out / "subgroup_counts.csv").set_index(["attribute", "level"])
    assert counts.loc[("age_group", "18-39"), "photos"] == 3
    assert counts.loc[("age_group", UNKNOWN), "photos"] == 2
    assert counts.loc[("race_ethnicity", UNKNOWN), "photos"] == 1
    assert counts.loc[("skin_tone", "MST 4"), "photos"] == 4
    assert counts.loc[("setting", "living"), "photos"] == 2 and counts.loc[("setting", UNKNOWN), "photos"] == 1
    # Body sites merge into regions whatever their case; blank and N/A are unknown
    assert counts.loc[("body_region", "head and neck"), "photos"] == 2
    assert counts.loc[("body_region", "upper limb"), "photos"] == 2
    assert counts.loc[("body_region", UNKNOWN), "photos"] == 2
    assert counts.loc[("tattoo", "tattoo"), "photos"] == 1 and counts.loc[("tattoo", "no tattoo"), "photos"] == 6

    metrics = pd.read_csv(out / "subgroup_metrics.csv")
    # Unknown and 1-photo subgroups (European, MST 1-3, trunk, lower limb, tattoo) are counted, not scored
    scored = set(zip(metrics["attribute"], metrics["level"]))
    assert scored == {("age_group", "18-39"), ("age_group", "40-59"), ("race_ethnicity", "Asian"),
                      ("skin_tone", "MST 4"), ("skin_tone", "MST 5-10"), ("setting", "living"),
                      ("setting", "autopsy"), ("body_region", "head and neck"), ("body_region", "upper limb"),
                      ("tattoo", "no tattoo")}
    full = metrics[(metrics["input_path"] == "full") & (metrics["family"] == FAMILY_ORDER[0])].set_index("level")
    assert full.loc["18-39", "accuracy"] == pytest.approx(2 / 3)
    assert full.loc["40-59", "accuracy"] == 1.0 and full.loc["18-39", "test_cases"] == 2
    # C is the living patient the full image misses
    assert full.loc["living", "accuracy"] == 0.5 and full.loc["autopsy", "accuracy"] == 1.0
    # Per-crop rows are crops: photo B has two
    crop = metrics[(metrics["input_path"] == "crop") & (metrics["family"] == FAMILY_ORDER[0])].set_index("level")
    assert crop.loc["18-39", "test_rows"] == 4 and crop.loc["18-39", "test_photos"] == 3
    # Identical families: the pooled row equals each family
    pooled = metrics[(metrics["input_path"] == "full") & (metrics["family"] == POOLED)].set_index("level")
    assert pooled.loc["18-39", "accuracy"] == pytest.approx(2 / 3)
    assert metrics["scenario_id"].dropna().str.match(r"S\d\d").all()

    gaps = pd.read_csv(out / "subgroup_gaps.csv")
    row = gaps[(gaps["input_path"] == "full") & (gaps["family"] == FAMILY_ORDER[0]) & (gaps["level"] == "40-59")
               & (gaps["metric"] == "accuracy")].iloc[0]
    assert row["reference"] == "18-39" and row["difference"] == pytest.approx(1 / 3)
    row = gaps[(gaps["input_path"] == "full") & (gaps["family"] == FAMILY_ORDER[0]) & (gaps["level"] == "living")
               & (gaps["metric"] == "accuracy")].iloc[0]
    assert row["reference"] == "autopsy" and row["difference"] == pytest.approx(-0.5)
    assert gaps[gaps["input_path"] == "roi"]["difference"].eq(0).all()
    assert (out / "subgroup_accuracy.png").is_file() and (out / "subgroup_macro_f1.png").is_file()

    # Per-class accuracy lists every subgroup, small or unknown: full images miss C, the MST 1-3 cut wound
    classes = pd.read_csv(out / "subgroup_class_accuracy.csv")
    full = classes[(classes["input_path"] == "full") & (classes["attribute"] == "skin_tone")]
    cut = full[(full["family"] == FAMILY_ORDER[0]) & (full["class"] == "cut wound")].set_index("level")
    assert cut.loc["MST 1-3", "accuracy"] == 0.0 and cut.loc["MST 1-3", "test_photos"] == 1
    assert cut.loc["MST 4", "accuracy"] == 1.0 and cut.loc["MST 4", "correct"] == 1
    assert cut.loc["MST 4", "test_cases"] == 1
    assert set(full["level"]) == {"MST 1-3", "MST 4", "MST 5-10", UNKNOWN}
    pooled = full[(full["family"] == POOLED) & (full["class"] == "cut wound")].set_index("level")
    assert pooled.loc["MST 1-3", "accuracy"] == 0.0 and np.isnan(pooled.loc["MST 1-3", "correct"])
    crop = classes[(classes["input_path"] == "crop") & (classes["family"] == FAMILY_ORDER[0])
                   & (classes["attribute"] == "skin_tone") & (classes["class"] == "cut wound")].set_index("level")
    assert crop.loc["MST 4", "test_rows"] == 2 and crop.loc["MST 4", "test_photos"] == 1


def test_full_and_crop_only_without_box_or_union(runs):
    # T38: no per-box run and union ROI not reported; only full and crop are scored
    tmp_path, args = runs
    i = args.index("--box-root")
    args = args[:i] + args[i + 2:] + ["--no-union"]
    assert subgroup_main(args) == 0
    metrics = pd.read_csv(tmp_path / "out" / "subgroup_metrics.csv")
    assert set(metrics["input_path"]) == {"full", "crop"}


def test_output_inside_repository_is_refused(runs):
    _, args = runs
    args[args.index("--output-dir") + 1] = str(Path(__file__).resolve().parents[1] / "results" / "subgroup")
    with pytest.raises(ValueError, match="inside the repository"):
        subgroup_main(args)


def test_photo_missing_from_sheet_is_an_error(runs):
    tmp_path, args = runs
    sheet = pd.read_csv(tmp_path / "sheet.csv", dtype=str, keep_default_na=False)
    sheet[sheet["filename"] != "D.JPG"].to_csv(tmp_path / "sheet.csv", index=False)
    with pytest.raises(ValueError, match="1 test photos are not in the sheet"):
        subgroup_main(args)


def test_subgroup_photos_lists_every_split(tmp_path):
    from src.evaluation.subgroup_photos import main as photos_main
    # Tattoo on E (test) and on a train photo of another sheet case filed under group 4; G has no tattoo entry
    sheet = [{"pseudonimized_id": case, "patient_id": f"P{case}", "filename": f"{name}.JPG", "skin_tone": "4",
              "wound_label": RAW_FOLDER[label], "race_ethnicity": "Asian", "age": "30", "autopsy": "1",
              "body_site": "arm", "tattoo": tattoo, "Inclusion": "TRUE"}
             for case, name, label, tattoo in (("3", "E", "cut wound", "1"), ("1", "A", "contusion", "0"),
                                               ("6", "T", "contusion", "1"), ("4", "G", "cut wound", ""))]
    split = [{"pseudonimized_id": group, "filename": f"{name}.JPG", "canonical_wound_label": label,
              "relative_path": f"{label}/{name}.JPG", "source_pseudonimized_id": case, "split": s}
             for group, case, name, label, s in (("3", "3", "E", "cut wound", "test"), ("1", "1", "A", "contusion", "train"),
                                                 ("4", "6", "T", "contusion", "train"), ("4", "4", "G", "cut wound", "validation"))]
    pd.DataFrame(sheet).to_csv(tmp_path / "sheet.csv", index=False)
    pd.DataFrame(split).to_csv(tmp_path / "split.csv", index=False)
    out = tmp_path / "out" / "tattoo.csv"
    assert photos_main(["--sheet", str(tmp_path / "sheet.csv"), "--split-manifest", str(tmp_path / "split.csv"),
                        "--output", str(out)]) == 0
    listed = pd.read_csv(out, dtype=str)
    # Train before test, matched on the sheet's own case ID, and no sheet attribute in the file
    assert listed["relative_path"].tolist() == ["contusion/T.JPG", "cut wound/E.JPG"]
    assert listed.columns.tolist() == ["relative_path", "split", "canonical_wound_label"]


def test_subgroup_confusion_counts(runs):
    from src.evaluation.subgroup_confusion import main as confusion_main
    tmp_path, args = runs
    keep = {"--sheet", "--main-root", "--crop-root", "--split-manifest", "--roi-manifest", "--box-manifest", "--min-photos"}
    chosen = [a for i in range(0, len(args), 2) if args[i] in keep for a in args[i:i + 2]]
    out = tmp_path / "confusion"
    assert confusion_main(chosen + ["--output-dir", str(out), "--families", "vgg16", "resnet152"]) == 0
    table = pd.read_csv(out / "subgroup_confusion.csv").set_index(
        ["input_path", "attribute", "subgroup", "family", "true_class", "predicted_class"]).sort_index()["count"]
    # 18-39 is A, B and C: on the full image C (cut wound) is called a contusion; per crop B has 2 crops, all right
    full = table.loc[("full", "age_group", "18-39", "vgg16")]
    assert (full.loc[("contusion", "contusion")], full.loc[("cut wound", "cut wound")],
            full.loc[("cut wound", "contusion")]) == (1, 1, 1)
    crop = table.loc[("crop", "age_group", "18-39", "resnet152")]
    assert (crop.loc[("cut wound", "cut wound")], crop.loc[("contusion", "contusion")], crop.sum()) == (3, 1, 4)
    # Unknowns are left out, and one figure per input path and attribute
    assert UNKNOWN not in table.index.get_level_values("subgroup")
    assert len(list(out.glob("confusion_*.png"))) == 2 * 5
