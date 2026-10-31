"""T30: more contrast minus original, paired on the same test images."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.evaluation.contrast_compare import main
from src.evaluation.evaluate import FAMILY_ORDER

CLASSES = ["contusion", "cut wound", "scratch"]


def write_run(root: Path, seed: int, flip: int = 0) -> None:
    """Prediction files of every family for main/full and roi_crop/crop; ``flip`` rows predicted wrong."""
    rng = np.random.default_rng(seed)
    for run, path in (("main", "full"), ("roi_crop", "crop")):
        for f in FAMILY_ORDER:
            n = 30
            y = np.arange(n) % 3
            probs = np.full((n, 3), 0.1)
            probs[np.arange(n), y] = 0.8
            probs[:flip] = np.roll(probs[:flip], 1, axis=1)
            probs = probs + rng.uniform(0, 0.01, probs.shape)
            probs /= probs.sum(axis=1, keepdims=True)
            frame = pd.DataFrame({"relative_path": [f"img{i}.jpg" for i in range(n)],
                                  "pseudonimized_id": [str(i // 3) for i in range(n)], "split": "test",
                                  "true_class": [CLASSES[k] for k in y],
                                  "predicted_class": [CLASSES[k] for k in probs.argmax(axis=1)]})
            for k, name in enumerate(CLASSES):
                frame[f"prob_{name.replace(' ', '_')}"] = probs[:, k]
            folder = root / run / path / f
            folder.mkdir(parents=True)
            frame.to_csv(folder / "test_predictions.csv", index=False)


def test_difference_is_contrast_minus_original(tmp_path):
    write_run(tmp_path / "original", 0, flip=6)
    write_run(tmp_path / "contrast", 1, flip=0)
    assert main(["--original-root", str(tmp_path / "original"), "--contrast-root", str(tmp_path / "contrast"),
                 "--output-dir", str(tmp_path / "out"), "--bootstrap", "50"]) == 0
    table = pd.read_csv(tmp_path / "out" / "contrast_vs_original.csv")
    acc = table[(table["metric"] == "accuracy") & (table["family"] == "efficientnet_b0") & (table["input_path"] == "full")].iloc[0]
    assert (acc["original"], acc["more_contrast"], acc["difference"]) == pytest.approx((0.8, 1.0, 0.2))
    assert acc["ci_low"] <= acc["difference"] <= acc["ci_high"]
    assert (tmp_path / "out" / "contrast_vs_original_macro_f1.png").exists()
    # Per class: the 6 flipped rows are 2 of each class's 10, so recall 0.8 -> 1.0 and 2 images gained
    per_class = pd.read_csv(tmp_path / "out" / "contrast_per_class.csv")
    recall = per_class[(per_class["metric"] == "recall") & (per_class["family"] == "vgg16")
                       & (per_class["input_path"] == "crop")].set_index("true_class")
    assert recall["difference"].tolist() == pytest.approx([0.2, 0.2, 0.2])
    assert recall.loc["cut wound", "test_images"] == 10
    changes = pd.read_csv(tmp_path / "out" / "contrast_per_class_changes.csv")
    row = changes[(changes["family"] == "vgg16") & (changes["input_path"] == "full")
                  & (changes["true_class"] == "scratch")].iloc[0]
    assert (row["original_correct"], row["more_contrast_correct"], row["gained"], row["lost"]) == (8, 10, 2, 0)
    assert (tmp_path / "out" / "contrast_per_class_recall.png").exists()


def test_class_scores_match_the_class_report():
    from src.evaluation.contrast_compare import class_scores
    from src.evaluation.metrics import Predictions, class_report
    rng = np.random.default_rng(3)
    probs = rng.dirichlet(np.ones(3), size=40)
    y_true = rng.integers(0, 3, size=40)
    cases = np.repeat([f"c{i}" for i in range(10)], 4)
    p = Predictions(frame=pd.DataFrame({"pseudonimized_id": cases}), classes=CLASSES,
                    y_true=y_true, y_pred=probs.argmax(axis=1), probs=probs, cases=cases)
    report = class_report(p).iloc[:3]
    np.testing.assert_allclose(class_scores(p), report[["recall", "precision", "f1", "auc"]].to_numpy(dtype=float))


def test_different_test_sets_are_refused(tmp_path):
    write_run(tmp_path / "original", 0)
    write_run(tmp_path / "contrast", 0)
    other = tmp_path / "contrast" / "main" / "full" / "vgg16" / "test_predictions.csv"
    frame = pd.read_csv(other)
    frame.loc[0, "relative_path"] = "another.jpg"
    frame.to_csv(other, index=False)
    with pytest.raises(ValueError, match="test sets differ"):
        main(["--original-root", str(tmp_path / "original"), "--contrast-root", str(tmp_path / "contrast"),
              "--output-dir", str(tmp_path / "out"), "--bootstrap", "10"])
