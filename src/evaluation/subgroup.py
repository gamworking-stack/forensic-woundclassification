#!/usr/bin/env python3
"""Test performance by age, race/ethnicity, skin tone, setting, body region and tattoo (T24).

Every test photo gets the patient's age group and race/ethnicity, the wound's Monk skin tone (MST)
group, the setting (living or autopsy), the body region and whether a tattoo shows, all from the
researcher's sheet. Each scenario's existing test predictions are then scored per
subgroup: full image and union ROI (S01-S16), per-box ROI at photo level (S17-S24) and, optionally,
per-crop ROI (S25-S32; one row per crop). Nothing is retrained or re-predicted.

Intervals come from one case-level bootstrap of the whole test set per input path: each resample
draws cases with replacement and every subgroup is rescored on its own rows. All families of a path
share the test set, so they share the resamples, and the pooled row (mean over the 8 families) gets
its interval from the per-resample family means, as in T15. Each subgroup is also compared with the
largest subgroup of its attribute (difference, interval, two-sided bootstrap p). Subgroups with fewer
than ``--min-photos`` test photos are counted but not scored. Post hoc and descriptive: decided after
every test result had been seen (plan: tickets/T24_subgroups.md).

The sheet's demographic columns are read here only. Outputs go to ``--output-dir``, which must be
outside the repository; per-photo demographics are never written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.paths import DATA_DIR  # noqa: E402
from src.data.prepare_relabel import read_sheet  # noqa: E402
from src.evaluation.evaluate import (BOX_PATH, BOX_SCENARIO_IDS, CROP_PATH, CROP_SCENARIO_IDS,  # noqa: E402
                                     FAMILY_ORDER, SCENARIO_IDS)
from src.evaluation.metrics import Predictions, load_predictions  # noqa: E402
from src.evaluation.plots import CLASS_COLOURS, INK, MUTED  # noqa: E402
from src.evaluation.single_box import photo_stem  # noqa: E402
from src.evaluation.summarize import POOLED, bootstrap_p  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBGROUP_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
UNKNOWN = "unknown"
PHOTO = "photo"  # added column: the split manifest's relative_path of each row's photo
PATH_LABELS = {"full": "full image", "roi": "union ROI", BOX_PATH: "per-box ROI (photo)", CROP_PATH: "per-crop ROI"}

# Age in years; 0, '-' and blanks mean not recorded (the 0s come with race 0 in the sheet)
AGE_GROUPS = ((1, 17, "<18"), (18, 39, "18-39"), (40, 59, "40-59"), (60, 200, "60+"))
RACES = ("Asian", "European")  # anything else ('-', '0', '?', 'ไม่พบ ID') is unknown
# Monk Skin Tone 1-10, merged where the test set is thin (it has MST 2-6 only)
SKIN_TONE_GROUPS = ((1, 3, "MST 1-3"), (4, 4, "MST 4"), (5, 10, "MST 5-10"))
# Per photo: the sheet's autopsy flag, body site merged into regions, and tattoo; anything else is unknown
SETTINGS = {"0": "living", "1": "autopsy"}
BODY_REGIONS = {**dict.fromkeys(("scalp", "face", "ear", "neck"), "head and neck"),
                **dict.fromkeys(("shoulder", "armpit", "arm", "elbow", "wrist", "hand", "finger"), "upper limb"),
                **dict.fromkeys(("chest", "abdomen", "back"), "trunk"),
                **dict.fromkeys(("leg", "knee", "foot"), "lower limb"), "multi": "multiple sites"}
TATTOOS = {"0": "no tattoo", "1": "tattoo"}
ATTRIBUTES = ("age_group", "race_ethnicity", "skin_tone", "setting", "body_region", "tattoo")
LEVEL_ORDER = ([g[2] for g in AGE_GROUPS] + list(RACES) + [g[2] for g in SKIN_TONE_GROUPS]
               + list(SETTINGS.values()) + list(dict.fromkeys(BODY_REGIONS.values())) + list(TATTOOS.values()))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test performance per subgroup of each sheet attribute.")
    parser.add_argument("--sheet", type=Path, required=True,
                        help="The researcher's sheet as CSV with age, race_ethnicity, skin_tone, autopsy, "
                             "body_site and tattoo (e.g. ../thesis_data_26_oct.csv).")
    parser.add_argument("--main-root", type=Path, required=True,
                        help="Main run (full/ and roi/ for all 8 families), already evaluated.")
    parser.add_argument("--box-root", type=Path, default=None,
                        help="Per-box run laid out as box/<family>/photo/, already summarised (T17); "
                             "omit to leave out S17-S24 (T38 has none).")
    parser.add_argument("--no-union", action="store_true",
                        help="Leave out the union ROI models of the main run (T38: not reported).")
    parser.add_argument("--crop-root", type=Path, default=None,
                        help="Per-crop run laid out as crop/<family>/; omit to leave out S25-S32.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Outside the repository.")
    parser.add_argument("--split-manifest", type=Path, default=DATA_DIR / "splits/case_level_split_manifest.csv")
    parser.add_argument("--roi-manifest", type=Path, default=DATA_DIR / "roi/roi_manifest.csv")
    parser.add_argument("--box-manifest", type=Path, default=DATA_DIR / "roi_box/roi_manifest.csv")
    parser.add_argument("--min-photos", type=int, default=10, help="Smallest subgroup that is scored.")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Case-level resamples.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def binned(value: str, bins: Tuple[Tuple[int, int, str], ...]) -> str:
    # The label of the bin holding an integer value; anything else is unknown
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return UNKNOWN
    return next((label for low, high, label in bins if low <= number <= high), UNKNOWN)


def subgroup_columns(sheet: pd.DataFrame) -> pd.DataFrame:
    """The subgroup labels of every sheet row."""
    missing = {"autopsy", "body_site", "tattoo"} - set(sheet.columns)
    if missing:
        raise ValueError(f"The sheet is missing columns: {sorted(missing)} (use the 26 Oct sheet or later)")
    return pd.DataFrame({
        "age_group": sheet["age"].map(lambda v: binned(v, AGE_GROUPS)),
        "race_ethnicity": sheet["race_ethnicity"].where(sheet["race_ethnicity"].isin(RACES), UNKNOWN),
        "skin_tone": sheet["skin_tone"].map(lambda v: binned(v, SKIN_TONE_GROUPS)),
        "setting": sheet["autopsy"].map(SETTINGS).fillna(UNKNOWN),
        "body_region": sheet["body_site"].str.lower().map(BODY_REGIONS).fillna(UNKNOWN),
        "tattoo": sheet["tattoo"].map(TATTOOS).fillna(UNKNOWN),
    }, index=sheet.index)


def test_photos(sheet_path: Path, split_manifest: Path) -> pd.DataFrame:
    """Test photos of the split manifest with their subgroup labels, one row per photo."""
    sheet, _ = read_sheet(sheet_path)
    sheet = sheet[sheet["Inclusion"].str.upper() == "TRUE"]
    # The manifest's group ID may join several sheet cases (T23); the sheet's own ID is kept beside it
    photos = pd.read_csv(split_manifest, dtype=str, keep_default_na=False)
    photos = photos[photos["split"] == "test"].copy()
    source = "source_pseudonimized_id" if "source_pseudonimized_id" in photos else "pseudonimized_id"
    labels = pd.concat([sheet[["pseudonimized_id", "filename"]], subgroup_columns(sheet)], axis=1)
    if labels.duplicated(["pseudonimized_id", "filename"]).any():
        raise ValueError(f"{sheet_path}: an included (case, filename) appears twice")
    merged = photos.merge(labels.rename(columns={"pseudonimized_id": source}), on=[source, "filename"],
                          how="left", validate="one_to_one")
    missing = merged[merged["age_group"].isna()]
    if len(missing):
        raise ValueError(f"{len(missing)} test photos are not in the sheet as included "
                         f"(e.g. {missing['relative_path'].head(3).tolist()})")
    return merged


def photo_keys(photos: pd.DataFrame, roi_manifest: Path, box_manifest: Path) -> Dict[str, Dict]:
    """Per input path, the key a prediction row is matched on and the photo it belongs to."""
    # Full images: the file names the raw photo, filed under its source label, so match case and stem
    full = {(r["pseudonimized_id"], photo_stem(r["filename"])): r["relative_path"] for r in photos.to_dict("records")}
    if len(full) != len(photos):
        raise ValueError("Two test photos share a case and file stem; full-image matching is ambiguous")
    union = pd.read_csv(roi_manifest, dtype=str, keep_default_na=False)
    box = pd.read_csv(box_manifest, dtype=str, keep_default_na=False)
    return {"full": full,
            "roi": dict(zip(union["roi_relative_path"], union["relative_path"])),
            BOX_PATH: dict(zip(photos["relative_path"], photos["relative_path"])),
            CROP_PATH: dict(zip(box["roi_relative_path"], box["relative_path"]))}


def prediction_file(root: Path, input_path: str, family: str) -> Path:
    # Per-box scenarios are scored at photo level in box/<family>/photo (T17)
    folder = root / input_path / family / ("photo" if input_path == BOX_PATH else "")
    path = folder / "test_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; evaluate every scenario first")
    return path


def labelled_predictions(path: Path, input_path: str, keys: Dict, photos: pd.DataFrame) -> Predictions:
    """A test prediction file with each row's photo and subgroup labels in its frame."""
    p = load_predictions(path, "test")
    frame = p.frame
    key = (pd.Series(list(zip(frame["pseudonimized_id"], frame["relative_path"].map(photo_stem))), index=frame.index)
           if input_path == "full" else frame["relative_path"])
    frame[PHOTO] = key.map(lambda k: keys.get(k))
    if frame[PHOTO].isna().any():
        unmatched = frame.loc[frame[PHOTO].isna(), "relative_path"].head(3).tolist()
        raise ValueError(f"{path}: {int(frame[PHOTO].isna().sum())} rows match no test photo (e.g. {unmatched})")
    # Every row must keep its photo's case and class
    expected = photos.set_index("relative_path").loc[frame[PHOTO]]
    if (expected["pseudonimized_id"].to_numpy() != frame["pseudonimized_id"].to_numpy()).any() or \
            (expected["canonical_wound_label"].to_numpy() != frame["true_class"].to_numpy()).any():
        raise ValueError(f"{path}: a matched row has another case or class than its photo")
    for attribute in ATTRIBUTES:
        frame[attribute] = expected[attribute].to_numpy()
    return p


def fast_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """Accuracy, balanced accuracy and macro F1 as ``summary_metrics`` defines them (classes present in y_true).

    From the confusion counts, so the subgroup bootstrap stays fast; NaN for an empty sample.
    """
    if len(y_true) == 0:
        return np.full(len(SUBGROUP_METRICS), np.nan)
    counts = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    support, predicted, hits = counts.sum(axis=1), counts.sum(axis=0), np.diag(counts)
    present = support > 0
    recall = hits[present] / support[present]
    # F1 = 2·TP / (support + predicted); zero when the class is never hit
    f1 = 2 * hits[present] / (support[present] + predicted[present])
    return np.array([hits.sum() / len(y_true), recall.mean(), f1.mean()])


def subgroup_masks(frame: pd.DataFrame, levels: Dict[str, List[str]]) -> Dict[Tuple[str, str], np.ndarray]:
    return {(a, level): (frame[a] == level).to_numpy() for a in levels for level in levels[a]}


def scored_levels(photos: pd.DataFrame, min_photos: int) -> Dict[str, List[str]]:
    """Per attribute, the known subgroups with at least ``min_photos`` test photos, in LEVEL_ORDER."""
    out = {}
    for attribute in ATTRIBUTES:
        sizes = photos[attribute].value_counts()
        out[attribute] = [level for level in LEVEL_ORDER if sizes.get(level, 0) >= min_photos]
    return out


def path_draws(predictions: Dict[str, Predictions], levels: Dict[str, List[str]], n_resamples: int, seed: int
               ) -> Tuple[Dict, Dict]:
    """Observed and bootstrap scores per (family, attribute, level): arrays over SUBGROUP_METRICS."""
    families = list(predictions)
    first = predictions[families[0]]
    for f in families[1:]:
        if predictions[f].signature() != first.signature():
            raise ValueError(f"{f} was scored on another test set than {families[0]}")
    case_ids = np.array(sorted(set(first.cases)))
    rows_of = {f: [np.flatnonzero(predictions[f].cases == c) for c in case_ids] for f in families}
    masks = {f: subgroup_masks(predictions[f].frame, levels) for f in families}

    def scores(f: str, idx: np.ndarray) -> Dict[Tuple[str, str], np.ndarray]:
        p = predictions[f]
        return {g: fast_metrics(p.y_true[idx][m[idx]], p.y_pred[idx][m[idx]], len(p.classes))
                for g, m in masks[f].items()}

    observed = {f: scores(f, np.arange(len(predictions[f].y_true))) for f in families}
    # One case draw per resample, shared by every family of the path
    rng = np.random.default_rng(seed)
    draws = {(f, g): np.empty((n_resamples, len(SUBGROUP_METRICS))) for f in families for g in masks[f]}
    for b in range(n_resamples):
        picked = rng.integers(0, len(case_ids), size=len(case_ids))
        for f in families:
            for g, values in scores(f, np.concatenate([rows_of[f][i] for i in picked])).items():
                draws[(f, g)][b] = values
    return observed, draws


def interval(values: np.ndarray) -> Tuple[float, float]:
    values = values[~np.isnan(values)]
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def path_tables(input_path: str, predictions: Dict[str, Predictions], levels: Dict[str, List[str]],
                n_resamples: int, seed: int) -> Tuple[List[Dict], List[Dict]]:
    """Metric rows and gap rows (subgroup minus the attribute's largest subgroup) for one input path."""
    observed, draws = path_draws(predictions, levels, n_resamples, seed)
    families = list(predictions)
    # The pooled row is the mean over families, observed and per resample
    for g in next(iter(observed.values())):
        observed.setdefault(POOLED, {})[g] = np.mean([observed[f][g] for f in families], axis=0)
        draws[(POOLED, g)] = np.mean([draws[(f, g)] for f in families], axis=0)
    ids = {**SCENARIO_IDS, **BOX_SCENARIO_IDS, **CROP_SCENARIO_IDS}
    metric_rows, gap_rows = [], []
    for f in families + [POOLED]:
        frame = predictions[families[0]].frame
        common = {"scenario_id": ids.get((input_path, f), ""), "input_path": input_path, "family": f}
        for attribute, names in levels.items():
            # Each subgroup is compared with the attribute's largest one (most photos)
            reference = max(names, key=lambda level: frame.loc[frame[attribute] == level, PHOTO].nunique())
            for level in names:
                g = (attribute, level)
                rows = frame[frame[attribute] == level]
                row = {**common, "attribute": attribute, "level": level, "test_rows": len(rows),
                       "test_photos": rows[PHOTO].nunique(), "test_cases": rows["pseudonimized_id"].nunique()}
                for j, m in enumerate(SUBGROUP_METRICS):
                    low, high = interval(draws[(f, g)][:, j])
                    row.update({m: observed[f][g][j], f"{m}_ci_low": low, f"{m}_ci_high": high})
                metric_rows.append(row)
                if level == reference:
                    continue
                for j, m in enumerate(SUBGROUP_METRICS):
                    d = draws[(f, g)][:, j] - draws[(f, (attribute, reference))][:, j]
                    d = d[~np.isnan(d)]
                    low, high = interval(d)
                    gap_rows.append({**common, "attribute": attribute, "level": level, "reference": reference,
                                     "metric": m, "level_value": observed[f][g][j],
                                     "reference_value": observed[f][(attribute, reference)][j],
                                     "difference": observed[f][g][j] - observed[f][(attribute, reference)][j],
                                     "ci_low": low, "ci_high": high, "p_value": bootstrap_p(d), "resamples": len(d)})
    return metric_rows, gap_rows


def counts_table(photos: pd.DataFrame) -> pd.DataFrame:
    """Test photos and cases per subgroup, with the class mix behind every subgroup score."""
    rows = []
    for attribute in ATTRIBUTES:
        for level, group in photos.groupby(attribute):
            rows.append({"attribute": attribute, "level": level, "photos": len(group),
                         "cases": group["pseudonimized_id"].nunique(),
                         **group["canonical_wound_label"].value_counts().to_dict()})
    return pd.DataFrame(rows).fillna(0)


def class_accuracy(input_path: str, predictions: Dict[str, Predictions]) -> pd.DataFrame:
    """Accuracy within each wound class (its recall) per subgroup, per family and pooled (mean over families).

    Every known or unknown subgroup is listed, however small: the row and photo counts say how much each value rests on.
    """
    rows = []
    for f, p in predictions.items():
        frame = p.frame
        correct = frame["true_class"] == frame["predicted_class"]
        for attribute in ATTRIBUTES:
            for (level, true_class), idx in frame.groupby([attribute, "true_class"]).groups.items():
                rows.append({"input_path": input_path, "family": f, "attribute": attribute, "level": level,
                             "class": true_class, "test_rows": len(idx), "test_photos": frame.loc[idx, PHOTO].nunique(),
                             "test_cases": frame.loc[idx, "pseudonimized_id"].nunique(),
                             "correct": int(correct[idx].sum()), "accuracy": float(correct[idx].mean())})
    table = pd.DataFrame(rows)
    # Families share the test set, so the counts agree and only the accuracy is averaged
    keys = ["input_path", "attribute", "level", "class", "test_rows", "test_photos", "test_cases"]
    pooled = table.groupby(keys, as_index=False)["accuracy"].mean().assign(family=POOLED, correct=np.nan)
    return pd.concat([table, pooled[table.columns]], ignore_index=True)


def plot_subgroups(metrics: pd.DataFrame, metric: str, path: Path) -> None:
    """Pooled (mean over families) score per subgroup with its 95% interval, one marker per input path."""
    pooled = metrics[metrics["family"] == POOLED]
    attributes = [a for a in ATTRIBUTES if a in set(pooled["attribute"])]
    paths = list(dict.fromkeys(pooled["input_path"]))
    markers = ("o", "s", "D", "^")
    heights = [pooled[pooled["attribute"] == a]["level"].nunique() for a in attributes]
    fig, axes = plt.subplots(len(attributes), 1, figsize=(7.2, 0.75 * sum(heights) + 1.2 * len(attributes)),
                             sharex=True, gridspec_kw={"height_ratios": heights}, squeeze=False)
    # Rows are subgroups in LEVEL_ORDER from the top; input paths stack in legend order within a row
    offset = np.linspace(0.27, -0.27, len(paths)) if len(paths) > 1 else np.zeros(1)
    for ax, attribute in zip(axes[:, 0], attributes):
        rows = pooled[pooled["attribute"] == attribute]
        levels = list(dict.fromkeys(rows["level"]))
        y = np.arange(len(levels))[::-1].astype(float)
        for k, input_path in enumerate(paths):
            data = rows[rows["input_path"] == input_path].set_index("level").loc[levels]
            colour = CLASS_COLOURS[k]
            ax.hlines(y + offset[k], data[f"{metric}_ci_low"], data[f"{metric}_ci_high"], color=colour, linewidth=1.5)
            ax.plot(data[metric], y + offset[k], linestyle="none", marker=markers[k], markersize=7, color=colour,
                    markeredgecolor="white", markeredgewidth=1, label=PATH_LABELS.get(input_path, input_path))
        sizes = rows.drop_duplicates("level").set_index("level").loc[levels, "test_photos"]
        ax.set_yticks(y, [f"{level}  (n={n})" for level, n in zip(levels, sizes)])
        ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
        ax.set_title(attribute.replace("_", " "), loc="left", fontsize=10, color=INK)
        ax.grid(axis="x", color="#e5e4df", linewidth=0.6)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[-1, 0].set_xlabel(f"{metric.replace('_', ' ')}, mean of 8 families (95% case-bootstrap interval)",
                           color=MUTED)
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.15), ncol=len(paths), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out = args.output_dir.resolve()
    if out == REPO_ROOT or REPO_ROOT in out.parents:
        raise ValueError(f"--output-dir {out} is inside the repository; subgroup results stay outside it")
    out.mkdir(parents=True, exist_ok=True)

    photos = test_photos(args.sheet, args.split_manifest)
    keys = photo_keys(photos, args.roi_manifest, args.box_manifest)
    roots = {"full": args.main_root, "roi": None if args.no_union else args.main_root, BOX_PATH: args.box_root,
             CROP_PATH: args.crop_root}
    by_path = {input_path: {f: labelled_predictions(prediction_file(root, input_path, f), input_path,
                                                    keys[input_path], photos) for f in FAMILY_ORDER}
               for input_path, root in roots.items() if root is not None}
    # Counts and scored subgroups over the photos the full-image models were scored on (matched to ROI)
    photos = photos[photos["relative_path"].isin(by_path["full"][FAMILY_ORDER[0]].frame[PHOTO])]
    levels = scored_levels(photos, args.min_photos)
    counts_table(photos).to_csv(out / "subgroup_counts.csv", index=False)

    metric_rows, gap_rows = [], []
    for input_path, predictions in by_path.items():
        m, g = path_tables(input_path, predictions, levels, args.bootstrap, args.seed)
        metric_rows += m
        gap_rows += g
        print(f"{input_path}: scored {sum(map(len, levels.values()))} subgroups for {len(predictions)} families",
              flush=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "subgroup_metrics.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(out / "subgroup_gaps.csv", index=False)
    pd.concat([class_accuracy(p, predictions) for p, predictions in by_path.items()]).to_csv(
        out / "subgroup_class_accuracy.csv", index=False)
    for metric in ("accuracy", "macro_f1"):
        plot_subgroups(metrics, metric, out / f"subgroup_{metric}.png")
    print(f"{len(photos)} test photos from {photos['pseudonimized_id'].nunique()} cases; "
          f"wrote subgroup_*.csv and *.png to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
