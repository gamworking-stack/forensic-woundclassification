#!/usr/bin/env python3
"""T35: pixel-based image quality of each test crop against whether a model classified it correctly.

Exploratory and unadjusted (no adjustment for wound class, no correction for multiple comparisons),
on the original images (T23 per-crop runs, no CLAHE). Quality measures per crop:

- ``megapixels``: width x height of the crop as cut from the photograph (box manifest ``x1``-``y2``,
  padding included). Read from the manifest, so it needs no image.
- ``area_fraction``: the crop's area as a share of its full photograph's area (manifest box and
  ``source_width`` x ``source_height``); no image needed either.
- With ``--image-root`` (the per-box crop folder; patient images, run on Raptor), on the crop resized
  so its longest side is ``--image-size`` (the scale the model sees, padding excluded):
  ``sharpness`` (variance of the Laplacian of CIELAB L*), ``brightness`` (mean L*, 0-100) and
  ``contrast`` (standard deviation of L*, RMS contrast).

Per model and measure: Spearman rho between the measure and the probability the model gave the true
class; the odds ratio of a correct prediction per 1 SD of the measure (log10 scale for megapixels and
sharpness); accuracy within quintiles of the measure. 95% intervals from a patient-level bootstrap
(a patient's crops are resampled together; the same resamples for every model and measure).

With ``--full-predictions-root`` each crop also gets the full-image model's verdict on its photograph
(same family), and the quintile table compares the two paired per crop (accuracy, crop minus full).

Writes ``quality_per_crop.csv``, ``quality_correlation.csv``, ``quality_quintiles.csv``, one scatter
figure per measure (``quality_scatter_<measure>.png``, a panel per model), ``quality_summary.png`` and,
with the full-image runs, ``crop_vs_full_<measure>.png``.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageFile, ImageOps  # noqa: E402
from scipy import ndimage, stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.paths import DATA_DIR  # noqa: E402
from src.evaluation.plots import FAMILY_LABELS, INK, MUTED  # noqa: E402
from src.evaluation.single_box import photo_stem  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True


FAMILIES = ["vgg16", "resnet152", "efficientnet_b0", "mobilenet_v3_small"]
MEASURES = {  # name: (axis label, log10 scale)
    "megapixels": ("Crop size (megapixels)", True),
    "area_fraction": ("Crop area / full-photograph area", True),
    "sharpness": ("Sharpness (variance of the Laplacian of L*)", True),
    "brightness": ("Brightness (mean L*, 0-100)", False),
    "contrast": ("Contrast (SD of L*)", False),
}
CORRECT, WRONG = "#2a78d6", "#eb6834"  # plots.CLASS_COLOURS[0:2]
QUINTILES = 5


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image quality of test crops against correct predictions.")
    parser.add_argument("--predictions-root", type=Path, required=True,
                        help="Folder with <family>/test_predictions.csv of the per-crop runs.")
    parser.add_argument("--families", nargs="+", default=FAMILIES)
    parser.add_argument("--box-manifest", type=Path, default=DATA_DIR / "roi_box/roi_manifest.csv")
    parser.add_argument("--patient-cases", type=Path, default=DATA_DIR / "splits/patient_cases.csv")
    parser.add_argument("--image-root", type=Path, default=None,
                        help="Per-box crop folder; adds sharpness, brightness and contrast.")
    parser.add_argument("--image-size", type=int, default=384, help="Longest side for the image measures.")
    parser.add_argument("--full-predictions-root", type=Path, default=None,
                        help="Folder with <family>/test_predictions.csv of the full-image runs: adds each crop's "
                             "photograph verdict and crop vs full-image accuracy by quintile.")
    parser.add_argument("--measures", nargs="+", default=None, choices=list(MEASURES),
                        help="Measures to analyse (default: every one available).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def lightness(rgb: np.ndarray) -> np.ndarray:
    """CIELAB L* (0-100, D65) of an sRGB uint8 array."""
    c = rgb.astype(np.float64) / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    y = linear @ np.array([0.2126, 0.7152, 0.0722])
    f = np.where(y > (6 / 29) ** 3, np.cbrt(y), y / (3 * (6 / 29) ** 2) + 4 / 29)
    return 116 * f - 16


def image_measures(path: Path, size: int) -> Dict[str, float]:
    """Sharpness, brightness and contrast of a crop at the model's scale (longest side = size)."""
    # Read as the training pipeline does (src.training.data.load_image): EXIF orientation, RGB
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = image.size
    scale = size / max(width, height)
    resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)
    lum = lightness(np.asarray(resized))
    return {"image_width": width, "image_height": height, "sharpness": float(ndimage.laplace(lum).var()),
            "brightness": float(lum.mean()), "contrast": float(lum.std())}


def crop_table(args: argparse.Namespace) -> pd.DataFrame:
    """One row per test crop: class, patient, quality measures, and per model correct / P(true class)."""
    boxes = pd.read_csv(args.box_manifest, dtype=str, keep_default_na=False).set_index("roi_relative_path")
    patients = pd.read_csv(args.patient_cases, dtype=str).set_index("relative_path")["patient_case"]
    table: Optional[pd.DataFrame] = None
    for family in args.families:
        preds = pd.read_csv(args.predictions_root / family / "test_predictions.csv")
        prob_true = [row[f"prob_{row['true_class'].replace(' ', '_')}"] for _, row in preds.iterrows()]
        own = pd.DataFrame({"crop": preds["relative_path"], f"{family}_correct": preds["true_class"] == preds["predicted_class"],
                            f"{family}_prob_true": prob_true})
        if table is None:
            table = preds[["relative_path", "true_class"]].rename(columns={"relative_path": "crop"})
        elif set(own["crop"]) != set(table["crop"]):
            raise ValueError(f"{family} predicts other test crops than {args.families[0]}")
        table = table.merge(own, on="crop", validate="one_to_one")

    missing = sorted(set(table["crop"]) - set(boxes.index))
    if missing:
        raise ValueError(f"{len(missing)} crops are not in {args.box_manifest}: {missing[:3]}")
    box = boxes.loc[table["crop"]]
    width = box["x2"].astype(int).to_numpy() - box["x1"].astype(int).to_numpy()
    height = box["y2"].astype(int).to_numpy() - box["y1"].astype(int).to_numpy()
    table.insert(2, "patient_case", patients.reindex(box["relative_path"]).to_numpy())
    if table["patient_case"].isna().any():
        raise ValueError("Some crops' photographs have no patient in " + str(args.patient_cases))
    table.insert(3, "crop_width", width)
    table.insert(4, "crop_height", height)
    table.insert(5, "megapixels", width * height / 1e6)
    photo_area = box["source_width"].astype(int).to_numpy() * box["source_height"].astype(int).to_numpy()
    table.insert(6, "area_fraction", width * height / photo_area)

    if args.full_predictions_root is not None:
        # The full-image model's verdict on each crop's photograph; full-image rows name the raw photo
        # file, so they are matched by group and file stem (as in patient_rescore.py)
        keys = list(zip(box["pseudonimized_id"], box["filename"].map(photo_stem)))
        for family in args.families:
            full = pd.read_csv(args.full_predictions_root / family / "test_predictions.csv", dtype={"pseudonimized_id": str})
            verdict = dict(zip(zip(full["pseudonimized_id"], full["relative_path"].map(photo_stem)),
                               full["true_class"] == full["predicted_class"]))
            unmatched = [k for k in keys if k not in verdict]
            if unmatched:
                raise ValueError(f"{len(unmatched)} crops' photographs have no {family} full-image prediction: {unmatched[:3]}")
            table[f"{family}_full_correct"] = [verdict[k] for k in keys]

    if args.image_root is not None:
        image = pd.DataFrame([image_measures(args.image_root / c, args.image_size) for c in table["crop"]])
        off = (np.abs(image["image_width"] - table["crop_width"]) > 1) | (np.abs(image["image_height"] - table["crop_height"]) > 1)
        # The crop may be stored rotated by its photo's EXIF orientation: compare sizes either way round
        off &= (np.abs(image["image_width"] - table["crop_height"]) > 1) | (np.abs(image["image_height"] - table["crop_width"]) > 1)
        if off.any():
            raise ValueError(f"{int(off.sum())} crop files differ in size from their manifest box")
        after = table.columns.get_loc("megapixels") + 1
        for k, name in enumerate(("sharpness", "brightness", "contrast")):
            table.insert(after + k, name, image[name].to_numpy())
    return table


def scaled(table: pd.DataFrame, measure: str) -> np.ndarray:
    x = table[measure].to_numpy(dtype=float)
    return np.log10(x) if MEASURES[measure][1] else x


def logistic_slope(x: np.ndarray, y: np.ndarray, iterations: int = 30) -> float:
    """Slope of a one-predictor logistic regression (Newton-Raphson); NaN if it does not converge."""
    if y.all() or not y.any():
        return float("nan")
    design = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-design @ beta))
        hessian = design.T @ (design * (p * (1 - p))[:, None])
        step = np.linalg.solve(hessian + 1e-9 * np.eye(2), design.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-8:
            return float(beta[1])
    return float("nan")


def interval(values: np.ndarray) -> List[float]:
    values = values[~np.isnan(values)]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if len(values) else [np.nan, np.nan]


def analyse(table: pd.DataFrame, families: List[str], measures: List[str], n_resamples: int, seed: int):
    """Correlation rows, quintile rows; the same patient resamples for every model and measure."""
    rng = np.random.default_rng(seed)
    cases = table["patient_case"].to_numpy()
    case_ids = np.array(sorted(set(cases)))
    rows_of = [np.flatnonzero(cases == c) for c in case_ids]
    resamples = [np.concatenate([rows_of[i] for i in rng.integers(0, len(case_ids), len(case_ids))])
                 for _ in range(n_resamples)]

    correlation, quintiles = [], []
    for measure in measures:
        x = scaled(table, measure)
        z = (x - x.mean()) / x.std()  # units of 1 SD of the whole test set, also in the resamples
        # Five equal-size groups by rank (ties broken by order), so no group is empty
        bins = pd.qcut(pd.Series(x).rank(method="first"), QUINTILES, labels=False).to_numpy()
        for family in families:
            correct = table[f"{family}_correct"].to_numpy(dtype=float)
            prob = table[f"{family}_prob_true"].to_numpy(dtype=float)
            full_column = f"{family}_full_correct"
            full = table[full_column].to_numpy(dtype=float) if full_column in table.columns else None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho = [stats.spearmanr(x[i], prob[i])[0] for i in resamples]
                slope = [logistic_slope(z[i], correct[i]) for i in resamples]
                acc = quintile_draws(correct, bins, resamples)
                if full is not None:
                    full_slope = [logistic_slope(z[i], full[i]) for i in resamples]
                    full_acc = quintile_draws(full, bins, resamples)
                    diff = quintile_draws(correct - full, bins, resamples)
            observed_slope = logistic_slope(z, correct)
            row = {
                "family": family, "measure": measure, "log10_scale": MEASURES[measure][1],
                "crops": len(table), "patients": len(case_ids),
                "spearman_rho": stats.spearmanr(x, prob)[0], **dict(zip(("rho_ci_low", "rho_ci_high"), interval(np.array(rho)))),
                "odds_ratio_per_sd": np.exp(observed_slope),
                **dict(zip(("or_ci_low", "or_ci_high"), np.exp(interval(np.array(slope))))),
                "accuracy": correct.mean(),
            }
            if full is not None:
                # The same odds ratio for the full-image model's verdict on each crop's photograph
                row.update({"full_odds_ratio_per_sd": np.exp(logistic_slope(z, full)),
                            **dict(zip(("full_or_ci_low", "full_or_ci_high"), np.exp(interval(np.array(full_slope))))),
                            "full_accuracy": full.mean()})
            correlation.append(row)
            for q in range(QUINTILES):
                inside = bins == q
                row = {
                    "family": family, "measure": measure, "quintile": q + 1,
                    "measure_low": table[measure].to_numpy()[inside].min(), "measure_high": table[measure].to_numpy()[inside].max(),
                    "measure_median": float(np.median(table[measure].to_numpy()[inside])),
                    "crops": int(inside.sum()), "patients": int(len(set(cases[inside]))), "accuracy": correct[inside].mean(),
                    **dict(zip(("ci_low", "ci_high"), interval(acc[:, q]))),
                }
                if full is not None:
                    # Crop model on the crop vs full-image model on the crop's photograph, paired per crop
                    row.update({"full_accuracy": full[inside].mean(),
                                **dict(zip(("full_ci_low", "full_ci_high"), interval(full_acc[:, q]))),
                                "crop_minus_full": (correct - full)[inside].mean(),
                                **dict(zip(("diff_ci_low", "diff_ci_high"), interval(diff[:, q])))})
                quintiles.append(row)
    return pd.DataFrame(correlation), pd.DataFrame(quintiles)


def quintile_draws(values: np.ndarray, bins: np.ndarray, resamples: List[np.ndarray]) -> np.ndarray:
    """Mean of ``values`` within each quintile, per resample (resamples x quintiles; NaN if empty)."""
    return np.array([[values[i][bins[i] == q].mean() if (bins[i] == q).any() else np.nan
                      for q in range(QUINTILES)] for i in resamples])


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_scatter(table: pd.DataFrame, correlation: pd.DataFrame, quintiles: pd.DataFrame,
                 families: List[str], measure: str, path: Path) -> None:
    label, log = MEASURES[measure]
    cols = 2 if len(families) > 1 else 1
    rows = int(np.ceil(len(families) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.0 * rows), squeeze=False, sharex=True, sharey=True)
    x = table[measure].to_numpy(dtype=float)
    for ax, family in zip(axes.flat, families):
        correct = table[f"{family}_correct"].to_numpy()
        prob = table[f"{family}_prob_true"].to_numpy(dtype=float)
        ax.scatter(x[correct], prob[correct], s=22, marker="o", color=CORRECT, alpha=0.75,
                   edgecolors="white", linewidths=0.6, label=f"correct ({correct.sum()})", zorder=2)
        ax.scatter(x[~correct], prob[~correct], s=30, marker="X", color=WRONG, alpha=0.9,
                   edgecolors="white", linewidths=0.6, label=f"wrong ({(~correct).sum()})", zorder=3)
        q = quintiles[(quintiles["family"] == family) & (quintiles["measure"] == measure)]
        ax.vlines(q["measure_median"], q["ci_low"], q["ci_high"], color=INK, linewidth=1.2, zorder=4)
        ax.plot(q["measure_median"], q["accuracy"], color=INK, linewidth=2, marker="s", markersize=6,
                markeredgecolor="white", label="accuracy by quintile (95% CI)", zorder=5)
        r = correlation[(correlation["family"] == family) & (correlation["measure"] == measure)].iloc[0]
        ax.set_title(FAMILY_LABELS.get(family, family), fontsize=10, color=INK, loc="left", fontweight="bold", pad=16)
        ax.text(0.0, 1.015, f"Spearman ρ {r.spearman_rho:+.2f} [{r.rho_ci_low:+.2f}, {r.rho_ci_high:+.2f}]   "
                          f"OR per SD {r.odds_ratio_per_sd:.2f} [{r.or_ci_low:.2f}, {r.or_ci_high:.2f}]",
                transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom")
        if log:
            ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.05)
        style(ax)
    for ax in axes.flat[len(families):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(label, fontsize=9, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("Probability of the true class / accuracy", fontsize=9, color=INK)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend([handles[2]] + handles[:2], [labels[2], "correct crop", "wrong crop"], loc="lower center",
               ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"{label} and correct classification of the {len(table)} test crops", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_summary(correlation: pd.DataFrame, families: List[str], measures: List[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(measures), figsize=(3.2 * len(measures), 0.5 * len(families) + 1.4),
                             sharey=True, squeeze=False)
    for ax, measure in zip(axes[0], measures):
        rows = correlation[correlation["measure"] == measure].set_index("family").loc[families]
        y = np.arange(len(families))[::-1]
        ax.axvline(0, color=MUTED, linewidth=1, linestyle="--", zorder=1)
        ax.hlines(y, rows["rho_ci_low"], rows["rho_ci_high"], color=INK, linewidth=1.5, zorder=2)
        ax.plot(rows["spearman_rho"], y, "o", color=INK, markeredgecolor="white", markersize=7, zorder=3)
        ax.set_yticks(y, [FAMILY_LABELS.get(f, f) for f in families])
        ax.set_xlim(-0.6, 0.6)
        ax.set_title(MEASURES[measure][0].split(" (")[0], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Spearman ρ with P(true class)", fontsize=8, color=INK)
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_crop_vs_full(quintiles: pd.DataFrame, families: List[str], measure: str, n_crops: int, path: Path) -> None:
    """Accuracy by quintile of the measure: crop model on the crop vs full-image model on its photograph."""
    label, log = MEASURES[measure]
    cols = 2 if len(families) > 1 else 1
    rows = int(np.ceil(len(families) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.8 * rows), squeeze=False, sharex=True, sharey=True)
    for ax, family in zip(axes.flat, families):
        q = quintiles[(quintiles["family"] == family) & (quintiles["measure"] == measure)]
        x = q["measure_median"].to_numpy()
        # Small horizontal offset so the two series' intervals do not overlap
        shift = (lambda v, d: v * 10 ** d) if log else (lambda v, d: v + d * (x.max() - x.min()) * 0.5)
        for column, low, high, colour, marker, name, d in (
                ("accuracy", "ci_low", "ci_high", CORRECT, "o", "wound crop (crop model)", -0.03),
                ("full_accuracy", "full_ci_low", "full_ci_high", WRONG, "s", "full photograph (full-image model)", 0.03)):
            xs = shift(x, d)
            ax.vlines(xs, q[low], q[high], color=colour, linewidth=1.3, zorder=2)
            ax.plot(xs, q[column], color=colour, linewidth=2, marker=marker, markersize=6,
                    markeredgecolor="white", label=name, zorder=3)
        ax.set_title(FAMILY_LABELS.get(family, family), fontsize=10, color=INK, loc="left", fontweight="bold")
        if log:
            ax.set_xscale("log")
        ax.set_ylim(0.3, 1.02)
        style(ax)
    for ax in axes.flat[len(families):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(f"{label} (median of each fifth of the crops)", fontsize=9, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy (95% CI)", fontsize=9, color=INK)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(f"Accuracy on the {n_crops} test crops by {label.lower()}", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    table = crop_table(args)
    measures = [m for m in (args.measures or MEASURES) if m in table.columns]
    if not measures:
        raise ValueError(f"None of {args.measures} is available (image measures need --image-root)")
    correlation, quintiles = analyse(table, args.families, measures, args.bootstrap, args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "quality_per_crop.csv", index=False)
    correlation.to_csv(out / "quality_correlation.csv", index=False)
    quintiles.to_csv(out / "quality_quintiles.csv", index=False)
    for measure in measures:
        plot_scatter(table, correlation, quintiles, args.families, measure, out / f"quality_scatter_{measure}.png")
        if "full_accuracy" in quintiles.columns:
            plot_crop_vs_full(quintiles, args.families, measure, len(table), out / f"crop_vs_full_{measure}.png")
    plot_summary(correlation, args.families, measures, out / "quality_summary.png")
    print(correlation.round(3).to_string(index=False))
    print(f"Wrote {len(table)} crops, {len(measures)} measures, {len(args.families)} models to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
