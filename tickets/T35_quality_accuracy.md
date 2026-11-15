# Ticket T35: Image Quality of the Test Crops against Correct Classification

- **Ticket ID**: `T35`
- **Phase**: Exploratory analysis on the T23 per-crop runs (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T18](T18_crop_roi.md) (per-crop runs), [T23](T23_relabel_rerun.md) (data), [T25](T25_patient_cases.md) (patients as cases)
- **Scenarios**: none new; the T23 test predictions of S25–S32 for VGG16, ResNet152, EfficientNet-B0 and MobileNetV3-Small
- **Status**: Done (2026-11-15; results §5)

---

## 1. Goal

The researcher asked (2026-11-15) for scatter plots exploring whether the image quality of a wound crop, measured from its pixels, goes with whether it is classified correctly. Four models, the original images (no CLAHE), no adjustment.

## 2. Decisions

| Item | Decision | Reason |
|---|---|---|
| Crops | The 276 T23 test crops (55 patients) | The models were never selected on them |
| Quality measures | Crop size (megapixels of the box as cut, padding included, from the manifest); sharpness (variance of the Laplacian of CIELAB L*); brightness (mean L*); contrast (SD of L*). The last three on the crop resized so its longest side is 384 px, the scale the model sees, without the square padding | Pixel-based, no reference image needed; size needs no image |
| Outcome on the plot | Each crop's probability for its true class (y), coloured and shaped by correct or wrong; accuracy within quintiles of the measure drawn over it, with 95% intervals | A correct/wrong outcome alone is 0 or 1 and does not scatter |
| Statistics | Spearman ρ (measure vs probability of the true class) and odds ratio of a correct prediction per 1 SD of the measure (log10 for size and sharpness), each with a 95% patient-level bootstrap interval (2,000 resamples, the same for every model and measure) | Crops of a patient are not independent |
| Adjustment | None: no adjustment for wound class, no correction for multiple comparisons | Researcher's choice; exploratory. Wound class is a strong confounder (gunshot crops are the smallest) |
| Where | Crop size on the laptop (manifest + the T23 test prediction files in `../discussion 29 oct/test_results/per_crop_exp3/`); image measures on Raptor (patient images) | Patient images stay on Raptor |

## 3. Files

| File | Change |
|---|---|
| `src/evaluation/quality_accuracy.py` | New |
| `tests/test_quality_accuracy.py` | L* and sharpness, blur lowers sharpness, a planted size effect found, image measures and quintiles |
| `scripts/slurm/quality_accuracy.sbatch` | New CPU job |

## 4. Run log

- 2026-11-15 (laptop): `pytest tests/test_quality_accuracy.py` 4 passed (Python 3.11, CPU, no torch). Crop size run on the laptop → `../results_figures/Image quality vs accuracy/`.
- 2026-11-15 (Raptor, commit `00eed02`): job 247170 (CPU, about 1 min): tests 4 passed; all four measures on the 276 crops; every crop file matched its manifest box size. Outputs in `$EXPERIMENTS_DIR/quality_accuracy/`, copied to `../results_figures/Image quality vs accuracy/` (no images; the per-crop table names crop files only, as the prediction files do).

## 5. Results (276 test crops, 55 patients; unadjusted)

| Measure | Spearman ρ with P(true class), range over the 4 models | Odds ratio of a correct prediction per 1 SD [95% CI] |
|---|---|---|
| Crop size (log) | +0.12 to +0.18 (VGG16 [+0.03, +0.31]; others include 0) | 0.89–1.04, all include 1 |
| Sharpness (log) | −0.00 to +0.10, all include 0 | 0.90–1.24, all include 1 |
| Brightness | −0.05 to +0.14, all include 0 | 0.60–0.78; EfficientNet-B0 0.60 [0.44, 0.88] |
| Contrast | −0.16 to −0.02, all include 0 | 1.10–1.67; ResNet152 1.46 [1.11, 1.97], EfficientNet-B0 1.67 [1.21, 2.26] |

- **No measure goes with correctness in all four models.** 4 of 32 intervals exclude the null, against about 1.6 expected by chance.
- **Crop size is U-shaped, not monotone**: the smallest fifth of crops (≤ 0.07 MP) is classified best (accuracy 0.89–0.95), the second and third fifths worst (0.66–0.82). Half the smallest fifth are gunshot wounds (28 of 56), the class the models get most right.
- **Contrast**: the lowest-contrast fifth has the lowest accuracy (0.66–0.75), hence OR > 1; yet ρ is slightly negative, because that fifth also holds contusions and scratches (median SD of L* 5.7 and 5.0, against 10.6–15.0 for the other classes) that are classified correctly with high probability.
- **Reading**: the signals line up with wound class (gunshot crops small, dark and high-contrast; contusion and scratch crops low-contrast), which this unadjusted analysis does not separate from image quality. Exploratory; 16 model × measure pairs, no multiplicity correction.
