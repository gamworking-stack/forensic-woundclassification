# Ticket T27: Manuscript Figures 1 and 2

- **Ticket ID**: `T27`
- **Phase**: Manuscript (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T23](T23_relabel_rerun.md), [T25](T25_patient_cases.md), [T26](T26_drop_unannotated.md)
- **Status**: Done (laptop, 2026-10-28)

## 1. Goal

The researcher asked (2026-10-28) for two manuscript figures: Figure 1, the study pipeline with the model architectures, and Figure 2, the inclusion and exclusion flow.

## 2. Content

| Figure | Panels / boxes | Source |
|---|---|---|
| 1 `figure1_pipeline` | a input photo with manual CVAT boxes; b full image and per-crop ROI (+8% margin), the two inputs the manuscript reports (researcher, 2026-10-28; union ROI and Experiments 1–2 left out); c pad to square, 384 px (ViT-B/16 224 px), training augmentation; d 8 classifiers (6 fine-tuned, CNN-SVM, AlexNet ensemble), each trained per input; e one prediction per photograph (full image) or per crop (Experiment 3), then evaluation | T05–T07, T18, `create_roi_dataset.py` (padding 0.08) |
| 2 `figure2_flow` | sheet photos and patients → exclusions by reason → included → patient-level split (photos, patients, crops) → 5-fold CV / final test | counted at run time from the sheet, `excluded_photos.csv`, `patient_cases.csv`, `roi_box/roi_manifest.csv` |

Schematic only: no photograph, no identifier. Figure 2 refuses to draw if sheet photos ≠ excluded + included.

## 3. Files

| File | Change |
|---|---|
| `src/visualization/manuscript_figures.py` | New |
| `runbook.md` | Step 11 |

## 4. Output

`../results_figures/manuscript/figure{1_pipeline,2_flow}.{png,tiff,pdf}`: 600 dpi; Figure 1 184 × 93 mm (double column), Figure 2 144 × 118 mm. Arial, vector PDF with embedded fonts.

Figure 2 counts (2026-10-28): 1,113 photographs (424 patients); excluded 81 (65 Inclusion FALSE, 7 imprint, 9 without a box; 33 patients with none left); included 1,032 (391 patients, 1,808 crops); train 728 / 275 / 1,310, validation 155 / 61 / 222, test 149 / 55 / 276; CV 883 photographs, 336 patients.
