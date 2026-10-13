# Ticket T14: Grad-CAM

- **Ticket ID**: `T14`
- **Phase**: Phase 6 (Figures and Results)
- **Spec Reference**: [ROADMAP.md](../ROADMAP.md) §3 (Grad-CAM "where applicable"), §5 T14
- **Scenarios**: S01–S12, S15–S16 (CNN-SVM excluded)
- **Status**: In Progress

---

## 1. Goal

Show where each fine-tuned model looks, on the same test images for every family and for both input paths, so full and ROI can be compared qualitatively.

---

## 2. Decisions (2026-10-14)

| Item | Decision | Reason |
|---|---|---|
| Families | The 6 single families and the AlexNet ensemble's representative member (seed 43, `ensemble_members.csv`) | ROADMAP: "where applicable". CNN-SVM's SVM on frozen concatenated features has no gradient path to one image map; excluded and stated in the thesis |
| Weights | The checkpoint that produced each scenario's test predictions: `best.pt` of the T08b selected run (S01–S12) or of the representative member (S15–S16) | Same model as the reported results |
| Images | **One fixed set**: 2 test images per class (14), from 14 different cases, drawn once from the matched test rows (images with an ROI crop), seed 42, before any map is computed; the same images for every family and both paths | A pick by "correct/wrong" would differ per model and path, so full and ROI would not show the same images. Correct/wrong is reported per panel instead |
| Method | Grad-CAM on the last convolutional block (ViT-B/16: the last encoder block's tokens, class token dropped, reshaped to 14 × 14) for the **predicted** class, fp32, ReLU, normalized to [0, 1], upsampled to the model input | Standard Grad-CAM; ViT needs the token reshape (ROADMAP T14) |
| Input shown | The image as the model sees it: padded to square and resized (no normalization) | Map and image align pixel for pixel |
| Patient images | Figures are overlays on patient photographs: they stay on Raptor (`<exp>/gradcam/`), and `fetch_results.sh` does not copy them. Use in the thesis only as the ethics approval allows | Patient data |

Target layers: EfficientNet-B0 and MobileNetV3-Small `features[-1]`, ResNet152 `layer4[-1]`, GoogLeNet `inception5b`, VGG16 `features[29]` (last ReLU), AlexNet `features[11]` (last ReLU), ViT-B/16 `encoder.layers[-1].ln_1`.

---

## 3. Requirements & Business Logic

1. **Examples** (`<exp>/gradcam/examples.csv`): class, case, full-image path, ROI-crop path, set (1 or 2).
2. **Per family, path and example** (`<exp>/gradcam/gradcam_results.csv`): true class, predicted class and its probability, correct, and whether the prediction equals the scenario's prediction file (`matches_prediction_file`; fp32 here vs bf16 in training can flip a borderline case).
   - **Full path only**: `cam_in_box`, the share of the map's mass inside the annotated wound box (the 8%-padded union box from the ROI manifest, mapped into the padded and resized input), and `box_share`, the box's share of the input area (what a uniform map would give). This is a numeric answer to "does the full-image model look at the wound?" that needs no one to view the photographs.
3. **Figures** (`<exp>/gradcam/<family>_set<k>.png`): one row per class; columns full input, full Grad-CAM, ROI input, ROI Grad-CAM; each map titled with predicted class, probability and ✔/✘.
4. **Automated Testing** (`tests/test_gradcam.py`, CPU, random weights):
   - Every family's map has the input's height and width, values in [0, 1] and no NaN.
   - The example draw is deterministic, 2 per class from distinct cases, test rows only.
   - Integration on synthetic runs: results CSV, examples CSV and figures are written; predictions match the prediction files.

---

## 4. Files to Create / Modify

| File | Purpose |
|---|---|
| `src/evaluation/gradcam.py` | CLI: draw examples, compute maps, write results and figures |
| `tests/test_gradcam.py` | Tests above |
| `runbook.md` | Command and outputs |

---

## 5. Acceptance Check

1. `pytest tests/` passes.
2. On Raptor, all 7 families x 2 paths x 14 images are computed; `matches_prediction_file` is true for (nearly) every row, with any exception listed here.
3. `cam_in_box` vs `box_share` summarized per family here; figures inspected on Raptor by the researcher.
4. Tick T14 in `ROADMAP.md` §9.
