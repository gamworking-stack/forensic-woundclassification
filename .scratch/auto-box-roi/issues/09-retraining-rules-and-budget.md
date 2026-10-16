# Retraining rules and GPU budget

Type: grilling
Status: resolved
Blocked by: 03

## Question

For retraining Experiments 1–3 on auto-box crops for all three arms (≈72 scenarios):
- Are the rules identical to S01–S32: lr grid {1e-5, 3e-5, 1e-4}, CNN-SVM grid, AlexNet ensemble seeds, max 200 epochs, patience 25, batch 32, seed 42?
- How are scenarios numbered (from S33)?
- In what order do the arms run?
- What happens if the GPU budget cannot cover all 72: drop the per-crop design, run fewer families, or wait?

## Answer

(2026-10-16, grilled with the researcher)

1. **Rules**: identical to S01–S32:
   - lr grid {1e-5, 3e-5, 1e-4} for the six single networks;
   - CNN-SVM with the 29-point grid;
   - AlexNet ensemble seeds 43–45 at lr 3e-5;
   - at most 200 epochs, patience 25, batch 32, seed 42.
   - Selection unit as in the original experiments: union at photo level, per-box at photo level (mean of crops, T17), per-crop at crop level (T18).
2. **Crops**: auto-box crops in **every split**. Train and validation photos go through the same threshold, box cap and fallback rules as test. For 4F's training photos, see "4F auto boxes on its own training photos".
3. **Numbering**: reserved by arm.
   - **S33–S56 = 4Z**, **S57–S80 = 4P**, **S81–S104 = 4F**.
   - Within an arm: union (8 families), then per-box (8), then per-crop (8), each in S01–S16 family order.
   - A dropped arm keeps its numbers unused.
4. **Recall gate, for every arm**: an arm goes forward only if, on its own selection data, **recall at the operating threshold** (F1-chosen, IoU ≥ 0.5, pooled boxes) is **≥ 0.50**. The selection data is validation for 4F, and train + validation for 4Z and 4P.
   - An arm that fails is dropped from crops, retraining and test, and reported as a negative finding.
   - This **amends the 4P gate** (previously max recall over up to 100 boxes per image, which a detector that scatters many boxes could pass).
5. **Run order**:
   1. Primary analysis first: existing models on auto-box crops, every passing arm.
   2. Then retraining in the order 4F → 4Z → 4P.
6. **If GPU or time runs short**, cuts are made in this order, fixed in advance:
   1. per-crop retraining;
   2. retraining of the passing arm with the lowest selection-data mAP@0.5.
   - The primary analysis is never cut.
   - Budget estimate: Experiments 1–3 plus CV used about 24 GPU-hours, so ≤ 72 retrained scenarios are expected to fit easily.
7. **Supplementary**:
   - **No cross-validation**: per-fold detectors would be needed, and the primary question is pre-specified on test.
   - **Grad-CAM**: the T14 method on the 14 T14 example photos, existing models, auto-box crops next to manual-box crops, for every passing arm and both primary designs (`gradcam.py`).
   - Verification of every number (T16-style) stays required.
