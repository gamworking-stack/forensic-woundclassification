# 4F auto boxes on its own training photos

Type: grilling
Status: resolved
Blocked by: 06

## Question

Retraining under arm 4F needs auto boxes on the **training** photos. But the 4F detector was fitted to those photos' manual boxes, so its boxes there will be near-perfect, while on validation and test they will not. Training crops would then be cleaner than test crops. How is this handled?
- (a) Cross-fit: k detectors, each boxes the training photos it did not see.
- (b) Accept it and state it as a limitation.
- (c) Use manual-box crops for training and auto boxes only for validation and test.
Also: does the same concern apply to choosing the threshold on validation?

## Answer

(2026-10-16, grilled with the researcher)

1. **Cross-fitting for 4F training photos**: 5 extra 4F detectors are trained. Each learns from 4 of 5 groups of training cases and boxes only the group it did not see. Every training photo therefore gets auto boxes from a detector that never saw it. Validation and test photos are boxed by the **main** 4F detector, which is trained on the whole training split.
2. **Folds and recipe**:
   - **Groups**: the existing `data/splits/cv_folds.csv`, restricted to training cases (fold sizes 64 / 57 / 60 / 55 / 59 cases). No new draw.
   - **Recipe**: the main 4F recipe, with the main detector's chosen **learning rate and best epoch count fixed** (no further selection, as in the T12b cross-validation).
   - **Threshold**: the main 4F operating threshold is applied to the cross-fitted boxes, so the same rules hold on every split.
   - **Reported**: recall of the cross-fitted boxes on the training split, next to the main detector's recall on validation, to show the box quality is comparable.
3. **Validation photos**: the main detector selects its checkpoint, learning rate and threshold on validation, so its validation boxes are mildly optimistic. This is accepted and stated as a limitation, like validation-based selection in Experiments 1–3.
4. **Scope**: only 4F retraining is affected. 4Z and 4P are not trained on our data, and the primary analysis (existing models on test crops) does not use training-photo boxes.
