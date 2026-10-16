# Classification comparisons and multiplicity

Type: grilling
Status: resolved
Blocked by: none

## Question

Which comparisons are primary and secondary, and over which set is the Holm correction applied? Candidates:
- **Primary**: auto-ROI (existing models) − manual-ROI, paired per family. Run per detector arm, for union and per-box designs: 8 families × 3 arms × 2 designs.
- **Secondary**:
  - retrained auto-ROI − manual-ROI;
  - auto-ROI − full image.

Also decide:
- whether the arms are compared with each other;
- whether results are pooled over families (as in the README's main table);
- the bootstrap settings (2,000 case resamples, seed 42, as before?).

## Answer

(2026-10-16, grilled with the researcher; fixed before any Experiment 4 result exists)

1. **Primary framing: non-inferiority.**
   - Auto boxes are declared **not inferior** to manual boxes when the lower bound of the 95% case-bootstrap interval of the **pooled** (8-family mean) auto − manual test macro-F1 difference is above **−Δ, with Δ = 0.03**.
   - Δ is about half the ROI gain seen in cross-validation (+0.060), so "not inferior" means auto boxes keep at least half of ROI's benefit.
   - Outcomes are reported as **not inferior / inferior** (upper bound below −Δ) / **inconclusive**.
   - Earlier pooled intervals were about 0.12 wide, so an inconclusive result is possible and will be stated as such.
2. **Primary comparisons**: existing models predicting on auto-box crops, against the same models on manual-box crops.
   - **union auto vs union manual** (the S09–S16 models);
   - **per-box auto vs per-box manual** (the S17–S24 models, photo level as the mean of crops);
   - for every detector arm that passes its gate, so up to 3 arms × 2 designs = 6 tests.
   - Experiment 3 with auto boxes stays results only.
3. **Multiplicity**:
   - Holm over the **pooled primary non-inferiority tests** (up to 6; only the arms that pass count), using one-sided p-values against −Δ.
   - Per-family results (8 families within each arm × design) are secondary, Holm over 8 families.
4. **Secondary comparisons**, two-sided, Holm over 8 families each, pooled effect reported:
   - (i) **retrained** auto − manual;
   - (ii) auto ROI − full image (the S01–S08 full-image models).
   - **Arms against each other**: descriptive only, no tests.
5. **Metric and statistics** as in Experiments 1–2:
   - photo-level test macro-F1;
   - paired case bootstrap with 2,000 resamples, seed 42, with the pooled effect computed within each resample;
   - two-sided p = min(1, 2 × min(P(d ≤ 0), P(d ≥ 0))).
   - Accuracy, balanced accuracy and macro AUC are reported without tests.
6. **Photos where the detector found nothing**: in the main analysis they enter **as they are**, i.e. the ROI model receives the full image and the test set stays 157 photos.
   - Their count is reported per arm.
   - A **sensitivity analysis** excluding them is descriptive only.
