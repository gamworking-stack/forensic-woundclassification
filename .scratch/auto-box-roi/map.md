# Map: Experiment 4, auto-box ROI

Label: wayfinder:map

## Destination

A frozen plan for **Experiment 4 (auto-box ROI)**, written as `tickets/T19_auto_box_roi.md` in the style of T17/T18. Every design choice is locked before any detector or classifier is run, so the next step is only writing code and running it on Raptor.

## Notes

- **Domain**: master's thesis on forensic wound classification. Terms are defined in `GLOSSARY.md`: manual box, auto box, wound detector, detector arm (4Z/4P/4F), union / per-box / per-crop ROI.
- **Skills**: grilling tickets call `grilling` and `domain-modeling`. Research tickets call `research`.
- **Chat language**: explain to the researcher in plain Thai; keep files in English.
- **Settled while charting (2026-10-16)**:
  - **Research question**:
    - Primary: can auto boxes replace manual boxes without a drop in classification, measured as auto-ROI − manual-ROI in the same family.
    - Secondary: auto-ROI − full image.
  - **Detector arms**: all three. 4Z is a zero-shot detector, no training. 4P is a public wound model. 4F is Faster R-CNN from torchvision, class-agnostic, fine-tuned on train-split manual boxes and selected on validation.
  - **Recall gate, every arm**: an arm goes forward only if recall at its operating threshold is ≥ 0.50 (IoU ≥ 0.5) on its own selection data. Otherwise it is dropped as a negative finding, and the retraining covers only the arms that pass. Never train a 4P model ourselves.
  - **Two uses of auto boxes**:
    - Primary: the existing Experiment 1–3 models predict on auto-box crops.
    - Secondary: retrain every scenario of Experiments 1–3 on auto-box crops, **for every detector arm that passes its gate** (≤ 72 scenarios plus learning-rate grids).
  - **Reference boxes**: manual boxes are the unpadded CVAT boxes, for training, scoring and the oversize rule in every arm. The 8% padding is applied only when cutting crops, to auto and manual boxes alike.
  - **From boxes to crops**: same as for manual boxes (union / per-box, 8% padding). The confidence threshold is chosen per detector on validation, by box-matching F1 at IoU ≥ 0.5. Overlapping boxes are merged by NMS (IoU 0.5). Boxes per photo are capped at the training-split maximum of manual boxes. A box covering more of the image than the largest training-split manual box (as an area share) is dropped, in every arm. 4Z, which trains nothing, chooses its threshold on train + validation.
  - **Photos with no box**: lower the threshold and keep the top box. If there is still none, fall back to the full image. Report these counts. The test set stays the same 157 photos.
  - **Detector evaluation, against manual boxes**:
    - mAP@0.5 and mAP@0.5:0.95;
    - precision and recall at IoU ≥ 0.5;
    - mean IoU of matched boxes;
    - share of photos with no auto box;
    - every metric per wound type, test set primary.
  - **Experiment 3 with auto boxes**: results only, no comparison, plus the share of auto crops that overlap no manual box.
  - **Test set**: the frozen split, used once more for this experiment. Nothing about a detector or threshold is chosen on test. Reported as a post hoc analysis.
  - **Supplementary**: verification of every number (T16-style) is required. Grad-CAM on the 14 T14 example photos; no cross-validation (see "Retraining rules and GPU budget").

## Decisions so far

<!-- one line per resolved ticket: [title](issues/NN-slug.md): gist -->

- [Which public wound detection models have usable weights](issues/02-public-wound-detectors.md): one forensic YOLOv8 model with public weights (simulated wounds, 3 of 7 types, AGPL); none trained on real forensic photos; MIT chronic-wound models exist (WILLIE gives boxes)
- [Which zero-shot detectors can find forensic wounds](issues/01-zero-shot-detectors.md): Grounding DINO, then OWLv2 (Apache-2.0, offline in transformers 4.57.6); expect low accuracy on small wounds, prompts matter most
- [Raptor environment and GPU budget for Experiment 4](issues/03-raptor-environment.md): venvs on /project with transformers 4.52.4 and ultralytics; Grounding DINO and OWLv2 load offline; ~24 GPU-h used so far, no SLURM limit
- [4Z setup: model, prompts and threshold](issues/04-zero-shot-setup.md): Grounding DINO vs OWLv2 × 4 frozen prompts (incl. proposal-definition prompts) × tiling on/off, winner by mAP@0.5 on train+val; oversize boxes dropped in all arms
- [4P: which public wound model, or drop the arm](issues/05-public-model-choice.md): Hanterdsith YOLOv8 (3 sizes × colour/grey) + WILLIE MINI if it loads, chosen on train+val; gate later amended to recall ≥ 0.50 at the operating threshold, else dropped as a negative finding; pickle scanned and re-saved
- [4F training recipe](issues/06-finetuned-detector-recipe.md): Faster R-CNN R50-FPN v2 from COCO, 1 class, 1024 px, flips + mild jitter, lr grid {0.005, 0.01, 0.02} chosen by validation mAP@0.5; reference boxes are unpadded CVAT boxes
- [Detection evaluation protocol](issues/07-detection-evaluation-protocol.md): pycocotools mAP@0.5 (primary) on unpadded boxes, F1-chosen threshold, pooled boxes plus photo-level rates, per wound type, case bootstrap 2,000; test scored once after all choices
- [Classification comparisons and multiplicity](issues/08-comparison-and-statistics.md): primary = non-inferiority of auto vs manual (pooled 8-family macro-F1, margin 0.03), union and per-box, Holm over passing arms; secondary two-sided: retrained auto − manual, auto − full; fallback photos kept, sensitivity excluding them
- [Retraining rules and GPU budget](issues/09-retraining-rules-and-budget.md): same rules as S01–S32 on auto crops in every split; S33–S56 4Z, S57–S80 4P, S81–S104 4F; one recall gate (≥ 0.50 at the operating threshold) for every arm, amending 4P; no CV, Grad-CAM on 14 examples

## Not yet specified

- **Thesis reporting**: which tables and figures go in the Results chapter, and how Experiment 4 is framed against the limitation "per-box crops still need manual boxes".
- **Detector failure analysis**: whether to look at where detectors miss or hallucinate wounds (per wound type, per photo condition) beyond the metrics already fixed.

## Out of scope

- **Running Experiment 4** (code, HPC runs, results): this map ends at the frozen plan.
- **A detector trained on `07_External_datasets/data_wound_seg` masks** (option d while charting): chronic wounds with masks only; not chosen.
- **Training our own "public" wound model for 4P**: it would only repeat 4F.
