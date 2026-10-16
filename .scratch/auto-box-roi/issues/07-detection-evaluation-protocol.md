# Detection evaluation protocol

Type: grilling
Status: resolved
Blocked by: none

## Question

How exactly are auto boxes scored against manual boxes?
- the matching rule: greedy by confidence, one-to-one, IoU ≥ 0.5;
- the mAP definition: COCO 101-point interpolation, or the area under the precision–recall curve;
- whether metrics carry case-level bootstrap intervals, as the classification metrics do;
- the per-wound-type breakdown, given that a photo's boxes all share its wound type;
- how the threshold-selection F1 on validation is computed;
- how a photo with many boxes (up to 35) is weighted.

Note from "4F training recipe": the reference boxes are the unpadded CVAT boxes, not the padded crop coordinates in the manifests.

## Answer

(2026-10-16, grilled with the researcher)

1. **mAP**: COCO evaluation with `pycocotools` (already in the container): one category "wound", 101-point interpolation, maxDets 100.
   - Reported: **mAP@0.5 (primary)**, mAP@0.5:0.95, and AP for small / medium / large boxes.
   - Reference: unpadded CVAT boxes.
2. **Operating-point metrics**: greedy one-to-one matching by score at IoU ≥ 0.5, as COCO does.
   - Reported: precision (matched auto / all auto), recall (matched manual / all manual), F1, and the mean IoU of matched pairs.
   - Also the share of photos with no auto box, before the fallback.
3. **Weighting**:
   - **Primary**: boxes pooled (COCO).
   - **Secondary, photo level**:
     - the share of photos with at least one manual box found (what matters for union ROI);
     - the share with every manual box found;
     - the mean per-photo recall.
4. **By wound type**: COCO mAP and recall on the photos of each type. Precision for type X counts the auto boxes on photos of type X.
5. **Intervals**: case-level bootstrap as for classification (`src/evaluation/metrics.py`).
   - Settings: 2,000 resamples, seed 42, 95% percentile intervals, all photos of a case kept together.
   - Applied on test to mAP@0.5, recall, precision and the photo-level rates, overall and per type.
   - Small types are not merged; their wide intervals are reported as they are.
6. **Order of operations, every arm**:
   1. Raw boxes, then drop oversize boxes, then NMS at IoU 0.5, then the top 100 by score. **mAP is computed here.**
   2. Threshold: the score that **maximises pooled box F1** at IoU ≥ 0.5 on the arm's selection data (validation for 4F; train + validation for 4Z and 4P). Every distinct score is a candidate, and ties go to the higher threshold.
   3. Apply the threshold, then the box cap. A photo with no box keeps its top-scoring box; with none at all, it falls back to the full image. **Operating-point metrics are computed here.**
7. **Test**: detection metrics on test are computed **once**, after every arm has fixed its model, prompt and threshold, in the same pass that cuts the test crops. Arms that fail a gate never touch test.
