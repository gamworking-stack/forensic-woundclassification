# Detection evaluation protocol

Type: grilling
Status: open
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
