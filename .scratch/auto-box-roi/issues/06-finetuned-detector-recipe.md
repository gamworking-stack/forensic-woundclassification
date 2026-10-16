# 4F training recipe

Type: grilling
Status: open
Blocked by: none

## Question

How is the 4F detector trained?
- torchvision Faster R-CNN variant and backbone weights (e.g. `fasterrcnn_resnet50_fpn_v2`, COCO weights);
- input size;
- augmentations;
- epochs and early stopping on validation mAP@0.5;
- learning rate;
- seed;
- whether the 6 photos with no manual box take part.
The model is class-agnostic; train and validation splits only.

Note from "4Z setup: model, prompts and threshold": the oversize-box rule (drop boxes larger than the largest training-split manual box) applies to this arm too.
