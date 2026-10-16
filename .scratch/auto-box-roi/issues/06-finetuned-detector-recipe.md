# 4F training recipe

Type: grilling
Status: resolved
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

## Answer

(2026-10-16, grilled with the researcher)

- **Data facts used**:
  - train split: 779 photos with 1,177 manual boxes;
  - photos mostly 4000 × 3000;
  - median box 2.8% of the image area, 10% of boxes under 0.3%;
  - smallest types by median area: gunshot 1.0%, penetration 1.2%, scratch 1.8%.
1. **Reference boxes**: the **unpadded CVAT boxes** are used for training 4F, for scoring every detector arm, and for the oversize rule. The 8% padding is a crop step: it is applied to auto boxes exactly as to manual boxes. The current manifests' `x1..y2` are padded crop coordinates, so the unpadded boxes are read from the CVAT exports.
2. **Model**: `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2`, starting from COCO weights (`FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`), with the box predictor replaced by 2 classes (background, wound). Class-agnostic.
3. **Input size**: `min_size=1024`, `max_size=1365`, fixed, not tuned. Images are cached downsized, as in classifier training.
4. **Augmentation**: horizontal flip, vertical flip, mild brightness/contrast jitter. No rotation, no random crop.
5. **Training**:
   - SGD, momentum 0.9, weight decay 1e-4, batch 4, linear warmup over 500 iterations;
   - **learning-rate grid {0.005, 0.01, 0.02}**;
   - at most 60 epochs, early stopping with patience 10;
   - fp32, seed 42.
   - Checkpoint and learning rate are both chosen by **class-agnostic mAP@0.5 on validation**.
6. **Photos without a manual box**: the 6 photos with no manual box (outside the 1,100) are left out. "No box" there means "not annotated", not "no wound".
