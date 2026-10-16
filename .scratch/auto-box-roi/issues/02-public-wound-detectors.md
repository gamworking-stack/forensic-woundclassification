# Which public wound detection models have usable weights

Type: research
Status: resolved
Blocked by: none
Findings: branch `research/public-wound-detectors`, file `docs/research/public-wound-detectors.md`

## Question

Is there a published wound **detection** model (boxes, or masks convertible to boxes) with downloadable weights, to serve as detector arm **4P**? Forensic or traumatic wounds are preferred; chronic wounds (diabetic foot, pressure, venous) are acceptable if nothing forensic exists. For each, find:
- training data;
- architecture;
- weight format and source;
- licence;
- reported metrics;
- whether it runs with torch 2.6 without network access.
If none exists, say so with what was searched.

## Answer

(2026-10-16; full findings with sources on branch `research/public-wound-detectors`, `docs/research/public-wound-detectors.md`)

- **One forensic detector has downloadable weights.** It is the Hanterdsith 2026 YOLOv8 model (n, m and x variants; Hugging Face `trapezius60/forensic_wound_detection`, revision pinned in the findings).
  - **Training data**: 595 images of **simulated** wounds, grayscale-filtered. It was tested on only 24 images (mAP@50 0.99).
  - **Classes**: 8, of which only gunshot entrance/exit, laceration and hesitation cuts overlap ours. There is no contusion, graze, scratch or stab class.
  - **Libraries**: needs `ultralytics`, which is AGPL-3.0. The model's licence is inconsistent across its pages (none / apache-2.0 / CC BY-NC 4.0); all allow academic use.
- **No model trained on real forensic photos has public weights.** Zimmermann et al. 2023 (Zurich, 1,753 real photos, segmentation) share theirs only "on reasonable request".
- **Chronic-wound models with downloadable weights (MIT licence)**:
  - WILLIE: boxes from masks, AP@0.5 96.2% on FUSeg/AZH/Medetec. Part of its model code is missing from GitHub.
  - FUSegNet: masks.
  - HarDNet-DFUS: masks.
  - Deepskin: TensorFlow only.
- **Not usable**: paper-only models (the AZH YOLOv3 localizer, DFUC2020, Tsai 2025), Roboflow-hosted models (no documented download; offline use needs an API key), and undocumented hobby checkpoints.
- **The agent's recommendation**: run the Hanterdsith model class-agnostically as a gated pilot, measuring box recall at IoU ≥ 0.5 on the **training split**. Keep 4P only if recall is adequate; otherwise drop it as a negative finding. Meanwhile, ask the Zurich group for their model. The decision itself belongs to "4P: which public wound model, or drop the arm".
