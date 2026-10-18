# Map: Experiment 5, more training images for the small classes

Label: wayfinder:map

## Destination

A frozen plan for **Experiment 5 (small-class augmentation)**, written as `tickets/T20_small_class_augmentation.md` in the style of T19. Every choice is locked before any augmented image is made or any model is trained, so what remains is code and runs on Raptor.

## Notes

- **Domain**: thesis on forensic wound classification. Terms are defined in `GLOSSARY.md`: small class, oversampled crop, synthetic crop, augmentation arm.
- **Skills**: grilling tickets call `grilling` and `domain-modeling`; research tickets call `research`.
- **Language**: chat with the researcher in plain English (Thai on request); files in English.
- **Settled while charting (2026-10-18)**:
  - **Arms**:
    - **5A**: oversampled crops with strong class-specific augmentation, no new images.
    - **5B**: synthetic crops from a diffusion model (LoRA) fine-tuned on training-split crops.
  - **Design**: per-box ROI (Experiment 2). Baselines are S17–S24 for test and the existing per-box CV (T17 §7) for CV.
  - **Small classes**: cut wound, scratch, penetration wound. In training crops that is 81 / 102 / 69 (cases 30 / 34 / 23). The other 4 classes stay real-only.
  - **Amount**: each small class is brought up to the median class's training crops (**156**), the same target in both arms. That adds about +75 cut, +54 scratch and +87 penetration in the main training split; per CV fold the target is recomputed the same way. Synthetic crops never exceed half of a class.
  - **Class weights**: recomputed from the enlarged counts (T05 rule).
  - **Validation and test**: real photos only, unchanged.
  - **Primary analysis**: 5-fold CV on the 356 development cases (frozen `cv_folds.csv`). The outcome is the **mean F1 of the 3 small classes**, pooled over 8 families and paired against the existing per-box CV out-of-fold predictions.
  - **Secondary analyses**: test, paired with S17–S24; macro-F1; per-class recall.
  - **Statistics**: superiority, two-sided, with the paired case bootstrap (2,000 resamples, seed 42) and Holm over the arms (5A, 5B).
  - **Families**: all 8.
  - **CV leakage**: 5B trains **one diffusion model per CV fold**, on that fold's training cases only, plus one on the training split for the test analysis. 5A augments inside each fold.
  - **Ethics gate for 5B**: 5B runs only if the ethics approval covers training a generative model on patient photographs. Otherwise 5B waits for an amendment or goes out of scope, and 5A proceeds alone.
  - **Synthetic image checks**: an automatic similarity filter against the training crops (to catch memorization), plus a **blinded review by the researcher and the advisor** of a shuffled sample of real and synthetic crops. The researcher must not view synthetic crops before that review.

## Decisions so far

<!-- one line per resolved ticket: [title](issues/NN-slug.md): gist -->

- [Which diffusion model and LoRA tooling can make wound crops on Raptor](issues/01-diffusion-models-on-raptor.md): SD 1.5 first (open, 512 px, memorizes less), then SDXL, FLUX.1 [dev]/SD 3.5; pinned diffusers 0.35.2 stack installed `--no-deps`; literature gains largest with tens of real images per class; DINO/CLIP nearest-neighbour memorization filter

## Not yet specified

- **Thesis reporting**: how Experiment 5 is framed next to Experiments 2–4, and which figures to use (e.g. example synthetic crops, if ethics allows showing them).
- **If 5B is ruled out by ethics**: whether a non-patient alternative (e.g. an external or public dataset for the generator) is worth a separate effort. That would likely be a new destination.

## Out of scope

- **Collecting new real cases**: the best fix, but slow; future work.
- **The internet wound dataset in `07_External_datasets/`**: it has no scratch, penetration or gunshot images, so it does not reach the smallest classes.
- **Copy-paste of wounds onto other skin**: of little use to a crop-based design.
- **The union ROI design**: per-box only, to keep the experiment small.
