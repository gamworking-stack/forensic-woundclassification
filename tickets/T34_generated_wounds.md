# Ticket T34: Generated Cut, Scratch and Penetration Wounds (GAN and Diffusion)

- **Ticket ID**: `T34`
- **Phase**: Extra experiment (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T18](T18_crop_roi.md) (per-crop path), [T12b](T12b_cross_validation.md) (CV folds), [T23](T23_relabel_rerun.md), [T30](T30_more_contrast.md) (paired comparison)
- **Scenarios**: per-crop models (S25–S32 recipe) retrained with generated crops added to training
- **Status**: In progress: stage 1 (generators training on Raptor, 2026-11-09)

---

## 1. Goal

The researcher asked (2026-11-09) for a GAN that generates cut, scratch and penetration wounds, the three smallest classes. Decisions taken with the researcher the same day: generate **wound crops** (not full photographs), use them to **augment training** and test whether recall of these classes improves, and **compare a GAN with a diffusion model**.

## 2. Data

Training split (T23): cut 94 crops (59 photos), scratch 77 (40), penetration 39 (33); all classes 1,310 crops. Test: cut 13 photos, scratch 12, penetration 4, so one penetration photo moves test recall by 25 points. Cut and scratch crops are long (median aspect 2.9 and 2.4; 90th percentile 5.4–5.6).

## 3. Decisions

| Item | Decision | Reason |
|---|---|---|
| Generator input | Training-split crops of **all 7 classes**, padded to a square and resized to 256 px exactly as the classifiers pad them (`export_generator_crops.py`) | 39–94 crops per class is too few for one generator per class; conditioning on the class lets skin, light and texture be learned from all 1,310 |
| GAN | Class-conditional StyleGAN2-ADA (NVlabs/stylegan3, `--cfg stylegan2`, commit c233a91), batch 32, gamma 1, x-flips, adaptive augmentation, 5,000 kimg, snapshot chosen by KID | ADA is made for small datasets |
| Diffusion | Stable Diffusion 1.5 (revision 451f4fe, CreativeML OpenRAIL-M) with a rank-16 LoRA on the UNet, trained on the same crops at 512 px with one caption per class ("a forensic close-up photograph of a {class} on human skin"); diffusers v0.31.0 example script, batch 8, 15,000 steps, lr 1e-4 cosine, min-SNR 5, bf16. Sampling: 30 steps, guidance 5, the NSFW filter off (it blanks wound photographs) | Usually copes better with tiny classes; its pretraining on internet images is reported. The licence bars medical advice, not research data |
| Leakage | Generators never see validation or test crops. For CV every fold gets its own generators, trained on that fold's training folds only | A generator trained on all development crops would leak a CV fold's validation crops into training |
| Memorisation | Each generated crop's nearest training crop (cosine similarity of the Inception features); flagged when closer than 95% of real validation crops are to their nearest training crop; flagged crops are dropped and the closest pairs shown to the researcher | 39 penetration crops are easy to memorise; a copy would also copy identifying marks such as tattoos |
| Realism | KID per class (StyleGAN's Inception features) against real training and validation crops, beside the KID of real validation against real training crops as the level of genuinely new wounds; a blinded real-or-generated rating by the researcher | FID is unstable with so few real images |
| Augmentation | Generated crops of the three classes added to training only, up to the size of the median class; validation and test stay real | Recall of the small classes is the target |
| Evaluation | Per-class recall and macro-F1 of the per-crop models: 5-fold CV of the development set (about 41 penetration photos) and the T23 test set, paired against the T23 models as in T30 | The test set alone is too small for penetration |

## 4. Stages

1. **Generators on the training split**: export, StyleGAN2-ADA (about 1 day on one H100), diffusion LoRA (hours). Realism and memorisation checks. Stop here if the crops are not realistic.
2. **Augmentation on the fixed split**: retrain the per-crop models (ResNet152, EfficientNet-B0, VGG16, MobileNetV3-Small as in T32, or all 8) with GAN crops and with diffusion crops; test-set comparison.
3. **CV**: generators per fold (5 × 2), CV retraining, per-class recall with intervals.

## 5. Files

| File | Change |
|---|---|
| `src/data/export_generator_crops.py` | New: training crops as a class-labelled 256 px generator dataset (`dataset.json` in StyleGAN's format) |
| `tests/test_generator_crops.py` | Training crops only, square, labels in the classifiers' class order; refuses an output inside the repo |
| `scripts/slurm/stylegan_train.sbatch` | New: tests, export, StyleGAN2-ADA training (optional `FOLD`) |
| `scripts/slurm/diffusion_lora_train.sbatch` | New: tests, 512 px export, Stable Diffusion LoRA training (optional `FOLD`) |
| `src/evaluation/generate_crops.py` | New: crops of chosen classes from a StyleGAN snapshot or a LoRA, one seed per crop |
| `src/evaluation/generated_quality.py`, `tests/test_generated_quality.py` | New: KID, nearest training crop and panels; tested with a stand-in detector |
| `scripts/slurm/generated_crops.sbatch` | New: `MODE=generate` or `MODE=quality` |

Tools on Raptor: `/project/a260012_FORWCL/tools/` (`stylegan3` c233a91, `diffusers_examples` v0.31.0, `venv_diffusers`: diffusers 0.31.0, transformers 4.46.3, peft 0.13.2, accelerate 1.1.1, datasets 3.1.0 on the container's PyTorch 2.6); weights and the Inception detector in `torch_cache/`. Outputs on Raptor only (patient-derived): `$DERIVED_DIR/generator_crops_{256,512}[_fold<k>]/`, `$EXPERIMENTS_DIR/generators/`.

## 6. Ethics

Generated crops are derived from patient photographs. Whether they may be shown in the thesis or a publication is for the ethics approval to say; until then they stay on Raptor like the photographs.

## 7. Run log

- 2026-11-09: NVlabs/stylegan3 cloned to `/project/a260012_FORWCL/tools/stylegan3` (c233a91); the researcher approved running it and the diffusion code on Raptor.
- Smoke tests: StyleGAN (job 243783, 8 kimg, 15 s/kimg, custom CUDA ops built) and LoRA (job 243785, 20 steps, 3.4 steps/s). Export: 1,310 training crops (contusion 198, cut 94, graze 297, gunshot 490, laceration 115, penetration 39, scratch 77).
- The first full StyleGAN run (243784) failed at its first KID: the host's CA-bundle path does not exist in the container. The Inception detector was fetched on the login node into `torch_cache/dnnlib` and the jobs now set the container's CA bundle.
- Running: StyleGAN2-ADA 5,000 kimg (job 243786, about 22 h) and LoRA 15,000 steps (job 243787, about 75 min). StyleGAN KID at kimg 0: 0.361.
