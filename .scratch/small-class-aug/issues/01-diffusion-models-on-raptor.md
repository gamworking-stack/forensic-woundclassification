# Which diffusion model and LoRA tooling can make wound crops on Raptor

Type: research
Status: resolved
Blocked by: none
Findings: branch `research/diffusion-models-for-wound-crops`, file `docs/research/diffusion-models-for-wound-crops.md`

## Question

Which open diffusion models could generate synthetic wound crops for arm 5B, fine-tuned with LoRA on about 70–100 training crops per class? Candidates include Stable Diffusion 1.5, SD 2.1, SDXL, SD 3.5 Medium, PixArt and FLUX.1-schnell. For each, find:
- the licence (research use, use of generated images in a thesis);
- the size of the weights and the GPU memory needed for LoRA training and for generation;
- whether `diffusers` + `peft` run offline under torch 2.6.0a0 (NGC 24.11 container, Python 3.12), and which versions to pin (the container's torch pre-release broke transformers 4.57; see T19 §2.10);
- training time for about 1,000 steps on one H100.

Also find published evidence on using synthetic images to improve classification of small classes in medical or dermatology images: which generation method was used (text-to-image LoRA, DreamBooth, image-to-image), how many images were added, and what gain was reported; plus any warnings (memorization, a drop on real test sets). End with a ranked short list (at most 3).

## Answer

(2026-10-18; full findings with sources on branch `research/diffusion-models-for-wound-crops`, `docs/research/diffusion-models-for-wound-crops.md`)

- **Short list**:
  1. **Stable Diffusion 1.5**: not gated, OpenRAIL-M, about 11 GB for LoRA, native 512 px (fits our crops and 384 px classifiers), and the smaller model memorizes less.
  2. **SDXL**: better quality, same licence family, LoRA under 16 GB, 1024 px.
  3. **FLUX.1 [dev]**: best recent dermatology LoRA result (DermaFlux 2026 preprint), but gated, 12B parameters, over 40 GB. **SD 3.5 Medium** is the lighter fallback.
- **Ruled out**: SD 2.1 (official repo no longer public), PixArt-Σ (no diffusers LoRA script, old PyTorch, 19 GB T5), FLUX.1 [schnell] (distilled; LoRA documented for [dev] only).
- **Licences**: all allow research use and generated images in a thesis.
  - OpenRAIL forbids use for medical advice or result interpretation, so the classifier must be stated as research-only.
  - FLUX [dev] requires a content review before sharing.
  - SD 3.5 and FLUX need a one-time authenticated download.
- **Pin set** (untested in the container): keep torch 2.6.0a0; diffusers 0.35.2, peft 0.17.1, accelerate 1.10.1, transformers 4.52.4, tokenizers 0.21.4, huggingface-hub 0.35.3, safetensors 0.6.2.
  - Install with `--no-deps` from a wheelhouse.
  - Use the training scripts from the diffusers `v0.35.2` tag.
  - diffusers ≥ 0.40 would pull a new torch.
  - Only safetensors weights load on this torch.
- **Cost**: an H100 covers every candidate. For SD 1.5, LoRA takes up to about 20 minutes per 1,000 steps (from a 2080 Ti benchmark), so a pilot is needed. Count 6 generators (main + 5 folds), or 18 with one LoRA per class.
- **Evidence**: gains are largest with tens of real images per class.
  - Sagers 2023 (SD + DreamBooth): up to +13 points, levelling off above 10 synthetic per real; −2.2 points at 228 real per class.
  - Akrout (textual inversion): +0.7. Derm-T2IM: +2–3.5.
  - Diabetic foot ulcer study: +2.5–4.5.
  - Ktena (Nature Medicine): a small drop in less skewed settings; real data mattered more in dermatology.
  - No public generator exists for traumatic or forensic wounds.
- **Memorization**:
  - Dar et al. 2025 found about 37% of patient training images memorized. Over-training raises it; augmentation and smaller models lower it.
  - Parameter-efficient fine-tuning cut extracted copies sharply compared with full fine-tuning.
  - Suggested filter: DINO (or SSCD) and CLIP embeddings; reject a synthetic crop whose nearest-training similarity exceeds the 95th percentile of train-to-held-out-real similarity; report rejections per class and fold. Keep LoRA training short.
