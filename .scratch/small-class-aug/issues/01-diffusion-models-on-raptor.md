# Which diffusion model and LoRA tooling can make wound crops on Raptor

Type: research
Status: open
Blocked by: none
Findings: branch `research/diffusion-models-for-wound-crops`, file `docs/research/diffusion-models-for-wound-crops.md`

## Question

Which open diffusion models could generate synthetic wound crops for arm 5B, fine-tuned with LoRA on about 70–100 training crops per class? Candidates include Stable Diffusion 1.5, SD 2.1, SDXL, SD 3.5 Medium, PixArt and FLUX.1-schnell. For each, find:
- the licence (research use, use of generated images in a thesis);
- the size of the weights and the GPU memory needed for LoRA training and for generation;
- whether `diffusers` + `peft` run offline under torch 2.6.0a0 (NGC 24.11 container, Python 3.12), and which versions to pin (the container's torch pre-release broke transformers 4.57; see T19 §2.10);
- training time for about 1,000 steps on one H100.

Also find published evidence on using synthetic images to improve classification of small classes in medical or dermatology images: which generation method was used (text-to-image LoRA, DreamBooth, image-to-image), how many images were added, and what gain was reported; plus any warnings (memorization, a drop on real test sets). End with a ranked short list (at most 3).
