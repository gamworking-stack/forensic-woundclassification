# 5B method: how synthetic crops are generated

Type: grilling
Status: open
Blocked by: 01, 02

## Question

Given the research and a positive ethics answer, decide:
- the model;
- the conditioning: one LoRA per class, or one LoRA with class tokens; text-to-image or image-to-image from real crops (and at what strength);
- the resolution;
- the prompts;
- the LoRA training settings (rank, steps, learning rate);
- the sampler, number of steps and guidance;
- the seeds;
- how many images are generated before filtering, to end with the target counts.

Inputs from the research: the short list is SD 1.5 (recommended start), SDXL, FLUX.1 [dev] / SD 3.5 Medium. Gains in the literature come from DreamBooth/LoRA text-to-image with a class token; they level off above about 10 synthetic per real. Our 1:1 cap is conservative.
