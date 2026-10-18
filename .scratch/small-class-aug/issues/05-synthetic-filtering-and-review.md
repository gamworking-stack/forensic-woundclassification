# 5B checks: memorization filter and blinded expert review

Type: grilling
Status: open
Blocked by: 04

## Question

- **Memorization filter**: which similarity measure is used against the training crops (e.g. CLIP or DINO embedding cosine, SSIM, LPIPS), with what threshold, and what happens to images above it?
- **Other filters**: is there any quality filter? A classifier-based filter is excluded, because it is circular.
- **Blinded review**: the researcher and the advisor review a shuffled sample of about 30 real and 30 synthetic crops per small class, rating the wound type and real vs synthetic.
  - Decide the exact sample, the rating form, and how the result is reported.
  - Decide whether the review can veto 5B (e.g. if most synthetic crops are rated unrealistic or the wrong type).
