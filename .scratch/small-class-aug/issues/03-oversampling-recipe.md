# 5A recipe: oversampled crops and their augmentation

Type: grilling
Status: open
Blocked by: none

## Question

How does 5A reach the target counts?
- **Copies**: offline copies saved as files, or online repeated sampling with fresh random augmentation each time?
- **Transforms**: which "strong" class-specific transforms go beyond the standard training augmentation (scale, rotation, elastic, blur, lighting, cutout)? Are any colour changes banned, given that colour matters for contusion?
- **Sampling**: which source crops are repeated (all equally, or by photo so each photo counts the same)?
- **Interaction**: how does it interact with the per-box photo weights (T17) and the recomputed class weights?
