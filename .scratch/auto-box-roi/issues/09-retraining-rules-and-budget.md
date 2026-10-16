# Retraining rules and GPU budget

Type: grilling
Status: open
Blocked by: 03

## Question

For retraining Experiments 1–3 on auto-box crops for all three arms (≈72 scenarios):
- Are the rules identical to S01–S32: lr grid {1e-5, 3e-5, 1e-4}, CNN-SVM grid, AlexNet ensemble seeds, max 200 epochs, patience 25, batch 32, seed 42?
- How are scenarios numbered (from S33)?
- In what order do the arms run?
- What happens if the GPU budget cannot cover all 72: drop the per-crop design, run fewer families, or wait?
