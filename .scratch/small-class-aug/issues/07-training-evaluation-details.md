# Training, CV and evaluation details

Type: grilling
Status: open
Blocked by: none

## Question

- **Scenario numbering**: S105 onwards; how are main-run and CV scenarios numbered?
- **Rules**: are the training rules identical to S17–S24 (lr grid, CNN-SVM grid, ensemble seeds, budget)?
- **CV**: does it reuse the per-box CV baseline exactly? The baseline's fold runs used the main run's chosen lr and epochs (T12b); should the augmented runs do the same, or select again?
- **Layout**: output folders and run names.
- **Verification**: what `verify` checks for T20, e.g. no synthetic or oversampled crop in validation or test, and generator fold separation.
- **Budget**: the GPU estimate, and the cut order if time runs short.
