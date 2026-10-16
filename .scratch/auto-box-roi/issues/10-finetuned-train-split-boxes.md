# 4F auto boxes on its own training photos

Type: grilling
Status: open
Blocked by: 06

## Question

Retraining under arm 4F needs auto boxes on the **training** photos. But the 4F detector was fitted to those photos' manual boxes, so its boxes there will be near-perfect, while on validation and test they will not. Training crops would then be cleaner than test crops. How is this handled?
- (a) Cross-fit: k detectors, each boxes the training photos it did not see.
- (b) Accept it and state it as a limitation.
- (c) Use manual-box crops for training and auto boxes only for validation and test.
Also: does the same concern apply to choosing the threshold on validation?
