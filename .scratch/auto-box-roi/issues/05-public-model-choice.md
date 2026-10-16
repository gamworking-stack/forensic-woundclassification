# 4P: which public wound model, or drop the arm

Type: grilling
Status: open
Blocked by: 02, 03

## Question

Given the research findings, which public model is arm 4P? If its output is a mask, how does it become boxes (one box per connected region, minimum area)? Or is 4P dropped under the map's rule, and with what wording for the thesis?

Inputs from the research: the only forensic option is the Hanterdsith YOLOv8 model (simulated wounds, 3 of our 7 types). Chronic-wound alternatives are WILLIE (boxes) and FUSegNet / HarDNet-DFUS (masks). The research suggests a **gated pilot**: box recall on the training split decides whether 4P stays. Decide:
- whether a pilot on the training split is allowed, given that nothing may be chosen on test;
- the pilot's pass criterion;
- whether to wait for a possible reply from the Zurich group (see "Ask the Zurich group for their forensic wound model").

Note from "4Z setup: model, prompts and threshold": the oversize-box rule (drop boxes larger than the largest training-split manual box) applies to this arm too.
