# Which zero-shot detectors can find forensic wounds

Type: research
Status: resolved
Blocked by: none
Findings: branch `research/zero-shot-wound-detectors`, file `docs/research/zero-shot-wound-detectors.md`

## Question

Which open-vocabulary, zero-shot object detectors could serve as detector arm **4Z**? Candidates include Grounding DINO, OWLv2, Florence-2 and YOLO-World. For each, find:
- licence;
- where the weights come from and how big they are;
- whether it runs offline on CPU or GPU with torch 2.6 / torchvision 0.20, and which extra libraries it needs;
- any published evidence of detecting wounds, lesions or skin findings, and which text prompts were used.

The outcome is a short list to choose from in "4Z setup".

## Answer

(2026-10-16; full findings with sources on branch `research/zero-shot-wound-detectors`, `docs/research/zero-shot-wound-detectors.md`)

- **No study runs a zero-shot detector on forensic wound photos.** The closest evidence is weak:
  - Grounding DINO-B with the prompt `"wound ."` gets mAP 6.2 / mAP50 12.2 on DFUC 2022 foot ulcers, and about 0 on small boxes.
  - GLIP-T gets 0.1 AP with the class name, but 25.9 AP with hand-written attribute prompts (DFUC 2020). The prompt matters more than the model.
  - Grounding DINO and OWLv2 both get 2.1 AP on the RF100-VL medical datasets.
- **Short list**:
  1. **Grounding DINO** (`IDEA-Research/grounding-dino-base`): Apache-2.0, scores for every box, runs offline in transformers. The `mm_grounding_dino` and `llmdet` checkpoints load with the same code.
  2. **OWLv2** (`google/owlv2-large-patch14-ensemble`): Apache-2.0, 960 px input suits small boxes, can also search with an example image.
  3. **SAM 3**, only if it installs: gated weights, a restrictive licence, and its official code needs torch ≥ 2.7.
- **Dropped**:
  - Florence-2: its boxes have no scores, so no AP.
  - YOLO-World: (A)GPL, and it fetches CLIP from Git.
  - Grounding DINO 1.5/1.6: cloud API only.
  - BiomedParse: non-commercial licence, and it segments rather than detects.
  - MedROV and OmDet-Turbo: no licence file or no medical evidence.
  - GLIP: an old CUDA build.
- **Container trap**: NGC 24.11 ships torch `2.6.0a0`, a pre-release that ranks below 2.6.0. Pin `transformers==4.57.6` (≤ 5.19.0) and load only `model.safetensors` weights. None of this has been tried in the container yet; it is now part of "Raptor environment and GPU budget for Experiment 4".

**Correction (2026-10-16, Raptor check):** use `transformers==4.52.4`, not 4.57.6. On the container's torch 2.6.0a0, 4.57.6 fails on import. With 4.52.4 both Grounding DINO and OWLv2 load offline. See "Raptor environment and GPU budget for Experiment 4".
