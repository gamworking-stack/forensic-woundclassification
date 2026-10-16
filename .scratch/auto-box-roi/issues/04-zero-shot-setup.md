# 4Z setup: model, prompts and threshold

Type: grilling
Status: resolved
Blocked by: 01, 03

## Question

Which zero-shot detector and which text prompt(s) does arm 4Z use? How is the prompt chosen on validation (fixed in advance, or a small set compared by validation box-matching F1)? Settle any model-specific settings (box and text thresholds) within the threshold rule already fixed on the map.

Inputs from the research:
- **Short list**: Grounding DINO first, OWLv2 second, SAM 3 only if it installs.
- **Expected accuracy**: low, especially for small boxes; prompt wording mattered most in the published evidence.
- **Decide**:
  - one model or two;
  - the candidate prompts (plain "wound" vs. per-type or attribute prompts such as "cut", "bruise", "abrasion");
  - how the prompt is chosen on validation.

## Answer

(2026-10-16, grilled with the researcher)

1. **Models**: Grounding DINO (`IDEA-Research/grounding-dino-base`) and OWLv2 (`google/owlv2-large-patch14-ensemble`) are both run in a selection stage. **Only the winner goes forward as 4Z** for the crops and the retraining. The detection metrics of both models are reported.
2. **Prompts**: four, frozen 2026-10-16. Lowercase, each phrase ending in `.`. OWLv2 receives the same phrases as a list. Every returned box counts as "wound"; its phrase label is ignored.
   - P1 class-agnostic: `wound .`
   - P2 class-agnostic: `injury on skin .`
   - P3 the 7 class names: `cut wound . graze wound . gunshot wound . contusion . laceration . scratch . penetration wound .`
   - P4 descriptive, following the proposal's definitions (DiMaio 2021, see `GLOSSARY.md`):
     `a straight incised cut with clean sharp edges on skin . a superficial scraped abrasion on skin . a narrow linear scrape abrasion on skin . a bruise with bleeding under intact skin . an irregular torn wound with abraded bruised edges and tissue bridges . a puncture or stab wound made by a pointed object . a round bullet wound on skin .`
     Features that cannot be seen in a photo (depth, mechanism) are kept so the prompt matches the proposal.
3. **Small wounds**: tiling is a pre-registered, no-training option. Each image is split into 2 × 2 overlapping tiles, and the tile boxes and full-image boxes are merged with NMS at IoU 0.5. It is a candidate in the selection, not a default.
4. **Selection**: 2 models × 4 prompts × 2 (with / without tiling) = **16 configurations**. Each is scored by class-agnostic **mAP@0.5 on train + validation together** (356 cases), against manual boxes. Ties go to the simpler setting: fewer words, then no tiling. 4Z trains nothing, so the development split is used whole for stability (scratch and penetration are few in validation alone). Test is never touched.
5. **Boxes kept**:
   - For mAP: every box, up to 100 per image, with no score threshold.
   - Grounding DINO's `text_threshold` is set to 0.
   - For crops: the operating threshold is chosen by box-matching F1 at IoU ≥ 0.5, on train + validation for 4Z (the map's rule says validation; 4Z is the stated exception).
6. **Too-large boxes**: a box is dropped if its area share of the image is larger than the largest manual box's share in the training split. The rule applies to **every detector arm** and is recorded in the map's rules.
