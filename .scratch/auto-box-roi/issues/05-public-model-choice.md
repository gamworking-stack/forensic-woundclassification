# 4P: which public wound model, or drop the arm

Type: grilling
Status: resolved
Blocked by: 02, 03

## Question

Given the research findings, which public model is arm 4P? If its output is a mask, how does it become boxes (one box per connected region, minimum area)? Or is 4P dropped under the map's rule, and with what wording for the thesis?

Inputs from the research: the only forensic option is the Hanterdsith YOLOv8 model (simulated wounds, 3 of our 7 types). Chronic-wound alternatives are WILLIE (boxes) and FUSegNet / HarDNet-DFUS (masks). The research suggests a **gated pilot**: box recall on the training split decides whether 4P stays. Decide:
- whether a pilot on the training split is allowed, given that nothing may be chosen on test;
- the pilot's pass criterion;
- whether to wait for a possible reply from the Zurich group (see "Ask the Zurich group for their forensic wound model").

Note from "4Z setup: model, prompts and threshold": the oversize-box rule (drop boxes larger than the largest training-split manual box) applies to this arm too.

## Answer

(2026-10-16, grilled with the researcher)

1. **Candidates**:
   - **Hanterdsith YOLOv8**, Hugging Face `trapezius60/forensic_wound_detection` at revision `0c61ba14…`, in three sizes: `best.pt` (v8n), `best_yolov8m_roboflow_v7.pt` (v8m), `best_yolov8x_robov9.pt` (v8x).
   - **WILLIE MINI** (`QianGroup/willie-weights`, MIT): a candidate **only if** it loads in a single smoke test from the code that is public (`ablations/mini_ablation_FINAL.py`), since its `willie_app/architectures/` folder is missing. If that one try fails, it is left out and the reason recorded.
   - **Excluded**: segmentation-only foot-ulcer models (FUSegNet, HarDNet-DFUS, Deepskin). They degrade off the foot, and excluding them means no mask-to-box conversion is needed.
2. **Input colour**: Hanterdsith was trained on grayscale-filtered images, so every Hanterdsith size is tried on colour and on grayscale input. WILLIE takes colour only. That gives **up to 7 configurations**.
3. **Selection**: as for 4Z. Class-agnostic mAP@0.5 on **train + validation**, every box kept (up to 100 per image), oversize rule applied. Ties go to the smaller model, then colour.
4. **Gate**: 4P is kept only if its selected configuration reaches **max recall ≥ 0.50 at IoU ≥ 0.5** on train + validation. If it does not:
   - 4P is **dropped**;
   - its development detection metrics are reported as a negative finding;
   - it is never run on test;
   - nothing is retrained for it.
5. **Classes**: all 8 Hanterdsith classes count as "wound" (class-agnostic). The count of boxes per source class is reported in an appendix.
6. **Loading the checkpoint safely**: the Hanterdsith `.pt` is a full pickle.
   - Scan it with `picklescan`.
   - Load it once in a throwaway job whose only writable bind is a scratch folder.
   - Re-save the weights as a plain `state_dict` / safetensors, and use only that file afterwards.
   - Record the SHA-256 of the original and of the re-saved file.
7. **Zurich model**: no waiting. If the Zurich group's model arrives before T19 is frozen, it joins the candidates under the same selection and gate; if later, it is future work.
8. **Thesis wording if dropped**: "No publicly available wound detector reached the pre-specified recall (≥ 0.50 at IoU ≥ 0.5) on the development photographs; arm 4P was therefore not carried forward." The detector's training domain is reported: simulated, grayscale-filtered wounds, or chronic ulcers.
