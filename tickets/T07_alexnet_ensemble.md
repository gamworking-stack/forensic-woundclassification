# Ticket T07: AlexNet Soft-Voting Ensemble

- **Ticket ID**: `T07`
- **Phase**: Phase 3 (Training Code)
- **Spec Reference**: [specs/phase3_training_code.md](../specs/phase3_training_code.md) §5
- **Scenarios**: S15 (full), S16 (ROI)
- **Status**: Completed

---

## 1. Goal

Train **3 AlexNet members** with the T05 library (`src/training/train.py --family alexnet`), each with a different seed, and combine them by **soft voting**: the unweighted mean of the members' class probabilities. Implement the combining step as `src/training/ensemble.py`, whose output files match every other scenario so T08 evaluates the ensemble like a single model.

---

## 2. Requirements & Business Logic

1. **Members**:
   - Same protocol as every fine-tuned family (spec §3): ImageNet-pretrained AlexNet, all layers fine-tuned, 384 px, same augmentation, loss, optimizer and early stopping on validation macro-F1.
   - Seeds **43, 44, 45** (spec §3). The seed changes the new head's initialisation, the shuffling order and the augmentation draws; split and data are identical.
   - Each member is an ordinary `train.py` run with its own output folder, `member_01`, `member_02`, `member_03`, under `<run>/<input_path>/alexnet_ensemble/`.
   - Each member selects its own best checkpoint on validation, independently of the others.
2. **Soft voting** (`ensemble.py`):
   - Read each member's `validation_predictions.csv` and `test_predictions.csv`.
   - Check, for every split, that all members have the same rows in the same order (`relative_path`, `pseudonimized_id`, `split`, `true_class`) and the same `prob_*` columns. Any mismatch is an error, never a silent re-sort.
   - Ensemble probability = mean of the member probabilities per image and class; `predicted_class` = argmax. Equal weights, so nothing is tuned and validation is not used for the combination.
   - Fail if a member is missing a prediction file or its run has not finished (`best.pt` absent).
3. **Outputs** (`<run>/<input_path>/alexnet_ensemble/`):
   - `validation_predictions.csv`, `test_predictions.csv`: same columns as T05 (spec §4.4).
   - `ensemble_members.csv`: `member, seed, best_epoch, best_validation_macro_f1, output_dir`, the same layout as EXP002.
   - `run_config.json`: member folders and seeds, class list, combination rule (`mean`), git and environment info.
   - `ensemble.log`.
4. **Grad-CAM note for T14**: the representative member is the one with the **highest validation macro-F1**, ties to the lower seed. `ensemble_members.csv` marks it in a `representative` column.
5. **HPC submission** (`scripts/hpc/submit_alexnet_ensemble.sh <full|roi>`):
   - Submit the 3 members as GPU jobs through `scripts/slurm/train_single.sbatch` (`FAMILY=alexnet`, `EXTRA_ARGS="--seed 4x"`, `OUTPUT_DIR=.../member_0k`).
   - Submit the combine job (`scripts/slurm/ensemble_combine.sbatch`, CPU partition `compute`) with `--dependency=afterok:<member ids>`, so it starts only when all 3 succeed.
   - Print all 4 job IDs, so they can be passed to `scripts/hpc/watch_jobs.sh`.
   - Resubmitting resumes unfinished members (`--resume`) and reruns the combine step.
6. **Automated Testing** (`tests/test_ensemble.py`, CPU, no downloads):
   - Unit: averaging hand-made member CSVs gives the expected probabilities and argmax; rows in a different order, a missing file or different class columns each raise a clear error.
   - Representative member: highest validation macro-F1, ties to the lower seed.
   - Integration: 3 `train.py --smoke --no-pretrained` AlexNet members on the synthetic dataset (`tests/synthetic_data.py`), then `ensemble.py`. Outputs exist, probabilities sum to 1, and the columns equal T05's.

---

## 3. Files to Create / Modify

| File | Purpose |
|---|---|
| `src/training/ensemble.py` | CLI: check members, average probabilities, write outputs |
| `scripts/slurm/ensemble_combine.sbatch` | CPU job running `ensemble.py` |
| `scripts/hpc/submit_alexnet_ensemble.sh` | Submit 3 member jobs and the dependent combine job |
| `tests/test_ensemble.py` | Tests above |
| `runbook.md` | Submit and watch commands |

---

## 4. CLI Interface

```bash
# On Raptor: members + combine for one input path
bash scripts/hpc/submit_alexnet_ensemble.sh roi

# Combine step on its own
python src/training/ensemble.py \
  --members /project/a260012_FORWCL/experiments/main/roi/alexnet_ensemble/member_0{1,2,3} \
  --output-dir /project/a260012_FORWCL/experiments/main/roi/alexnet_ensemble
```

---

## 5. Acceptance Check

1. `pytest tests/` passes.
2. On Raptor, `submit_alexnet_ensemble.sh full` and `... roi` each finish 3 members and the combine job with exit code 0.
3. `ensemble_members.csv` lists 3 members with distinct seeds and best epochs, and marks one representative. For the full path, the members' validation macro-F1 are in the EXP002 range (0.61–0.69).
4. Tick T07 in `ROADMAP.md` §9.

---

## 6. HPC Run

Members and the combine job finished on Raptor (first evaluated 2026-10-01; checked 2026-10-14).

| Path | Seed | Best epoch | Validation macro-F1 | Representative |
|---|---|---|---|---|
| full | 43 | 31 | 0.643 | yes |
| full | 44 | 32 | 0.639 | |
| full | 45 | 47 | 0.632 | |
| ROI | 43 | 21 | 0.744 | yes |
| ROI | 44 | 22 | 0.712 | |
| ROI | 45 | 18 | 0.677 | |

- Full-path members are in the EXP002 range (0.61–0.69). Seed 43 is the representative member on both paths (T14 Grad-CAM).
- Soft-voted test macro-F1 [95% case-bootstrap CI]: S15 full 0.662 [0.553, 0.748], S16 ROI 0.728 [0.607, 0.817].
