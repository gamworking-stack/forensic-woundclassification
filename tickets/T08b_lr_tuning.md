# Ticket T08b: Learning-Rate Tuning on Validation

- **Ticket ID**: `T08b`
- **Phase**: Phase 3 (Training Code)
- **Spec Reference**: [specs/phase3_training_code.md](../specs/phase3_training_code.md) §3, §5
- **Scenarios**: S01–S12 (the six single fine-tuned families, full and ROI)
- **Status**: In Progress (code and tests done; Raptor run pending)

---

## 1. Goal

Tune the **learning rate** of every single fine-tuned scenario (S01–S12) on the validation split, as the proposal's hyperparameter-tuning step, while the test split stays untouched until the setting is chosen. Each scenario is trained once per grid value; the run with the best validation macro-F1 is the scenario's result, and only that run predicts the test split.

---

## 2. Decision (2026-10-13, before any S01–S12 run)

| Item | Decision | Reason |
|---|---|---|
| Tuned | Learning rate only, grid **{1e-5, 3e-5, 1e-4}** (3e-5 is the spec §3 default) | The setting that matters most when fine-tuning; the values are far enough apart to show through seed noise |
| Fixed | Batch size 32, weight decay, epochs (max 200, patience 25), augmentation, loss: spec §3 | Validation has 164 images from 63 cases. EXP002 seeds alone spread validation macro-F1 over 0.61–0.69, so finer settings cannot be told apart. Dropout exists only in some heads |
| Selection | Highest validation macro-F1 at the run's best epoch; ties to lower validation loss, then to the smaller learning rate | Same rule as the epoch selection inside each run |
| Unit | Per scenario (family × input path) | ROADMAP T08b: "chosen values recorded per scenario"; full and ROI are tuned the same way |
| Not tuned | S13–S16 | Their test results were seen on 2026-10-01 (T08), so their settings stay frozen. S13/S14 tuned their own SVM on validation (T06); S15/S16 members use the spec §3 default. Report in the thesis methods |
| Not done | k-fold cross-validation | Optional in the proposal "if the data size allows"; out of scope here |

The grid and the rule are fixed here, before any S01–S12 result exists, and are not changed after.

---

## 3. Requirements & Business Logic

1. **Tuning runs** (`train.py --skip-test`):
   - Ordinary T05 runs with `--lr <value>`, output in `<run>/<input_path>/<family>/lr_<value>/` (for example `lr_3e-5`).
   - `--skip-test` writes `validation_predictions.csv` only; the test split is never loaded.
   - Resume works as in T05.
2. **Selection** (`src/training/select_lr.py --scenario-dir <run>/<input_path>/<family>`):
   - Every grid folder must hold a finished run (`best.pt`, `validation_predictions.csv`) whose `run_config.json` learning rate matches its folder; otherwise it is an error.
   - Read the best epoch's validation macro-F1 and loss from `best.pt`, apply the rule in §2, and write `tuning_summary.csv` (`lr, best_epoch, validation_macro_f1, validation_loss, selected, output_dir`) **before** the test split is touched.
   - Predict the test split for the selected run only: rerun `train.py --resume` on its folder without `--skip-test`, which loads `best.pt` and writes both prediction files (no further training).
   - Copy the selected run's `run_config.json`, `training_history.csv`, `validation_predictions.csv` and `test_predictions.csv` into the scenario folder, test last, so T08 and `submit_all.sh` see an ordinary finished scenario. Checkpoints stay in the `lr_*` folder (T14 reads the selected one from `tuning_summary.csv`).
   - Rerunning is safe: nothing is retrained, and the same files are written again.
3. **HPC submission** (`scripts/hpc/submit_all.sh`):
   - For each of S01–S12: skip it when the scenario folder has `test_predictions.csv` or its select job is queued.
   - Submit each grid run that is neither finished nor queued through `train_single.sbatch` (`OUTPUT_DIR=.../lr_<value>`, `EXTRA_ARGS="--lr <value> --skip-test"`, job name `S01_efficientnet_b0_full_lr3e-5`).
   - Submit the select job (`scripts/slurm/tune_select.sbatch`, one GPU) with `afterok` on the grid jobs it waits for and `--kill-on-invalid-dep=yes`.
   - S13–S16 are submitted as before.
4. **Automated Testing** (`tests/test_tuning.py`, CPU, no downloads):
   - `--skip-test` writes no test predictions; a later `--resume` without it adds them without retraining.
   - Selection rule: best macro-F1 wins; ties go to lower loss, then to the smaller learning rate; a missing or mismatched grid folder raises.
   - Integration: three `--smoke --skip-test` runs, then `select_lr.py`. The scenario folder has the four files, `tuning_summary.csv` marks one run, its test predictions equal the selected run's, a rerun changes nothing, and T08 finds the scenario but not the `lr_*` folders.

---

## 4. Files to Create / Modify

| File | Purpose |
|---|---|
| `src/training/config.py` | `--skip-test` flag |
| `src/training/train.py` | Skip the test prediction when `--skip-test` is set |
| `src/training/select_lr.py` | CLI: choose the learning rate, predict test for it, fill the scenario folder |
| `scripts/slurm/tune_select.sbatch` | GPU job running `select_lr.py` |
| `scripts/hpc/submit_all.sh` | Submit grid runs and select jobs for S01–S12 |
| `tests/test_tuning.py` | Tests above |
| `runbook.md` | Tuning layout and commands |

---

## 5. CLI Interface

```bash
# On Raptor: everything unfinished, S01-S12 with tuning
bash scripts/hpc/watch_jobs.sh $(bash scripts/hpc/submit_all.sh)

# Selection on its own, once the three grid runs have finished
python src/training/select_lr.py \
  --scenario-dir /project/a260012_FORWCL/experiments/main/roi/efficientnet_b0
```

---

## 6. Acceptance Check

1. `pytest tests/` passes.
2. On Raptor, `submit_all.sh` finishes 3 grid runs and 1 select job for each of S01–S12 with exit code 0.
3. Every S01–S12 scenario folder has `tuning_summary.csv` with one selected row and no `test_predictions.csv` in any unselected `lr_*` folder.
4. `evaluate.py --experiments-root .../experiments/main` evaluates 16 scenarios with 0 missing.
5. Record the chosen learning rate per scenario here, and tick T08b in `ROADMAP.md` §9.
