# Ticket Memo: Files Used, Created and Changed

One entry per ticket. Update it in the same commit that closes or extends a ticket.

- **Used**: read by the ticket's code or needed to run it, but not changed by the ticket.
- **Created**: new files added by the ticket.
- **Changed**: existing files the ticket edited.
- **Outside Git**: files the ticket writes on Raptor or outside the repo. They are not committed (patient data, crops, checkpoints, run outputs).

`<exp>` = `/project/a260012_FORWCL/experiments/main` and `<project>` = `/project/a260012_FORWCL` on Raptor.

To list a ticket's committed files from history:

```bash
git log --reverse --format='%h %s' --name-status --grep 'T08'
```

---

## T01: Metadata validation

- **Commits**: `38c8ce5`
- **Used**
  - `../thesis_data_30_sep_id_only.csv` (outside the repo; `thesis_data_30_sep.xlsx` also accepted)
  - `specs/phase1_dataset_and_split.md`
- **Created**
  - `src/data/__init__.py`
  - `src/data/validate_metadata.py`
  - `tests/test_validate_metadata.py`
  - `data/validated_metadata.csv`
  - `data/metadata_validation_report.json`
  - Repo setup in the same commit: `.gitignore`, `ROADMAP.md`, `requirements.txt`, `specs/phase1_dataset_and_split.md`, `tickets/T01_validate_metadata.md`, `tickets/T02_case_level_split.md`
- **Changed**: none

## T02: Case-level split

- **Commits**: `7127423`
- **Used**
  - `data/validated_metadata.csv`
  - `specs/phase1_dataset_and_split.md`
- **Created**
  - `src/data/create_case_split.py`
  - `tests/test_split_leakage.py`
  - `data/splits/case_level_split_manifest.csv` (frozen)
  - `data/splits/case_split_assignments.csv`
  - `data/splits/case_level_split_summary.csv`
  - `data/splits/case_level_split_report.json`
- **Changed**
  - `ROADMAP.md` (§9 tick)
  - `tickets/T02_case_level_split.md`

## T03: CVAT ROI cropping

- **Commits**: `5063ff9` (spec and ticket), `6cbb937`, `a85411d` (Raptor live run)
- **Used**
  - `data/splits/case_level_split_manifest.csv`
  - `src/data/validate_metadata.py` (`LABEL_NORMALIZATION_MAP`)
  - CVAT YOLO 1.1 exports: `~/Downloads/task_*_annotations_2026_09_30_*.zip` locally, `<project>/annotations/cvat_yolo_2026_09_30/` on Raptor
  - `../Researcher_ROI_results/dataset_information/roi_manifest.csv` (outside the repo; image sizes for the dry run)
  - `<project>/raw_data/wound_ml_7classes/` (raw images, live run only)
  - `specs/phase2_roi_preprocessing.md`
- **Created**
  - `specs/phase2_roi_preprocessing.md`, `tickets/T03_cvat_roi_cropping.md`
  - `src/data/create_roi_dataset.py`
  - `src/data/compare_roi_manifests.py`
  - `tests/test_roi_preprocessing.py`
  - `data/roi/roi_manifest.csv`
  - `data/roi/crop_summary.json`
  - `scripts/slurm/t03_roi_crops.sbatch`
  - `scripts/hpc/sync_to_raptor.sh`
  - `runbook.md`
- **Changed**
  - `ROADMAP.md` (§9 tick)
  - `tickets/T03_cvat_roi_cropping.md`
- **Outside Git**: 1,100 crops in `<project>/derived_data/roi_union_p008/<split>/<class>/` (SLURM job `221586`)

## T04: Full vs ROI audit

- **Commits**: `5063ff9` (ticket), `07ebbaf`
- **Used**
  - `data/splits/case_level_split_manifest.csv`
  - `data/roi/roi_manifest.csv`
  - `data/roi/crop_summary.json`
  - `src/data/create_roi_dataset.py` (`SPLIT_NAMES`)
  - `specs/phase2_roi_preprocessing.md`
- **Created**
  - `tickets/T04_full_vs_roi_audit.md`
  - `src/data/audit_full_vs_roi.py`
  - `data/audits/full_vs_roi_audit_table.csv`
  - `data/audits/full_vs_roi_audit_report.json`
- **Changed**
  - `tests/test_roi_preprocessing.py` (audit tests)
  - `ROADMAP.md` (§9 tick)
  - `tickets/T04_full_vs_roi_audit.md`

## T05: Shared training library

- **Commits**: `925dfa5` (spec and ticket), `9ac8689`
- **Used**
  - `data/splits/case_level_split_manifest.csv`
  - `data/roi/roi_manifest.csv`
  - `src/data/create_roi_dataset.py` (`find_source_image`, `verify_no_case_leakage`)
  - `specs/phase3_training_code.md`
- **Created**
  - `specs/phase3_training_code.md`, `tickets/T05_shared_training_library.md`
  - `src/training/__init__.py`
  - `src/training/config.py`
  - `src/training/data.py`
  - `src/training/transforms.py`
  - `src/training/models.py`
  - `src/training/engine.py`
  - `src/training/train.py`
  - `tests/test_training.py`
- **Changed**
  - `runbook.md` (weight prefetch, local test commands)
  - `scripts/hpc/sync_to_raptor.sh`
  - `ROADMAP.md` (§9 tick)
  - `tickets/T05_shared_training_library.md`

## T06: CNN-SVM

- **Commits**: `3d7d527`, `b8554b6` (wider grid), `dccf3a4` (HPC result, grid frozen)
- **Used**
  - `data/splits/case_level_split_manifest.csv`, `data/roi/roi_manifest.csv`
  - `src/training/models.py` (`feature_extractor`, `input_size`)
  - `src/training/engine.py` (`classification_metrics`, `seed_everything`)
  - `src/training/transforms.py`, `src/training/config.py`
  - `<project>/raw_data/wound_ml_7classes/`, `<project>/derived_data/roi_union_p008/`
- **Created**
  - `tickets/T06_cnn_svm.md`
  - `src/training/cnn_svm.py`
  - `src/training/common.py` (code moved out of `train.py`)
  - `tests/synthetic_data.py`
  - `tests/test_cnn_svm.py`
- **Changed**
  - `src/training/data.py` (`load_manifest` arguments, `prepare_rows`)
  - `src/training/train.py` (uses `common.py`)
  - `tests/test_training.py`
  - `scripts/slurm/train_single.sbatch` (`FAMILY=cnn_svm`)
  - `runbook.md`
  - `ROADMAP.md` (§9 tick)
  - `tickets/T06_cnn_svm.md`
- **Outside Git**: `<exp>/<full|roi>/cnn_svm/` (jobs `221604`, `221605`)

## T07: AlexNet ensemble

- **Commits**: `d7e041a`
- **Used**
  - `src/training/train.py` (members are ordinary `--family alexnet` runs)
  - `src/training/common.py`, `src/training/engine.py` (`load_checkpoint`)
  - `scripts/slurm/train_single.sbatch`
  - `scripts/hpc/watch_jobs.sh`
  - `tests/synthetic_data.py`
- **Created**
  - `tickets/T07_alexnet_ensemble.md`
  - `src/training/ensemble.py`
  - `scripts/slurm/ensemble_combine.sbatch`
  - `scripts/hpc/submit_alexnet_ensemble.sh`
  - `tests/test_ensemble.py`
- **Changed**
  - `runbook.md`
- **Outside Git**: `<exp>/<full|roi>/alexnet_ensemble/` and its `member_01` to `member_03`
- **Closed** 2026-10-14: HPC run recorded in the ticket (§6), `ROADMAP.md` §9 ticked.

## T08: Evaluation module

- **Commits**: `ba32f8d`, `ea7d7f5` (first HPC run recorded)
- **Used**
  - `<exp>/<full|roi>/<family>/validation_predictions.csv` and `test_predictions.csv` (from T05, T06, T07)
  - `specs/phase3_training_code.md` (§4.4 columns, §5.1 scenario IDs)
- **Created**
  - `tickets/T08_evaluation_module.md`
  - `src/evaluation/__init__.py`
  - `src/evaluation/metrics.py`
  - `src/evaluation/plots.py`
  - `src/evaluation/evaluate.py`
  - `tests/test_evaluation.py`
- **Changed**
  - `runbook.md` (evaluate command)
  - `ROADMAP.md` (§9 tick)
  - `tickets/T08_evaluation_module.md`
- **Outside Git**: per scenario `<split>_metrics.json`, `<split>_classification_report.csv`, `<split>_confusion_matrix.csv` and `.png`; `<exp>/evaluation_index.csv`
- **Changed later**: `tickets/T08_evaluation_module.md` (§7, full 16-scenario run, 2026-10-14)

## T08b: Learning-rate tuning

- **Commits**: see `git log --grep 'T08b'`
- **Used**
  - `src/training/engine.py` (`load_checkpoint`), `src/training/common.py` (`Logger`)
  - `scripts/slurm/train_single.sbatch` (grid runs, via `OUTPUT_DIR` and `EXTRA_ARGS`)
  - `src/evaluation/evaluate.py` (`find_scenarios`, in the tests)
  - `tests/synthetic_data.py`, `tests/test_training.py` (`run_args`)
- **Created**
  - `tickets/T08b_lr_tuning.md`
  - `src/training/select_lr.py`
  - `scripts/slurm/tune_select.sbatch`
  - `tests/test_tuning.py`
- **Changed**
  - `src/training/config.py`, `src/training/train.py` (`--skip-test`)
  - `scripts/hpc/submit_all.sh` (S01-S12 grid runs and select jobs)
  - `specs/phase3_training_code.md` (§5 T08b interface)
  - `runbook.md`
- **Outside Git**: `<exp>/<full|roi>/<family>/lr_<value>/` grid runs, `tuning_summary.csv`, `tuning.log`
- **Closed** 2026-10-14: chosen learning rates and acceptance recorded in the ticket (§7), `ROADMAP.md` §9 ticked.

## T09 (partial, no ticket file yet): SLURM templates

- **Commits**: `4a045e5`, `b961312`, `ea7d7f5`
- **Used**
  - `src/training/train.py`, `src/training/cnn_svm.py`
  - `data/splits/case_level_split_manifest.csv`, `data/roi/roi_manifest.csv`
  - `scripts/hpc/submit_alexnet_ensemble.sh` (called by `submit_all.sh`)
- **Created**
  - `scripts/slurm/train_single.sbatch`
  - `scripts/hpc/watch_jobs.sh`
  - `scripts/hpc/submit_all.sh`
- **Changed**
  - `scripts/slurm/train_single.sbatch` (log follow-up in `b961312`)
  - `runbook.md`

## T11: HPC smoke test

- **Commits**: `4a045e5`
- **Used**
  - `scripts/slurm/train_single.sbatch`
  - `src/training/train.py`
- **Created**
  - `scripts/hpc/submit_t11_smoke.sh`
- **Changed**
  - `runbook.md`
  - `ROADMAP.md` (§9 tick)
- **Outside Git**: EfficientNet-B0 2-epoch runs, full and ROI (jobs `221591`, `221592`)

## T13: ROC/AUC curves

- **Commits**: see `git log --grep 'T13'`
- **Used**
  - `<exp>/<full|roi>/<family>/validation_predictions.csv` and `test_predictions.csv`
  - `src/evaluation/metrics.py` (`load_predictions`, `one_vs_rest_auc`)
- **Created**
  - `tickets/T13_roc_auc_curves.md`
- **Changed**
  - `src/evaluation/metrics.py` (`roc_points`)
  - `src/evaluation/plots.py` (`plot_roc_curves`)
  - `src/evaluation/evaluate.py` (writes the curves per scenario and split)
  - `tests/test_evaluation.py`
  - `runbook.md` (evaluation outputs)
- **Outside Git**: per scenario `<split>_roc_curves.csv` and `.png`
- **Closed** 2026-10-14: HPC run recorded in the ticket (§6), `ROADMAP.md` §9 ticked.

## T15: Summary table and paired full vs ROI

- **Commits**: see `git log --grep 'T15'`
- **Used**
  - `<exp>/<full|roi>/<family>/test_predictions.csv` and `test_metrics.json` (from T08)
  - `src/evaluation/metrics.py` (`load_predictions`, `summary_metrics`), `src/evaluation/evaluate.py` (scenario IDs)
  - `tests/test_evaluation.py` (`write_predictions`, `write_scenario`)
- **Created**
  - `tickets/T15_summary_and_full_vs_roi.md`
  - `src/evaluation/summarize.py`
  - `tests/test_summary.py`
- **Changed**
  - `src/evaluation/plots.py` (`plot_full_vs_roi`)
  - `runbook.md`
  - `ROADMAP.md` (§9 tick)
- **Outside Git**: `<exp>/summary/summary_table.csv`, `full_vs_roi.csv`, `full_vs_roi_macro_f1.png`
