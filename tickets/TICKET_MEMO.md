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

## T00: Skeleton and requirements

- **Commits**: see `git log --grep 'T00'`
- **Used**
  - `ROADMAP.md` §4 (layout), every `import` in `src/` and `tests/`
- **Created**
  - `tickets/T00_skeleton.md`
  - `specs/phase0_environment.md`
- **Changed**
  - `requirements.txt` (only packages the code imports)
  - `ROADMAP.md` (§4 as-built note, §9 tick)
- **Outside Git**: the fresh test environment (scratch only)

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

## T09: SLURM templates

- **Commits**: `4a045e5`, `b961312`, `ea7d7f5`, T08b commit; record: see `git log --grep 'T09'`
- **Used**
  - `src/training/train.py`, `src/training/cnn_svm.py`, `src/training/ensemble.py`, `src/training/select_lr.py`
  - `data/splits/case_level_split_manifest.csv`, `data/roi/roi_manifest.csv`
- **Created**
  - `scripts/slurm/train_single.sbatch`, `scripts/slurm/ensemble_combine.sbatch` (T07), `scripts/slurm/tune_select.sbatch` (T08b), `scripts/slurm/t03_roi_crops.sbatch` (T03)
  - `scripts/hpc/submit_all.sh`, `scripts/hpc/watch_jobs.sh`
  - `tickets/T09_slurm_templates.md`
- **Changed**
  - `scripts/slurm/train_single.sbatch` (log follow-up in `b961312`), `scripts/hpc/submit_all.sh` (T08b grid)
  - `runbook.md`, `ROADMAP.md` (§9 tick)

## T09b: Cluster environment

- **Created**: `tickets/T09b_cluster_environment.md`
- **Changed**: `runbook.md` (container versions), `ROADMAP.md` (§9 tick)
- **Outside Git**: `<project>/torch_cache/`; container `/common/sif/pytorch-24.11-py3.sif` (cluster-provided)

## T09c: Data transfer

- **Used**: `scripts/hpc/sync_to_raptor.sh` (created in T03, `a85411d`)
- **Created**: `scripts/hpc/fetch_results.sh`, `tickets/T09c_data_transfer.md`
- **Changed**: `scripts/hpc/sync_to_raptor.sh` (refuses uncommitted changes), `.gitignore` (`results/`), `runbook.md`, `ROADMAP.md` (§9 tick)
- **Outside Git**: `<project>/raw_data/wound_ml_7classes/`, `<project>/annotations/cvat_yolo_2026_09_30/`; laptop `../results_figures/`

## T10: Runbook

- **Created**: `tickets/T10_runbook.md`
- **Changed**: `runbook.md` (connecting, data locations, transfer, progress and failures, support), `ROADMAP.md` (§9 tick)

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

## T12: All 16 scenarios

- **Used**: `scripts/hpc/submit_all.sh`, `scripts/hpc/submit_alexnet_ensemble.sh`, `scripts/slurm/*.sbatch`, `src/training/*`
- **Created**: `tickets/T12_all_scenarios.md`
- **Changed**: `ROADMAP.md` (§9 tick, `83cb643`)
- **Outside Git**: `<exp>/<full|roi>/<family>/` for all 16 scenarios

## T12b: Cross-validation (supplementary)

- **Commits**: see `git log --grep 'cv'`
- **Used**
  - `data/splits/case_level_split_manifest.csv`, `data/roi/roi_manifest.csv`
  - `src/data/create_case_split.py` (`build_group_matrices`)
  - main run: `<exp>/<full|roi>/<family>/tuning_summary.csv`, `ensemble_members.csv`, CNN-SVM `run_config.json`
  - `src/evaluation/summarize.py` (`paired_comparison`), `scripts/slurm/train_single.sbatch`
- **Created**
  - `tickets/T12b_cross_validation.md`
  - `src/data/create_cv_folds.py`, `data/splits/cv_folds.csv`, `data/splits/cv_folds_report.json`
  - `src/training/cv_plan.py`, `scripts/hpc/submit_cv.sh`, `src/evaluation/cv_summary.py`
  - `tests/test_cv.py`
- **Changed**
  - `src/training/data.py`, `config.py`, `train.py` (CV mode, `--checkpoint last`)
  - `src/training/cnn_svm.py` (CV mode, `--svm-params-from`), `src/training/ensemble.py` (`--splits`)
  - `scripts/slurm/ensemble_combine.sbatch` (`ENSEMBLE_SPLITS`)
  - `runbook.md`, `tickets/T12_all_scenarios.md`
- **Outside Git**: `<project>/experiments/cv/` (fold runs, `oof_predictions.csv`, `summary/`, `cv_plan.tsv`)

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

## T14: Grad-CAM

- **Commits**: see `git log --grep 'T14'`
- **Used**
  - `<exp>/<full|roi>/<family>/tuning_summary.csv`, `ensemble_members.csv`, the selected runs' `best.pt`, `run_config.json`, `test_predictions.csv`
  - `data/roi/roi_manifest.csv` (test rows, wound boxes)
  - `src/training/models.py`, `data.py` (`load_image`), `transforms.py`, `src/data/create_roi_dataset.py` (`find_source_image`)
- **Created**
  - `tickets/T14_gradcam.md`
  - `src/evaluation/gradcam.py`
  - `tests/test_gradcam.py`
- **Changed**
  - `runbook.md`
- **Outside Git**: `<exp>/gradcam/` (overlays on patient photographs; not copied off Raptor)

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

## T16: Final verification

- **Commits**: see `git log --grep 'T16'`
- **Used**
  - every file of the main run (`<exp>/<full|roi>/<family>/`, grid runs, members, `summary/`, `gradcam/`) and the CV run (`experiments/cv/`)
  - `data/splits/case_split_assignments.csv`, `data/splits/cv_folds.csv`
  - `src/training/engine.py` (`EarlyStopping`), `select_lr.py` (`LR_GRID`, `choose`), `src/evaluation/metrics.py`, `summarize.py`, `gradcam.py` (`FAMILIES`)
- **Created**
  - `tickets/T16_final_verification.md`
  - `src/evaluation/verify.py`
  - `tests/test_verify.py`
- **Changed**
  - `scripts/hpc/fetch_results.sh` (copies `verification/`)
  - `runbook.md`
  - `ROADMAP.md` (§9 tick)
- **Outside Git**: `<exp>/verification/verification_report.csv`

## T17: Per-box ROI experiment (extra)

- **Commits**: see `git log --grep 'T17'`
- **Used**
  - `data/splits/case_level_split_manifest.csv`, the 7 CVAT exports of 30 Sep 2026
  - `src/data/create_roi_dataset.py` (`--crop-mode individual`), `src/data/compare_roi_manifests.py`
  - `src/evaluation/summarize.py` (`load_pairs`, `paired_comparison`), `src/evaluation/metrics.py`
  - the evaluated main run (`<exp>/<full|roi>/<family>/`) for the comparison
- **Created**
  - `tickets/T17_box_roi.md`
  - `data/roi_box/roi_manifest.csv`, `data/roi_box/crop_summary.json` (dry run)
  - `src/training/photo.py`, `src/evaluation/box_summary.py`
  - `tests/test_box_roi.py`
- **Changed**
  - `src/training/config.py`, `data.py`, `engine.py`, `train.py`, `common.py`, `cnn_svm.py`, `ensemble.py` (`box` input path)
  - `src/evaluation/evaluate.py` (S17-S24), `src/evaluation/plots.py` (forest-plot labels)
  - `scripts/slurm/train_single.sbatch`, `scripts/slurm/t03_roi_crops.sbatch`, `scripts/hpc/submit_all.sh`, `scripts/hpc/submit_alexnet_ensemble.sh`
  - `tests/synthetic_data.py` (`build_synthetic_box_dataset`)
  - `runbook.md`, `ROADMAP.md`
- **Outside Git**: `<project>/derived_data/roi_box_p008/`, `<project>/experiments/roi_box/`
