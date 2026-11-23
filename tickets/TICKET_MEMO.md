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
  - `../04_Early_experiments/Researcher_ROI_results/dataset_information/roi_manifest.csv` (outside the repo; image sizes for the dry run; moved from `../Researcher_ROI_results/` on 2026-10-15)
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
  - `src/training/photo.py`, `src/evaluation/box_summary.py`, `src/evaluation/cv_box_summary.py`, `src/evaluation/verify_box.py`
  - `tests/test_box_roi.py`
- **Changed**
  - `src/training/config.py`, `data.py`, `engine.py`, `train.py`, `common.py`, `cnn_svm.py`, `ensemble.py` (`box` input path)
  - `src/evaluation/evaluate.py` (S17-S24), `src/evaluation/plots.py` (forest-plot labels), `src/evaluation/gradcam.py` (`--box-manifest`), `src/evaluation/verify.py` (photo-level histories)
  - `src/training/cv_plan.py` (`--paths`), `scripts/hpc/submit_cv.sh` (`PATHS`)
  - `scripts/slurm/train_single.sbatch`, `scripts/slurm/t03_roi_crops.sbatch`, `scripts/hpc/submit_all.sh`, `scripts/hpc/submit_alexnet_ensemble.sh`
  - `tests/synthetic_data.py` (`build_synthetic_box_dataset`)
  - `runbook.md`, `ROADMAP.md`
- **Outside Git**: `<project>/derived_data/roi_box_p008/`, `<project>/experiments/roi_box/` (incl. `gradcam/`, `verification/`), `<project>/experiments/cv_box/`

## T18: Per-crop ROI experiment (extra)

- **Commits**: see `git log --grep 'T18'`
- **Used**
  - `data/roi_box/roi_manifest.csv` and `<project>/derived_data/roi_box_p008/` (the T17 crops)
  - `src/evaluation/metrics.py`, `src/evaluation/verify.py` (selection and budget checks)
- **Created**
  - `tickets/T18_crop_roi.md`
  - `src/evaluation/crop_summary.py`, `src/evaluation/verify_crop.py`
  - `tests/test_crop_roi.py`
- **Changed**
  - `src/training/config.py`, `data.py`, `train.py`, `cnn_svm.py` (`crop` input path)
  - `src/evaluation/evaluate.py` (S25-S32)
  - `scripts/slurm/train_single.sbatch`, `scripts/hpc/submit_all.sh`, `scripts/hpc/submit_alexnet_ensemble.sh`
  - `runbook.md`, `ROADMAP.md`
- **Outside Git**: `<project>/experiments/roi_crop/` (incl. `verification/`)

## T21: Single-box test photos, one full image vs one crop (extra)

- **Commits**: see `git log --grep 'T21'`
- **Used**
  - `data/roi_box/roi_manifest.csv`, `data/roi/roi_manifest.csv`
  - `src/evaluation/summarize.py` (`scenario_folder`, `paired_comparison`), `src/evaluation/metrics.py`, `src/evaluation/plots.py`
  - the evaluated main run (`<exp>/main/<full|roi>/<family>/`) and per-crop run (`<exp>/roi_crop/crop/<family>/`)
- **Created**
  - `tickets/T21_single_box.md`
  - `src/evaluation/single_box.py`
  - `tests/test_single_box.py`
- **Changed**
  - `runbook.md`, `ROADMAP.md`
- **Outside Git**: `<project>/experiments/roi_crop/single_box/`, `<project>/experiments/roi_crop/summary/single_box_*`

## T22: One crop per photo, the most confident (extra)

- **Commits**: see `git log --grep 'T22'`
- **Used**
  - `data/roi_box/roi_manifest.csv`
  - `src/evaluation/evaluate.py` (`evaluate_scenario`), `src/evaluation/summarize.py`, `src/evaluation/metrics.py`
  - the evaluated main run (`<exp>/main/full/<family>/`) and per-crop run (`<exp>/roi_crop/crop/<family>/`)
- **Created**
  - `tickets/T22_top_crop.md`
  - `src/evaluation/top_crop.py`
  - `tests/test_top_crop.py`
- **Changed**
  - `src/training/photo.py` (`photo_top_crop_frame`)
  - `runbook.md`, `ROADMAP.md`
- **Outside Git**: `<project>/experiments/roi_crop/crop/<family>/top_crop/`, `<project>/experiments/roi_crop/summary/top_crop_*`

## T23: Relabelled rerun of every experiment (23 Oct 2026 labels)

- **Commits**: see `git log --grep 'T23'`
- **Used**
  - `../thesis_data_23_oct.csv` (outside the repo; holds `patient_id`, never committed)
  - `data/validated_metadata.csv`, `data/roi/roi_manifest.csv`, `data/roi_box/roi_manifest.csv` (where each photo is filed, box sizes)
  - the 7 CVAT exports of 23 Oct 2026 (`../23 oct updated annotation CVAT/`; laceration as CVAT XML)
- **Created**
  - `tickets/T23_relabel_rerun.md`
  - `src/data/prepare_relabel.py`, `src/data/paths.py`
  - `scripts/hpc/relabel_env.sh`
  - `tests/test_relabel.py`
  - `data/relabel_2026_10_23/` (metadata, exclusions, split, CV folds, ROI and per-box manifests, audit, T19 reference boxes)
- **Changed**
  - `src/data/create_roi_dataset.py` (`source_label`, crop class from the manifest row, CVAT XML exports), `src/detection/boxes.py`, `src/detection/common.py`, `src/evaluation/verify_auto.py`
  - `scripts/hpc/sync_to_raptor.sh` (uploads the 23 Oct exports)
  - every script's `data/...` default (`DATA_DIR`), `scripts/slurm/*.sbatch`, `scripts/hpc/submit_*.sh`, `fetch_results.sh`
  - `runbook.md`, `ROADMAP.md`
- **Outside Git**: `<project>/experiments_relabel_2026_10_23/`, `<project>/derived_data/relabel_2026_10_23/`, `../results_figures/relabel_2026_10_23/`

## T24: Subgroups by age, race/ethnicity and skin tone (extra)

- **Commits**: see `git log --grep 'T24'`
- **Used**
  - `../thesis_data_26_oct.csv` (outside the repo; holds `patient_id` and demographics, never committed); checked again on `../thesis_data_27_oct.csv` (same test subgroups)
  - `data/relabel_2026_10_23/splits/case_level_split_manifest.csv`, `roi/roi_manifest.csv`, `roi_box/roi_manifest.csv`
  - `src/data/prepare_relabel.py` (`read_sheet`), `src/evaluation/single_box.py` (`photo_stem`), `src/evaluation/summarize.py`, `src/evaluation/metrics.py`
  - the T23 test predictions (`<exp>/main`, `roi_box`, `roi_crop`), copied to a temporary folder
- **Created**
  - `tickets/T24_subgroups.md`
  - `src/evaluation/subgroup.py`
  - `tests/test_subgroup.py`
- **Changed**
  - `runbook.md`, `ROADMAP.md`
- **Outside Git**: `../results_figures/relabel_2026_10_23/subgroup/`

## T25: One case per patient (extra)

- **Commits**: see `git log --grep 'T25'`
- **Used**
  - `../thesis_data_27_oct_corrected.csv` (outside the repo; holds `patient_id`, never committed)
  - `data/relabel_2026_10_23/splits/case_level_split_manifest.csv`, `roi/roi_manifest.csv`, `roi_box/roi_manifest.csv`
  - `src/data/prepare_relabel.py` (`read_sheet`), `src/evaluation/single_box.py` (`photo_stem`), `evaluate.py`, `summarize.py`, `crop_summary.py`, `subgroup.py`
  - the T23 prediction files (`<exp>/main`, `roi_box`, `roi_crop`), copied to a temporary folder
- **Created**
  - `tickets/T25_patient_cases.md`
  - `src/data/patient_cases.py`, `src/evaluation/patient_rescore.py`
  - `data/relabel_2026_10_23/splits/patient_cases.csv`
  - `tests/test_patient_cases.py`
- **Changed**
  - `runbook.md`, `ROADMAP.md`
- **Outside Git**: `../results_figures/relabel_2026_10_23_patient/`

## T26: Photos without a CVAT box dropped (extra)

- **Commits**: see `git log --grep 'T26'`
- **Used**
  - `data/relabel_2026_10_23/roi/crop_summary.json` (`unannotated_records`)
  - `src/data/create_case_split.py` (`generate_summary`), `src/data/prepare_relabel.py` (`SOURCE_ID`)
- **Created**
  - `tickets/T26_drop_unannotated.md`
  - `src/data/drop_unannotated.py`
  - `tests/test_drop_unannotated.py`
- **Changed**
  - `data/relabel_2026_10_23/` (metadata, exclusions, split manifest, patient cases, case assignments, CV folds, summary, reports)
  - `tests/test_relabel.py`, `ROADMAP.md`, `tickets/T25_patient_cases.md`

## T27: Manuscript figures (extra)

- **Commits**: see `git log --grep 'T27'`
- **Used**
  - `../thesis_data_27_oct_corrected.csv` (outside the repo; only counts are drawn)
  - `data/relabel_2026_10_23/excluded_photos.csv`, `splits/patient_cases.csv`, `roi_box/roi_manifest.csv`
  - `src/data/prepare_relabel.py` (`read_sheet`, `PATIENT_PLACEHOLDERS`)
- **Created**
  - `tickets/T27_manuscript_figures.md`
  - `src/visualization/manuscript_figures.py`
- **Changed**
  - `runbook.md`, `ROADMAP.md`
- **Outside Git**: `../results_figures/manuscript/`

## T28: Precision and recall with intervals (extra)

- **Commits**: see `git log --grep 'T28'`
- **Used**
  - `src/evaluation/metrics.py` (`load_predictions`; resampling as `bootstrap_draws`), `src/evaluation/patient_rescore.py`
  - the T23 test predictions (`<exp>/main/full`, `roi_crop/crop`), copied to a temporary folder; the T25 `test_metrics.json`
- **Created**
  - `tickets/T28_precision_recall.md`
  - `src/evaluation/precision_recall.py`
  - `tests/test_precision_recall.py`
- **Changed**
  - `ROADMAP.md`
- **Outside Git**: `../results_figures/relabel_2026_10_23_patient/precision_recall.csv`, `../08_Manuscript/Table 2 (precision recall).docx`

## T29: Confidence threshold, "Halloween experiment" (extra)

- **Commits**: see `git log --grep 'T29'`
- **Used**
  - `src/evaluation/metrics.py` (`load_predictions`, `summary_metrics`; resampling as `bootstrap_draws`), `precision_recall.py`, `patient_rescore.py`, `summarize.py` (`POOLED`), `plots.py`
  - the T23 test predictions (`<exp>/main/full`, `roi_crop/crop`), copied to a temporary folder; the T25 `test_metrics.json`
- **Created**
  - `tickets/T29_confidence_threshold.md`
  - `src/evaluation/confidence_threshold.py`
  - `tests/test_confidence_threshold.py`
- **Changed**
  - `runbook.md`, `ROADMAP.md`
- **Outside Git**: `../results_figures/Halloween experiment/`

## T30: More contrast experiment (extra)

- **Commits**: see `git log --grep 'T30'`
- **Used**
  - the T23 dataset (`data/relabel_2026_10_23/`), `src/data/create_roi_dataset.py` (`IMAGE_EXTENSIONS`, `find_source_image`), the T03 crop job, `submit_all.sh`
- **Created**
  - `tickets/T30_more_contrast.md`
  - `src/data/create_contrast_dataset.py`, `tests/test_contrast_dataset.py`
  - `scripts/slurm/contrast_images.sbatch`, `scripts/hpc/contrast_env.sh`
- **Changed**
  - `scripts/slurm/train_single.sbatch` (`RAW_IMAGE_ROOT`), `runbook.md`, `ROADMAP.md`
- **Outside Git**: `<project>/derived_data/relabel_2026_10_23_more_contrast/`, `<project>/experiments_relabel_2026_10_23_more_contrast/`, `../results_figures/More contrast experiment/`

## T31: Grad-CAM of the tattoo photos (extra)

- **Commits**: see `git log --grep 'T31'`
- **Used**
  - `src/evaluation/subgroup.py` (`subgroup_columns`), `src/data/prepare_relabel.py` (`read_sheet`), `src/evaluation/gradcam.py`
- **Created**
  - `tickets/T31_gradcam_tattoo.md`
  - `src/evaluation/subgroup_photos.py`, `scripts/slurm/gradcam_tattoo.sbatch`
- **Changed**
  - `src/evaluation/gradcam_panels.py` (`--photo-list`), `tests/test_gradcam.py`, `tests/test_subgroup.py`, `runbook.md`, `ROADMAP.md`
- **Outside Git**: `<project>/experiments_relabel_2026_10_23/tattoo_photos.csv`, `<run>/roi_crop/gradcam_tattoo/` in the T23 and T30 experiment folders

## T32: Confusion matrices per subgroup (extra)

- **Commits**: see `git log --grep 'T32'`
- **Used**
  - `src/evaluation/subgroup.py` (`test_photos`, `photo_keys`, `labelled_predictions`, `scored_levels`), `src/evaluation/plots.py` (`FAMILY_LABELS`, `INK`)
- **Created**
  - `tickets/T32_subgroup_confusion.md`, `src/evaluation/subgroup_confusion.py`
- **Changed**
  - `tests/test_subgroup.py`, `runbook.md`, `ROADMAP.md`
- **Outside Git**: `../results_figures/relabel_2026_10_23_patient/subgroup_confusion/`, `../08_Manuscript/Figures 3-4 (confusion matrices by subgroup).docx`

## T33: Computational cost of the models (extra)

- **Commits**: see `git log --grep 'T33'`
- **Used**
  - `src/training/models.py` (`build_model`, `feature_extractor`, `input_size`), `src/training/engine.py` (`train_one_epoch`, `_autocast`), `src/training/cnn_svm.py` (`fit_svm`, `DEFAULT_EXTRACTORS`), `src/training/data.py` (`load_manifest`, `prepare_rows`)
- **Created**
  - `tickets/T33_compute_cost.md`, `src/evaluation/compute_cost.py`, `tests/test_compute_cost.py`, `scripts/slurm/compute_cost.sbatch`
- **Changed**
  - `runbook.md`, `ROADMAP.md`
- **Outside Git**: `<project>/experiments_relabel_2026_10_23/compute_cost/`, `../results_figures/Computational resources/`, `../08_Manuscript/Table 5 (computational cost).docx`

## T38: The 19 Nov 2026 dataset (new annotation, new photos, split by patient)

- **Commits**: see `git log --grep 'T38'`
- **Used**
  - `src/data/prepare_relabel.py` (`OUTPUT_COLUMNS`, `SOURCE_ID`), `src/data/validate_metadata.py` (`LABEL_NORMALIZATION_MAP`), `src/data/create_case_split.py`, `src/data/create_roi_dataset.py`, `src/data/create_cv_folds.py`, `src/data/audit_full_vs_roi.py`
- **Created**
  - `tickets/T38_nov19_dataset.md`, `src/data/prepare_nov19.py`, `tests/test_nov19.py`, `scripts/hpc/nov19_env.sh`
- **Changed**
  - `scripts/hpc/sync_to_raptor.sh`, `requirements.txt`, `runbook.md`, `ROADMAP.md`
- **Outside Git**: `../Current dataset T23 (1,032 photos).xlsx` (the researcher's workbook, the source), `../datasets/relabel_2026_11_19/` (the dataset's lists; file names hold forensic case numbers), `../19 Nov annotation/`, `~/Desktop/รูปตามประเภท/` (new photos), on Raptor `<project>/datasets/relabel_2026_11_19/`, `<project>/annotations/cvat_2026_11_19/`
