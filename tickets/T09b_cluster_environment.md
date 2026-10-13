# Ticket T09b: Cluster Environment

- **Ticket ID**: `T09b`
- **Phase**: Phase 4 (HPC)
- **Spec Reference**: [ROADMAP.md](../ROADMAP.md) §5 T09b
- **Status**: Completed (written after the fact)

---

## 1. Goal

A Python environment on Raptor with PyTorch on CUDA, verified on a GPU node.

## 2. Decision

| ROADMAP plan | What was used | Reason |
|---|---|---|
| micromamba environment from `requirements.txt` | the cluster's NGC container `/common/sif/pytorch-24.11-py3.sif` through `module load apptainer/1.3.5` | PyTorch, torchvision, CUDA, scikit-learn, pandas and Pillow come prebuilt for the H100 driver; nothing to install or keep in sync, and every job uses the same image |

- Jobs run `apptainer exec --nv --bind /project/a260012_FORWCL <sif> python ...` (`--nv` exposes the GPU).
- Pretrained weights are downloaded once on the login node into `TORCH_HOME=/project/a260012_FORWCL/torch_cache` (compute nodes may have no internet); every job exports `TORCH_HOME`.
- The training code reads `requirements.txt` versions only as minimums; the container's versions are recorded per run.

## 3. Acceptance Check

Every training run records its environment in `run_config.json` (`environment`). Checked 2026-10-14 for S03, S12 and S13:

| Item | Value |
|---|---|
| Python | 3.12.3 |
| torch | 2.6.0a0 (NGC 24.11) |
| torchvision | 0.20.0a0 |
| CUDA | 12.6 |
| Device | NVIDIA H100 PCIe (`raptor-g-002`) |

`torch.cuda.is_available()` is therefore true on the GPU nodes (T11 smoke runs 221591/221592 and all later runs trained on the H100). `pytest` suites ran in the same container (T05, T08b: 5 passed; T15: 6 passed). ✔
