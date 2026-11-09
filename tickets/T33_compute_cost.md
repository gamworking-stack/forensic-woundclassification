# Ticket T33: Computational Cost of the Models

- **Ticket ID**: `T33`
- **Phase**: Extra analysis (ROADMAP §5 "Extra experiments")
- **Spec Reference**: [T05](T05_shared_training_library.md) (training recipe), [T06](T06_cnn_svm.md) (CNN-SVM), [T23](T23_relabel_rerun.md)
- **Scenarios**: none new; the 8 families of S01–S16 and S25–S32 (same networks and input sizes)
- **Status**: Done (Raptor, 2026-11-09; results §4)

---

## 1. Goal

The researcher asked (2026-11-09) for a manuscript table of input size, number of parameters, computational complexity, inference latency, throughput and peak training VRAM of every model.

## 2. Decisions

| Item | Decision | Reason |
|---|---|---|
| Models | Built as trained (`build_model`, 7 classes), random weights; no checkpoint loaded | Weights do not change the cost; full-image, per-crop and CLAHE models share network and input size |
| Input size | The family's own: 384 × 384, ViT-B/16 224 × 224 | `models.input_size` |
| Parameters | All parameters (every family is fully fine-tuned) | |
| FLOPs | PyTorch `FlopCounterMode` on one image: convolutions and matrix products, a multiply-add = 2 FLOPs | Built in, no extra package; the convention is stated in the table note |
| Latency | One image, model only (no decoding or resizing), median of 200 GPU passes (bfloat16 autocast, as `predict`) and of 30 CPU passes (float32, 8 threads) | GPU as used here; CPU closer to a forensic unit without a GPU |
| Throughput | Batches of 32 on the GPU (the evaluation batch) | |
| Training memory | `torch.cuda.max_memory_allocated` over 3 steps of `train_one_epoch` (batch 32, bfloat16, AdamW, weighted loss): weights, gradients, optimiser state and activations; CUDA context not included | The training loop itself, not a copy of it |
| AlexNet ensemble | Its 3 members one after the other (soft vote) for parameters, FLOPs and inference; training memory of one member | Members are trained one at a time |
| CNN-SVM | VGG16 + ResNet152 + EfficientNet-B0 feature extractors plus the SVM refitted from the cached train+validation features of the T23 run, as `cnn_svm.py` fits its final model. SVM FLOPs: support vectors × 7,424 features × 3 (RBF); its CPU time is added to both latencies and to the batch time. Memory: peak of feature extraction (batch 64, one extractor at a time); no network is trained | Its real prediction pipeline |
| Node | `raptor-g-002` (H100 PCIe, AMD EPYC 9454), alone on the GPU | `raptor-g-001` lends its GPU out in shards, which would disturb the timing |

## 3. Files

| File | Change |
|---|---|
| `src/evaluation/compute_cost.py` | New |
| `tests/test_compute_cost.py` | FLOP convention, ensemble = 3 members, SVM refit of a smoke CNN-SVM run, the table |
| `scripts/slurm/compute_cost.sbatch` | New: tests, then the benchmark (one GPU) |
| `runbook.md` (T23 step 16), `ROADMAP.md`, `tickets/TICKET_MEMO.md` | Command, scope, memo |

Outputs in `$EXPERIMENTS_DIR/compute_cost/` (copied to `../results_figures/Computational resources/`): `compute_cost.csv` (per family, with interquartile ranges and reserved memory), `svm_cost.csv` (both CNN-SVM runs), `compute_cost.json` (settings and environment). Word table `../08_Manuscript/Table 5 (computational cost).docx`.

## 4. Run log and results

- 2026-11-09 (Raptor, job 243757, commit a3c8681): first run. ViT-B/16 came out at 22.5 GFLOPs against torchvision's 17.56 GMACs (35.1 GFLOPs): under `no_grad` nn.MultiheadAttention takes a fused path the counter does not see. Counting with gradients on (job 243759, commit 0c230e9) still missed the two attention products (16.85 GMACs): the fused CPU attention kernel is not in the counter's table either, so the reference test failed and the job stopped.
- 2026-11-09 (Raptor, job 243760, 2 min, commit 19439ef): attention on the math backend while counting. `pytest tests/test_compute_cost.py` 6 passed, ViT-B/16 and GoogLeNet within 1% of torchvision's catalogue at 224 px. The other families' counts did not change between runs; GPU times agreed within 3% and CPU times within 10% between jobs 243757 and 243760.

| Family | Input | Parameters (M) | GFLOPs | GPU latency (ms) | CPU latency (ms) | GPU throughput (images/s) | Peak training memory (GiB) |
|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | 384 | 4.02 | 2.26 | 4.57 | 19.9 | 3,144 | 4.19 |
| ResNet152 | 384 | 58.16 | 67.66 | 9.94 | 119.1 | 1,104 | 8.89 |
| MobileNetV3-Small | 384 | 1.53 | 0.32 | 2.97 | 6.6 | 10,550 | 0.98 |
| GoogLeNet | 384 | 5.61 | 8.80 | 3.99 | 38.6 | 3,626 | 2.95 |
| VGG16 | 384 | 134.29 | 90.44 | 1.73 | 156.9 | 1,380 | 6.13 |
| ViT-B/16 | 224 | 85.80 | 35.13 | 3.55 | 70.1 | 3,578 | 3.46 |
| CNN-SVM (full-image SVM) | 384 | 196.41 | 160.38 | 18.76 | 298.3 | 200 | 5.17 (extraction) |
| AlexNet ensemble | 384 | 171.10 | 12.44 | 1.67 | 42.5 | 4,986 | 1.34 (one member) |

- CNN-SVM's SVM: 801 support vectors (full image, RBF C=100 gamma=1e-6) and 1,028 (per crop, gamma=1e-5); 0.018 and 0.023 GFLOPs; 3.2 and 3.9 ms for one image, 98 and 128 ms for 32 on the CPU (scikit-learn `predict_proba`), which limits its throughput.
- One image on the GPU is bound by kernel launches, not arithmetic: VGG16 (16 layers) takes 1.7 ms against MobileNetV3-Small's 3.0 ms (over 50 small layers) at 280 times the FLOPs. Batch throughput and CPU latency follow FLOPs more closely.
- Word table: `../08_Manuscript/Table 5 (computational cost).docx` (landscape, Table 1 style), built by `build_table5.py` kept with its note text in `../results_figures/Computational resources/`.
