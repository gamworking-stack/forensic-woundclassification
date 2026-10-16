# Raptor environment and GPU budget for Experiment 4

Type: task (HITL)
Status: resolved
Blocked by: none

## Question

Facts the detector and retraining decisions wait on, checked on Raptor by the researcher:
1. Does `/common/sif/pytorch-24.11-py3.sif` contain `transformers`, `pycocotools`, `torchmetrics`, `ultralytics`? (`apptainer exec ... python -c "import X"` for each)
2. Can compute nodes reach the internet, or must weights be downloaded on the login node into `/project/a260012_FORWCL/` (as for `torch_cache`)?
3. Can extra Python packages be installed for the container (e.g. `pip install --user` or a bind-mounted folder)?
4. From the detector research: can `pip install --user "transformers==4.57.6"` (no `[torch]` extra) and `ultralytics` be installed for the container? Do `IDEA-Research/grounding-dino-base` and `google/owlv2-large-patch14-ensemble` (safetensors only) and the Hanterdsith `best.pt` load offline on a compute node, after the weights are downloaded on the login node?
5. GPU budget: how many GPU-hours are available, and what did one Experiment 1–3 scenario with its learning-rate grid cost (from the SLURM logs or `sacct`)?

## Answer

(2026-10-16; checked by the researcher on Raptor with `.scratch/auto-box-roi/check_raptor.sh` and a follow-up fix)

1. **Container contents**:
   - present: torch `2.6.0a0+df5bbc09d1.nv24.11` (a pre-release), torchvision `0.20.0a0`, **pycocotools**, OpenCV 4.10, numpy 1.26.4;
   - absent: transformers, torchmetrics, ultralytics.
2. **Internet**: login and compute nodes both reach the internet (huggingface.co HTTP 200 from raptor-m-003). Weights are still kept in a shared cache, `/project/a260012_FORWCL/hf_cache` (`HF_HOME`), and loaded with `HF_HUB_OFFLINE=1`.
3. **Extra packages**: these go into venvs on `/project` created with `--system-site-packages`, never into `~/.local`.
   - **Certificates**: pass `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` / `PIP_CERT=/etc/ssl/certs/ca-certificates.crt` with `--env`, because the host's Red Hat CA path does not exist in the Ubuntu container.
   - `venv_exp4`: **transformers 4.52.4**. Version 4.57.6, the one the research suggested, installs but fails on import: `masking_utils` needs `TransformGetItemToIndex`, which the torch 2.6.0a0 pre-release lacks.
   - `venv_exp4_yolo`: **ultralytics 8.4.178**, using the container's numpy 1.26.4 and OpenCV 4.10. Its own `opencv-python` (needs `libxcb`) and numpy 2.5 were uninstalled. Ultralytics warns that "torchvision 0.20 is incompatible with torch 2.6"; this pairing is NVIDIA's own, so the warning is a false alarm on this container, still to be confirmed when YOLO first runs.
4. **Weights downloaded**, loaded offline on a compute node with one synthetic image on CPU:
   - `IDEA-Research/grounding-dino-base` (snapshot `12bdfa31…`): OK, 15 s.
   - `google/owlv2-large-patch14-ensemble` (snapshot `95e26936…`): OK, 51 s.
   - Hanterdsith `best.pt` (revision `0c61ba14…`): downloaded but **not loaded**; that decision belongs to the 4P ticket.
5. **GPU budget**:
   - No SLURM limit on account `a260012`.
   - Since 2026-09-01: 306 GPU jobs, about **24 GPU-hours** in total (0.1 h per job) for Experiments 1–3 and CV.
   - Retraining for all three arms (≈72 scenarios) should therefore need on the order of tens of GPU-hours.
   - Any allowance outside SLURM is unconfirmed (admin: nattawet.sri@cmu.ac.th).

## Comments

**2026-10-16, first run of `check_raptor.sh`:**
- **Container**:
  - has torch `2.6.0a0+df5bbc09d1.nv24.11`, torchvision `0.20.0a0`, **pycocotools** and OpenCV 4.10;
  - has **no** transformers, torchmetrics or ultralytics.
- **Compute nodes reach the internet** (HTTP 200 from huggingface.co on raptor-m-003).
- **Package installs**:
  - Venvs on `/project` can be created.
  - `pip install` failed with `Could not find a suitable TLS CA certificate bundle, invalid path: /etc/ssl/certs/ca-bundle.crt`: the host's Red Hat CA path is passed into the Ubuntu container.
  - The script now sets `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` / `PIP_CERT` to `/etc/ssl/certs/ca-certificates.crt`. Rerun pending.
- **GPU**:
  - No SLURM TRES limit is set on account `a260012`.
  - Since 2026-09-01 there were 306 GPU jobs totalling about 24 GPU-hours (mean 0.1 h per job); this covers Experiments 1–3 and CV.
  - The actual allowance is still to be confirmed with the admin.
