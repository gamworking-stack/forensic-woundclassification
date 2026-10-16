#!/bin/bash
# Experiment 4 environment check (ticket "Raptor environment and GPU budget for Experiment 4").
# Run on the Raptor LOGIN node:  bash ~/check_raptor.sh 2>&1 | tee ~/exp4_check.txt
# Uses only a synthetic image; no patient photo is read. Installs go into a venv on /project,
# never into ~/.local (which the container would pick up in every existing run).
set -u
module load apptainer/1.3.5
SIF=/common/sif/pytorch-24.11-py3.sif
P=/project/a260012_FORWCL
VENV=$P/venv_exp4            # transformers + pycocotools
VENV_YOLO=$P/venv_exp4_yolo  # ultralytics, kept apart in case it changes numpy/opencv
export HF_HOME=$P/hf_cache
# The host (Red Hat) passes its CA path /etc/ssl/certs/ca-bundle.crt into the Ubuntu container,
# where it does not exist; point pip, requests and huggingface_hub at the container's own bundle.
CA=/etc/ssl/certs/ca-certificates.crt
CA_ENV="SSL_CERT_FILE=$CA,REQUESTS_CA_BUNDLE=$CA,CURL_CA_BUNDLE=$CA,PIP_CERT=$CA"
run() { apptainer exec --bind "$P" --env HF_HOME="$HF_HOME,$CA_ENV" "$SIF" "$@"; }
echo "-- host CA variables: $(env | grep -E '^(SSL_CERT_FILE|REQUESTS_CA_BUNDLE|CURL_CA_BUNDLE|PIP_CERT)=' | tr '\n' ' ')"
run test -f "$CA" && echo "-- container CA bundle $CA: present" || echo "-- container CA bundle $CA: MISSING"

echo "================ 1. Libraries already in the container"
run python -c "import torch, torchvision; print('torch', torch.__version__, '| torchvision', torchvision.__version__)"
for m in transformers pycocotools torchmetrics ultralytics cv2; do
  run python -c "import $m; print('$m', getattr($m, '__version__', 'installed'))" 2>/dev/null || echo "$m: NOT installed"
done

echo "================ 3. Can extra packages be installed? (venv on /project)"
if run python -m venv --system-site-packages "$VENV"; then
  run "$VENV/bin/pip" install -q --no-warn-script-location "transformers==4.57.6" pycocotools \
    && echo "venv_exp4 install: OK" || echo "venv_exp4 install: FAILED"
  run "$VENV/bin/python" -c "import transformers; from transformers.utils import is_torch_available; print('transformers', transformers.__version__, '| sees torch:', is_torch_available())"
else
  echo "python -m venv: FAILED inside the container"
fi
if run python -m venv --system-site-packages "$VENV_YOLO"; then
  run "$VENV_YOLO/bin/pip" install -q ultralytics \
    && echo "venv_exp4_yolo install: OK" || echo "venv_exp4_yolo install: FAILED"
  run "$VENV_YOLO/bin/python" -c "import ultralytics, torch, numpy; print('ultralytics', ultralytics.__version__, '| torch', torch.__version__, '| numpy', numpy.__version__)"
fi

echo "================ 4a. Download weights on the login node (to $HF_HOME)"
run "$VENV/bin/python" - <<'EOF'
from huggingface_hub import hf_hub_download, snapshot_download
for repo in ("IDEA-Research/grounding-dino-base", "google/owlv2-large-patch14-ensemble"):
    try:
        path = snapshot_download(repo, allow_patterns=["*.json", "*.txt", "*.safetensors"])
        print("downloaded", repo, "->", path)
    except Exception as e:
        print("download FAILED", repo, type(e).__name__, e)
try:  # downloaded only, NOT loaded: an untrusted pickle; loading it is decided in the 4P ticket
    path = hf_hub_download("trapezius60/forensic_wound_detection", "best.pt",
                           revision="0c61ba140e2c800b25e562e092b777082c7a10fe")
    print("downloaded Hanterdsith best.pt (not loaded) ->", path)
except Exception as e:
    print("download FAILED best.pt", type(e).__name__, e)
EOF

cat > "$P/exp4_check_load.py" <<'EOF'
import time
import torch
from PIL import Image
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, Owlv2ForObjectDetection,
                          Owlv2Processor)

image = Image.new("RGB", (800, 600), (200, 150, 140))  # synthetic, no patient data
try:
    t = time.time()
    name = "IDEA-Research/grounding-dino-base"
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(name, use_safetensors=True).eval()
    with torch.no_grad():
        model(**proc(images=image, text="wound.", return_tensors="pt"))
    print(f"Grounding DINO offline load + 1 image on CPU: OK ({time.time() - t:.0f} s)")
except Exception as e:
    print("Grounding DINO: FAILED", type(e).__name__, e)
try:
    t = time.time()
    name = "google/owlv2-large-patch14-ensemble"
    proc = Owlv2Processor.from_pretrained(name)
    model = Owlv2ForObjectDetection.from_pretrained(name, use_safetensors=True).eval()
    with torch.no_grad():
        model(**proc(text=[["a photo of a wound"]], images=image, return_tensors="pt"))
    print(f"OWLv2 offline load + 1 image on CPU: OK ({time.time() - t:.0f} s)")
except Exception as e:
    print("OWLv2: FAILED", type(e).__name__, e)
EOF

echo "================ 2 + 4b. On a compute node: internet? do the models load offline?"
srun -p compute -c 4 --mem=16G -t 0:30:00 bash -c "
  curl -sS -m 10 -o /dev/null -w 'compute-node internet: HTTP %{http_code}\n' https://huggingface.co 2>/dev/null \
    || echo 'compute-node internet: NO'
  apptainer exec --bind $P --env HF_HOME=$HF_HOME,HF_HUB_OFFLINE=1,$CA_ENV $SIF $VENV/bin/python $P/exp4_check_load.py
"

echo "================ 5. GPU budget"
echo "-- limits on the account (empty = no limit set in SLURM; ask the admin)"
sacctmgr -n show assoc user="$USER" format=Account%22,Partition%12,GrpTRESMins%30,MaxTRESMins%30
echo "-- GPU use since 1 Sep 2026 (Experiments 1-3 and CV)"
sacct -u "$USER" -S 2026-09-01 -X -n -P --format=ElapsedRaw,AllocTRES | awk -F'|' '
  match($2, /gres\/gpu=[0-9]+/) { g = substr($2, RSTART + 9, RLENGTH - 9); s += $1 * g; n++ }
  END { printf "GPU jobs: %d | GPU-hours: %.0f | mean per job: %.1f h\n", n, s / 3600, (n ? s / 3600 / n : 0) }'
echo "================ done"
