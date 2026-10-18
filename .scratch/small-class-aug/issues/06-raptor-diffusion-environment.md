# Raptor environment for diffusion training

Type: task (HITL)
Status: open
Blocked by: 01

## Question

On Raptor:
1. Create a `venv_exp5` with the pinned `diffusers` / `peft` / `accelerate` versions.
2. Download the chosen base model into `hf_cache`.
3. Run a 10-step LoRA training and generate one image on a GPU node, offline.
4. Record the time and memory used.

The researcher runs a checklist script, as for T19.
