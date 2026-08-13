#!/bin/bash
# Trains an MDLM generative model on QM9 (absorbing-state diffusion).
# Mirrors scripts/train_qm9_no-guidance.sh with MODEL=mdlm, adapted for our
# venv + no slurm. Needed because HuggingFace has no MDLM-QM9 checkpoint: the
# published MDLM releases are all OpenWebText (context 1024, GPT-2 tokenizer),
# incompatible with QM9's 32-token SMILES vocabulary.
#
# Usage (inside tmux, survives disconnects):
#   CUDA_VISIBLE_DEVICES=1,2 bash scripts/server_train_mdlm_qm9.sh
#
# Writes to outputs/qm9/mdlm_no-guidance, which is the path
# guidance_eval scripts expect for the MDLM base model.
set -euo pipefail

ROOT=/home/aailab/wp03052/Synthetic-Data/DLRT
RUN_NAME=mdlm_no-guidance

source /home/aailab/wp03052/venvs/dlrt_env/bin/activate
cd "${ROOT}"

export HF_HOME="${ROOT}/.hf_cache"
export PYTHONPATH="${ROOT}:${ROOT}/guidance_eval:${HF_HOME}/modules"
export HYDRA_FULL_ERROR=1
export NCCL_P2P_LEVEL=NVL
export WANDB_MODE=offline
export WANDB_DIR="${ROOT}"

python -u -m main \
  diffusion=absorbing_state \
  parameterization=subs \
  T=0 \
  time_conditioning=False \
  zero_recon_loss=False \
  data=qm9 \
  "data.cache_dir=${ROOT}/.data_cache" \
  data.label_col=null \
  data.label_col_pctile=null \
  data.num_classes=null \
  eval.generate_samples=False \
  loader.global_batch_size=2048 \
  loader.eval_global_batch_size=4096 \
  loader.num_workers=0 \
  loader.persistent_workers=False \
  backbone=dit \
  model=small \
  model.length=32 \
  optim.lr=3e-4 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=1000 \
  lr_scheduler.lr_min=3e-6 \
  training.guidance=null \
  training.compute_loss_on_pad_tokens=True \
  training.use_simple_ce_loss=False \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
  trainer.max_steps=25_000 \
  trainer.val_check_interval=1.0 \
  wandb.group=train_generative \
  wandb.name="qm9_${RUN_NAME}" \
  "hydra.run.dir=${ROOT}/outputs/qm9/${RUN_NAME}"
