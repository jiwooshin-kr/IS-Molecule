#!/bin/bash
# Trains the D-CBG noisy classifier baseline on our server (venv, no slurm).
# Mirrors scripts/train_qm9_classifier.sh, which assumes conda + sbatch.
#
# Usage (run inside tmux so it survives disconnects):
#   bash scripts/server_train_cbg_classifier.sh <uniform|absorbing_state> <qed|ring_count>
#
# `uniform` matches the UDLM base model (kuleshov-group/udlm-qm9) that our
# method is evaluated against, so use that for an apples-to-apples comparison.
# Writes to outputs/qm9/classifier/<PROP>_<DIFFUSION>_T-0, which is the path
# guidance.classifier_checkpoint_path is expected to point into.
set -euo pipefail

DIFFUSION="${1:-uniform}"
PROP="${2:-qed}"
T=0
RUN_NAME="${PROP}_${DIFFUSION}_T-${T}"

source /home/aailab/wp03052/venvs/dlrt_env/bin/activate
cd /home/aailab/wp03052/Synthetic-Data/DLRT

export HF_HOME="${PWD}/.hf_cache"
export PYTHONPATH="${PWD}:${PWD}/guidance_eval:${HF_HOME}/modules"
export HYDRA_FULL_ERROR=1
export NCCL_P2P_LEVEL=NVL
# No wandb account is configured on this box; keep the logger code path but
# write runs to ./wandb instead of the network.
export WANDB_MODE=offline
export WANDB_DIR="${PWD}"
# The repo defaults num_workers to the CPU count (40 here). Forked dataloader
# workers hit "CUDA error: initialization error" and take the run down during
# sanity checking, so data loading stays on the main process (QM9 is small and
# pre-tokenized, so this is not a bottleneck).

python -u -m main \
  mode=train_classifier \
  diffusion="${DIFFUSION}" \
  T=${T} \
  data=qm9 \
  data.cache_dir="${PWD}/.data_cache" \
  data.label_col="${PROP}" \
  data.label_col_pctile=90 \
  data.num_classes=2 \
  loader.global_batch_size=2048 \
  loader.eval_global_batch_size=4096 \
  loader.num_workers=0 \
  loader.persistent_workers=False \
  classifier_backbone=dit \
  classifier_model=tiny-classifier \
  model.length=32 \
  optim.lr=3e-4 \
  lr_scheduler=cosine_decay_warmup \
  lr_scheduler.warmup_t=1000 \
  lr_scheduler.lr_min=3e-6 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
  callbacks.checkpoint_monitor.monitor=val/cross_entropy \
  trainer.val_check_interval=1.0 \
  trainer.max_steps=25_000 \
  wandb.group=train_classifier \
  wandb.name="qm9-classifier_${RUN_NAME}" \
  hydra.run.dir="${PWD}/outputs/qm9/classifier/${RUN_NAME}"
