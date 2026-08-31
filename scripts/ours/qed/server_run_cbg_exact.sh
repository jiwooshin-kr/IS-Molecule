#!/bin/bash
# Re-runs the exact (non-approximated) D-CBG baseline at a smaller sampling
# batch size. D-CBG without `use_approx` evaluates the classifier on every
# vocabulary substitution, which needs ~5 GiB more than a 3090 has free at
# batch_size=64; the total sample count is kept at 1024.
#
# Usage (inside tmux):
#   bash scripts/ours/qed/server_run_cbg_exact.sh
set -uo pipefail

PROP=qed
BATCH_SIZE=8
BATCHES=128         # 8 * 128 = 1024 samples
ROOT=/home/aailab/wp03052/Synthetic-Data/Molecule
CLASS_CKPT="${ROOT}/outputs/qm9/classifier/${PROP}_uniform_T-0/checkpoints/best.ckpt"

source /home/aailab/wp03052/venvs/dlrt_env/bin/activate
cd "${ROOT}"
export HF_HOME="${ROOT}/.hf_cache"
export PYTHONPATH="${ROOT}:${ROOT}/guidance_eval:${HF_HOME}/modules"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export WANDB_DIR="${ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for GAMMA in 1 2 5; do
  TAG="s1024_cbg_exact_gamma${GAMMA}"
  CSV="${ROOT}/results/qed/${TAG}.csv"
  if [ -s "${CSV}" ]; then echo "[skip] ${TAG}"; continue; fi
  echo "=============== ${TAG} ==============="
  if python -u -W ignore guidance_eval/our_qm9_eval.py \
      data=qm9 "data.label_col=${PROP}" "data.cache_dir=${ROOT}/.data_cache" \
      backbone=hf_dit model=hf model.length=32 \
      model.pretrained_model_name_or_path=kuleshov-group/udlm-qm9 \
      diffusion=uniform parameterization=d3pm \
      time_conditioning=True zero_recon_loss=True \
      sampling.steps=32 "sampling.batch_size=${BATCH_SIZE}" \
      "sampling.num_sample_batches=${BATCHES}" sampling.use_cache=False \
      eval.disable_ema=True seed=1 \
      guidance=cbg guidance.condition=1 \
      classifier_model=tiny-classifier classifier_backbone=dit \
      "guidance.classifier_checkpoint_path=${CLASS_CKPT}" \
      "guidance.gamma=${GAMMA}" guidance.use_approx=False \
      "++eval.results_csv_path=${CSV}" \
      "++eval.generated_samples_path=${ROOT}/results/qed/${TAG}_samples.json" \
      > "${ROOT}/results/qed/${TAG}.log" 2>&1; then
    grep -E "^\s+Valid|Mean:" "${ROOT}/results/qed/${TAG}.log"
  else
    echo "  FAILED"; tail -4 "${ROOT}/results/qed/${TAG}.log"
  fi
done
echo "######## Done ########"
