#!/bin/bash
# MDLM-QM9 unguided validity as a function of the sampling step budget T.
#
# Why this run exists
# -------------------
# MDLM-QM9 at T=32 gives 52.7 % validity (results/mdlm_unguided.csv) against
# 97.9 % for UDLM-QM9 at the same budget (results/s1024_unguided.csv). Before
# building anything on top of the MDLM base model we need to know which of two
# things that is:
#
#   (a) a weak checkpoint  -> validity stays low no matter how large T gets,
#       and the fix is more training;
#   (b) a step-budget artifact -> validity climbs towards ~0.95 as T grows,
#       the checkpoint is fine, and the T=32 number is a *sampler* deficiency.
#
# (b) is the interesting case: MDLM freezes a position once it is unmasked
# (`copy_flag`), so at T = L = 32 roughly 26 % of steps unmask two or more
# positions at once and write them conditionally independently given x_t. The
# true joint is not that product, and that error is exactly what a
# confidence-ordered decoder is meant to remove. So this curve is the baseline
# the reward-tilted confidence decoding work has to improve on.
#
# Evidence already in hand that the checkpoint is converged: `best.ckpt` was
# last written at 12:41 while step 15000 landed at 12:00 and step 20000 at
# 13:22, so validation loss stopped improving around step ~17.5k and the run
# continued to 25k without beating it. The eval below uses `best.ckpt` with EMA
# enabled, i.e. the best checkpoint, not the last one.
#
# All rows are unguided: `guidance.t_min=2.0 guidance.t_max=2.0` puts every t
# outside the guidance window, so `_our_denoise` falls through to the base
# sampler and skips the reward calls entirely (our_guidance.py:297-308). That
# makes each row a pure `Diffusion._ddpm_denoise` trajectory.
#
# Usage (inside tmux, survives disconnects):
#   bash scripts/ours/server_run_mdlm_tsweep.sh
set -uo pipefail

PROP=qed
BATCH_SIZE=64
BATCHES=16          # 1024 samples, matching every other table
SEED=1

ROOT=/home/aailab/wp03052/Synthetic-Data/Molecule
MODEL_CKPT="${ROOT}/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt"

source /home/aailab/wp03052/venvs/dlrt_env/bin/activate
cd "${ROOT}"
export HF_HOME="${ROOT}/.hf_cache"
export PYTHONPATH="${ROOT}:${ROOT}/guidance_eval:${HF_HOME}/modules"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export WANDB_DIR="${ROOT}"
mkdir -p "${ROOT}/results"

if [ ! -s "${MODEL_CKPT}" ]; then
  echo "FATAL: no checkpoint at ${MODEL_CKPT}"
  exit 1
fi

# Mirrors scripts/ours/server_run_comparison_mdlm.sh exactly except for
# `sampling.steps`, which is swept. `training.guidance=null` is required: the
# DiT builds a class-conditioning embedding whenever it is non-null
# (models/dit.py:381) and the MDLM checkpoint has no `cond_map` weights.
base_args() {
  local steps="$1"
  echo data=qm9 "data.label_col=${PROP}" "data.cache_dir=${ROOT}/.data_cache" \
    diffusion=absorbing_state parameterization=subs T=0 \
    time_conditioning=False zero_recon_loss=False \
    training.guidance=null \
    backbone=dit model=small model.length=32 \
    "eval.checkpoint_path=${MODEL_CKPT}" \
    "sampling.steps=${steps}" "sampling.batch_size=${BATCH_SIZE}" \
    "sampling.num_sample_batches=${BATCHES}" sampling.use_cache=False \
    eval.disable_ema=False "seed=${SEED}"
}

# Unguided: every t falls outside [t_min, t_max], so no reward is ever called.
UNGUIDED_ARGS=(
  guidance=ours "guidance.reward=${PROP}"
  guidance.mixture_sampling=marginal
  guidance.num_x0_samples=2 guidance.lambda_=0.0
  guidance.t_min=2.0 guidance.t_max=2.0
  guidance.num_reward_workers=1
)

run_eval() {
  local tag="$1"; local steps="$2"
  local csv="${ROOT}/results/${tag}.csv"
  if [ -s "${csv}" ]; then echo "[skip] ${tag}"; return 0; fi
  echo "=============== ${tag} (T=${steps}) ==============="
  local log="${ROOT}/results/${tag}.log"
  local start=${SECONDS}
  if python -u -W ignore guidance_eval/our_qm9_eval.py \
      $(base_args "${steps}") "${UNGUIDED_ARGS[@]}" \
      "++eval.results_csv_path=${csv}" \
      "++eval.generated_samples_path=${ROOT}/results/${tag}_samples.json" \
      > "${log}" 2>&1; then
    grep -E "^\s+Valid|Mean:" "${log}"
    echo "  $((SECONDS - start))s"
  else
    echo "  FAILED -- see ${log}"; tail -20 "${log}"
  fi
}

# L = 32, so T=32 is one write per step on average and T=8/16 force
# multi-position writes. T=64..256 drive P(two or more at once) towards zero.
for T in 8 16 32 64 128 256; do
  run_eval "mdlm_unguided_T${T}" "${T}"
done

echo "######## Done ########"
ls -1 "${ROOT}"/results/mdlm_unguided_T*.csv 2>/dev/null
