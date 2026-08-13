#!/bin/bash
# Table 5: the mixture-sampling comparison at 150 denoising steps instead of 32.
#
# Table 4 fixes sampling.steps=32, which is the paper's setting. Two things make
# the step count worth its own table. The reward-tilted posterior commits to (or
# averages over) a fresh set of N candidates at every step, so the number of
# steps sets how many times the tilt is applied along a trajectory -- 150 steps
# apply it 4.7x as often at the same N. And for `exact`, each step commits to one
# sampled x_0, so more steps means more commitments but a smaller move per
# commitment; whether that helps or hurts is not predictable from the 32-step
# runs.
#
# Scope: `exact` and `marginal` only. `aggregate_x0` and D-CBG are excluded --
# D-CBG's exact branch costs ~55 min per config at 32 steps and would be ~4.3 h
# at 150. The unguided row is included as the same-step-count reference, without
# which a change in novel QED cannot be attributed to the step count rather than
# the variant.
#
# N is extended down to 10 and 50, which Table 4 does not cover: at 32 steps
# small N was dropped because the candidate histogram is an atomic measure, but
# `exact` and `marginal` are affected differently and more steps may compensate
# for fewer candidates per step.
#
# T = 0 (continuous time, configs/config.yaml), so there is no t-grid snapping
# and 150 steps is well posed.
#
# Usage (inside tmux, survives disconnects):
#   bash scripts/server_run_steps150.sh
set -uo pipefail

PROP=qed
BATCH_SIZE=64
BATCHES=16          # 1024 samples, matching Table 4
STEPS=150
SEED=1
WORKERS=32

ROOT=/home/aailab/wp03052/Synthetic-Data/DLRT

source /home/aailab/wp03052/venvs/dlrt_env/bin/activate
cd "${ROOT}"

export HF_HOME="${ROOT}/.hf_cache"
export PYTHONPATH="${ROOT}:${ROOT}/guidance_eval:${HF_HOME}/modules"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export WANDB_DIR="${ROOT}"
mkdir -p "${ROOT}/results"

BASE_ARGS=(
  data=qm9 "data.label_col=${PROP}" "data.cache_dir=${ROOT}/.data_cache"
  backbone=hf_dit model=hf model.length=32
  model.pretrained_model_name_or_path=kuleshov-group/udlm-qm9
  diffusion=uniform parameterization=d3pm
  time_conditioning=True zero_recon_loss=True
  "sampling.steps=${STEPS}" "sampling.batch_size=${BATCH_SIZE}"
  "sampling.num_sample_batches=${BATCHES}" sampling.use_cache=False
  eval.disable_ema=True "seed=${SEED}"
)

run_eval() {
  local tag="$1"; shift
  local csv="${ROOT}/results/${tag}.csv"
  if [ -s "${csv}" ]; then
    echo "[skip] ${tag}"
    return 0
  fi
  echo "=============== ${tag} ==============="
  local log="${ROOT}/results/${tag}.log"
  local start=${SECONDS}
  if python -u -W ignore guidance_eval/our_qm9_eval.py \
      "${BASE_ARGS[@]}" "$@" \
      "++eval.results_csv_path=${csv}" \
      "++eval.generated_samples_path=${ROOT}/results/${tag}_samples.json" \
      > "${log}" 2>&1; then
    grep -E "^\s+Valid|Mean:" "${log}"
    echo "  last ESS: $(grep -o 'ESS=[0-9.]*' "${log}" | tail -1)" \
         "| $((SECONDS - start))s"
  else
    echo "  FAILED -- see ${log}"
    tail -4 "${log}"
  fi
}

echo "######## 150-step unguided reference ########"
# Unsatisfiable window, so the guided branch never fires: the base UDLM sampler
# at 150 steps, run through the same code path as the rows below.
run_eval "s1024_st150_unguided" \
  guidance=ours "guidance.reward=${PROP}" \
  guidance.mixture_sampling=aggregate_x0 \
  "guidance.num_reward_workers=${WORKERS}" \
  guidance.num_x0_samples=2 guidance.lambda_=0.0 \
  guidance.t_min=2.0 guidance.t_max=2.0

for MODE in exact marginal; do
  echo "######## ${MODE}: N sweep at lambda = 5000, guidance over all t ########"
  MODE_ARGS=(
    guidance=ours "guidance.reward=${PROP}"
    "guidance.mixture_sampling=${MODE}"
    "guidance.num_reward_workers=${WORKERS}"
  )
  for N in 10 50 100 300 500; do
    run_eval "s1024_st150_ours_${MODE}_N${N}_lam5000_win0.0-1.0" \
      "${MODE_ARGS[@]}" \
      "guidance.num_x0_samples=${N}" guidance.lambda_=5000 \
      guidance.t_min=0.0 guidance.t_max=1.0
  done

  echo "######## ${MODE}: best 32-step window setting ########"
  run_eval "s1024_st150_ours_${MODE}_N500_lam1000_win0.0-0.75" \
    "${MODE_ARGS[@]}" \
    guidance.num_x0_samples=500 guidance.lambda_=1000 \
    guidance.t_min=0.0 guidance.t_max=0.75
done

echo "######## Done ########"
ls -1 "${ROOT}"/results/s1024_st150_*.csv
