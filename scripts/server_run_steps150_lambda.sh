#!/bin/bash
# lambda probe for the 150-step runs, `exact` variant at N = 100.
#
# Every 1024-sample run so far inherits lambda_ = 5000, which was selected on the
# 32-step `aggregate_x0` grid (Table 3) and never retuned for `exact`/`marginal`
# or for a different step count. On that grid novel QED rises monotonically in
# lambda, so lowering it costs ~0.02 at 32 steps.
#
# The reason to expect a different answer at 150 steps: composing exact tilted
# reverse steps targets the same global tilt regardless of step count, so in
# theory lambda need not be rescaled. But at finite N each step's self-normalized
# importance estimate is biased toward over-concentration, and that bias
# accumulates once per step -- 150 steps accumulate it 4.7x as often, so the same
# lambda acts as a stronger tilt.
#
# N = 100 with lambda = 5000 is computed by scripts/server_run_steps150.sh, so
# these three runs complete a four-point lambda curve at matched N and steps.
#
# Runs on GPU 1 so it does not disturb that sweep on GPU 0. Reward workers are
# halved for the same reason; RDKit misses are sparse enough that this is not the
# bottleneck (load average was ~4 of 40 cores with the main sweep running).
#
# Usage (inside tmux, survives disconnects):
#   bash scripts/server_run_steps150_lambda.sh
set -uo pipefail

PROP=qed
BATCH_SIZE=64
BATCHES=16          # 1024 samples
STEPS=150
SEED=1
WORKERS=16

ROOT=/home/aailab/wp03052/Synthetic-Data/DLRT

source /home/aailab/wp03052/venvs/dlrt_env/bin/activate
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES=1
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

OURS_ARGS=(
  guidance=ours "guidance.reward=${PROP}"
  guidance.mixture_sampling=exact
  "guidance.num_reward_workers=${WORKERS}"
)

echo "######## lambda curve: exact, N=100, 150 steps ########"
for LAM in 100 500 1000; do
  run_eval "s1024_st150_ours_exact_N100_lam${LAM}_win0.0-1.0" "${OURS_ARGS[@]}" \
    guidance.num_x0_samples=100 "guidance.lambda_=${LAM}" \
    guidance.t_min=0.0 guidance.t_max=1.0
done

echo "######## Done ########"
ls -1 "${ROOT}"/results/s1024_st150_ours_exact_N100_lam*.csv
