#!/bin/bash
# Multi-seed comparison of the three mixture-sampling variants at small N.
#
# Everything in Tables 4 and 5 is a single seed, and the gaps between variants
# (+0.007 to +0.008 novel QED) sit near the per-cell standard error of ~0.005, so
# no individual cell separates them. This sweep replaces that with 5 seeds.
#
# Small N is the regime where the variants should differ most, and where the
# prediction has a sign. `marginal` is the Rao-Blackwellization of `exact`'s
# component draw -- it is the conditional expectation of exact's kernel given the
# candidate set -- so its per-position marginals have strictly lower estimation
# variance, paid for by losing the joint. At N = 5 or 10, `exact` commits to one
# of very few candidates and that variance is largest, so if the variance
# argument holds anywhere it holds here: `marginal` should beat `exact` at N = 5
# and the gap should narrow as N grows. At N = 300--500 (Table 4) `exact` wins,
# so the prediction is a crossover, not a uniform ordering.
#
# 32 steps and lambda = 5000, matching the Table 4 default, so these cells are
# directly comparable to its N = 100--500 rows.
#
# Usage (inside tmux). Takes N and a GPU index so the two N values can run
# concurrently on separate GPUs:
#   bash scripts/ours/qed/server_run_seeds_smallN.sh 5 0
#   bash scripts/ours/qed/server_run_seeds_smallN.sh 10 1
set -uo pipefail

if [ $# -ne 2 ]; then
  echo "usage: $0 <num_x0_samples> <gpu_index>" >&2
  exit 2
fi
N="$1"
GPU="$2"

PROP=qed
BATCH_SIZE=64
BATCHES=16          # 1024 samples, matching Tables 4 and 5
STEPS=32
LAM=5000
SEEDS="1 2 3 4 5"
WORKERS=16

ROOT=/home/aailab/wp03052/Synthetic-Data/Molecule

source /home/aailab/wp03052/venvs/dlrt_env/bin/activate
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HOME="${ROOT}/.hf_cache"
export PYTHONPATH="${ROOT}:${ROOT}/guidance_eval:${HF_HOME}/modules"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export WANDB_DIR="${ROOT}"
mkdir -p "${ROOT}/results/qed"

run_eval() {
  local tag="$1"; shift
  local csv="${ROOT}/results/qed/${tag}.csv"
  if [ -s "${csv}" ]; then
    echo "[skip] ${tag}"
    return 0
  fi
  echo "=============== ${tag} ==============="
  local log="${ROOT}/results/qed/${tag}.log"
  local start=${SECONDS}
  if python -u -W ignore guidance_eval/our_qm9_eval.py \
      data=qm9 "data.label_col=${PROP}" "data.cache_dir=${ROOT}/.data_cache" \
      backbone=hf_dit model=hf model.length=32 \
      model.pretrained_model_name_or_path=kuleshov-group/udlm-qm9 \
      diffusion=uniform parameterization=d3pm \
      time_conditioning=True zero_recon_loss=True \
      "sampling.steps=${STEPS}" "sampling.batch_size=${BATCH_SIZE}" \
      "sampling.num_sample_batches=${BATCHES}" sampling.use_cache=False \
      eval.disable_ema=True \
      "$@" \
      "++eval.results_csv_path=${csv}" \
      "++eval.generated_samples_path=${ROOT}/results/qed/${tag}_samples.json" \
      > "${log}" 2>&1; then
    grep -E "^\s+Valid|Mean:" "${log}"
    echo "  last ESS: $(grep -o 'ESS=[0-9.]*' "${log}" | tail -1)" \
         "| $((SECONDS - start))s"
  else
    echo "  FAILED -- see ${log}"
    tail -4 "${log}"
  fi
}

echo "######## N = ${N}, lambda = ${LAM}, ${STEPS} steps, seeds ${SEEDS} ########"
for MODE in edlm marginal; do
  for SEED in ${SEEDS}; do
    run_eval "s1024_seed${SEED}_ours_${MODE}_N${N}_lam${LAM}_win0.0-1.0" \
      "seed=${SEED}" \
      guidance=ours "guidance.reward=${PROP}" \
      "guidance.mixture_sampling=${MODE}" \
      "guidance.num_reward_workers=${WORKERS}" \
      "guidance.num_x0_samples=${N}" "guidance.lambda_=${LAM}" \
      guidance.t_min=0.0 guidance.t_max=1.0
  done
done

echo "######## Done (N = ${N}) ########"
ls -1 "${ROOT}"/results/s1024_seed*_ours_*_N${N}_lam${LAM}_win0.0-1.0.csv
