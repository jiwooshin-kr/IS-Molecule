#!/bin/bash
# Reruns the Table 4 (head-to-head) "Ours" configurations with
# guidance.mixture_sampling=marginal instead of aggregate_x0. The marginal
# variant averages the N *normalized* posteriors q(.|x_t, x_0^(n)) before
# sampling, which differs from aggregate_x0 under uniform diffusion because
# the posterior normalizer depends on x_0.
#
# Same budget as server_run_comparison.sh: 1024 samples, seed 1, 32 steps,
# same HuggingFace UDLM-QM9 base model.
#
# Usage (inside tmux, survives disconnects):
#   bash scripts/ours/qed/server_run_marginal.sh
set -uo pipefail

PROP=qed
BATCH_SIZE=64
BATCHES=16          # 1024 samples, matching Table 4
STEPS=32
SEED=1
WORKERS=32

ROOT=/home/aailab/wp03052/Synthetic-Data/Molecule

source /home/aailab/wp03052/venvs/dlrt_env/bin/activate
cd "${ROOT}"

export HF_HOME="${ROOT}/.hf_cache"
export PYTHONPATH="${ROOT}:${ROOT}/guidance_eval:${HF_HOME}/modules"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export WANDB_DIR="${ROOT}"
mkdir -p "${ROOT}/results/qed"

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
  local csv="${ROOT}/results/qed/${tag}.csv"
  if [ -s "${csv}" ]; then
    echo "[skip] ${tag}"
    return 0
  fi
  echo "=============== ${tag} ==============="
  local log="${ROOT}/results/qed/${tag}.log"
  local start=${SECONDS}
  if python -u -W ignore guidance_eval/our_qm9_eval.py \
      "${BASE_ARGS[@]}" "$@" \
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

OURS_ARGS=(
  guidance=ours "guidance.reward=${PROP}"
  guidance.mixture_sampling=marginal
  "guidance.num_reward_workers=${WORKERS}"
)

# The four "Ours" rows of Table 4 (tab:headline), marginal variant.
run_eval "s1024_ours_marginal_N100_lam5000_win0.0-1.0" "${OURS_ARGS[@]}" \
  guidance.num_x0_samples=100 guidance.lambda_=5000 \
  guidance.t_min=0.0 guidance.t_max=1.0

run_eval "s1024_ours_marginal_N300_lam5000_win0.0-1.0" "${OURS_ARGS[@]}" \
  guidance.num_x0_samples=300 guidance.lambda_=5000 \
  guidance.t_min=0.0 guidance.t_max=1.0

run_eval "s1024_ours_marginal_N500_lam5000_win0.0-1.0" "${OURS_ARGS[@]}" \
  guidance.num_x0_samples=500 guidance.lambda_=5000 \
  guidance.t_min=0.0 guidance.t_max=1.0

run_eval "s1024_ours_marginal_N500_lam1000_win0.0-0.75" "${OURS_ARGS[@]}" \
  guidance.num_x0_samples=500 guidance.lambda_=1000 \
  guidance.t_min=0.0 guidance.t_max=0.75

echo "######## Done ########"
ls -1 "${ROOT}"/results/qed/s1024_ours_marginal_*.csv
