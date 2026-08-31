#!/bin/bash
# Reruns the Table 4 (head-to-head) "Ours" configurations with
# guidance.mixture_sampling=edlm, the alternative to our `marginal` method.
#
# `edlm` is EDLM's Algorithm 1 (Denoising via Importance Sampling) with the
# learned energy E_phi(x_0, x_t) replaced by our reward -lambda*r(x_0): draw
# n* ~ Cat(w), then sample q(.|x_t, x_0^(n*)). It samples the *joint* mixture of
# Eq. (8) exactly, so it keeps the cross-position correlations that `marginal`
# discards; `marginal` instead reproduces the mixture's per-position marginals
# exactly and samples positions independently. Neither dominates on fidelity --
# they target the same marginals and differ only in the joint.
#
# EDLM runs its algorithm on masked (absorbing-state) diffusion; the runs here
# are uniform diffusion, so these rows double as a check of whether that
# substitution transfers.
#
# Table 1 rejected this mode at N=10, where the tilt is too weak to pay for
# committing to a single sampled x_0; these runs test N=100..500.
#
# NOTE: the tags below still say `exact`, the old name for this mode, because
# the CSVs they skip-check against were produced under that name. The retired
# third mode `aggregate_x0` was deleted, not renamed: it substituted the
# weighted histogram xbar_0 into the *unnormalized* posterior and divided once
# at the end, which for uniform diffusion is a ratio of averages instead of an
# average of ratios and is biased.
#
# Same budget as server_run_comparison.sh: 1024 samples, seed 1, 32 steps, same
# HuggingFace UDLM-QM9 base model.
#
# Usage (inside tmux, survives disconnects):
#   bash scripts/ours/qed/server_run_exact.sh
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
  guidance.mixture_sampling=edlm
  "guidance.num_reward_workers=${WORKERS}"
)

echo "######## lambda = 0 control ########"
# Separates the two things `exact` changes at once: the reward tilt, and
# committing each step to a single sampled x_0.
#
# At lambda_ = 0 the weights are exactly uniform, so drawing n* ~ Cat(w) and
# sampling q(.|x_t, x_0^(n*)) is identical to drawing one x_0 ~ p_theta and
# sampling its posterior -- i.e. exact ancestral sampling of the *true*
# untilted reverse step, with no Monte Carlo error and no dependence on N. The
# base sampler instead plugs the denoiser marginal x_theta into the posterior
# and samples positions independently, which is the factorized projection of
# that same step. So this row measures the projection, not the guidance.
#
# Table 2 saw 61.7 % validity here against 96.7 % for the base sampler at
# N = 10 on 128 samples. If that reproduces at 1024 samples, the validity cost
# of `exact` is inherent to sampling the true reverse step and not something
# larger N or larger lambda can buy back. N = 100 is used only so the row lines
# up with the first tilted row; the result cannot depend on it.
run_eval "s1024_ours_exact_N100_lam0_win0.0-1.0" "${OURS_ARGS[@]}" \
  guidance.num_x0_samples=100 guidance.lambda_=0.0 \
  guidance.t_min=0.0 guidance.t_max=1.0

echo "######## The four Table 4 rows ########"
run_eval "s1024_ours_exact_N100_lam5000_win0.0-1.0" "${OURS_ARGS[@]}" \
  guidance.num_x0_samples=100 guidance.lambda_=5000 \
  guidance.t_min=0.0 guidance.t_max=1.0

run_eval "s1024_ours_exact_N300_lam5000_win0.0-1.0" "${OURS_ARGS[@]}" \
  guidance.num_x0_samples=300 guidance.lambda_=5000 \
  guidance.t_min=0.0 guidance.t_max=1.0

run_eval "s1024_ours_exact_N500_lam5000_win0.0-1.0" "${OURS_ARGS[@]}" \
  guidance.num_x0_samples=500 guidance.lambda_=5000 \
  guidance.t_min=0.0 guidance.t_max=1.0

run_eval "s1024_ours_exact_N500_lam1000_win0.0-0.75" "${OURS_ARGS[@]}" \
  guidance.num_x0_samples=500 guidance.lambda_=1000 \
  guidance.t_min=0.0 guidance.t_max=0.75

echo "######## Done ########"
ls -1 "${ROOT}"/results/s1024_ours_exact_*.csv
