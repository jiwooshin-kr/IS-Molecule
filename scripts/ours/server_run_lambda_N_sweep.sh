#!/bin/bash
# lambda x N sweep for the two surviving mixture-sampling modes.
#
#   marginal : our method -- average the N individually normalized posteriors
#   edlm     : EDLM Algorithm 1 with the learned energy replaced by our reward
#
# (`aggregate_x0` and `support_floor` were deleted on 2026-08-13; every earlier
# `ours_N*_lam*` result without a mode in its tag was produced by aggregate_x0
# and is NOT comparable to these on uniform diffusion.)
#
# Together with the D-CBG gamma sweeps already in results/{qed,ring_count}/,
# this gives the reward-diversity frontier both methods are compared on --
# never at single hyperparameter points, since the paper tunes per model *and*
# property.
#
# Usage (one invocation per GPU, inside its own tmux session):
#   MODEL=udlm PROP=qed        CUDA_VISIBLE_DEVICES=0 bash scripts/ours/server_run_lambda_N_sweep.sh
#   MODEL=udlm PROP=ring_count CUDA_VISIBLE_DEVICES=1 bash scripts/ours/server_run_lambda_N_sweep.sh
#   MODEL=mdlm PROP=qed        CUDA_VISIBLE_DEVICES=2 bash scripts/ours/server_run_lambda_N_sweep.sh
#   MODEL=mdlm PROP=ring_count CUDA_VISIBLE_DEVICES=3 bash scripts/ours/server_run_lambda_N_sweep.sh
#
# Resumable: a config whose CSV already exists is skipped, so re-running after a
# crash costs nothing.
set -uo pipefail

MODEL="${MODEL:?set MODEL=mdlm|udlm}"
PROP="${PROP:?set PROP=qed|ring_count}"

STEPS=32
SEED=1
BATCH_SIZE=64
BATCHES=16          # 1024 samples, the budget every other table uses
WORKERS=32

# N is the number of x_0 candidates per denoising step: the Monte Carlo budget
# of the estimator, so this axis measures estimator variance, not tilt.
#
# Overridable, because extending a grid is the common case and the CSV-exists
# skip makes a superset re-run cost only the new cells:
#   N_GRID="10 30 100 300 500 1000 2000" bash ...
# Cost is markedly super-linear past N=500 -- measured 156 s at N=500 but 8 min
# at N=1000 and 12 min at N=2000, against the 40 + 0.23 N seconds that fits the
# small-N range. Budget from the measured points, not from that fit.
read -r -a N_GRID <<< "${N_GRID:-10 30 100 300 500}"

# lambda is the tilt: q(x_0) \propto p_theta(x_0) exp(lambda * r(x_0)). It is
# NOT transferable across properties, and the reason is not the reward's *range*
# but the gaps between candidates:
#
#   ring_count is integer valued (`_mol_ring_count` = len(GetSymmSSSR)), so the
#   smallest nonzero gap between two candidates is 1. Once lambda * 1 >> 1 the
#   softmax is already a hard argmax over the tied top set, and raising lambda
#   further changes nothing at all. Measured on UDLM at N=30 (kept in
#   results/_lambda_calibration/): lambda = 20, 100, 1,000 and 10,000 give
#   *identical* output -- min ESS 14.6, validity 95.31 %, novel ring 1.909. The
#   whole transition lives in lambda ~ [0.5, 20].
#
#   QED is continuous, so candidates almost never tie and the tilt keeps biting
#   much longer: min ESS 29.1 -> 12.1 -> 7.5 -> 5.5 across lambda = 1, 20, 100,
#   1,000, flattening only after ~10^2-10^3.
#
# Hence two grids that differ by ~100x, each log-spaced across its own live
# range with one saturated anchor at the top.
#
# The ring grid carries two points (100, 1000) well past its own saturation
# point on purpose: saturation was calibrated at N=30, and more candidates mean
# more distinct reward values and fewer ties, so the point where lambda stops
# biting can move with N. Those cells are what test that.
case "${PROP}" in
  qed)        DEFAULT_LAM="0 2 5 20 50 200 1000" ;;
  ring_count) DEFAULT_LAM="0 0.5 1 2 5 10 30 100 1000" ;;
  *) echo "FATAL: PROP must be qed or ring_count, got '${PROP}'" >&2; exit 1 ;;
esac
read -r -a LAM_GRID <<< "${LAM_GRID:-${DEFAULT_LAM}}"

ROOT=/home/aailab/wp03052/Synthetic-Data/Molecule
RESULTS="${ROOT}/results/${PROP}"
MDLM_CKPT="${ROOT}/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt"

source /home/aailab/wp03052/venvs/dlrt_env/bin/activate
cd "${ROOT}"
export HF_HOME="${ROOT}/.hf_cache"
export PYTHONPATH="${ROOT}:${ROOT}/guidance_eval:${HF_HOME}/modules"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export WANDB_DIR="${ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${RESULTS}"

# `s1024_` is the UDLM prefix the existing tags use; `mdlm_` marks MDLM. Keep
# them -- make_results_table.py's parse_tag routes on exactly this.
case "${MODEL}" in
  udlm)
    TAG_PREFIX=s1024
    BASE_ARGS=(
      backbone=hf_dit model=hf model.length=32
      model.pretrained_model_name_or_path=kuleshov-group/udlm-qm9
      diffusion=uniform parameterization=d3pm
      time_conditioning=True zero_recon_loss=True
      eval.disable_ema=True
    ) ;;
  mdlm)
    TAG_PREFIX=mdlm
    if [ ! -s "${MDLM_CKPT}" ]; then echo "FATAL: missing ${MDLM_CKPT}" >&2; exit 1; fi
    BASE_ARGS=(
      diffusion=absorbing_state parameterization=subs T=0
      time_conditioning=False zero_recon_loss=False
      training.guidance=null
      backbone=dit model=small model.length=32
      "eval.checkpoint_path=${MDLM_CKPT}"
      eval.disable_ema=False
    ) ;;
  *) echo "FATAL: MODEL must be mdlm or udlm, got '${MODEL}'" >&2; exit 1 ;;
esac

BASE_ARGS+=(
  data=qm9 "data.label_col=${PROP}" "data.cache_dir=${ROOT}/.data_cache"
  "sampling.steps=${STEPS}" "sampling.batch_size=${BATCH_SIZE}"
  "sampling.num_sample_batches=${BATCHES}" sampling.use_cache=False
  "seed=${SEED}"
)

run_eval() {
  local tag="$1"; shift
  local csv="${RESULTS}/${tag}.csv"
  if [ -s "${csv}" ]; then echo "[skip] ${tag}"; return 0; fi
  echo "=============== ${tag} ==============="
  local log="${RESULTS}/${tag}.log"
  local start=${SECONDS}
  if python -u -W ignore guidance_eval/our_qm9_eval.py \
      "${BASE_ARGS[@]}" "$@" \
      "++eval.results_csv_path=${csv}" \
      "++eval.generated_samples_path=${RESULTS}/${tag}_samples.json" \
      > "${log}" 2>&1; then
    grep -E "^\s+Valid|Mean:" "${log}"
    # ESS is the effective sample size of the N importance weights, logged to the
    # tqdm postfix each step. ESS -> 1 means the tilt collapsed onto one
    # candidate and larger N is buying nothing; ESS -> N means lambda is doing
    # nothing. It is the diagnostic that says which end of the grid we are on.
    echo "  last ESS: $(grep -o 'ESS=[0-9.]*' "${log}" | tail -1)" \
         "| $((SECONDS - start))s"
  else
    echo "  FAILED -- see ${log}"; tail -6 "${log}"
  fi
}

echo "######## ${MODEL} x ${PROP}: ${#N_GRID[@]} N x ${#LAM_GRID[@]} lambda x 2 modes ########"
echo "lambda grid: ${LAM_GRID[*]}"
echo "N grid     : ${N_GRID[*]}"

# N outermost and ascending, mode innermost: a sweep killed part-way still
# leaves a complete lambda curve at every N it finished, with both modes matched
# at each point -- which is exactly what the frontier plot needs. Ordering by
# mode instead would leave one mode entirely unrun.
for N in "${N_GRID[@]}"; do
  for LAM in "${LAM_GRID[@]}"; do
    for MODE in marginal edlm; do
      run_eval "${TAG_PREFIX}_ours_${MODE}_N${N}_lam${LAM}_win0.0-1.0" \
        guidance=ours "guidance.reward=${PROP}" \
        "guidance.mixture_sampling=${MODE}" \
        "guidance.num_x0_samples=${N}" \
        "guidance.lambda_=${LAM}" \
        guidance.t_min=0.0 guidance.t_max=1.0 \
        "guidance.num_reward_workers=${WORKERS}"
    done
  done
done

echo "######## Done: ${MODEL} x ${PROP} ########"
ls -1 "${RESULTS}"/${TAG_PREFIX}_ours_*.csv 2>/dev/null | wc -l
