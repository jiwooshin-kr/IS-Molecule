#!/bin/bash
# The 4-arm confidence-unmasking ablation. MDLM only -- see below.
#
# In absorbing-state diffusion the posterior at a masked position gives
# `P(stay masked) = move_chance_s / move_chance_t`, the *same* value at every
# masked position and independent of the token distribution. So the base
# sampler's choice of which position to unmask carries zero information. This
# ablation asks whether spending that free choice well is worth anything.
#
#   A  bernoulli     the schedule's coin flip                 (baseline)
#   B  random_k      deterministic count, random positions    (isolates the count)
#   C  xtheta_conf   ranked by the denoiser marginal          (isolates ranking)
#   D  xbar_conf     ranked by the reward-weighted histogram  (OURS)
#
# C is the real baseline; **C -> D is the contribution**. D at lambda -> 0 should
# converge to C, up to the O(1/sqrt(N)) noise in xbar_0 as an estimate of
# x_theta -- which doubles as an implementation-correctness check.
#
# UDLM cannot run this at all: uniform diffusion has no mask state, it resamples
# every position at every step, so there is no "which position to unmask" to
# decide. The script refuses anything but MDLM.
#
# All four arms run under `mixture_sampling=marginal`, because D is only defined
# there: under `edlm` the per-position distribution is a one-hot on the single
# drawn candidate, so confidence is 1 at every masked position and the ranking
# degenerates to ties. The EDLM CHECK block at the end verifies exactly that --
# under `edlm`, arms B and D must come out statistically indistinguishable.
#
# Usage (waits for the main sweep to finish; run inside tmux):
#   PROP=qed CUDA_VISIBLE_DEVICES=0 bash scripts/ours/server_run_confidence_unmasking.sh
set -uo pipefail

PROP="${PROP:?set PROP=qed|ring_count}"
MODEL=mdlm
STEPS=32            # = model.length on purpose: MDLM writes at most L tokens,
                    # so T > L only adds no-op steps.
# Seed replication: the run-to-run noise on `hits` is ~4-6 molecules, which is
# the same size as the between-arm differences, so single-seed rows here are not
# interpretable on their own. Seed 1 keeps the original tag, so cells already
# computed are reused.
read -r -a SEED_GRID <<< "${SEED_GRID:-1}"
BATCH_SIZE=64
BATCHES=16          # 1024 samples
WORKERS=32
N="${N:-300}"       # fixed by default: this ablation varies the position rule,
                    # not the Monte Carlo budget. N=300 is where the tilt is
                    # clearly alive on MDLM (median ESS/N ~ 0.44).
# Which arms to run, and over which lambdas -- overridable so the baseline arms
# can be replicated on exactly the grid the blend/CV runs used.
read -r -a ARM_GRID <<< "${ARM_GRID:-bernoulli random_k xtheta_conf xbar_conf}"

case "${PROP}" in
  qed)        DEFAULT_LAM="0 2 5 20 50 200 1000" ;;
  ring_count) DEFAULT_LAM="0 0.5 1 2 5 10 30" ;;
  *) echo "FATAL: PROP must be qed or ring_count, got '${PROP}'" >&2; exit 1 ;;
esac
read -r -a LAM_GRID <<< "${LAM_GRID:-${DEFAULT_LAM}}"

ROOT=/home/aailab/wp03052/Synthetic-Data/Molecule
RESULTS="${ROOT}/results/${PROP}/confidence_unmasking"
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

if [ ! -s "${MDLM_CKPT}" ]; then echo "FATAL: missing ${MDLM_CKPT}" >&2; exit 1; fi

# Do not contend for GPUs: wait until no evaluation is actually running.
#
# Match on `our_qm9_eval`, the process that holds the GPU -- NOT on a wrapper
# script name. A `tmux new-session -d -s x "... bash <wrapper>.sh ..."` launcher
# process can outlive both the session and the job while still carrying the
# wrapper's name on its command line, and waiting on that name then blocks
# forever against a machine that is completely idle. That is exactly what
# happened on 2026-08-15 and cost the ablation a 9-hour window.
#
# The bracket on the first character keeps the pattern from matching the shell
# that ssh spawns to run this very command.
echo "waiting for the GPUs to go idle ..."
while [ "$(pgrep -fc '[o]ur_qm9_eval' || true)" -gt 0 ]; do
  sleep 120
done
echo "GPUs are idle, starting at $(date)"

BASE_ARGS=(
  data=qm9 "data.label_col=${PROP}" "data.cache_dir=${ROOT}/.data_cache"
  diffusion=absorbing_state parameterization=subs T=0
  time_conditioning=False zero_recon_loss=False
  training.guidance=null
  backbone=dit model=small model.length=32
  "eval.checkpoint_path=${MDLM_CKPT}"
  eval.disable_ema=False
  "sampling.steps=${STEPS}" "sampling.batch_size=${BATCH_SIZE}"
  "sampling.num_sample_batches=${BATCHES}" sampling.use_cache=False
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
    echo "  last ESS: $(grep -o 'ESS=[0-9.]*' "${log}" | tail -1)" \
         "| $((SECONDS - start))s"
  else
    echo "  FAILED -- see ${log}"; tail -6 "${log}"
  fi
}

ours_args() {  # <mixture_sampling> <position_selection> <lambda>
  echo guidance=ours "guidance.reward=${PROP}" \
    "guidance.mixture_sampling=$1" \
    "++guidance.position_selection=$2" \
    "guidance.num_x0_samples=${N}" "guidance.lambda_=$3" \
    guidance.t_min=0.0 guidance.t_max=1.0 \
    "guidance.num_reward_workers=${WORKERS}"
}

# lambda outermost so that a run killed part-way still has all four arms matched
# at every lambda it finished -- the arms are the comparison, so they must move
# together.
echo "######## arms [${ARM_GRID[*]}]: ${MODEL} x ${PROP}, N=${N} ########"
for SEED in "${SEED_GRID[@]}"; do
  SFX=""; [ "${SEED}" = "1" ] || SFX="_s${SEED}"
  for LAM in "${LAM_GRID[@]}"; do
    for ARM in "${ARM_GRID[@]}"; do
      run_eval "${MODEL}_cu_${ARM}_marginal_N${N}_lam${LAM}${SFX}" "seed=${SEED}" \
        $(ours_args marginal "${ARM}" "${LAM}")
    done
  done
done

# EDLM CHECK: D is undefined under edlm (one-hot => confidence 1 everywhere), so
# it must reduce to B. Two lambdas is enough to see it.
echo "######## edlm degeneracy check (xbar_conf must match random_k) ########"
for LAM in "${LAM_GRID[1]}" "${LAM_GRID[-1]}"; do
  for ARM in random_k xbar_conf; do
    run_eval "${MODEL}_cu_${ARM}_edlm_N${N}_lam${LAM}" \
      $(ours_args edlm "${ARM}" "${LAM}")
  done
done

echo "######## Done: ${MODEL} x ${PROP} ########"
ls -1 "${RESULTS}"/*.csv 2>/dev/null | wc -l
