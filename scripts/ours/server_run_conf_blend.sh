#!/bin/bash
# Sweeps the C-D shrinkage weight of `position_selection=blend_conf`.
#
# The 4-arm ablation found xbar_conf (D) worse than xtheta_conf (C): xbar_0 is a
# Monte Carlo estimate over N candidates and its ranking is noisier than the
# exact x_theta, and the reward tilt it carries does not pay for that noise.
# blend_conf interpolates: beta=0 IS C, beta=1 IS D (exactly -- verified), so
# this sweep contains the previous two arms as endpoints and any interior
# optimum is the real contribution.
#
# Prediction worth recording before looking: if the diagnosis is right the best
# beta is interior and should move towards 1 as N grows, because xbar_0's error
# shrinks as O(1/sqrt(N)).
#
# Usage (waits for the GPUs to go idle; run inside tmux):
#   PROP=qed CUDA_VISIBLE_DEVICES=0 bash scripts/ours/server_run_conf_blend.sh
set -uo pipefail

PROP="${PROP:?set PROP=qed|ring_count}"
MODEL=mdlm
STEPS=32; BATCH_SIZE=64; BATCHES=16; WORKERS=32
# Seed replication. The run-to-run noise here is large -- two runs of an
# identical algorithm differing only in random stream gave QED hits of 30 and 49
# -- so single-seed differences between c values are not interpretable. Seed 1
# keeps the original tag so every cell already computed is reused.
read -r -a SEED_GRID <<< "${SEED_GRID:-1}"
# N is swept because the theory makes a prediction about it: beta* rises with
# ESS, and ESS rises with N, so the optimal blend should move towards xbar_0 as
# N grows. Without an N axis that prediction cannot be tested.
read -r -a N_GRID <<< "${N_GRID:-100 300 1000}"
BETA_GRID=(0 0.25 0.5 0.75 1)
# `cv_conf` is the principled alternative to picking a beta: it corrects xbar_0
# with a control variate built from the SAME candidates, whose mean is x_theta
# in closed form. Zero-mean correction, so unlike the blend it adds no bias, and
# it has no free parameter. Verified offline: variance falls to 53 % of the raw
# SNIS estimator with the means unchanged.
RUN_CV=${RUN_CV:-1}
# Fixed-c sweep. The variance-optimal c = Cov/Var minimises the error of the
# *estimate*, but this arm uses the score as a *ranking* -- and the beta sweep
# already showed those two objectives come apart (its MSE-optimal interior
# turned out to be a U-shaped worst case). So the coefficient is swept against
# the metric we actually care about rather than derived. c is not restricted to
# [0,1]; Cov/Var can exceed 1. c=0 reproduces xbar_conf exactly, which is a free
# consistency check against the beta=1 cells.
read -r -a CV_GRID <<< "${CV_GRID:-1}"
# lambda=0 is in the grid on purpose: with no tilt the two sources estimate the
# SAME quantity, so b=0 and the theory says beta*=0 exactly -- pure x_theta wins
# and any beta>0 only adds noise. That is the cleanest falsifiable prediction
# here, and it doubles as an implementation check.
case "${PROP}" in
  qed)        DEFAULT_LAM="0 50 200 1000" ;;
  ring_count) DEFAULT_LAM="0 2 5 10" ;;
  *) echo "FATAL: PROP must be qed or ring_count" >&2; exit 1 ;;
esac
# Overridable like the other axes. lambda=0 is in the default because the blend
# needs it as a control, but it carries no information for the c sweep: with
# uniform weights the correction collapses to a plain interpolation
# (1-c) xbar_0 + c x_theta, so nothing about the tilt is being measured.
read -r -a LAM_GRID <<< "${LAM_GRID:-${DEFAULT_LAM}}"

ROOT=/home/aailab/wp03052/Synthetic-Data/Molecule
RESULTS="${ROOT}/results/${PROP}/conf_blend"
MDLM_CKPT="${ROOT}/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt"

source /home/aailab/wp03052/venvs/dlrt_env/bin/activate
cd "${ROOT}"
export HF_HOME="${ROOT}/.hf_cache"
export PYTHONPATH="${ROOT}:${ROOT}/guidance_eval:${HF_HOME}/modules"
export HYDRA_FULL_ERROR=1 WANDB_MODE=offline WANDB_DIR="${ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${RESULTS}"
[ -s "${MDLM_CKPT}" ] || { echo "FATAL: missing ${MDLM_CKPT}" >&2; exit 1; }

# Wait for THIS card only, so shards on different GPUs run concurrently. Asking
# "are any evaluations running anywhere" would serialise the whole machine.
# Checked with nvidia-smi rather than a process name: a detached `tmux
# new-session` launcher can outlive its job while still carrying the wrapper's
# name on its command line, which once blocked a queued job for nine hours
# against a completely idle machine.
GPU="${CUDA_VISIBLE_DEVICES:-0}"
if [ "${WAIT_FOR_IDLE:-1}" -eq 1 ]; then
  echo "waiting for GPU ${GPU} to go idle ..."
  while [ "$(nvidia-smi -i "${GPU}" --query-compute-apps=pid \
             --format=csv,noheader 2>/dev/null | grep -c .)" -gt 0 ]; do
    sleep 120
  done
fi
echo "starting on GPU ${GPU} at $(date)"

BASE_ARGS=(
  data=qm9 "data.label_col=${PROP}" "data.cache_dir=${ROOT}/.data_cache"
  diffusion=absorbing_state parameterization=subs T=0
  time_conditioning=False zero_recon_loss=False training.guidance=null
  backbone=dit model=small model.length=32
  "eval.checkpoint_path=${MDLM_CKPT}" eval.disable_ema=False
  "sampling.steps=${STEPS}" "sampling.batch_size=${BATCH_SIZE}"
  "sampling.num_sample_batches=${BATCHES}" sampling.use_cache=False
)

run_eval() {
  local tag="$1"; shift
  local csv="${RESULTS}/${tag}.csv"
  if [ -s "${csv}" ]; then echo "[skip] ${tag}"; return 0; fi
  echo "=============== ${tag} ==============="
  local log="${RESULTS}/${tag}.log"; local start=${SECONDS}
  if python -u -W ignore guidance_eval/our_qm9_eval.py "${BASE_ARGS[@]}" "$@" \
      "++eval.results_csv_path=${csv}" \
      "++eval.generated_samples_path=${RESULTS}/${tag}_samples.json" \
      > "${log}" 2>&1; then
    grep -E "^\s+Valid|Mean:" "${log}"
    echo "  $((SECONDS - start))s"
  else
    echo "  FAILED -- see ${log}"; tail -6 "${log}"
  fi
}

for SEED in "${SEED_GRID[@]}"; do
 SFX=""; [ "${SEED}" = "1" ] || SFX="_s${SEED}"
 for N in "${N_GRID[@]}"; do
  for LAM in "${LAM_GRID[@]}"; do
    for BETA in "${BETA_GRID[@]}"; do
      run_eval "${MODEL}_cb_beta${BETA}_N${N}_lam${LAM}${SFX}" "seed=${SEED}" \
        guidance=ours "guidance.reward=${PROP}" \
        guidance.mixture_sampling=marginal \
        ++guidance.position_selection=blend_conf \
        "++guidance.conf_blend=${BETA}" \
        "guidance.num_x0_samples=${N}" "guidance.lambda_=${LAM}" \
        guidance.t_min=0.0 guidance.t_max=1.0 \
        "guidance.num_reward_workers=${WORKERS}"
    done
    if [ "${RUN_CV}" -eq 1 ]; then
      for C in "${CV_GRID[@]}"; do
        # c=1 keeps the original tag so the cells already run are reused.
        if [ "${C}" = "1" ]; then tag="${MODEL}_cb_cv_N${N}_lam${LAM}${SFX}"
        else tag="${MODEL}_cb_cv${C}_N${N}_lam${LAM}${SFX}"; fi
        run_eval "${tag}" "seed=${SEED}" \
          guidance=ours "guidance.reward=${PROP}" \
          guidance.mixture_sampling=marginal \
          ++guidance.position_selection=cv_conf \
          "++guidance.cv_coeff=${C}" \
          "guidance.num_x0_samples=${N}" "guidance.lambda_=${LAM}" \
          guidance.t_min=0.0 guidance.t_max=1.0 \
          "guidance.num_reward_workers=${WORKERS}"
      done
    fi
  done
 done
done
echo "######## Done: ${MODEL} x ${PROP} conf_blend ########"
ls -1 "${RESULTS}"/*.csv 2>/dev/null | wc -l
