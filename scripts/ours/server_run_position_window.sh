#!/bin/bash
# Does confidence-based unmasking help if it is confined to the LATE steps?
#
# The 5-arm ablation established that no position rule beats the plain Bernoulli
# sampler on hits, and decomposed why: introducing a ranking at all costs ~43
# novel molecules out of ~360 (paired, p<0.05), because ranking commits to the
# most confident -- hence most typical -- positions and walks towards the mode of
# the data. Making the ranking reward-aware then wins ~5 hits back (C->E,
# p<0.05), i.e. our contribution repairs part of a wound the ranking itself
# inflicts.
#
# Early unmasking decisions fix the molecular scaffold, which is where novelty is
# decided; late ones are local substitutions. So this asks whether the quality
# gain survives when the ranking is restricted to late t, leaving the scaffold to
# the schedule's own coin flip.
#
# The reward tilt (lambda) stays on for the WHOLE trajectory -- only the position
# rule is windowed. They are separate mechanisms and `position_t_*` is separate
# from `t_min`/`t_max` for exactly this reason.
#
# Convention: t runs 1 (noise) -> 0 (clean), so `position_t_max = 0.25` means the
# ranking is active only on the last quarter of the trajectory.
#
# Usage (inside tmux):
#   PROP=qed CUDA_VISIBLE_DEVICES=0 bash scripts/ours/server_run_position_window.sh
set -uo pipefail

PROP="${PROP:-qed}"
MODEL=mdlm
ARM="${ARM:-cv_conf}"
STEPS=32; BATCH_SIZE=64; BATCHES=16; WORKERS=32
N="${N:-300}"
read -r -a SEED_GRID <<< "${SEED_GRID:-1 2 3 4 5}"
read -r -a WIN_GRID  <<< "${WIN_GRID:-0.25 0.5 0.75 1.0}"
case "${PROP}" in
  qed)        DEFAULT_LAM="50 200 1000" ;;
  ring_count) DEFAULT_LAM="2 5 10" ;;
  *) echo "FATAL: PROP must be qed or ring_count" >&2; exit 1 ;;
esac
read -r -a LAM_GRID <<< "${LAM_GRID:-${DEFAULT_LAM}}"

ROOT=/home/aailab/wp03052/Synthetic-Data/Molecule
RESULTS="${ROOT}/results/${PROP}/position_window"
MDLM_CKPT="${ROOT}/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt"

source /home/aailab/wp03052/venvs/dlrt_env/bin/activate
cd "${ROOT}"
export HF_HOME="${ROOT}/.hf_cache"
export PYTHONPATH="${ROOT}:${ROOT}/guidance_eval:${HF_HOME}/modules"
export HYDRA_FULL_ERROR=1 WANDB_MODE=offline WANDB_DIR="${ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${RESULTS}"
[ -s "${MDLM_CKPT}" ] || { echo "FATAL: missing ${MDLM_CKPT}" >&2; exit 1; }

# Wait for THIS card, checked with nvidia-smi: a detached tmux launcher can
# outlive its job while still carrying the wrapper's name on its command line,
# which once blocked a queued job for nine hours against an idle machine.
GPU="${CUDA_VISIBLE_DEVICES:-0}"
if [ "${WAIT_FOR_IDLE:-0}" -eq 1 ]; then
  echo "waiting for GPU ${GPU} to go idle ..."
  while [ "$(nvidia-smi -i "${GPU}" --query-compute-apps=pid \
             --format=csv,noheader 2>/dev/null | grep -c .)" -gt 0 ]; do sleep 120; done
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
    grep -E "^\s+Valid|Mean:" "${log}"; echo "  $((SECONDS - start))s"
  else
    echo "  FAILED -- see ${log}"; tail -6 "${log}"
  fi
}

echo "######## ${ARM} restricted to late t: ${MODEL} x ${PROP}, N=${N} ########"
echo "guidance window stays 0-1; only the position rule is windowed."
for SEED in "${SEED_GRID[@]}"; do
  for LAM in "${LAM_GRID[@]}"; do
    for WIN in "${WIN_GRID[@]}"; do
      run_eval "${MODEL}_pw_${ARM}_w${WIN}_N${N}_lam${LAM}_s${SEED}" \
        "seed=${SEED}" \
        guidance=ours "guidance.reward=${PROP}" \
        guidance.mixture_sampling=marginal \
        "++guidance.position_selection=${ARM}" \
        ++guidance.cv_coeff=1.0 \
        guidance.t_min=0.0 guidance.t_max=1.0 \
        ++guidance.position_t_min=0.0 "++guidance.position_t_max=${WIN}" \
        "guidance.num_x0_samples=${N}" "guidance.lambda_=${LAM}" \
        "guidance.num_reward_workers=${WORKERS}"
    done
  done
done
echo "######## Done: ${MODEL} x ${PROP} position window ########"
ls -1 "${RESULTS}"/*.csv 2>/dev/null | wc -l
