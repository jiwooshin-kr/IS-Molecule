#!/bin/bash
# Does screening candidates for validity beat not screening, at a matched
# reward budget?
#
#   over1  : draw N candidates i.i.d. from x_theta, score all N with QED.
#            The method exactly as every earlier result ran it.
#   over10 : draw 10N, parse them (parse costs ~0.04 ms against ~0.81 ms for
#            QED, i.e. 5 % of the full evaluation), keep N -- every valid
#            candidate in random order, padded with random invalid ones if
#            fewer than N are valid -- and score only those N with QED.
#
# **Both arms make exactly N x batch QED calls per step.** The expensive
# operation is held fixed and only its *targeting* changes, so this isolates
# "were the candidates worth scoring?" from "how many did we score?".
#
# Statistically this changes the proposal from p_theta to p_theta(.|valid),
# which is a different target -- the same one `invalid_reward = -inf` defines --
# reached with N usable candidates instead of N x (valid fraction). That
# fraction was measured as low as 0.007 early in a UDLM trajectory, i.e. 2 of
# 300 candidates.
#
# Prediction worth recording: the gain should be much larger on UDLM, whose
# candidate validity stays at 0.007-0.03 for most of the trajectory, than on
# MDLM, which reaches 0.69 by the end and so has little to screen.
#
# lambda is swept wide and high on purpose. Screening removes the invalid
# candidates, and with them the ~0.5 valid/invalid reward gap that used to
# dominate the tilt; what is left is the ~0.1 spread among valid QEDs, so the
# same tilt needs a markedly larger lambda.
#
# Usage (inside tmux):
#   MODEL=udlm PROP=qed CUDA_VISIBLE_DEVICES=0 bash scripts/ours/server_run_oversample.sh
set -uo pipefail

MODEL="${MODEL:?set MODEL=mdlm|udlm}"
PROP="${PROP:-qed}"
STEPS=32; BATCH_SIZE=64; BATCHES=16; WORKERS=32
N="${N:-300}"
read -r -a SEED_GRID <<< "${SEED_GRID:-1 2 3}"
read -r -a LAM_GRID  <<< "${LAM_GRID:-50 200 1000 5000}"
read -r -a OVER_GRID <<< "${OVER_GRID:-1 10}"
# `marginal` keeps the mode out of its tag so the cells already computed under
# the earlier grid are reused; `edlm` gets a suffix.
read -r -a MODE_GRID <<< "${MODE_GRID:-marginal edlm}"
# exclude_invalid restricts the target to parseable molecules. It is an axis
# because it changes the *target*: comparing over1 against overK is only fair
# with it held fixed. `False` keeps the tag unsuffixed so earlier cells are
# reused.
read -r -a EXCL_GRID <<< "${EXCL_GRID:-False}"
# Replace the Monte Carlo mixture with the exact base kernel wherever the
# weights come out uniform (lambda_=0, or every candidate ties because none
# parses). Absorbing state only; a no-op on UDLM. See our_guidance.py.
EXACT_UNIF="${EXACT_UNIF:-False}"
# Appended to every tag. EXACT_UNIF is not part of the tag, so re-running a cell
# with it flipped needs an explicit marker or the existing CSV is skipped.
TAG_EXTRA="${TAG_EXTRA:-}"

ROOT=/home/aailab/wp03052/Synthetic-Data/Molecule
RESULTS="${ROOT}/results/${PROP}/oversample"
MDLM_CKPT="${ROOT}/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt"

source /home/aailab/wp03052/venvs/dlrt_env/bin/activate
cd "${ROOT}"
export HF_HOME="${ROOT}/.hf_cache"
export PYTHONPATH="${ROOT}:${ROOT}/guidance_eval:${HF_HOME}/modules"
export HYDRA_FULL_ERROR=1 WANDB_MODE=offline WANDB_DIR="${ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${RESULTS}"

case "${MODEL}" in
  udlm)
    TAG=s1024
    BASE_ARGS=(backbone=hf_dit model=hf model.length=32
      model.pretrained_model_name_or_path=kuleshov-group/udlm-qm9
      diffusion=uniform parameterization=d3pm
      time_conditioning=True zero_recon_loss=True eval.disable_ema=True) ;;
  mdlm)
    TAG=mdlm
    [ -s "${MDLM_CKPT}" ] || { echo "FATAL: missing ${MDLM_CKPT}" >&2; exit 1; }
    BASE_ARGS=(diffusion=absorbing_state parameterization=subs T=0
      time_conditioning=False zero_recon_loss=False training.guidance=null
      backbone=dit model=small model.length=32
      "eval.checkpoint_path=${MDLM_CKPT}" eval.disable_ema=False) ;;
  *) echo "FATAL: MODEL must be mdlm or udlm" >&2; exit 1 ;;
esac
BASE_ARGS+=(
  data=qm9 "data.label_col=${PROP}" "data.cache_dir=${ROOT}/.data_cache"
  "sampling.steps=${STEPS}" "sampling.batch_size=${BATCH_SIZE}"
  "sampling.num_sample_batches=${BATCHES}" sampling.use_cache=False
)

# Wait for THIS card only, via nvidia-smi: a detached tmux launcher can outlive
# its job while still carrying the wrapper's name on its command line, which
# once blocked a queued job for nine hours against an idle machine.
GPU="${CUDA_VISIBLE_DEVICES:-0}"
if [ "${WAIT_FOR_IDLE:-0}" -eq 1 ]; then
  echo "waiting for GPU ${GPU} ..."
  while [ "$(nvidia-smi -i "${GPU}" --query-compute-apps=pid \
             --format=csv,noheader 2>/dev/null | grep -c .)" -gt 0 ]; do sleep 120; done
fi
echo "starting ${MODEL} x ${PROP} on GPU ${GPU} at $(date)"

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
    # `pool` is the valid fraction of the oversampled pool and `short` the
    # fraction of sequences that could not supply N valid candidates -- the
    # cases where the padding with invalids kicks in.
    echo "  pool=$(grep -o 'pool=[0-9.]*' "${log}" | tail -1)" \
         "short=$(grep -o 'short=[0-9.]*' "${log}" | tail -1)" \
         "| $((SECONDS - start))s"
  else
    echo "  FAILED -- see ${log}"; tail -6 "${log}"
  fi
}

read -r -a N_GRID <<< "${N_GRID:-${N}}"
for SEED in "${SEED_GRID[@]}"; do
  for NN in "${N_GRID[@]}"; do
    for LAM in "${LAM_GRID[@]}"; do
      for MODE in "${MODE_GRID[@]}"; do
        MSFX=""; [ "${MODE}" = "marginal" ] || MSFX="_${MODE}"
        for OVER in "${OVER_GRID[@]}"; do
         for EXCL in "${EXCL_GRID[@]}"; do
          ESFX=""; [ "${EXCL}" = "False" ] || ESFX="_ex"
          run_eval "${TAG}_ov${OVER}${ESFX}_N${NN}_lam${LAM}_s${SEED}${MSFX}${TAG_EXTRA}" \
            "seed=${SEED}" \
            guidance=ours "guidance.reward=${PROP}" \
            "guidance.mixture_sampling=${MODE}" \
            "++guidance.oversample=${OVER}" \
            "++guidance.exclude_invalid=${EXCL}" \
            "++guidance.exact_uniform_step=${EXACT_UNIF}" \
            "guidance.num_x0_samples=${NN}" "guidance.lambda_=${LAM}" \
            guidance.t_min=0.0 guidance.t_max=1.0 \
            "guidance.num_reward_workers=${WORKERS}"
         done
        done
      done
    done
  done
done
echo "######## Done: ${MODEL} x ${PROP} oversample ########"
ls -1 "${RESULTS}"/${TAG}_ov*.csv 2>/dev/null | wc -l
