#!/bin/bash
# Ring-count counterpart of scripts/ours/qed/server_run_dcbg_repro.sh: reproduction
# check for the D-CBG ring-count rows of arXiv 2412.10193v3, Tables 29 and 32.
#
# Nothing about the *generative* model changes for a new property -- the base
# MDLM/UDLM is unconditional and D-CBG steers it from outside, so upstream's own
# eval reuses `outputs/qm9/<model>_no-guidance` for both properties. What is
# property-specific is the noisy classifier, hence the ring_count checkpoints
# this script waits for. Our own method needs neither: `guidance.reward` already
# accepts ring_count and `our_guidance._mol_ring_count` is the same
# `len(GetSymmSSSR(mol))` that upstream's eval measures.
#
# The main-paper (γ, use_approx) setting differs from QED's, and differs between
# the two models -- the paper tunes it per (model, property):
#
#   MDLM D-CBG ring count: gamma=10, use_approx=True   (Table 29)
#   UDLM D-CBG ring count: gamma=8,  use_approx=False  (Table 32)
#
# MDLM falls back to the approximation because the exact arm collapses: 143.8
# valid at gamma=1 and *zero* valid for gamma >= 2. So on MDLM only gamma=1 is
# worth running exactly; the curve lives on the approximate arm.
#
# Reference values (paper, mean +- std over 5 seeds, counts out of 1024,
# "Novel Ring Count Mean" is the property averaged over novel sequences only):
#
#   MDLM (Table 29)                      UDLM (Table 32)
#   g   approx  valid  novel  nRing      g   approx  valid  novel  nRing
#   1   False   143.8  121.2  5.00       1   False   797.4  279.0  4.12
#   1   True    455.4  229.4  2.52       2   False   829.4  336.4  4.54
#   2   True    327.6  176.4  2.82       3   False   862.6  363.8  4.70
#   3   True    223.8  136.0  3.25       5   False   889.2  404.0  4.76
#   5   True    135.0   94.4  4.11       8   False   897.2  432.0  4.84 *
#   8   True    121.2   92.6  4.54       10  False   891.6  431.4  4.76
#   10  True    113.0   85.6  4.75 *     8   True    796.6  304.4  4.66
#                                        10  True    816.6  359.8  4.70
#   (* = the row that appears in Table 5.)
#
# Original data mean ring count is 1.74, and the classifier's positive class is
# `ring_count > p90 = 3`, i.e. molecules with >= 4 rings (7.83 % of QM9) --
# `dataloader.py:282` assigns class 0 to `value <= p90`.
#
# `use_approx=False` expands the classifier over the whole vocabulary at every
# position and OOMs a 24 GB card at batch 64; the proven setting is 8x128.
#
# Usage (inside tmux, survives disconnects):
#   bash scripts/ours/ring_count/server_run_dcbg_repro_ring.sh
set -uo pipefail

PROP=ring_count
STEPS=32
SEED=1
APPROX_BATCH=64;  APPROX_BATCHES=16    # 1024 samples
EXACT_BATCH=8;    EXACT_BATCHES=128    # 1024 samples, fits in 24 GB

ROOT=/home/aailab/wp03052/Synthetic-Data/Molecule
RESULTS="${ROOT}/results/${PROP}"
MDLM_CKPT="${ROOT}/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt"
CLASS_UNIFORM="${ROOT}/outputs/qm9/classifier/${PROP}_uniform_T-0/checkpoints/best.ckpt"
CLASS_ABSORB="${ROOT}/outputs/qm9/classifier/${PROP}_absorbing_state_T-0/checkpoints/best.ckpt"

source /home/aailab/wp03052/venvs/dlrt_env/bin/activate
cd "${ROOT}"
export HF_HOME="${ROOT}/.hf_cache"
export PYTHONPATH="${ROOT}:${ROOT}/guidance_eval:${HF_HOME}/modules"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export WANDB_DIR="${ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${RESULTS}"

if [ ! -s "${MDLM_CKPT}" ]; then echo "FATAL: missing ${MDLM_CKPT}"; exit 1; fi

echo "Waiting for the ring_count classifiers to finish training ..."
while [ ! -s "${CLASS_UNIFORM}" ] || [ ! -s "${CLASS_ABSORB}" ] \
      || pgrep -f "mode=train_classifier" > /dev/null; do
  command sleep 120
done
echo "Both ring_count classifiers ready."

udlm_args() {  # <batch> <batches>
  echo data=qm9 "data.label_col=${PROP}" "data.cache_dir=${ROOT}/.data_cache" \
    backbone=hf_dit model=hf model.length=32 \
    model.pretrained_model_name_or_path=kuleshov-group/udlm-qm9 \
    diffusion=uniform parameterization=d3pm \
    time_conditioning=True zero_recon_loss=True \
    "sampling.steps=${STEPS}" "sampling.batch_size=$1" \
    "sampling.num_sample_batches=$2" sampling.use_cache=False \
    eval.disable_ema=True "seed=${SEED}"
}

mdlm_args() {  # <batch> <batches>
  echo data=qm9 "data.label_col=${PROP}" "data.cache_dir=${ROOT}/.data_cache" \
    diffusion=absorbing_state parameterization=subs T=0 \
    time_conditioning=False zero_recon_loss=False \
    training.guidance=null \
    backbone=dit model=small model.length=32 \
    "eval.checkpoint_path=${MDLM_CKPT}" \
    "sampling.steps=${STEPS}" "sampling.batch_size=$1" \
    "sampling.num_sample_batches=$2" sampling.use_cache=False \
    eval.disable_ema=False "seed=${SEED}"
}

cbg_args() {  # <classifier ckpt> <gamma> <use_approx>
  echo guidance=cbg guidance.condition=1 \
    classifier_model=tiny-classifier classifier_backbone=dit \
    "guidance.classifier_checkpoint_path=$1" \
    "guidance.gamma=$2" "guidance.use_approx=$3"
}

run_eval() {
  local tag="$1"; shift
  local csv="${RESULTS}/${tag}.csv"
  if [ -s "${csv}" ]; then echo "[skip] ${tag}"; return 0; fi
  echo "=============== ${tag} ==============="
  local log="${RESULTS}/${tag}.log"
  local start=${SECONDS}
  if python -u -W ignore guidance_eval/our_qm9_eval.py \
      "$@" \
      "++eval.results_csv_path=${csv}" \
      "++eval.generated_samples_path=${RESULTS}/${tag}_samples.json" \
      > "${log}" 2>&1; then
    grep -E "^\s+Valid|Mean:" "${log}"
    echo "  $((SECONDS - start))s"
  else
    echo "  FAILED -- see ${log}"; tail -6 "${log}"
  fi
}

echo "######## unguided references ########"
# t outside [t_min, t_max] makes `_our_denoise` fall through to the base
# sampler, so these are the untouched MDLM / UDLM trajectories scored on ring
# count. Needed as the zero point of the frontier.
UNGUIDED=(guidance=ours "guidance.reward=${PROP}" guidance.mixture_sampling=marginal
          guidance.num_x0_samples=2 guidance.lambda_=0.0
          guidance.t_min=2.0 guidance.t_max=2.0 guidance.num_reward_workers=1)
run_eval "mdlm_unguided"  $(mdlm_args "${APPROX_BATCH}" "${APPROX_BATCHES}") "${UNGUIDED[@]}"
run_eval "s1024_unguided" $(udlm_args "${APPROX_BATCH}" "${APPROX_BATCHES}") "${UNGUIDED[@]}"

echo "######## approx arm (fast) ########"
for GAMMA in 1 2 3 5 8 10; do
  run_eval "mdlm_cbg_approx_gamma${GAMMA}" \
    $(mdlm_args "${APPROX_BATCH}" "${APPROX_BATCHES}") \
    $(cbg_args "${CLASS_ABSORB}" "${GAMMA}" True)
done
for GAMMA in 1 2 3 5 8 10; do
  run_eval "s1024_cbg_approx_gamma${GAMMA}" \
    $(udlm_args "${APPROX_BATCH}" "${APPROX_BATCHES}") \
    $(cbg_args "${CLASS_UNIFORM}" "${GAMMA}" True)
done

echo "######## exact arm: main-paper setting first ########"
run_eval "s1024_cbg_exact_gamma8" \
  $(udlm_args "${EXACT_BATCH}" "${EXACT_BATCHES}") \
  $(cbg_args "${CLASS_UNIFORM}" 8 False)
# MDLM exact is only viable at gamma=1 (zero valid molecules from gamma=2 on).
run_eval "mdlm_cbg_exact_gamma1" \
  $(mdlm_args "${EXACT_BATCH}" "${EXACT_BATCHES}") \
  $(cbg_args "${CLASS_ABSORB}" 1 False)

echo "######## exact arm: rest of the UDLM curve ########"
for GAMMA in 1 2 3 5 10; do
  run_eval "s1024_cbg_exact_gamma${GAMMA}" \
    $(udlm_args "${EXACT_BATCH}" "${EXACT_BATCHES}") \
    $(cbg_args "${CLASS_UNIFORM}" "${GAMMA}" False)
done
# Confirms the collapse the paper reports rather than assuming it.
run_eval "mdlm_cbg_exact_gamma2" \
  $(mdlm_args "${EXACT_BATCH}" "${EXACT_BATCHES}") \
  $(cbg_args "${CLASS_ABSORB}" 2 False)

echo "######## Done ########"
ls -1 "${RESULTS}"/*.csv 2>/dev/null
