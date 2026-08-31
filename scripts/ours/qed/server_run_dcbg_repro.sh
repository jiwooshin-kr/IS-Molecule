#!/bin/bash
# Reproduction check for the D-CBG rows of "Simple Guidance Mechanisms for
# Discrete Diffusion Models" (arXiv 2412.10193v3), Tables 20 and 23.
#
# The paper's appendix settles the two things Table 5 leaves ambiguous:
#   MDLM D-CBG main-paper setting: gamma=3, use_approx=False  (Table 20)
#   UDLM D-CBG main-paper setting: gamma=10, use_approx=False (Table 23)
# Both report Num. Valid / Num. Novel as counts out of 1024 and "Novel QED
# Mean", i.e. the property averaged over *novel* sequences only, over five
# seeds. Our CSVs store Valid as a fraction of 1024 and Novel as a fraction of
# valid, so converting is: valid = Valid * 1024, novel = Novel * valid.
#
# Reference values to compare against (paper, mean +- std over 5 seeds):
#
#   MDLM (Table 20)                     UDLM (Table 23)
#   g   approx  valid   novel  nQED     g   approx  valid   novel  nQED
#   1   False   526.4   170.8  0.56     1   False   933.4   134.6  0.53
#   1   True    476.0   221.4  0.46     1   True    996.4   132.2  0.47
#   2   False   524.8   139.0  0.58     2   False   911.2   119.8  0.57
#   2   True    363.4   172.4  0.47     2   True    974.8   110.6  0.49
#   3   False   417.6   116.6  0.58 *   3   False   941.0    96.4  0.58
#   3   True    244.0   114.0  0.47     3   True    925.0   111.4  0.51
#   4   False   200.4    66.4  0.58     5   False   967.6    77.8  0.59
#   5   False    24.6    11.6  0.58     10  False   994.8    63.8  0.61 *
#   5   True    120.6    48.4  0.50     10  True    783.2    67.0  0.56
#   10  True     72.6    21.2  0.55
#   (* = the row that appears in Table 5. MDLM exact collapses to 0 valid for
#    gamma >= 6, so gammas above 5 are not worth running on that arm.)
#
# IMPORTANT: `use_approx=False` evaluates the classifier over the whole
# vocabulary at every position, so it needs a small batch -- batch_size=64 OOMs
# on a 24 GB 3090 (5 GiB single allocation). The proven setting from
# scripts/ours/server_run_cbg_exact.sh is batch_size=8 with 128 batches, which
# keeps the 1024-sample budget. The approximate arm is fine at 64x16.
#
# Usage (inside tmux, survives disconnects):
#   bash scripts/ours/qed/server_run_dcbg_repro.sh
set -uo pipefail

PROP=qed
STEPS=32            # upstream's SAMPLING_STEPS default for QM9
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

for f in "${MDLM_CKPT}" "${CLASS_UNIFORM}" "${CLASS_ABSORB}"; do
  if [ ! -s "${f}" ]; then echo "FATAL: missing ${f}"; exit 1; fi
done

# UDLM: the released HuggingFace checkpoint, not the locally trained UDLM that
# upstream's own eval targets -- see pdfs/mdlm/02_udlm_dcbg_fidelity.md.
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

# MDLM: our own 25k-step checkpoint (HuggingFace publishes no MDLM-QM9).
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

# Cheap arm first so the picture is complete early, then the expensive exact arm
# ordered with the two main-paper settings up front.
echo "######################## approx arm (fast) ########################"
for GAMMA in 1 2 3 5 10; do
  run_eval "mdlm_cbg_approx_gamma${GAMMA}" \
    $(mdlm_args "${APPROX_BATCH}" "${APPROX_BATCHES}") \
    $(cbg_args "${CLASS_ABSORB}" "${GAMMA}" True)
done
for GAMMA in 1 2 3 5 10; do
  run_eval "s1024_cbg_approx_gamma${GAMMA}" \
    $(udlm_args "${APPROX_BATCH}" "${APPROX_BATCHES}") \
    $(cbg_args "${CLASS_UNIFORM}" "${GAMMA}" True)
done

echo "################ exact arm: main-paper settings ################"
run_eval "mdlm_cbg_exact_gamma3" \
  $(mdlm_args "${EXACT_BATCH}" "${EXACT_BATCHES}") \
  $(cbg_args "${CLASS_ABSORB}" 3 False)
run_eval "s1024_cbg_exact_gamma10" \
  $(udlm_args "${EXACT_BATCH}" "${EXACT_BATCHES}") \
  $(cbg_args "${CLASS_UNIFORM}" 10 False)

echo "################ exact arm: rest of the curve ################"
for GAMMA in 1 2 4 5; do
  run_eval "mdlm_cbg_exact_gamma${GAMMA}" \
    $(mdlm_args "${EXACT_BATCH}" "${EXACT_BATCHES}") \
    $(cbg_args "${CLASS_ABSORB}" "${GAMMA}" False)
done
for GAMMA in 1 2 3 5; do
  run_eval "s1024_cbg_exact_gamma${GAMMA}" \
    $(udlm_args "${EXACT_BATCH}" "${EXACT_BATCHES}") \
    $(cbg_args "${CLASS_UNIFORM}" "${GAMMA}" False)
done

echo "######## Done ########"
ls -1 "${RESULTS}"/*cbg*.csv 2>/dev/null
