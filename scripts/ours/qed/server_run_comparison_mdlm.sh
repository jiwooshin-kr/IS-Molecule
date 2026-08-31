#!/bin/bash
# MDLM (absorbing-state) counterpart of scripts/ours/qed/server_run_comparison.sh.
# Waits for scripts/ours/server_train_mdlm_qm9.sh and the absorbing-state D-CBG
# classifier, then runs both guidance mechanisms on the trained MDLM.
#
# MDLM is the interesting second axis for two reasons. Its posterior normalizer
# is 1 - alpha_t whatever x_0 is, so where the normalization happens relative to
# the aggregation over the N candidates does not matter here -- the distinction
# that makes `marginal` the only faithful form under uniform diffusion collapses
# in absorbing state. (This is why the retired `aggregate_x0` mode gave results
# identical to `marginal` on MDLM while being biased on UDLM, and why the
# mdlm_* CSVs produced under that name remain valid.)
#
# Second, its posterior has no uniform floor, so a token that none of the N
# candidates proposed gets *exactly* zero probability rather than the small floor
# uniform diffusion leaves -- MDLM is where the atomic truncation of x_theta's
# ~1/N tail actually bites.
#
# Usage (inside tmux):
#   bash scripts/ours/qed/server_run_comparison_mdlm.sh
set -uo pipefail

PROP=qed
BATCH_SIZE=64
BATCHES=16          # 1024 samples
STEPS=32
SEED=1
WORKERS=32

ROOT=/home/aailab/wp03052/Synthetic-Data/Molecule
MODEL_CKPT="${ROOT}/outputs/qm9/mdlm_no-guidance/checkpoints/best.ckpt"
CLASS_CKPT="${ROOT}/outputs/qm9/classifier/${PROP}_absorbing_state_T-0/checkpoints/best.ckpt"

source /home/aailab/wp03052/venvs/dlrt_env/bin/activate
cd "${ROOT}"
export HF_HOME="${ROOT}/.hf_cache"
export PYTHONPATH="${ROOT}:${ROOT}/guidance_eval:${HF_HOME}/modules"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export WANDB_DIR="${ROOT}"
mkdir -p "${ROOT}/results/qed"

echo "Waiting for MDLM checkpoint and absorbing-state classifier ..."
while [ ! -s "${MODEL_CKPT}" ] || [ ! -s "${CLASS_CKPT}" ] \
      || pgrep -f "qm9_mdlm_no-guidance" > /dev/null \
      || pgrep -f "mode=train_classifier" > /dev/null; do
  command sleep 120
done
echo "Both ready."

# MDLM: absorbing-state diffusion with the `subs` parameterization.
BASE_ARGS=(
  data=qm9 "data.label_col=${PROP}" "data.cache_dir=${ROOT}/.data_cache"
  diffusion=absorbing_state parameterization=subs T=0
  time_conditioning=False zero_recon_loss=False
  # The DiT builds a class-conditioning embedding whenever
  # `training.guidance` is non-null (models/dit.py:381), and it defaults to
  # non-null. MDLM was trained with it disabled, so the checkpoint has no
  # `cond_map` weights and loading fails unless we disable it here too.
  # DITClassifier ignores this flag, so D-CBG's classifier is unaffected.
  training.guidance=null
  backbone=dit model=small model.length=32
  "eval.checkpoint_path=${MODEL_CKPT}"
  "sampling.steps=${STEPS}" "sampling.batch_size=${BATCH_SIZE}"
  "sampling.num_sample_batches=${BATCHES}" sampling.use_cache=False
  eval.disable_ema=False "seed=${SEED}"
)

run_eval() {
  local tag="$1"; shift
  local csv="${ROOT}/results/qed/${tag}.csv"
  if [ -s "${csv}" ]; then echo "[skip] ${tag}"; return 0; fi
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
    echo "  FAILED -- see ${log}"; tail -4 "${log}"
  fi
}

OURS_ARGS=(
  guidance=ours "guidance.reward=${PROP}"
  guidance.mixture_sampling=marginal
  "guidance.num_reward_workers=${WORKERS}"
)

echo "######## MDLM unguided reference ########"
run_eval "mdlm_unguided" "${OURS_ARGS[@]}" \
  guidance.num_x0_samples=2 guidance.lambda_=0.0 \
  guidance.t_min=2.0 guidance.t_max=2.0

echo "######## MDLM + ours: (N, lambda) ########"
for N in 100 300 500; do
  for LAM in 1000 5000; do
    run_eval "mdlm_ours_N${N}_lam${LAM}_win0.0-1.0" "${OURS_ARGS[@]}" \
      "guidance.num_x0_samples=${N}" "guidance.lambda_=${LAM}" \
      guidance.t_min=0.0 guidance.t_max=1.0
  done
done

echo "######## MDLM + ours: time window ########"
for WIN in "0.0 0.75" "0.0 0.5" "0.0 0.25" "0.5 1.0"; do
  read -r TMIN TMAX <<< "${WIN}"
  run_eval "mdlm_ours_N500_lam1000_win${TMIN}-${TMAX}" "${OURS_ARGS[@]}" \
    guidance.num_x0_samples=500 guidance.lambda_=1000 \
    "guidance.t_min=${TMIN}" "guidance.t_max=${TMAX}"
done

# The support-floor sweep was removed with `guidance.support_floor`. Its result
# is kept in results/mdlm_ours_N500_lam1000_floor*.csv: 0.5122 QED at floor 0
# against 0.5108 / 0.5061 at 0.01 / 0.1, so the floor did not help here either.

echo "######## MDLM + D-CBG baseline ########"
CBG_ARGS=(
  guidance=cbg guidance.condition=1
  classifier_model=tiny-classifier classifier_backbone=dit
  "guidance.classifier_checkpoint_path=${CLASS_CKPT}"
)
for GAMMA in 1 2 5; do
  run_eval "mdlm_cbg_approx_gamma${GAMMA}" \
    "${CBG_ARGS[@]}" "guidance.gamma=${GAMMA}" guidance.use_approx=True
done

echo "######## Done ########"
ls -1 "${ROOT}"/results/mdlm_*.csv 2>/dev/null
