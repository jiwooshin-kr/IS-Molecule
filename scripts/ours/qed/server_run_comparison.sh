#!/bin/bash
# Runs the comparison on the server: our reward-tilted posterior guidance
# against the D-CBG baseline, both on top of the same HuggingFace UDLM-QM9 base
# model so the only difference is the guidance mechanism.
#
# Every configuration uses the paper's 1024-sample budget. That is affordable
# because the reward path was optimized: a table-join decoder replaced
# tokenizer.batch_decode (~560x faster) and the RDKit calls are deduplicated and
# spread over a process pool, which took N=300 from minutes to ~18 s per batch.
#
# Usage (inside tmux, survives disconnects):
#   bash scripts/ours/qed/server_run_comparison.sh
#
# One CSV per configuration lands in results/. Aggregate with
#   python scripts/ours/make_results_table.py results/qed
set -uo pipefail

PROP=qed
BATCH_SIZE=64
BATCHES=16          # 1024 samples, matching the paper's Table 5
STEPS=32
SEED=1
WORKERS=32

ROOT=/home/aailab/wp03052/Synthetic-Data/Molecule
CLASS_CKPT="${ROOT}/outputs/qm9/classifier/${PROP}_uniform_T-0/checkpoints/best.ckpt"

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

# run_eval <tag> <extra hydra args...>
# Hydra chdirs into its run directory, so output paths must be absolute.
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

echo "######## Phase 0: unguided reference ########"
# An unsatisfiable window means the guided branch never fires, so this is
# exactly the base UDLM sampler run through our code path.
run_eval "s1024_unguided" "${OURS_ARGS[@]}" \
  guidance.num_x0_samples=2 guidance.lambda_=0.0 \
  guidance.t_min=2.0 guidance.t_max=2.0

echo "######## Phase 1: (N, lambda) grid, guidance over all t ########"
# Importance sampling can only reweight among the N proposals drawn from
# p_theta, never invent a better x_0, so N sets the ceiling and lambda controls
# how sharply we move toward it. Both have to grow together. Small N is dropped
# here: the candidate histogram is an atomic measure, so N also fixes how much
# of x_theta's tail survives each step.
for N in 100 300 500; do
  for LAM in 200 1000 5000; do
    run_eval "s1024_ours_N${N}_lam${LAM}_win0.0-1.0" "${OURS_ARGS[@]}" \
      "guidance.num_x0_samples=${N}" "guidance.lambda_=${LAM}" \
      guidance.t_min=0.0 guidance.t_max=1.0
  done
done

# Phase 2 (support floor) was removed together with `guidance.support_floor`
# and the `aggregate_x0` branch it lived in. The sweep it produced is kept in
# results/ for the record: raising the floor moved QED monotonically the wrong
# way (0.5346 at floor 0 -> 0.5084 / 0.4921 / 0.4789 at 0.01 / 0.1 / 0.5) while
# validity climbed toward the unguided 97.85 %, i.e. the knob only dialled
# guidance off and was strictly worse than lowering lambda.

echo "######## Phase 3: guidance time window ########"
# t counts down from 1 (pure noise) to 0 (clean). t_max < 1 switches guidance
# off for the early high-noise steps, where the x_0 candidates are mostly
# unparseable; t_min > 0 switches it off for the late steps, where the sequence
# is already largely determined.
for N in 300 500; do
  for WIN in "0.0 0.75" "0.0 0.5" "0.0 0.25" "0.25 1.0" "0.5 1.0" "0.25 0.75"; do
    read -r TMIN TMAX <<< "${WIN}"
    run_eval "s1024_ours_N${N}_lam1000_win${TMIN}-${TMAX}" "${OURS_ARGS[@]}" \
      "guidance.num_x0_samples=${N}" guidance.lambda_=1000 \
      "guidance.t_min=${TMIN}" "guidance.t_max=${TMAX}"
  done
done

echo "######## Phase 4: D-CBG baseline ########"
echo "Waiting for ${CLASS_CKPT} ..."
while [ ! -s "${CLASS_CKPT}" ] || pgrep -f "mode=train_classifier" > /dev/null; do
  command sleep 60
done
echo "Classifier ready."

CBG_ARGS=(
  guidance=cbg guidance.condition=1
  classifier_model=tiny-classifier classifier_backbone=dit
  "guidance.classifier_checkpoint_path=${CLASS_CKPT}"
)
for GAMMA in 1 2 5; do
  run_eval "s1024_cbg_approx_gamma${GAMMA}" \
    "${CBG_ARGS[@]}" "guidance.gamma=${GAMMA}" guidance.use_approx=True
done
for GAMMA in 1 2 5; do
  run_eval "s1024_cbg_exact_gamma${GAMMA}" \
    "${CBG_ARGS[@]}" "guidance.gamma=${GAMMA}" guidance.use_approx=False
done

echo "######## Done ########"
ls -1 "${ROOT}"/results/*.csv
