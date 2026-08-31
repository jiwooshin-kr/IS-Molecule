#!/bin/bash
# MDLM few-step generation: does `marginal` diverge from `edlm` once a step
# writes more than one token?
#
# The question
# -----------
# At k=1 unmasked position per step the two mixture samplers are the SAME Markov
# chain: `marginal` returns E_n[q(.|x_t,x_0^(n))] and `edlm` draws n* ~ Cat(w)
# then samples that component, and with one position updating the joint equals
# that position's marginal. So the whole difference lives in steps with k >= 2,
# where `marginal` draws each position independently from the reward-weighted
# candidate histogram (reachable set <= N^k) while `edlm` copies k tokens from
# one candidate (reachable set <= N). This sweep puts k on an axis.
#
# How k is fixed
# --------------
# No code change is needed. `position_selection=random_k` unmasks
# `k = round(n_masked * p_unmask)` positions, and for the loglinear schedule
# p_unmask = dt/t exactly, so with L = 32 and T steps:
#
#     step i:  n_masked = 32 - i*(32/T),  t = (T-i)/T,  k = 32/T  -- every step
#
# i.e. k = L/T deterministically. Hence STEPS = 32/k below. Positions are chosen
# uniformly at random, which is not a heuristic: in the base absorbing-state
# sampler every masked position has the same "stay masked" probability, so
# conditional on the count the position set is uniform over subsets. random_k is
# therefore the true kernel conditioned on the count. A confidence rule would
# also be unfair here -- under `edlm` q_xs at a masked position is a one-hot, so
# every position scores 1 and the ranking degenerates to random anyway.
#
# Two null controls are built in and both must come out at zero:
#   k = 1, any lambda : the same Markov chain (above).
#   lambda = 0, any k : weights uniform, and the candidates were drawn
#                       position-independently from x_theta, so copying k tokens
#                       from one candidate IS k independent draws from x_theta.
#
# Arm A settings throughout (`oversample=1`, `exclude_invalid=False`,
# `exact_uniform_step=False`) so every cell is comparable with the earlier
# lambda x N sweep. `exclude_invalid` is deliberately off: it was measured as
# exactly a no-op (B - A = 0 on every metric, 8 cells), and it saves no compute
# either, since `smiles_reward` returns `invalid_reward` right after a failed
# parse and never reaches QED.
#
# Sizing
# ------
# 2048 samples/run (BATCHES=32), 5 seeds. At N=300, lambda>=200 the counting
# noise on the hits@0.6 difference is ~8.3 per 1024 samples, so 5 x 2048 gives
# se ~2.6 per cell. That resolves a trend across k and N; it does NOT resolve a
# single cell. Read the sweep as a trend.
#
# Usage -- one lambda per card, four tmux sessions:
#   LAM_GRID=0    CUDA_VISIBLE_DEVICES=0 bash scripts/ours/server_run_fewstep.sh
#   LAM_GRID=20   CUDA_VISIBLE_DEVICES=1 bash scripts/ours/server_run_fewstep.sh
#   LAM_GRID=200  CUDA_VISIBLE_DEVICES=2 bash scripts/ours/server_run_fewstep.sh
#   LAM_GRID=1000 CUDA_VISIBLE_DEVICES=3 bash scripts/ours/server_run_fewstep.sh
set -uo pipefail

PROP="${PROP:-qed}"
MODEL=mdlm
LENGTH=32
BATCH_SIZE=64
BATCHES="${BATCHES:-32}"      # 2048 samples
WORKERS=32

# k = tokens unmasked per step; STEPS is derived as LENGTH/k.
read -r -a K_GRID    <<< "${K_GRID:-1 2 4 8}"
# N ordered by value, not ascending: N=300 is the operating point and answers the
# question on its own, N=1000 is 60 % of the total cost and goes last. Each N is
# a complete block -- do NOT read paired contrasts from a partial block. Reading
# the screening sweep at 36/80 cells once flipped the SIGN of a conclusion,
# because an N-ascending grid makes a partial read a small-N read.
read -r -a N_GRID    <<< "${N_GRID:-300 30 10 100 1000}"
read -r -a LAM_GRID  <<< "${LAM_GRID:-0 20 200 1000}"
read -r -a MODE_GRID <<< "${MODE_GRID:-marginal edlm}"
read -r -a SEED_GRID <<< "${SEED_GRID:-1 2 3 4 5}"

ROOT=/home/aailab/wp03052/Synthetic-Data/Molecule
RESULTS="${ROOT}/results/${PROP}/fewstep"
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

[ -s "${MDLM_CKPT}" ] || { echo "FATAL: missing ${MDLM_CKPT}" >&2; exit 1; }

# Wait on THIS card only, via nvidia-smi. A global `pgrep our_qm9_eval` wait is
# wrong here: four cards run four of these at once, so each would block on the
# other three forever. Off by default.
GPU="${CUDA_VISIBLE_DEVICES:-0}"
if [ "${WAIT_FOR_IDLE:-0}" -eq 1 ]; then
  echo "waiting for GPU ${GPU} ..."
  while [ "$(nvidia-smi -i "${GPU}" --query-compute-apps=pid \
             --format=csv,noheader 2>/dev/null | grep -c .)" -gt 0 ]; do sleep 120; done
fi

BASE_ARGS=(
  data=qm9 "data.label_col=${PROP}" "data.cache_dir=${ROOT}/.data_cache"
  diffusion=absorbing_state parameterization=subs T=0
  time_conditioning=False zero_recon_loss=False
  training.guidance=null
  backbone=dit model=small "model.length=${LENGTH}"
  "eval.checkpoint_path=${MDLM_CKPT}"
  eval.disable_ema=False
  "sampling.batch_size=${BATCH_SIZE}"
  "sampling.num_sample_batches=${BATCHES}" sampling.use_cache=False
)

run_eval() {
  local tag="$1"; shift
  local csv="${RESULTS}/${tag}.csv"
  if [ -s "${csv}" ]; then echo "[skip] ${tag}"; return 0; fi
  local log="${RESULTS}/${tag}.log"
  local start=${SECONDS}
  if python -u -W ignore guidance_eval/our_qm9_eval.py \
      "${BASE_ARGS[@]}" "$@" \
      "++eval.results_csv_path=${csv}" \
      "++eval.generated_samples_path=${RESULTS}/${tag}_samples.json" \
      > "${log}" 2>&1; then
    printf '  %-46s %6ss  %s\n' "${tag}" "$((SECONDS - start))" \
      "$(grep -o 'ESS=[0-9.]*' "${log}" | tail -1)"
  else
    echo "  FAILED ${tag} -- see ${log}"; tail -6 "${log}"
  fi
}

echo "######## fewstep: ${MODEL} x ${PROP} | k=[${K_GRID[*]}] N=[${N_GRID[*]}]"
echo "######## lambda=[${LAM_GRID[*]}] seeds=[${SEED_GRID[*]}] on GPU ${GPU} at $(date)"

# Loop order: N outermost (complete blocks, cheapest-first within the value
# ordering above), mode INNERMOST so that a run killed part-way still has both
# modes present for every cell it finished -- the two modes are the comparison,
# so they must move together.
for NN in "${N_GRID[@]}"; do
  echo "=============== N=${NN} ==============="
  for K in "${K_GRID[@]}"; do
    STEPS=$(( LENGTH / K ))
    if [ $(( STEPS * K )) -ne "${LENGTH}" ]; then
      echo "  [skip] k=${K} does not divide L=${LENGTH}"; continue
    fi
    for LAM in "${LAM_GRID[@]}"; do
      for SEED in "${SEED_GRID[@]}"; do
        for MODE in "${MODE_GRID[@]}"; do
          run_eval "${MODEL}_fs_k${K}_N${NN}_lam${LAM}_${MODE}_s${SEED}" \
            "seed=${SEED}" "sampling.steps=${STEPS}" \
            guidance=ours "guidance.reward=${PROP}" \
            "guidance.mixture_sampling=${MODE}" \
            "++guidance.position_selection=random_k" \
            "++guidance.oversample=1" \
            "++guidance.exclude_invalid=False" \
            "++guidance.exact_uniform_step=False" \
            "guidance.num_x0_samples=${NN}" "guidance.lambda_=${LAM}" \
            guidance.t_min=0.0 guidance.t_max=1.0 \
            guidance.position_t_min=0.0 guidance.position_t_max=1.0 \
            "guidance.num_reward_workers=${WORKERS}"
        done
      done
    done
  done
done
echo "######## Done at $(date): $(ls -1 "${RESULTS}"/${MODEL}_fs_*.csv 2>/dev/null | wc -l) csv"
