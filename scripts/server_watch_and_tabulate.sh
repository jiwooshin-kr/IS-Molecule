#!/bin/bash
# Regenerates the results table while the comparison sweep runs, and once more
# after it finishes. Launch alongside scripts/server_run_comparison.sh so the
# aggregated table is always current without anyone having to be connected:
#
#   tmux new-session -d -s agg "bash scripts/server_watch_and_tabulate.sh"
#
# Output: results/sweep_table.md and results/sweep_table.tex (rsync these back).
set -uo pipefail

ROOT=/home/aailab/wp03052/Synthetic-Data/DLRT
source /home/aailab/wp03052/venvs/dlrt_env/bin/activate
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${ROOT}/guidance_eval"

tabulate_now() {
  python scripts/make_results_table.py "${ROOT}/results" "${ROOT}/results" \
    > "${ROOT}/results/sweep_table.txt" 2>&1
}

# Refresh every 5 minutes for as long as the sweep is alive.
while pgrep -f server_run_comparison.sh > /dev/null; do
  tabulate_now
  command sleep 300
done

# Final pass after the sweep exits.
tabulate_now
echo "Sweep finished; final table in ${ROOT}/results/sweep_table.{md,tex,txt}"
