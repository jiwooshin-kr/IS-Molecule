#!/bin/bash
# Regenerates the results table while the 150-step sweeps run, and once more
# after both finish, so the aggregated table is current without anyone being
# connected. Companion to scripts/server_watch_and_tabulate.sh, which watches
# server_run_comparison.sh instead.
#
#   tmux new-session -d -s agg "bash scripts/server_watch_steps150.sh"
#
# Watches both 150-step sweeps:
#   server_run_steps150.sh         (GPU 0, exact + marginal N sweep)
#   server_run_steps150_lambda.sh  (GPU 1, lambda curve at N=100)
#
# Output: results/sweep_table.{md,tex,txt} plus a one-line-per-run summary in
# results/steps150_summary.txt, both inside results/ (which sync.sh excludes, so
# a local rsync cannot delete them).
set -uo pipefail

ROOT=/home/aailab/wp03052/Synthetic-Data/DLRT
source /home/aailab/wp03052/venvs/dlrt_env/bin/activate
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${ROOT}/guidance_eval"

SWEEPS='server_run_steps150.sh|server_run_steps150_lambda.sh'

tabulate_now() {
  python scripts/make_results_table.py "${ROOT}/results" "${ROOT}/results" \
    > "${ROOT}/results/sweep_table.txt" 2>&1
  # Compact per-run view of just the 150-step cells, for reading over ssh.
  {
    date '+generated %Y-%m-%d %H:%M'
    printf '%-52s %6s %6s %6s %7s %7s\n' TAG VALID UNIQ NOVEL QED NOVQED
    for csv in "${ROOT}"/results/s1024_st150_*.csv; do
      [ -s "${csv}" ] || continue
      python - "${csv}" <<'PY'
import sys, os, pandas as pd
p = sys.argv[1]
f = pd.read_csv(p)
g = f[f['Seed'] != -1]
if g.empty:
    raise SystemExit
r = g.iloc[-1]
n = int(r['Num Samples']); v = r['Valid'] * n
print('%-52s %6d %6d %6d %7.3f %7.3f' % (
    os.path.basename(p)[:-4], round(v), round(r['Unique'] * v),
    round(r['Novel'] * v), r['QED Mean'], r['Novel QED Mean']))
PY
    done
  } > "${ROOT}/results/steps150_summary.txt" 2>&1
}

# Refresh every 5 minutes for as long as either sweep is alive.
while pgrep -f "${SWEEPS}" > /dev/null; do
  tabulate_now
  command sleep 300
done

tabulate_now
echo "Both 150-step sweeps finished."
echo "Table:   ${ROOT}/results/sweep_table.{md,tex,txt}"
echo "Summary: ${ROOT}/results/steps150_summary.txt"
echo "Raw logs: ${ROOT}/results/_sweep_st150.out, _sweep_st150_lambda.out"
