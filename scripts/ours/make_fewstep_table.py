#!/usr/bin/env python3
"""Per-cell mean and sd over seeds for the few-step sweep.

One row per (N, k, lambda) and per metric: `marginal` mean+-sd over seeds,
`edlm` mean+-sd over seeds, and the *paired* difference, computed per seed and
then averaged. Nothing is pooled across cells -- each setting stands alone.

The paired difference is not "mixing settings": it is the comparison itself,
taken at a fixed (N, k, lambda). Pairing per seed is what makes it paired.

Counts are reported per run (2048 samples by default), not normalised, so do not
put runs of different sample counts in the same directory: `novel` is
`set(valid) - set(train)`, which saturates with pool size.

Usage:
  python3 scripts/ours/make_fewstep_table.py                     # print + write csv
  python3 scripts/ours/make_fewstep_table.py --bars 0.6 0.65 0.7
  python3 scripts/ours/make_fewstep_table.py --results <dir> --out <csv>
"""
import argparse
import collections
import csv
import json
import math
import os
import re
import statistics as st

TAG_RE = re.compile(
  r'^mdlm_fs_k(?P<k>\d+)_N(?P<N>\d+)_lam(?P<lam>[0-9.]+)'
  r'_(?P<mode>marginal|edlm)_s(?P<seed>\d+)$')

DEFAULT_RESULTS = 'results/qed/fewstep'
DEFAULT_OUT = 'pdfs/QED/fewstep/notes/fewstep_by_seed.csv'


def read_run(results, tag, bars):
  """Metrics for one run, or None if either file is missing/unreadable."""
  csv_path = os.path.join(results, tag + '.csv')
  json_path = os.path.join(results, tag + '_samples.json')
  if not (os.path.getsize(csv_path) if os.path.exists(csv_path) else 0):
    return None
  with open(csv_path) as fh:
    rows = list(csv.DictReader(fh))
  if not rows:
    return None
  # Row 0 is the QM9 reference row the eval script prepends; the run is last.
  row = rows[-1]
  n_samples = float(row['Num Samples'])
  out = {
    'valid': float(row['Valid']) * n_samples,
    'novel_qed_mean': float(row['Novel QED Mean']),
  }
  try:
    with open(json_path) as fh:
      q = json.load(fh)['qed_novel']
  except (OSError, ValueError, KeyError):
    return None
  out['novel'] = float(len(q))
  out['max'] = max(q) if q else 0.0
  for b in bars:
    out['hits@%g' % b] = float(sum(1 for x in q if x >= b))
  return out


def agg(values):
  """mean, sd, se over seeds. sd is None with a single seed."""
  n = len(values)
  mean = st.mean(values)
  sd = st.stdev(values) if n > 1 else None
  se = sd / math.sqrt(n) if sd is not None else None
  return mean, sd, se, n


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--results', default=DEFAULT_RESULTS)
  ap.add_argument('--out', default=DEFAULT_OUT)
  ap.add_argument('--bars', type=float, nargs='+', default=[0.6, 0.65])
  # The grid must be homogeneous: a cell aggregated over more seeds than its
  # neighbours is not comparable with them, and if those extra seeds were run
  # *because* the cell looked extreme, pooling them re-imports that selection.
  # Follow-up seeds belong in their own file, not in the grid.
  ap.add_argument('--seeds', default='1-5',
                  help="seeds to aggregate, e.g. '1-5' or '6-15' or 'all'")
  args = ap.parse_args()
  if args.seeds == 'all':
    keep = None
  elif '-' in args.seeds:
    lo, hi = (int(x) for x in args.seeds.split('-'))
    keep = set(range(lo, hi + 1))
  else:
    keep = {int(x) for x in args.seeds.split(',')}

  runs = collections.defaultdict(dict)   # (N,k,lam) -> mode -> seed -> metrics
  for name in sorted(os.listdir(args.results)):
    if not name.endswith('.csv'):
      continue
    m = TAG_RE.match(name[:-4])
    if not m:
      continue
    seed = int(m['seed'])
    if keep is not None and seed not in keep:
      continue
    got = read_run(args.results, name[:-4], args.bars)
    if got is None:
      continue
    cell = (int(m['N']), int(m['k']), float(m['lam']))
    runs[cell].setdefault(m['mode'], {})[seed] = got

  metrics = ['valid', 'novel'] + ['hits@%g' % b for b in args.bars] \
            + ['novel_qed_mean', 'max']
  rows = []
  for cell in sorted(runs):
    N, k, lam = cell
    per_mode = runs[cell]
    for metric in metrics:
      rec = {'N': N, 'k': k, 'steps': 32 // k, 'lam': lam, 'metric': metric}
      for mode in ('marginal', 'edlm'):
        seeds = per_mode.get(mode, {})
        if seeds:
          mean, sd, se, n = agg([v[metric] for v in seeds.values()])
          rec[mode + '_mean'], rec[mode + '_sd'] = mean, sd
          rec[mode + '_n'] = n
      # Paired difference: same seed on both sides, then aggregate.
      shared = sorted(set(per_mode.get('marginal', {}))
                      & set(per_mode.get('edlm', {})))
      if shared:
        d = [per_mode['marginal'][s][metric] - per_mode['edlm'][s][metric]
             for s in shared]
        mean, sd, se, n = agg(d)
        rec['diff_mean'], rec['diff_sd'], rec['diff_se'], rec['diff_n'] = \
          mean, sd, se, n
      rows.append(rec)

  fields = ['N', 'k', 'steps', 'lam', 'metric',
            'marginal_mean', 'marginal_sd', 'marginal_n',
            'edlm_mean', 'edlm_sd', 'edlm_n',
            'diff_mean', 'diff_sd', 'diff_se', 'diff_n']
  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
  with open(args.out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    for r in rows:
      w.writerow({f: r.get(f, '') for f in fields})

  def fmt(v, wide):
    if v is None or v == '':
      return ' ' * wide
    return ('%*.4f' if abs(v) < 10 else '%*.1f') % (wide, v)

  for metric in metrics:
    print('\n=== %s : marginal +-sd | edlm +-sd | paired diff +-SE (over seeds) ==='
          % metric)
    print('%5s %3s %5s %7s | %-17s %-17s %-19s'
          % ('N', 'k', 'T', 'lam', 'marginal', 'edlm', 'diff'))
    for r in rows:
      if r['metric'] != metric:
        continue
      print('%5d %3d %5d %7g | %s+-%s %s+-%s %s+-%s (n=%s)' % (
        r['N'], r['k'], r['steps'], r['lam'],
        fmt(r.get('marginal_mean'), 8), fmt(r.get('marginal_sd'), 7),
        fmt(r.get('edlm_mean'), 8), fmt(r.get('edlm_sd'), 7),
        fmt(r.get('diff_mean'), 8), fmt(r.get('diff_se'), 7),
        r.get('diff_n', 0)))
  print('\nwrote %s (%d rows)' % (args.out, len(rows)))


if __name__ == '__main__':
  main()
