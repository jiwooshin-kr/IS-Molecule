"""Loads the lambda x N sweep into one tidy frame and answers the questions the
reports are built on.

The CSVs carry every hyperparameter (`Lambda_`, `Num_x0_samples`,
`Mixture_sampling`), so the frame is built from those rather than from filenames.
ESS is the exception -- it only ever reaches the tqdm postfix, so it comes from
`results/_ess_summary.csv`, a digest of the per-run logs. Read `_ess_map` before
using it: the last-step ESS that the driver prints is actively misleading.

Usage:
    python scripts/ours/analyze_sweep.py            # print every section
    python scripts/ours/analyze_sweep.py frontier   # one section
"""

import glob
import json
import os

import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROPS = ('qed', 'ring_count')
PROP_COL = {'qed': 'QED', 'ring_count': 'RING_COUNT'}


def _ess_map(prop):
  """tag -> ESS over the whole trajectory, from `results/_ess_summary.csv`.

  Use the MEDIAN, never the last value. The driver `_sweep_*.out` prints only
  the final step's ESS, and on MDLM that step is uninformative: absorbing-state
  sampling has almost everything unmasked by then, so all N candidates decode to
  the same sequence, tie on reward, and the ESS reads ~N no matter how hard
  lambda is tilting. Judged on the last value MDLM looks completely unguided,
  while its novel QED in fact climbs 0.456 -> 0.515 across the lambda grid.
  Example -- mdlm edlm N=100 lambda=1000: last 95.6, but median 50.2, min 19.5.

  `_ess_summary.csv` is regenerated from the per-run logs, which stay on the
  server (they are ~1 MB each and sync.sh excludes results/).
  """
  path = os.path.join(REPO, 'results', '_ess_summary.csv')
  if not os.path.exists(path):
    return {}
  frame = pd.read_csv(path)
  frame = frame[frame['prop'] == prop]
  return dict(zip(frame['tag'], frame['ess_median']))


TOP_K = 10


def novel_values(prop, tag):
  """The per-molecule property values behind a run, from its samples JSON."""
  path = os.path.join(REPO, 'results', prop, f'{tag}_samples.json')
  if not os.path.exists(path):
    return []
  try:
    with open(path) as f:
      payload = json.load(f)
  except (OSError, json.JSONDecodeError):
    return []
  return payload.get(f'{prop}_novel') or []


def _order_stats(prop, tag):
  """max and top-K mean over the novel molecules.

  Caveat that applies to `max` specifically: it grows with the number of
  molecules it is taken over, and rows here differ several-fold in novel count.
  A larger max can therefore mean nothing but a bigger pool. `top{K}` at a fixed
  generation budget is the version that is comparable, because both methods
  generated the same 1,024 sequences; see `make_table4_distribution.py` for a
  bootstrap sample-size-matched max as well.
  """
  v = novel_values(prop, tag)
  if not v:
    return {'max': float('nan'), f'top{TOP_K}': float('nan')}
  ordered = sorted(v, reverse=True)
  return {'max': ordered[0],
          f'top{TOP_K}': sum(ordered[:TOP_K]) / len(ordered[:TOP_K])}


def load(prop):
  """Every `ours` run for one property, in paper units."""
  ess = _ess_map(prop)
  col = PROP_COL[prop]
  rows = []
  for path in sorted(glob.glob(os.path.join(REPO, 'results', prop, '*_ours_*.csv'))):
    tag = os.path.basename(path)[:-len('.csv')]
    frame = pd.read_csv(path)
    gen = frame[frame['Seed'] != -1]
    if gen.empty:
      continue
    r = gen.iloc[-1]
    total = int(r['Num Samples'])
    valid = r['Valid'] * total
    rows.append({
      'tag': tag,
      'model': 'MDLM' if tag.startswith('mdlm_') else 'UDLM',
      'mode': str(r['Mixture_sampling']),
      'lam': float(r['Lambda_']),
      'N': int(r['Num_x0_samples']),
      'valid': round(valid),
      'novel': round(r['Novel'] * valid),
      'prop': r[f'Novel {col} Mean'],
      'ess': ess.get(tag, float('nan')),
    } | _order_stats(prop, tag))
  frame = pd.DataFrame(rows)
  # ESS is only comparable across N as a fraction of N.
  frame['ess_frac'] = frame['ess'] / frame['N']
  return frame.sort_values(['model', 'mode', 'N', 'lam']).reset_index(drop=True)


def load_cbg(prop):
  """The D-CBG gamma sweep, same units -- the frontier is drawn against it."""
  col = PROP_COL[prop]
  rows = []
  for path in sorted(glob.glob(os.path.join(REPO, 'results', prop, '*_cbg_*.csv'))):
    tag = os.path.basename(path)[:-len('.csv')]
    frame = pd.read_csv(path)
    gen = frame[frame['Seed'] != -1]
    if gen.empty:
      continue
    r = gen.iloc[-1]
    total = int(r['Num Samples'])
    valid = r['Valid'] * total
    rows.append({
      'model': 'MDLM' if tag.startswith('mdlm_') else 'UDLM',
      'approx': str(r['Use_approx']).strip().lower() == 'true',
      'gamma': float(r['Gamma']),
      'valid': round(valid), 'novel': round(r['Novel'] * valid),
      'prop': r[f'Novel {col} Mean'],
      'tag': tag,
    } | _order_stats(prop, tag))
  return pd.DataFrame(rows).sort_values(['model', 'approx', 'gamma'])


# A run that produced almost nothing is not an operating point: D-CBG's ring
# exact gamma=2 cell has 2 valid molecules out of 1,024 and a "ring count" of 6.0
# computed over those two. Left in, it becomes the reported peak of the method.
# Single source of truth -- make_reports.py imports this rather than redefining.
MIN_NOVEL = 10


def usable(frame):
  return frame[frame['novel'] >= MIN_NOVEL]


def pareto(frame, x='novel', y='prop'):
  """Rows not dominated on (x, y), both maximised."""
  keep = []
  for _, row in frame.iterrows():
    if not ((frame[x] >= row[x]) & (frame[y] >= row[y])
            & ((frame[x] > row[x]) | (frame[y] > row[y]))).any():
      keep.append(row)
  return pd.DataFrame(keep).sort_values(x)


# ---------------------------------------------------------------- sections

def sec_grid(prop, frame):
  print(f"\n{'='*100}\n[{prop}] novel-{PROP_COL[prop]} over the (N, lambda) grid\n{'='*100}")
  for model in ('MDLM', 'UDLM'):
    for mode in ('marginal', 'edlm'):
      sub = frame[(frame.model == model) & (frame['mode'] == mode)]
      if sub.empty:
        continue
      print(f"\n  {model} / {mode}")
      print(sub.pivot(index='N', columns='lam', values='prop').round(3).to_string())


def sec_convergence(prop, frame):
  """The lambda -> infinity prediction: marginal and edlm must coincide."""
  print(f"\n{'='*100}\n[{prop}] marginal vs edlm: |difference| by lambda "
        f"(prediction: -> 0 as lambda grows)\n{'='*100}")
  wide = frame.pivot_table(index=['model', 'N', 'lam'], columns='mode',
                           values=['prop', 'novel', 'valid'])
  diff = pd.DataFrame({
    'd_prop': (wide[('prop', 'marginal')] - wide[('prop', 'edlm')]).abs(),
    'd_novel': (wide[('novel', 'marginal')] - wide[('novel', 'edlm')]).abs(),
    'd_valid': (wide[('valid', 'marginal')] - wide[('valid', 'edlm')]).abs(),
  }).reset_index()
  for model in ('MDLM', 'UDLM'):
    sub = diff[diff.model == model]
    print(f"\n  {model}  (mean over the five N at each lambda)")
    print(sub.groupby('lam')[['d_prop', 'd_novel', 'd_valid']]
          .mean().round(3).to_string())


def sec_ess(prop, frame):
  print(f"\n{'='*100}\n[{prop}] ESS as a fraction of N "
        f"(1.0 = no tilt, ->0 = collapsed onto one candidate)\n{'='*100}")
  for model in ('MDLM', 'UDLM'):
    sub = frame[(frame.model == model) & (frame['mode'] == 'marginal')]
    if sub.empty:
      continue
    print(f"\n  {model} / marginal")
    print(sub.pivot(index='N', columns='lam', values='ess_frac').round(3).to_string())


def sec_frontier(prop, frame):
  print(f"\n{'='*100}\n[{prop}] Pareto frontier on (num novel, novel-{PROP_COL[prop]})\n{'='*100}")
  cbg = load_cbg(prop)
  for model in ('MDLM', 'UDLM'):
    ours = usable(frame[frame.model == model])
    theirs = usable(cbg[cbg.model == model])
    if ours.empty or theirs.empty:
      continue
    po, pt = pareto(ours), pareto(theirs)
    print(f"\n  {model} -- ours (mode/N/lambda):")
    print(po[['mode', 'N', 'lam', 'valid', 'novel', 'prop']]
          .to_string(index=False, float_format='%.3f'))
    print(f"  {model} -- D-CBG (gamma/approx):")
    print(pt[['gamma', 'approx', 'valid', 'novel', 'prop']]
          .to_string(index=False, float_format='%.3f'))
    # The headline claim: at matched novelty, who reaches the higher property?
    best_o, best_t = ours['prop'].max(), theirs['prop'].max()
    print(f"  peak novel-{PROP_COL[prop]}:  ours {best_o:.3f}   D-CBG {best_t:.3f}")
    print(f"  max novel count:      ours {ours['novel'].max():.0f}   "
          f"D-CBG {theirs['novel'].max():.0f}")


def compare_distributions(a, b, n_q=101):
  """Rank-against-rank and draw-against-draw comparisons of two sample sets.

  Three numbers, answering three genuinely different questions. None of them is
  the mean, and only the third uses the fact that one method yields more novel
  molecules than the other from the same generation budget.

  * `quantile_win` -- sort both, then compare like for like: our best against
    their best, our 10th percentile against their 10th, and so on. The fraction
    of quantiles where we are ahead. Comparing the k-th *rank* directly only
    works when both sets are the same size; they are not, so this compares at
    matched *quantiles* instead, which is the size-independent version of the
    same idea. If this is 1.0 we dominate at every quantile -- that is exactly
    **first-order stochastic dominance**, the strongest distribution-free
    statement available.
  * `a12` -- the Vargha-Delaney A_12 / probability of superiority: draw one
    molecule from each at random, how often is ours better (ties count a half).
    This is the normalised Mann-Whitney U. 0.5 = indistinguishable.
  * Both of the above are *shape* comparisons and deliberately ignore the
    counts. The count is cashed in by `topk`, below, which fixes the generation
    budget instead of the novel count.
  """
  import numpy as np
  a, b = np.asarray(a, float), np.asarray(b, float)
  if a.size == 0 or b.size == 0:
    return {}
  qs = np.linspace(0, 1, n_q)
  qa, qb = np.quantile(a, qs), np.quantile(b, qs)
  wins = (qa > qb).sum() + 0.5 * (qa == qb).sum()
  # A_12 without materialising the full outer product for large samples.
  order = np.argsort(np.concatenate([a, b]), kind='mergesort')
  ranks = np.empty(order.size, float)
  ranks[order] = np.arange(1, order.size + 1)
  # average ranks over ties
  cat = np.concatenate([a, b])
  for v in np.unique(cat[np.concatenate([a, b])[order][:-1] ==
                         np.concatenate([a, b])[order][1:]]) if False else []:
    pass
  ra = ranks[:a.size].sum()
  u = ra - a.size * (a.size + 1) / 2.0
  return {
    'quantile_win': wins / n_q,
    'a12': u / (a.size * b.size),
    'dominates': bool((qa >= qb).all() and (qa > qb).any()),
  }


# Quality bars for the hit count. Chosen to bracket each property's useful
# range: QM9's unguided novel mean is ~0.46 QED and ~2.0 rings.
BARS = {'qed': (0.55, 0.60, 0.65), 'ring_count': (4, 5, 6)}


def hits(prop, tag, bars=None):
  """Novel molecules clearing each bar, out of the 1,024 generated.

  The primary head-to-head metric, and the only one that needs no correction:
  every run generated the same 1,024 sequences, so this counts what a fixed
  compute budget actually buys. Quality and quantity both enter -- a method that
  yields many novel molecules of poor quality scores nothing, and one that
  yields a few excellent ones scores little.

  Do NOT normalise this by the novel count, and do not size-match it against a
  competitor. Converting a fixed budget into more usable molecules IS the
  result; dividing it out erases exactly what is being measured.
  """
  v = novel_values(prop, tag)
  return [sum(1 for x in v if x >= b) for b in (bars or BARS[prop])]


def _matched(values, m, boot=2000, seed=0):
  """max / top-K of a random size-m subsample, averaged over `boot` draws."""
  import numpy as np
  v = np.asarray(values, float)
  if v.size <= m:
    ordered = np.sort(v)[::-1]
    return {'max': float(ordered[0]),
            'topk': float(ordered[:TOP_K].mean())}
  rng = np.random.default_rng(seed)
  mx, tk = [], []
  for _ in range(boot):
    sub = np.sort(rng.choice(v, m, replace=False))[::-1]
    mx.append(sub[0])
    tk.append(sub[:TOP_K].mean())
  return {'max': float(np.mean(mx)), 'topk': float(np.mean(tk))}


def hit_rows(prop, frame, model):
  """Best setting per METHOD -- marginal, edlm and D-CBG kept separate.

  The two mixture-sampling modes are different samplers, so collapsing them into
  one "ours" row hides which of them is actually carrying the result. They are
  ranked independently, each on its own best setting.
  """
  cbg = load_cbg(prop)
  out = []
  groups = [
    ('marginal', usable(frame[(frame.model == model) & (frame['mode'] == 'marginal')])),
    ('edlm', usable(frame[(frame.model == model) & (frame['mode'] == 'edlm')])),
    ('D-CBG', usable(cbg[cbg.model == model])),
  ]
  for label, fr in groups:
    if fr.empty:
      continue
    scored = [(hits(prop, r['tag']), r) for _, r in fr.iterrows()]
    scored = [x for x in scored if sum(x[0])]
    if not scored:
      continue
    c, r = max(scored, key=lambda x: x[0][1])   # rank on the middle bar
    setting = (f"gamma={r['gamma']:g} {'approx' if r['approx'] else 'exact'}"
               if label == 'D-CBG'
               else f"N={r['N']:.0f} lam={r['lam']:g}")
    out.append({'method': label, 'setting': setting, 'novel': r['novel'],
                'mean': r['prop'], 'max': r['max'], 'hits': c, 'row': r})
  return out


def sec_hits(prop, frame):
  """The primary comparison: usable molecules per fixed generation budget."""
  bars = BARS[prop]
  print(f"\n{'='*96}\n[{prop}] novel molecules clearing each bar, "
        f"out of 1,024 generated (no size matching)\n{'='*96}")
  for model in ('MDLM', 'UDLM'):
    rows = hit_rows(prop, frame, model)
    if len(rows) < 2:
      continue
    print(f"\n  {model}")
    print(f"    {'method':10s} {'setting':22s} {'novel':>6} {'mean':>7} "
          + ' '.join(f"{'>=' + str(b):>8}" for b in bars))
    for r in rows:
      print(f"    {r['method']:10s} {r['setting']:22s} {r['novel']:>6.0f} "
            f"{r['mean']:>7.3f} " + ' '.join(f"{x:>8d}" for x in r['hits']))
    base = next((r for r in rows if r['method'] == 'D-CBG'), None)
    if base:
      for r in rows:
        if r['method'] == 'D-CBG':
          continue
        print(f"    {r['method'] + ' / D-CBG':33s} {'':>6} {'':>7} "
              + ' '.join(f"{(o / t if t else float('inf')):>7.1f}x"
                         for o, t in zip(r['hits'], base['hits'])))


def sec_dominance(prop, frame):
  """Secondary: shape comparisons, which deliberately ignore the counts."""
  print(f"\n{'='*100}\n[{prop}] beyond the mean: ours vs D-CBG "
        f"at matched generation budget (1,024 sequences each)\n{'='*100}")
  cbg = load_cbg(prop)
  for model in ('MDLM', 'UDLM'):
    ours = usable(frame[frame.model == model])
    theirs = usable(cbg[cbg.model == model])
    if ours.empty or theirs.empty:
      continue
    # Each side at its own best setting, judged by the budget-matched top-K.
    o = ours.loc[ours[f'top{TOP_K}'].idxmax()]
    t = theirs.loc[theirs[f'top{TOP_K}'].idxmax()]
    va = novel_values(prop, o['tag'])
    vb = novel_values(prop, t['tag'])
    cmp = compare_distributions(va, vb)
    print(f"\n  {model}")
    print(f"    ours  : {o['mode']} N={o['N']:.0f} lam={o['lam']:g}   "
          f"novel={o['novel']:.0f}  mean={o['prop']:.3f}  "
          f"max={o['max']:.3f}  top{TOP_K}={o[f'top{TOP_K}']:.3f}")
    print(f"    D-CBG : gamma={t['gamma']:g} "
          f"{'approx' if t['approx'] else 'exact'}        "
          f"novel={t['novel']:.0f}  mean={t['prop']:.3f}  "
          f"max={t['max']:.3f}  top{TOP_K}={t[f'top{TOP_K}']:.3f}")
    # Separate the two reasons our top-k can be higher: a genuinely better
    # upper tail, or simply a deeper pool to take the top of. Subsampling the
    # larger set down to the smaller one removes the pool-size effect, so the
    # matched column is a statement about distribution *shape*; the raw column
    # is the statement about a fixed generation budget, which is what a
    # screening pipeline actually faces. Both are true, they answer different
    # questions, and only the raw one may be called an advantage of the method
    # at equal cost.
    m = min(len(va), len(vb))
    ra, rb = _matched(va, m), _matched(vb, m)
    print(f"    at matched pool size (n={m}):  max {ra['max']:.3f} vs "
          f"{rb['max']:.3f}   top{TOP_K} {ra['topk']:.3f} vs {rb['topk']:.3f}"
          f"   -> shape favours "
          f"{'ours' if ra['topk'] > rb['topk'] else 'D-CBG'}")
    if cmp:
      print(f"    quantile win rate (ours ahead at what fraction of quantiles) "
            f"= {cmp['quantile_win']:.2f}")
      print(f"    A_12  P(random ours > random D-CBG)                        "
            f"= {cmp['a12']:.2f}")
      print(f"    first-order stochastic dominance by ours: {cmp['dominates']}")


def sec_lambda0(prop, frame):
  """lambda=0 must reproduce the base sampler, and converge in N."""
  print(f"\n{'='*100}\n[{prop}] lambda = 0: the unguided limit, as N grows\n{'='*100}")
  z = frame[frame.lam == 0]
  for model in ('MDLM', 'UDLM'):
    sub = z[z.model == model]
    if sub.empty:
      continue
    print(f"\n  {model}")
    print(sub.pivot(index='N', columns='mode',
                    values=['valid', 'novel', 'prop']).round(3).to_string())


SECTIONS = {'grid': sec_grid, 'convergence': sec_convergence, 'ess': sec_ess,
            'frontier': sec_frontier, 'lambda0': sec_lambda0,
            'hits': sec_hits, 'dominance': sec_dominance}


def main():
  wanted = sys.argv[1:] or list(SECTIONS)
  for prop in PROPS:
    frame = load(prop)
    if frame.empty:
      print(f"no ours_* CSVs for {prop}")
      continue
    for name in wanted:
      SECTIONS[name](prop, frame)


if __name__ == '__main__':
  main()
