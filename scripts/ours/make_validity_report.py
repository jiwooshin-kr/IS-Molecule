"""Generates pdfs/Validity/src/validity.tex.

Validity is a cross-cutting thread that fits neither property sweep: it is where
the two guidance mechanisms differ most, it is what drives the opposite curve
directions, and it is the subject of the oversampling experiment. This report
collects it.

    results/<prop>/*_ours_*.csv        the lambda x N sweep (validity grids)
    results/<prop>/*_cbg_*.csv         D-CBG, for the contrast
    results/qed/oversample/*.csv       the oversampling arms, once they land

The oversampling section renders only when those runs exist, so this can be built
before the experiment finishes and rebuilt afterwards.
"""

import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_sweep as A          # noqa: E402
import make_reports as M           # noqa: E402

BAR = {'qed': 0.60, 'ring_count': 5}
# The bar is a choice and the counts are steeply sensitive to it: on one MDLM run
# the novel molecules clearing QED 0.50 / 0.55 / 0.60 / 0.65 / 0.70 numbered
# 211 / 115 / 37 / 3 / 0. Reporting a single bar would hide whether a conclusion
# survives moving it, so every bar is carried through.
BARS = {'qed': (0.50, 0.55, 0.60, 0.65, 0.70), 'ring_count': (3, 4, 5, 6, 7)}
LAB = {'qed': 'QED', 'ring_count': 'ring count'}


def dial_table(prop):
  """What each method's strength dial does to validity."""
  frame, cbg = A.load(prop), A.load_cbg(prop)
  rows = []
  for model in ('MDLM', 'UDLM'):
    sub = frame[(frame.model == model) & (frame['mode'] == 'marginal')]
    if sub.empty:
      continue
    big = sub[sub.N == sub.N.max()].sort_values('lam')
    weak = float(big['valid'].iloc[0])
    strong = float(big['valid'].iloc[-1])
    rows.append([model, f'ours ($N={int(sub.N.max())}$)', f'{weak:.0f}',
                 f"{big['valid'].max():.0f}", f'{strong:.0f}',
                 r'\textbf{rises}' if strong > weak else 'falls'])
    t = cbg[cbg.model == model]
    for approx in (False, True):
      d = t[t.approx == approx].sort_values('gamma')
      if d.empty:
        continue
      w, s = float(d['valid'].iloc[0]), float(d['valid'].iloc[-1])
      rows.append([model, f"D-CBG {'approx.' if approx else 'exact'}",
                   f'{w:.0f}', f"{d['valid'].max():.0f}", f'{s:.0f}',
                   r'\textbf{collapses}' if s < 0.3 * w else
                   ('falls' if s < w else 'rises')])
  return rows


ARM_EU = r'A$_{\mathrm{eu}}$'
ARM_LABEL = {(1, False): ARM_EU, (1, True): 'B', (5, True): 'C'}
# A' is the earlier lambda x N sweep, reused to fill the cells arm A never ran.
# It is NOT arm A: it predates `exact_uniform_step`, so wherever the weights come
# out uniform it takes a Monte Carlo step where A substitutes the closed-form
# base kernel.
#
# The trigger is a *reward tie*, `spread < 1e-9` on the weights -- not a validity
# flag, so it fires in arm A even though arm A never masks on validity. Validity
# still reaches it, because `invalid_reward = 0` is what an unparseable candidate
# scores in every arm. Two distinct causes, measured at lambda=5000, N=300
# (median over the trajectory):
#   every candidate unparseable  -- MDLM 0.266, UDLM 0.156
#   ties among parseable ones    -- MDLM +0.047 (max +0.719), UDLM +0.000
#     i.e. duplicates decoding to the same molecule, or distinct molecules of
#     equal QED. This is the same duplicate effect behind the ESS floor, and it
#     is an MDLM phenomenon: on UDLM `unif` equals `allbad` in 1022 of 1024
#     logged steps.
# Total uniform rate: MDLM median 0.359, UDLM 0.156. Kept as its own column so
# the two configurations are never averaged together.
ARM_PRIME = 'A'
ARM_ORDER = [ARM_PRIME, ARM_EU, 'B', 'C']
ARM_DESC = {
  ARM_PRIME: r'\textbf{A} \ \texttt{over1}, unrestricted target --- the '
             r'method as every earlier result ran it',
  ARM_EU: r'\textbf{A$_{\mathrm{eu}}$} \ as A, plus the exact-uniform step '
          r'($\lambda = 5000$ only; an ablation, not the baseline)',
  'B': r'\textbf{B} \ \texttt{over1}, valid-restricted',
  'C': r'\textbf{C} \ \texttt{over5}, valid-restricted + screening',
}


def _hits_from_json(path, prop):
  """hits at every bar, read off the per-run sample dump."""
  out = {f'hits@{b:g}': np.nan for b in BARS[prop]}
  js = path[:-4] + '_samples.json'
  if not os.path.exists(js):
    return out
  try:
    vals = json.load(open(js)).get(f'{prop}_novel') or []
  except (OSError, json.JSONDecodeError):
    return out
  return {f'hits@{b:g}': sum(1 for x in vals if x >= b) for b in BARS[prop]}


def _row_from_csv(path, prop):
  """The (valid, novel, mean) triple every arm reports, from one results CSV."""
  frame = pd.read_csv(path)
  gen = frame[frame['Seed'] != -1]
  if gen.empty:
    return None
  r = gen.iloc[-1]
  total = int(r['Num Samples'])
  valid = r['Valid'] * total
  col = 'QED' if prop == 'qed' else 'RING_COUNT'
  return {'valid': round(valid), 'novel': round(r['Novel'] * valid),
          'prop': r[f'Novel {col} Mean']}


def load_prior_sweep(prop='qed'):
  """The pre-existing lambda x N sweep, restricted to the oversampling grid.

  Those runs already cover the (N, lambda) cells arm A skipped -- A was run at
  lambda=5000 only, which is the one lambda the earlier sweep lacks, so the two
  are exactly complementary and together fill the grid.
  """
  rows = []
  for path in glob.glob(os.path.join(A.REPO, 'results', prop,
                                     '*_ours_*_N*_lam*_win0.0-1.0.csv')):
    m = re.match(r'(?P<pfx>mdlm|s1024)_ours_(?P<mode>edlm|marginal)'
                 r'_N(?P<N>\d+)_lam(?P<lam>[\d.]+)_win0\.0-1\.0$',
                 os.path.basename(path)[:-4])
    if not m:
      continue
    base = _row_from_csv(path, prop)
    if base is None:
      continue
    rows.append({
      'model': 'MDLM' if m.group('pfx') == 'mdlm' else 'UDLM',
      'arm': ARM_PRIME, 'mode': m.group('mode'),
      'N': int(m.group('N')), 'lam': float(m.group('lam'))}
      | base | _hits_from_json(path, prop))
  frame = pd.DataFrame(rows)
  return frame


def load_oversample(prop='qed'):
  """One row per run. Nothing is pooled -- this sweep has a single seed."""
  rows = []
  for path in glob.glob(os.path.join(A.REPO, 'results', prop, 'oversample',
                                     '*.csv')):
    m = re.match(r'(?P<pfx>mdlm|s1024)_ov(?P<ov>\d+)(?P<ex>_ex)?'
                 r'_N(?P<N>\d+)_lam(?P<lam>[\d.]+)_s(?P<seed>\d+)'
                 r'(?P<mode>_edlm)?(?P<noeu>_noeu)?$',
                 os.path.basename(path)[:-4])
    if not m:
      continue
    if m.group('noeu'):
      # Run specifically to complete A' at lambda=5000: same configuration as
      # the earlier sweep (no exact-uniform substitution), same code as arm A.
      arm = ARM_PRIME
    else:
      arm = ARM_LABEL.get((int(m.group('ov')), m.group('ex') is not None))
    if arm is None:
      continue
    frame = pd.read_csv(path)
    gen = frame[frame['Seed'] != -1]
    if gen.empty:
      continue
    r = gen.iloc[-1]
    total = int(r['Num Samples'])
    valid = r['Valid'] * total
    hits = {f'hits@{b:g}': np.nan for b in BARS[prop]}
    js = path[:-4] + '_samples.json'
    if os.path.exists(js):
      try:
        vals = json.load(open(js)).get(f'{prop}_novel') or []
        hits = {f'hits@{b:g}': sum(1 for x in vals if x >= b)
                for b in BARS[prop]}
      except (OSError, json.JSONDecodeError):
        pass
    col = 'QED' if prop == 'qed' else 'RING_COUNT'
    rows.append({
      'model': 'MDLM' if m.group('pfx') == 'mdlm' else 'UDLM',
      'arm': arm,
      'mode': 'edlm' if m.group('mode') else 'marginal',
      'N': int(m.group('N')), 'lam': float(m.group('lam')),
      'valid': round(valid), 'novel': round(r['Novel'] * valid),
      'prop': r[f'Novel {col} Mean']} | hits)
  return pd.DataFrame(rows)


def load_walltime(prop='qed'):
  """Per-cell wall clock, scraped from the launcher logs into _walltime.csv."""
  # Lives under pdfs/, not results/: it is scraped from the launcher logs rather
  # than produced by a run, and `rsync --delete` against the server wiped it once
  # when it sat in results/qed/oversample/, where the server has no such file.
  path = os.path.join(A.REPO, 'pdfs', 'Validity', 'notes', 'walltime.csv')
  if prop != 'qed':
    return None
  if not os.path.exists(path):
    return None
  t = pd.read_csv(path)
  rec = []
  for tag, sec in zip(t.tag, t.sec):
    m = re.match(r'(mdlm|s1024)_ov(\d+)(_ex)?_N(\d+)_lam([\d.]+)_s\d+'
                 r'(_edlm)?$', str(tag))
    if not m:
      continue
    arm = ARM_LABEL.get((int(m.group(2)), m.group(3) is not None))
    if arm is None:
      continue
    rec.append({'model': 'MDLM' if m.group(1) == 'mdlm' else 'UDLM',
                'arm': arm, 'mode': 'edlm' if m.group(6) else 'marginal',
                'N': int(m.group(4)), 'lam': float(m.group(5)), 'sec': sec})
  return pd.DataFrame(rec)


def cost_tables(ov, prop='qed'):
  """Hits against wall clock, which is the comparison that decides the method.

  The paired C - B contrast holds N fixed, and N is not what a user spends --
  time is. Screening draws 5N candidates and must decode and parse all of them,
  so at matched N it is 1.2x to 7.5x slower. Ordering every configuration by
  measured cost asks the question the paired table cannot: given a time budget,
  is it better to screen, or to raise N?
  """
  t = load_walltime(prop)
  if t is None:
    return []
  d = ov.merge(t, on=['model', 'arm', 'mode', 'N', 'lam'], how='left')
  d = d[d.arm.isin(['B', 'C']) & d.sec.notna()]
  if d.empty:
    return []
  hcols = [f'hits@{bar:g}' for bar in BARS[prop]]
  out = []
  for model in ('MDLM', 'UDLM'):
    sub_ = d[d.model == model]
    if sub_.empty:
      continue
    g = sub_.groupby(['arm', 'N']).agg(
      cost=('sec', 'mean'), **{c: (c, 'max') for c in hcols}).reset_index()
    g = g.sort_values('cost')
    rows = [[r.arm, str(int(r.N)), f'{r.cost:.0f}']
            + [f'{r[c]:.0f}' for c in hcols] for _, r in g.iterrows()]
    out.append(M.table(
      ['arm', r'$N$', 'cost (s)'] + [f'$\\geq {bar:g}$' for bar in BARS[prop]],
      rows, 'lrr' + 'r' * len(hcols),
      f'{model}: hits against measured cost, ordered by cost. Each row takes '
      f'the best hits over $\\lambda$ and mixture mode at that $(\\mathrm{{arm}}, N)$, '
      f'i.e.\\ it assumes the tilt is tuned. Cost is the mean wall clock of the '
      f'8 cells behind the row.',
      f'tab:ov-cost-{model}',
      note=r'Read down the cost column and compare rows of similar cost. '
           r'\textbf{B} at $N=300$ costs 121\,s (MDLM) / 143\,s (UDLM); '
           r'\textbf{C} at $N=100$ costs 320\,s / 371\,s, i.e.\ 2.6$\times$ more, '
           r'and does not clearly beat it.',
      size=r'\footnotesize'))
  return out


def quality_tables(ov, prop='qed'):
  """valid / novel / novel-mean per (arm, N), the quantities hits is built from.

  hits is a single number that folds three things together -- how many parse, how
  many of those are new, and how good they are. When hits moves, this table says
  which of the three moved, and the three do not always move together: screening
  raised MDLM's novel count by 27 while raising UDLM's by only 4, yet both gained
  hits.
  """
  t = load_walltime(prop)
  hcol = f'hits@{BAR[prop]:g}'
  out = []
  for model in ('MDLM', 'UDLM'):
    sub_ = ov[ov.model == model]
    if sub_.empty:
      continue
    if t is not None:
      sub_ = sub_.merge(t, on=['model', 'arm', 'mode', 'N', 'lam'], how='left')
    rows = []
    for arm in ARM_ORDER:
      a = sub_[sub_.arm == arm]
      if a.empty:
        continue
      for n in sorted(a.N.unique()):
        g = a[a.N == n]
        best = g.loc[g[hcol].idxmax()] if g[hcol].notna().any() else None
        rows.append([
          arm, str(int(n)),
          f"{g['valid'].mean():.0f}", f"{g['novel'].mean():.0f}",
          f"{g['prop'].mean():.4f}",
          f"{g['valid'].max():.0f}", f"{g['novel'].max():.0f}",
          f"{g['prop'].max():.4f}",
          '--' if best is None else f'{best.lam:g}',
        ])
    out.append(M.table(
      ['arm', r'$N$', 'valid', 'novel', 'novel-mean',
       'valid', 'novel', 'novel-mean', r'best $\lambda$'],
      rows, 'lr rrr rrr r',
      f'{model}: the three quantities behind hits. First block is the mean over '
      f'the $\\lambda \\times$ mode cells at that $(\\mathrm{{arm}}, N)$, second block '
      f'the best single cell. \\emph{{best}} $\\lambda$ is the one maximising '
      f'hits at the {LAB[prop]} $\\geq {BAR[prop]:g}$ bar.',
      f'tab:ov-quality-{model}',
      note=r'\textbf{mean} over cells $\vert$ \textbf{max} over cells. '
           r'\emph{valid} and \emph{novel} are counts out of 1{,}024 generated; '
           r'\emph{novel-mean} is the mean property over the novel molecules, '
           r'i.e.\ the metric the paper reports. Note that novel-mean and the '
           r'novel count trade off against each other, so neither alone ranks '
           r'the arms.',
      size=r'\footnotesize'))
  return out


def quality_by_lambda(ov, prop='qed', metric='prop', fmt='%.4f'):
  """The lambda-resolved form of quality_tables: rows (arm, N), columns lambda.

  quality_tables averages lambda away, which hides the shape the curves actually
  have. Laid out this way each arm's row block is directly plottable: x = N,
  one series per lambda, solid/dashed for the two mixture modes.
  """
  out = []
  lams = sorted(ov.lam.unique())
  for model in ('MDLM', 'UDLM'):
    sub_ = ov[ov.model == model]
    if sub_.empty:
      continue
    rows = []
    for arm in ARM_ORDER:
      a = sub_[sub_.arm == arm]
      if a.empty:
        continue
      for n in sorted(a.N.unique()):
        cells = []
        for mode in ('marginal', 'edlm'):
          for lam in lams:
            c = a[(a.N == n) & (a.lam == lam) & (a['mode'] == mode)]
            cells.append(fmt % c.iloc[0][metric]
                         if len(c) and pd.notna(c.iloc[0][metric]) else '--')
        rows.append([arm, str(int(n))] + cells)
    header = ['arm', r'$N$']
    for mode in ('marginal', 'edlm'):
      header += [f'{mode[:4]}.\\,{lam:g}' for lam in lams]
    out.append(M.table(
      header, rows, 'lr' + 'r' * (2 * len(lams)),
      f'{model}: {LAB[prop]} novel-mean per run, resolved by $\\lambda$. '
      f'Columns are (mixture mode) $\\times$ $\\lambda$; rows group by arm. '
      f'Read one arm block as a plot: $x = N$, one series per $\\lambda$, '
      f'\\texttt{{marginal}} against \\texttt{{edlm}}.',
      f'tab:ov-prop-lambda-{model}',
      note=r'Every cell is a single run of 1{,}024 sequences; nothing is '
           r'averaged. Arm A$_{\mathrm{eu}}$ ran at $\lambda = 5000$ only. The '
           r'earlier sweep behind arm A also covers $N \in \{500, 1000, 2000\}$ '
           r'and $\lambda \in \{0, 2, 5\}$, which are outside this grid and are '
           r'reported in \texttt{qed\_sweep.pdf}; the same numbers are exported '
           r'to \texttt{pdfs/Validity/notes/quality\_by\_lambda.csv} for plotting.',
      size=r'\scriptsize'))
  return out


def per_run_tables(ov, prop, metric, fmt):
  """N x lambda rows, (mode, arm) columns. Every cell is exactly one run."""
  out = []
  arms = [a for a in ARM_ORDER if a in set(ov['arm'])]
  for model in ('MDLM', 'UDLM'):
    sub = ov[ov.model == model]
    if sub.empty:
      continue
    header = [r'$N$', r'$\lambda$']
    for mode in ('marginal', 'edlm'):
      header += [f'{mode[:4]}.\,{a}' for a in arms]
    rows = []
    for n in sorted(sub.N.unique()):
      for lam in sorted(sub.lam.unique()):
        cells = []
        for mode in ('marginal', 'edlm'):
          for a in arms:
            hit = sub[(sub.N == n) & (sub.lam == lam)
                      & (sub['mode'] == mode) & (sub.arm == a)]
            cells.append(fmt % hit.iloc[0][metric]
                         if len(hit) and pd.notna(hit.iloc[0][metric]) else '--')
        rows.append([str(n), f'{lam:g}'] + cells)
    out.append(M.table(
      header, rows, 'rr' + 'r' * (2 * len(arms)),
      f'{model}: {metric} per run. Columns are '
      f'(mixture mode) $\\times$ (arm); every cell is a single run of 1{{,}}024 '
      f'sequences, with no averaging.',
      f'tab:ov-{prop}-{model}-{metric}', size=r'\scriptsize'))
  return out


def build():
  body = [M.PREAMBLE,
          r'\title{Validity\\'
          r'\large What the reward-tilted proposal does to parseability, '
          r'and whether screening for it helps}',
          r'\author{}\date{2026-08-19}',
          r'\begin{document}\maketitle',
          r'\section{Why validity is the interesting axis}', r"""
Our reward is evaluated on a \emph{clean} $x_0$ candidate and returns
\texttt{invalid\_reward} $= 0$ when RDKit cannot parse it. An unparseable
candidate is therefore treated as a property-zero one and suppressed by the
tilt: raising $\lambda$ buys \emph{parseability} before it buys property.

D-CBG has no such term. It perturbs the per-position logits with the gradient of
a noisy classifier, and nothing in that gradient prefers a parseable string.

This single difference explains two things reported elsewhere. It is why the
novel \emph{count} rises with $\lambda$ for us and falls with $\gamma$ for D-CBG
--- novel molecules are counted as a subset of valid ones, so whatever happens to
validity happens to the novel count. And it is why our curve in
(novel, property) space runs up and to the right while D-CBG's runs up and to
the left.

\textbf{One caveat, stated up front:} the effect is not symmetric across base
models, and the naive claim ``our method keeps validity high'' is false. See
Section~\ref{sec:asym}.
""",
          r'\section{What each dial does to validity}']
  for prop in ('qed', 'ring_count'):
    rows = dial_table(prop)
    if not rows:
      continue
    body.append(M.table(
      ['model', 'method', 'weakest dial', 'best', 'strongest dial', 'trend'],
      rows, 'll rrr l',
      f'{LAB[prop]}: valid molecules out of 1{{,}}024 as the strength dial is '
      f'turned up. Ours along $\\lambda$ at the largest $N$; D-CBG along '
      f'$\\gamma$.', f'tab:val-dial-{prop}'))

  body.append(r'\section{The asymmetry}\label{sec:asym}')
  body.append(r"""
On \textbf{MDLM} the contrast is as stark as the mechanism predicts: ours climbs
with $\lambda$ while D-CBG's exact arm collapses, reaching \emph{zero} valid
molecules at $\gamma \geq 6$ --- which is why the original paper does not report
that region. The collapse is not cosmetic: it caps how far D-CBG can push the
property at all on this base model.

On \textbf{UDLM} the claim fails. D-CBG's exact arm \emph{raises} validity as
$\gamma$ grows and is above ours at every setting, while ours peaks at moderate
$\lambda$ and then declines slightly. A plausible reason is that uniform
diffusion resamples every position at every step and so can repair a local
mistake later, whereas absorbing-state diffusion cannot revisit an unmasked
token --- but that is a hypothesis this report does not test.

So the defensible claim is narrower and about robustness rather than level:

\begin{quote}
\textbf{Our dial does not destroy validity.} It has no collapse regime, so the
property can be pushed as far as compute allows. D-CBG on MDLM has one, and it
bounds what that method can reach.
\end{quote}
""")

  ov = load_oversample('qed')
  prior = load_prior_sweep('qed')
  if not prior.empty and not ov.empty:
    # Only the cells the oversampling grid actually visits. The earlier sweep
    # also ran N up to 2000 and lambda down to 0; carrying those in would add
    # rows that every other arm leaves blank.
    prior = prior[prior.N.isin(sorted(ov.N.unique()))
                  & prior.lam.isin(sorted(ov.lam.unique()))]
    ov = pd.concat([ov, prior], ignore_index=True)
  body.append(r'\section{Does screening candidates for validity help?}')
  body.append(r"""
If the tilt spends its early range on validity, that range is being consumed by
something a cheap filter could do instead. Three arms separate the two effects,
all on QED with a single seed per cell:

\begin{itemize}
  \item[\textbf{A}] \texttt{oversample}$=1$, unrestricted target. The method as
        every earlier result ran it: draw $N$ candidates, score all of them, and
        give an unparseable one the reward \texttt{invalid\_reward} $=0$.
        Run at $\lambda = 5000$ only.
  \item[\textbf{A$'$}] the earlier $\lambda \times N$ sweep, the one behind
        \texttt{qed\_sweep.pdf}, reused here to fill the cells A never ran --- and
        it is exactly complementary, since that sweep covered
        $\lambda \leq 1000$ and A covers only $\lambda = 5000$.
        \textbf{It is not arm A.} Those runs predate
        \texttt{exact\_uniform\_step}, so wherever the importance weights come
        out uniform, A$'$ takes a Monte Carlo step where A substitutes the
        closed-form base kernel.

        The trigger is a \emph{reward tie} --- \texttt{spread} $< 10^{-9}$ on the
        weights --- and \textbf{not} a validity flag, which is why it fires in
        arm A even though arm A never masks on validity. Validity still reaches
        it, because \texttt{invalid\_reward} $=0$ is what an unparseable
        candidate scores in \emph{every} arm; masking the logits
        (\texttt{exclude\_invalid}) is a separate mechanism. Two causes,
        measured at $\lambda = 5000$, $N = 300$ as medians over the trajectory:
        every candidate unparseable (MDLM $0.266$, UDLM $0.156$), and ties among
        parseable candidates --- duplicates decoding to the same molecule, or
        distinct molecules of equal QED --- which add $0.047$ on MDLM (up to
        $0.719$) and essentially nothing on UDLM. Total uniform rate: median
        $0.359$ on MDLM against $0.156$ on UDLM. The duplicate half is the same
        effect behind the ESS floor, and it is an MDLM phenomenon. So A$'$ is
        reported as its own column and never averaged with A.
  \item[\textbf{B}] \texttt{oversample}$=1$, \emph{valid-restricted} target.
        Same candidates, but $q(x_0) = 0$ wherever RDKit cannot read $x_0$, so
        invalid candidates carry zero weight. Isolates restricting the target.
  \item[\textbf{C}] \texttt{oversample}$=5$, valid-restricted \emph{and}
        screened. Draw $5N$, keep $N$ preferring the parseable ones, score those.
        Isolates improving the proposal.
\end{itemize}

\textbf{B versus C is the effect of screening}, at the same target and the same
$N$. A versus B is the effect of restricting the target at all.

Why this is correct rather than a heuristic: restricting to valid molecules makes
the proposal $p_\theta(\cdot \mid \text{valid})$, and the resulting
normaliser $Z = P_\theta(\text{valid} \mid x_t)$ does not depend on $x_0$, so it
cancels under self-normalisation and Eq.~(8) is unchanged. Conditioning i.i.d.\
draws on an event gives i.i.d.\ draws from the conditional, and a uniformly
random subset of them preserves that --- so C is a valid estimator for the same
target as B, reached with $N$ usable candidates instead of
$N \times (\text{valid fraction})$, a fraction measured as low as $0.007$ early
in a UDLM trajectory.

The cost argument is measured, not assumed: parsing costs $0.042$\,ms against
$0.811$\,ms for parse-plus-QED, so the reward is $91\,\%$ of a full evaluation.
Screening therefore leaves the expensive operation almost untouched. Note the
budgets are matched on $N$, not on cost: C scores more \emph{valid} candidates
than B, so it makes more reward calls. A cost-matched comparison is available
post hoc against the $\lambda \times N$ sweep, which runs \texttt{over1} up to
$N = 2000$.

$\lambda$ starts at 20 rather than 0. At $\lambda = 0$ there is no importance
weighting to reduce the variance of, so the claim this experiment tests does not
apply there.
""")
  if ov.empty:
    body.append(r'\emph{No oversampling runs found yet.}')
  else:
    have = sorted(set(ov['arm']))
    body.append('Arms present in this build: '
                + ', '.join(ARM_DESC[a] for a in have) + '.\n')
    if 'C' not in have or len(ov[ov.arm == 'C']) < len(ov[ov.arm == 'B']):
      body.append(r"""
\emph{Arm C is still running; its columns are partly blank and will fill in on
the next rebuild.}
""")
    body += quality_tables(ov, 'qed')
    body += quality_by_lambda(ov, 'qed')
    # Tidy export so the curves can be plotted without re-deriving them.
    ov.sort_values(['model', 'arm', 'mode', 'lam', 'N']).to_csv(
      os.path.join(A.REPO, 'pdfs', 'Validity', 'notes',
                   'quality_by_lambda.csv'), index=False)
    for metric, fmt in ([(f'hits@{b:g}', '%.0f') for b in BARS['qed']]
                        + [('valid', '%.0f'), ('novel', '%.0f'),
                           ('prop', '%.4f')]):
      body += per_run_tables(ov, 'qed', metric, fmt)

    # paired B - A and C - B, within (model, mode, N, lambda)
    key = ['model', 'mode', 'N', 'lam']
    rows, hrows = [], []
    hcols = [f'hits@{bar:g}' for bar in BARS['qed']]
    # Each contrast moves exactly one flag: A_eu -> B adds `exclude_invalid`
    # (both already carry the exact-uniform step), B -> C adds screening. A_eu
    # rather than A is the left side precisely so the exact-uniform flag is held
    # fixed; the flag's own effect is isolated separately, in tab:ov-exactunif.
    for a, b, what in ((ARM_EU, 'B', 'restricting the target'),
                       ('B', 'C', r'\textbf{screening}'),
                       (ARM_PRIME, 'C', r'\emph{end-to-end}')):
      if a not in have or b not in have:
        continue
      x = ov[ov.arm == a].set_index(key)
      y = ov[ov.arm == b].set_index(key)
      for model in ('MDLM', 'UDLM'):
        cells, hcells = [], []
        # Two tables rather than one twelve-column one: the quality metrics each
        # carry their own unit, while the hits columns are one metric read at
        # five bars and belong together.
        cols = [('valid', '%+.1f'), ('novel', '%+.1f'), ('prop', '%+.4f')]
        cols += [(f'hits@{bar:g}', '%+.1f') for bar in BARS['qed']]
        for col, f in cols:
          xi = x[x.index.get_level_values('model') == model][col].astype(float)
          yi = y[y.index.get_level_values('model') == model][col].astype(float)
          d = (yi - xi).dropna()
          if len(d) < 2:
            (hcells if col.startswith('hits@') else cells).append('--')
            continue
          mu = d.mean()
          se = d.std(ddof=1) / np.sqrt(len(d))
          star = r'$^{*}$' if se and abs(mu / se) > 2.1 else ''
          txt = (f'{f % mu} $\\pm$ {se:.4f}{star}' if col == 'prop'
                 else f'{f % mu} $\\pm$ {se:.1f}{star}')
          (hcells if col.startswith('hits@') else cells).append(txt)
        rows.append([model, f'{b} $-$ {a}', what, str(len(d))] + cells)
        hrows.append([model, f'{b} $-$ {a}', what, str(len(d))] + hcells)
    if rows:
      body.append(M.table(
        ['model', 'contrast', 'isolates', 'cells', '$\Delta$ valid',
         '$\Delta$ novel', '$\Delta$ QED'],
        rows, 'lll r rrr',
        'Paired within each $(N, \lambda)$ and mixture mode, which removes both '
        'and leaves only the contrast named.', 'tab:ov-paired',
        note=r'$^{*}$ marks $|t| > 2.1$. These are paired across grid cells, not '
             r'across seeds: this sweep has one seed per cell, so the spread '
             r'mixes cell-to-cell variation with run-to-run noise and the '
             r'$\pm$ is not a pure standard error.',
        size=r'\footnotesize'))
      body.append(M.table(
        ['model', 'contrast', 'isolates', 'cells']
        + [f'$\\geq {bar:g}$' for bar in BARS['qed']],
        hrows, 'lll r' + 'r' * len(BARS['qed']),
        'The same paired contrast read at every quality bar. A conclusion that '
        'holds at one bar and reverses at another is a property of the bar, not '
        'of the method, so all five are shown.', 'tab:ov-paired-bars',
        note=r'$\Delta$ hits: the change in the number of novel molecules '
             r'clearing each QED bar, out of 1{,}024 generated. The high bars '
             r'count in the single digits per run, so those columns are '
             r'noise-dominated and read as unresolved rather than null.',
        size=r'\footnotesize'))

      # Split the two substantive contrasts by mixture mode. Screening is a
      # change to the *proposal*, so it should not care which sampler draws from
      # the mixture -- if it helped in only one mode that would point at an
      # interaction, and the claim would have to be narrowed. Halving the cells
      # halves the power, so read the bars, not the stars, for consistency.
      mrows = []
      for a, b, what in (('B', 'C', 'screening'),
                         (ARM_PRIME, 'C', 'end-to-end')):
        if a not in have or b not in have:
          continue
        for mode in ('marginal', 'edlm'):
          xm = ov[(ov.arm == a) & (ov['mode'] == mode)].set_index(key)
          ym = ov[(ov.arm == b) & (ov['mode'] == mode)].set_index(key)
          for model in ('MDLM', 'UDLM'):
            xi = xm[xm.index.get_level_values('model') == model]
            yi = ym[ym.index.get_level_values('model') == model]
            idx = xi.index.intersection(yi.index)
            if len(idx) < 2:
              continue
            cells = []
            for col in hcols:
              dd = (yi.loc[idx, col].astype(float)
                    - xi.loc[idx, col].astype(float)).dropna()
              mu = dd.mean()
              se = dd.std(ddof=1) / np.sqrt(len(dd))
              star = r'$^{*}$' if se and abs(mu / se) > 2.1 else ''
              cells.append(f'{mu:+.1f} $\\pm$ {se:.1f}{star}')
            mrows.append([what, mode, model, str(len(idx))] + cells)
      if mrows:
        body.append(M.table(
          ['contrast', 'mode', 'model', 'cells']
          + [f'$\\geq {bar:g}$' for bar in BARS['qed']],
          mrows, 'lll r' + 'r' * len(hcols),
          r'$\Delta$ hits split by mixture mode. Screening changes the '
          r'\emph{proposal}, which is upstream of the choice of sampler, so it '
          r'should help in both --- and it does, at the 0.55 and 0.60 bars in '
          r'all four (mode, model) combinations.', 'tab:ov-bymode',
          note=r'20 cells per row against 40 in the pooled table, so the power is '
               r'halved and the outer bars lose significance in some '
               r'combinations while keeping their sign. The one place the modes '
               r'differ in kind is the 0.50 bar on UDLM: $+12.1$ under '
               r'\texttt{marginal} against $+5.2$ (n.s.) under \texttt{edlm}.',
          size=r'\footnotesize'))

      # Filling A' also buys a contrast we never had: B - A' at matched
       # (N, lambda, mode). The two differ in `exclude_invalid` and
       # `exact_uniform_step`, and the former is measured at exactly zero, so
       # what is left is the exact-uniform substitution on its own. Launch
       # settings are identical across the two sweeps (32 steps, seed 1,
       # 64 x 16 samples, 32 reward workers), so this is not a drift artifact --
       # though it is still two sweeps run five days apart, not one experiment.
      if ARM_PRIME in have and ARM_EU in have:
        # A - A' restricted to lambda=5000, where both exist: the two differ in
        # `exact_uniform_step` and in nothing else -- same code, same seed, same
        # launch settings, run the same morning. This replaces an earlier
        # B - A' contrast that differed in two flags across two sweeps five days
        # apart, and so could not attribute the effect.
        lam_eu = sorted(set(ov[ov.arm == ARM_EU].lam.unique())
                        & set(ov[ov.arm == ARM_PRIME].lam.unique()))
        xp = ov[(ov.arm == ARM_PRIME) & ov.lam.isin(lam_eu)].set_index(key)
        yp = ov[(ov.arm == ARM_EU) & ov.lam.isin(lam_eu)].set_index(key)
        erows = []
        for model in ('MDLM', 'UDLM'):
          xi = xp[xp.index.get_level_values('model') == model]
          yi = yp[yp.index.get_level_values('model') == model]
          idx = xi.index.intersection(yi.index)
          if len(idx) < 2:
            continue
          cells = []
          for col, f in ([('valid', '%+.1f'), ('novel', '%+.1f'),
                          ('prop', '%+.4f')]
                         + [(c, '%+.1f') for c in hcols]):
            dd = (yi.loc[idx, col].astype(float)
                  - xi.loc[idx, col].astype(float)).dropna()
            mu = dd.mean()
            se = dd.std(ddof=1) / np.sqrt(len(dd))
            star = r'$^{*}$' if se and abs(mu / se) > 2.1 else ''
            cells.append(f'{f % mu} $\\pm$ '
                         + (f'{se:.4f}' if col == 'prop' else f'{se:.1f}')
                         + star)
          erows.append([model, str(len(idx))] + cells)
        if erows:
          body.append(M.table(
            ['model', 'cells', r'$\Delta$ valid', r'$\Delta$ novel',
             r'$\Delta$ QED']
            + [f'$\\Delta$ h$_{{\\geq {bar:g}}}$' for bar in BARS['qed']],
            erows, 'lr rrr' + 'r' * len(hcols),
            r'$\mathrm{A}_{\mathrm{eu}} - \mathrm{A}$: what the exact-uniform step does on '
            r'its own, paired at matched $(N, \text{mode})$ at $\lambda = 5000$. '
            r'The only difference between the two is the flag.',
            'tab:ov-exactunif',
            note=r'The substitution replaces a Monte Carlo estimate of the base '
                 r'kernel with its closed form wherever the $N$ weights tie. For '
                 r'absorbing state the two have the \emph{same mean}, since the '
                 r'posterior is linear in $x_0$ --- so this removes variance, not '
                 r'bias, and an equal mean does not imply an equal metric. UDLM '
                 r'should move least: its uniform-weight rate is 0.156 against '
                 r"MDLM's 0.359. Whichever way it comes out, \texttt{ours.yaml} "
                 r"calling it an improvement ``at no cost'' claims more than a "
                 r'variance argument can support.',
            size=r'\footnotesize'))

      # Where the screening gain lives. The prediction on record was that the
      # gain would concentrate at *small* lambda, reasoning from exclude_invalid:
      # at large lambda exp(lambda*0.5) already suppresses invalid candidates, so
      # excluding them changes nothing. That reasoning holds for B - A, which is
      # exactly zero. It is wrong for screening, whose gain *grows* with lambda:
      # screening does not reweight the pool, it enlarges the valid part of it,
      # and at large lambda the sampler keeps the best valid candidate it was
      # offered. Five times the valid candidates is a better maximum to keep, and
      # a max-statistic gain grows as the weights concentrate.
      if 'B' in have and 'C' in have:
        db = (ov[ov.arm == 'C'].set_index(key)[hcols].astype(float)
              - ov[ov.arm == 'B'].set_index(key)[hcols].astype(float))
        db = db.dropna(how='all').reset_index()
        brows = []
        for model in ('MDLM', 'UDLM'):
          sub_ = db[db.model == model]
          for lam in sorted(sub_.lam.unique()):
            r = sub_[sub_.lam == lam][hcols].mean()
            brows.append([model, f'{lam:g}']
                         + [f'{r[c]:+.1f}' for c in hcols])
        body.append(M.table(
          ['model', r'$\lambda$'] + [f'$\geq {bar:g}$' for bar in BARS['qed']],
          brows, 'lr' + 'r' * len(hcols),
          r'Screening gain ($\mathrm{C} - \mathrm{B}$) against tilt strength, '
          r'averaged over $N$ and mixture mode. The gain grows with $\lambda$ on '
          r'both models, opposite to the direction predicted for '
          r'\texttt{exclude\_invalid}.', 'tab:ov-gain-lambda',
          note=r'Each row averages 8 cells ($N \in \{10, 30, 100, 300\}$ '
               r'$\times$ 2 mixture modes).',
          size=r'\footnotesize'))

      body.append(r'\section{Is screening worth its cost?}')
      body.append(r"""
Every contrast so far pairs at fixed $N$, and $N$ is not the quantity anyone
spends --- time is. Screening draws $5N$ candidates and has to decode and parse
all of them, so the arms are not cost-matched at equal $N$. Measured slowdown of
C over B: 1.26, 1.66, 4.71, 7.50 on MDLM and 1.23, 2.25, 4.82, 6.28 on UDLM at
$N = 10, 30, 100, 300$.

\paragraph{The verdict changes.} At matched \emph{cost} the advantage largely
goes away. B at $N=300$ costs 121\,s on MDLM and reaches
$221 / 115 / 37 / 7 / 1$ hits across the five bars; C at $N=100$ costs
320\,s --- 2.6$\times$ as much --- and reaches $208 / 114 / 42 / 10 / 0$. It is
\emph{worse} at the 0.50 bar, tied at 0.55, better at 0.60 and 0.65. UDLM tells
the same story: B at $N=300$ (143\,s) gives $266 / 156 / 67 / 14 / 3$ against C
at $N=100$ (371\,s) giving $233 / 154 / 74 / 16 / 0$. \textbf{So the paired
$\mathrm{C}-\mathrm{B}$ table overstates the benefit}, because holding $N$ fixed
hides that screening bought its extra valid candidates with time that could have
been spent on more candidates instead.

\paragraph{But the two knobs do different things,} which is why the comparison is
not a wash. Raising $N$ adds hits at \emph{every} bar and most at the low ones;
screening moves probability mass \emph{up}, gaining at 0.60--0.65 and losing at
0.50. If the deliverable is "molecules above a demanding threshold" the two are
not interchangeable at any budget.

\paragraph{The cost penalty is ours, not the method's.} The reward-call model
predicts screening should cost about 1.2$\times$, not 5$\times$: parsing is
0.042\,ms against 0.811\,ms for parse${+}$QED, and only $N$ candidates ever
reach QED. Checking the gap arithmetically at $N=300$ on MDLM --- 39.3\,M extra
candidates over a measured 786\,s, i.e.\ 20\,$\mu$s each --- parsing accounts
for just 52\,s of it once spread over 32 workers. The remaining $\sim$734\,s is
\emph{serial decoding}, turning candidate token tensors into SMILES strings in
Python. That is the same bottleneck the shelved \texttt{fast\_decode} attempt
failed to remove. So the honest statement is: screening is a clear win per
reward call and a loss per second \emph{in this implementation}, and the cost
verdict would flip with a decoder that is not Python string joins. That is a
falsifiable claim about our code, not a property of the estimator.
""")
      body += cost_tables(ov, 'qed')

      body.append(r'\subsection{A cost model, and why QED is the worst case}')
      body.append(r"""
Split the per-candidate cost into the part that scales with the oversampling
factor $k$ and the part that does not. Every one of the $kN$ candidates must be
decoded and validity-checked, at cost $c = d + v$; only the $N$ survivors reach
the reward, at cost $r$. Then
\[
  \mathrm{cost}_{\mathrm{B}} = N (c + r), \qquad
  \mathrm{cost}_{\mathrm{C}} = N (k c + r), \qquad
  \rho \;=\; \frac{\mathrm{cost}_{\mathrm{C}}}{\mathrm{cost}_{\mathrm{B}}}
        \;=\; \frac{k + \theta}{1 + \theta},
  \qquad \theta \equiv r / c .
\]
$\rho \to 1$ as $\theta \to \infty$: \textbf{screening is asymptotically free in
the reward cost}. Requiring $\rho \leq 1.2$ gives $\theta \geq 5(k - 1.2)$, i.e.\
$\theta \geq 4, 9, 19, 44$ for $k = 2, 3, 5, 10$.

\paragraph{Where QM9 sits.} $c \approx 20\,\mu$s, dominated by serial decoding,
and $r \approx 0.811\,\mathrm{ms} / 32 = 25\,\mu$s once the RDKit calls are spread
over the worker pool --- so $\theta \approx 1.25$ and $\rho \approx 2.8$, against
4.7 and 7.5 measured at $N = 100$ and $300$. \textbf{QED is close to the worst
case this method can be run in.} Not because QED is cheap in absolute terms
--- 0.811\,ms is 40$\times$ our decode --- but because we parallelised the
expensive part 32-fold and left the cheap part serial, which collapses $\theta$
to about 1.

\paragraph{A neural reward moves $\theta$ the right way,} and by a lot. At
$r = 500\,\mu$s effective, $\rho = 1.15$; at $r = 5\,$ms, $\rho = 1.02$. A GPU
property predictor or a preference model is also \emph{harder} to parallelise
32-fold than 32 independent RDKit calls --- it is one batched GPU op, not an
embarrassingly parallel CPU pool --- so its effective $r$ stays high. Both
effects push in the same direction. \textbf{The regime where screening is
cheapest is exactly the regime where reward models are actually used}, which
means the QM9 cost result is a floor on the method rather than a verdict on it.

\paragraph{One caveat on the language plan.} The decomposition needs the
feasibility check to be cheap \emph{relative to} the quality reward. Parse
against QED is a structural gap: 0.042 vs 0.811\,ms, different kinds of
computation. A safety classifier against a preference model is not --- both are
neural forward passes, and the gap is only the size ratio of the two networks.
That makes it a quantitative claim to be measured, not one that can be asserted
from the shape of the pipeline.
""")
      body.append(M.table(
        [r'$\theta = r/c$', r'$\rho$ at $k{=}5$', 'regime'],
        [['0.3', '4.08', 'reward as cheap as decoding'],
         ['1.25', '2.78', r'\textbf{QM9 + QED, 32 workers} (ours)'],
         ['3', '2.00', 'unparallelised RDKit'],
         ['10', '1.36', 'small GPU predictor'],
         ['19', '1.20', r'break-even target for $k=5$'],
         ['100', '1.04', 'preference model / LLM judge']],
        'rrl',
        r'Screening overhead $\rho$ against the reward-to-scan cost ratio '
        r'$\theta$. The rows are the model evaluated at illustrative $\theta$; '
        r'only the QM9 row is measured.', 'tab:ov-costmodel',
        note=r'$c = d + v$ is decode plus validity check, paid on all $kN$ '
             r'candidates; $r$ is the reward, paid on $N$. Both are effective '
             r'costs, i.e.\ after whatever parallelism each enjoys.',
        size=r'\footnotesize'))

    body.append(r"""
\textbf{Note for anyone reusing this.} Screening is incompatible with
\texttt{position\_selection=cv\_conf}: that control variate needs
$\mathbb{E}[\bar{x}_0^{(0)}] = x_\theta$ exactly, which holds only when the
candidates are drawn from $x_\theta$ itself.

Two earlier versions of this experiment were discarded rather than reported.
The first padded short pools with invalid candidates while they still carried
weight $e^{0}$, which makes the proposal depend on how many happened to be valid
and is not a correct estimator of anything; the padding fired on 53\,\% of
MDLM and 79\,\% of UDLM sequences, so this was the dominant path, not a corner.
Its numerical effect was probably small --- at $\lambda \geq 50$ the exclusion
is a no-op, verified byte-identical at $\lambda = 200$, because
$e^{\lambda \cdot 0.5}$ against $e^{0}$ already suppresses invalid candidates ---
but the justification did not hold, so it was rerun.
""")

  body.append(r'\end{document}')
  return '\n'.join(body)


def main():
  path = os.path.join(M.PDFS, 'Validity', 'src', 'validity.tex')
  os.makedirs(os.path.dirname(path), exist_ok=True)
  text = build()
  with open(path, 'w') as f:
    f.write(text)
  print(f'wrote {os.path.relpath(path, A.REPO)}  ({len(text)} chars)')


if __name__ == '__main__':
  main()
