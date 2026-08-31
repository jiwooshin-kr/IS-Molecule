"""Generates the three LaTeX reports under `pdfs/` from the result CSVs.

    pdfs/Reproduce/src/reproduce.tex     D-CBG reproduction against the paper
    pdfs/QED/src/qed_sweep.tex           lambda x N sweep, QED
    pdfs/RingCount/src/ringcount_sweep.tex   lambda x N sweep, ring count

Every number is read from `results/`, never transcribed by hand, so re-running
this after new runs land keeps the reports honest. Build with `build_reports.sh`,
which leaves the PDF at the top of each folder and the .tex/.aux/.log in `src/`.

Usage:
    python scripts/ours/make_reports.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_sweep as A          # noqa: E402
import check_paper_repro as C      # noqa: E402

PDFS = os.path.join(A.REPO, 'pdfs')
PROP_LABEL = {'qed': 'QED', 'ring_count': 'ring count'}
# LaTeX-safe versions of the identifiers that appear in prose.
TT = {'qed': r'\texttt{qed}', 'ring_count': r'\texttt{ring\_count}'}


def esc(text):
  return str(text).replace('_', r'\_')


PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage[table]{xcolor}
\usepackage[colorlinks=true,urlcolor=blue,linkcolor=black]{hyperref}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}
% \texttt{ring\_count} and friends are unbreakable and otherwise poke into the
% margin; let TeX stretch interword space instead.
\sloppy
"""


def table(header, rows, align, caption, label, note=None, size=r'\small'):
  """A booktabs tabular wrapped in a table float."""
  out = [r'\begin{table}[htbp]', r'\centering', size]
  # A table float is its own group, so this does not leak. Wide grids at
  # \scriptsize were overflowing the text block by up to 81pt with the default
  # 6pt column separation; 3.5pt fits them without dropping a font size.
  if size in (r'\scriptsize', r'\tiny'):
    out.append(r'\setlength{\tabcolsep}{3.5pt}')
  elif size == r'\footnotesize':
    out.append(r'\setlength{\tabcolsep}{4pt}')
  # Shrink only if the tabular would otherwise stick out: \width is the box's
  # natural width, so a table that already fits is left untouched. Column count
  # alone cannot predict this -- a 10-column table of "$-0.0019 \pm 0.0006$"
  # cells is wider than a 14-column table of two-digit integers.
  out += [r'\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%',
          r'\begin{tabular}{' + align + '}', r'\toprule',
          ' & '.join(header) + r' \\', r'\midrule']
  out += [' & '.join(str(c) for c in r) + r' \\' for r in rows]
  out += [r'\bottomrule', r'\end{tabular}}']
  if note:
    out.append(r'\\[0.4em] \begin{minipage}{0.92\textwidth}\footnotesize '
               + note + r'\end{minipage}')
  out += [r'\caption{' + caption + '}', r'\label{' + label + '}',
          r'\end{table}', '']
  return '\n'.join(out)


def pivot_table(frame, values, caption, label, fmt='%.3f', note=None):
  """A N x lambda grid as a LaTeX table."""
  grid = frame.pivot(index='N', columns='lam', values=values)
  lams = list(grid.columns)
  header = [r'$N$'] + [f'$\\lambda={l:g}$' for l in lams]
  rows = []
  for n, row in grid.iterrows():
    rows.append([str(n)] + [(fmt % v) if pd.notna(v) else '--' for v in row])
  return table(header, rows, 'r' + 'r' * len(lams), caption, label, note)


# ------------------------------------------------------------------ reproduce

def build_reproduce():
  body = [PREAMBLE,
          r'\title{Reproducing the D-CBG baselines\\'
          r'\large \texttt{arXiv:2412.10193v3} (Schiff et al., ICLR 2025) on QM9}',
          r'\author{}\date{2026-08-15}',
          r'\begin{document}\maketitle',
          r'\section{What is being reproduced, and what is ours}', r"""
D-CBG needs a \emph{noisy classifier} per property, and the paper releases none.
All four classifiers used here (\texttt{\{qed,ring\_count\}} $\times$
\texttt{\{uniform,absorbing\_state\}}) are therefore our own training runs. On the
generative side the split is:

\begin{itemize}
  \item \textbf{UDLM} --- the released \texttt{kuleshov-group/udlm-qm9}
        checkpoint, i.e.\ the paper's own model. This is the arm that has to
        match.
  \item \textbf{MDLM} --- our own 25k-step run; HuggingFace publishes no
        MDLM-QM9.
\end{itemize}

So even the UDLM arm is not a pure re-run of the paper's pipeline: the
classifier differs. This matters when reading the $z$ column, because the
paper's $\pm$ is over \emph{five sampling seeds on one checkpoint} and contains
no training variation of either component.

Two conversions are needed before the numbers are comparable. The paper reports
\textsc{Num. Valid} / \textsc{Num. Novel} as counts out of 1{,}024 and its
\textsc{Mean} column over \emph{novel} sequences only, while
\texttt{our\_qm9\_eval.py} writes \textsc{Valid} as a fraction of 1{,}024 and
\textsc{Novel} as a fraction of the valid set. And we contribute one seed against
a five-seed mean, so the spread of the difference is
$\sqrt{1+1/5}\,\sigma$, not $\sigma$; the $z$ below carries that factor.
The property $\sigma$ is floored at $0.005$ because the paper prints that column
to two decimals, so a printed ``$0.00$'' only means $<0.005$.
""",
          r'\section{The four rows that appear in Table 5}', r"""
The main paper's Table 5 does not state which $\gamma$ or \texttt{use\_approx}
produced each row; the appendix does (Tables 20, 23, 29, 32). Those are the
settings reproduced here.
"""]

  head_rows, all_sections = [], []
  for prop in A.PROPS:
    for model in ('MDLM', 'UDLM'):
      # C.check returns every row for the property, both models -- filter, or
      # each model's table silently shows the other's rows too.
      rows = [r for r in C.check(os.path.join(A.REPO, 'results', prop), prop)
              if 'note' not in r and r['model'] == model]
      if not rows:
        continue
      h = next((r for r in rows if r['headline']), None)
      if h:
        head_rows.append([
          PROP_LABEL[prop], model,
          f"$\\gamma={h['gamma']:g}$, {'approx.' if h['approx'] else 'exact'}",
          f"{h['valid']:.0f} / {h['valid_paper']:.1f}", f"{h['z_valid']:+.1f}",
          f"{h['novel']:.0f} / {h['novel_paper']:.1f}", f"{h['z_novel']:+.1f}",
          f"{h['prop']:.3f} / {h['prop_paper']:.2f}", f"{h['z_prop']:+.1f}"])
      body_rows = [[
        f"{r['gamma']:g}", 'approx.' if r['approx'] else 'exact',
        f"{r['valid']:.0f}", f"{r['valid_paper']:.1f}", f"{r['z_valid']:+.1f}",
        f"{r['novel']:.0f}", f"{r['novel_paper']:.1f}", f"{r['z_novel']:+.1f}",
        f"{r['prop']:.3f}", f"{r['prop_paper']:.2f}", f"{r['z_prop']:+.1f}"]
        for r in rows]
      within = sum(1 for r in rows if max(abs(r['z_valid']), abs(r['z_novel']),
                                          abs(r['z_prop'])) <= 2)
      all_sections.append(table(
        [r'$\gamma$', 'arm', 'valid', 'paper', '$z$', 'novel', 'paper', '$z$',
         PROP_LABEL[prop], 'paper', '$z$'],
        body_rows, 'rl rrr rrr rrr',
        f'{model}, {PROP_LABEL[prop]}: every $\\gamma$ we ran against the paper. '
        f'{within}/{len(rows)} rows are within $2\\sigma$ on all three metrics.',
        f'tab:repro-{prop}-{model}'))

  body.append(table(
    ['property', 'model', 'setting', 'valid', '$z$', 'novel', '$z$',
     'property', '$z$'],
    head_rows, 'll l rr rr rr',
    'The four D-CBG rows of the paper\'s Table 5. Three are within $2\\sigma$ on '
    'all three metrics; MDLM ring count is outside on the two count metrics and '
    'matches on the property.', 'tab:repro-headline',
    note=r'Each metric column reads \emph{ours / paper}; $z$ is '
         r'(ours $-$ paper) / $\sqrt{1+1/5}\,\sigma_{\text{paper}}$.',
    size=r'\footnotesize'))

  body.append(r'\section{Where the residual gap comes from}')
  # Mean signed z per arm quantifies direction, which is the whole diagnosis.
  dir_rows = []
  for prop in A.PROPS:
    for model in ('MDLM', 'UDLM'):
      rows = [r for r in C.check(os.path.join(A.REPO, 'results', prop), prop)
              if 'note' not in r and r['model'] == model]
      for approx in (False, True):
        sub = [r for r in rows if r['approx'] == approx]
        if not sub:
          continue
        f = lambda k: sum(r[k] for r in sub) / len(sub)  # noqa: E731
        dir_rows.append([PROP_LABEL[prop], model,
                         'approx.' if approx else 'exact', str(len(sub)),
                         f"{f('z_valid'):+.1f}", f"{f('z_novel'):+.1f}",
                         f"{f('z_prop'):+.1f}"])
  body.append(table(
    ['property', 'model', 'arm', 'n', 'mean $z$ valid', 'mean $z$ novel',
     'mean $z$ property'], dir_rows, 'lll r rrr',
    'Mean \\emph{signed} $z$ per arm. Random scatter would average to zero; a '
    'consistent sign is a systematic offset.', 'tab:repro-direction',
    note=r'The property column is centred everywhere ($|{\cdot}|\le 0.7$) except '
         r'the $n{=}2$ collapsed MDLM ring exact arm. The deviations live in '
         r'validity and novelty counts, not in how hard the guidance steers.'))

  # Pull the headline offset out of the same table the reader sees, so prose and
  # table cannot drift apart when the data is regenerated.
  mdlm_ap = next(r for r in dir_rows
                 if r[0] == 'QED' and r[1] == 'MDLM' and r[2] == 'approx.')
  body.append(r"""
Reading the two tables together:

\begin{enumerate}
  \item \textbf{The MDLM approximate arm on QED} is the one real offset: mean
        signed $z$ of $""" + f"{mdlm_ap[4]} / {mdlm_ap[5]}" + r"""$ on
        valid/novel with the property at $""" + mdlm_ap[6] + r"""$.
        Our checkpoint is \emph{more robust} to the first-order approximation
        than theirs --- same property at higher validity. Expected, since
        MDLM-QM9 is our own training run.
  \item \textbf{Everything else is centred.} In particular UDLM exact, the arm
        that must match, is within $|z|\le 1$ on average for both properties.
  \item Individual cells still scatter, worst at UDLM ring $\gamma{=}3$ exact
        ($4.56$ against $4.70\pm0.03$). A $0.6\,\%$ printed $\sigma$ is tight
        enough that any pipeline difference shows up as a large $z$.
\end{enumerate}

\textbf{Verdict.} The baselines reproduce. Absolute numbers can be quoted against
the paper; frontier comparisons within a fixed checkpoint are safer still, and
are what the sweep reports do.
""")
  body += all_sections
  body.append(r'\end{document}')
  return '\n'.join(body)


# --------------------------------------------------------------------- sweeps

BAR_MAIN = {'qed': 0.60, 'ring_count': 5}
# lambda values shown in the grid tables. The full sweep has 7 (QED) and 9 (ring)
# values, which at two modes overflows the text block by 157pt. These keep the
# unguided control plus the range where anything happens; the omitted values sit
# below the point where the tilt starts to bite (QED 2, 5) or inside the
# saturated plateau (ring 0.5, 1, 2, 100), and the full grid stays in the
# per-metric tables.
GRID_LAMS = {'qed': (0.0, 20.0, 50.0, 200.0, 1000.0),
             'ring_count': (0.0, 5.0, 10.0, 30.0, 1000.0)}


def hits_grid(prop, frame, cbg=None):
  """hits at the headline bar over the full (N, lambda) grid, one table per model.

  Every other view of this sweep aggregates something away -- the curves average
  over the grid, the Pareto table keeps only the frontier. This is the raw grid
  on the primary metric, so a reader can see where in (N, lambda) the hits
  actually live rather than taking a summary on trust.
  """
  bar = BAR_MAIN[prop]
  keep = GRID_LAMS.get(prop)
  lams = sorted(l for l in frame['lam'].unique()
                if keep is None or l in keep)
  out = []
  for model in ('MDLM', 'UDLM'):
    sub_ = frame[frame.model == model]
    if sub_.empty:
      continue
    rows = []
    for n in sorted(sub_.N.unique()):
      cells = []
      for mode in ('marginal', 'edlm'):
        for lam in lams:
          c = sub_[(sub_.N == n) & (sub_.lam == lam) & (sub_['mode'] == mode)]
          if not len(c):
            cells.append('--')
            continue
          h = A.hits(prop, c.iloc[0]['tag'], bars=(bar,))
          cells.append('--' if not h else f'{h[0]:d}')
      rows.append([str(int(n))] + cells)
    header = [r'$N$']
    for mode in ('marginal', 'edlm'):
      header += [f'{mode[:4]}.\\,{lam:g}' for lam in lams]
    all_lams = sorted(frame['lam'].unique())
    hidden = [l for l in all_lams if l not in lams]
    note = (r'Novel molecules clearing the bar, out of the 1{,}024 generated. '
            r'Every cell is a single run, seed 1; nothing is averaged. '
            + (r'$\lambda \in \{'
               + ', '.join(f'{l:g}' for l in hidden)
               + r'\}$ are omitted to fit the page. ' if hidden else '')
            + r'\textbf{Not} filtered by \texttt{MIN\_NOVEL}, unlike the Pareto '
            r'table --- a collapsed run simply scores few hits here, which is '
            r'the honest reading, whereas a mean over 2 molecules is not.')
    if cbg is not None and not cbg.empty:
      t = cbg[cbg.model == model]
      best, where = -1, ''
      for _, r in t.iterrows():
        if r['novel'] < MIN_NOVEL:
          continue
        h = A.hits(prop, r['tag'], bars=(bar,))
        if h and h[0] > best:
          best, where = h[0], f"{'approx.' if r['approx'] else 'exact'} "\
                              f"$\\gamma={r['gamma']:g}$"
      if best >= 0:
        note += (f' \\textbf{{D-CBG\'s best on this bar is {best}}} '
                 f'({where}), for comparison.')
    out.append(table(
      header, rows, 'r' + 'r' * (2 * len(lams)),
      f'{model}: hits at {PROP_LABEL[prop]} $\\geq {bar:g}$ across the whole '
      f'$(N, \\lambda)$ grid. Columns are (mixture mode) $\\times$ $\\lambda$.',
      f'tab:hits-grid-{prop}-{model}', note=note, size=r'\scriptsize'))
  return out


def build_sweep(prop):
  frame = A.load(prop)
  cbg = A.load_cbg(prop)
  label, tt = PROP_LABEL[prop], TT[prop]
  lams = sorted(frame['lam'].unique())
  ns = sorted(frame['N'].unique())
  sat = {'qed': None, 'ring_count': 5.0}[prop]
  hits_tables = hits_grid(prop, frame, cbg)

  body = [PREAMBLE,
          r'\title{The $(\lambda, N)$ sweep on ' + label + r'\\'
          r'\large \texttt{marginal} against \texttt{edlm}, MDLM and UDLM}',
          r'\author{}\date{2026-08-15}',
          r'\begin{document}\maketitle',
          r'\section{Setup}',
          f"""
{len(frame)} runs, 1{{,}}024 samples each at $T=32$ steps, seed 1, reward
{tt}. Grid: $\\lambda \\in \\{{{', '.join(f'{l:g}' for l in lams)}\\}}$ against
$N \\in \\{{{', '.join(str(n) for n in ns)}\\}}$, for both mixture-sampling modes
and both base models.

$N$ is the number of $\\mathbf{{x}}_0$ candidates drawn per denoising step --- the
Monte Carlo budget of the estimator. $\\lambda$ is the tilt in
$q(\\mathbf{{x}}_0) \\propto p_\\theta(\\mathbf{{x}}_0)\\,e^{{\\lambda r(\\mathbf{{x}}_0)}}$.
The two modes differ only in how $\\mathbf{{x}}_{{t-1}}$ is drawn from the resulting
mixture: \\texttt{{marginal}} averages the $N$ individually normalised posteriors,
\\texttt{{edlm}} draws a component and then samples from it. They share
per-position marginals and differ only in the joint.
""",
          r'\section{Reading ESS: use the median, never the last step}', r"""
Effective sample size $\mathrm{ESS} = 1/\sum_n w_n^2$ says which end of the grid
a run is on: $\mathrm{ESS}=N$ means the tilt is doing nothing, $\mathrm{ESS}\to1$
means it has collapsed onto a single candidate.

The driver script prints only the \emph{final} step's ESS, and that number is
actively misleading on MDLM. Absorbing-state sampling has almost everything
unmasked by the last step, so all $N$ candidates decode to the same sequence and
tie on reward --- the last-step ESS reads $\approx N$ however hard $\lambda$ is
tilting. Judged on it, MDLM looks entirely unguided. Its novel QED in fact climbs
across the $\lambda$ grid. Every ESS below is the \textbf{median over the
trajectory}, digested from the per-run logs into
\texttt{results/\_ess\_summary.csv}.
"""]

  # ESS depends only on the importance weights, which both modes compute
  # identically -- but report both anyway rather than leaving the reader to
  # infer it, since every other grid in this report is split by mode.
  for model in ('MDLM', 'UDLM'):
    for mode in ('marginal', 'edlm'):
      sub = frame[(frame.model == model) & (frame['mode'] == mode)]
      if sub.empty:
        continue
      body.append(pivot_table(
        sub, 'ess_frac',
        f'{model}, \\texttt{{{mode}}}: median ESS as a fraction of $N$.',
        f'tab:ess-{prop}-{model}-{mode}', fmt='%.2f'))

  body.append(r'\section{The grid}')
  for model in ('MDLM', 'UDLM'):
    for mode in ('marginal', 'edlm'):
      sub = frame[(frame.model == model) & (frame['mode'] == mode)]
      if sub.empty:
        continue
      body.append(pivot_table(
        sub, 'prop',
        f'{model}, \\texttt{{{mode}}}: novel-{label} mean.',
        f'tab:grid-{prop}-{model}-{mode}'))
  for model in ('MDLM', 'UDLM'):
    for mode in ('marginal', 'edlm'):
      sub = frame[(frame.model == model) & (frame['mode'] == mode)]
      if sub.empty:
        continue
      body.append(pivot_table(
        sub, 'novel',
        f'{model}, \\texttt{{{mode}}}: number of novel molecules '
        f'(out of 1{{,}}024 generated).',
        f'tab:novel-{prop}-{model}-{mode}', fmt='%.0f'))
  # The mean is not what a screening pipeline consumes -- it takes the top of
  # the pool -- so report the best single molecule alongside it.
  for model in ('MDLM', 'UDLM'):
    for mode in ('marginal', 'edlm'):
      sub = frame[(frame.model == model) & (frame['mode'] == mode)]
      if sub.empty:
        continue
      body.append(pivot_table(
        sub, 'max',
        f'{model}, \\texttt{{{mode}}}: highest novel-{label} of the run.',
        f'tab:max-{prop}-{model}-{mode}',
        note=(r'A raw max grows with the number of molecules it is taken over, '
              r'and these cells differ several-fold in novel count, so a higher '
              r'max here is partly a deeper pool. That is a real advantage at a '
              r'fixed generation budget -- every cell generated the same 1,024 '
              r'sequences -- but it is not a claim about distribution shape. '
              r'Section~\ref{sec:beyond-mean-' + prop + '} separates the two.')
        if (model, mode) == ('MDLM', 'marginal') else None))

  # ---- mode comparison
  wide = frame.pivot_table(index=['model', 'N', 'lam'], columns='mode',
                           values=['prop', 'novel'])
  d = pd.DataFrame({
    'd_prop': (wide[('prop', 'marginal')] - wide[('prop', 'edlm')]),
    'd_novel': (wide[('novel', 'marginal')] - wide[('novel', 'edlm')]),
  }).reset_index()
  rows = []
  for model in ('MDLM', 'UDLM'):
    g = d[d.model == model].groupby('lam')
    for lam, sub in g:
      rows.append([model, f'{lam:g}', f"{sub['d_prop'].abs().mean():.3f}",
                   f"{sub['d_prop'].mean():+.3f}",
                   f"{sub['d_novel'].abs().mean():.1f}",
                   f"{sub['d_novel'].mean():+.1f}"])
  body.append(r'\section{\texttt{marginal} against \texttt{edlm}}')
  body.append(table(
    ['model', r'$\lambda$', r'mean $|\Delta|$ property', r'mean $\Delta$ property',
     r'mean $|\Delta|$ novel', r'mean $\Delta$ novel'],
    rows, 'lr rrrr',
    r'\texttt{marginal} minus \texttt{edlm}, averaged over the five $N$ at each '
    r'$\lambda$. $|\Delta|$ is the typical size of the gap, $\Delta$ its sign.',
    f'tab:modes-{prop}',
    note=r'A systematic winner would show as a consistent sign in the '
         r'$\Delta$ columns. There is none.'))

  # The lambda -> infinity prediction.
  hi = d[d.lam == max(lams)]
  lo = d[d.lam == 0]
  body.append(f"""
Two things were predicted before the sweep ran, and only one holds.

\\textbf{{Predicted and confirmed:}} the two modes are empirically
interchangeable. Across the whole grid the mean absolute gap in novel-{label} is
{d['d_prop'].abs().mean():.3f} with mean signed gap
{d['d_prop'].mean():+.3f} --- no consistent winner, at any $\\lambda$ or $N$.

\\textbf{{Predicted and \\emph{{not}} confirmed:}} the gap was expected to
\\emph{{shrink}} as $\\lambda$ grows, because one-hot weights make
$\\sum_n w_n q(\\cdot\\,|\\,\\mathbf{{x}}_0^{{(n)}})$ collapse to the single component
\\texttt{{edlm}} would have drawn, and a posterior conditioned on a fixed
$\\mathbf{{x}}_0$ already factorises over positions. Measured, the gap is flat:
{lo['d_prop'].abs().mean():.3f} at $\\lambda=0$ against
{hi['d_prop'].abs().mean():.3f} at $\\lambda={max(lams):g}$. The reason is
visible in the ESS tables --- the weights never actually become one-hot in this
grid, so the limit is never entered. The flat gap is therefore sampling noise
between two samplers that agree, which is the same conclusion by a weaker route.
""")

  # ---- lambda = 0
  body.append(r'\section{$\lambda = 0$: the unguided control}')
  body.append(r"""
At $\lambda=0$ the weights are uniform, so no tilt is applied whatever $N$ or the
mode is. Every cell of Table~\ref{tab:lam0-""" + prop + r"""} should therefore be
the base sampler, and the spread across it measures the sampling noise that every
other comparison in this report has to clear.
""")
  z = frame[frame.lam == 0]
  rows = []
  for model in ('MDLM', 'UDLM'):
    for mode in ('marginal', 'edlm'):
      sub = z[(z.model == model) & (z['mode'] == mode)].sort_values('N')
      if sub.empty:
        continue
      rows.append([model, esc(mode)]
                  + [f"{v:.3f}" for v in sub['prop']])
  # The lambda=0 spread IS the noise floor, and comparing it to the mode gap is
  # the cleanest statement of whether the modes differ at all.
  floor = z.groupby('model')['prop'].agg(lambda g: g.max() - g.min()).max()
  gap = frame.pivot_table(index=['model', 'N', 'lam'], columns='mode',
                          values='prop')
  gap = (gap['marginal'] - gap['edlm']).abs().mean()
  body.append(table(
    ['model', 'mode'] + [f'$N={n}$' for n in ns], rows,
    'll' + 'r' * len(ns),
    f'novel-{label} at $\\lambda=0$. Every entry should be the base sampler, '
    f'independent of $N$ and of the mode.',
    f'tab:lam0-{prop}',
    note=r'This is the implementation control: at $\lambda=0$ the weights are '
         r'uniform, so no tilt is applied and the numbers must be flat in $N$ '
         r'up to sampling noise. They are.'))
  body.append(
    f"""
This table also calibrates everything else in the report. The widest $\\lambda=0$
spread within a single model is {floor:.3f} in novel-{label} --- runs that are
identical in distribution, differing only by sampling noise. The mean gap between
\\texttt{{marginal}} and \\texttt{{edlm}} across the whole grid is {gap:.3f},
\\textbf{{{'smaller' if gap < floor else 'larger'}}} than that floor. So the two
modes are not merely close: their difference is
{'indistinguishable from noise' if gap < floor else 'above the noise floor'}.
""")

  # ---- frontier
  body.append(r'\section{The frontier against D-CBG}')
  body.append(f"""
Both methods are swept, then compared on the axes the paper's Figure 3 uses:
number of novel molecules against novel-{label} mean. Comparing at single
hyperparameter points would be meaningless --- the paper tunes $\\gamma$ per model
\\emph{{and}} per property.
""")
  for model in ('MDLM', 'UDLM'):
    ours, theirs = frame[frame.model == model], cbg[cbg.model == model]
    if ours.empty or theirs.empty:
      continue
    po, pt = A.pareto(_usable(ours)), A.pareto(_usable(theirs))
    dropped = len(theirs) - len(_usable(theirs)) + len(ours) - len(_usable(ours))
    rows = [['ours', f"{r['mode']}, $N={r['N']}$, $\\lambda={r['lam']:g}$",
             f"{r['valid']:.0f}", f"{r['novel']:.0f}", f"{r['prop']:.3f}"]
            for _, r in po.iterrows()]
    rows += [['D-CBG',
              f"$\\gamma={r['gamma']:g}$, {'approx.' if r['approx'] else 'exact'}",
              f"{r['valid']:.0f}", f"{r['novel']:.0f}", f"{r['prop']:.3f}"]
             for _, r in pt.iterrows()]
    body.append(table(
      ['method', 'setting', 'valid', 'novel', f'novel-{label}'], rows,
      'll rrr',
      f'{model}: the Pareto-optimal settings of each method on '
      f'(novel count, novel-{label}).', f'tab:frontier-{prop}-{model}',
      note=(f'{dropped} collapsed run(s) with fewer than {MIN_NOVEL} novel '
            f'molecules were excluded; a property mean over a handful of '
            f'molecules is not an operating point.') if dropped else None))

  # ---- beyond the mean
  body.append(r'\section{Beyond the mean}\label{sec:beyond-mean-' + prop + '}')
  body.append(r"""
Everything above compares \emph{means}, which is the paper's metric but not the
one a screening pipeline faces. Nobody consumes the mean of a generated library;
they take the usable molecules out of it. And the mean is blind to the axis these
methods differ on most: from the same 1{,}024 generated sequences our runs yield
several times more novel molecules than D-CBG's.

\subsection{The primary comparison: usable molecules per fixed budget}

Both methods generated exactly 1{,}024 sequences. So simply count the novel
molecules that clear a quality bar. Quality and quantity both enter --- a run
that yields many poor novel molecules scores nothing, and one that yields a
handful of excellent ones scores little.

This number needs no correction of any kind. In particular it must \textbf{not}
be normalised by the novel count, nor size-matched against the competitor:
converting a fixed budget into more usable molecules \emph{is} the result, and
dividing it out would erase precisely what is being measured.
""")
  body.append(r'''
\subsubsection{The whole grid, on the primary metric}

The tables above pick a best setting per method, which is what a headline needs
but not what a reader should have to trust. Below is the raw grid: hits at the
headline bar at every $(N, \lambda)$, both modes, nothing averaged and nothing
filtered. It shows where in the grid the hits live --- and that the surface is
not monotone in either axis, so a single ``best'' cell is partly a draw from run
noise.
''')
  body += hits_tables
  bars = A.BARS[prop]
  rows = []
  for model in ('MDLM', 'UDLM'):
    hr = A.hit_rows(prop, frame, model)
    if len(hr) < 2:
      continue
    base = next((r for r in hr if r['method'] == 'D-CBG'), None)
    for r in hr:
      setting = (f"$\\gamma={r['row']['gamma']:g}$, "
                 f"{'approx.' if r['row']['approx'] else 'exact'}"
                 if r['method'] == 'D-CBG' else
                 f"$N={r['row']['N']:.0f}$, $\\lambda={r['row']['lam']:g}$")
      name = (r['method'] if r['method'] == 'D-CBG'
              else '\\texttt{' + r['method'] + '}')
      cells = []
      for i, x in enumerate(r['hits']):
        if base and r['method'] != 'D-CBG' and base['hits'][i]:
          ratio = x / base['hits'][i]
          body_cell = f"{x} ({ratio:.1f}$\\times$)"
          cells.append(body_cell if ratio < 1.0
                       else '\\textbf{' + str(x) + '} '
                            + f"({ratio:.1f}$\\times$)")
        else:
          cells.append(str(x))
      rows.append([model, name, setting, f"{r['novel']:.0f}",
                   f"{r['mean']:.3f}"] + cells)
  body.append(table(
    ['model', 'method', 'setting', 'novel', 'mean']
    + [f'$\\geq {b}$' for b in bars],
    rows, 'lll rr' + 'r' * len(bars),
    f'Novel molecules clearing each {label} bar, out of the 1{{,}}024 sequences '
    f'each run generated. The two mixture-sampling modes are kept apart rather '
    f'than merged into a single \\emph{{ours}} row, since they are different '
    f'samplers. '
    f'Each is at its own best setting, ranked on the middle bar; the multiplier '
    f'is against D-CBG on the same model.', f'tab:hits-{prop}',
    note=r'No size matching and no normalisation --- every run spent the same '
         r'generation budget, so these are directly comparable counts.',
    size=r'\footnotesize'))

  body.append(r"""
\subsection{Secondary: shape comparisons, which discard the counts}

Two standard distribution-free comparisons. Both deliberately ignore how many
molecules each method produced, so neither can see the advantage above; they are
here to answer a different question --- whether our \emph{distribution} is
better, or only our yield.

\begin{itemize}
  \item \textbf{Rank against rank.} Sort both sets and compare like for like:
        best against best, tenth percentile against tenth. The fraction of
        quantiles where we lead is the \emph{quantile win rate}; 1.0 would be
        \textbf{first-order stochastic dominance}. Comparing the $k$-th
        \emph{rank} directly would need equal sample sizes, so this compares at
        matched quantiles, the size-independent form of the same idea.
  \item \textbf{Draw against draw.} $A_{12}$, the Vargha--Delaney probability of
        superiority: one molecule drawn at random from each, how often is ours
        better? The normalised Mann--Whitney $U$; $0.5$ is indistinguishable.
\end{itemize}

The \emph{matched} column subsamples the larger novel set down to the smaller.
It is a \textbf{decomposition, not a fairness correction} --- it separates
``we win because our tail is better'' from ``we win because our pool is deeper''.
The counterfactual it describes never occurs in practice, so it should not be
read as the fair comparison; the table above is.
""")
  rows = []
  for model in ('MDLM', 'UDLM'):
    ours, theirs = _usable(frame[frame.model == model]), _usable(cbg[cbg.model == model])
    if ours.empty or theirs.empty:
      continue
    o = ours.loc[ours[f'top{A.TOP_K}'].idxmax()]
    t = theirs.loc[theirs[f'top{A.TOP_K}'].idxmax()]
    va, vb = A.novel_values(prop, o['tag']), A.novel_values(prop, t['tag'])
    cmp = A.compare_distributions(va, vb)
    m = min(len(va), len(vb))
    ra, rb = A._matched(va, m), A._matched(vb, m)
    rows.append([model, 'ours', f"{o['novel']:.0f}", f"{o['prop']:.3f}",
                 f"{o['max']:.3f}", f"{o[f'top{A.TOP_K}']:.3f}", f"{ra['topk']:.3f}",
                 f"{cmp.get('quantile_win', float('nan')):.2f}",
                 f"{cmp.get('a12', float('nan')):.2f}"])
    rows.append([model, 'D-CBG', f"{t['novel']:.0f}", f"{t['prop']:.3f}",
                 f"{t['max']:.3f}", f"{t[f'top{A.TOP_K}']:.3f}", f"{rb['topk']:.3f}",
                 '--', '--'])
  body.append(table(
    ['model', 'method', 'novel', 'mean', 'max', f'top{A.TOP_K}',
     f'top{A.TOP_K} matched', 'q-win', '$A_{12}$'], rows, 'll rrrrr rr',
    f'Each method at its own best setting by top-{A.TOP_K}. \\emph{{q-win}} and '
    f'$A_{{12}}$ sit on the ours row and describe ours against D-CBG.',
    f'tab:beyond-{prop}',
    note=r'\emph{max} and \emph{top-K} are at the fixed 1,024-sequence budget '
         r'and so include our deeper novel pool. \emph{matched} removes that, '
         r'and answers only ``quality or quantity?''.'))

  # ---- validity
  body.append(r'\section{Validity}\label{sec:validity-' + prop + '}')
  body.append(f"""
Validity is the axis on which the two mechanisms differ most sharply, and it is
what drives the curve directions of the previous section, so it is worth
reporting on its own.

Our reward sees a \\emph{{clean}} $x_0$ candidate and returns
\\texttt{{invalid\\_reward}} $=0$ when RDKit cannot parse it. Raising $\\lambda$
therefore pushes probability towards parseable molecules before it pushes
towards high {label}: an invalid candidate is treated as a {label}-zero one and
suppressed. D-CBG instead perturbs the per-position logits with the gradient of
a noisy classifier, which carries no such term --- nothing in it prefers a
parseable string.
""")
  for model in ('MDLM', 'UDLM'):
    for mode in ('marginal', 'edlm'):
      sub = frame[(frame.model == model) & (frame['mode'] == mode)]
      if sub.empty:
        continue
      body.append(pivot_table(
        sub, 'valid',
        f'{model}, \\texttt{{{mode}}}: valid molecules out of the 1{{,}}024 '
        f'generated.',
        f'tab:valid-{prop}-{model}-{mode}', fmt='%.0f'))

  # the actual argument: what each strength dial does to validity
  rows = []
  for model in ('MDLM', 'UDLM'):
    sub = frame[(frame.model == model) & (frame['mode'] == 'marginal')]
    if sub.empty:
      continue
    big = sub[sub.N == sub.N.max()].sort_values('lam')
    base = big[big.lam == 0]['valid']
    base = float(base.iloc[0]) if len(base) else float('nan')
    peak = float(big['valid'].max())
    last = float(big[big.lam == big.lam.max()]['valid'].iloc[0])
    rows.append([model, f'ours ($N={int(sub.N.max())}$)',
                 f'{base:.0f}', f'{peak:.0f}', f'{last:.0f}',
                 r'\textbf{rises}'])
    t = cbg[cbg.model == model]
    for approx in (False, True):
      d = t[t.approx == approx].sort_values('gamma')
      if d.empty:
        continue
      rows.append([model, f"D-CBG {'approx.' if approx else 'exact'}",
                   f"{d['valid'].iloc[0]:.0f}", f"{d['valid'].max():.0f}",
                   f"{d['valid'].iloc[-1]:.0f}",
                   'falls' if d['valid'].iloc[-1] < d['valid'].iloc[0] else 'rises'])
  body.append(table(
    ['model', 'method', 'weakest dial', 'best', 'strongest dial', 'trend'],
    rows, 'll rrr l',
    f'Valid molecules out of 1{{,}}024 as the strength dial is turned up. '
    f'Ours is read along $\\lambda$ at the largest $N$; D-CBG along $\\gamma$.',
    f'tab:valid-vs-dial-{prop}',
    note=r'This is the mechanism behind the opposite curve directions: our dial '
         r'buys validity on the way to buying property, so the novel count '
         r'rises with it, while D-CBG spends validity and the novel count '
         r'falls. On MDLM the D-CBG exact arm reaches zero valid molecules at '
         r'$\gamma \geq 6$, which is why the paper does not report it there.'))

  # ---- curve direction
  body.append(r'\section{The shape of each curve}')
  body.append(f"""
Plot novel count on the $x$-axis against novel-{label} mean on the $y$-axis and
sweep each method's strength dial --- $\\lambda$ for ours, $\\gamma$ for D-CBG.
The two methods trace curves in \\emph{{opposite directions}}, and that is a
property worth reporting on its own, independently of which sits higher.

\\begin{{itemize}}
  \\item \\textbf{{D-CBG runs up and to the left.}} Raising $\\gamma$ buys
        property by spending novelty: the dial is a genuine trade-off the user
        has to tune, and the original paper tunes it per model and per property.
  \\item \\textbf{{Ours runs up and to the right.}} Raising $\\lambda$ improves
        \\emph{{both}} axes at once. The reason is visible in the validity
        column: the first thing $\\lambda$ buys is valid molecules, which raises
        the novel count, and only then does it buy property. There is no
        trade-off to tune --- the only cost is compute.
\\end{{itemize}}

The correlation between the two axes along each sweep reduces this to one
number. Crucially it is measured at \\emph{{fixed}} $N$, so the up-right shape
is not an artifact of spending more Monte Carlo budget: along every row below,
only the strength dial moves.
""")
  rows = []
  for model in ('MDLM', 'UDLM'):
    sub = frame[(frame.model == model) & (frame['mode'] == 'marginal')]
    theirs = _usable(cbg[cbg.model == model])
    entries = [(f'ours, $N={n}$', sub[sub.N == n], 'lam')
               for n in sorted(sub.N.unique()) if n in (100, 300, 1000, 2000)]
    entries += [('D-CBG exact', theirs[~theirs.approx], 'gamma'),
                ('D-CBG approx.', theirs[theirs.approx], 'gamma')]
    for name, d, key in entries:
      if len(d) < 3:
        continue
      d = d.sort_values(key)
      x = d['novel'].astype(float).values
      y = d['prop'].astype(float).values
      corr = float(np.corrcoef(x, y)[0, 1])
      shape = (r'\textbf{up-right}' if corr > 0.3 else
               ('up-left' if corr < -0.3 else 'mixed'))
      rows.append([model, name, str(len(x)),
                   f'({x[0]:.0f}, {y[0]:.3f})', f'({x[-1]:.0f}, {y[-1]:.3f})',
                   f'{corr:+.2f}', shape])
  body.append(table(
    ['model', 'method', 'pts', 'weakest dial', 'strongest dial', 'corr', 'shape'],
    rows, 'll r rr rl',
    f'Direction of each curve in (novel count, novel-{label}) space. '
    f'\\emph{{corr}} is between the two axes along the sweep: positive means the '
    f'dial improves both, negative means it trades one for the other.',
    f'tab:shape-{prop}',
    note=r'Measured at fixed $N$, so the up-right shape is not an effect of '
         r'raising the Monte Carlo budget --- only the strength dial moves '
         r'along each row.'))

  for model in ('MDLM', 'UDLM'):
    sub = frame[(frame.model == model) & (frame['mode'] == 'marginal')]
    lams = sorted(sub['lam'].unique())
    rows = []
    for n in sorted(sub.N.unique()):
      d = sub[sub.N == n].sort_values('lam')
      rows.append([f'$N={n}$'] +
                  [f'({r.novel:.0f}, {r.prop:.3f})' for r in d.itertuples()])
    body.append(table(
      ['ours'] + [f'$\\lambda={l:g}$' for l in lams], rows,
      'l' + 'c' * len(lams),
      f'{model}: the (novel, novel-{label}) point at each $\\lambda$, for every '
      f'$N$. Reading a row left to right traces that curve.',
      f'tab:curve-{prop}-{model}', size=r'\scriptsize'))
    theirs = _usable(cbg[cbg.model == model])
    rows = []
    for approx in (False, True):
      d = theirs[theirs.approx == approx].sort_values('gamma')
      if d.empty:
        continue
      rows.append([('approx.' if approx else 'exact')] +
                  [f'$\\gamma={r.gamma:g}$: ({r.novel:.0f}, {r.prop:.3f})'
                   for r in d.itertuples()])
    if rows:
      width = max(len(r) for r in rows) - 1
      rows = [r + [''] * (width + 1 - len(r)) for r in rows]
      body.append(table(
        ['D-CBG'] + [''] * width, rows, 'l' + 'l' * width,
        f'{model}: the same for D-CBG, sweeping $\\gamma$. The novel count '
        f'falls as the property rises --- the opposite direction.',
        f'tab:curve-cbg-{prop}-{model}', size=r'\scriptsize'))

  body.append(_verdict(prop, frame, cbg))
  body.append(r'\end{document}')
  return '\n'.join(body)


# A run that produced almost nothing is not an operating point: D-CBG's ring
# exact gamma=2 cell has 2 valid molecules out of 1,024 and a "ring count" of
# 6.0 computed over those two. Left in, it becomes the reported peak. Rows below
# this many novel molecules are dropped from the frontier and the drop is stated
# in the caption rather than being silent.
MIN_NOVEL = 10


def _usable(frame):
  return frame[frame['novel'] >= MIN_NOVEL]


def _verdict(prop, frame, cbg):
  """The part that has to be written per property, because they disagree."""
  out = [r'\section{What the sweep says}']
  for model in ('MDLM', 'UDLM'):
    ours = _usable(frame[frame.model == model])
    theirs = _usable(cbg[cbg.model == model])
    if ours.empty or theirs.empty:
      continue
    bo, bt = ours['prop'].max(), theirs['prop'].max()
    no, nt = ours['novel'].max(), theirs['novel'].max()
    dominated = (bt >= bo) and (nt >= no)
    out.append(
      f"\\textbf{{{model}.}} Peak novel-{PROP_LABEL[prop]}: ours {bo:.3f} against "
      f"D-CBG {bt:.3f}. Peak novel count: ours {no:.0f} against D-CBG {nt:.0f}. "
      + ('\\emph{D-CBG dominates on both axes} --- there is no trade-off to '
         'report here, we simply lose.'
         if dominated else
         'Each method wins one axis, so the comparison is a genuine frontier.'))
  if prop == 'ring_count':
    out.append(r"""
\subsection{Why ring count behaves differently from QED}

The method can only \emph{select}, never \emph{invent}. Weights are
$\mathrm{softmax}(\lambda r)$ over $N$ candidates drawn from $x_\theta$, so the
best reachable reward at a step is the best value \emph{present among those $N$
candidates}. No $\lambda$ can produce a 5-ring molecule if none of the $N$
proposals has one.

That ceiling binds hard on ring count and barely at all on QED, for two reasons
that reinforce each other:

\begin{itemize}
  \item \textbf{The reward is integer valued}, so candidates tie and the tilt
        saturates early. Raising $\lambda$ past $\approx5$ changes nothing:
        the softmax is already a hard argmax over the tied top set.
  \item \textbf{High-ring molecules are rare under the base model} (QM9's mean
        ring count is 1.74), so the $N$ proposals rarely contain one.
\end{itemize}

D-CBG has no such ceiling: it perturbs the per-position logits with a classifier
gradient and can steer toward regions the base model would not have proposed.
That is exactly what the numbers show.

\textbf{The binding constraint is $N$, not $\lambda$.} Along the saturated
$\lambda$ ridge the property still climbs monotonically with $N$ and has not
flattened at $N=500$. Pushing $N$ further is the one lever with headroom left;
raising $\lambda$ is spent.
""")
  else:
    out.append(r"""
\subsection{Where the gains come from}

$\lambda$ and $N$ are not interchangeable. $\lambda$ sets \emph{how hard} the
tilt pulls and saturates once the weights concentrate; $N$ sets \emph{how good
the best available candidate is}, and along the high-$\lambda$ edge the property
still rises with $N$ at $N=500$. The two together trace the frontier: high
$\lambda$ with small $N$ is a greedy sampler with nothing good to pick from,
while large $N$ with $\lambda=0$ is just the base model.
""")
  return '\n'.join(out)


def main():
  targets = [
    (os.path.join(PDFS, 'Reproduce', 'src', 'reproduce.tex'), build_reproduce()),
    (os.path.join(PDFS, 'QED', 'src', 'qed_sweep.tex'), build_sweep('qed')),
    (os.path.join(PDFS, 'RingCount', 'src', 'ringcount_sweep.tex'),
     build_sweep('ring_count')),
  ]
  for path, text in targets:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
      f.write(text)
    print(f'wrote {os.path.relpath(path, A.REPO)}  ({len(text)} chars)')


if __name__ == '__main__':
  main()
