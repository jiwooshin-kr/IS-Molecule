"""Generates pdfs/QED/fewstep/src/fewstep.tex from the few-step sweep.

Reads `pdfs/QED/fewstep/notes/fewstep_by_seed.csv`, which
`make_fewstep_table.py` writes from `results/qed/fewstep/`. Figures land in
`pdfs/QED/fewstep/src/` and are \includegraphics'd from there.

Exploratory report: the point is the trend across k, not a paper table.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_reports as M                 # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, 'pdfs', 'QED', 'fewstep')
SRC = os.path.join(OUT, 'src')
CSV = os.path.join(OUT, 'notes', 'fewstep_by_seed.csv')
LAMS = [0.0, 20.0, 200.0, 1000.0]
# The complete factorial ran k in {1,2,4,8} at every (N, lambda). k=16 and 32
# were a follow-up at N=1000 only, so they stay out of the grids and get their
# own section; the frontier, which is a scatter over cells, uses everything.
KS = [1, 2, 4, 8]
KS_ALL = [1, 2, 4, 8, 16, 32]


def load():
  d = pd.read_csv(CSV)
  d['calls'] = d.N * d.steps
  return d


def diff_grid(d, metric, fmt='%+.2f', lams=None):
  """Delta grid: rows (N, k), columns lambda."""
  lams = lams or LAMS
  sub = d[(d.metric == metric) & d.diff_mean.notna()]
  rows = []
  for n in sorted(sub.N.unique()):
    for k in KS:
      cells = []
      for lam in lams:
        c = sub[(sub.N == n) & (sub.k == k) & (sub.lam == lam)]
        if c.empty:
          cells.append('--')
          continue
        m, se = float(c.diff_mean.iloc[0]), float(c.diff_se.iloc[0])
        cell = f'${fmt % m} \\pm {se:.2f}$'
        if abs(m) > 2.5 * se and se > 0:
          cell = r'\textbf{' + cell + '}'
        cells.append(cell)
      rows.append([str(n) if k == KS[0] else '', str(k), str(32 // k)] + cells)
  header = [r'$N$', r'$k$', r'$T$'] + [f'$\\lambda={l:g}$' for l in lams]
  return rows, header


def fig_delta(d):
  """Delta hits@0.6 against k, one panel per lambda, one line per N."""
  sub = d[(d.metric == 'hits@0.6') & d.diff_mean.notna()]
  fig, axes = plt.subplots(1, 4, figsize=(13, 3.1), sharey=True)
  for ax, lam in zip(axes, LAMS):
    for n in sorted(sub.N.unique()):
      c = sub[(sub.N == n) & (sub.lam == lam)].sort_values('k')
      if c.empty:
        continue
      ax.errorbar(c.k, c.diff_mean, yerr=c.diff_se, marker='o', ms=4,
                  capsize=2, lw=1.3, label=f'$N={n}$')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xscale('log', base=2)
    ax.set_xticks(KS)
    ax.set_xticklabels(KS)
    ax.set_xlabel('$k$ (tokens unmasked per step)')
    ax.set_title(f'$\\lambda={lam:g}$' + ('  (control)' if lam == 0 else ''))
    ax.grid(alpha=0.3)
  axes[0].set_ylabel(r'hits@0.6:  marginal $-$ edlm')
  axes[-1].legend(fontsize=7, ncol=1)
  fig.tight_layout()
  path = os.path.join(SRC, 'fewstep_delta.pdf')
  fig.savefig(path, bbox_inches='tight')
  plt.close(fig)
  return 'fewstep_delta.pdf'


def fig_frontier(d):
  """hits@0.6 against reward calls per sequence, with the Pareto front."""
  sub = d[(d.metric == 'hits@0.6') & (d.lam > 0)]
  pts = []
  for _, r in sub.iterrows():
    for mode in ('marginal', 'edlm'):
      pts.append((r.calls, float(r[mode + '_mean']), int(r.k), mode))
  fig, ax = plt.subplots(figsize=(6.2, 4.2))
  marks = {1: 'o', 2: 's', 4: '^', 8: 'D', 16: 'v', 32: 'P'}
  for k in KS_ALL:
    xs = [p[0] for p in pts if p[2] == k]
    ys = [p[1] for p in pts if p[2] == k]
    ax.scatter(xs, ys, s=26, marker=marks[k], alpha=0.75,
               label=f'$k={k}$ ($T={32 // k}$)')
  front = sorted(p for p in pts
                 if not any(q[1] >= p[1] and q[0] <= p[0] and q[:2] != p[:2]
                            for q in pts))
  ax.plot([p[0] for p in front], [p[1] for p in front], 'k-', lw=1.1,
          alpha=0.8, zorder=0, label='Pareto front')
  old = sub[(sub.N == 300) & (sub.k == 1) & (sub.lam == 200)]
  if not old.empty:
    ax.annotate('previous operating point\n($N=300$, $T=32$)',
                xy=(float(old.calls.iloc[0]), float(old.edlm_mean.iloc[0])),
                xytext=(0.35, 0.16), textcoords='axes fraction', fontsize=8,
                arrowprops=dict(arrowstyle='->', lw=0.9))
  ax.set_xscale('log')
  ax.set_xlabel('reward evaluations per sequence  ($N \\times T$)')
  ax.set_ylabel('hits@0.6  (novel molecules, of 2048)')
  ax.grid(alpha=0.3)
  ax.legend(fontsize=8)
  fig.tight_layout()
  path = os.path.join(SRC, 'fewstep_frontier.pdf')
  fig.savefig(path, bbox_inches='tight')
  plt.close(fig)
  return 'fewstep_frontier.pdf'


def tradeoff_rows(d, lam=200.0):
  """Both modes side by side, decomposed into the three factors of `hits`."""
  rows = []
  for n in sorted(d.N.unique()):
    # k=16 and 32 only ran at N=1000, so take whatever is present per N rather
    # than the factorial list; the caption says which rows those are.
    ks = sorted(d[(d.N == n) & (d.lam == lam)].k.unique())
    for k in ks:
      def g(metric, mode):
        c = d[(d.metric == metric) & (d.N == n) & (d.k == k) & (d.lam == lam)]
        return float(c[mode + '_mean'].iloc[0]) if not c.empty else float('nan')
      cells = [str(n) if k == ks[0] else '', str(k), str(32 // k), f'{n * 32 // k}']
      for mode in ('marginal', 'edlm'):
        v, nov, h = g('valid', mode), g('novel', mode), g('hits@0.6', mode)
        cells += [f'{v:.0f}', f'{nov / v:.2f}', f'{h / nov:.3f}', f'{h:.1f}']
      rows.append(cells)
  return rows


def extension_rows(d, lam=200.0):
  rows = []
  for k in KS_ALL:
    c = d[(d.metric == 'hits@0.6') & (d.N == 1000) & (d.k == k) & (d.lam == lam)]
    if c.empty:
      continue
    calls = 1000 * (32 // k)
    m, e = float(c.marginal_mean.iloc[0]), float(c.edlm_mean.iloc[0])
    dm, se = float(c.diff_mean.iloc[0]), float(c.diff_se.iloc[0])
    best = max(m, e)
    rows.append([str(k), str(32 // k), str(calls), f'{m:.1f}', f'{e:.1f}',
                 f'${dm:+.1f}$', f'${dm / se:+.2f}$' if se else '--',
                 f'{1000 * best / calls:.1f}'])
  return rows


def frontier_rows(d):
  sub = d[(d.metric == 'hits@0.6') & (d.lam > 0)]
  pts = []
  for _, r in sub.iterrows():
    for mode in ('marginal', 'edlm'):
      pts.append((int(r.calls), float(r[mode + '_mean']), int(r.N), int(r.k),
                  int(r.steps), float(r.lam), mode))
  front = sorted(p for p in pts
                 if not any(q[1] >= p[1] and q[0] <= p[0] and q[:2] != p[:2]
                            for q in pts))
  return [[f'{c}', f'{h:.1f}', f'{n}', f'{k}', f'{t}', f'{lam:g}',
           M.esc(mode)] for c, h, n, k, t, lam, mode in front]


def build():
  d = load()
  f_delta, f_front = fig_delta(d), fig_frontier(d)
  o = [M.PREAMBLE, r'\begin{document}',
       r'\title{MDLM few-step generation: \texttt{marginal} vs \texttt{edlm} '
       r'at $k$ tokens per step}',
       r'\author{QED / fewstep}', r'\date{2026-08-26}', r'\maketitle', '']

  o.append(r"""\begin{center}
\fcolorbox{black}{yellow!25}{\begin{minipage}{0.93\textwidth}
\textbf{\large Read this before quoting any hits number in this report.}

\medskip
Every hits@0.6 figure below is counted on \texttt{novel}, defined as
$\mathrm{set(valid)} \setminus \mathrm{QM9}$, and that metric is
\textbf{confounded by molecule size in a way that scales with $k$}. QM9 contains
only molecules of $\le 9$ heavy atoms; RDKit validity does not enforce that, and
QED rewards size (it peaks near 300\,Da, far above anything QM9 can reach). Two
biases therefore compound and both grow with $k$: molecules outside the size
range are novel \emph{by construction}, and they score higher QED. Measured on
$N=1000$, $\lambda=200$, \texttt{edlm}: mean heavy-atom count runs $9.0 \to 13.6$
and the fraction above 9 heavy atoms runs $38.9\,\% \to 95.7\,\%$ as
$k: 1 \to 32$.

\medskip
Recounted on metrics that are not size-confounded (5-seed means, same runs):

\begin{center}\small
\begin{tabular}{lrrrr}
\toprule
$k$ (at $N=1000$, $\lambda=200$) & 1 & 4 & 8 & 32 \\
\midrule
as reported below (novel) & 69.6 & 111.2 & \textbf{137.4} & 98.2 \\
unique valid, no QM9 subtraction & \textbf{166.0} & 123.8 & 139.6 & 98.2 \\
restricted to $\le 9$ heavy atoms & \textbf{115.0} & 30.2 & 11.6 & 0.2 \\
\bottomrule
\end{tabular}
\end{center}

\medskip
\textbf{Invalidated:} every claim that few-step raises hits, the reward-call
Pareto frontier, and the reading of $\mathrm{novel}/\mathrm{valid}\to 1$ as
``duplication disappears'' (it is the size drift). \textbf{Still valid:} the
\texttt{marginal} vs \texttt{edlm} contrast (both modes drift identically, to
one decimal place in heavy-atom count, so it cancels in $\Delta$), the two null
controls, the validity collapse being a property of few-step itself, and the
\texttt{allbad} analysis.

\medskip
Synthetic accessibility is \emph{flat} across $k$ (SA 4.74 / 4.74 / 4.75 at
$k=1/8/32$ against 3.61 for QM9), so the larger molecules are not unmakeable
junk --- this is a confounded comparison, not reward hacking.

\medskip
Recount data: \texttt{results/qed/fewstep/\_recount.csv} (1240 runs), produced by
\texttt{scripts/ours/recount\_size\_controlled.py}. The choice of replacement
metric is an open decision; this report has not been rewritten under it.
\end{minipage}}
\end{center}

\bigskip""")

  o.append(r'\section{What this sweep is}')
  o.append(
    r'''At $k=1$ unmasked position per step the two mixture samplers are the
\emph{same Markov chain}: \texttt{marginal} returns
$\mathbb{E}_n[q(\cdot\,|\,x_t,x_0^{(n)})]$ while \texttt{edlm} draws
$n^\ast\sim\mathrm{Cat}(w)$ and samples that component, and with one position
updating the joint equals that position's marginal. The whole difference
therefore lives in steps with $k\ge 2$, where \texttt{marginal} draws each
position independently from the reward-weighted candidate histogram while
\texttt{edlm} copies $k$ tokens from a single candidate. This sweep puts $k$ on
an axis.

No code change was needed. \texttt{position\_selection=random\_k} unmasks
$\mathrm{round}(n_{\text{masked}}\cdot p_{\text{unmask}})$ positions and for the
log-linear schedule $p_{\text{unmask}} = \mathrm{d}t/t$ exactly, so
$k = L/T$ deterministically at every step; verified numerically for
$T\in\{32,16,8,4\}$ with no rounding drift and no leftover mask. Uniformly random
positions are not a heuristic here: every masked position carries the same
``stay masked'' probability, so conditional on the count the position set is
uniform over subsets --- \texttt{random\_k} \emph{is} the true kernel
conditioned on the count.

Arm A settings throughout (\texttt{oversample=1}, \texttt{exclude\_invalid=False},
\texttt{exact\_uniform\_step=False}), $N\in\{10,30,100,300,1000\}$,
$\lambda\in\{0,20,200,1000\}$, seeds $1$--$5$, 2048 samples per run.
800 runs, no failures. \textbf{Exploratory: read the trend, not a single cell.}''')

  o.append(r'\section{Two built-in null controls}')
  o.append(
    r'''Both must come out at zero. $k=1$ is the structural null above.
$\lambda=0$ is a second, independent null at \emph{every} $k$: the weights are
uniform, and the candidates were drawn position-independently from
$x_\theta$, so copying $k$ tokens from one candidate \emph{is} $k$ independent
draws from $x_\theta$. The $\lambda=0$ column below is clean everywhere
($|\Delta|\le 2.6$ at every $N$ and $k$), which is what licenses reading the
$\lambda>0$ columns as a reward-driven effect.

Tested formally rather than by eye, the $k=1$ row passes: of the 15 cells
at $\lambda>0$, exactly one exceeds $p<0.05$ against 0.8 expected by chance, and
the 15 cells at $k\ge2,\ \lambda=0$ give zero. The $\pm3$ to $\pm5$ values that
look large at $N\ge300$ are not: hit counts there are 55--82, so counting noise
makes the per-seed $\Delta$ swing by $\pm15$, and those cells have $|t|<2$.

The one cell that did exceed it --- $N=100$, $\lambda=1000$, $\Delta=-10.6$ with
all five seeds negative and $t=-7.77$ --- was rerun with ten fresh seeds and did
not replicate: seeds 6--15 give $\Delta=-3.0\pm2.8$ ($t=-1.06$) at
$\lambda=1000$ and $-0.9\pm2.3$ ($t=-0.39$) at $\lambda=200$. Its two
``impossible'' features both dissolved --- the standard deviation went
$3.05\to8.24$ (counting noise predicts 8.9) and the seed-wise correlation
between the modes went $+0.97\to+0.06$. Both were five-sample artifacts. The
out-of-sample seeds are the honest test here, since the cell was selected for
looking extreme; quoting the 15-seed pooled figure would re-import that
selection. \textbf{Verdict: noise. Both null controls hold.}''')

  rows, header = diff_grid(d, 'hits@0.6')
  o.append(M.table(
    header, rows, 'rrr' + 'r' * len(LAMS),
    r'$\Delta$ hits@0.6 (\texttt{marginal} $-$ \texttt{edlm}), mean $\pm$ '
    r'standard error over 5 seeds. Bold marks $|\Delta| > 2.5\,\mathrm{se}$. '
    r'The $\lambda=0$ column and the $k=1$ row are the two null controls. '
    r'Every cell is 5 seeds; the follow-up seeds of Check~A are kept out of the '
    r'grid on purpose (see text) and live in '
    r'\texttt{notes/fewstep\_all\_seeds.csv}.',
    'tab:fs-delta-hits', size=r'\footnotesize'))

  o.append(r'\section{Result: \texttt{edlm} wins, and the gap grows with $k$ '
           r'and $N$}')
  o.append(
    r'''The sign is the opposite of the pre-registered prediction. Crossover was
expected to pay once parents are elites; instead the conditional-independence
error dominates. Writing $k$ positions independently breaks SMILES ring-closure
digits and brackets, which are non-separable across positions, and at $k\ge 4$
that cost swamps the extra reachable combinations. The largest effect is
$-19.4\pm3.6$ at $N=1000$, $k=8$, $\lambda=200$.''')
  o.append(r'\begin{figure}[htbp]\centering'
           r'\includegraphics[width=\linewidth]{' + f_delta + r'}'
           r'\caption{$\Delta$ hits@0.6 against $k$. $\lambda=0$ is the '
           r'control panel; error bars are $\pm1$ standard error over 5 seeds.}'
           r'\label{fig:fs-delta}\end{figure}')

  o.append(r'\section{The trade-off few-step actually makes}')
  o.append(
    r'''Validity collapses with $k$ exactly as expected --- and hits go
\emph{up} anyway, \emph{for both samplers}, provided $N$ is large enough. The
best $k$ is interior and it moves right with $N$: reading Table~\ref{tab:fs-tradeoff}
by row, hits@0.6 peak at $k=2$ for $N=10$ and $N=30$, at $k=2$--$4$ for $N=100$,
at $k=4$ for $N=300$, and at $k=8$ for $N=1000$ --- where the follow-up rows
confirm it turns over, 137.4 at $k=8$ against 109.2 at $k=16$ and 98.2 at
$k=32$. Larger $N$ buys more candidates and so survives a bigger per-step commit
before the state degenerates: \texttt{allbad} at $k=8$ is 0.824 at $N=300$
against 0.728 at $N=1000$. Writing
$\mathrm{hits@0.6} = \mathrm{valid} \times
\frac{\mathrm{novel}}{\mathrm{valid}} \times
\frac{\mathrm{hits@0.6}}{\mathrm{novel}}$
splits that into three factors, and going $k=1 \to 8$ at $N=1000$ moves each one
in a different direction:

\begin{center}\small
\begin{tabular}{lcccc}
\toprule
 & valid & novel/valid & hits@0.6/novel & $=$ hits@0.6 \\
\midrule
\texttt{marginal} & $\times 0.32$ & $\times 2.09$ & $\times 2.47$ & $\times 1.63$ \\
\texttt{edlm}     & $\times 0.33$ & $\times 2.14$ & $\times 2.78$ & $\times 1.97$ \\
\bottomrule
\end{tabular}
\end{center}

Validity falls threefold, and \emph{both} of the other factors more than double,
at one eighth of the reward calls. The middle factor is the one that is easy to
miss: at $k=1$ only 45\,\% of valid molecules are novel, at $k=8$ it is 95\,\% ---
few-step buys diversity because there is far less duplication among the
survivors. The two samplers differ only in the last factor, which is exactly the
coherence effect \texttt{edlm} wins on.''')
  o.append(M.table(
    [r'$N$', r'$k$', r'$T$', 'calls',
     'valid', r'$\frac{\text{novel}}{\text{valid}}$',
     r'$\frac{\text{hits@0.6}}{\text{novel}}$', 'hits@0.6',
     'valid', r'$\frac{\text{novel}}{\text{valid}}$',
     r'$\frac{\text{hits@0.6}}{\text{novel}}$', 'hits@0.6'],
    tradeoff_rows(d), 'rrrr|rrrr|rrrr',
    r'$\lambda=200$, counts out of 2048 samples; columns 5--8 are '
    r'\texttt{marginal} and 9--12 are \texttt{edlm}. \emph{hits@0.6} is the '
    r'number of \emph{novel} molecules scoring $\mathrm{QED}\ge0.60$, the '
    r'same quantity as in Table~\ref{tab:fs-delta-hits}; \emph{valid} and '
    r'\emph{novel} are counts of the 2048 generated sequences, and '
    r'\emph{novel} is deduplicated ($\mathrm{set(valid)}\setminus$ train). '
    r'Reward evaluations per sequence are $N\times T$. The three factors '
    r'multiply to hits@0.6, so each row shows where few-step gains and where '
    r'it pays. The $k=16$ and $k=32$ rows exist only at $N=1000$: they were a '
    r'follow-up once the factorial grid put that optimum on its boundary.',
    'tab:fs-tradeoff', size=r'\footnotesize'))

  o.append(r'\section{Pushing $k$ further at $N=1000$}')
  o.append(
    r'''Since the factorial grid put the $N=1000$ optimum on its boundary,
$k=16$ ($T=2$) and $k=32$ ($T=1$) were run there as a follow-up.
\textbf{$k=8$ really is the peak}: hits@0.6 falls from 137.4 to 109.2 to 98.2.
But the reward budget halves each time, so hits \emph{per call} keeps rising ---
34.4, 54.6 and 98.2 per 1000 calls --- and $k=32$ stays on the frontier.

Two things fall out. First, the mode difference \textbf{vanishes again} at
$k=32$ ($\Delta=+3.4$, $t=0.84$), and the effective sample size says why. Steps
where no candidate parses carry uniform weights and hence $\mathrm{ESS}=N$
exactly, so removing them recovers the concentration in the steps that matter:
$\mathrm{ESS}_{\text{live}} = (\mathrm{ESS} - \texttt{allbad}\cdot N)/(1-\texttt{allbad})$
falls $302 \to 61 \to 1.5$ for $k = 1 \to 8 \to 32$. At 1.5 effective parents the
weighted histogram is a one-hot, so \texttt{marginal} \emph{is} \texttt{edlm}.
The difference peaks at $k=8$ because that is where the concentration is
intermediate --- the same non-monotonicity predicted for $\lambda$, arriving
through $k$ instead.

Second, $T=1$ gives a free implementation check. With one step \texttt{edlm}
returns a candidate verbatim (it is literally best-of-$N$), so if the reward ever
picks a parseable candidate the output must be valid, i.e.
$\mathrm{valid} = 1 - \texttt{allbad}$. Measured: $\texttt{allbad}=0.874$
predicts 0.126 against 0.118 observed.''')

  o.append(M.table(
    [r'$k$', r'$T$', 'reward calls', r'\texttt{marginal}', r'\texttt{edlm}',
     r'$\Delta$', r'$t$', 'hits per 1000 calls'],
    extension_rows(d), 'rrrrrrrr',
    r'$N=1000$, $\lambda=200$, hits@0.6. $k=16$ and $32$ are a follow-up at '
    r'this $N$ only. Absolute hits peak at $k=8$; hits per reward call keep '
    r'rising to $k=32$.', 'tab:fs-extension'))

  o.append(r'\section{Hits per reward call: no $k=1$ point is on the frontier}')
  o.append(
    r'''This is the axis the reframed claim needs (hits against reward-call
budget). Every point on the Pareto front has $k\ge 2$. The previous operating
point ($N=300$, $k=1$, $T=32$, $\lambda=200$: 55.2 hits at 9600 calls) is well
inside it --- $N=300$, $k=4$ gives 78.4 hits at 2400 calls, i.e.
$4\times$ fewer reward evaluations and 42\,\% more hits.''')
  o.append(M.table(
    ['reward calls', 'hits@0.6', r'$N$', r'$k$', r'$T$', r'$\lambda$', 'mode'],
    frontier_rows(d), 'rrrrrrl',
    r'Pareto frontier of hits@0.6 against reward evaluations per sequence '
    r'($\lambda>0$ only).', 'tab:fs-frontier'))
  o.append(r'\begin{figure}[htbp]\centering'
           r'\includegraphics[width=0.78\linewidth]{' + f_front + r'}'
           r'\caption{Every cell of the sweep, hits against reward budget.}'
           r'\label{fig:fs-frontier}\end{figure}')

  o.append(r'\section{Validity, as a difference}')
  rows, header = diff_grid(d, 'valid', fmt='%+.1f')
  o.append(M.table(
    header, rows, 'rrr' + 'r' * len(LAMS),
    r'$\Delta$ valid molecules out of 2048 (\texttt{marginal} $-$ '
    r'\texttt{edlm}), 5 seeds; bold marks $|\Delta| > 2.5\,\mathrm{se}$. The '
    r'validity cost of per-position mixing is confirmed only at $\lambda=20$, '
    r'$k\ge4$, large $N$ (and $\lambda=200$, $N=1000$, $k=8$). Cells where '
    r'\texttt{marginal} comes out ahead reach at most $t=+1.4$, i.e. none of '
    r'them is distinguishable from zero --- the mixed signs elsewhere are noise '
    r'on a standard error of 5--20 counts.', 'tab:fs-delta-valid',
    size=r'\footnotesize'))

  o.append(r'\section{Predictions, scored}')
  o.append(M.table(
    ['prediction', 'outcome'],
    [[r'$k=1 \Rightarrow \Delta=0$', r'partly --- 2 of 20 cells deviate'],
     [r'$\lambda=0 \Rightarrow \Delta=0$', r'\textbf{held} at every $N$, $k$'],
     [r'$N=10$, $k\ge2 \Rightarrow \Delta\approx0$', r'held, small and mixed'],
     [r'$N\uparrow$, $k\uparrow \Rightarrow |\Delta|$ grows',
      r'magnitude \textbf{held}, sign \textbf{wrong}'],
     [r'$\lambda\le50 \Rightarrow \Delta<0$',
      r'held --- but $\lambda\ge200$ is negative too'],
     [r'$k$ large $\Rightarrow \Delta$ collapses (\texttt{allbad}$\to1$)',
      r'\textbf{wrong} --- $k=8$ is the maximum']],
    'll', r'Written down before the sweep was read.', 'tab:fs-predictions'))

  o.append(r'\section{What is and is not resolvable here}')
  o.append(
    r'''Seed-to-seed noise was measured directly from five existing replicates
(\texttt{random\_k}, marginal, $N=300$, 1024 samples): single-run standard
deviation 3.58 on hits@0.6, 7.3 on valid, 0.0046 on novel-QED mean, 0.010 on
max. At 2048 samples with 5 seeds this gives $\mathrm{se}\approx1.6$ on the
paired hits difference. Consequences: hits@0.6 and hits@0.65 are the only
metrics that resolve this comparison; \texttt{max} has a noise floor several
times its own effect and is reported only for honesty; the count of novel
molecules is null on the mode axis because the validity loss and the diversity
gain cancel, and is the right metric for the $k$ axis instead.''')

  o.append(r'\end{document}')
  return '\n\n'.join(o)


def main():
  os.makedirs(SRC, exist_ok=True)
  path = os.path.join(SRC, 'fewstep.tex')
  text = build()
  with open(path, 'w') as f:
    f.write(text)
  print(f'wrote {os.path.relpath(path, REPO)}  ({len(text)} chars)')


if __name__ == '__main__':
  main()
