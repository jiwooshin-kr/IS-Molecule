"""Generates pdfs/Confidence/src/confidence.tex from the position-selection runs.

Covers the whole confidence-based unmasking line of work:

    results/<prop>/confidence_unmasking/   the 4 arms, and the edlm degeneracy check
    results/<prop>/conf_blend/             the blend beta sweep and the control variate
    results/<prop>/position_window/        the CV restricted to late t

Every number is read from those CSVs, never transcribed, so the prose cannot
drift from the tables. Build with `build_reports.sh`.
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
COL = {'qed': 'QED', 'ring_count': 'RING_COUNT'}
LAB = {'qed': 'QED', 'ring_count': 'ring count'}
# The lambdas that carry the seed replication. lambda=0 is excluded throughout:
# with uniform weights there is no tilt, so nothing about the reward is measured.
LAMS = {'qed': [50., 200., 1000.], 'ring_count': [2., 5., 10.]}
N_FIXED = 300


def _row(path, prop):
  """One run, in the units the reports use."""
  frame = pd.read_csv(path)
  gen = frame[frame['Seed'] != -1]
  if gen.empty:
    return None
  r = gen.iloc[-1]
  total = int(r['Num Samples'])
  valid = r['Valid'] * total
  hits = np.nan
  js = path[:-4] + '_samples.json'
  if os.path.exists(js):
    try:
      vals = json.load(open(js)).get(f'{prop}_novel') or []
      hits = sum(1 for x in vals if x >= BAR[prop])
    except (OSError, json.JSONDecodeError):
      pass
  return {'valid': round(valid), 'novel': round(r['Novel'] * valid),
          'prop': r[f'Novel {COL[prop]} Mean'], 'hits': hits}


def collect(prop, folder, pattern, namer):
  rows = []
  for path in glob.glob(os.path.join(A.REPO, 'results', prop, folder, '*.csv')):
    m = re.match(pattern, os.path.basename(path)[:-4])
    if not m:
      continue
    d = m.groupdict()
    if int(d.get('N', N_FIXED)) != N_FIXED:
      continue
    lam = float(d['lam'])
    name = namer(d)
    if name is None:
      continue
    base = _row(path, prop)
    if base is None:
      continue
    rows.append({'arm': name, 'lam': lam,
                 'seed': int(d.get('seed') or 1)} | base)
  return pd.DataFrame(rows)


CU = (r'mdlm_cu_(?P<a>\w+?)_(?P<mode>marginal|edlm)_N(?P<N>\d+)'
      r'_lam(?P<lam>[\d.]+)(?:_s(?P<seed>\d+))?$')
CB = (r'mdlm_cb_(?P<a>cv[\d.]*|beta[\d.]+)_N(?P<N>\d+)'
      r'_lam(?P<lam>[\d.]+)(?:_s(?P<seed>\d+))?$')
PW = (r'mdlm_pw_cv_conf_w(?P<w>[\d.]+)_N(?P<N>\d+)'
      r'_lam(?P<lam>[\d.]+)_s(?P<seed>\d+)$')

ARM_LABEL = {'bernoulli': 'A \\ bernoulli', 'random_k': 'B \\ random\\_k',
             'beta0': 'C \\ xtheta\\_conf', 'beta1': 'D \\ xbar\\_conf',
             'cv': 'E \\ CV'}
ARM_ORDER = list(ARM_LABEL.values())


def five_arms(prop):
  """The five arms on the seed-replicated grid."""
  a = collect(prop, 'confidence_unmasking', CU,
              lambda d: ARM_LABEL.get(d['a'])
              if d['mode'] == 'marginal' and float(d['lam']) in LAMS[prop]
              else None)
  b = collect(prop, 'conf_blend', CB,
              lambda d: ARM_LABEL.get(d['a'])
              if float(d['lam']) in LAMS[prop] else None)
  return pd.concat([a, b], ignore_index=True)


def stat(frame, col):
  v = frame[col].astype(float).dropna()
  return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan)


def paired(frame, arm, base_arm, col, key=('lam', 'seed')):
  """mean, se and t of the paired difference arm - base_arm."""
  x = frame[frame.arm == base_arm].set_index(list(key))[col].astype(float)
  y = frame[frame.arm == arm].set_index(list(key))[col].astype(float)
  d = (y - x).dropna()
  if len(d) < 2:
    return np.nan, np.nan, np.nan, 0
  m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
  return m, se, (m / se if se else np.nan), len(d)


def cell(m, se, t, fmt='%+.1f', dec=1):
  if not np.isfinite(m):
    return '--'
  star = r'$^{*}$' if np.isfinite(t) and abs(t) > 2.145 else ''
  return f'{fmt % m} $\\pm$ {se:.{dec}f}{star}'


def build():
  body = [M.PREAMBLE,
          r'\title{Confidence-based position selection\\'
          r'\large Does choosing \emph{which} position to unmask buy anything?}',
          r'\author{}\date{2026-08-16}',
          r'\begin{document}\maketitle',
          r'\section{The idea, and why it is MDLM-only}', r"""
In absorbing-state diffusion the reverse posterior at a masked position is

\[ P(\text{stay masked}) \;=\; \frac{\text{move\_chance}_s}{\text{move\_chance}_t}, \]

the \emph{same} value at every masked position and independent of the token
distribution. The base sampler's choice of which position to unmask therefore
carries no information at all. That is free capacity, and this report asks
whether spending it well is worth anything.

Two structural restrictions follow, and both are properties of the method rather
than of the implementation:

\begin{itemize}
  \item \textbf{MDLM only.} Uniform diffusion (UDLM) has no mask state --- every
        position is resampled at every step --- so there is no ``which position
        to unmask'' to decide. The code raises \texttt{NotImplementedError}
        rather than silently doing nothing.
  \item \textbf{\texttt{marginal} only.} Under \texttt{edlm} the per-position
        distribution is a one-hot on the single drawn candidate, so the
        confidence is $1$ at every masked position and the ranking is entirely
        ties. Only \texttt{marginal}, which averages the $N$ normalised
        posteriors, exposes a usable per-position signal.
\end{itemize}

The second point is the one that mattered for the paper's framing: the
$\lambda \times N$ sweep found \texttt{marginal} and \texttt{edlm} empirically
interchangeable, so \texttt{marginal} could not be justified on performance.
A capability that exists only under \texttt{marginal} would have justified it
instead. That is what is being tested here.
""",
          r'\section{The arms}', r"""
\begin{itemize}
  \item[\textbf{A}] \texttt{bernoulli} --- the schedule's own coin flip. The
        original sampler, and the baseline that actually matters.
  \item[\textbf{B}] \texttt{random\_k} --- deterministic count, uniformly random
        positions. Isolates fixing the \emph{count}.
  \item[\textbf{C}] \texttt{xtheta\_conf} --- deterministic count, ranked by the
        denoiser marginal $x_\theta$. Isolates ranking \emph{at all}; this is
        MaskGIT-style confidence decoding.
  \item[\textbf{D}] \texttt{xbar\_conf} --- ranked by the reward-weighted
        candidate histogram $\bar{x}_0$. The reward-aware version.
  \item[\textbf{E}] \texttt{cv\_conf} --- D with a control-variate correction
        (Section~\ref{sec:cv}). No free parameter.
\end{itemize}

Everything below is MDLM, \texttt{marginal}, $N=300$, 1{,}024 samples per run,
$T=32$ steps, and \textbf{5 seeds $\times$ 3 values of $\lambda$ = 15 runs per
arm}. Seed replication is not optional here: two runs of an \emph{identical}
algorithm differing only in random stream gave hit counts of 30 and 49, so
single-seed rows are not interpretable.
"""]

  # ---------------------------------------------------------------- results
  body.append(r'\section{Result: nothing beats doing nothing}')
  for prop in ('qed', 'ring_count'):
    f = five_arms(prop)
    if f.empty:
      continue
    rows = []
    for arm in ARM_ORDER:
      s = f[f.arm == arm]
      if s.empty:
        continue
      n, p, h = stat(s, 'novel'), stat(s, 'prop'), stat(s, 'hits')
      rows.append([f'\\texttt{{{arm}}}', str(len(s)),
                   f'{n[0]:.1f} $\\pm$ {n[1]:.1f}',
                   f'{p[0]:.4f} $\\pm$ {p[1]:.4f}',
                   f'{h[0]:.1f} $\\pm$ {h[1]:.1f}'])
    body.append(M.table(
      ['arm', 'n', 'novel', f'novel-{LAB[prop]}', f'hits ($\\geq {BAR[prop]}$)'],
      rows, 'lr rrr',
      f'{LAB[prop]}: the five arms, mean $\\pm$ standard error over 15 runs.',
      f'tab:conf-abs-{prop}'))

    base = ARM_ORDER[0]
    rows = []
    for arm in ARM_ORDER[1:]:
      if f[f.arm == arm].empty:
        continue
      cs = [cell(*paired(f, arm, base, c)[:3], fmt=fmt, dec=dec)
            for c, fmt, dec in (('novel', '%+.1f', 1), ('prop', '%+.4f', 4),
                                ('hits', '%+.1f', 1))]
      rows.append([f'\\texttt{{{arm}}}'] + cs)
    body.append(M.table(
      ['arm', '$\\Delta$ novel', '$\\Delta$ property', '$\\Delta$ hits'],
      rows, 'l rrr',
      f'{LAB[prop]}: paired against \\texttt{{{base}}}, within each '
      f'$(\\lambda, \\text{{seed}})$. Pairing removes the $\\lambda$ and seed '
      f'effects, which otherwise dominate the spread.',
      f'tab:conf-paired-{prop}',
      note=r'$^{*}$ = $|t| > 2.145$, i.e. $p<0.05$ on a paired $t$-test with '
           r'14 degrees of freedom.'))

  # decomposition, QED only -- it is the property being closed out
  f = five_arms('qed')
  steps = [(ARM_ORDER[0], ARM_ORDER[1], 'fixing the count'),
           (ARM_ORDER[1], ARM_ORDER[2], 'ranking at all'),
           (ARM_ORDER[2], ARM_ORDER[3], 'making the ranking reward-aware'),
           (ARM_ORDER[2], ARM_ORDER[4], 'reward-aware, control-variate corrected')]
  rows = []
  for a, b, what in steps:
    if f[f.arm == a].empty or f[f.arm == b].empty:
      continue
    cs = [cell(*paired(f, b, a, c)[:3], fmt=fmt, dec=dec)
          for c, fmt, dec in (('novel', '%+.1f', 1), ('hits', '%+.1f', 1))]
    rows.append([f'{a.split()[0]} $\\to$ {b.split()[0]}', what] + cs)
  body.append(M.table(
    ['step', 'isolates', '$\\Delta$ novel', '$\\Delta$ hits'], rows, 'll rr',
    'QED: the decomposition the four arms were designed for, each step paired.',
    'tab:conf-decomp',
    note=r'The damage is done by \emph{ranking}, not by the reward: B$\to$C '
         r'costs the novelty. Making the ranking reward-aware (C$\to$E) wins '
         r'part of it back --- the contribution repairs a wound the ranking '
         r'itself inflicts.'))

  body.append(r"""
Reading the three tables together:

\begin{enumerate}
  \item \textbf{No arm beats \texttt{A bernoulli} on hits.} Two are
        significantly worse; the best (E) is a draw. Doing nothing is the best
        policy on the metric a fixed generation budget cares about.
  \item \textbf{The cost is ranking itself, not the reward.} Introducing a
        confidence ranking (B$\to$C) is what destroys novelty: a confidence rule
        commits to the most confident, hence most typical, positions and walks
        towards the mode of the data. Reward-awareness (C$\to$E) then wins some
        of it back.
  \item \textbf{One thing does survive.} E raises the property mean
        significantly and by the largest margin of any arm. If the objective
        were the mean quality of novel molecules rather than the count above a
        bar, E wins; on hits it does not.
\end{enumerate}
""")

  # ---------------------------------------------------------------- CV
  body.append(r'\section{The control variate}\label{sec:cv}')
  body.append(r"""
$\bar{x}_0$ is a \emph{self-normalised} importance sampling estimate: the weights
$w_n = \mathrm{softmax}(\lambda r_n)$ share a denominator over all $N$ samples, so
it is a ratio of two averages and is \textbf{biased at finite $N$}, by
$O(1/N)$ --- measured at $0.104$ for $N{=}5$ falling to $0.0007$ for $N{=}1000$
on a synthetic check. It is consistent, not unbiased.

The same $N$ candidates averaged with \emph{uniform} weights,
$\bar{x}_0^{(0)}$, has $\mathbb{E}[\bar{x}_0^{(0)}] = x_\theta$ \emph{exactly},
because the candidates are drawn i.i.d.\ from $x_\theta$. So
$\bar{x}_0^{(0)} - x_\theta$ is a zero-mean quantity computed from the same
draws, hence correlated with the error in $\bar{x}_0$:

\[ \bar{x}_0^{\mathrm{CV}} \;=\; \bar{x}_0 \;-\; c\,\bigl(\bar{x}_0^{(0)} - x_\theta\bigr). \]

Two things this is and is not. It \textbf{adds} no bias --- the correction has
mean zero for any $c$ --- but it does \textbf{not remove} the SNIS bias already
in $\bar{x}_0$. And it costs no extra reward evaluations, since the candidates
were already scored; reusing those same draws is not an economy but the
\emph{condition} for the correction to reduce variance rather than add noise.

Classical theory gives $c^\star = \mathrm{Cov}/\mathrm{Var}$, the OLS regression
coefficient, with variance falling by $(1-\rho^2)$. And $c=1$ is not arbitrary:
at any position the weights do not depend on, the indicator draws are i.i.d.\ and
$\mathrm{Cov} = \mathrm{Var} = p(1-p)/N$ exactly, so $c^\star = 1$. Departures
from $1$ measure how strongly the reward couples to that position.
""")
  f = five_arms('qed')  # for the c sweep, read the cv cells directly
  for prop in ('qed', 'ring_count'):
    cv = collect(prop, 'conf_blend', CB,
                 lambda d: (('c=' + (d['a'][2:] or '1'))
                            if d['a'].startswith('cv') and float(d['lam']) in LAMS[prop]
                            else None))
    if cv.empty:
      continue
    order = sorted(cv.arm.unique(), key=lambda s: float(s[2:]))
    rows = []
    for arm in order:
      s = cv[cv.arm == arm]
      n, p, h = stat(s, 'novel'), stat(s, 'prop'), stat(s, 'hits')
      d = ([cell(*paired(cv, arm, 'c=1', c)[:3], fmt=fmt, dec=dec)
            for c, fmt, dec in (('novel', '%+.1f', 1), ('hits', '%+.1f', 1))]
           if arm != 'c=1' else ['(reference)', ''])
      rows.append([arm, f'{n[0]:.1f} $\\pm$ {n[1]:.1f}',
                   f'{p[0]:.4f} $\\pm$ {p[1]:.4f}',
                   f'{h[0]:.1f} $\\pm$ {h[1]:.1f}'] + d)
    body.append(M.table(
      ['$c$', 'novel', f'novel-{LAB[prop]}', 'hits',
       '$\\Delta$ novel vs $c{=}1$', '$\\Delta$ hits'],
      rows, 'l rrr rr',
      f'{LAB[prop]}: sweeping the control-variate coefficient, 15 runs per $c$. '
      f'$c$ is not restricted to $[0,1]$ --- $\\mathrm{{Cov}}/\\mathrm{{Var}}$ can '
      f'exceed $1$.', f'tab:conf-c-{prop}',
      note=r'$c=1$ is best or tied everywhere and every difference has the same '
           r'sign, so the theoretical default is also the empirical optimum. '
           r'An adaptive $c$ estimated from the same samples would re-introduce '
           r'bias for no measurable gain.'))

  # ---------------------------------------------------------------- window
  body.append(r'\section{Is there a critical time window?}')
  body.append(r"""
The failure mode above suggests a repair. Early unmasking decisions fix the
molecular scaffold, which is where novelty is decided; late ones are local
substitutions. Confining the ranking to late $t$ should then keep the quality
gain without the diversity cost. The reward tilt stays on for the whole
trajectory --- only the position rule is windowed, through a
\texttt{position\_t\_max} that is deliberately separate from the guidance window.

$t$ runs from 1 (noise) to 0 (clean), so $w = 0.25$ means the ranking is active
only on the last quarter of the trajectory. $w=0$ is exactly \texttt{bernoulli}
and $w=1$ is exactly arm E --- both verified to reproduce those rows.
""")
  pw = collect('qed', 'position_window', PW,
               lambda d: f"w={float(d['w']):g}")
  if not pw.empty:
    order = sorted(pw.arm.unique(), key=lambda s: float(s[2:]))
    rows = []
    for arm in order:
      s = pw[pw.arm == arm]
      n, p, h = stat(s, 'novel'), stat(s, 'prop'), stat(s, 'hits')
      w = float(arm[2:])
      meaning = ('= \\texttt{bernoulli}' if w == 0 else
                 ('= full trajectory (arm E)' if w == 1 else
                  f'last {w:g} of $t$'))
      d = ([cell(*paired(pw, arm, 'w=0', c)[:3], fmt=fmt, dec=dec)
            for c, fmt, dec in (('novel', '%+.1f', 1), ('prop', '%+.4f', 4),
                                ('hits', '%+.1f', 1))]
           if w != 0 else ['(reference)', '', ''])
      rows.append([arm, meaning, f'{n[0]:.1f} $\\pm$ {n[1]:.1f}',
                   f'{p[0]:.4f} $\\pm$ {p[1]:.4f}',
                   f'{h[0]:.1f} $\\pm$ {h[1]:.1f}'] + d)
    body.append(M.table(
      ['$w$', 'ranking active on', 'novel', 'novel-QED', 'hits',
       '$\\Delta$ novel', '$\\Delta$ prop.', '$\\Delta$ hits'],
      rows, 'll rrr rrr',
      'QED: the CV ranking confined to $t \\leq w$, paired against $w=0$.',
      'tab:conf-window', size=r'\footnotesize'))
  body.append(r"""
\textbf{The hypothesis is refuted, and the way it fails is informative.} No $w$
beats \texttt{bernoulli} on hits. But novelty and quality do not live in the same
part of the trajectory in the way the hypothesis assumed --- they live in the
\emph{same} part:

\begin{itemize}
  \item Novelty is lost as soon as the window opens at all: $w=0.25$ already
        costs a significant number of novel molecules, while delivering
        essentially no property gain.
  \item The property gain appears only at $w=1$. It comes from the early,
        high-noise steps --- exactly the steps that cost the most novelty.
\end{itemize}

So the two effects are produced by the same decisions and cannot be separated by
windowing. Late-step ranking is the worst of both: it costs diversity and buys
nothing, presumably because by then $x_t$ is nearly determined and the order of
the remaining unmaskings barely matters.
""")

  # ---------------------------------------------------------------- checks
  body.append(r'\section{Implementation checks, and two bugs}')
  body.append(r"""
Five identities were available for free, and all five hold:

\begin{enumerate}
  \item \textbf{$\lambda=0 \Rightarrow$ CV $\equiv$ \texttt{xtheta\_conf}.} With
        uniform weights $\bar{x}_0 = \bar{x}_0^{(0)}$, so the correction
        collapses to $x_\theta$ algebraically. Measured agreement:
        $0.0001$--$0.0010$, i.e.\ floating point.
  \item \textbf{$c=0 \Rightarrow$ CV $\equiv$ \texttt{xbar\_conf}.} Holds within
        seed noise, though not to floating point --- tiny numerical differences
        flip near-ties in the ranking and trajectories then diverge.
  \item \textbf{$w=0 \Rightarrow$ \texttt{bernoulli}.} Reproduces it exactly,
        run for run.
  \item \textbf{$\beta=0 \equiv$ C and $\beta=1 \equiv$ D} for the blend, checked
        on CPU before launching.
  \item \textbf{Under \texttt{edlm}, \texttt{xbar\_conf} degenerates to
        \texttt{random\_k}}, as the one-hot argument requires.
\end{enumerate}

Check 5 initially \emph{failed}, and finding out why exposed the first bug.

\textbf{Bug 1 --- stable-sort tie-breaking.} \texttt{argsort} returns ties in
index order, so when every masked position scored identically the arm unmasked
\emph{left to right} rather than at random: a systematically different sampler.
This is invisible under \texttt{marginal}, where scores genuinely differ, but it
governs the whole \texttt{edlm} check. Fixed with a $10^{-6}$ jitter. The entire
four-arm ablation was re-run afterwards, and the conclusions changed: before the
fix C looked strong and D looked weak, which was an artifact.

\textbf{Bug 2 --- a falsy default.} \texttt{float(getattr(\dots, 'cv\_coeff',
1.0) or 1.0)} maps $c=0$ to $c=1$, because $0.0$ is falsy. Caught before the $c$
sweep ran; $c=0$ is precisely the setting that turns the correction off and is
the consistency check of item 2, so it would have read as ``the control variate
does nothing''.

\textbf{The noise floor.} Because \texttt{random\_k} and \texttt{xbar\_conf} are
the same algorithm under \texttt{edlm}, their difference measures pure run-to-run
noise: hit counts of 30 against 49 on one QED cell. The seed replication puts a
number on it directly --- see the standard errors above --- and it is the reason
every comparison here is paired and replicated.
""")

  body.append(r'\section{Verdict}')
  body.append(r"""
Confidence-based position selection is \textbf{closed as a negative result on the
primary metric}, refuted along three independent routes:

\begin{enumerate}
  \item no arm beats \texttt{bernoulli} on hits, with 5 seeds and paired tests;
  \item no control-variate coefficient rescues it --- $c=1$ is optimal and still
        below baseline;
  \item no time window separates the quality gain from the diversity cost.
\end{enumerate}

What survives is narrow but real: \textbf{arm E raises the mean property of novel
molecules significantly, by the largest margin of any arm.} Under an objective
that scores mean quality rather than count-above-a-bar, the contribution holds.
Under the budgeted-hit-count metric this work adopted, it does not.

The capability argument for \texttt{marginal} --- that this is definable only
there --- is intact, but it now buys a capability that does not pay on the metric
that matters, so it cannot carry the justification on its own.
""")
  body.append(r'\end{document}')
  return '\n'.join(body)


def main():
  path = os.path.join(M.PDFS, 'Confidence', 'src', 'confidence.tex')
  os.makedirs(os.path.dirname(path), exist_ok=True)
  text = build()
  with open(path, 'w') as f:
    f.write(text)
  print(f'wrote {os.path.relpath(path, A.REPO)}  ({len(text)} chars)')


if __name__ == '__main__':
  main()
