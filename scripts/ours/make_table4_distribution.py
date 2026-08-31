"""Distributional view of Table~\\ref{tab:headline} plus the Pareto figure.

`guidance_eval/our_qm9_eval.py` dumps every valid/novel sample and its QED into
`results/<tag>_samples.json`, so the head-to-head rows can be re-read as
distributions instead of the two summary means the CSV carries. This writes

    <out_dir>/table4_dist.tex      quantiles / max / top-k / count above threshold
    <out_dir>/table4_quartiles.tex mean of each quartile bin
    <out_dir>/table4_dist.md       both of the above, for reading in the terminal
    <out_dir>/pareto_qed.pdf       quality-vs-novelty Pareto panels
    <out_dir>/novel_qed_dist.pdf   box + strip plot of the novel QED distributions

Usage:
    python scripts/ours/make_table4_distribution.py <results_dir> [out_dir]  # out_dir defaults to results_dir
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Figures are sized to the 6.5in text block of results.tex and included at
# `width=\textwidth`, so points here are points on the page.
FIG_WIDTH = 6.5
plt.rcParams.update({
  'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
  'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
  'figure.dpi': 200,
})

# The four families of Table 4, in the same order. `block` drives the midrules
# and the figure legend; `marker` separates the three mixture-sampling variants
# within the Ours family so that the figure needs only two hues.
ROWS = [
  ('Unguided', 'Unguided UDLM', '---', 's1024_unguided'),
  ('D-CBG (approx.)', 'D-CBG (approx.)', r'$\gamma = 1$',
   's1024_cbg_approx_gamma1'),
  ('D-CBG (approx.)', 'D-CBG (approx.)', r'$\gamma = 2$',
   's1024_cbg_approx_gamma2'),
  ('D-CBG (approx.)', 'D-CBG (approx.)', r'$\gamma = 5$',
   's1024_cbg_approx_gamma5'),
  ('D-CBG (exact)', 'D-CBG (exact)', r'$\gamma = 1$', 's1024_cbg_exact_gamma1'),
  ('D-CBG (exact)', 'D-CBG (exact)', r'$\gamma = 2$', 's1024_cbg_exact_gamma2'),
  ('D-CBG (exact)', 'D-CBG (exact)', r'$\gamma = 5$', 's1024_cbg_exact_gamma5'),
  ('Ours', 'Ours', r'$N{=}100$, $\lambda{=}5000$',
   's1024_ours_N100_lam5000_win0.0-1.0'),
  ('Ours', 'Ours', r'$N{=}300$, $\lambda{=}5000$',
   's1024_ours_N300_lam5000_win0.0-1.0'),
  ('Ours', 'Ours', r'$N{=}500$, $\lambda{=}5000$',
   's1024_ours_N500_lam5000_win0.0-1.0'),
  ('Ours', 'Ours', r'$N{=}500$, $\lambda{=}1000$, $t \le 0.75$',
   's1024_ours_N500_lam1000_win0.0-0.75'),
  ('Ours (marginal)', 'Ours (marginal)', r'$N{=}100$, $\lambda{=}5000$',
   's1024_ours_marginal_N100_lam5000_win0.0-1.0'),
  ('Ours (marginal)', 'Ours (marginal)', r'$N{=}300$, $\lambda{=}5000$',
   's1024_ours_marginal_N300_lam5000_win0.0-1.0'),
  ('Ours (marginal)', 'Ours (marginal)', r'$N{=}500$, $\lambda{=}5000$',
   's1024_ours_marginal_N500_lam5000_win0.0-1.0'),
  ('Ours (marginal)', 'Ours (marginal)',
   r'$N{=}500$, $\lambda{=}1000$, $t \le 0.75$',
   's1024_ours_marginal_N500_lam1000_win0.0-0.75'),
  ('Ours (exact)', 'Ours (exact)', r'$N{=}100$, $\lambda{=}5000$',
   's1024_ours_exact_N100_lam5000_win0.0-1.0'),
  ('Ours (exact)', 'Ours (exact)', r'$N{=}300$, $\lambda{=}5000$',
   's1024_ours_exact_N300_lam5000_win0.0-1.0'),
  ('Ours (exact)', 'Ours (exact)', r'$N{=}500$, $\lambda{=}5000$',
   's1024_ours_exact_N500_lam5000_win0.0-1.0'),
  ('Ours (exact)', 'Ours (exact)', r'$N{=}500$, $\lambda{=}1000$, $t \le 0.75$',
   's1024_ours_exact_N500_lam1000_win0.0-0.75'),
]

# Fixed hues, categorical slots 1 (blue) and 2 (orange) of the reference
# palette, plus neutral ink for the unguided anchor. Two hues suffice because
# marker shape carries the variant within each family.
BLUE, ORANGE, INK, GRID = '#2a78d6', '#eb6834', '#52514e', '#d8d7d2'
STYLE = {
  'Unguided':        dict(color=INK, marker='X', fill=INK),
  'D-CBG (approx.)': dict(color=ORANGE, marker='o', fill='none'),
  'D-CBG (exact)':   dict(color=ORANGE, marker='o', fill=ORANGE),
  'Ours':            dict(color=BLUE, marker='^', fill='none'),
  'Ours (marginal)': dict(color=BLUE, marker='s', fill='none'),
  'Ours (exact)':    dict(color=BLUE, marker='D', fill=BLUE),
}

TOP_K = 20        # budget-matched depth: top 20 novel molecules out of 1024
THRESHOLD = 0.6   # well above the QM9 upper quartile (0.515)
BOOT = 4000       # subsamples for the sample-size-matched max


def stats(qed: np.ndarray, match_n: int, rng: np.random.Generator) -> dict:
  """Distributional summaries of one configuration's novel QED values."""
  descending = np.sort(qed)[::-1]
  # `max` grows with the number of novel samples, and the rows differ by 4x in
  # that count, so also report the max of a random subsample of the size of the
  # smallest row -- that column is comparable across rows, the raw max is not.
  matched = (qed.max() if len(qed) <= match_n else
             np.mean([rng.choice(qed, match_n, replace=False).max()
                      for _ in range(BOOT)]))
  quarters = np.array_split(np.sort(qed), 4)
  return {
    'n': len(qed),
    'mean': qed.mean(),
    'q1': np.percentile(qed, 25),
    'median': np.median(qed),
    'q3': np.percentile(qed, 75),
    'max': qed.max(),
    'max_matched': matched,
    f'top{TOP_K}': descending[:TOP_K].mean(),
    'above': int((qed > THRESHOLD).sum()),
    'quartile_means': [q.mean() for q in quarters],
  }


def collect(results_dir: str) -> list:
  rows = []
  for block, method, setting, tag in ROWS:
    path = os.path.join(results_dir, f'{tag}_samples.json')
    if not os.path.exists(path):
      print(f'  missing: {path}')
      continue
    with open(path) as f:
      payload = json.load(f)
    rows.append({'block': block, 'method': method, 'setting': setting,
                 'tag': tag, 'qed': np.asarray(payload['qed_novel'], float),
                 'qed_valid': np.asarray(payload['qed_valid'], float)})
  match_n = min(len(r['qed']) for r in rows)
  rng = np.random.default_rng(0)
  for row in rows:
    row.update(stats(row['qed'], match_n, rng))
  return rows, match_n


def _bold(value: float, best: float, fmt: str = '.3f') -> str:
  """Bolds the best cell, comparing at printed precision so ties both bold."""
  text = format(value, fmt)
  return rf'\textbf{{{text}}}' if text == format(best, fmt) else text


def write_tex(rows: list, match_n: int, out_dir: str) -> None:
  cols = ['n', 'mean', 'q1', 'median', 'q3', 'max', 'max_matched',
          f'top{TOP_K}', 'above']
  best = {c: max(r[c] for r in rows) for c in cols}
  head = (r'Method & Setting & Novel & Mean & Q1 & Med. & Q3 & Max & '
          rf'Max$_{{{match_n}}}$ & Top-{TOP_K} & $>{THRESHOLD}$ \\')
  lines = [r'\begin{tabular}{ll rrrrr rrr r}', r'\toprule', head, r'\midrule',
           r'QM9 data & --- & --- & 0.465 & 0.427 & 0.473 & 0.515 & --- & '
           r'--- & --- & --- \\', r'\midrule']
  previous = rows[0]['block']
  for row in rows:
    if row['block'] != previous:
      lines.append(r'\midrule')
      previous = row['block']
    cells = [row['method'], row['setting'], _bold(row['n'], best['n'], 'd')]
    cells += [_bold(row[c], best[c]) for c in cols[1:-1]]
    cells.append(_bold(row['above'], best['above'], 'd'))
    lines.append(' & '.join(cells) + r' \\')
  lines += [r'\bottomrule', r'\end{tabular}']
  path = os.path.join(out_dir, 'table4_dist.tex')
  with open(path, 'w') as f:
    f.write('\n'.join(lines) + '\n')

  lines = [r'\begin{tabular}{ll rrrr r}', r'\toprule',
           r'Method & Setting & Bottom 25\% & 25--50\% & 50--75\% & Top 25\% & '
           r'Spread \\', r'\midrule']
  previous = rows[0]['block']
  for row in rows:
    if row['block'] != previous:
      lines.append(r'\midrule')
      previous = row['block']
    means = row['quartile_means']
    cells = [row['method'], row['setting']]
    cells += [format(m, '.3f') for m in means]
    cells.append(format(means[-1] - means[0], '.3f'))
    lines.append(' & '.join(cells) + r' \\')
  lines += [r'\bottomrule', r'\end{tabular}']
  with open(os.path.join(out_dir, 'table4_quartiles.tex'), 'w') as f:
    f.write('\n'.join(lines) + '\n')


def write_md(rows: list, match_n: int, out_dir: str) -> None:
  header = (f'| Method | Setting | Novel | Mean | Q1 | Med | Q3 | Max | '
            f'Max@{match_n} | Top-{TOP_K} | >{THRESHOLD} | Q1 bin | Q2 bin | '
            f'Q3 bin | Q4 bin |')
  out = [header, '|' + '---|' * 15]
  for row in rows:
    setting = (row['setting'].replace('$', '').replace('{=}', '=')
               .replace(r'\gamma', 'gamma').replace(r'\lambda', 'lambda')
               .replace(r'\le', '<=').replace('---', '-'))
    quart = ' | '.join(f'{m:.3f}' for m in row['quartile_means'])
    out.append(
      f"| {row['method']} | {setting} | {row['n']} | {row['mean']:.3f} | "
      f"{row['q1']:.3f} | {row['median']:.3f} | {row['q3']:.3f} | "
      f"{row['max']:.3f} | {row['max_matched']:.3f} | "
      f"{row[f'top{TOP_K}']:.3f} | {row['above']} | {quart} |")
  path = os.path.join(out_dir, 'table4_dist.md')
  with open(path, 'w') as f:
    f.write('\n'.join(out) + '\n')


def _pareto_front(points: list) -> list:
  """Points not dominated on (x up, y up), ordered by x."""
  front = [p for p in points
           if not any(q[0] >= p[0] and q[1] >= p[1] and q[:2] != p[:2]
                      for q in points)]
  return sorted(front)


def _staircase(front: list, xmin: float, ymin: float) -> tuple:
  """The boundary of the dominated region: y(x) = max{y' : x' >= x}.

  Drawn as a staircase rather than a polyline through the front, because the
  segment between two front points is not attainable -- 115 novel molecules do
  not come with the QED of the 82-molecule configuration.
  """
  path = [(xmin, front[0][1])]
  for i, (x, y) in enumerate(front):
    path.append((x, y))
    path.append((x, front[i + 1][1] if i + 1 < len(front) else ymin))
  return [p[0] for p in path], [p[1] for p in path]


def _scatter_panel(ax, rows, ykey, ylabel, title, annotate):
  for row in rows:
    style = STYLE[row['block']]
    ax.plot(row['n'], row[ykey], marker=style['marker'], markersize=6,
            markerfacecolor=style['fill'], markeredgecolor=style['color'],
            markeredgewidth=1.3, linestyle='none', zorder=3)
  ax.margins(0.1)
  xlim, ylim = ax.get_xlim(), ax.get_ylim()
  front = _pareto_front([(r['n'], r[ykey]) for r in rows])
  ax.plot(*_staircase(front, xlim[0], ylim[0]), color=INK, linewidth=1.0,
          linestyle=(0, (4, 3)), alpha=0.55, zorder=2)
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)
  for label, row in annotate.items():
    ax.annotate(label, (row['n'], row[ykey]),
                textcoords='offset points', xytext=(7, -3), fontsize=7,
                color=STYLE[row['block']]['color'])
  ax.set_xlabel('Novel molecules per 1024 samples')
  ax.set_ylabel(ylabel)
  ax.set_title(title, fontsize=9.5, loc='left')
  ax.grid(True, color=GRID, linewidth=0.6)
  ax.set_axisbelow(True)
  for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
  for side in ('left', 'bottom'):
    ax.spines[side].set_color(INK)


def plot_pareto(rows: list, out_dir: str) -> None:
  by_tag = {r['tag']: r for r in rows}
  annotate = {
    r'$\gamma$=5': by_tag['s1024_cbg_exact_gamma5'],
    r'$\gamma$=2': by_tag['s1024_cbg_exact_gamma2'],
    'N=500': by_tag['s1024_ours_exact_N500_lam5000_win0.0-1.0'],
  }
  fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 3.2))
  _scatter_panel(axes[0], rows, 'mean', 'Novel QED, mean',
                 '(a) Mean quality vs. novelty', annotate)
  _scatter_panel(axes[1], rows, f'top{TOP_K}',
                 f'Novel QED, top-{TOP_K} mean',
                 f'(b) Top-{TOP_K} quality vs. novelty', annotate)
  handles = [Line2D([], [], linestyle='none', marker=s['marker'],
                    markersize=6, markerfacecolor=s['fill'],
                    markeredgecolor=s['color'], markeredgewidth=1.3, label=name)
             for name, s in STYLE.items()]
  handles.append(Line2D([], [], color=INK, linewidth=1.0,
                        linestyle=(0, (4, 3)), alpha=0.55,
                        label='Pareto frontier'))
  fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
             bbox_to_anchor=(0.5, -0.01))
  fig.tight_layout(rect=(0, 0.13, 1, 1))
  for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(out_dir, f'pareto_qed.{ext}'), dpi=200)
  plt.close(fig)


def plot_distributions(rows: list, out_dir: str) -> None:
  """Box + strip plot of the novel QED distributions behind the two means."""
  by_tag = {r['tag']: r for r in rows}
  picks = [
    ('Unguided', 's1024_unguided'),
    ('D-CBG exact\n$\\gamma{=}1$', 's1024_cbg_exact_gamma1'),
    ('D-CBG exact\n$\\gamma{=}2$', 's1024_cbg_exact_gamma2'),
    ('D-CBG exact\n$\\gamma{=}5$', 's1024_cbg_exact_gamma5'),
    ('Ours exact\n$N{=}100$', 's1024_ours_exact_N100_lam5000_win0.0-1.0'),
    ('Ours exact\n$N{=}300$', 's1024_ours_exact_N300_lam5000_win0.0-1.0'),
    ('Ours exact\n$N{=}500$', 's1024_ours_exact_N500_lam5000_win0.0-1.0'),
  ]
  fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.1))
  rng = np.random.default_rng(1)
  for i, (label, tag) in enumerate(picks):
    row = by_tag[tag]
    color = STYLE[row['block']]['color']
    values = row['qed']
    ax.scatter(rng.normal(i, 0.055, len(values)), values, s=3, alpha=0.25,
               color=color, linewidths=0, zorder=2)
    box = ax.boxplot([values], positions=[i], widths=0.42, whis=(5, 95),
                     showfliers=False, patch_artist=True, zorder=3)
    box['boxes'][0].set(facecolor='white', edgecolor=color, linewidth=1.2,
                        alpha=0.85)
    for part in ('whiskers', 'caps', 'medians'):
      for artist in box[part]:
        artist.set(color=color, linewidth=1.2)
    ax.plot(i, values.mean(), marker='_', markersize=13, color=color,
            markeredgewidth=1.8, zorder=4)
    ax.annotate(f'n={len(values)}\n{row["above"]} above 0.6', (i, 0.175),
                ha='center', va='center', fontsize=6.5, color=INK)
  # Reference lines get a legend rather than in-plot labels: at this aspect
  # ratio every y around 0.465 and 0.6 has a box or a dot in it.
  refs = [
    ax.axhline(0.465, color=INK, linewidth=0.9, linestyle=(0, (4, 3)),
               alpha=0.6, zorder=1, label='QM9 data mean, 0.465'),
    ax.axhline(THRESHOLD, color=INK, linewidth=0.9, linestyle=(0, (1, 2)),
               alpha=0.6, zorder=1, label=f'QED = {THRESHOLD}'),
  ]
  ax.legend(handles=refs, loc='upper left', frameon=False, fontsize=6.5,
            handlelength=2.6, borderaxespad=0.2, labelspacing=0.25)
  ax.set_xlim(-0.6, len(picks) - 0.45)
  ax.set_ylim(0.145, 0.755)
  ax.set_xticks(range(len(picks)))
  ax.set_xticklabels([p[0] for p in picks])
  ax.set_ylabel('QED (novel samples)')
  ax.set_title('Box: quartiles and median; whiskers: 5th\u201395th percentile; '
               'bar: mean', loc='left')
  ax.grid(True, axis='y', color=GRID, linewidth=0.6)
  ax.set_axisbelow(True)
  for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
  for side in ('left', 'bottom'):
    ax.spines[side].set_color(INK)
  fig.tight_layout()
  for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(out_dir, f'novel_qed_dist.{ext}'), dpi=200)
  plt.close(fig)


def main() -> None:
  results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'
  # Default alongside the inputs, NOT into `pdfs/` -- see the same note in
  # make_results_table.py. `pdfs/` is local-only.
  out_dir = sys.argv[2] if len(sys.argv) > 2 else results_dir
  os.makedirs(out_dir, exist_ok=True)
  rows, match_n = collect(results_dir)
  write_tex(rows, match_n, out_dir)
  write_md(rows, match_n, out_dir)
  plot_pareto(rows, out_dir)
  plot_distributions(rows, out_dir)
  with open(os.path.join(out_dir, 'table4_dist.md')) as f:
    print(f.read())
  print(f'Wrote table4_dist.tex, table4_quartiles.tex, table4_dist.md, '
        f'pareto_qed.pdf, novel_qed_dist.pdf to {out_dir}/ '
        f'(matched max at n={match_n})')


if __name__ == '__main__':
  main()
