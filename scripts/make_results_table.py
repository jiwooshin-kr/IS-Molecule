"""Aggregates results/*.csv from the comparison sweep into LaTeX + Markdown.

`guidance_eval/our_qm9_eval.py` writes one CSV per configuration, each holding
a data-reference row (Seed == -1) and a generated-samples row. This collects
the generated-samples rows into one table.

Usage:
    # pull the CSVs off the server first, e.g.
    #   rsync -av wp03052@143.248.84.179:.../DLRT/results/ ./results/
    python scripts/make_results_table.py results pdfs
"""

import glob
import os
import re
import sys

import pandas as pd

# Ordered so that the baseline rows come first and our sweeps read top to
# bottom; anything unmatched is appended alphabetically.
_TAG_ORDER = ['unguided', 'cbg_', 'ours_']


def parse_tag(tag: str) -> dict:
  """Turns a result filename into descriptive columns."""
  prefix, base = '', re.sub(r'^(final_|s\d+_)', '', tag)
  # Optional `st<steps>_` / `seed<n>_` markers for sweeps that vary the sampling
  # step count or the seed. The authoritative values come from the CSV (see
  # `collect`); these only keep the markers from breaking the patterns below.
  base = re.sub(r'^(st\d+_|seed\d+_)+', '', base)
  if base.startswith('mdlm_'):
    prefix, base = 'MDLM ', base[len('mdlm_'):]
  else:
    prefix = 'UDLM '
  if base == 'unguided':
    return {'Method': f'{prefix}unguided', 'N': '--', 'Strength': '--',
            'Window': '--'}
  match = re.match(r'cbg_(approx|exact)_gamma([\d.]+)$', base)
  if match:
    variant = 'approx.' if match.group(1) == 'approx' else 'exact'
    return {'Method': f'{prefix}D-CBG ({variant})', 'N': '--',
            'Strength': f'gamma={match.group(2)}', 'Window': 'all t'}
  # `aggregate_x0` was the unlabelled default in the early sweeps, so the mode is
  # optional; later sweeps name it explicitly.
  match = re.match(
    r'ours_(?:(marginal|exact|aggregate_x0)_)?'
    r'N(\d+)_lam([\d.]+)(?:_floor([\d.]+))?_win([\d.]+)-([\d.]+)$', base)
  if match:
    mode, n, lam, floor, t_min, t_max = match.groups()
    window = ('all t' if (float(t_min), float(t_max)) == (0.0, 1.0)
              else f'{t_min}-{t_max}')
    strength = f'lambda={lam}'
    if floor is not None:
      strength += f', floor={floor}'
    method = f'{prefix}ours' + (f' ({mode})' if mode else '')
    return {'Method': method, 'N': n, 'Strength': strength,
            'Window': window}
  return {'Method': tag, 'N': '--', 'Strength': '--', 'Window': '--'}


def sort_key(tag: str):
  for i, prefix in enumerate(_TAG_ORDER):
    if tag == prefix or tag.startswith(prefix):
      return (i, tag)
  return (len(_TAG_ORDER), tag)


def collect(results_dir: str) -> pd.DataFrame:
  rows = []
  reference = None
  paths = sorted(glob.glob(os.path.join(results_dir, '*.csv')),
                 key=lambda p: sort_key(
                   os.path.basename(p)[:-len('.csv')]))
  for path in paths:
    tag = os.path.basename(path)[:-len('.csv')]
    frame = pd.read_csv(path)
    prop_cols = [c for c in frame.columns
                 if c.endswith('Mean') or c.endswith('Median')]
    if not prop_cols:
      continue
    prop = prop_cols[0].split(' ')[0]
    generated = frame[frame['Seed'] != -1]
    if generated.empty:
      continue
    row = generated.iloc[-1]
    if reference is None:
      ref_rows = frame[frame['Seed'] == -1]
      if not ref_rows.empty:
        ref = ref_rows.iloc[0]
        n_ref = int(ref['Num Samples'])
        reference = {
          'Method': 'QM9 training data', 'N': '--', 'Strength': '--',
          'Window': '--', 'Steps': '--', 'Seed': '--',
          'Gen.': n_ref, 'Valid': n_ref, 'Unique': n_ref, 'Novel': n_ref,
          f'{prop} mean': ref[f'{prop} Mean'],
          f'novel {prop} mean': ref[f'Novel {prop} Mean'],
          f'novel {prop} med.': ref[f'Novel {prop} Median'],
        }
    # The paper's Table 5 reports raw counts out of the generated samples, and
    # its "Mean" column is over novel sequences only. The CSVs store fractions
    # instead, with Unique/Novel taken as fractions *of the valid* set, so
    # convert to counts here to make the two directly comparable.
    total = int(row['Num Samples'])
    valid_count = row['Valid'] * total
    # `our_qm9_eval.py` stores `config.sampling.steps` in the 'T' column, which
    # is the real step count regardless of what the filename says.
    rows.append(parse_tag(tag) | {
      'Steps': int(row['T']),
      'Seed': int(row['Seed']),
      'Gen.': total,
      'Valid': round(valid_count),
      'Unique': round(row['Unique'] * valid_count),
      'Novel': round(row['Novel'] * valid_count),
      f'{prop} mean': row[f'{prop} Mean'],
      f'novel {prop} mean': row[f'Novel {prop} Mean'],
      f'novel {prop} med.': row[f'Novel {prop} Median'],
    })
  if reference is not None:
    rows.insert(0, reference)
  return pd.DataFrame(rows)


def main() -> None:
  results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'
  out_dir = sys.argv[2] if len(sys.argv) > 2 else 'pdfs'
  os.makedirs(out_dir, exist_ok=True)
  table = collect(results_dir)
  if table.empty:
    print(f"No usable CSVs in {results_dir}/")
    return
  float_cols = table.select_dtypes('number').columns
  table[float_cols] = table[float_cols].round(3)

  md_path = os.path.join(out_dir, 'sweep_table.md')
  with open(md_path, 'w') as f:
    f.write(table.to_markdown(index=False, floatfmt='.3f'))
    f.write('\n')
  tex_path = os.path.join(out_dir, 'sweep_table.tex')
  with open(tex_path, 'w') as f:
    f.write(table.to_latex(index=False, float_format='%.3f',
                           escape=True, longtable=False))
  print(table.to_string(index=False))
  print(f"\nWrote {md_path} and {tex_path}")


if __name__ == '__main__':
  main()
