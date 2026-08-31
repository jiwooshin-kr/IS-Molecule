"""Checks our D-CBG sweeps against the reported numbers of arXiv 2412.10193v3.

The paper's per-method appendix tables are the only place the (gamma, use_approx)
of each Table 5 row is stated, so those are what we reproduce against:

    Table 20 (p35)  MDLM D-CBG  QED         main-paper row: gamma=3,  approx=False
    Table 23 (p37)  UDLM D-CBG  QED         main-paper row: gamma=10, approx=False
    Table 29 (p40)  MDLM D-CBG  ring count  main-paper row: gamma=10, approx=True
    Table 32 (p42)  UDLM D-CBG  ring count  main-paper row: gamma=8,  approx=False

Unit mismatch, handled in `ours_row`: the paper reports Num. Valid / Num. Novel
as counts out of 1,024 and its Mean column is over *novel* sequences only, while
`our_qm9_eval.py` writes Valid as a fraction of 1,024 and Novel as a fraction of
the valid set.

The z column is (ours - paper mean) / paper std. Two caveats make it indicative
rather than a test:
  * we generate with a single seed and the paper averages five, so the
    denominator carries a sqrt(1 + 1/5) factor (applied, see SEED_FACTOR);
  * the paper's std is over *sampling* seeds on one checkpoint. It contains no
    training variation at all, so for MDLM -- where we use our own 25k-step
    checkpoint rather than theirs -- a large z is expected and is not by itself
    evidence of a broken reproduction. UDLM uses the released HF checkpoint and
    so is the arm that must actually match.

Usage:
    python scripts/ours/check_paper_repro.py            # both properties
    python scripts/ours/check_paper_repro.py qed
"""

import glob
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (gamma, use_approx) -> (valid, valid_sd, novel, novel_sd, prop, prop_sd),
# transcribed from the PDF. Only the gammas we ran are listed.
PAPER = {
  ('qed', 'MDLM'): {   # Table 20
    (1, False): (526.4, 16.53, 170.8, 14.69, 0.56, 0.00),
    (1, True): (476.0, 15.23, 221.4, 7.57, 0.46, 0.01),
    (2, False): (524.8, 22.04, 139.0, 11.79, 0.58, 0.00),
    (2, True): (363.4, 17.57, 172.4, 9.42, 0.47, 0.01),
    (3, False): (417.6, 19.69, 116.6, 8.91, 0.58, 0.00),
    (3, True): (244.0, 13.38, 114.0, 6.16, 0.47, 0.01),
    (4, False): (200.4, 10.31, 66.4, 7.70, 0.58, 0.00),
    (5, False): (24.6, 3.21, 11.6, 2.30, 0.58, 0.01),
    (5, True): (120.6, 6.69, 48.4, 5.41, 0.50, 0.01),
    (10, True): (72.6, 7.13, 21.2, 1.79, 0.55, 0.02),
  },
  ('qed', 'UDLM'): {   # Table 23
    (1, False): (933.4, 7.50, 134.6, 7.33, 0.53, 0.01),
    (1, True): (996.4, 5.86, 132.2, 8.87, 0.47, 0.01),
    (2, False): (911.2, 8.14, 119.8, 16.47, 0.57, 0.01),
    (2, True): (974.8, 8.26, 110.6, 9.53, 0.49, 0.01),
    (3, False): (941.0, 11.31, 96.4, 8.82, 0.58, 0.01),
    (3, True): (925.0, 8.09, 111.4, 9.56, 0.51, 0.01),
    (5, False): (967.6, 1.52, 77.8, 10.83, 0.59, 0.01),
    (5, True): (828.2, 17.85, 86.2, 13.01, 0.53, 0.01),
    (10, False): (994.8, 2.86, 63.8, 8.11, 0.61, 0.00),
    (10, True): (783.2, 13.88, 67.0, 3.08, 0.56, 0.01),
  },
  ('ring_count', 'MDLM'): {   # Table 29
    (1, False): (143.8, 6.61, 121.2, 7.01, 5.00, 0.07),
    (1, True): (455.4, 24.81, 229.4, 15.21, 2.52, 0.11),
    (2, False): (0.4, 0.55, 0.4, 0.55, 3.00, 4.47),
    (2, True): (327.6, 15.82, 176.4, 17.30, 2.82, 0.15),
    (3, True): (223.8, 15.01, 136.0, 9.80, 3.25, 0.26),
    (5, True): (135.0, 9.90, 94.4, 11.06, 4.11, 0.26),
    (8, True): (121.2, 12.91, 92.6, 11.95, 4.54, 0.16),
    (10, True): (113.0, 8.75, 85.6, 8.82, 4.75, 0.23),
  },
  ('ring_count', 'UDLM'): {   # Table 32
    (1, False): (797.4, 9.91, 279.0, 23.37, 4.12, 0.04),
    (1, True): (978.4, 3.91, 166.0, 11.81, 2.49, 0.07),
    (2, False): (829.4, 11.59, 336.4, 10.55, 4.54, 0.03),
    (2, True): (892.2, 12.19, 171.2, 13.59, 3.09, 0.08),
    (3, False): (862.6, 7.73, 363.8, 12.76, 4.70, 0.03),
    (3, True): (763.8, 11.52, 198.8, 10.76, 3.76, 0.09),
    (5, False): (889.2, 12.40, 404.0, 9.64, 4.76, 0.03),
    (5, True): (726.8, 15.64, 245.8, 6.83, 4.47, 0.07),
    (8, False): (897.2, 11.58, 432.0, 19.07, 4.84, 0.02),
    (8, True): (796.6, 15.84, 304.4, 10.01, 4.66, 0.06),
    (10, False): (891.6, 12.18, 431.4, 11.46, 4.76, 0.05),
    (10, True): (816.6, 15.14, 359.8, 21.87, 4.70, 0.05),
  },
}

# The row each model/property contributes to the main paper's Table 5.
HEADLINE = {
  ('qed', 'MDLM'): (3, False),
  ('qed', 'UDLM'): (10, False),
  ('ring_count', 'MDLM'): (10, True),
  ('ring_count', 'UDLM'): (8, False),
}


def ours_row(path):
  """Reads one result CSV into paper units."""
  frame = pd.read_csv(path)
  generated = frame[frame['Seed'] != -1]
  if generated.empty:
    return None
  row = generated.iloc[-1]
  prop_col = next((c for c in frame.columns if c.endswith(' Mean')
                   and not c.startswith('Novel')), None)
  if prop_col is None:
    return None
  prop = prop_col[:-len(' Mean')]
  total = int(row['Num Samples'])
  valid = row['Valid'] * total
  return {
    'gamma': int(float(row['Gamma'])),
    # our_qm9_eval writes the python bool through, so it round-trips as a string.
    'approx': str(row['Use_approx']).strip().lower() == 'true',
    'valid': valid,
    'novel': row['Novel'] * valid,
    'prop': row[f'Novel {prop} Mean'],
  }


# The paper prints the property mean to two decimals, so a std shown as "0.00"
# only means "< 0.005" -- dividing by it literally gives inf on rows that in fact
# agree. Floor the denominator at the half-ulp of the printed value: the z then
# reads as "how many rounding steps apart", which is the most the printed table
# can resolve.
PROP_SD_FLOOR = 0.005

# We contribute one seed; the paper's cell is a mean over five. The spread of
# (ours - theirs) is therefore sqrt(sigma^2 + sigma^2/5) = 1.095 sigma, not
# sigma, so dividing by the printed std alone overstates every deviation by 9 %.
SEED_FACTOR = (1.0 + 1.0 / 5.0) ** 0.5


def z(ours, mean, sd, floor=0.0):
  sd = max(sd, floor) * SEED_FACTOR
  if sd == 0:
    return 0.0 if ours == mean else float('inf')
  return (ours - mean) / sd


def check(prop_dir, prop):
  out = []
  for path in sorted(glob.glob(os.path.join(prop_dir, '*_cbg_*.csv'))):
    tag = os.path.basename(path)[:-len('.csv')]
    model = 'MDLM' if tag.startswith('mdlm_') else 'UDLM'
    mine = ours_row(path)
    if mine is None:
      continue
    key = (mine['gamma'], mine['approx'])
    ref = PAPER.get((prop, model), {}).get(key)
    if ref is None:
      out.append({'model': model, 'gamma': mine['gamma'],
                  'approx': mine['approx'], 'note': 'no paper row'} | mine)
      continue
    v_m, v_s, n_m, n_s, p_m, p_s = ref
    out.append({
      'model': model, 'gamma': mine['gamma'], 'approx': mine['approx'],
      'headline': HEADLINE.get((prop, model)) == key,
      'valid': mine['valid'], 'valid_paper': v_m, 'z_valid': z(mine['valid'], v_m, v_s),
      'novel': mine['novel'], 'novel_paper': n_m, 'z_novel': z(mine['novel'], n_m, n_s),
      'prop': mine['prop'], 'prop_paper': p_m,
      'z_prop': z(mine['prop'], p_m, p_s, PROP_SD_FLOOR),
    })
  # Filename order puts gamma 10 between 1 and 2; sort so each arm reads as a
  # curve in gamma.
  out.sort(key=lambda r: (r['model'], r['approx'], r['gamma']))
  return out


def render(prop, rows):
  print(f"\n{'=' * 96}\n{prop}\n{'=' * 96}")
  print(f"{'':1} {'model':5} {'g':>3} {'approx':6} | "
        f"{'valid':>7} {'paper':>7} {'z':>7} | "
        f"{'novel':>6} {'paper':>6} {'z':>7} | "
        f"{'prop':>5} {'paper':>5} {'z':>7}")
  print('-' * 96)
  for r in rows:
    if 'note' in r:
      print(f"  {r['model']:5} {r['gamma']:>3} {str(r['approx']):6} | "
            f"{r['valid']:>7.0f} {'--':>7} {'--':>7} | "
            f"{r['novel']:>6.0f} {'--':>6} {'--':>7} | "
            f"{r['prop']:>5.2f} {'--':>5} {'--':>7}   ({r['note']})")
      continue
    worst = max(abs(r['z_valid']), abs(r['z_novel']), abs(r['z_prop']))
    flag = '*' if r['headline'] else ('!' if worst > 2 else ' ')
    print(f"{flag} {r['model']:5} {r['gamma']:>3} {str(r['approx']):6} | "
          f"{r['valid']:>7.0f} {r['valid_paper']:>7.1f} {r['z_valid']:>+7.1f} | "
          f"{r['novel']:>6.0f} {r['novel_paper']:>6.1f} {r['z_novel']:>+7.1f} | "
          f"{r['prop']:>5.2f} {r['prop_paper']:>5.2f} {r['z_prop']:>+7.1f}")


def summarise(prop, rows):
  scored = [r for r in rows if 'note' not in r]
  for model in ('MDLM', 'UDLM'):
    sub = [r for r in scored if r['model'] == model]
    if not sub:
      continue
    within = sum(1 for r in sub
                 if max(abs(r['z_valid']), abs(r['z_novel']),
                        abs(r['z_prop'])) <= 2)
    head = next((r for r in sub if r['headline']), None)
    line = f"  {prop:11} {model}: {within}/{len(sub)} rows within 2 sigma on all three metrics"
    if head is not None:
      line += (f"; Table 5 row (g={head['gamma']}, approx={head['approx']}) "
               f"z = {head['z_valid']:+.1f} / {head['z_novel']:+.1f} / "
               f"{head['z_prop']:+.1f}")
    print(line)


def main() -> None:
  wanted = sys.argv[1:] or ['qed', 'ring_count']
  summaries = []
  for prop in wanted:
    prop_dir = os.path.join(REPO, 'results', prop)
    if not os.path.isdir(prop_dir):
      print(f"skipping {prop}: no {prop_dir}")
      continue
    rows = check(prop_dir, prop)
    if not rows:
      print(f"skipping {prop}: no *_cbg_*.csv in {prop_dir}")
      continue
    render(prop, rows)
    summaries.append((prop, rows))
  print(f"\n{'=' * 96}\nsummary  (* = the paper's Table 5 setting, "
        f"! = some metric beyond 2 sigma)\n{'=' * 96}")
  for prop, rows in summaries:
    summarise(prop, rows)


if __name__ == '__main__':
  main()
