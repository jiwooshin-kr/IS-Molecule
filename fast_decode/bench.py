"""Correctness and speed of `fast_decode` against the reference decode.

Correctness first: a decode that differs anywhere silently corrupts every reward,
so identity against the reference is asserted on random ids, on edge cases, and
on the real QM9 table when it can be loaded.

Run:
    python -m fast_decode.bench            # synthetic table
    python -m fast_decode.bench --real     # also load the QM9 tokenizer
"""

import sys
import time

import numpy as np

from fast_decode.decode import build_lut, decode_ids


def reference(ids, table):
  """Exactly `our_guidance._decode_smiles`, minus the tensor handling."""
  rows = ids.tolist() if hasattr(ids, 'tolist') else ids
  return [''.join([table[i] for i in row]) for row in rows]


def _check(table, rows, length, seed=0, label=''):
  rng = np.random.default_rng(seed)
  ids = rng.integers(0, len(table), size=(rows, length))
  lut = build_lut(table)
  assert lut is not None, f'{label}: table could not be baked'
  got, want = decode_ids(ids, lut), reference(ids, table)
  assert got == want, (
    f'{label}: MISMATCH\n  first differing row: '
    + next(f'{i}: {g!r} != {w!r}' for i, (g, w) in enumerate(zip(got, want))
           if g != w))
  return ids, lut


def main():
  # A table with the same shape of problem as QM9's: single chars, multi-char
  # bracket atoms, and empty strings where special ids were mapped out.
  table = list('CNOFcnos()[]=#123456+-/\\') + [
    '[NH3+]', '[nH]', '[O-]', '[CH-]', '', '', '', '', '']
  for rows, length, label in [(1, 1, 'single'), (7, 32, 'small'),
                              (1000, 32, 'medium')]:
    _check(table, rows, length, label=label)
  print(f'  synthetic table ({len(table)} tokens, width '
        f'{max(len(t) for t in table)}): identical to reference')

  # all-empty and all-longest rows
  ids = np.array([[table.index('')] * 32, [table.index('[NH3+]')] * 32])
  lut = build_lut(table)
  assert decode_ids(ids, lut) == reference(ids, table)
  print('  edge cases (all-empty row, all-widest row): identical')

  table_real = None
  if '--real' in sys.argv:
    try:
      import tokenizer as T
      tok = T.SMILESTokenizer.from_pretrained('yairschiff/qm9-tokenizer')
      size = len(tok)
      toks = tok.convert_ids_to_tokens(list(range(size)))
      special = set(tok.all_special_ids)
      table_real = ['' if i in special else (toks[i] or '')
                    for i in range(size)]
      _check(table_real, 2000, 32, label='qm9')
      print(f'  real QM9 table ({size} tokens): identical to reference')
    except Exception as exc:                                  # noqa: BLE001
      print(f'  real QM9 table: skipped ({type(exc).__name__}: {exc})')

  for rows in (19_200, 192_000):
    ids = np.random.default_rng(1).integers(0, len(table), size=(rows, 32))
    lut = build_lut(table)
    t = time.perf_counter(); reference(ids, table); t_ref = time.perf_counter() - t
    t = time.perf_counter(); decode_ids(ids, lut);  t_fast = time.perf_counter() - t
    print(f'  {rows:>7,} rows: reference {t_ref:6.3f} s   fast {t_fast:6.3f} s'
          f'   speedup {t_ref / t_fast:5.1f}x')


if __name__ == '__main__':
  main()
