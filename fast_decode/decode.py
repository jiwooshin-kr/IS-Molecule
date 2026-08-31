"""Fast id -> SMILES decode by memoising identical candidate rows.

The reference implementation in `our_guidance._decode_smiles` is

    [''.join([table[i] for i in row]) for row in ids.cpu().tolist()]

Two things were measured before choosing this approach, on 192k rows of 32
tokens:

  * The cost splits 41 % `.cpu().tolist()` (which builds 192k lists and 6.1M
    int objects) and 59 % the per-row join.
  * Baking the table into a NUL-padded byte lookup and decoding the whole batch
    as one buffer is *slower* (0.7x). The QM9 vocabulary contains bracket atoms
    like `[NH3+]`, so the padded width is 6 while most tokens are 1 character;
    five sixths of the buffer becomes padding that still has to be scanned and
    stripped.

What does work is that the candidate rows repeat, heavily, and increasingly so
along the trajectory. Measured on MDLM with `oversample=10`, the fraction of
*distinct* rows among 32,000 candidates runs 1.00 -> 0.78 -> 0.58 -> 0.16 ->
0.02 -> 0.0013 from the first step to the last: by the end, 32,000 candidates
are 41 distinct sequences. This is the same phenomenon that puts a floor under
the effective sample size.

So: find the distinct rows, decode only those, and scatter the results back.

**This is memoisation, not deduplication.** Every input row receives its own
output string, duplicates included, so the candidate multiset the importance
weights see is untouched. Dropping duplicates would change the weight
distribution and break the estimator; this cannot.
"""

import numpy as np


def decode_ids(ids, table, min_rows=4096):
  """Token ids of shape (rows, L) to a list of `rows` strings.

  Falls back to the plain per-row decode when there are too few rows for the
  uniquing to pay for itself.
  """
  arr = ids.detach().cpu().numpy() if hasattr(ids, 'detach') else np.asarray(ids)
  if arr.ndim != 2:
    raise ValueError(f'expected (rows, L), got {arr.shape}')
  rows = arr.shape[0]
  if rows == 0:
    return []
  if rows < min_rows:
    return [''.join([table[i] for i in row]) for row in arr.tolist()]
  # `np.unique(..., axis=0)` sorts and is comparatively slow; viewing each row as
  # one opaque scalar lets the 1-D path do the work instead.
  contig = np.ascontiguousarray(arr)
  view = contig.view([('', contig.dtype)] * contig.shape[1]).ravel()
  _, first, inverse = np.unique(view, return_index=True, return_inverse=True)
  distinct = [''.join([table[i] for i in row])
              for row in contig[first].tolist()]
  return [distinct[k] for k in inverse]
