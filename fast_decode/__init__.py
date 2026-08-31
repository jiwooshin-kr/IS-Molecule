"""Drop-in fast replacement for the SMILES id -> string decode.

Kept in its own package so the main implementation (`our_guidance.py`) stays
readable and keeps working unchanged when this is absent.
"""

from fast_decode.decode import decode_ids

__all__ = ['decode_ids']
