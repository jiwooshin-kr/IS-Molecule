"""Does MDLM sampling leave <mask> tokens in the decoded output?

`guidance_eval/qm9_eval.py` (the original harness) strips only <bos>, <eos> and
<pad> before handing the string to RDKit, while
`our_guidance.clean_smiles` also strips <mask>. For an absorbing-state model
those differ exactly when a position is still masked at the end of sampling: the
original counts such a sample invalid, ours strips the marker and may parse the
remainder as a valid molecule. That would inflate our validity numbers relative
to the paper's.

This script samples one batch and reports, per cleaning function, how many
sequences contain a residual mask marker and how the validity count differs.

Usage: same args as our_qm9_eval.py, e.g.

    python guidance_eval/diag_mask_residue.py \
      data=qm9 data.label_col=qed diffusion=absorbing_state \
      parameterization=subs T=0 time_conditioning=False \
      zero_recon_loss=False training.guidance=null \
      backbone=dit model=small model.length=32 \
      eval.checkpoint_path=.../best.ckpt \
      sampling.steps=32 sampling.batch_size=64 \
      guidance=ours guidance.t_min=2.0 guidance.t_max=2.0
"""

import os
import typing

import hydra
import lightning as L
import omegaconf
import rdkit
import torch
from rdkit import Chem as rdChem

import dataloader
import our_guidance

rdkit.rdBase.DisableLog('rdApp.error')
rdkit.rdBase.DisableLog('rdApp.warning')

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)
omegaconf.OmegaConf.register_new_resolver(
  'if_then_else', lambda condition, x, y: x if condition else y)


def original_clean(text: str) -> str:
  """Exactly what guidance_eval/qm9_eval.py does (note: no <mask>)."""
  return text.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '')


def parses(smiles: str) -> bool:
  if not smiles:
    return False
  try:
    return rdChem.MolFromSmiles(smiles) is not None
  except Exception:
    return False


@hydra.main(version_base=None, config_path='../configs',
            config_name='config')
def main(config: omegaconf.DictConfig) -> None:
  L.seed_everything(config.seed)
  tokenizer = dataloader.get_tokenizer(config)
  ckpt = config.eval.get('checkpoint_path', '')
  if ckpt:
    model = our_guidance.OurGuidedDiffusion.load_from_checkpoint(
      ckpt, tokenizer=tokenizer, config=config, logger=False)
  else:
    model = our_guidance.OurGuidedDiffusion(config, tokenizer=tokenizer)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
  model.eval()

  mask_token = tokenizer.mask_token
  print(f'mask_token={mask_token!r} mask_index={model.mask_index}')

  raw: typing.List[str] = []
  for _ in range(config.sampling.num_sample_batches):
    sample = model.sample()
    # Count residual masks on the *token ids*, which is unambiguous.
    n_masked_pos = int((sample == model.mask_index).sum().item())
    n_masked_seq = int((sample == model.mask_index).any(dim=-1).sum().item())
    print(f'  batch: {n_masked_pos} masked positions across '
          f'{n_masked_seq}/{sample.shape[0]} sequences')
    raw.extend(model.tokenizer.batch_decode(sample))

  n = len(raw)
  orig = [original_clean(t) for t in raw]
  ours = [our_guidance.clean_smiles(t) for t in raw]

  contains_marker = sum(1 for s in orig if mask_token and mask_token in s)
  differ = sum(1 for a, b in zip(orig, ours) if a.strip() != b)
  v_orig = sum(1 for s in orig if parses(s))
  v_ours = sum(1 for s in ours if parses(s))

  print()
  print(f'samples                              : {n}')
  print(f'decoded strings containing {mask_token!r:>10}: {contains_marker}')
  print(f'strings where the two cleanings differ: {differ}')
  print(f'valid under original cleaning        : {v_orig} '
        f'({100 * v_orig / n:.2f}%)')
  print(f'valid under our cleaning             : {v_ours} '
        f'({100 * v_ours / n:.2f}%)')
  print(f'validity inflation from <mask> strip : '
        f'{100 * (v_ours - v_orig) / n:+.2f} points')


if __name__ == '__main__':
  main()
