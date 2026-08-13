"""QM9 evaluation for our reward-tilted posterior guidance.

Mirrors the metrics of `guidance_eval/qm9_eval.py` (validity / uniqueness /
novelty and property percentiles) so that the resulting CSVs are directly
comparable with the paper's D-CFG / D-CBG / FUDGE / NOS numbers. The only
differences are that it builds an `our_guidance.OurGuidedDiffusion` and that
it can run without a local checkpoint (loading HuggingFace weights via
`backbone=hf_dit`), since our method needs no trained guidance component.

Usage (from the repo root, with `PYTHONPATH` including `guidance_eval/`):

    python guidance_eval/our_qm9_eval.py \
      data=qm9 data.label_col=qed \
      backbone=hf_dit model=hf model.length=32 \
      model.pretrained_model_name_or_path=kuleshov-group/udlm-qm9 \
      diffusion=uniform parameterization=d3pm \
      time_conditioning=True zero_recon_loss=True \
      guidance=ours guidance.reward=qed guidance.lambda_=10 \
      guidance.num_x0_samples=10 \
      sampling.steps=32 sampling.num_sample_batches=4 \
      sampling.batch_size=64 sampling.use_cache=False \
      eval.disable_ema=True \
      eval.results_csv_path=... eval.generated_samples_path=...
"""

import json
import os
import typing

import datasets
import hydra
import lightning as L
import numpy as np
import omegaconf
import pandas as pd
import rdkit
import torch
from rdkit import Chem as rdChem
from tqdm.auto import tqdm

import dataloader
import our_guidance

rdkit.rdBase.DisableLog('rdApp.error')

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)
omegaconf.OmegaConf.register_new_resolver(
  'if_then_else', lambda condition, x, y: x if condition else y)


def get_mol_property_fn(
    prop: str,
) -> typing.Callable[[rdChem.Mol], typing.Union[int, float]]:
  if prop in our_guidance.REWARD_FNS:
    return our_guidance.REWARD_FNS[prop]
  raise NotImplementedError(
    f"Property function for {prop} not implemented")


def build_model(config, tokenizer):
  """Loads a local checkpoint if given, else builds from config (HF weights)."""
  checkpoint_path = config.eval.get('checkpoint_path', '')
  if checkpoint_path:
    model = our_guidance.OurGuidedDiffusion.load_from_checkpoint(
      checkpoint_path, tokenizer=tokenizer, config=config, logger=False)
  else:
    model = our_guidance.OurGuidedDiffusion(config, tokenizer=tokenizer)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
  model.eval()
  return model


@hydra.main(version_base=None, config_path='../configs',
            config_name='config')
def main(config: omegaconf.DictConfig) -> None:
  # `OurGuidedDiffusion` delegates every method other than `ours` to the
  # parent implementation, so this script doubles as the baseline runner. That
  # matters for the comparison: baselines then share this script's model
  # construction (HuggingFace weights, no local generative checkpoint) and its
  # metric code with our method.
  assert config.guidance is not None, (
    'Set `guidance=ours` (our method) or a baseline such as `guidance=cbg`.')
  L.seed_everything(config.seed)

  qm9_dataset = datasets.load_dataset(
    'yairschiff/qm9', trust_remote_code=True, split='train')
  tokenizer = dataloader.get_tokenizer(config)
  model = build_model(config, tokenizer)

  label_col = config.data.label_col
  mol_property_fn = get_mol_property_fn(label_col)
  guidance_fields = {k.capitalize(): v for k, v in config.guidance.items()}
  result_dicts = []

  # Reference row: the training data itself.
  result_dicts.append({
    'Seed': -1,
    'T': -1,
    'Num Samples': len(qm9_dataset),
    'Valid': 1.0,
    'Unique': 1.0,
    'Novel': 1.0,
  } | {
    f'{label_col.upper()} Mean': np.mean(qm9_dataset[label_col]),
    f'{label_col.upper()} 25%ile': np.percentile(qm9_dataset[label_col], q=25),
    f'{label_col.upper()} Median': np.median(qm9_dataset[label_col]),
    f'{label_col.upper()} 75%ile': np.percentile(qm9_dataset[label_col], q=75),
    f'Novel {label_col.upper()} Mean': np.mean(qm9_dataset[label_col]),
    f'Novel {label_col.upper()} 25%ile': np.percentile(qm9_dataset[label_col], q=25),
    f'Novel {label_col.upper()} Median': np.median(qm9_dataset[label_col]),
    f'Novel {label_col.upper()} 75%ile': np.percentile(qm9_dataset[label_col], q=75),
  } | {k: -1 for k in guidance_fields})

  samples = []
  for _ in tqdm(range(config.sampling.num_sample_batches),
                desc='Gen. batches', leave=False):
    sample = model.sample()
    samples.extend(model.tokenizer.batch_decode(sample))

  valids, invalids, mol_property = [], [], []
  for text in samples:
    smiles = our_guidance.clean_smiles(text)
    try:
      mol = rdChem.MolFromSmiles(smiles)
    except Exception:
      mol = None
    if mol is None or not smiles:
      invalids.append(smiles)
    else:
      valids.append(smiles)
      mol_property.append(mol_property_fn(mol))

  valid = len(valids)
  valid_pct = valid / len(samples)
  unique = len(set(valids))
  novel_smiles = set(valids) - set(qm9_dataset['canonical_smiles'])
  novel = len(novel_smiles)
  unique_pct = unique / valid if valid else 0.
  novel_pct = novel / valid if valid else 0.
  mol_property_novel = [
    mol_property_fn(rdChem.MolFromSmiles(s)) for s in novel_smiles]

  def pct(values, q):
    return np.percentile(values, q=q) if len(values) else 0.

  result_dicts.append({
    'Seed': config.seed,
    'T': config.sampling.steps,
    'Num Samples': len(samples),
    'Valid': valid_pct,
    'Unique': unique_pct,
    'Novel': novel_pct,
  } | {
    f'{label_col.upper()} Mean': np.mean(mol_property) if mol_property else 0.,
    f'{label_col.upper()} 25%ile': pct(mol_property, 25),
    f'{label_col.upper()} Median': pct(mol_property, 50),
    f'{label_col.upper()} 75%ile': pct(mol_property, 75),
    f'Novel {label_col.upper()} Mean': np.mean(mol_property_novel) if mol_property_novel else 0.,
    f'Novel {label_col.upper()} 25%ile': pct(mol_property_novel, 25),
    f'Novel {label_col.upper()} Median': pct(mol_property_novel, 50),
    f'Novel {label_col.upper()} 75%ile': pct(mol_property_novel, 75),
  } | guidance_fields)

  print("Guidance:", ", ".join(
    f"{k} - {v}" for k, v in guidance_fields.items()))
  print(f"\tValid: {valid:,d} / {len(samples):,d} ({100 * valid_pct:0.2f}%) ",
        f"Unique (of valid): {unique:,d} / {valid:,d} ({100 * unique_pct:0.2f}%) ",
        f"Novel (of valid): {novel:,d} / {valid:,d} ({100 * novel_pct:0.2f}%)")
  print(f"\t{label_col.upper()} Mean: "
        f"{np.mean(mol_property) if mol_property else 0.:0.3f}, "
        f"Median: {np.median(mol_property) if mol_property else 0.:0.3f}")
  print(f"\tNovel {label_col.upper()} Mean: "
        f"{np.mean(mol_property_novel) if mol_property_novel else 0.:0.3f}, "
        f"Median: {np.median(mol_property_novel) if mol_property_novel else 0.:0.3f}")

  samples_path = config.eval.get('generated_samples_path', '')
  if samples_path:
    with open(samples_path, 'w') as f:
      json.dump({
        'valid': valids,
        'novel': list(novel_smiles),
        f"{label_col}_valid": mol_property,
        f"{label_col}_novel": mol_property_novel,
      }, f, indent=4)
  csv_path = config.eval.get('results_csv_path', '')
  if csv_path:
    pd.DataFrame.from_records(result_dicts).to_csv(csv_path)


if __name__ == '__main__':
  main()
