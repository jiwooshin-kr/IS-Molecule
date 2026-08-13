"""HF 사전학습 UDLM-QM9 모델로 샘플링 스모크 테스트.

사용법 (서버, 프로젝트 루트에서):
    python scripts/smoke_test_sampling.py \
        data=qm9 data.cache_dir=$PWD/.data_cache \
        backbone=hf_dit model=hf model.length=32 \
        model.pretrained_model_name_or_path=kuleshov-group/udlm-qm9 \
        diffusion=uniform parameterization=d3pm \
        time_conditioning=True zero_recon_loss=True \
        sampling.steps=32 sampling.num_sample_batches=1 \
        sampling.batch_size=8 sampling.use_cache=False
"""
import hydra
import torch
from rdkit import Chem as rdChem

import dataloader
import diffusion


@hydra.main(version_base=None, config_path='../configs',
            config_name='config')
def main(config):
  tokenizer = dataloader.get_tokenizer(config)
  model = diffusion.Diffusion(
    config, tokenizer=tokenizer).to('cuda')
  model.eval()
  with torch.no_grad():
    sample = model.sample()
  texts = tokenizer.batch_decode(sample)
  n_valid = 0
  for t in texts:
    smiles = (t.replace('<bos>', '')
              .replace('<eos>', '')
              .replace('<pad>', ''))
    mol = rdChem.MolFromSmiles(smiles)
    valid = mol is not None and len(smiles) > 0
    n_valid += valid
    print(f"[{'valid' if valid else 'INVALID'}] {smiles}")
  print(f"\n{n_valid}/{len(texts)} valid SMILES")


if __name__ == '__main__':
  main()
