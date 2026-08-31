#!/bin/bash
rsync -av --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.DS_Store' \
  --exclude '.ipynb_checkpoints' \
  --exclude '.idea' \
  --exclude '.claude' \
  --exclude 'CLAUDE.md' \
  --exclude 'outputs' \
  --exclude 'watch_folder' \
  --exclude 'results' \
  --exclude '.hf_cache' \
  --exclude 'wandb' \
  --exclude 'pdfs' \
  --exclude '*.pdf' \
  --exclude '.data_cache' \
  --exclude '*.ckpt' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  ./ wp03052@143.248.84.179:/home/aailab/wp03052/Synthetic-Data/Molecule
