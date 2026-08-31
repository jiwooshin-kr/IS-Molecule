# Is the UDLM + D-CBG pipeline faithful to upstream?

**Date:** 2026-08-13
**Question:** the D-CBG baseline we compare against — classifier training, guidance
arguments, base model — does the whole pipeline match
`kuleshov-group/discrete-diffusion-guidance`?
**Answer:** the D-CBG mechanism is faithful in every argument, the noisy classifier
is trained with upstream's exact recipe, and both classifiers are healthy and
non-degenerate, so **D-CBG is a fair baseline**. One real deviation remains: our
UDLM base model is the released HuggingFace checkpoint, not the locally trained
one upstream's eval script targets, which means our UDLM rows are not
number-for-number comparable with the paper's tables.

---

## 1. D-CBG guidance arguments: exact match

Upstream `scripts/eval_qm9_guidance.sh` builds, for `GUIDANCE=cbg`
(lines 91, 152, 158-161):

```
guidance=cbg guidance.condition=${CONDITION}          # CONDITION defaults to 1
classifier_model=tiny-classifier classifier_backbone=dit
guidance.classifier_checkpoint_path=${CLASS_CKPT}/checkpoints/best.ckpt
guidance.gamma=${GAMMA}
guidance.use_approx=${USE_APPROX}
```

Ours (`scripts/ours/server_run_comparison.sh:133-145`) is identical, including
`condition=1`, `tiny-classifier`, and both `use_approx` settings. The
`guidance.topk=40 classifier_model.pooling=no_pooling` extras in that upstream
branch are **FUDGE-only** (guarded by `if [ "${GUIDANCE}" = "fudge" ]`), so
correctly absent from ours.

Classifier checkpoint path also matches: upstream maps `udlm`+`qed` →
`outputs/qm9/classifier/qed_uniform_T-0` (line 138) and `mdlm`+`qed` →
`qed_absorbing_state_T-0` (line 136); we use both, per base model.

The classifier is not constructed by our script at all — `Diffusion.sample()`
loads it from `config.guidance.classifier_checkpoint_path`
(`diffusion.py:897-900`), which is upstream's own code path.

## 2. Two upstream flags we omit — both harmless

Upstream's eval passes `data.label_col_pctile=90 data.num_classes=2` explicitly;
our `BASE_ARGS` does not. Verified harmless: **those are already the defaults**
in `configs/data/qm9.yaml` (`label_col_pctile: 90`, `num_classes: 2`), so both
runs use the same values. The classifier target is "top 10 % by QED" either way,
and `guidance.condition=1` means the same class.

Upstream also passes `mode=qm9_eval`. `config.mode` is only read by `main.py` for
dispatch and never by `dataloader.py`, `classifier.py`, or `diffusion.py`; our
standalone eval bypasses `main.py`, so the flag is inert.

## 3. Classifier training: flag-for-flag match

`scripts/train_qm9_classifier.sh:45-68` against
`scripts/ours/server_train_cbg_classifier.sh:35-61` — all 18 flags agree:
`mode=train_classifier`, `diffusion`, `T=0`, `data.label_col`,
`label_col_pctile=90`, `num_classes=2`, `global_batch_size=2048`,
`eval_global_batch_size=4096`, `classifier_backbone=dit`,
`classifier_model=tiny-classifier`, `model.length=32`, `optim.lr=3e-4`,
`cosine_decay_warmup` / `warmup_t=1000` / `lr_min=3e-6`,
`checkpoint_every_n_steps=5000`, `checkpoint_monitor.monitor=val/cross_entropy`,
`val_check_interval=1.0`, `max_steps=25000`, `hydra.run.dir`.

Ours adds only `data.cache_dir`, `loader.num_workers=0`,
`loader.persistent_workers=False` — environment.

## 4. Both classifiers are healthy — D-CBG is a fair baseline

Extracted from the offline wandb runs (396 validation evaluations each):

| classifier | best val/CE | @ step | acc | precision | recall |
|:---|---:|---:|---:|---:|---:|
| `qed_uniform_T-0` (UDLM) | 0.2089 | 17198 | 0.9255 | 0.8087 | 0.8087 |
| `qed_absorbing_state_T-0` (MDLM) | 0.1791 | 23120 | 0.9286 | 0.7506 | 0.7506 |

The split is the 90th percentile, so the base rate is 10 % and a trivial
all-negative predictor would score 0.90 accuracy with **zero** recall. Both
classifiers reach ~0.75–0.81 precision *and* recall, so they carry real signal
across the noise levels. This matters for the paper: it means the D-CBG rows —
including `s1024_cbg_exact_gamma5` reaching QED 0.5999, above our best 0.5484 —
reflect a properly trained baseline, not a crippled one. The comparison is fair
and that gap is real.

Convergence (mean val/CE per window):

| window | uniform | absorbing_state |
|:---|---:|---:|
| 0–5 k | 0.2538 | 0.2242 |
| 10–15 k | 0.2230 | 0.1939 |
| 20–25 k | 0.2159 | 0.1898 |

**Caveat:** the absorbing-state classifier's best is at step 23120 of 25000, so
it had not fully plateaued. MDLM D-CBG may be very slightly understated. The
uniform one peaked at 17198 and is settled.

## 5. The one real deviation: UDLM base weights

| | upstream | ours |
|:---|:---|:---|
| weights | `outputs/qm9/udlm_no-guidance/checkpoints/best.ckpt` (trained locally by `train_qm9_no-guidance.sh MODEL=udlm`) | `kuleshov-group/udlm-qm9` (released HF checkpoint) |
| loader | `model=small backbone=dit` | `model=hf backbone=hf_dit` |
| EMA | enabled (`eval.disable_ema` defaults to `False`) | **disabled** (`eval.disable_ema=True`) |
| trained length | 32 | HF config says `model_length: 128`, evaluated at 32 |

Three consequences:

1. **Our UDLM rows are not number-for-number comparable with the paper's
   tables** — different weights. Internal comparisons (ours vs D-CBG on the same
   HF checkpoint) are unaffected and remain apples-to-apples, which is what the
   comparison script was built for.
2. **EMA asymmetry.** Upstream evaluates EMA weights; a plain HF checkpoint has
   no EMA shadow parameters, so `disable_ema=True` is forced. Not a bug, but it
   is a different evaluation of the model than upstream performs.
3. The released checkpoint was trained at sequence length 128 and we run it at
   32. Empirically fine — unguided UDLM gives 97.85 % validity at T=32, the
   strongest base number we have — but it is an untested regime.

The substitution is sanctioned by upstream's README (`README.md:231` documents
`model.pretrained_model_name_or_path="kuleshov-group/udlm-qm9"`), and it is what
makes our method runnable without training a generative model. It just needs
stating wherever UDLM numbers are reported next to the paper's.

---

## Housekeeping

Both classifier directories still carry the five periodic snapshots
(`79-5000` … `396-25000`, 341 MB each) alongside `best.ckpt` and `last.ckpt`:
2.3 GB per directory, **~3.4 GB reclaimable** by the same cleanup applied to
`mdlm_no-guidance`. The `outputs/qm9/2026.08.*/` hydra run directories total
about 3 MB and are harmless.
