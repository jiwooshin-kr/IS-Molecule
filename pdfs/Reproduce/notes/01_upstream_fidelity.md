# Is our MDLM setup faithful to `kuleshov-group/discrete-diffusion-guidance`?

**Date:** 2026-08-13
**Question:** the MDLM we run on QM9 — is it the upstream implementation, configured
the way upstream configures it?
**Answer:** the model code and the training configuration are faithful. Five
deviations exist in the *evaluation* path; four are harmless (three verified by
measurement, one is a robustness improvement), and one — `sampling.use_cache` —
is distribution-neutral but changes the NFE accounting enough to revise the
conclusions of `00_base_model_health.md`.

---

## 1. Model code: unmodified upstream

`git status` on the core files is clean; nothing upstream was edited:

```
diffusion.py  main.py  dataloader.py  models/  configs/  noise_schedule.py  classifier.py
→ no modifications
```

Everything we added is a *new* file: `our_guidance.py`,
`configs/guidance/ours.yaml`, `guidance_eval/our_qm9_eval.py`,
`guidance_eval/diag_mask_residue.py`, `scripts/server_*.sh`. So MDLM is
literally upstream's implementation — `diffusion=absorbing_state
parameterization=subs`, `Diffusion._ddpm_denoise`.

No MDLM-QM9 checkpoint exists upstream to download: the HuggingFace collection
referenced in `README.md:214-215` publishes only `udlm-lm1b` and `udlm-qm9`.
Training our own was necessary.

## 2. Training configuration: matches upstream flag for flag

Upstream `scripts/train_qm9_no-guidance.sh` with `MODEL=mdlm` (lines 51-57,
72-99) against our `scripts/server_train_mdlm_qm9.sh`. All 24 flags agree:

| flag | upstream | ours |
|:---|:---|:---|
| `diffusion` | absorbing_state | ✓ |
| `parameterization` | subs | ✓ |
| `T` | 0 | ✓ |
| `time_conditioning` | False | ✓ |
| `zero_recon_loss` | False | ✓ |
| `data.label_col` / `_pctile` / `num_classes` | null | ✓ |
| `eval.generate_samples` | False | ✓ |
| `loader.global_batch_size` | 2048 | ✓ |
| `loader.eval_global_batch_size` | 4096 | ✓ |
| `backbone` / `model` / `model.length` | dit / small / 32 | ✓ |
| `optim.lr` | 3e-4 | ✓ |
| `lr_scheduler` / `warmup_t` / `lr_min` | cosine_decay_warmup / 1000 / 3e-6 | ✓ |
| `training.guidance` | null | ✓ |
| `training.compute_loss_on_pad_tokens` | True | ✓ |
| `training.use_simple_ce_loss` | False | ✓ |
| `checkpoint_every_n_steps` | 5000 | ✓ |
| `trainer.max_steps` | 25000 | ✓ |
| `trainer.val_check_interval` | 1.0 | ✓ |
| `hydra.run.dir` | outputs/qm9/mdlm_no-guidance | ✓ |

Ours adds only environment-specific flags: `data.cache_dir` (keeps the cache
inside the project) and `loader.num_workers=0 loader.persistent_workers=False`
(forked workers touching CUDA crash on this box).

**GPU count is not a confound.** Upstream runs `srun` with `--gres=gpu:4`; we
ran 2 GPUs. But `loader.global_batch_size` is genuinely global —
`configs/config.yaml:29` sets `loader.batch_size = div_up(global_batch_size,
devices*nodes)` and line 73 sets `accumulate_grad_batches` to compensate — so
the optimizer sees a 2048-example batch per step either way, and 25 k steps
means the same amount of data.

## 3. Sampler: our unguided fallback equals `_ddpm_denoise`

Our `_our_denoise` falls through to the base sampler when `t` is outside
`[t_min, t_max]` (`our_guidance.py:297-308`). That branch is line-for-line
equivalent to upstream `Diffusion._ddpm_denoise` (`diffusion.py:1231-1253`):

| | upstream | ours |
|:---|:---|:---|
| posterior | `x_theta*(mc_t-mc_s)`; `[mask]=mc_s`; `/= mc_t` | identical (`our_guidance.py:246-249`) |
| sampling | `_sample_categorical(q_xs)` | identical |
| carry-over | `copy_flag` zero/one-hot + `torch.where` | identical |

The surrounding loop in `our_guidance._diffusion_sample:409-454` also matches
`diffusion._diffusion_sample:1121-1209` — same `timesteps` linspace, same `dt`,
same `T>0` branch, same sigma/move_chance construction, same NFE counting, same
cache invalidation, same `xt = xs`.

So the T-sweep in `00_base_model_health.md` measured the genuine upstream base
sampler.

## 4. Evaluation harness: five deviations

`guidance_eval/our_qm9_eval.py` vs upstream `guidance_eval/qm9_eval.py`.
Validity / uniqueness / novelty / percentile computation is the same. The
differences:

### (a) `sampling.use_cache` — a no-op at our batch size

Upstream's eval script sets `sampling_use_cache=True` for MDLM and `False` for
UDLM (`scripts/eval_qm9_guidance.sh:70-83`). **All our MDLM runs used `False`.**

It is *distribution-neutral*: `_process_sigma` zeroes sigma when
`time_conditioning=False` (`diffusion.py:334-335`), so for MDLM `log_x_theta`
depends only on `xt`, and reusing it while `xt` is unchanged is exact. That is
also why upstream disables caching for UDLM, which *is* time-conditioned.

It is also **NFE-neutral here**, which is not obvious and was worth measuring.
The cache is discarded whenever `not torch.allclose(xs, xt)`
(`diffusion.py:1204-1207`) — and that compares the *whole batch*. With
`batch_size=64` and `L=32`, roughly `64 * 32 / T` positions unmask somewhere in
the batch per step, so at any usable T some sequence always changes and the
cache never survives a step. Caching would only bite at batch size 1, or at T so
large that entire steps pass with no unmasking anywhere in the batch.

Re-running with `use_cache=True` (1024 samples, seed 1) confirms both halves:

| T | use_cache | valid | unique | novel | NFEs |
|---:|:---|---:|---:|---:|---:|
| 32 | False | 52.73 % (540/1024) | 92.22 % | 39.44 % | 32 |
| 32 | **True** | **52.73 % (540/1024)** | 92.22 % | 39.44 % | **32** |
| 256 | False | 69.14 % (708/1024) | 95.62 % | 31.07 % | 256 |
| 256 | **True** | **69.14 % (708/1024)** | 95.62 % | 31.07 % | **256** |

Bit-identical valid/unique/novel counts, identical NFE. So the deviation costs
nothing and the T-sweep conclusions in `00_base_model_health.md` stand as
written: T=256 really does cost 256 network evaluations, and the 16.4-point
validity headroom at T=32 really is bought with ~8x the compute.

Still worth flipping to `True` for MDLM going forward, purely to match upstream
and to keep the option of small-batch runs honest.

Raw CSVs: `results/mdlm_unguided_cache_T{32,256}.csv` on the server.

### (b) `<mask>` stripping — verified harmless

`our_guidance.clean_smiles` strips `<mask>` (plus `<unk>` and the BERT-style
markers) while upstream strips only `<bos>/<eos>/<pad>`. For an absorbing-state
model those differ exactly when a position is still masked at the end of
sampling: upstream counts such a sample invalid, ours could strip the marker and
parse the remainder as valid — inflating our validity.

Measured with `guidance_eval/diag_mask_residue.py` (MDLM, T=32, 256 samples):

```
decoded strings containing '<mask>' : 0
strings where the two cleanings differ: 0
valid under upstream cleaning : 142 (55.47%)
valid under our cleaning      : 142 (55.47%)
inflation                     : +0.00 points
```

Zero effect. The final step drives `move_chance_s` to ~0, so nothing stays
masked.

### (c) Determinism flags dropped — restore for paper runs

Upstream sets, before sampling:

```python
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
```

We do not. This does not bias any expected value, but it means a given seed is
not bit-reproducible on our runs. Worth restoring before final numbers.

### (d) Exception handling — ours is stricter-but-safer

Upstream catches only `rdkit.Chem.rdchem.KekulizeException` and would crash on
any other RDKit exception; ours catches `Exception` and counts the molecule
invalid. Ours cannot crash mid-sweep. No observed difference.

### (e) Inherited quirk in both: novelty is not canonicalised

Both harnesses compare *raw generated* SMILES strings against the *canonical*
training SMILES (`set(valids) - set(qm9_dataset['canonical_smiles'])`), without
canonicalising the generated side. A non-canonical spelling of a training
molecule therefore counts as novel, so novelty is overestimated — in upstream's
numbers too. Comparability with the paper is preserved; absolute novelty is not
trustworthy. Do not fix this silently, or our numbers stop matching the paper's.

---

## Verdict

Nothing needs to be "brought over" — the MDLM we are running *is* upstream's,
trained with upstream's QM9 recipe. The one thing to change going forward is
`sampling.use_cache=True` for MDLM runs, to match upstream and to report honest
NFE counts.
