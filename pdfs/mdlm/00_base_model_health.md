# MDLM-QM9 base model health check

**Date:** 2026-08-12
**Question:** MDLM-QM9 gives 52.73 % validity at T=32 against 97.85 % for UDLM-QM9 at
the same budget. Is that a weak checkpoint, a step-budget artifact, or a
model-class limitation?
**Answer:** converged checkpoint; ~16 points of the T=32 deficit is a step-budget /
sampler artifact; the remaining ~30 points is a model-class gap that more steps
do not close.

Artifacts in this folder: `val_nll_history.csv`, `mdlm_unguided_T{8,16,32,64,128,256}.csv`.
Sweep script: `scripts/server_run_mdlm_tsweep.sh`. Raw logs on the server at
`results/mdlm_unguided_T*.log`, console at `results/_tsweep.out`.

---

## 1. The checkpoint is converged

Extracted from the offline wandb run
(`outputs/qm9/mdlm_no-guidance/wandb/offline-run-20260801_065258-*/run-*.wandb`);
the offline `wandb-summary.json` holds only `runtime`, so the history had to be
read out of the datastore records. 396 validation evaluations over 25 k steps.

| window | mean val/nll | min val/nll |
|:---|---:|---:|
| 0–5 k | 0.92356 | 0.78382 |
| 5–10 k | 0.78261 | 0.75794 |
| 10–15 k | 0.76803 | 0.74972 |
| 15–20 k | 0.75915 | **0.73866** |
| 20–25 k | 0.75519 | 0.74335 |

Minimum val/nll = **0.73866 at step 17450**. From 5 k to 25 k the mean improved
0.783 → 0.755, i.e. **3.6 % over 20 k steps**; the last 10 k bought 1.7 %.

This matches the checkpoint timestamps independently: `best.ckpt` was last
written at 12:41 while step 15000 landed at 12:00 and step 20000 at 13:22, so
Lightning stopped finding new minima around step ~17.5 k and the run continued
to 25 k without beating it.

The eval scripts load `checkpoints/best.ckpt` with `eval.disable_ema=False`
(`scripts/server_run_comparison_mdlm.sh:25,60`), i.e. the EMA weights of the
best checkpoint — not the last one. **More training is not the fix.**

## 2. Architecture is not the confound

| | hidden | n_blocks | n_heads | cond_dim | dropout |
|:---|---:|---:|---:|---:|---:|
| MDLM (`model=small`) | 768 | 12 | 12 | 128 | 0.1 |
| UDLM-QM9 (`kuleshov-group/udlm-qm9`) | 768 | 12 | 12 | 128 | 0.1 |

Identical. The MDLM↔UDLM gap is not model capacity. It remains confounded with
**training recipe**: ours is 25 k steps at lr 3e-4; the released UDLM-QM9
checkpoint's budget is unknown and probably larger. Given the val/nll plateau
above, that is unlikely to be the dominant term, but any cross-model claim
should say so.

## 3. Validity vs step budget (unguided, 1024 samples, seed 1)

All rows are pure base-sampler trajectories: `guidance.t_min=2.0
guidance.t_max=2.0` puts every `t` outside the guidance window so
`_our_denoise` falls through to `Diffusion._ddpm_denoise` and never calls the
reward (`our_guidance.py:297-308`).

| T | valid | Δ | gap to 100 % | frac. of gap closed | uniq | novel | QED mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 18.26 % | — | 81.74 % | — | 89.84 % | 54.01 % | 0.4355 |
| 16 | 37.30 % | +19.04 | 62.70 % | 0.233 | 92.67 % | 42.41 % | 0.4543 |
| 32 | 52.73 % | +15.43 | 47.27 % | 0.246 | 92.22 % | 39.44 % | 0.4542 |
| 64 | 63.28 % | +10.55 | 36.72 % | 0.223 | 93.21 % | 36.57 % | 0.4537 |
| 128 | 66.31 % | +3.03 | 33.69 % | **0.082** | 93.52 % | 33.58 % | 0.4575 |
| 256 | 69.14 % | +2.83 | 30.86 % | **0.084** | 95.62 % | 31.07 % | 0.4533 |

T=32 reproduces the existing `results/mdlm_unguided.csv` figure (52.73 %)
exactly, so the sweep is a faithful re-run.

**Two regimes.** Up to T=64 each doubling closes a constant ~23 % of the
remaining gap. From T=64 onward that collapses to ~8 %. Extrapolating the
second regime, reaching UDLM's 97.85 % would take ~30 further doublings — i.e.
MDLM-QM9 validity does **not** approach UDLM's at any usable budget. Practical
ceiling: **~70 %**.

**Two side observations.** QED mean is flat at ~0.454 across the whole sweep —
the step budget buys validity, not quality. And novelty *falls* monotonically
(54.0 % → 31.1 %) while uniqueness rises: longer trajectories land closer to
the training distribution.

---

## 4. What this means for reward-tilted confidence decoding

**The headroom is real and quantified.** At T=32 the base sampler leaves
**52.73 % → 69.14 % = 16.4 validity points** on the table relative to its own
large-T ceiling. The mechanism responsible is known: MDLM freezes a position
once unmasked (`copy_flag`, `our_guidance.py:376-379`), so at T = L = 32 about
26 % of steps unmask two or more positions at once and write them
*conditionally independently* given `x_t`. The true joint is not that product.
Raising T dilutes those events; a confidence-ordered decoder is meant to remove
them at fixed NFE.

So the experiment has a clean success criterion and a known upper bound:
**does arm C/D at T=32 recover ~69 % without raising NFE above 32?**

**But QM9 is a weak showcase for MDLM.** The ceiling (~70 %) sits far below
UDLM's 97.85 % at T=32. That is consistent with the published UDLM claim that
uniform diffusion beats masked diffusion at *small vocabularies* — QM9's
vocabulary is 40 tokens. The honest reading is that on QM9 the masked model is
the wrong model class, and confidence decoding cannot repair that; it can only
recover the sampler's own gap.

The method needs masking (reward-aware position selection is undefined for
UDLM — its kernel has no mask slot to rank), so UDLM is not an option for the
headline. That points at a **large-vocabulary** task where masked diffusion is
the appropriate class: `amazon_polarity` / `lm1b` at L=128 with the GPT-2
vocabulary. Cost: `REWARD_FNS` is RDKit-only (`our_guidance.py:71-74`) and would
need a text reward. Upside: an off-the-shelf sentiment classifier works
directly, because the reward only ever sees clean `x_0` — which is the
classifier-free selling point, demonstrated outside molecules.

### Recommended sequencing

1. **MDLM-QM9 as mechanism validation** — code is nearly ready, runs are
   ~30–60 s each, and it carries the implementation correctness check
   (arm D must converge to arm C as λ→0). Target: 52.73 % → ~69 % at T=32.
2. **Large-vocab task as headline** — after the mechanism is confirmed.

Do not report QM9 MDLM numbers head-to-head against UDLM numbers without
stating the small-vocabulary caveat.
