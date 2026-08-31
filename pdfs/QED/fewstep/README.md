# QED / few-step

MDLM few-step generation: fixed k unmaskings per step (k = L/T, L = 32), so
k in {1,2,4,8} <-> sampling.steps in {32,16,8,4}. Compares
`mixture_sampling=marginal` against `edlm` under `position_selection=random_k`.

- runs:   `results/qed/fewstep/`   (server-side, rsynced back; never synced out)
- PDF:    `fewstep.pdf` at this level
- source: `src/`     (tex + build logs)
- scraped tables: `notes/`   (never in `results/`, which is rsync-mirrored)

Arm A settings throughout: `oversample=1`, `exclude_invalid=False`,
`exact_uniform_step=False` -- so every cell is comparable with the earlier
lambda x N sweep in `results/qed/`.
