"""Re-count every few-step run on metrics that are not confounded by molecule size.

The shipped metric is hits@bar among `novel = set(raw generated strings) - QM9`.
Two problems compound in it: the subtraction is string-level so rediscoveries
leak in, and molecules outside QM9's size range (>9 heavy atoms) are novel by
construction while also scoring higher QED. Both biases scale with k.

Emits, per run: unique-valid counts (molecule-level dedup, no QM9 subtraction),
the same restricted to <=9 heavy atoms, true novelty (canonicalised difference),
and mean heavy-atom count.
"""
import csv, glob, json, os, re
from rdkit import Chem, RDLogger
import datasets
RDLogger.DisableLog("rdApp.*")

TAG = re.compile(r"mdlm_fs_k(\d+)_N(\d+)_lam([0-9.]+)_(marginal|edlm)_s(\d+)_samples\.json$")
QM9 = set(datasets.load_dataset("yairschiff/qm9", trust_remote_code=True,
                                split="train")["canonical_smiles"])
BARS = (0.6, 0.65)
rows = []
files = sorted(glob.glob("results/qed/fewstep/*_samples.json"))
for i, f in enumerate(files):
    m = TAG.search(os.path.basename(f))
    if not m:
        continue
    d = json.load(open(f))
    uniq = {}                      # canonical smiles -> (qed, heavy atoms)
    for smi, q in zip(d["valid"], d["qed_valid"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        uniq[Chem.MolToSmiles(mol)] = (q, mol.GetNumHeavyAtoms())
    small = {c: v for c, v in uniq.items() if v[1] <= 9}
    truenovel = {c: v for c, v in uniq.items() if c not in QM9}
    rec = dict(k=int(m[1]), N=int(m[2]), lam=float(m[3]), mode=m[4], seed=int(m[5]),
               n_valid=len(d["valid"]), n_uniq=len(uniq), n_small=len(small),
               n_truenovel=len(truenovel),
               ha_mean=(sum(v[1] for v in uniq.values()) / len(uniq)) if uniq else 0.0)
    for b in BARS:
        rec[f"uniq_hits{b}"] = sum(1 for q, _ in uniq.values() if q >= b)
        rec[f"small_hits{b}"] = sum(1 for q, _ in small.values() if q >= b)
        rec[f"truenovel_hits{b}"] = sum(1 for q, _ in truenovel.values() if q >= b)
    rows.append(rec)
    if i % 200 == 0:
        print(f"  {i}/{len(files)}", flush=True)
out = "results/qed/fewstep/_recount.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out}  ({len(rows)} runs)")
