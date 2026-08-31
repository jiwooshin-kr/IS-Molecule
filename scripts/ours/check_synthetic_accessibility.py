import json, sys, statistics as st
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")
sys.path.append(RDConfig.RDContribDir + "/SA_Score")
import sascorer
import datasets, os
def summarise(smis, qs):
    sa, ha, ok = [], [], 0
    for s, q in zip(smis, qs):
        m = Chem.MolFromSmiles(s)
        if not m: continue
        sa.append(sascorer.calculateScore(m)); ha.append(m.GetNumHeavyAtoms())
    return st.mean(sa), st.median(sa), 100.0*sum(1 for x in sa if x > 4.5)/len(sa), st.mean(ha)
qm9 = datasets.load_dataset("yairschiff/qm9", trust_remote_code=True, split="train")["canonical_smiles"][:4000]
a, b, c, d = summarise(qm9, [0]*len(qm9))
print("SA score: 1 = easy to make, 10 = hard.  >4.5 is the usual 'hard' flag.")
print("%-22s SA mean=%.2f median=%.2f  hard=%4.1f%%  HA=%.1f" % ("QM9 (reference)", a, b, c, d))
for k in (1, 8, 32):
    d_ = json.load(open(f"results/qed/fewstep/mdlm_fs_k{k}_N1000_lam200_edlm_s1_samples.json"))
    a, b, c, e = summarise(d_["novel"], d_["qed_novel"])
    print("%-22s SA mean=%.2f median=%.2f  hard=%4.1f%%  HA=%.1f" % (f"ours k={k} (T={32//k})", a, b, c, e))
