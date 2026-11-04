#!/usr/bin/env python3
"""Reproduce the PC9 / osimertinib analysis (Table 1 of the paper) with selbound only.

Input: the GEO files of GSE150949 converted by ``build_inputs.py`` (pc9_counts.npz,
pc9_genes.txt, GSE150949_metaData_with_lineage.txt.gz) in DATA.

Steps: QC -> thinning to 3000 nuclear UMIs -> clone calling from lineage barcodes ->
held-out Pearson-residual PCA -> response-adjusted bounds vs the live-imaging budgets.
"""
import collections, json, pathlib, sys
import numpy as np, pandas as pd, scipy.sparse as sp
import selbound as sb

DATA = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/pc9")
T, MTMAX = 3000, 0.2
rng = np.random.default_rng(20261026)

X = sp.load_npz(DATA / "pc9_counts.npz").tocsr(); genes = np.array(open(DATA / "pc9_genes.txt").read().split("\n"))
m = pd.read_csv(DATA / "GSE150949_metaData_with_lineage.txt.gz", sep="\t")
mt = np.array([g.startswith("MT-") for g in genes])
umi = np.asarray(X.sum(1)).ravel(); nuc = np.asarray(X[:, ~mt].sum(1)).ravel()
keep = (nuc >= T) & (1 - nuc / np.maximum(umi, 1) < MTMAX)
m = m[keep].copy(); Xn = X[keep][:, ~mt]

# clone calling: link two lineage barcodes if they co-occur in >= 2 cells and in >= 50% of the rarer one
bl = m.lineage_barcode.apply(lambda s: s.split(",") if isinstance(s, str) else [])
cnt = collections.Counter(x for l in bl for x in l); co = collections.Counter()
for l in bl:
    if 2 <= len(l) <= 3:
        for i in range(len(l)):
            for j in range(i + 1, len(l)):
                co[tuple(sorted((l[i], l[j])))] += 1
parent = {x: x for x in cnt}
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for (a_, b_), w in co.items():
    if w >= 2 and w >= 0.5 * min(cnt[a_], cnt[b_]):
        ra, rb = find(a_), find(b_)
        if ra != rb: parent[ra] = rb
def assign(l):
    if len(l) == 0 or len(l) >= 4: return None
    roots = {find(x) for x in l}
    if len(roots) == 1: return roots.pop()
    w = collections.Counter()
    for x in l: w[find(x)] += cnt[x]
    (r1, w1), (r2, w2) = w.most_common(2)
    return r1 if w1 >= 3 * w2 else None
clone = bl.apply(assign).fillna("").values.astype(str)

Xt = sb.thin(Xn, T, rng)
samp = m.sample_name.values.astype(str)
fit = sb.split_fit_eval(samp, rng, frac=0.3, cap=1000)
emb = sb.PearsonResidualPCA(depth=T, n_hvg=2000, n_pcs=50).fit(Xt, samp, fit, rng)
Y = emb.transform(Xt)[:, :20].astype(float)
print("cells", len(samp), "clones", len(set(clone) - {""}))

# survival budgets digitised from live imaging (proxy, upper) at days 3, 7, 14
BUD = {3: (5.88, 17.30), 7: (36.41, 204.04), 14: (108.04, 1838.99)}
TRE = ["3_rep1", "3_rep2", "7_rep1", "7_rep2", "14_rep1_low", "14_rep2_low", "14_rep1_med", "14_rep2_med", "14_rep1_high", "14_rep2_high"]
out = {}
for t in TRE:
    lo, hi = BUD[int(t.split("_")[0])]
    r = sb.assess_selection(Y, samp, "0", t, fit_mask=fit, alpha=0.05, n_tests=20, budget=hi, loss=False)
    out[t] = dict(identity=r.bounds["identity"], translation=r.bounds["translation"], budget_proxy=lo, budget_upper=hi)
    print(r.summary())
json.dump(out, open("pc9_table1_selbound.json", "w"), indent=1)
