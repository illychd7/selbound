#!/usr/bin/env python3
"""Power versus number of cells and effect size, and computing time (paper: Fig. 4).

Semi-synthetic populations as in the benchmark: disjoint halves of barcoded PC9 day-0 cells,
clone-level log-normal selection (sigma = 0.5) with oracle budget B = sup g / E g, the real
day-3 within-clone common response, plus a clone-intrinsic shift carrying a fraction phi of
the response energy (phi = 0: the null "selection + common response").
For each n, n baseline EVAL cells and n treated EVAL cells are subsampled; the translation is
learned on separate FIT cells. Rejection: M_L(nu | mu0 + delta) > B at alpha = 0.05.

Usage: python power_runtime.py /home/claude/pc9/pc9_pp.npz  -> power.csv, runtime.csv
"""
import sys, time
import numpy as np, pandas as pd
import selbound as sb

PP = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/pc9/pc9_pp.npz"
Q, REPS = 20, 20
NS = [200, 400, 800, 1600]
PHIS = [0.0, 0.05, 0.10, 0.20, 0.30]

z = np.load(PP, allow_pickle=True)
Yall = z["pc"][:, :Q].astype(float); samp = z["sample"]; clone = z["clone"]
lab0 = (samp == "0") & (clone != ""); Y0, c0 = Yall[lab0], clone[lab0]
dreal, _ = sb.within_clone_shift(Yall[samp == "0"], clone[samp == "0"], Yall[samp == "3_rep1"], clone[samp == "3_rep1"])
cl = np.unique(c0); ci = np.searchsorted(cl, c0); E_delta = dreal @ dreal


def make(seed, phi, sigma_g=0.5):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Y0)); ref, src = perm[: len(perm) // 2], perm[len(perm) // 2:]
    g = np.exp(sigma_g * rng.standard_normal(len(cl))); w = g[ci[src]]; B = float(w.max() / w.mean())
    nt = int(1.5 * len(src)); pick = src[rng.choice(len(src), nt, replace=True, p=w / w.sum())]
    Yt = Y0[pick] + 1e-6 * rng.standard_normal((nt, Q)) + dreal
    if phi > 0:
        r = rng.standard_normal((len(cl), Q))
        wt = np.bincount(ci[pick], minlength=len(cl)) / nt; r = r - wt @ r
        r *= np.sqrt(phi / (1 - phi) * E_delta / (wt @ (r ** 2).sum(1))); Yt += r[ci[pick]]
    X0 = Y0[ref]
    f0, ft = rng.random(len(X0)) < 0.3, rng.random(len(Yt)) < 0.3
    return X0[f0], X0[~f0], Yt[ft], Yt[~ft], B, rng


rows = []
for phi in PHIS:
    for s in range(REPS):
        X0f, X0e, Xtf, Xte, B, rng = make(1000 + s, phi)
        delta = sb.Translation().fit(X0f, Xtf).delta_
        for n in NS:
            a = rng.choice(len(X0e), min(n, len(X0e)), replace=False); b = rng.choice(len(Xte), n, replace=False)
            m = sb.lower_bound(X0e[a] + delta, Xte[b], np.random.default_rng(s), alpha=0.05)
            rows.append(dict(phi=phi, seed=s, n=n, n0=len(a), B=B, M_L=m, reject=m > B,
                             ceiling=sb.ceiling(len(a), n, 0.05)))
    print("phi", phi, "done", flush=True)
P = pd.DataFrame(rows); P.to_csv("power.csv", index=False)
print(P.groupby(["phi", "n"]).agg(power=("reject", "mean"), median_ML=("M_L", "median"), median_B=("B", "median")).round(2).unstack("n").to_string())

# computing time of one bound (20 dimensions), n cells per condition
rt, rng = [], np.random.default_rng(0)
for n in [1000, 3000, 10000, 30000, 100000]:
    X, Y = rng.standard_normal((n, Q)), rng.standard_normal((n, Q)) + .3
    ts = []
    for _ in range(3):
        t0 = time.perf_counter(); sb.lower_bound(X, Y, 0); ts.append(time.perf_counter() - t0)
    rt.append(dict(n_per_condition=n, seconds=float(np.median(ts))))
    print(rt[-1], flush=True)
pd.DataFrame(rt).to_csv("runtime.csv", index=False)
