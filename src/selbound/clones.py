"""Clone-level decomposition of drug-induced state change (Proposition 2.5).

  mean shift = composition (selection between clones) + within-clone response     (exact)
  within-clone response energy = common translation + clone-specific deviation    (split-half)
  clone-specific = state-dependent (predicted from neighbouring clones) + clone-intrinsic

Clone-specific shares are biased upward by finite sampling of clones; interpret them only
against a calibrated no-plasticity floor (``calibrate_floor``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def within_clone_shift(Y0, c0, Yt, ct, nmin: int = 2):
    """Clone-weighted mean within-clone shift and number of shared clones."""
    c0, ct = np.asarray(c0).astype(str), np.asarray(ct).astype(str)
    s0 = pd.Series(c0[c0 != ""]).value_counts(); st = pd.Series(ct[ct != ""]).value_counts()
    sh = [c for c in st.index if c in s0.index and s0[c] >= nmin and st[c] >= nmin]
    d = np.array([Yt[ct == c].mean(0) - Y0[c0 == c].mean(0) for c in sh])
    w = np.array([st[c] for c in sh], float)
    return (w[:, None] * d).sum(0) / w.sum(), len(sh)


def clone_decomposition(Y0, c0, Yt, ct, nsplit: int = 50, k: int = 10, nperm: int = 200, seed: int = 0,
                        nmin: int = 4, min_clones: int = 15) -> dict:
    """Decompose the baseline-to-time-t change of clone-labelled cells.

    Y0, Yt : embeddings of baseline and time-t cells; c0, ct : clone labels ("" = unassigned).
    Returns a dict; ``skipped=True`` when fewer than ``min_clones`` clones have >= ``nmin``
    cells at both times. Keys of interest: share_selection_incl_extinction, frac_common,
    frac_clone_specific, frac_specific_state_dependent, p_clone_specific, p_state_dependent,
    within_clone_var_ratio.
    """
    rng = np.random.default_rng(seed)
    Y0, Yt = np.asarray(Y0, float), np.asarray(Yt, float)
    c0, ct = np.asarray(c0).astype(str), np.asarray(ct).astype(str)
    s0 = pd.Series(c0[c0 != ""]).value_counts(); st = pd.Series(ct[ct != ""]).value_counts()
    K = [c for c in st.index if c in s0.index and s0[c] >= nmin and st[c] >= nmin]
    if len(K) < min_clones:
        return {"n_clones": len(K), "skipped": True}
    i0 = {c: np.where(c0 == c)[0] for c in K}; it = {c: np.where(ct == c)[0] for c in K}
    n0 = np.array([len(i0[c]) for c in K], float); nt = np.array([len(it[c]) for c in K], float)
    p0, pt = n0 / n0.sum(), nt / nt.sum()
    M0 = np.array([Y0[i0[c]].mean(0) for c in K]); Mt = np.array([Yt[it[c]].mean(0) for c in K])
    Delta = pt @ Mt - p0 @ M0; Sel = (pt - p0) @ M0; Resp = pt @ (Mt - M0)
    nD = Delta @ Delta
    res = dict(n_clones=len(K), n_cells_t=int(nt.sum()), mean_shift=float(np.sqrt(nD)),
               share_selection=float(Sel @ Delta / nD), share_response=float(Resp @ Delta / nD),
               norm_selection=float(np.linalg.norm(Sel)), norm_response=float(np.linalg.norm(Resp)))
    lab0 = c0 != ""; all0 = pd.Series(c0[lab0]).value_counts(); labt = ct != ""
    ft = pd.Series(ct[labt & np.isin(ct, all0.index)]).value_counts()
    w = np.array([ft.get(c, 0) / all0[c] for c in c0[lab0]]); w = w / w.sum()
    Sel_ext = w @ Y0[lab0] - Y0[lab0].mean(0); Delta_all = Yt[labt].mean(0) - Y0[lab0].mean(0)
    res["share_selection_incl_extinction"] = float(Sel_ext @ Delta_all / (Delta_all @ Delta_all))
    res["frac_baseline_clones_extinct"] = float(np.mean([ft.get(c, 0) == 0 for c in all0.index[all0 >= 3]]))
    splits = []
    for _ in range(nsplit):
        A0, B0, At, Bt = {}, {}, {}, {}
        for c in K:
            a = rng.permutation(i0[c]); b = rng.permutation(it[c])
            A0[c], B0[c] = a[: len(a) // 2], a[len(a) // 2:]; At[c], Bt[c] = b[: len(b) // 2], b[len(b) // 2:]
        m0a = np.array([Y0[A0[c]].mean(0) for c in K]); m0b = np.array([Y0[B0[c]].mean(0) for c in K])
        mta = np.array([Yt[At[c]].mean(0) for c in K]); mtb = np.array([Yt[Bt[c]].mean(0) for c in K])
        da, db = mta - m0a, mtb - m0b; dela, delb = pt @ da, pt @ db
        D = ((m0a[:, None, :] - m0a[None, :, :]) ** 2).sum(-1); np.fill_diagonal(D, np.inf)
        splits.append((dela, delb, da - dela, db - delb, np.argsort(D, 1)[:, :k]))

    def stats(perm_pair=None, perm_pos=None):
        Ec, Es, Est = [], [], []
        for dela, delb, ra, rb, nn0 in splits:
            rb_ = rb if perm_pair is None else rb[perm_pair]
            if perm_pos is None:
                nn = nn0
            else:
                inv = np.empty_like(perm_pos); inv[perm_pos] = np.arange(len(perm_pos)); nn = inv[nn0[perm_pos]]
            Ec.append(dela @ delb); Es.append(pt @ (ra * rb_).sum(1)); Est.append(pt @ (ra[nn].mean(1) * rb).sum(1))
        return np.median(Ec), np.median(Es), np.median(Est), Es

    Ec, Es, Est, Es_all = stats()
    nul_s = [stats(perm_pair=rng.permutation(len(K)))[1] for _ in range(nperm)]
    nul_t = [stats(perm_pos=rng.permutation(len(K)))[2] for _ in range(nperm)]
    tot = Ec + max(Es, 0)
    res.update(E_common=float(Ec), E_clone_specific=float(Es), E_state_dependent=float(Est),
               frac_common=float(Ec / tot), frac_clone_specific=float(max(Es, 0) / tot),
               frac_specific_state_dependent=float(np.clip(Est / Es, 0, 1)) if Es > 0 else float("nan"),
               p_clone_specific=float((1 + np.sum(np.array(nul_s) >= Es)) / (1 + nperm)),
               p_state_dependent=float((1 + np.sum(np.array(nul_t) >= Est)) / (1 + nperm)),
               E_spec_q10_q90=[float(np.quantile(Es_all, .1)), float(np.quantile(Es_all, .9))])
    v0 = np.array([np.var(Y0[i0[c]], 0, ddof=1).sum() for c in K]); vt = np.array([np.var(Yt[it[c]], 0, ddof=1).sum() for c in K])
    res["within_clone_var_ratio"] = float((pt @ vt) / (pt @ v0))
    return res


def calibrate_floor(Y0, c0, Yt, ct, seeds=range(4), sigma_g: float = 0.5, **kw) -> list:
    """No-plasticity floor of the clone-specific share for this dataset.

    Splits the barcoded baseline cells into disjoint halves, resamples one half with
    clone-level log-normal fitness and adds the observed common within-clone shift, so the
    true clone-specific share is zero. Returns frac_clone_specific for each seed.
    """
    Y0, Yt = np.asarray(Y0, float), np.asarray(Yt, float)
    c0, ct = np.asarray(c0).astype(str), np.asarray(ct).astype(str)
    lab = c0 != ""
    X0, cl0 = Y0[lab], c0[lab]
    dreal, _ = within_clone_shift(Y0, c0, Yt, ct)
    cl = np.unique(cl0); ci = np.searchsorted(cl, cl0); out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X0)); rf, src = perm[: len(perm) // 2], perm[len(perm) // 2:]
        g = np.exp(sigma_g * rng.standard_normal(len(cl))); pr = g[ci[src]]; pr = pr / pr.sum()
        nt = int(1.5 * len(src)); pick = src[rng.choice(len(src), nt, replace=True, p=pr)]
        Ys = X0[pick] + 1e-6 * rng.standard_normal((nt, X0.shape[1])) + dreal
        r = clone_decomposition(X0[rf], cl0[rf], Ys, cl0[pick], seed=seed, **{"nmin": 3, "min_clones": 10, **kw})
        out.append(r.get("frac_clone_specific", float("nan")))
    return out
