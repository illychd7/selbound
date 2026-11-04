"""Finite-sample lower confidence bounds for the minimal selective enrichment.

M*(nu | mu) = inf{M >= 1 : nu(A) <= M mu(A) for every measurable A} = exp D_inf(nu || mu).

The bound splits the reference sample into three disjoint parts: a distance reference S,
a threshold set T and an estimation set U. For nested tail sets
B_k = {y : d(y, S) > s_k}, with s_k the q_k-quantile of d(T, S), it combines a one-sided
Clopper-Pearson upper bound for mu(B_k) (from U) with a one-sided Clopper-Pearson lower
bound for nu(B_k) (from the target sample), each at level alpha / (2K). Then

    M_L = max_k LB_k / UB_k   satisfies   P(M* >= M_L) >= 1 - alpha.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import beta

#: Default quantile grid defining the nested tail sets.
GRID = (0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 1.0)


def cp_lower(k: int, n: int, a: float) -> float:
    """One-sided Clopper-Pearson lower bound for a binomial proportion at level a."""
    return 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))


def cp_upper(k: int, n: int, a: float) -> float:
    """One-sided Clopper-Pearson upper bound for a binomial proportion at level a."""
    return 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))


def nn_dist(Q: np.ndarray, S: np.ndarray, chunk: int = 4000) -> np.ndarray:
    """Euclidean distance from every row of Q to its nearest row of S."""
    return np.concatenate([cdist(Q[i:i + chunk], S).min(1) for i in range(0, len(Q), chunk)])


@dataclass
class TailSet:
    q: float
    threshold: float
    k_ref: int
    n_ref: int
    k_target: int
    n_target: int
    lower_target: float
    upper_ref: float
    ratio: float


def _split(n: int, rng: np.random.Generator, mS: int, fracT: float):
    p = rng.permutation(n)
    mS = min(mS, n // 3)
    rest = p[mS:]
    nT = int(round(fracT * len(rest)))
    return p[:mS], rest[:nT], rest[nT:]


def lower_bound(Y_ref: np.ndarray, Y_target: np.ndarray, rng: np.random.Generator | int = 0,
                alpha: float = 0.05, grid=GRID, mS: int = 400, fracT: float = 0.4,
                return_detail: bool = False):
    """(1 - alpha) lower confidence bound for M*(nu | mu).

    Parameters
    ----------
    Y_ref : (n0, q) array
        Cells from the reference distribution mu (for a response-adjusted bound, the baseline
        cells already passed through the response model, e.g. ``Y0 + delta``).
    Y_target : (nt, q) array
        Cells from the target distribution nu (e.g. treated cells).
    rng : Generator or int
        Random generator (or seed) for the three-way split of the reference cells.
    alpha : float
        Error level of this single bound. Divide by the number of bounds you report for
        familywise control (see :func:`selbound.test_selection`).
    grid : sequence of float
        Quantiles of d(T, S) that define the nested tail sets (1.0 = maximum).
    mS, fracT : int, float
        Size of the distance reference set (capped at n0 // 3) and fraction of the remaining
        reference cells used to set thresholds.
    return_detail : bool
        Also return the per-tail-set counts and ratios.
    """
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    Y_ref, Y_target = np.asarray(Y_ref, float), np.asarray(Y_target, float)
    iS, iT, iU = _split(len(Y_ref), rng, mS, fracT)
    S, T, U = Y_ref[iS], Y_ref[iT], Y_ref[iU]
    dT, dU, dt = nn_dist(T, S), nn_dist(U, S), nn_dist(Y_target, S)
    K = len(grid)
    a = alpha / (2 * K)
    best, detail = 0.0, []
    for q in grid:
        s = np.quantile(dT, q, method="inverted_cdf") if q < 1 else dT.max()
        k0, kt = int((dU > s).sum()), int((dt > s).sum())
        LB, UB = cp_lower(kt, len(dt), a), cp_upper(k0, len(dU), a)
        r = LB / UB
        detail.append(TailSet(q, float(s), k0, len(dU), kt, len(dt), LB, UB, r))
        best = max(best, r)
    return (best, detail) if return_detail else best


def lower_bound_mixture(Y_ref, parts, rng: np.random.Generator | int = 0, alpha: float = 0.05,
                        grid=GRID, mS: int = 400, fracT: float = 0.4) -> float:
    """Gate-agnostic bound when the target is an unknown mixture of sorted fractions ``parts``.

    Uses nu(B) >= min_j nu_j(B) for every mixture, with Bonferroni over (J + 1) K events.
    """
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    Y_ref = np.asarray(Y_ref, float)
    iS, iT, iU = _split(len(Y_ref), rng, mS, fracT)
    S = Y_ref[iS]
    dT, dU = nn_dist(Y_ref[iT], S), nn_dist(Y_ref[iU], S)
    dj = [nn_dist(np.asarray(P, float), S) for P in parts]
    K = len(grid)
    a = alpha / ((len(parts) + 1) * K)
    best = 0.0
    for q in grid:
        s = np.quantile(dT, q, method="inverted_cdf") if q < 1 else dT.max()
        UB = cp_upper(int((dU > s).sum()), len(dU), a)
        LB = min(cp_lower(int((d > s).sum()), len(d), a) for d in dj)
        best = max(best, LB / UB)
    return best


def pvalue(Y_ref, Y_target, M0: float, seed: int = 0, **kw) -> float:
    """p-value for H0: M* <= M0, by bisection on alpha with a fixed split."""
    lo, hi = 1e-12, 1.0
    if lower_bound(Y_ref, Y_target, np.random.default_rng(seed), alpha=hi, **kw) <= M0:
        return 1.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if lower_bound(Y_ref, Y_target, np.random.default_rng(seed), alpha=mid, **kw) > M0:
            hi = mid
        else:
            lo = mid
    return hi


def ceiling(n_ref: int, n_target: int, alpha: float = 0.05, grid=GRID, mS: int = 400, fracT: float = 0.4) -> float:
    """Largest bound attainable for given sample sizes (no reference cell and every target cell
    in the tail). Useful to check whether a design can detect a given enrichment at all."""
    mS = min(mS, n_ref // 3)
    rest = n_ref - mS
    n_est = rest - int(round(fracT * rest))
    a = alpha / (2 * len(grid))
    return cp_lower(n_target, n_target, a) / cp_upper(0, n_est, a)
