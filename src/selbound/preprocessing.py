"""Count thinning and a held-out Pearson-residual PCA.

The bound compares distributions of cell states, so technical depth must be equalised first:
every cell is thinned to exactly ``depth`` UMIs (multivariate hypergeometric sampling), and the
embedding (analytic Pearson residuals, variable genes, PCA) is fitted on a balanced subset of
FIT cells only and then applied to all cells.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA


def thin(X, depth: int, rng: np.random.Generator | int = 0) -> sp.csr_matrix:
    """Down-sample every row of an integer count matrix to exactly ``depth`` counts.
    Rows with fewer than ``depth`` counts must be removed beforehand."""
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    X = sp.csr_matrix(X).astype(np.int64)
    data = np.empty_like(X.data)
    for i in range(X.shape[0]):
        a, b = X.indptr[i], X.indptr[i + 1]
        if X.data[a:b].sum() < depth:
            raise ValueError(f"row {i} has fewer than {depth} counts")
        data[a:b] = rng.multivariate_hypergeometric(X.data[a:b], depth, method="marginals")
    Y = sp.csr_matrix((data, X.indices.copy(), X.indptr.copy()), shape=X.shape)
    Y.eliminate_zeros()
    return Y


def split_fit_eval(groups, rng: np.random.Generator | int = 0, frac: float = 0.3, cap: int = 1000) -> np.ndarray:
    """Boolean FIT mask: min(cap, frac * n_g) random cells of every group; the rest is EVAL."""
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    groups = np.asarray(groups)
    fit = np.zeros(len(groups), bool)
    for g in np.unique(groups):
        ii = np.where(groups == g)[0]
        fit[rng.choice(ii, min(cap, int(frac * len(ii))), replace=False)] = True
    return fit


class PearsonResidualPCA:
    """Analytic Pearson residuals (Lause et al. 2021) + variable genes + PCA, fitted on a
    balanced subset of FIT cells (equal cells per group), applied to all cells.

    Assumes every cell has the same depth (use :func:`thin` first).
    """

    def __init__(self, depth: int, n_hvg: int = 2000, n_pcs: int = 50, theta: float = 100.0):
        self.depth, self.n_hvg, self.n_pcs, self.theta = depth, n_hvg, n_pcs, theta

    def _resid(self, C, cols):
        M = self.depth * self.p_[cols]
        R = (C[:, cols].toarray() - M) / np.sqrt(M + M ** 2 / self.theta)
        c = np.sqrt(self.n_bal_)
        return np.clip(R, -c, c)

    def fit(self, X, groups, fit_mask, rng: np.random.Generator | int = 0):
        rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
        X = sp.csr_matrix(X); groups = np.asarray(groups)
        nbal = min((fit_mask & (groups == g)).sum() for g in np.unique(groups))
        bal = np.concatenate([rng.choice(np.where(fit_mask & (groups == g))[0], nbal, replace=False) for g in np.unique(groups)])
        self.bal_, self.n_bal_ = bal, len(bal)
        Cb = X[bal]
        self.p_ = np.asarray(Cb.sum(0)).ravel() / (self.depth * len(bal))
        ok = np.where(self.p_ > 0)[0]
        rv = np.zeros(len(self.p_))
        for k in range(0, len(ok), 3000):
            cols = ok[k:k + 3000]; rv[cols] = self._resid(Cb, cols).var(0)
        self.hvg_ = np.argsort(rv)[::-1][: self.n_hvg]
        self.pca_ = PCA(n_components=self.n_pcs, svd_solver="full", random_state=0).fit(self._resid(Cb, self.hvg_).astype(np.float32))
        return self

    def transform(self, X, chunk: int = 5000) -> np.ndarray:
        X = sp.csr_matrix(X)
        Z = np.vstack([self._resid(X[i:i + chunk], self.hvg_) for i in range(0, X.shape[0], chunk)]).astype(np.float32)
        return self.pca_.transform(Z).astype(np.float32)
