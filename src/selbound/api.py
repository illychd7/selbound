"""High-level interface: is pre-treatment selection plus a shared response sufficient?"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bound import lower_bound, ceiling
from .response import get_model


@dataclass
class SelectionResult:
    """Outcome of :func:`assess_selection` for one reference/target pair.

    bounds      : lower confidence bound M_L for each response model
    statistic   : min over models (Proposition 2.2: a valid test statistic without
                  multiplicity correction across models)
    budget      : selection budget supplied by the user (None if not given)
    reject      : True if statistic > budget ("selection plus a response in the class does not
                  suffice"); None if no budget was given
    loss        : reverse-direction bound M_L(mu | nu) with the identity model (Section 2.7);
                  large values mean baseline states were lost, which selection can produce
    ceiling     : largest bound this design can return (sample-size limit)
    """
    reference: str
    target: str
    n_ref: int
    n_target: int
    alpha: float
    bounds: dict
    statistic: float
    budget: float | None
    reject: bool | None
    loss: float | None
    ceiling: float
    models: dict = field(repr=False, default_factory=dict)

    def summary(self) -> str:
        b = ", ".join(f"{k} {v:.2f}" for k, v in self.bounds.items())
        s = f"{self.target} vs {self.reference}: M_L ({b}); statistic {self.statistic:.2f}"
        if self.budget is not None:
            s += f" vs budget {self.budget:.2f} -> {'REJECT' if self.reject else 'compatible with selection + shared response'}"
        return s


def assess_selection(Y, groups, reference, target, fit_mask=None, models=("identity", "translation"),
                   alpha: float = 0.05, n_tests: int = 1, budget: float | None = None, loss: bool = True,
                   seed: int = 0, fit_frac: float = 0.3, fit_cap: int = 1000) -> SelectionResult:
    """Response-adjusted selection test for one treated sample.

    Parameters
    ----------
    Y : (n, q) array
        Cell embedding (e.g. a held-out PCA; see :class:`selbound.preprocessing.PearsonResidualPCA`).
    groups : (n,) array
        Sample label of every cell.
    reference, target : labels in ``groups`` (baseline and treated sample).
    fit_mask : (n,) bool array, optional
        Cells used to fit response models. Defaults to a random min(fit_cap, fit_frac * n)
        cells per sample; bounds use the remaining (EVAL) cells only.
    models : sequence of response models (names or :class:`selbound.response.ResponseModel`).
        "identity" is the pure-selection reference; the test statistic is the minimum over the
        non-identity models when any is given.
    alpha, n_tests : familywise level and the number of bounds it is shared over; every bound
        is computed at alpha / n_tests. Count all bounds you will report (e.g. samples x models).
    budget : selection budget B (see :mod:`selbound.budget`); enables the reject decision.
    loss : also compute the reverse-direction bound (identity model).
    """
    Y = np.asarray(Y, float); groups = np.asarray(groups)
    if fit_mask is None:
        from .preprocessing import split_fit_eval
        fit_mask = split_fit_eval(groups, seed, fit_frac, fit_cap)
    fit_mask = np.asarray(fit_mask, bool); ev = ~fit_mask
    r, t = groups == reference, groups == target
    Y0f, Ytf, Y0e, Yte = Y[fit_mask & r], Y[fit_mask & t], Y[ev & r], Y[ev & t]
    a = alpha / n_tests
    fitted, bounds = {}, {}
    for m in models:
        mod = get_model(m).fit(Y0f, Ytf)
        fitted[mod.name] = mod
        bounds[mod.name] = lower_bound(mod.transform(Y0e), Yte, np.random.default_rng(seed), alpha=a)
    adj = [v for k, v in bounds.items() if k != "identity"]
    stat = min(adj) if adj else bounds["identity"]
    rev = lower_bound(Yte, Y0e, np.random.default_rng(seed), alpha=a) if loss else None
    return SelectionResult(str(reference), str(target), len(Y0e), len(Yte), alpha, bounds, float(stat), budget,
                           None if budget is None else bool(stat > budget), rev, ceiling(len(Y0e), len(Yte), a), fitted)


def assess_anndata(adata, groupby: str, reference, target, use_rep: str = "X_pca", n_dims: int | None = 20,
                 fit_key: str | None = None, **kw) -> SelectionResult:
    """:func:`assess_selection` on an AnnData object.

    use_rep : key in ``adata.obsm`` holding the embedding (first ``n_dims`` columns are used)
    fit_key : boolean column of ``adata.obs`` marking FIT cells (random split if None)
    """
    Y = np.asarray(adata.obsm[use_rep])
    if n_dims is not None:
        Y = Y[:, :n_dims]
    fit = None if fit_key is None else adata.obs[fit_key].to_numpy(bool)
    return assess_selection(Y, adata.obs[groupby].to_numpy(), reference, target, fit_mask=fit, **kw)
