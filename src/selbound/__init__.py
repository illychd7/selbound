"""selbound: response-adjusted selection bounds for single-cell perturbation data.

Is pre-treatment selection plus a shared drug response enough to explain a treated
single-cell population? selbound returns a finite-sample lower bound on the pre-treatment
enrichment that a dataset still requires after a held-out response model, and compares it with
a ceiling on enrichment derived from survival.
"""
from .bound import lower_bound, lower_bound_mixture, pvalue, ceiling, GRID
from .response import Identity, Translation, AxisStretch, AffineMomentMatching, Callable, get_model
from .budget import survival_budget, proxy_budget, batch_tolerance
from .clones import clone_decomposition, within_clone_shift, calibrate_floor
from .preprocessing import thin, split_fit_eval, PearsonResidualPCA
from .api import assess_selection, assess_anndata, SelectionResult

__version__ = "0.1.0"
__all__ = ["lower_bound", "lower_bound_mixture", "pvalue", "ceiling", "GRID",
           "Identity", "Translation", "AxisStretch", "AffineMomentMatching", "Callable", "get_model",
           "survival_budget", "proxy_budget", "batch_tolerance",
           "clone_decomposition", "within_clone_shift", "calibrate_floor",
           "thin", "split_fit_eval", "PearsonResidualPCA",
           "assess_selection", "assess_anndata", "SelectionResult"]
