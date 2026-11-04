"""Selection budgets: how much enrichment survival and division can produce.

Proposition 2.4. If every founder cell has at most 2^d descendants and a fraction f of founders
has at least one descendant, the selective enrichment of any state is at most 2^d / f.
"""
from __future__ import annotations

import numpy as np


def survival_budget(max_divisions: float, frac_surviving: float) -> float:
    """Upper selection budget 2^d / f (Proposition 2.4).

    max_divisions  : largest number of divisions any founder lineage completed by time t
    frac_surviving : fraction of founder lineages with at least one living descendant at t
    """
    if not 0 < frac_surviving <= 1:
        raise ValueError("frac_surviving must be in (0, 1]")
    if max_divisions < 0:
        raise ValueError("max_divisions must be >= 0")
    return float(2.0 ** max_divisions / frac_surviving)


def proxy_budget(divisions, alive) -> float:
    """Central budget from tracked lineages: max progeny / mean progeny.

    divisions : cumulative divisions of each tracked founder lineage at time t
    alive     : boolean, whether the lineage is alive at time t (dead lineages contribute 0)
    """
    divisions, alive = np.asarray(divisions, float), np.asarray(alive, bool)
    prog = np.where(alive, 2.0 ** np.nan_to_num(divisions), 0.0)
    if prog.sum() == 0:
        raise ValueError("no surviving lineage")
    return float(prog.max() / prog.mean())


def batch_tolerance(bound: float, budget: float) -> float:
    """Proposition 2.6: technical inflation c* = M_L / budget that a rejection survives.

    A rejection (M_L > budget) remains valid as long as library-specific technical distortion
    inflates M* by less than c*.
    """
    return float(bound / budget)
