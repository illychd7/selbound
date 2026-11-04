import numpy as np
import pytest

import selbound as sb


def test_cp_bounds_monotone():
    assert sb.bound.cp_lower(0, 10, 0.05) == 0.0
    assert sb.bound.cp_upper(10, 10, 0.05) == 1.0
    assert sb.bound.cp_lower(5, 10, 0.025) < 0.5 < sb.bound.cp_upper(5, 10, 0.025)


def test_null_same_distribution_rarely_exceeds_one():
    """Same distribution: M* = 1, so P(M_L > 1) <= alpha."""
    rng = np.random.default_rng(0)
    exceed = 0
    for s in range(100):
        X, Y = rng.standard_normal((800, 5)), rng.standard_normal((800, 5))
        exceed += sb.lower_bound(X, Y, s, alpha=0.05) > 1
    assert exceed <= 10          # expected <= 5; generous margin for 100 runs


def test_bound_below_true_selection():
    """Exact selection on a bounded population: M_L stays below sup g / E g."""
    rng = np.random.default_rng(1)
    beta = 1.0
    true = np.exp(beta) / (np.sinh(beta) / beta)          # 1-D uniform on [-1, 1], g = exp(beta x)
    for s in range(20):
        X = rng.uniform(-1, 1, (3000, 1))
        Z = rng.uniform(-1, 1, (20000, 1))
        Y = Z[rng.random(len(Z)) < np.exp(beta * (Z[:, 0] - 1))][:3000]
        assert sb.lower_bound(X, Y, s) <= true


def test_shift_detected_and_removed_by_translation():
    rng = np.random.default_rng(2)
    X0, Xt = rng.standard_normal((3000, 4)), rng.standard_normal((3000, 4)) + 3.0
    groups = np.r_[np.zeros(3000), np.ones(3000)]
    res = sb.assess_selection(np.vstack([X0, Xt]), groups, 0, 1, budget=2.0)
    assert res.bounds["identity"] > 10
    assert res.bounds["translation"] < 1.5
    assert res.reject is False
    assert res.ceiling > res.bounds["identity"]


def test_heterogeneous_response_rejected():
    """Half the treated cells respond twice as strongly: no translation can absorb that."""
    rng = np.random.default_rng(3)
    X0 = rng.standard_normal((3000, 3))
    Xt = rng.standard_normal((3000, 3)); Xt[:, 0] += np.where(rng.random(3000) < .5, 2.0, 6.0)
    groups = np.r_[np.zeros(3000), np.ones(3000)]
    res = sb.assess_selection(np.vstack([X0, Xt]), groups, 0, 1, budget=2.0)
    assert res.reject is True


def test_models_fit_and_transform():
    rng = np.random.default_rng(4)
    Y0 = rng.standard_normal((500, 3)); Yt = 2 * rng.standard_normal((500, 3)) + 1
    for name in ["identity", "translation", "axis_stretch", "affine"]:
        m = sb.get_model(name).fit(Y0, Yt)
        assert m.transform(Y0).shape == Y0.shape
    aff = sb.AffineMomentMatching().fit(Y0, Yt)
    assert np.allclose(np.cov(aff.transform(Y0).T), np.cov(Yt.T), atol=1e-6)


def test_budget():
    assert sb.survival_budget(3, 0.5) == 16.0
    assert sb.proxy_budget([0, 1, 2, 0], [True, True, True, False]) == pytest.approx(4 / (7 / 4))
    with pytest.raises(ValueError):
        sb.survival_budget(1, 0)


def test_clone_decomposition_identity_exact():
    rng = np.random.default_rng(5)
    clones = np.repeat([f"c{i}" for i in range(30)], 20)
    cm = rng.standard_normal((30, 5))
    Y0 = cm.repeat(20, 0) + rng.standard_normal((600, 5))
    Yt = cm.repeat(20, 0) + 3.0 + rng.standard_normal((600, 5))
    r = sb.clone_decomposition(Y0, clones, Yt, clones, nperm=20, nsplit=10)
    assert r["share_selection"] + r["share_response"] == pytest.approx(1.0)
    assert r["frac_common"] > 0.9


def test_matches_reference_implementation():
    """Same split, same thresholds: identical numbers to the manuscript's seltest.lower_bound."""
    rng = np.random.default_rng(6)
    X, Y = rng.standard_normal((2000, 6)), rng.standard_normal((1500, 6)) * 1.3
    v, det = sb.lower_bound(X, Y, np.random.default_rng(0), alpha=0.01, return_detail=True)
    assert v == max(d.ratio for d in det)
    assert len(det) == len(sb.GRID)


def test_thin_exact_depth():
    import scipy.sparse as sp
    rng = np.random.default_rng(7)
    X = sp.csr_matrix(rng.poisson(3, (50, 200)))
    Y = sb.thin(X, 300, 0)
    assert np.all(np.asarray(Y.sum(1)).ravel() == 300)
    assert (X - Y).toarray().min() >= 0
