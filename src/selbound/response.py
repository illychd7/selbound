"""Shared-response models K: baseline state -> treated state.

Every model is fitted on FIT cells and applied to held-out EVAL baseline cells, so that the
bound M_L(nu | mu K) is computed on cells the model has not seen (Theorem 2.1 requires K to be
fixed independently of the evaluation cells).

The class of response models decides what the test can reject. Flexible classes keep the test
valid but remove its power (a model that can map mu onto nu exactly returns M_L ~ 1 whatever
the mechanism), so choose the class before looking at the bound and calibrate it.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm


class ResponseModel:
    name = "response"

    def fit(self, Y0: np.ndarray, Yt: np.ndarray) -> "ResponseModel":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, X):
        return self.transform(X)


class Identity(ResponseModel):
    """No response: tests pure selection, nu = S_g mu."""
    name = "identity"

    def transform(self, X):
        return np.asarray(X, float)


class Translation(ResponseModel):
    """One common shift delta = mean(Yt) - mean(Y0). The default, calibrated response model."""
    name = "translation"

    def fit(self, Y0, Yt):
        self.delta_ = np.asarray(Yt, float).mean(0) - np.asarray(Y0, float).mean(0)
        return self

    def transform(self, X):
        return np.asarray(X, float) + self.delta_


class AxisStretch(ResponseModel):
    """Translation plus a stretch along the response axis u = delta / |delta|.

    K(y) = m_t + (y - m_0) + (k - 1) <y - m_0, u> u, with k the ratio of standard deviations
    along u. One parameter more than the translation; allows the response magnitude to vary
    between cells along its own direction.
    """
    name = "axis_stretch"

    def fit(self, Y0, Yt):
        Y0, Yt = np.asarray(Y0, float), np.asarray(Yt, float)
        self.m0_, self.mt_ = Y0.mean(0), Yt.mean(0)
        d = self.mt_ - self.m0_
        self.u_ = d / np.linalg.norm(d)
        self.k_ = float(np.sqrt(np.var((Yt - self.mt_) @ self.u_) / np.var((Y0 - self.m0_) @ self.u_)))
        return self

    def transform(self, X):
        Z = np.asarray(X, float) - self.m0_
        return self.mt_ + Z + (self.k_ - 1) * np.outer(Z @ self.u_, self.u_)


class AffineMomentMatching(ResponseModel):
    """Gaussian optimal-transport affine map matching the first two moments.

    Very flexible: in calibration it returned M_L ~ 1 even with 30% clone-specific response,
    so it keeps validity but has almost no power. Provided as a sensitivity analysis only.
    """
    name = "affine"

    def __init__(self, ridge: float = 1e-6):
        self.ridge = ridge

    def fit(self, Y0, Yt):
        Y0, Yt = np.asarray(Y0, float), np.asarray(Yt, float)
        self.m0_, self.mt_ = Y0.mean(0), Yt.mean(0)
        q = Y0.shape[1]
        S0 = np.cov(Y0.T) + self.ridge * np.eye(q)
        St = np.cov(Yt.T) + self.ridge * np.eye(q)
        r0 = np.real(sqrtm(S0))
        ir0 = np.linalg.inv(r0)
        self.A_ = ir0 @ np.real(sqrtm(r0 @ St @ r0)) @ ir0
        return self

    def transform(self, X):
        return self.mt_ + (np.asarray(X, float) - self.m0_) @ self.A_.T


class Callable(ResponseModel):
    """Wrap a user-supplied, already fitted map f: (n, q) -> (n, q), e.g. a perturbation model
    (scGen, CPA, CellOT) trained on data disjoint from the evaluation cells."""

    def __init__(self, f, name: str = "custom"):
        self.f, self.name = f, name

    def transform(self, X):
        return np.asarray(self.f(np.asarray(X, float)), float)


MODELS = {"identity": Identity, "translation": Translation, "axis_stretch": AxisStretch,
          "affine": AffineMomentMatching}


def get_model(m) -> ResponseModel:
    if isinstance(m, ResponseModel):
        return m
    if isinstance(m, str):
        return MODELS[m]()
    if callable(m):
        return Callable(m)
    raise TypeError(f"cannot interpret response model {m!r}")
