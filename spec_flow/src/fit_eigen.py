"""Eigenvalue least-squares Pixel-R fitting (TSMC spec)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from kron import grounded_eigh
from pixel_r import PixelRModel, build_Gs

BuildGsParamsFn = Callable[..., np.ndarray]


@dataclass
class FitResult:
    Rx: float
    Ry: float
    Rz: float
    residual: float
    relative_spectral_error: float
    lam_M: np.ndarray
    lam_S: np.ndarray
    Q: np.ndarray
    Gs: np.ndarray
    success: bool
    message: str


@dataclass
class FitLogRParamsEigenResult:
    params: dict
    residual: float
    relative_spectral_error: float
    lam_M: np.ndarray
    lam_S: np.ndarray
    Q: np.ndarray
    Gs: np.ndarray
    success: bool
    message: str


def proj_eigs(Q: np.ndarray, Gs: np.ndarray) -> np.ndarray:
    """diag(Q.T @ Gs @ Q) without forming the full product."""
    # (Q * (Gs @ Q)).sum(axis=0) == diag(Q.T @ Gs @ Q)
    return np.sum(Q * (Gs @ Q), axis=0)


# Back-compat alias used by older imports / tests.
_proj_eigs = proj_eigs


def relative_spectral_error(lam_M: np.ndarray, lam_S: np.ndarray) -> float:
    residual = float(np.sum((np.asarray(lam_M) - np.asarray(lam_S)) ** 2))
    denom = float(np.sum(np.asarray(lam_M) ** 2)) + 1e-30
    return float(np.sqrt(residual / denom))


def fit_log_r_params_eigen(
    Gprime: np.ndarray,
    build_Gs_fn: BuildGsParamsFn,
    param_names: Sequence[str],
    *,
    x0: Optional[Sequence[float]] = None,
    bounds: Optional[Tuple[float, float]] = None,
    max_nfev: int = 300,
) -> FitLogRParamsEigenResult:
    """
    min_R || lamM - diag(Q.T Gs(R) Q) ||^2 over positive R parameters.

    ``build_Gs_fn(*values)`` is called with values in the same order as
    ``param_names``. Optimization is in log10(R).
    """
    names = list(param_names)
    if not names:
        raise ValueError("param_names must be non-empty")

    lam_M, Q = grounded_eigh(Gprime)
    g_scale = max(float(np.median(np.diag(Gprime))), 1e-6)
    r_scale = 1.0 / g_scale
    n_par = len(names)
    if x0 is None:
        x0_arr = np.full(n_par, r_scale, dtype=float)
    else:
        x0_arr = np.asarray(x0, dtype=float)
        if x0_arr.shape != (n_par,):
            raise ValueError(f"x0 length {x0_arr.size} != {n_par}")
    if bounds is None:
        bounds = (r_scale * 1e-4, r_scale * 1e4)
    lo, hi = float(bounds[0]), float(bounds[1])
    lo = max(lo, 1e-12)
    hi = max(hi, lo * 10.0)
    log_lo = np.full(n_par, np.log10(lo))
    log_hi = np.full(n_par, np.log10(hi))
    log_x0 = np.clip(np.log10(x0_arr), log_lo, log_hi)

    def fun(log_theta: np.ndarray) -> np.ndarray:
        vals = (10.0 ** log_theta).tolist()
        Gs = build_Gs_fn(*vals)
        return lam_M - proj_eigs(Q, Gs)

    res = least_squares(
        fun,
        x0=log_x0,
        bounds=(log_lo, log_hi),
        method="trf",
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=max_nfev,
    )
    vals = (10.0 ** res.x).tolist()
    params = {name: float(v) for name, v in zip(names, vals)}
    Gs = build_Gs_fn(*vals)
    lam_S = proj_eigs(Q, Gs)
    residual = float(np.sum((lam_M - lam_S) ** 2))
    rel = relative_spectral_error(lam_M, lam_S)
    return FitLogRParamsEigenResult(
        params=params,
        residual=residual,
        relative_spectral_error=rel,
        lam_M=lam_M,
        lam_S=lam_S,
        Q=Q,
        Gs=Gs,
        success=bool(res.success),
        message=str(res.message),
    )


def fit_pixel_r(
    Gprime: np.ndarray,
    model: PixelRModel,
    *,
    x0: Optional[Tuple[float, float, float]] = None,
    bounds: Optional[Tuple[float, float]] = None,
) -> FitResult:
    """
    min_{Rx,Ry,Rz} || lamM - diag(Q.T Gs Q) ||^2

    Optimization is performed in log10(R) space for scale stability.
    """

    def _build(Rx: float, Ry: float, Rz: float) -> np.ndarray:
        return build_Gs(model, float(Rx), float(Ry), float(Rz))

    fit = fit_log_r_params_eigen(
        Gprime,
        _build,
        ("Rx", "Ry", "Rz"),
        x0=x0,
        bounds=bounds,
        max_nfev=300,
    )
    return FitResult(
        Rx=float(fit.params["Rx"]),
        Ry=float(fit.params["Ry"]),
        Rz=float(fit.params["Rz"]),
        residual=fit.residual,
        relative_spectral_error=fit.relative_spectral_error,
        lam_M=fit.lam_M,
        lam_S=fit.lam_S,
        Q=fit.Q,
        Gs=fit.Gs,
        success=fit.success,
        message=fit.message,
    )
