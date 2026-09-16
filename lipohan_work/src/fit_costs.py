"""Fit Pixel-R (Rx, Ry, Rz) under a named cost function."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares, minimize

from _bootstrap import ensure_paths

ensure_paths()

from costs import (  # noqa: E402
    SCALAR_COSTS,
    VECTOR_COSTS,
    FitContext,
    canonical_cost,
    make_residual_fn,
    make_scalar_fn,
)
from pixel_r import build_Gs  # noqa: E402


@dataclass
class CostFitResult:
    cost: str
    Rx: float
    Ry: float
    Rz: float
    Gs: np.ndarray
    training_loss: float
    success: bool
    message: str
    nfev: int
    elapsed_s: float
    warm_start_from: str = ""


def _bounds(ctx: FitContext, bounds: Optional[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    if bounds is None:
        lo, hi = ctx.r_scale * 1e-4, ctx.r_scale * 1e4
    else:
        lo, hi = float(bounds[0]), float(bounds[1])
    lo = max(lo, 1e-12)
    hi = max(hi, lo * 10.0)
    log_lo = np.full(3, np.log10(lo))
    log_hi = np.full(3, np.log10(hi))
    return log_lo, log_hi


def _log_x0(
    ctx: FitContext,
    x0: Optional[Tuple[float, float, float]],
    log_lo: np.ndarray,
    log_hi: np.ndarray,
) -> np.ndarray:
    if x0 is None:
        x0_arr = np.full(3, ctx.r_scale, dtype=float)
    else:
        x0_arr = np.asarray(x0, dtype=float)
    return np.clip(np.log10(x0_arr), log_lo, log_hi)


def fit_cost(
    ctx: FitContext,
    cost: str,
    *,
    x0: Optional[Tuple[float, float, float]] = None,
    bounds: Optional[Tuple[float, float]] = None,
    max_nfev: int = 300,
    warm_start: bool = True,
) -> CostFitResult:
    """Minimize the named cost over log10(Rx, Ry, Rz)."""
    c = canonical_cost(cost)
    log_lo, log_hi = _bounds(ctx, bounds)
    log_x0 = _log_x0(ctx, x0, log_lo, log_hi)
    warm = ""
    t0 = time.perf_counter()

    if c == "minimax" and warm_start and x0 is None:
        # Minimax is non-smooth; start from the voltage LS solution.
        warm_fit = fit_cost(
            ctx, "voltage", x0=x0, bounds=bounds, max_nfev=max_nfev, warm_start=False
        )
        log_x0 = np.clip(
            np.log10(np.array([warm_fit.Rx, warm_fit.Ry, warm_fit.Rz])), log_lo, log_hi
        )
        warm = "voltage"

    if c in VECTOR_COSTS:
        fun = make_residual_fn(ctx, c)
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
        log_theta = np.asarray(res.x, dtype=float)
        success = bool(res.success)
        message = str(res.message)
        nfev = int(res.nfev)
        training_loss = float(np.sum(np.square(res.fun)))
    elif c in SCALAR_COSTS:
        fun = make_scalar_fn(ctx, c)
        res = minimize(
            fun,
            x0=log_x0,
            method="Nelder-Mead",
            options={"maxiter": max(max_nfev, 400), "xatol": 1e-8, "fatol": 1e-12, "adaptive": True},
        )
        log_theta = np.clip(np.asarray(res.x, dtype=float), log_lo, log_hi)
        success = bool(res.success)
        message = str(res.message)
        nfev = int(getattr(res, "nfev", 0))
        training_loss = float(res.fun)
    else:
        raise ValueError(
            f"unknown cost {cost!r}; known: {sorted(VECTOR_COSTS | SCALAR_COSTS)}"
        )

    Rx, Ry, Rz = (10.0 ** log_theta).tolist()
    Gs = build_Gs(ctx.model, Rx, Ry, Rz)
    return CostFitResult(
        cost=c,
        Rx=float(Rx),
        Ry=float(Ry),
        Rz=float(Rz),
        Gs=Gs,
        training_loss=training_loss,
        success=success,
        message=message,
        nfev=nfev,
        elapsed_s=float(time.perf_counter() - t0),
        warm_start_from=warm,
    )
