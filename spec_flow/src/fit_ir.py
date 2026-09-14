"""Multi-stimulus mixed-BC IR Pixel-R fitting (physics-response path)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from correlate import _factor_sink_block, build_stimuli, solve_mixed_bc
from pixel_r import PixelRModel, build_Gs
from ports import PortSet


@dataclass
class FitIRResult:
    Rx: float
    Ry: float
    Rz: float
    residual: float
    relative_ir_error: float
    Gs: np.ndarray
    stimulus_names: List[str]
    success: bool
    message: str


def fit_pixel_r_ir(
    Gprime: np.ndarray,
    model: PixelRModel,
    ports: PortSet,
    *,
    seed: int = 0,
    x0: Optional[Tuple[float, float, float]] = None,
    bounds: Optional[Tuple[float, float]] = None,
) -> FitIRResult:
    """
    min_{Rx,Ry,Rz} sum_j || V_S(I_j) - V_M(I_j) ||^2

    Mixed BC: pads voltage-fixed, sinks current-driven. Optimization in log10(R).
    """
    if ports.n_pads == 0:
        raise ValueError("need at least one pad for mixed-BC IR fit")
    stimuli = build_stimuli(ports, seed=seed)
    if not stimuli:
        raise ValueError("need at least one sink stimulus for IR fit")

    n_p = ports.n_pads
    V_pad = np.asarray(ports.pad_voltages, dtype=float)
    vref = float(np.mean(V_pad))
    lu_m, Gsp_m = _factor_sink_block(Gprime, n_p)

    drops_m: List[np.ndarray] = []
    names: List[str] = []
    for name, I_sink in stimuli:
        Vm = solve_mixed_bc(Gprime, n_p, V_pad, I_sink, _lu=lu_m, _Gsp=Gsp_m)
        drops_m.append(vref - Vm[n_p:])
        names.append(name)
    target = np.concatenate(drops_m)

    g_scale = max(float(np.median(np.diag(Gprime))), 1e-6)
    r_scale = 1.0 / g_scale
    if x0 is None:
        x0 = (r_scale, r_scale, r_scale)
    if bounds is None:
        bounds = (r_scale * 1e-4, r_scale * 1e4)

    lo, hi = bounds
    lo = max(lo, 1e-12)
    hi = max(hi, lo * 10.0)
    log_lo = np.log10(lo)
    log_hi = np.log10(hi)
    log_x0 = np.clip(np.log10(np.asarray(x0, dtype=float)), log_lo, log_hi)

    def fun(log_theta: np.ndarray) -> np.ndarray:
        Rx, Ry, Rz = (10.0 ** log_theta).tolist()
        Gs = build_Gs(model, float(Rx), float(Ry), float(Rz))
        lu_s, Gsp_s = _factor_sink_block(Gs, n_p)
        drops_s: List[np.ndarray] = []
        for _, I_sink in stimuli:
            Vs = solve_mixed_bc(Gs, n_p, V_pad, I_sink, _lu=lu_s, _Gsp=Gsp_s)
            drops_s.append(vref - Vs[n_p:])
        return np.concatenate(drops_s) - target

    res = least_squares(
        fun,
        x0=log_x0,
        bounds=(log_lo, log_hi),
        method="trf",
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=300,
    )
    Rx, Ry, Rz = (10.0 ** res.x).tolist()
    Gs = build_Gs(model, Rx, Ry, Rz)
    residual_vec = fun(res.x)
    residual = float(np.sum(residual_vec**2))
    denom = float(np.sum(target**2)) + 1e-30
    rel = float(np.sqrt(residual / denom))

    return FitIRResult(
        Rx=float(Rx),
        Ry=float(Ry),
        Rz=float(Rz),
        residual=residual,
        relative_ir_error=rel,
        Gs=Gs,
        stimulus_names=names,
        success=bool(res.success),
        message=str(res.message),
    )
