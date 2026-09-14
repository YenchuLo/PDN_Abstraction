"""Shared multi-stimulus IR fit of global Rx, Ry, Rz (spec_flow IR objective).

Matches ``spec_flow.fit_ir.fit_pixel_r_ir``:
  min_{Rx,Ry,Rz} sum_j || drop_S(I_j) - drop_M(I_j) ||^2
with mixed BC, log10(R), all stimuli (including random). Topology is supplied
via ``build_Gs(Rx, Ry, Rz)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from _bootstrap import ensure_paths

ensure_paths()

from correlate import _factor_sink_block, build_stimuli, e_ir, solve_mixed_bc  # noqa: E402
from ports_dual import DualPortSet  # noqa: E402

BuildGsFn = Callable[[float, float, float], np.ndarray]
BuildGsParamsFn = Callable[..., np.ndarray]


@dataclass
class FitRxRyRzResult:
    Rx: float
    Ry: float
    Rz: float
    residual: float
    relative_ir_error: float
    e_worst: float
    e_ir_by_stimulus: Dict[str, float]
    heldout_e_worst: float
    heldout_e_ir_by_stimulus: Dict[str, float]
    Gs: np.ndarray
    e_ir_x0: float
    success: bool
    message: str
    stimulus_names: List[str]


def _eval_e_by_stimulus(
    Gprime: np.ndarray,
    Gs: np.ndarray,
    ports: DualPortSet,
    stimuli: List[Tuple[str, np.ndarray]],
    *,
    lu_m=None,
    Gsp_m=None,
) -> Dict[str, float]:
    n_p = ports.n_pads
    V_pad = np.asarray(ports.pad_voltages, dtype=float)
    vref = float(np.mean(V_pad))
    if lu_m is None or Gsp_m is None:
        lu_m, Gsp_m = _factor_sink_block(Gprime, n_p)
    lu_s, Gsp_s = _factor_sink_block(Gs, n_p)
    out: Dict[str, float] = {}
    for name, I_sink in stimuli:
        Vm = solve_mixed_bc(Gprime, n_p, V_pad, I_sink, _lu=lu_m, _Gsp=Gsp_m)
        Vs = solve_mixed_bc(Gs, n_p, V_pad, I_sink, _lu=lu_s, _Gsp=Gsp_s)
        out[name] = e_ir(vref - Vs[n_p:], vref - Vm[n_p:])
    return out


def fit_rxryrz_ir(
    Gprime: np.ndarray,
    ports: DualPortSet,
    build_Gs: BuildGsFn,
    *,
    seed: int = 0,
    heldout_seed: int = 1,
    x0: Optional[Tuple[float, float, float]] = None,
    bounds: Optional[Tuple[float, float]] = None,
    max_nfev: int = 300,
) -> FitRxRyRzResult:
    """
    min_{Rx,Ry,Rz} sum_j ||drop_S(I_j) - drop_M(I_j)||^2  (spec_flow IR).

    Optimization in log10(R). ``build_Gs(Rx, Ry, Rz)`` stamps the model.
    ``heldout_seed`` only affects post-fit diagnostic metrics.
    """
    if ports.n_pads == 0:
        raise ValueError("need at least one pad for mixed-BC IR fit")
    stimuli = build_stimuli(ports.as_port_set(), seed=seed)
    if not stimuli:
        raise ValueError("need at least one sink stimulus for IR fit")
    names = [n for n, _ in stimuli]
    heldout = build_stimuli(ports.as_port_set(), seed=heldout_seed)

    n_p = ports.n_pads
    V_pad = np.asarray(ports.pad_voltages, dtype=float)
    vref = float(np.mean(V_pad))
    lu_m, Gsp_m = _factor_sink_block(Gprime, n_p)

    drops_m: List[np.ndarray] = []
    for _name, I_sink in stimuli:
        Vm = solve_mixed_bc(Gprime, n_p, V_pad, I_sink, _lu=lu_m, _Gsp=Gsp_m)
        drops_m.append(vref - Vm[n_p:])
    target = np.concatenate(drops_m)

    g_scale = max(float(np.median(np.diag(Gprime))), 1e-6)
    r_scale = 1.0 / g_scale
    if x0 is None:
        x0 = (r_scale, r_scale, r_scale)
    if bounds is None:
        bounds = (r_scale * 1e-4, r_scale * 1e4)
    lo, hi = float(bounds[0]), float(bounds[1])
    lo = max(lo, 1e-12)
    hi = max(hi, lo * 10.0)
    log_lo = np.log10(lo)
    log_hi = np.log10(hi)
    log_x0 = np.clip(np.log10(np.asarray(x0, dtype=float)), log_lo, log_hi)

    Gs0 = build_Gs(float(x0[0]), float(x0[1]), float(x0[2]))
    e_by0 = _eval_e_by_stimulus(
        Gprime, Gs0, ports, stimuli, lu_m=lu_m, Gsp_m=Gsp_m
    )
    e_x0 = float(e_by0.get("original", max(e_by0.values())))

    def fun(log_theta: np.ndarray) -> np.ndarray:
        Rx, Ry, Rz = (10.0 ** log_theta).tolist()
        Gs = build_Gs(float(Rx), float(Ry), float(Rz))
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
        max_nfev=max_nfev,
    )
    Rx, Ry, Rz = (10.0 ** res.x).tolist()
    Gs = build_Gs(float(Rx), float(Ry), float(Rz))
    residual_vec = fun(res.x)
    residual = float(np.sum(residual_vec**2))
    denom = float(np.sum(target**2)) + 1e-30
    rel = float(np.sqrt(residual / denom))

    e_by = _eval_e_by_stimulus(Gprime, Gs, ports, stimuli, lu_m=lu_m, Gsp_m=Gsp_m)
    e_worst = float(e_by.get("original", max(e_by.values())))
    held_by: Dict[str, float] = {}
    for name, I_sink in heldout:
        one = _eval_e_by_stimulus(
            Gprime, Gs, ports, [(name, I_sink)], lu_m=lu_m, Gsp_m=Gsp_m
        )
        held_by[f"heldout_{name}"] = one[name]
    held_worst = float(
        held_by.get("heldout_original", max(held_by.values()) if held_by else 0.0)
    )
    return FitRxRyRzResult(
        Rx=float(Rx),
        Ry=float(Ry),
        Rz=float(Rz),
        residual=residual,
        relative_ir_error=rel,
        e_worst=e_worst,
        e_ir_by_stimulus=e_by,
        heldout_e_worst=held_worst,
        heldout_e_ir_by_stimulus=held_by,
        Gs=Gs,
        e_ir_x0=e_x0,
        success=bool(res.success),
        message=str(res.message),
        stimulus_names=names,
    )


@dataclass
class FitLogRParamsResult:
    """IR fit of an arbitrary positive R-parameter vector (log10 space)."""

    params: Dict[str, float]
    residual: float
    relative_ir_error: float
    e_worst: float
    e_ir_by_stimulus: Dict[str, float]
    heldout_e_worst: float
    heldout_e_ir_by_stimulus: Dict[str, float]
    Gs: np.ndarray
    e_ir_x0: float
    success: bool
    message: str
    stimulus_names: List[str]


def fit_log_r_params_ir(
    Gprime: np.ndarray,
    ports: DualPortSet,
    build_Gs: BuildGsParamsFn,
    param_names: List[str],
    *,
    seed: int = 0,
    heldout_seed: int = 1,
    x0: Optional[Sequence[float]] = None,
    bounds: Optional[Tuple[float, float]] = None,
    max_nfev: int = 400,
) -> FitLogRParamsResult:
    """
    min_R sum_j ||drop_S(I_j) - drop_M(I_j)||^2 over positive R parameters.

    ``build_Gs(*values)`` is called with values in the same order as
    ``param_names``. Optimization is in log10(R).
    """
    if not param_names:
        raise ValueError("param_names must be non-empty")
    if ports.n_pads == 0:
        raise ValueError("need at least one pad for mixed-BC IR fit")
    stimuli = build_stimuli(ports.as_port_set(), seed=seed)
    if not stimuli:
        raise ValueError("need at least one sink stimulus for IR fit")
    names = [n for n, _ in stimuli]
    heldout = build_stimuli(ports.as_port_set(), seed=heldout_seed)

    n_p = ports.n_pads
    V_pad = np.asarray(ports.pad_voltages, dtype=float)
    vref = float(np.mean(V_pad))
    lu_m, Gsp_m = _factor_sink_block(Gprime, n_p)

    drops_m: List[np.ndarray] = []
    for _name, I_sink in stimuli:
        Vm = solve_mixed_bc(Gprime, n_p, V_pad, I_sink, _lu=lu_m, _Gsp=Gsp_m)
        drops_m.append(vref - Vm[n_p:])
    target = np.concatenate(drops_m)

    g_scale = max(float(np.median(np.diag(Gprime))), 1e-6)
    r_scale = 1.0 / g_scale
    n_par = len(param_names)
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

    Gs0 = build_Gs(*x0_arr.tolist())
    e_by0 = _eval_e_by_stimulus(
        Gprime, Gs0, ports, stimuli, lu_m=lu_m, Gsp_m=Gsp_m
    )
    e_x0 = float(e_by0.get("original", max(e_by0.values())))

    def fun(log_theta: np.ndarray) -> np.ndarray:
        vals = (10.0 ** log_theta).tolist()
        Gs = build_Gs(*vals)
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
        max_nfev=max_nfev,
    )
    vals = (10.0 ** res.x).tolist()
    params = {name: float(v) for name, v in zip(param_names, vals)}
    Gs = build_Gs(*vals)
    residual_vec = fun(res.x)
    residual = float(np.sum(residual_vec**2))
    denom = float(np.sum(target**2)) + 1e-30
    rel = float(np.sqrt(residual / denom))

    e_by = _eval_e_by_stimulus(Gprime, Gs, ports, stimuli, lu_m=lu_m, Gsp_m=Gsp_m)
    e_worst = float(e_by.get("original", max(e_by.values())))
    held_by: Dict[str, float] = {}
    for name, I_sink in heldout:
        one = _eval_e_by_stimulus(
            Gprime, Gs, ports, [(name, I_sink)], lu_m=lu_m, Gsp_m=Gsp_m
        )
        held_by[f"heldout_{name}"] = one[name]
    held_worst = float(
        held_by.get("heldout_original", max(held_by.values()) if held_by else 0.0)
    )
    return FitLogRParamsResult(
        params=params,
        residual=residual,
        relative_ir_error=rel,
        e_worst=e_worst,
        e_ir_by_stimulus=e_by,
        heldout_e_worst=held_worst,
        heldout_e_ir_by_stimulus=held_by,
        Gs=Gs,
        e_ir_x0=e_x0,
        success=bool(res.success),
        message=str(res.message),
        stimulus_names=names,
    )
