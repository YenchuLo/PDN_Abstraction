"""IR evaluation / optional α polish for VoltSpot-style Eq. 2 grid.

Default is extract-only (α_x=α_y=1), matching the paper's no-fit validation.
Optional --polish fits two global scales on sheet conductances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from _bootstrap import ensure_paths

ensure_paths()

from correlate import _factor_sink_block, build_stimuli, e_ir, solve_mixed_bc  # noqa: E402
from ports_dual import DualPortSet  # noqa: E402
from voltspot_model import VoltSpotModel, build_Gs_voltspot  # noqa: E402

DEFAULT_ALPHA_BOUNDS = (1e-4, 1e4)


@dataclass
class FitVoltSpotResult:
    alpha_x: float
    alpha_y: float
    relative_ir_error: float
    e_worst: float
    e_ir_by_stimulus: Dict[str, float]
    heldout_e_worst: float
    heldout_e_ir_by_stimulus: Dict[str, float]
    Gs: np.ndarray
    e_ir_extract_only: float
    e_ir_after_alpha: float
    polish: bool
    success: bool
    message: str


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


def fit_voltspot_ir(
    Gprime: np.ndarray,
    model: VoltSpotModel,
    ports: DualPortSet,
    *,
    seed: int = 0,
    heldout_seed: int = 1,
    polish: bool = False,
    bounds: Optional[Tuple[float, float]] = None,
) -> FitVoltSpotResult:
    stimuli_all = build_stimuli(ports.as_port_set(), seed=seed)
    if not stimuli_all:
        raise ValueError("need at least one sink stimulus")
    stimuli = [(n, i) for n, i in stimuli_all if n != "random"] or stimuli_all
    heldout = build_stimuli(ports.as_port_set(), seed=heldout_seed)

    n_p = ports.n_pads
    V_pad = np.asarray(ports.pad_voltages, dtype=float)
    vref = float(np.mean(V_pad))
    lu_m, Gsp_m = _factor_sink_block(Gprime, n_p)

    Gs0 = build_Gs_voltspot(model, alpha_x=1.0, alpha_y=1.0)
    e_by0 = _eval_e_by_stimulus(
        Gprime, Gs0, ports, stimuli_all, lu_m=lu_m, Gsp_m=Gsp_m
    )
    e_extract = float(e_by0.get("original", max(e_by0.values())))

    if not polish:
        held_by = {
            f"heldout_{n}": _eval_e_by_stimulus(
                Gprime, Gs0, ports, [(n, i)], lu_m=lu_m, Gsp_m=Gsp_m
            )[n]
            for n, i in heldout
        }
        return FitVoltSpotResult(
            alpha_x=1.0,
            alpha_y=1.0,
            relative_ir_error=e_extract,
            e_worst=float(max(e_by0.values())),
            e_ir_by_stimulus=e_by0,
            heldout_e_worst=float(max(held_by.values()) if held_by else e_extract),
            heldout_e_ir_by_stimulus=held_by,
            Gs=Gs0,
            e_ir_extract_only=e_extract,
            e_ir_after_alpha=e_extract,
            polish=False,
            success=True,
            message="extract_only (no α polish; VoltSpot paper style)",
        )

    drops_m: List[np.ndarray] = []
    norms: List[float] = []
    for name, I_sink in stimuli:
        Vm = solve_mixed_bc(Gprime, n_p, V_pad, I_sink, _lu=lu_m, _Gsp=Gsp_m)
        d = vref - Vm[n_p:]
        drops_m.append(d)
        norms.append(float(np.linalg.norm(d)) + 1e-30)

    if bounds is None:
        bounds = DEFAULT_ALPHA_BOUNDS
    lo, hi = float(bounds[0]), float(bounds[1])
    lo = max(lo, 1e-6)
    hi = max(hi, lo * 10)
    log_lo = np.array([np.log10(lo), np.log10(lo)])
    log_hi = np.array([np.log10(hi), np.log10(hi)])
    weights = [10.0 if name == "original" else 1.0 for name, _ in stimuli]

    def fun(log_theta: np.ndarray) -> np.ndarray:
        ax, ay = (10.0 ** log_theta).tolist()
        Gs = build_Gs_voltspot(model, alpha_x=ax, alpha_y=ay)
        lu_s, Gsp_s = _factor_sink_block(Gs, n_p)
        parts = []
        for j, (_, I_sink) in enumerate(stimuli):
            Vs = solve_mixed_bc(Gs, n_p, V_pad, I_sink, _lu=lu_s, _Gsp=Gsp_s)
            d = vref - Vs[n_p:]
            parts.append(weights[j] * (d - drops_m[j]) / norms[j])
        return np.concatenate(parts)

    best = None
    for s0 in (np.zeros(2), np.log10([0.1, 0.1]), np.log10([10.0, 10.0])):
        s0 = np.clip(s0, log_lo, log_hi)
        res = least_squares(
            fun, x0=s0, bounds=(log_lo, log_hi), method="trf", max_nfev=120
        )
        ax, ay = (10.0 ** res.x).tolist()
        Gs_t = build_Gs_voltspot(model, alpha_x=ax, alpha_y=ay)
        e_by_t = _eval_e_by_stimulus(
            Gprime, Gs_t, ports, stimuli_all, lu_m=lu_m, Gsp_m=Gsp_m
        )
        score = float(e_by_t.get("original", max(e_by_t.values())))
        if best is None or score < best[0]:
            best = (score, ax, ay, Gs_t, e_by_t, res)
    if best is None:
        raise RuntimeError("VoltSpot α polish failed to produce a candidate")
    score, ax, ay, Gs, e_by, res = best
    held_by = {
        f"heldout_{n}": _eval_e_by_stimulus(
            Gprime, Gs, ports, [(n, i)], lu_m=lu_m, Gsp_m=Gsp_m
        )[n]
        for n, i in heldout
    }
    return FitVoltSpotResult(
        alpha_x=float(ax),
        alpha_y=float(ay),
        relative_ir_error=float(e_by.get("original", score)),
        e_worst=float(max(e_by.values())),
        e_ir_by_stimulus=e_by,
        heldout_e_worst=float(max(held_by.values()) if held_by else score),
        heldout_e_ir_by_stimulus=held_by,
        Gs=Gs,
        e_ir_extract_only=e_extract,
        e_ir_after_alpha=float(e_by.get("original", score)),
        polish=True,
        success=bool(res.success),
        message=str(res.message),
    )
