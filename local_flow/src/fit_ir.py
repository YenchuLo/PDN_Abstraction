"""Uniform IR init + multi-stimulus Feng block-τ localized Pixel-R fit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

import importlib.util
from pathlib import Path

from _bootstrap import ensure_paths

ensure_paths()

from correlate import _factor_sink_block, build_stimuli, e_ir, solve_mixed_bc  # noqa: E402
from localized_pixel import (  # noqa: E402
    LocalizedPixelModel,
    build_Gs,
    build_Gs_from_regions,
)
from partitions import expand_region_params  # noqa: E402
from ports import PortSet  # noqa: E402

# Load spec_flow's fit_ir under a unique name (local module is also fit_ir.py).
import sys

_SPEC_FIT = Path(__file__).resolve().parents[2] / "spec_flow" / "src" / "fit_ir.py"
_spec = importlib.util.spec_from_file_location("spec_flow_fit_ir", _SPEC_FIT)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load {_SPEC_FIT}")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["spec_flow_fit_ir"] = _mod
_spec.loader.exec_module(_mod)
fit_pixel_r_ir = _mod.fit_pixel_r_ir

OMEGA = 0.5
TAU_BOUND = 0.1
MAX_FENG_ITERS = 15
RESIDUAL_TOL = 0.005  # 0.5% relative residual change
# Polish when #params is not far above multi-stimulus residual length.
POLISH_PARAM_MARGIN = 1.0
POLISH_BASE_NFEV = 200


@dataclass
class FitLocalizedResult:
    Rx_r: np.ndarray
    Ry_r: np.ndarray
    Rz_r: np.ndarray
    Rx_cell: np.ndarray
    Ry_cell: np.ndarray
    Rz_cell: np.ndarray
    Rx_uniform: float
    Ry_uniform: float
    Rz_uniform: float
    residual: float
    relative_ir_error: float
    residual_uniform: float
    relative_ir_error_uniform: float
    e_ir_by_stimulus: Dict[str, float]
    e_ir_by_stimulus_uniform: Dict[str, float]
    e_worst: float
    Gs: np.ndarray
    Gs_uniform: np.ndarray
    n_feng_iters: int
    polished: bool
    success: bool
    message: str
    stimulus_names: List[str]


def _eval_e_by_stimulus(
    Gprime: np.ndarray,
    Gs: np.ndarray,
    ports: PortSet,
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


def _multi_stimulus_residual(
    Gprime: np.ndarray,
    Gs: np.ndarray,
    ports: PortSet,
    stimuli: List[Tuple[str, np.ndarray]],
    *,
    lu_m=None,
    Gsp_m=None,
    target: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    n_p = ports.n_pads
    V_pad = np.asarray(ports.pad_voltages, dtype=float)
    vref = float(np.mean(V_pad))
    if lu_m is None or Gsp_m is None:
        lu_m, Gsp_m = _factor_sink_block(Gprime, n_p)
    if target is None:
        drops_m: List[np.ndarray] = []
        for _, I_sink in stimuli:
            Vm = solve_mixed_bc(Gprime, n_p, V_pad, I_sink, _lu=lu_m, _Gsp=Gsp_m)
            drops_m.append(vref - Vm[n_p:])
        target = np.concatenate(drops_m)
    lu_s, Gsp_s = _factor_sink_block(Gs, n_p)
    drops_s: List[np.ndarray] = []
    for _, I_sink in stimuli:
        Vs = solve_mixed_bc(Gs, n_p, V_pad, I_sink, _lu=lu_s, _Gsp=Gsp_s)
        drops_s.append(vref - Vs[n_p:])
    residual_vec = np.concatenate(drops_s) - target
    residual = float(np.sum(residual_vec**2))
    return residual, target


def _damp_tau(tau: float, bound: float = TAU_BOUND) -> float:
    if abs(tau - 1.0) > bound:
        return 1.0 + bound if tau > 1.0 else 1.0 - bound
    return float(tau)


def _safe_ratio(num: float, den: float, default: float = 1.0) -> float:
    if abs(den) < 1e-30:
        return default
    return float(num / den)


def _region_edge_sets(
    model: LocalizedPixelModel,
) -> Tuple[
    List[List[Tuple[int, int]]],
    List[List[Tuple[int, int]]],
    List[List[Tuple[int, int]]],
]:
    """Per-region lists of EW edges, NS edges, and (pad, sink) pairs.

    An edge is attributed to a region if *either* endpoint belongs to it
    (shared boundary edges contribute to both). Padless regions have an
    empty pad-pair list (Rz is unidentifiable there).
    """
    part = model.partition
    nr = part.n_regions
    ew_by: List[List[Tuple[int, int]]] = [[] for _ in range(nr)]
    ns_by: List[List[Tuple[int, int]]] = [[] for _ in range(nr)]
    pads_by: List[List[Tuple[int, int]]] = [[] for _ in range(nr)]
    ctr = part.cell_to_region
    star = model.star
    for a, b in star.ew_shared or star.ew_edges:
        a_i, b_i = int(a), int(b)
        for rid in {ctr[a_i], ctr[b_i]}:
            ew_by[rid].append((a_i, b_i))
    for a, b in star.ns_shared or star.ns_edges:
        a_i, b_i = int(a), int(b)
        for rid in {ctr[a_i], ctr[b_i]}:
            ns_by[rid].append((a_i, b_i))
    for p, s in enumerate(star.pad_attach):
        s_i = int(s)
        if 0 <= s_i < len(ctr):
            pads_by[ctr[s_i]].append((int(p), s_i))
    return ew_by, ns_by, pads_by


def _joule_and_ir(
    V: np.ndarray,
    n_p: int,
    G: np.ndarray,
    ew: Sequence[Tuple[int, int]],
    ns: Sequence[Tuple[int, int]],
    pad_pairs: Sequence[Tuple[int, int]],
    sinks: Sequence[int],
    *,
    Rx: Optional[np.ndarray] = None,
    Ry: Optional[np.ndarray] = None,
    Rz: Optional[np.ndarray] = None,
    use_model_g: bool = False,
) -> Tuple[float, float, float, float]:
    """Return (P_H, P_V, P_Z, mean_abs_IR) for a region under port voltages V."""
    Vs = V[n_p:]
    Vp = V[:n_p]
    vref = float(np.mean(Vp)) if n_p else 0.0

    ph = 0.0
    for a, b in ew:
        dv = float(Vs[a] - Vs[b])
        if use_model_g:
            assert Rx is not None
            g = 1.0 / (Rx[a] + Rx[b])
        else:
            g = float(-G[n_p + a, n_p + b])
        ph += g * dv * dv

    pv = 0.0
    for a, b in ns:
        dv = float(Vs[a] - Vs[b])
        if use_model_g:
            assert Ry is not None
            g = 1.0 / (Ry[a] + Ry[b])
        else:
            g = float(-G[n_p + a, n_p + b])
        pv += g * dv * dv

    pz = 0.0
    for p, s in pad_pairs:
        dv = float(Vp[p] - Vs[s])
        if use_model_g:
            assert Rz is not None
            g = 1.0 / Rz[s]
        else:
            g = float(-G[p, n_p + s])
        pz += g * dv * dv
    ir_sum = 0.0
    for s in sinks:
        ir_sum += abs(vref - float(Vs[s]))
    mean_ir = ir_sum / max(len(sinks), 1)
    return ph, pv, pz, mean_ir


def _feng_iteration(
    model: LocalizedPixelModel,
    Gprime: np.ndarray,
    ports: PortSet,
    stimuli: List[Tuple[str, np.ndarray]],
    Rx_r: np.ndarray,
    Ry_r: np.ndarray,
    Rz_r: np.ndarray,
    *,
    lu_m,
    Gsp_m,
    omega: float = OMEGA,
    bound: float = TAU_BOUND,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_p = ports.n_pads
    V_pad = np.asarray(ports.pad_voltages, dtype=float)
    Rx_c, Ry_c, Rz_c = expand_region_params(model.partition, Rx_r, Ry_r, Rz_r)
    Gs = build_Gs(model, Rx_c, Ry_c, Rz_c)
    lu_s, Gsp_s = _factor_sink_block(Gs, n_p)
    ew_by, ns_by, pads_by = _region_edge_sets(model)
    nr = model.n_regions

    # Accumulate τ factors per region over stimuli (start at 0; average later).
    tau_h = np.zeros(nr, dtype=float)
    tau_v = np.zeros(nr, dtype=float)
    tau_z = np.zeros(nr, dtype=float)
    counts = np.zeros(nr, dtype=float)
    power_eps = 1e-30

    for _, I_sink in stimuli:
        Vm = solve_mixed_bc(Gprime, n_p, V_pad, I_sink, _lu=lu_m, _Gsp=Gsp_m)
        Vs = solve_mixed_bc(Gs, n_p, V_pad, I_sink, _lu=lu_s, _Gsp=Gsp_s)
        for rid in range(nr):
            cells_r = model.partition.region_cells[rid]
            ph_m, pv_m, pz_m, ir_m = _joule_and_ir(
                Vm, n_p, Gprime, ew_by[rid], ns_by[rid], pads_by[rid], cells_r
            )
            ph_s, pv_s, pz_s, ir_s = _joule_and_ir(
                Vs,
                n_p,
                Gs,
                ew_by[rid],
                ns_by[rid],
                pads_by[rid],
                cells_r,
                Rx=Rx_c,
                Ry=Ry_c,
                Rz=Rz_c,
                use_model_g=True,
            )
            ir_ratio = _safe_ratio(ir_s, ir_m, 1.0)
            # Fall back to IR-only τ when directional Joule on dense G' is tiny.
            if ph_m > power_eps and ph_s > power_eps:
                th = omega * _safe_ratio(ph_s, ph_m) + (1.0 - omega) * ir_ratio
            else:
                th = ir_ratio
            if pv_m > power_eps and pv_s > power_eps:
                tv = omega * _safe_ratio(pv_s, pv_m) + (1.0 - omega) * ir_ratio
            else:
                tv = ir_ratio
            if not pads_by[rid]:
                # Padless region: Rz does not appear in G_S; leave it unchanged.
                tz = 1.0
            elif pz_m > power_eps and pz_s > power_eps:
                tz = omega * _safe_ratio(pz_s, pz_m) + (1.0 - omega) * ir_ratio
            else:
                tz = ir_ratio
            tau_h[rid] += th
            tau_v[rid] += tv
            tau_z[rid] += tz
            counts[rid] += 1.0

    Rx_new = Rx_r.copy()
    Ry_new = Ry_r.copy()
    Rz_new = Rz_r.copy()
    for rid in range(nr):
        c = max(counts[rid], 1.0)
        th = _damp_tau(tau_h[rid] / c, bound)
        tv = _damp_tau(tau_v[rid] / c, bound)
        tz = _damp_tau(tau_z[rid] / c, bound)
        # Clamp away from zero to keep R positive and finite.
        th = max(th, 1e-3)
        tv = max(tv, 1e-3)
        tz = max(tz, 1e-3)
        Rx_new[rid] = float(Rx_r[rid] / th)
        Ry_new[rid] = float(Ry_r[rid] / tv)
        Rz_new[rid] = float(Rz_r[rid] / tz)
    return Rx_new, Ry_new, Rz_new


def _optional_polish(
    model: LocalizedPixelModel,
    Gprime: np.ndarray,
    ports: PortSet,
    stimuli: List[Tuple[str, np.ndarray]],
    Rx_r: np.ndarray,
    Ry_r: np.ndarray,
    Rz_r: np.ndarray,
    *,
    lu_m,
    Gsp_m,
    target: np.ndarray,
    max_nfev: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    nr = model.n_regions
    n_par = 3 * nr
    # Skip only when heavily underdetermined vs multi-stimulus residual length.
    if n_par > POLISH_PARAM_MARGIN * len(target):
        return Rx_r, Ry_r, Rz_r, False
    if max_nfev is None:
        max_nfev = max(POLISH_BASE_NFEV, 2 * n_par)

    n_p = ports.n_pads
    V_pad = np.asarray(ports.pad_voltages, dtype=float)
    vref = float(np.mean(V_pad))
    x0 = np.concatenate([Rx_r, Ry_r, Rz_r])
    lo = np.full(n_par, np.min(x0) * 1e-3)
    hi = np.full(n_par, np.max(x0) * 1e3)
    lo = np.maximum(lo, 1e-12)
    log_lo = np.log10(lo)
    log_hi = np.log10(hi)
    log_x0 = np.clip(np.log10(x0), log_lo, log_hi)

    def fun(log_theta: np.ndarray) -> np.ndarray:
        vals = 10.0 ** log_theta
        rx = vals[:nr]
        ry = vals[nr : 2 * nr]
        rz = vals[2 * nr :]
        Gs = build_Gs_from_regions(model, rx, ry, rz)
        lu_s, Gsp_s = _factor_sink_block(Gs, n_p)
        drops_s: List[np.ndarray] = []
        for _, I_sink in stimuli:
            Vs = solve_mixed_bc(Gs, n_p, V_pad, I_sink, _lu=lu_s, _Gsp=Gsp_s)
            drops_s.append(vref - Vs[n_p:])
        resid = np.concatenate(drops_s) - target
        # Up-weight the first stimulus ("original") so polish does not
        # trade away the primary IR map for synthetic currents.
        n_s = ports.n_sinks
        w = np.ones(resid.size, dtype=float)
        w[:n_s] = 4.0
        return resid * np.sqrt(w)

    res = least_squares(
        fun,
        x0=log_x0,
        bounds=(log_lo, log_hi),
        method="trf",
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        max_nfev=max_nfev,
    )
    vals = 10.0 ** res.x
    return vals[:nr], vals[nr : 2 * nr], vals[2 * nr :], True


def fit_localized_ir(
    Gprime: np.ndarray,
    model: LocalizedPixelModel,
    ports: PortSet,
    *,
    seed: int = 0,
    max_feng_iters: int = MAX_FENG_ITERS,
    residual_tol: float = RESIDUAL_TOL,
) -> FitLocalizedResult:
    if ports.n_pads == 0:
        raise ValueError("need at least one pad for mixed-BC IR fit")
    stimuli = build_stimuli(ports, seed=seed)
    if not stimuli:
        raise ValueError("need at least one sink stimulus for IR fit")

    n_p = ports.n_pads
    lu_m, Gsp_m = _factor_sink_block(Gprime, n_p)

    # --- uniform IR init (spec_flow) ---
    uni = fit_pixel_r_ir(Gprime, model.star, ports, seed=seed)
    Rx_u, Ry_u, Rz_u = float(uni.Rx), float(uni.Ry), float(uni.Rz)
    nr = model.n_regions
    Rx_r = np.full(nr, Rx_u, dtype=float)
    Ry_r = np.full(nr, Ry_u, dtype=float)
    Rz_r = np.full(nr, Rz_u, dtype=float)

    Gs_uni = build_Gs_from_regions(model, Rx_r, Ry_r, Rz_r)
    residual_uni, target = _multi_stimulus_residual(
        Gprime, Gs_uni, ports, stimuli, lu_m=lu_m, Gsp_m=Gsp_m
    )
    denom = float(np.sum(target**2)) + 1e-30
    rel_uni = float(np.sqrt(residual_uni / denom))
    e_uni = _eval_e_by_stimulus(
        Gprime, Gs_uni, ports, stimuli, lu_m=lu_m, Gsp_m=Gsp_m
    )

    # --- Feng block τ (keep best residual iterate) ---
    best_Rx, best_Ry, best_Rz = Rx_r.copy(), Ry_r.copy(), Rz_r.copy()
    best_residual = residual_uni
    residual_prev = residual_uni
    n_iters = 0
    for it in range(max_feng_iters):
        n_iters = it + 1
        Rx_try, Ry_try, Rz_try = _feng_iteration(
            model,
            Gprime,
            ports,
            stimuli,
            Rx_r,
            Ry_r,
            Rz_r,
            lu_m=lu_m,
            Gsp_m=Gsp_m,
        )
        Gs = build_Gs_from_regions(model, Rx_try, Ry_try, Rz_try)
        residual, _ = _multi_stimulus_residual(
            Gprime, Gs, ports, stimuli, lu_m=lu_m, Gsp_m=Gsp_m, target=target
        )
        # Accept only improving (or equal) steps to avoid Feng divergence on dense G'.
        if residual <= best_residual * (1.0 + 1e-12):
            best_residual = residual
            best_Rx, best_Ry, best_Rz = Rx_try.copy(), Ry_try.copy(), Rz_try.copy()
            Rx_r, Ry_r, Rz_r = Rx_try, Ry_try, Rz_try
        else:
            # Reject step; keep previous accepted params.
            Rx_r, Ry_r, Rz_r = best_Rx.copy(), best_Ry.copy(), best_Rz.copy()
            residual = best_residual

        if residual_prev > 1e-30:
            rel_change = abs(residual - residual_prev) / residual_prev
        else:
            rel_change = 0.0
        residual_prev = residual
        if rel_change < residual_tol and it > 0:
            break

    Rx_r, Ry_r, Rz_r = best_Rx, best_Ry, best_Rz

    polished = False
    Rx_p, Ry_p, Rz_p, did_polish = _optional_polish(
        model,
        Gprime,
        ports,
        stimuli,
        Rx_r,
        Ry_r,
        Rz_r,
        lu_m=lu_m,
        Gsp_m=Gsp_m,
        target=target,
    )
    if did_polish:
        Gs_p = build_Gs_from_regions(model, Rx_p, Ry_p, Rz_p)
        residual_p, _ = _multi_stimulus_residual(
            Gprime, Gs_p, ports, stimuli, lu_m=lu_m, Gsp_m=Gsp_m, target=target
        )
        e_p = _eval_e_by_stimulus(
            Gprime, Gs_p, ports, stimuli, lu_m=lu_m, Gsp_m=Gsp_m
        )
        e_orig_p = float(e_p.get("original", residual_p))
        e_orig_uni = float(e_uni.get("original", rel_uni))
        # Accept polish if original IR improves (or ties) vs uniform baseline.
        if e_orig_p <= e_orig_uni * (1.0 + 1e-9):
            Rx_r, Ry_r, Rz_r = Rx_p, Ry_p, Rz_p
            best_residual = residual_p
            polished = True

    Gs = build_Gs_from_regions(model, Rx_r, Ry_r, Rz_r)
    residual = best_residual
    rel = float(np.sqrt(residual / denom))
    e_by = _eval_e_by_stimulus(Gprime, Gs, ports, stimuli, lu_m=lu_m, Gsp_m=Gsp_m)
    Rx_c, Ry_c, Rz_c = expand_region_params(model.partition, Rx_r, Ry_r, Rz_r)

    return FitLocalizedResult(
        Rx_r=np.asarray(Rx_r, dtype=float),
        Ry_r=np.asarray(Ry_r, dtype=float),
        Rz_r=np.asarray(Rz_r, dtype=float),
        Rx_cell=Rx_c,
        Ry_cell=Ry_c,
        Rz_cell=Rz_c,
        Rx_uniform=Rx_u,
        Ry_uniform=Ry_u,
        Rz_uniform=Rz_u,
        residual=residual,
        relative_ir_error=rel,
        residual_uniform=residual_uni,
        relative_ir_error_uniform=rel_uni,
        e_ir_by_stimulus=e_by,
        e_ir_by_stimulus_uniform=e_uni,
        e_worst=float(max(e_by.values()) if e_by else rel),
        Gs=Gs,
        Gs_uniform=Gs_uni,
        n_feng_iters=n_iters,
        polished=polished,
        success=True,
        message=f"feng_iters={n_iters} polished={polished} best_rel={rel:.6g}",
        stimulus_names=[n for n, _ in stimuli],
    )
