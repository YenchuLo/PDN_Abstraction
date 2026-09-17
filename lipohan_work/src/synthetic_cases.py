"""Synthetic PDN cases and held-out current evaluation for cost studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from _bootstrap import ensure_paths

ensure_paths()

from correlate import _factor_sink_block, e_ir, solve_mixed_bc  # noqa: E402
from pixel_r import PixelRModel, build_Gs, build_pixel_model  # noqa: E402
from ports import GridCell, PortSet  # noqa: E402

CurrentStimulus = Tuple[str, np.ndarray]
BoundaryStimulus = Tuple[str, np.ndarray, np.ndarray]


@dataclass
class SyntheticCase:
    name: str
    description: str
    ports: PortSet
    model: PixelRModel
    Gprime: np.ndarray
    truth_params: Dict[str, float]


def build_fixture(n: int = 5) -> Tuple[PortSet, PixelRModel]:
    """Build a sparse-pad square lattice with a nonuniform physical load map."""
    if n < 3:
        raise ValueError("synthetic grid size must be at least 3")
    vdd = 1.8
    center = 0.5 * (n - 1)
    cells: List[GridCell] = []
    for iy in range(n):
        for ix in range(n):
            dx = (ix - 0.70 * (n - 1)) / max(0.28 * n, 1.0)
            dy = (iy - 0.35 * (n - 1)) / max(0.24 * n, 1.0)
            hotspot = np.exp(-0.5 * (dx * dx + dy * dy))
            gradient = 0.25 * (ix + iy) / max(2 * (n - 1), 1)
            draw = 0.010 * (0.35 + 2.0 * hotspot + gradient)
            k = iy * n + ix
            cells.append(
                GridCell(
                    ix=ix,
                    iy=iy,
                    node=f"s{k}",
                    x=float(ix),
                    y=float(iy),
                    current=-float(draw),
                )
            )

    pad_axis = sorted({0, int(round(center)), n - 1})
    pad_attach = [iy * n + ix for iy in pad_axis for ix in pad_axis]
    pad_nodes = [f"p{k}" for k in range(len(pad_attach))]
    pad_xy = [(cells[s].x, cells[s].y) for s in pad_attach]
    ports = PortSet(
        pad_nodes=pad_nodes,
        pad_voltages=[vdd] * len(pad_nodes),
        cells=cells,
        cell_size=1.0,
        nx=n,
        ny=n,
        bbox=(-0.5, -0.5, n - 0.5, n - 0.5),
        pad_xy=pad_xy,
        pad_attach=pad_attach,
        vdd=vdd,
    )
    return ports, build_pixel_model(ports)


def build_variable_gs(
    model: PixelRModel,
    Rx_cell: Sequence[float],
    Ry_cell: Sequence[float],
    Rz_cell: Sequence[float],
) -> np.ndarray:
    """Stamp a localized Pixel-R truth while fitting remains global 3R."""
    n_p = model.n_pads
    n_s = model.n_sinks
    rx = np.asarray(Rx_cell, dtype=float)
    ry = np.asarray(Ry_cell, dtype=float)
    rz = np.asarray(Rz_cell, dtype=float)
    if rx.shape != (n_s,) or ry.shape != (n_s,) or rz.shape != (n_s,):
        raise ValueError("per-cell resistance arrays must match n_sinks")
    if np.any(rx <= 0.0) or np.any(ry <= 0.0) or np.any(rz <= 0.0):
        raise ValueError("synthetic resistances must be positive")

    G = np.zeros((n_p + n_s, n_p + n_s), dtype=float)

    def stamp(i: int, j: int, conductance: float) -> None:
        G[i, i] += conductance
        G[j, j] += conductance
        G[i, j] -= conductance
        G[j, i] -= conductance

    for a, b in model.ew_shared or model.ew_edges:
        stamp(n_p + int(a), n_p + int(b), 1.0 / (rx[int(a)] + rx[int(b)]))
    for a, b in model.ns_shared or model.ns_edges:
        stamp(n_p + int(a), n_p + int(b), 1.0 / (ry[int(a)] + ry[int(b)]))
    for pad, sink in enumerate(model.pad_attach):
        stamp(pad, n_p + int(sink), 1.0 / rz[int(sink)])
    return 0.5 * (G + G.T)


def add_long_range_coupling(
    G: np.ndarray,
    model: PixelRModel,
    *,
    conductance: float = 0.12,
) -> np.ndarray:
    """Add nonlocal sink edges that a nearest-neighbour Pixel-R cannot express."""
    out = np.array(G, dtype=float, copy=True)
    n = model.ports.nx
    mid = n // 2
    loc = {(c.ix, c.iy): k for k, c in enumerate(model.ports.cells)}
    coordinate_pairs = [
        ((0, 0), (n - 1, n - 1)),
        ((n - 1, 0), (0, n - 1)),
        ((0, mid), (n - 1, mid)),
        ((mid, 0), (mid, n - 1)),
    ]
    n_p = model.n_pads
    for left, right in coordinate_pairs:
        if left not in loc or right not in loc:
            continue
        i = n_p + loc[left]
        j = n_p + loc[right]
        out[i, i] += conductance
        out[j, j] += conductance
        out[i, j] -= conductance
        out[j, i] -= conductance
    return 0.5 * (out + out.T)


def perturb_laplacian_conductances(
    G: np.ndarray,
    *,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """Apply independent log-normal noise to edges while preserving row sums."""
    if sigma < 0.0:
        raise ValueError("noise sigma must be non-negative")
    matrix = np.asarray(G, dtype=float)
    n = matrix.shape[0]
    out = np.zeros_like(matrix)
    rng = np.random.default_rng(seed)
    for i in range(n):
        for j in range(i + 1, n):
            conductance = max(0.0, -float(matrix[i, j]))
            if conductance <= 0.0:
                continue
            factor = float(np.exp(rng.normal(-0.5 * sigma * sigma, sigma)))
            noisy = conductance * factor
            out[i, i] += noisy
            out[j, j] += noisy
            out[i, j] -= noisy
            out[j, i] -= noisy
    return 0.5 * (out + out.T)


def build_cases(n: int = 5, seed: int = 0) -> List[SyntheticCase]:
    """Return exact, local-mismatch, nonlocal, and combined benchmark cases."""
    ports, model = build_fixture(n)
    base_rx, base_ry, base_rz = 0.80, 1.40, 0.35
    exact = build_Gs(model, base_rx, base_ry, base_rz)
    xy = np.asarray([(c.x, c.y) for c in ports.cells], dtype=float)
    x = xy[:, 0] / max(n - 1, 1)
    y = xy[:, 1] / max(n - 1, 1)

    rx_grad = base_rx * np.exp(0.65 * (x - 0.5) + 0.25 * np.sin(2.0 * np.pi * y))
    ry_grad = base_ry * np.exp(-0.55 * (y - 0.5) + 0.20 * np.cos(2.0 * np.pi * x))
    radial = np.square(x - 0.68) + np.square(y - 0.32)
    rz_grad = base_rz * np.exp(0.75 * np.exp(-radial / 0.07) - 0.18)
    gradient = build_variable_gs(model, rx_grad, ry_grad, rz_grad)

    rng = np.random.default_rng(seed + 101)
    rx_rand = base_rx * np.exp(rng.normal(0.0, 0.42, ports.n_sinks))
    ry_rand = base_ry * np.exp(rng.normal(0.0, 0.42, ports.n_sinks))
    rz_rand = base_rz * np.exp(rng.normal(0.0, 0.35, ports.n_sinks))
    random_local = build_variable_gs(model, rx_rand, ry_rand, rz_rand)

    nonlocal_only = add_long_range_coupling(exact, model)
    combined = add_long_range_coupling(gradient, model)
    params = {"Rx": base_rx, "Ry": base_ry, "Rz": base_rz}
    return [
        SyntheticCase(
            "exact_global",
            "Truth is exactly representable by one global (Rx,Ry,Rz).",
            ports,
            model,
            exact,
            params,
        ),
        SyntheticCase(
            "smooth_local",
            "Per-cell R follows smooth gradients and a via hotspot.",
            ports,
            model,
            gradient,
            params,
        ),
        SyntheticCase(
            "random_local",
            "Per-cell R has reproducible log-normal local disorder.",
            ports,
            model,
            random_local,
            params,
        ),
        SyntheticCase(
            "long_range",
            "Uniform local R plus four nonlocal sink couplings.",
            ports,
            model,
            nonlocal_only,
            params,
        ),
        SyntheticCase(
            "combined",
            "Smooth local heterogeneity plus nonlocal sink couplings.",
            ports,
            model,
            combined,
            params,
        ),
    ]


def build_current_ensemble(
    ports: PortSet,
    *,
    seed: int,
    n_physical: int = 8,
    n_hotspot: int = 8,
    n_signed: int = 4,
) -> List[CurrentStimulus]:
    """Build independent physical, hotspot, smooth, and signed load patterns."""
    rng = np.random.default_rng(seed)
    n_s = ports.n_sinks
    total_draw = max(-float(np.sum(ports.lumped_sink_currents())), 1e-9)
    xy = np.asarray([(c.x, c.y) for c in ports.cells], dtype=float)
    patterns: List[CurrentStimulus] = []

    for index in range(n_physical):
        raw = rng.lognormal(mean=0.0, sigma=0.85, size=n_s)
        current = -total_draw * raw / np.sum(raw)
        patterns.append((f"physical_{index:02d}", current))

    span_x = max(float(np.ptp(xy[:, 0])), 1.0)
    span_y = max(float(np.ptp(xy[:, 1])), 1.0)
    for index in range(n_hotspot):
        cx = rng.uniform(float(np.min(xy[:, 0])), float(np.max(xy[:, 0])))
        cy = rng.uniform(float(np.min(xy[:, 1])), float(np.max(xy[:, 1])))
        sigma = rng.uniform(0.12, 0.30)
        radius = np.square((xy[:, 0] - cx) / span_x) + np.square(
            (xy[:, 1] - cy) / span_y
        )
        raw = 0.03 + np.exp(-0.5 * radius / (sigma * sigma))
        current = -total_draw * raw / np.sum(raw)
        patterns.append((f"hotspot_{index:02d}", current))

    xn = (xy[:, 0] - np.mean(xy[:, 0])) / span_x
    yn = (xy[:, 1] - np.mean(xy[:, 1])) / span_y
    for index, phase in enumerate((0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi)):
        raw = 1.2 + 0.55 * np.sin(2.0 * np.pi * xn + phase) + 0.35 * np.cos(
            2.0 * np.pi * yn - phase
        )
        current = -total_draw * raw / np.sum(raw)
        patterns.append((f"smooth_{index:02d}", current))

    reference_norm = max(float(np.linalg.norm(ports.lumped_sink_currents())), 1e-9)
    for index in range(n_signed):
        current = rng.normal(0.0, 1.0, n_s)
        current -= np.mean(current)
        current *= reference_norm / max(float(np.linalg.norm(current)), 1e-12)
        patterns.append((f"signed_{index:02d}", current))
    return patterns


def build_boundary_ensemble(
    ports: PortSet,
    *,
    seed: int,
) -> List[BoundaryStimulus]:
    """Vary both sink currents and individual pad voltages around nominal VDD."""
    rng = np.random.default_rng(seed)
    base_voltage = np.asarray(ports.pad_voltages, dtype=float)
    base_current = ports.lumped_sink_currents()
    pad_xy = np.asarray(ports.pad_locations(), dtype=float)
    x = pad_xy[:, 0] - np.mean(pad_xy[:, 0])
    y = pad_xy[:, 1] - np.mean(pad_xy[:, 1])
    x /= max(float(np.max(np.abs(x))), 1.0)
    y /= max(float(np.max(np.abs(y))), 1.0)
    amplitude = 0.02 * max(float(np.mean(np.abs(base_voltage))), 1.0)
    checker = np.asarray(
        [1.0 if index % 2 == 0 else -1.0 for index in range(ports.n_pads)]
    )
    patterns: List[BoundaryStimulus] = [
        ("pad_gradient_x", base_current.copy(), base_voltage + amplitude * x),
        ("pad_gradient_y", base_current.copy(), base_voltage + amplitude * y),
        ("pad_checker", base_current.copy(), base_voltage + amplitude * checker),
    ]
    random_pad = rng.normal(0.0, 1.0, ports.n_pads)
    random_pad -= np.mean(random_pad)
    random_pad /= max(float(np.max(np.abs(random_pad))), 1e-12)
    patterns.append(
        ("pad_random_original", base_current.copy(), base_voltage + amplitude * random_pad)
    )
    currents = build_current_ensemble(
        ports,
        seed=seed + 41,
        n_physical=4,
        n_hotspot=4,
        n_signed=0,
    )[:8]
    for index, (name, current) in enumerate(currents):
        delta = rng.normal(0.0, 1.0, ports.n_pads)
        delta -= np.mean(delta)
        delta /= max(float(np.max(np.abs(delta))), 1e-12)
        patterns.append(
            (
                f"pad_random_{index:02d}_{name}",
                current,
                base_voltage + amplitude * delta,
            )
        )
    return patterns


def _evaluate_boundaries(
    Gprime: np.ndarray,
    Gs: np.ndarray,
    ports: PortSet,
    stimuli: Sequence[BoundaryStimulus],
) -> Dict[str, Any]:
    n_p = ports.n_pads
    lu_m, Gsp_m = _factor_sink_block(Gprime, n_p)
    lu_s, Gsp_s = _factor_sink_block(Gs, n_p)
    rows: List[Dict[str, float | str]] = []
    all_relative: List[np.ndarray] = []

    def family_of(name: str) -> str:
        parts = name.split("_")
        if parts and parts[0].startswith("seed"):
            parts = parts[1:]
        if not parts:
            return name
        if parts[0] == "pad" and len(parts) > 1:
            return f"pad_{parts[1]}"
        return parts[0]

    for name, current, pad_voltage in stimuli:
        V_pad = np.asarray(pad_voltage, dtype=float)
        if V_pad.shape != (n_p,):
            raise ValueError(f"pad voltage shape {V_pad.shape} != ({n_p},)")
        vref = float(np.mean(V_pad))
        Vm = solve_mixed_bc(Gprime, n_p, V_pad, current, _lu=lu_m, _Gsp=Gsp_m)
        Vs = solve_mixed_bc(Gs, n_p, V_pad, current, _lu=lu_s, _Gsp=Gsp_s)
        reference = vref - Vm[n_p:]
        predicted = vref - Vs[n_p:]
        absolute_error = np.abs(predicted - reference)
        relative_floor = max(0.02 * float(np.max(np.abs(reference))), 1e-12)
        stable_relative = absolute_error / (np.abs(reference) + relative_floor)
        all_relative.append(stable_relative)
        rows.append(
            {
                "name": name,
                "family": family_of(name),
                "e_ir": float(e_ir(predicted, reference)),
                "max_abs_error": float(np.max(absolute_error)),
                "stable_rel_p95": float(np.quantile(stable_relative, 0.95)),
                "stable_rel_max": float(np.max(stable_relative)),
                "within_20pct": float(np.mean(stable_relative <= 0.20)),
            }
        )
    errors = np.asarray([float(row["e_ir"]) for row in rows], dtype=float)
    rel_concat = np.concatenate(all_relative) if all_relative else np.zeros(0)
    return {
        "n_stimuli": len(rows),
        "e_ir_mean": float(np.mean(errors)) if errors.size else float("nan"),
        "e_ir_median": float(np.median(errors)) if errors.size else float("nan"),
        "e_ir_p95": float(np.quantile(errors, 0.95)) if errors.size else float("nan"),
        "e_ir_worst": float(np.max(errors)) if errors.size else float("nan"),
        "stable_rel_p95": float(np.quantile(rel_concat, 0.95))
        if rel_concat.size
        else float("nan"),
        "stable_rel_max": float(np.max(rel_concat)) if rel_concat.size else float("nan"),
        "within_20pct": float(np.mean(rel_concat <= 0.20))
        if rel_concat.size
        else float("nan"),
        "by_stimulus": rows,
    }


def evaluate_boundary_ensemble(
    Gprime: np.ndarray,
    Gs: np.ndarray,
    ports: PortSet,
    stimuli: Sequence[BoundaryStimulus],
) -> Dict[str, Any]:
    """Evaluate simultaneous held-out pad-voltage and sink-current patterns."""
    return _evaluate_boundaries(Gprime, Gs, ports, stimuli)


def evaluate_current_ensemble(
    Gprime: np.ndarray,
    Gs: np.ndarray,
    ports: PortSet,
    stimuli: Sequence[CurrentStimulus],
) -> Dict[str, Any]:
    """Evaluate a fitted model on currents not used to construct FitContext."""
    base_voltage = np.asarray(ports.pad_voltages, dtype=float)
    boundaries = [(name, current, base_voltage) for name, current in stimuli]
    result = _evaluate_boundaries(Gprime, Gs, ports, boundaries)
    rows = result["by_stimulus"]
    family_mean: Dict[str, float] = {}
    for family in sorted({str(row["family"]) for row in rows}):
        vals = [float(row["e_ir"]) for row in rows if row["family"] == family]
        family_mean[family] = float(np.mean(vals))
    result["family_e_ir_mean"] = family_mean
    return result