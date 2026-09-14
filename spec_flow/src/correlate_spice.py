"""IR correlation from three ngspice voltage results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ports import PortSet
from spice_emit import pad_name, sink_name


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def e_ir(pred: np.ndarray, ref: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    denom = np.linalg.norm(ref)
    if denom < 1e-30:
        return float(np.linalg.norm(pred - ref))
    return float(np.linalg.norm(pred - ref) / denom)


def sink_ir_drop(voltages: Dict[str, float], n_pads: int, n_sinks: int) -> np.ndarray:
    """IR_k = mean(V_pads) - V_sink_k."""
    pads = np.array([voltages[pad_name(k)] for k in range(n_pads)], dtype=float)
    vref = float(np.mean(pads)) if n_pads else 0.0
    drop = np.array(
        [vref - voltages[sink_name(k)] for k in range(n_sinks)],
        dtype=float,
    )
    return drop / 1000.0



def pair_metrics(name: str, pred: np.ndarray, ref: np.ndarray) -> dict:
    return {
        "pair": name,
        "e_ir": e_ir(pred, ref),
        "pearson_r": _pearson(ref, pred),
    }


def correlate_three(
    voltages: Dict[str, Dict[str, float]],
    ports: PortSet,
) -> dict:
    n_p, n_s = ports.n_pads, ports.n_sinks
    drops = {
        key: sink_ir_drop(voltages[key], n_p, n_s)
        for key in ("original", "reduced_gprime", "pixel_r")
    }
    pairs = [
        pair_metrics("original_vs_pixel_r", drops["pixel_r"], drops["original"]),
        pair_metrics(
            "original_vs_reduced", drops["reduced_gprime"], drops["original"]
        ),
        pair_metrics(
            "reduced_vs_pixel_r", drops["pixel_r"], drops["reduced_gprime"]
        ),
    ]
    e_worst = max(p["e_ir"] for p in pairs)
    return {
        "ir_pairs": pairs,
        "e_worst": e_worst,
        "e_ir_original_vs_pixel_r": pairs[0]["e_ir"],
        "e_ir_original_vs_reduced": pairs[1]["e_ir"],
        "e_ir_reduced_vs_pixel_r": pairs[2]["e_ir"],
        "pearson_original_vs_pixel_r": pairs[0]["pearson_r"],
        "drops": {k: v.tolist() for k, v in drops.items()},
    }


def _vdd_ref(ports: PortSet) -> float:
    vdd = float(getattr(ports, "vdd", 0.0) or 0.0)
    if vdd > 0.0:
        return vdd
    pads = np.asarray(ports.pad_voltages, dtype=float)
    mean_pad = float(np.mean(pads)) if pads.size else 0.0
    if mean_pad <= 0.0:
        raise ValueError("cannot resolve Vdd from ports.vdd or pad_voltages")
    return mean_pad


def _ir_pct(drop: np.ndarray, vdd: float) -> np.ndarray:
    return np.asarray(drop, dtype=float) / float(vdd) * 100.0


def _scatter_volt(ref: np.ndarray, pred: np.ndarray, title: str, path: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ref, pred, s=12, alpha=0.7)
    lo = float(min(ref.min(), pred.min()))
    hi = float(max(ref.max(), pred.max()))
    if lo == hi:
        lo, hi = lo - 1e-6, hi + 1e-6
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Reference IR drop")
    ax.set_ylabel("Predicted IR drop")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _scatter_ir_pct(
    ref_pct: np.ndarray,
    pred_pct: np.ndarray,
    title: str,
    path: Path,
) -> str:
    """Correlation scatter in IR % with ideal and ±20% relative error lines.

    x = feasibility / model (pred), y = real / reference (ref).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = np.asarray(ref_pct, dtype=float)
    pred = np.asarray(pred_pct, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pred, ref, alpha=0.7, label="IR Drop (%)")

    lo = float(min(ref.min(), pred.min()))
    hi = float(max(ref.max(), pred.max()))
    if lo == hi:
        lo, hi = lo - 1e-6, hi + 1e-6
    x_vals = [lo, hi]
    ax.plot(x_vals, x_vals, "k--", label="Ideal (y = x)")
    ax.plot(x_vals, [1.2 * x for x in x_vals], "r--", label="+20% Error")
    ax.plot(x_vals, [0.8 * x for x in x_vals], "b--", label="-20% Error")

    ax.set_xlabel("Feasibility_IR_Drop (%)")
    ax.set_ylabel("Real PDN IR Drop (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _spatial_ir_side_by_side(
    ports: PortSet,
    drop_ref_pct: np.ndarray,
    drop_pred_pct: np.ndarray,
    titles: Sequence[str],
    path: Path,
) -> str:
    """Side-by-side spatial IR Drop (%) maps at physical cell (x, y)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.asarray([c.x for c in ports.cells], dtype=float)
    ys = np.asarray([c.y for c in ports.cells], dtype=float)
    ref = np.asarray(drop_ref_pct, dtype=float)
    pred = np.asarray(drop_pred_pct, dtype=float)
    if len(xs) != len(ref) or len(xs) != len(pred):
        raise ValueError(
            f"spatial plot size mismatch: cells={len(xs)} ref={len(ref)} pred={len(pred)}"
        )

    min_val = float(min(ref.min(), pred.min()))
    max_val = float(max(ref.max(), pred.max()))
    if min_val == max_val:
        min_val, max_val = min_val - 1e-6, max_val + 1e-6

    point_size = 60
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1.scatter(
        xs,
        ys,
        c=ref,
        cmap="jet",
        vmin=min_val,
        vmax=max_val,
        s=point_size,
        alpha=0.8,
        edgecolors="none",
    )
    ax1.set_title(titles[0], fontsize=14, fontweight="bold")
    ax1.set_xlabel("X Coordinate", fontsize=12)
    ax1.set_ylabel("Y Coordinate", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)

    sc2 = ax2.scatter(
        xs,
        ys,
        c=pred,
        cmap="jet",
        vmin=min_val,
        vmax=max_val,
        s=point_size,
        alpha=0.8,
        edgecolors="none",
    )
    ax2.set_title(titles[1], fontsize=14, fontweight="bold")
    ax2.set_xlabel("X Coordinate", fontsize=12)
    ax2.set_ylabel("Y Coordinate", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax1.set_aspect("equal", adjustable="box")
    ax2.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(sc2, ax=[ax1, ax2], orientation="vertical", fraction=0.03, pad=0.05)
    cbar.set_label("IR Drop (%)", rotation=270, labelpad=15, fontsize=12)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_spice_ir(
    report: dict,
    ports: PortSet,
    out_dir: str | Path,
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    drop_o = np.asarray(report["drops"]["original"], dtype=float)
    drop_r = np.asarray(report["drops"]["reduced_gprime"], dtype=float)
    drop_p = np.asarray(report["drops"]["pixel_r"], dtype=float)
    vdd = _vdd_ref(ports)
    ref_pct = _ir_pct(drop_r, vdd)
    pred_pct = _ir_pct(drop_p, vdd)
    e_red = float(report["e_ir_reduced_vs_pixel_r"])

    written.append(
        _scatter_volt(
            drop_o,
            drop_p,
            f"Original vs Pixel-R (e={report['e_ir_original_vs_pixel_r']:.4g})",
            out_dir / "ir_scatter_original_vs_pixel_r.png",
        )
    )
    written.append(
        _scatter_volt(
            drop_o,
            drop_r,
            f"Original vs Reduced (e={report['e_ir_original_vs_reduced']:.4g})",
            out_dir / "ir_scatter_original_vs_reduced.png",
        )
    )
    # Primary reduced-vs-Pixel-R pair (matches my_flow model-vs-reduced style).
    written.append(
        _scatter_ir_pct(
            ref_pct,
            pred_pct,
            f"Reduced vs Pixel-R (e={e_red:.4g}) +-20% Error Lines",
            out_dir / "ir_scatter_reduced_vs_pixel_r.png",
        )
    )
    written.append(
        _spatial_ir_side_by_side(
            ports,
            ref_pct,
            pred_pct,
            (
                "Reduced VDD IR Drop Distribution (%)",
                "Pixel-R VDD IR Drop Distribution (%)",
            ),
            out_dir / "ir_spatial_reduced_vs_pixel_r.png",
        )
    )

    err_map = np.full((ports.ny, ports.nx), np.nan)
    for k, c in enumerate(ports.cells):
        err_map[c.iy, c.ix] = abs(drop_p[k] - drop_o[k])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(err_map, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax, label="|ΔIR| Pixel-R vs original")
    ax.set_title("Sink IR-drop error map")
    ax.set_xlabel("ix")
    ax.set_ylabel("iy")
    p = out_dir / "ir_error_heatmap.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p))
    return written
