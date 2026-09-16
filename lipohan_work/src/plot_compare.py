"""Comparison plots for a cost-function sweep."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


def plot_sweep(rows: Sequence[Dict[str, Any]], out_dir: str | Path) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    if not rows:
        return written

    labels = [str(r["cost"]) for r in rows]
    x = np.arange(len(labels))

    def _bar(metric: str, ylabel: str, fname: str, *, log: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        vals = [float(r[metric]) for r in rows]
        ax.bar(x, vals, color="#4c78a8")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        if log:
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        path = out / fname
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))

    _bar("e_ir_original", r"$e_{\mathrm{IR}}$ (original stimulus)", "bar_e_ir_original.png")
    _bar("J_inf", r"$J_\infty$ max relative IR error", "bar_j_inf.png")
    _bar("frac_within_20pct_original", "fraction of sinks within 20% IR", "bar_within_20pct.png")
    _bar("J_lambda", r"$J_\lambda$ eigenvalue MSE", "bar_j_lambda.png", log=True)
    _bar("J_F", r"$J_F$ Frobenius $\|G_S-G'_M\|_F^2$", "bar_j_f.png", log=True)
    _bar("J_V", r"$J_V$ IR-vector MSE", "bar_j_v.png", log=True)

    # IR % scatter with ±20% bands, one panel per cost
    n = len(rows)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.8 * nrow), squeeze=False)
    for k, row in enumerate(rows):
        ax = axes[k // ncol][k % ncol]
        ref = np.asarray(row.get("ir_pct_original_ref", []), dtype=float)
        pred = np.asarray(row.get("ir_pct_original_pred", []), dtype=float)
        if ref.size == 0 or pred.size == 0:
            ax.set_visible(False)
            continue
        ax.scatter(pred, ref, s=10, alpha=0.7)
        lo = float(min(ref.min(), pred.min()))
        hi = float(max(ref.max(), pred.max()))
        if lo == hi:
            lo, hi = lo - 1e-6, hi + 1e-6
        xs = np.array([lo, hi])
        ax.plot(xs, xs, "k--", lw=1, label="ideal")
        ax.plot(xs, 1.2 * xs, "r--", lw=1, label="+20%")
        ax.plot(xs, 0.8 * xs, "b--", lw=1, label="-20%")
        ax.set_title(
            f"{row['cost']}  e_IR={row['e_ir_original']:.3g}  "
            f"≤20%={row['frac_within_20pct_original']:.0%}"
        )
        ax.set_xlabel("Feasibility IR (%)")
        ax.set_ylabel("Kron IR (%)")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].set_visible(False)
    fig.suptitle("Original-stimulus IR %  (ideal and ±20% relative bands)")
    fig.tight_layout()
    path = out / "ir_scatter_by_cost.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    # Fitted R comparison
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.25
    Rx = [float(r["Rx"]) for r in rows]
    Ry = [float(r["Ry"]) for r in rows]
    Rz = [float(r["Rz"]) for r in rows]
    ax.bar(x - width, Rx, width, label="Rx")
    ax.bar(x, Ry, width, label="Ry")
    ax.bar(x + width, Rz, width, label="Rz")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("ohm")
    ax.set_title("Fitted Pixel-R parameters")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / "bar_r_params.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))
    return written
