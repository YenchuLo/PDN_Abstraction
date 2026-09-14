#!/usr/bin/env python3
"""Stage 08: e_IR of Kron G' vs localized G_S (+ R maps / error heatmap)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from _bootstrap import ensure_paths

ensure_paths()

from correlate import _factor_sink_block, e_ir, solve_mixed_bc  # noqa: E402
from correlate_spice import (  # noqa: E402
    _ir_pct,
    _scatter_ir_pct,
    _spatial_ir_side_by_side,
    _vdd_ref,
)
from grid_reff import sink_impedance  # noqa: E402
from io_artifacts import load_gprime, load_ports, save_metrics  # noqa: E402
from ports import PortSet  # noqa: E402

R_JSON = "localized_r.json"
SPECTRA_NPZ = "localized_spectra.npz"


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _cell_map(ports: PortSet, values: Sequence[float]) -> np.ndarray:
    m = np.full((ports.ny, ports.nx), np.nan)
    for k, c in enumerate(ports.cells):
        m[c.iy, c.ix] = float(values[k])
    return m


def _plot_r_maps(
    ports: PortSet,
    Rx: np.ndarray,
    Ry: np.ndarray,
    Rz: np.ndarray,
    path: Path,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Rz_plot = np.asarray(Rz, dtype=float).copy()
    has = ports.sink_has_pad()
    for k, ok in enumerate(has):
        if not ok:
            Rz_plot[k] = np.nan
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#e8e8e8")
    for ax, vals, title in zip(
        axes,
        (Rx, Ry, Rz_plot),
        ("Rx (half-arm)", "Ry (half-arm)", "Rz (NaN if sink-only)"),
    ):
        im = ax.imshow(_cell_map(ports, vals), origin="lower", cmap=cmap)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("ix")
        ax.set_ylabel("iy")
    fig.suptitle("Localized Pixel-R parameters")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _plot_reff_single(
    ports: PortSet,
    r_dp: np.ndarray,
    path: Path,
    title: str,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(_cell_map(ports, r_dp), origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$R_{\mathrm{eff}}$ (Ω)")
    ax.set_title(title)
    ax.set_xlabel("ix")
    ax.set_ylabel("iy")
    fig.suptitle(r"Driving-point $R_{\mathrm{eff}}=Z_{kk}$  (Rx, Ry, Rz together)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _plot_reff_side_by_side(
    ports: PortSet,
    r_ref: np.ndarray,
    r_pred: np.ndarray,
    path: Path,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = np.asarray(r_ref, dtype=float)
    pred = np.asarray(r_pred, dtype=float)
    vmin = float(min(np.nanmin(ref), np.nanmin(pred)))
    vmax = float(max(np.nanmax(ref), np.nanmax(pred)))
    if vmin == vmax:
        vmin, vmax = vmin - 1e-6, vmax + 1e-6
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, vals, title in zip(
        axes,
        (ref, pred),
        (
            r"Reduced $G'$  $R_{\mathrm{eff}}$ (Ω)",
            r"Localized Pixel-R  $R_{\mathrm{eff}}$ (Ω)",
        ),
    ):
        im = ax.imshow(
            _cell_map(ports, vals), origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("ix")
        ax.set_ylabel("iy")
    fig.suptitle(r"Sink driving-point $R_{\mathrm{eff}}=Z_{kk}$  (pads grounded)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _plot_error_heatmap(
    ports: PortSet,
    drop_m: np.ndarray,
    drop_s: np.ndarray,
    path: Path,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    err_map = np.full((ports.ny, ports.nx), np.nan)
    for k, c in enumerate(ports.cells):
        err_map[c.iy, c.ix] = abs(float(drop_s[k]) - float(drop_m[k]))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(err_map, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax, label="|ΔIR| localized vs reduced")
    ax.set_title("Sink IR-drop error map")
    ax.set_xlabel("ix")
    ax.set_ylabel("iy")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Correlate localized Pixel-R IR vs G'")
    p.add_argument("out", type=Path, help="Component OUT directory")
    p.add_argument("seed", type=int, nargs="?", default=0)
    args = p.parse_args(argv)

    ports, _ = load_ports(args.out)
    Gprime, _ = load_gprime(args.out)
    fit = json.loads((args.out / R_JSON).read_text())
    spectra = dict(np.load(args.out / SPECTRA_NPZ))
    Gs = np.asarray(spectra["Gs"], dtype=float)
    Gs_uni = np.asarray(spectra["Gs_uniform"], dtype=float)
    Rx_cell = np.asarray(spectra["Rx_cell"], dtype=float)
    Ry_cell = np.asarray(spectra["Ry_cell"], dtype=float)
    Rz_cell = np.asarray(spectra["Rz_cell"], dtype=float)

    n_p = ports.n_pads
    V_pad = np.asarray(ports.pad_voltages, dtype=float)
    I = ports.lumped_sink_currents()
    lu_m, Gsp_m = _factor_sink_block(Gprime, n_p)
    lu_s, Gsp_s = _factor_sink_block(Gs, n_p)
    lu_u, Gsp_u = _factor_sink_block(Gs_uni, n_p)
    Vm = solve_mixed_bc(Gprime, n_p, V_pad, I, _lu=lu_m, _Gsp=Gsp_m)
    Vs = solve_mixed_bc(Gs, n_p, V_pad, I, _lu=lu_s, _Gsp=Gsp_s)
    Vu = solve_mixed_bc(Gs_uni, n_p, V_pad, I, _lu=lu_u, _Gsp=Gsp_u)
    vref = float(np.mean(V_pad))
    drop_m = vref - Vm[n_p:]
    drop_s = vref - Vs[n_p:]
    drop_u = vref - Vu[n_p:]
    e_loc = e_ir(drop_s, drop_m)
    e_uni = e_ir(drop_u, drop_m)
    pearson = _pearson(drop_m, drop_s)

    metrics = {
        "cell_size": ports.cell_size,
        "nx": ports.nx,
        "ny": ports.ny,
        "bbox": list(ports.bbox),
        "block_size": fit.get("block_size"),
        "n_regions": fit.get("n_regions"),
        "n_pads": int(ports.n_pads),
        "n_sinks": int(ports.n_sinks),
        "n_padless": int(ports.n_padless),
        "max_metal_pitch": float(ports.max_metal_pitch),
        "Rx_uniform": fit.get("Rx_uniform"),
        "Ry_uniform": fit.get("Ry_uniform"),
        "Rz_uniform": fit.get("Rz_uniform"),
        "residual": fit.get("residual"),
        "relative_ir_error": fit.get("relative_ir_error"),
        "residual_uniform": fit.get("residual_uniform"),
        "relative_ir_error_uniform": fit.get("relative_ir_error_uniform"),
        "fit_method": fit.get("fit_method"),
        "fit_success": fit.get("fit_success"),
        "n_feng_iters": fit.get("n_feng_iters"),
        "polished": fit.get("polished"),
        "seed": args.seed,
        "ir_source": "gprime_vs_localized",
        "e_ir_reduced_vs_localized": e_loc,
        "e_ir_reduced_vs_uniform": e_uni,
        "pearson_reduced_vs_localized": pearson,
        "e_worst": e_loc,
        "e_ir_by_stimulus": fit.get("e_ir_by_stimulus"),
        "e_ir_by_stimulus_uniform": fit.get("e_ir_by_stimulus_uniform"),
    }

    plots: list[str] = []
    try:
        vdd = _vdd_ref(ports)
        ref_pct = _ir_pct(drop_m, vdd)
        pred_pct = _ir_pct(drop_s, vdd)
        plots.append(
            _scatter_ir_pct(
                ref_pct,
                pred_pct,
                f"Reduced vs Localized (e={e_loc:.4g}) +-20% Error Lines",
                args.out / "ir_scatter_reduced_vs_localized.png",
            )
        )
        plots.append(
            _spatial_ir_side_by_side(
                ports,
                ref_pct,
                pred_pct,
                (
                    "Reduced VDD IR Drop Distribution (%)",
                    "Localized Pixel-R VDD IR Drop Distribution (%)",
                ),
                args.out / "ir_spatial_reduced_vs_localized.png",
            )
        )
        plots.append(
            _plot_error_heatmap(
                ports, drop_m, drop_s, args.out / "ir_error_heatmap.png"
            )
        )
        plots.append(
            _plot_r_maps(
                ports, Rx_cell, Ry_cell, Rz_cell, args.out / "localized_r_maps.png"
            )
        )
        Z_m = sink_impedance(Gprime, n_p)
        Z_s = sink_impedance(Gs, n_p)
        r_dp_m = np.diag(Z_m).copy()
        r_dp_s = np.diag(Z_s).copy()
        (args.out / "localized_reff.json").write_text(
            json.dumps(
                {
                    "kind": "driving_point_Zkk",
                    "note": "Pads-grounded driving-point Reff; Rx, Ry, Rz together.",
                    "nx": int(ports.nx),
                    "ny": int(ports.ny),
                    "cell_size": float(ports.cell_size),
                    "r_eff_reduced": [float(x) for x in r_dp_m],
                    "r_eff_localized": [float(x) for x in r_dp_s],
                    "stats": {
                        "reduced": {
                            "min": float(np.min(r_dp_m)),
                            "median": float(np.median(r_dp_m)),
                            "mean": float(np.mean(r_dp_m)),
                            "max": float(np.max(r_dp_m)),
                        },
                        "localized": {
                            "min": float(np.min(r_dp_s)),
                            "median": float(np.median(r_dp_s)),
                            "mean": float(np.mean(r_dp_s)),
                            "max": float(np.max(r_dp_s)),
                        },
                    },
                },
                indent=2,
            )
        )
        plots.append(
            _plot_reff_single(
                ports,
                r_dp_s,
                args.out / "localized_reff_map.png",
                r"Localized Pixel-R  $R_{\mathrm{eff}}$ (Ω)",
            )
        )
        plots.append(
            _plot_reff_side_by_side(
                ports,
                r_dp_m,
                r_dp_s,
                args.out / "reff_spatial_reduced_vs_localized.png",
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"warning: plots skipped: {exc}", file=sys.stderr)
    metrics["plots"] = plots
    save_metrics(args.out, metrics)
    print(
        json.dumps(
            {
                "stage": "08_correlate",
                "out": str(args.out),
                "e_ir_reduced_vs_localized": e_loc,
                "e_ir_reduced_vs_uniform": e_uni,
                "pearson_reduced_vs_localized": pearson,
                "plots": plots,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
