#!/usr/bin/env python3
"""Grid-level Kron-branch resistance of G' at the Pixel-R lattice.

Maps the *local stencil* of the reduced port Laplacian, which is what Pixel-R
\\(R_z, 2 R_x, 2 R_y\\) correspond to — not pads-grounded two-terminal Reff
(that quantity shorts all C4 pads together and makes laterals look like vias).

  R_z[k]   = -1 / G'[pad_p, sink_k]  for pad_attach[p]=k; NaN if sink-only
  R_EW     = -1 / G'[sink_a, sink_b]   east neighbor
  R_NS     = -1 / G'[sink_a, sink_b]   north neighbor

Writes ``grid_reff.npz``, ``grid_reff.json``, and ``grid_reff_maps.png``.
Standalone:

    python spec_flow/src/grid_reff.py "$COMP"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ports import PortSet

GRID_REFF_NPZ = "grid_reff.npz"
GRID_REFF_JSON = "grid_reff.json"
GRID_REFF_PNG = "grid_reff_maps.png"
GRID_REFF_DP_PNG = "grid_reff_dp.png"
SINK_REG = 1e-9


def neighbor_pairs(
    ports: PortSet,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """East-west and south-north sink index pairs on the port lattice."""
    loc = {(c.ix, c.iy): k for k, c in enumerate(ports.cells)}
    ew: List[Tuple[int, int]] = []
    ns: List[Tuple[int, int]] = []
    for (ix, iy), k in loc.items():
        ke = loc.get((ix + 1, iy))
        if ke is not None:
            ew.append((k, ke))
        kn = loc.get((ix, iy + 1))
        if kn is not None:
            ns.append((k, kn))
    return ew, ns


def sink_impedance(Gprime: np.ndarray, n_pads: int, *, reg: float = SINK_REG) -> np.ndarray:
    """Z_ss = (G_ss + reg I)^{-1} with pads voltage-fixed (grounded for Reff)."""
    n = int(Gprime.shape[0])
    n_s = n - int(n_pads)
    if n_s <= 0:
        raise ValueError("need at least one sink for grid R_eff")
    Gss = np.asarray(Gprime[n_pads:, n_pads:], dtype=float) + np.eye(n_s) * float(reg)
    return np.linalg.inv(Gss)


def two_terminal_r(Z: np.ndarray, a: int, b: int) -> float:
    return float(Z[a, a] + Z[b, b] - Z[a, b] - Z[b, a])


def kron_branch_r(G: np.ndarray, i: int, j: int) -> float:
    """Resistance of the Kron edge i—j: 1 / (-G_ij). NaN if no coupling."""
    g = float(-0.5 * (G[i, j] + G[j, i]))
    if not np.isfinite(g) or g <= 0.0:
        return float("nan")
    return 1.0 / g


def _cell_map(ports: PortSet, values: Sequence[float]) -> np.ndarray:
    m = np.full((ports.ny, ports.nx), np.nan, dtype=float)
    for k, c in enumerate(ports.cells):
        m[c.iy, c.ix] = float(values[k])
    return m


def _edge_origin_map(
    ports: PortSet,
    pairs: Sequence[Tuple[int, int]],
    r_edge: Sequence[float],
) -> np.ndarray:
    """Place each edge R on the western / southern cell; NaN if no neighbor."""
    m = np.full((ports.ny, ports.nx), np.nan, dtype=float)
    for (a, _b), r in zip(pairs, r_edge):
        c = ports.cells[a]
        m[c.iy, c.ix] = float(r)
    return m


def _finite_stats(vals: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "min": float(np.min(a)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "max": float(np.max(a)),
    }


def compute_grid_reff(
    Gprime: np.ndarray,
    ports: PortSet,
    *,
    reg: float = SINK_REG,
) -> Dict[str, Any]:
    """Return Kron-branch maps (primary) plus pads-grounded driving-point."""
    n_p = int(ports.n_pads)
    G = np.asarray(Gprime, dtype=float)
    if G.shape != (ports.n_ports, ports.n_ports):
        raise ValueError(f"G' shape {tuple(G.shape)} != n_ports={ports.n_ports}")
    attach = ports.resolved_pad_attach()

    ew, ns = neighbor_pairs(ports)
    r_z = np.full(int(ports.n_sinks), np.nan, dtype=float)
    for p, s in enumerate(attach):
        s_i = int(s)
        if 0 <= s_i < ports.n_sinks:
            r_z[s_i] = kron_branch_r(G, p, n_p + s_i)
    r_ew = np.array([kron_branch_r(G, n_p + a, n_p + b) for a, b in ew], dtype=float)
    r_ns = np.array([kron_branch_r(G, n_p + a, n_p + b) for a, b in ns], dtype=float)

    Z = sink_impedance(G, n_p, reg=reg)
    r_dp = np.diag(Z).copy()
    r_ew_gnd = np.array([two_terminal_r(Z, a, b) for a, b in ew], dtype=float)
    r_ns_gnd = np.array([two_terminal_r(Z, a, b) for a, b in ns], dtype=float)

    return {
        "kind": "kron_branch",
        "reg": float(reg),
        "nx": int(ports.nx),
        "ny": int(ports.ny),
        "cell_size": float(ports.cell_size),
        "n_pads": n_p,
        "n_sinks": int(ports.n_sinks),
        "r_z": r_z,
        "r_ew": r_ew,
        "r_ns": r_ns,
        "r_pad_sink": r_dp,
        "r_ew_grounded": r_ew_gnd,
        "r_ns_grounded": r_ns_gnd,
        "ew_pairs": ew,
        "ns_pairs": ns,
        "r_z_map": _cell_map(ports, r_z),
        "r_ew_map": _edge_origin_map(ports, ew, r_ew),
        "r_ns_map": _edge_origin_map(ports, ns, r_ns),
        "r_pad_sink_map": _cell_map(ports, r_dp),
        "stats": {
            "R_z": _finite_stats(r_z),
            "R_ew": _finite_stats(r_ew),
            "R_ns": _finite_stats(r_ns),
            "R_pad_sink_grounded": _finite_stats(r_dp),
        },
    }


def plot_grid_reff_maps(ports: PortSet, data: Dict[str, Any], path: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#e8e8e8")
    panels = (
        (data["r_z_map"], r"$R_z$  pad–sink  (Ω)"),
        (data["r_ew_map"], r"$R_{\mathrm{EW}}$  ($\approx 2 R_x$)  (Ω)"),
        (data["r_ns_map"], r"$R_{\mathrm{NS}}$  ($\approx 2 R_y$)  (Ω)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (grid, title) in zip(axes, panels):
        masked = np.ma.masked_invalid(np.asarray(grid, dtype=float))
        im = ax.imshow(masked, origin="lower", cmap=cmap)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("ix")
        ax.set_ylabel("iy")
    nx, ny = int(data["nx"]), int(data["ny"])
    cell = float(data["cell_size"])
    fig.suptitle(
        f"Benchmark Kron-branch $R$  (from $G'$)  {nx}×{ny}  cell={cell:g}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def plot_driving_point_map(ports: PortSet, data: Dict[str, Any], path: Path) -> str:
    """One Reff per cell: pads-grounded driving-point Z_kk (all paths together)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#e8e8e8")
    grid = np.asarray(data["r_pad_sink_map"], dtype=float)
    masked = np.ma.masked_invalid(grid)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(masked, origin="lower", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$R_{\mathrm{eff}}$ (Ω)")
    nx, ny = int(data["nx"]), int(data["ny"])
    cell = float(data["cell_size"])
    ax.set_title(r"Driving-point $R_{\mathrm{eff}}=Z_{kk}$  (pads grounded)")
    ax.set_xlabel("ix")
    ax.set_ylabel("iy")
    fig.suptitle(f"Combined grid $R_{{\\mathrm{{eff}}}}$  {nx}×{ny}  cell={cell:g}")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _jsonable(
    data: Dict[str, Any],
    png: Optional[str],
    dp_png: Optional[str] = None,
) -> Dict[str, Any]:
    ew = data["ew_pairs"]
    ns = data["ns_pairs"]
    r_ew = np.asarray(data["r_ew"], dtype=float)
    r_ns = np.asarray(data["r_ns"], dtype=float)
    payload: Dict[str, Any] = {
        "kind": data["kind"],
        "note": (
            "Kron-branch R = -1/G'_ij (Pixel-R stencil). "
            "Driving-point Reff is Z_kk with pads grounded (all paths together)."
        ),
        "reg": data["reg"],
        "nx": data["nx"],
        "ny": data["ny"],
        "cell_size": data["cell_size"],
        "n_pads": data["n_pads"],
        "n_sinks": data["n_sinks"],
        "stats": data["stats"],
        "r_z": [float(x) for x in np.asarray(data["r_z"]).ravel()],
        "r_pad_sink_grounded": [
            float(x) for x in np.asarray(data["r_pad_sink"]).ravel()
        ],
        "ew_edges": [
            {"a": int(a), "b": int(b), "R": float(r)} for (a, b), r in zip(ew, r_ew)
        ],
        "ns_edges": [
            {"a": int(a), "b": int(b), "R": float(r)} for (a, b), r in zip(ns, r_ns)
        ],
    }
    if png is not None:
        payload["plot"] = png
    if dp_png is not None:
        payload["driving_point_plot"] = dp_png
    return payload


def write_grid_reff(
    out_dir: str | Path,
    Gprime: np.ndarray,
    ports: PortSet,
    *,
    reg: float = SINK_REG,
) -> Dict[str, str]:
    """Compute, persist, and plot grid Kron-branch R. Returns artifact paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = compute_grid_reff(Gprime, ports, reg=reg)
    npz_path = out / GRID_REFF_NPZ
    np.savez_compressed(
        npz_path,
        r_z=np.asarray(data["r_z"], dtype=float),
        r_ew=np.asarray(data["r_ew"], dtype=float),
        r_ns=np.asarray(data["r_ns"], dtype=float),
        r_pad_sink_grounded=np.asarray(data["r_pad_sink"], dtype=float),
        r_ew_grounded=np.asarray(data["r_ew_grounded"], dtype=float),
        r_ns_grounded=np.asarray(data["r_ns_grounded"], dtype=float),
        r_z_map=np.asarray(data["r_z_map"], dtype=float),
        r_ew_map=np.asarray(data["r_ew_map"], dtype=float),
        r_ns_map=np.asarray(data["r_ns_map"], dtype=float),
        r_pad_sink_map=np.asarray(data["r_pad_sink_map"], dtype=float),
        ew_pairs=np.asarray(data["ew_pairs"], dtype=int),
        ns_pairs=np.asarray(data["ns_pairs"], dtype=int),
    )
    png_path = out / GRID_REFF_PNG
    plot: Optional[str] = None
    try:
        plot = plot_grid_reff_maps(ports, data, png_path)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: grid R plot skipped: {exc}", file=sys.stderr)
    dp_plot: Optional[str] = None
    try:
        dp_plot = plot_driving_point_map(ports, data, out / GRID_REFF_DP_PNG)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: driving-point R plot skipped: {exc}", file=sys.stderr)
    json_path = out / GRID_REFF_JSON
    json_path.write_text(json.dumps(_jsonable(data, plot, dp_plot), indent=2))
    paths = {"npz": str(npz_path), "json": str(json_path)}
    if plot is not None:
        paths["png"] = plot
    if dp_plot is not None:
        paths["dp_png"] = dp_plot
    return paths


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Grid-level Kron-branch R maps from G' (Pixel-R stencil)"
    )
    p.add_argument("out", type=Path, help="Component OUT directory (has Gprime.npy)")
    args = p.parse_args(argv)

    from io_artifacts import load_gprime, load_ports

    ports, _ = load_ports(args.out)
    Gprime, _ = load_gprime(args.out)
    paths = write_grid_reff(args.out, Gprime, ports)
    print(
        json.dumps(
            {"stage": "grid_reff", "out": str(args.out), **paths},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
