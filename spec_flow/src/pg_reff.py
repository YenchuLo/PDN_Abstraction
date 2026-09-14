#!/usr/bin/env python3
"""Per-grid driving-point R_eff of the unreduced PG mesh.

Grounds every nonzero-V C4/package node from the SPICE deck (all C4s, not
only the one pad kept per overlapping cell). Injects 1 A at each grid sink
representative on the assembled G_M.  R_eff[k] = V_sink[k].

Writes ``pg_reff.npz``, ``pg_reff.json``, and ``pg_reff_map.png``.

    python spec_flow/src/pg_reff.py "$COMP"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from graph import ConductanceSystem, _is_gnd
from ports import PortSet

PG_REFF_NPZ = "pg_reff.npz"
PG_REFF_JSON = "pg_reff.json"
PG_REFF_PNG = "pg_reff_map.png"
REG = 1e-12


def real_vdd_source_nodes(
    voltages: Sequence[tuple],
    node_map: Dict[str, str],
) -> List[str]:
    """Live terminals of nonzero V sources, after short-merge."""
    seen = set()
    out: List[str] = []
    for n1, n2, v in voltages:
        if abs(float(v)) < 1e-30:
            continue
        g1, g2 = _is_gnd(str(n1)), _is_gnd(str(n2))
        if g1 == g2:
            continue
        live = str(n2) if g1 else str(n1)
        root = node_map.get(live, live)
        if _is_gnd(root) or root in seen:
            continue
        seen.add(root)
        out.append(root)
    return out


def _cell_map(ports: PortSet, values: Sequence[float]) -> np.ndarray:
    m = np.full((ports.ny, ports.nx), np.nan, dtype=float)
    for k, c in enumerate(ports.cells):
        m[c.iy, c.ix] = float(values[k])
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


def compute_mesh_reff(
    system: ConductanceSystem,
    probe_nodes: Sequence[str],
    ground_nodes: Sequence[str],
    *,
    reg: float = REG,
    rhs_batch: Optional[int] = None,
) -> np.ndarray:
    """Driving-point R at each probe with ``ground_nodes`` voltage-fixed at 0."""
    G = system.G.tocsc()
    n = G.shape[0]
    gset = {system.index[name] for name in ground_nodes if name in system.index}
    if not gset:
        raise ValueError("no real VDD source nodes present in G")
    free = np.array([i for i in range(n) if i not in gset], dtype=int)
    if free.size == 0:
        raise ValueError("all nodes are grounded; no free mesh")
    Gff = G[free[:, None], free]
    if reg > 0:
        Gff = Gff + sparse.eye(free.size, format="csc") * float(reg)
    lu = splu(Gff.tocsc())
    pos = {int(i): k for k, i in enumerate(free)}

    n_p = len(probe_nodes)
    slots: List[Optional[int]] = []
    for name in probe_nodes:
        if name not in system.index:
            slots.append(None)
            continue
        gi = system.index[name]
        slots.append(None if gi in gset else pos[gi])

    if rhs_batch is None:
        target_bytes = 2 * 1024**3
        rhs_batch = max(1, min(n_p, int(target_bytes / max(8 * free.size, 1))))
    rhs_batch = max(1, int(rhs_batch))

    r = np.full(n_p, np.nan, dtype=float)
    for start in range(0, n_p, rhs_batch):
        end = min(start + rhs_batch, n_p)
        rhs = np.zeros((free.size, end - start), dtype=float)
        for j, sl in enumerate(slots[start:end]):
            if sl is not None:
                rhs[sl, j] = 1.0
        vf = lu.solve(rhs)
        for j, sl in enumerate(slots[start:end]):
            if sl is not None:
                r[start + j] = float(vf[sl, j])
    return r


def compute_pg_reff(
    system: ConductanceSystem,
    ports: PortSet,
    voltages: Sequence[tuple],
    *,
    reg: float = REG,
) -> Dict[str, Any]:
    grounds = real_vdd_source_nodes(voltages, system.node_map)
    probes = [c.node for c in ports.cells]
    r = compute_mesh_reff(system, probes, grounds, reg=reg)
    return {
        "kind": "real_pg_c4_grounded",
        "reg": float(reg),
        "nx": int(ports.nx),
        "ny": int(ports.ny),
        "cell_size": float(ports.cell_size),
        "n_sinks": int(ports.n_sinks),
        "n_c4": int(len(grounds)),
        "n_mesh": int(len(system.nodes)),
        "c4_nodes": grounds,
        "r_eff": r,
        "r_eff_map": _cell_map(ports, r),
        "stats": _finite_stats(r),
    }


def plot_pg_reff_map(data: Dict[str, Any], path: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#e8e8e8")
    masked = np.ma.masked_invalid(np.asarray(data["r_eff_map"], dtype=float))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(masked, origin="lower", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$R_{\mathrm{eff}}$ (Ω)")
    nx, ny = int(data["nx"]), int(data["ny"])
    cell = float(data["cell_size"])
    n_c4 = int(data["n_c4"])
    ax.set_title(r"Real PG  $R_{\mathrm{eff}}=V_{\mathrm{sink}}/1\,\mathrm{A}$")
    ax.set_xlabel("ix")
    ax.set_ylabel("iy")
    fig.suptitle(
        f"Unreduced mesh, {n_c4} C4 pads grounded  {nx}×{ny}  cell={cell:g}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def write_pg_reff(
    out_dir: str | Path,
    system: ConductanceSystem,
    ports: PortSet,
    voltages: Sequence[tuple],
    *,
    reg: float = REG,
) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = compute_pg_reff(system, ports, voltages, reg=reg)
    npz_path = out / PG_REFF_NPZ
    np.savez_compressed(
        npz_path,
        r_eff=np.asarray(data["r_eff"], dtype=float),
        r_eff_map=np.asarray(data["r_eff_map"], dtype=float),
    )
    png: Optional[str] = None
    try:
        png = plot_pg_reff_map(data, out / PG_REFF_PNG)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: real-PG R_eff plot skipped: {exc}", file=sys.stderr)
    payload = {
        "kind": data["kind"],
        "note": (
            "Driving-point of the unreduced SPICE mesh. "
            "Real nonzero-V C4 nodes grounded; 1 A at each grid sink."
        ),
        "reg": data["reg"],
        "nx": data["nx"],
        "ny": data["ny"],
        "cell_size": data["cell_size"],
        "n_sinks": data["n_sinks"],
        "n_c4": data["n_c4"],
        "n_mesh": data["n_mesh"],
        "stats": data["stats"],
        "r_eff": [float(x) for x in np.asarray(data["r_eff"]).ravel()],
    }
    if png is not None:
        payload["plot"] = png
    json_path = out / PG_REFF_JSON
    json_path.write_text(json.dumps(payload, indent=2))
    paths = {"npz": str(npz_path), "json": str(json_path)}
    if png is not None:
        paths["png"] = png
    return paths


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Per-grid R_eff from the unreduced PG (real C4 pads grounded)"
    )
    p.add_argument("out", type=Path, help="Component OUT directory")
    args = p.parse_args(argv)

    from graph import assemble_conductance, galvanic_component_of
    from io_artifacts import load_net, load_ports

    ports, node_map = load_ports(args.out)
    net = load_net(args.out)
    keep = galvanic_component_of(net, node_map, ports.port_nodes)
    system = assemble_conductance(net, node_map, keep_roots=keep)
    paths = write_pg_reff(args.out, system, ports, net.voltages)
    print(
        json.dumps(
            {"stage": "pg_reff", "out": str(args.out), **paths},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
