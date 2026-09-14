"""Shared OUT/ artifact paths and load/save helpers for staged flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ports import (
    VIA_STUB_ZERO,
    GridCell,
    PortSet,
    PORT_MODE_C4_GRID,
    PORT_MODE_IMAGINARY_LEGACY,
)
from spice_parser import BumpSite, NodeCoord, SpiceNetlist


def _normalize_port_mode(raw: str) -> str:
    mode = str(raw or PORT_MODE_C4_GRID)
    if mode == PORT_MODE_IMAGINARY_LEGACY:
        return PORT_MODE_C4_GRID
    return mode


NET_NPZ = "net.npz"
NET_META = "net_meta.json"
COMPONENTS_JSON = "components.json"
PORTS_JSON = "ports.json"
NODE_MAP_JSON = "node_map.json"
GPRIME_NPY = "Gprime.npy"
SYSTEM_META = "system_meta.json"
GRID_REFF_NPZ = "grid_reff.npz"
GRID_REFF_JSON = "grid_reff.json"
GRID_REFF_PNG = "grid_reff_maps.png"
GRID_REFF_DP_PNG = "grid_reff_dp.png"
PG_REFF_NPZ = "pg_reff.npz"
PG_REFF_JSON = "pg_reff.json"
PG_REFF_PNG = "pg_reff_map.png"
PIXEL_MODEL_JSON = "pixel_model.json"
PIXEL_R_JSON = "pixel_r.json"
SPECTRA_NPZ = "spectra.npz"
METRICS_JSON = "metrics.json"
SPICE_DIR = "spice"


def resolve_net_dir(out_dir: str | Path) -> Path:
    """Directory containing ``net.npz`` (``out`` or its parent for ``compk/``)."""
    out = Path(out_dir)
    if (out / NET_NPZ).is_file():
        return out
    parent = out.parent
    if (parent / NET_NPZ).is_file():
        return parent
    raise FileNotFoundError(
        f"missing artifact {NET_NPZ} under {out} or parent {parent}"
    )


def ensure_out(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def require(out_dir: str | Path, *names: str) -> Path:
    out = Path(out_dir)
    if not out.is_dir():
        raise FileNotFoundError(f"output directory not found: {out}")
    for name in names:
        path = out / name
        if not path.exists():
            raise FileNotFoundError(f"missing artifact {name} under {out}")
    return out


def save_net(out_dir: str | Path, net: SpiceNetlist, spice_path: str | Path) -> None:
    out = ensure_out(out_dir)
    r_n1 = np.array([a for a, _, _ in net.resistors], dtype=object)
    r_n2 = np.array([b for _, b, _ in net.resistors], dtype=object)
    r_val = np.array([v for _, _, v in net.resistors], dtype=float)
    i_n1 = np.array([a for a, _, _ in net.currents], dtype=object)
    i_n2 = np.array([b for _, b, _ in net.currents], dtype=object)
    i_val = np.array([v for _, _, v in net.currents], dtype=float)
    v_n1 = np.array([a for a, _, _ in net.voltages], dtype=object)
    v_n2 = np.array([b for _, b, _ in net.voltages], dtype=object)
    v_val = np.array([v for _, _, v in net.voltages], dtype=float)
    c_names = np.array(list(net.coords.keys()), dtype=object)
    c_layer = np.array([net.coords[n].layer for n in c_names], dtype=int)
    c_x = np.array([net.coords[n].x for n in c_names], dtype=float)
    c_y = np.array([net.coords[n].y for n in c_names], dtype=float)
    node_names = np.array(sorted(net.node_names), dtype=object)
    bump_net = np.array([b.net for b in net.bumps], dtype=object)
    bump_bx = np.array([b.bx for b in net.bumps], dtype=int)
    bump_by = np.array([b.by for b in net.bumps], dtype=int)
    bump_node = np.array([b.node for b in net.bumps], dtype=object)
    bump_x = np.array([b.x for b in net.bumps], dtype=float)
    bump_y = np.array([b.y for b in net.bumps], dtype=float)
    np.savez_compressed(
        out / NET_NPZ,
        r_n1=r_n1,
        r_n2=r_n2,
        r_val=r_val,
        i_n1=i_n1,
        i_n2=i_n2,
        i_val=i_val,
        v_n1=v_n1,
        v_n2=v_n2,
        v_val=v_val,
        c_names=c_names,
        c_layer=c_layer,
        c_x=c_x,
        c_y=c_y,
        node_names=node_names,
        bump_net=bump_net,
        bump_bx=bump_bx,
        bump_by=bump_by,
        bump_node=bump_node,
        bump_x=bump_x,
        bump_y=bump_y,
    )
    meta = {
        "spice_path": str(spice_path),
        "bbox": list(net.bbox),
        "n_resistors": len(net.resistors),
        "n_currents": len(net.currents),
        "n_voltages": len(net.voltages),
        "n_nodes": len(net.node_names),
        "n_coords": len(net.coords),
        "n_bumps": len(net.bumps),
        "n_bumps_vdd": sum(1 for b in net.bumps if b.net == "VDD"),
    }
    (out / NET_META).write_text(json.dumps(meta, indent=2))


def load_net(out_dir: str | Path) -> SpiceNetlist:
    out = resolve_net_dir(out_dir)
    data = np.load(out / NET_NPZ, allow_pickle=True)
    net = SpiceNetlist()
    net.resistors = [
        (str(a), str(b), float(v))
        for a, b, v in zip(data["r_n1"], data["r_n2"], data["r_val"])
    ]
    net.currents = [
        (str(a), str(b), float(v))
        for a, b, v in zip(data["i_n1"], data["i_n2"], data["i_val"])
    ]
    net.voltages = [
        (str(a), str(b), float(v))
        for a, b, v in zip(data["v_n1"], data["v_n2"], data["v_val"])
    ]
    for name, layer, x, y in zip(
        data["c_names"], data["c_layer"], data["c_x"], data["c_y"]
    ):
        net.coords[str(name)] = NodeCoord(int(layer), float(x), float(y))
    net.node_names = {str(n) for n in data["node_names"]}
    if "bump_node" in data.files:
        net.bumps = [
            BumpSite(
                net=str(bn),
                bx=int(bx),
                by=int(by),
                node=str(node),
                x=float(x),
                y=float(y),
            )
            for bn, bx, by, node, x, y in zip(
                data["bump_net"],
                data["bump_bx"],
                data["bump_by"],
                data["bump_node"],
                data["bump_x"],
                data["bump_y"],
            )
        ]
    return net


def save_components(out_dir: str | Path, payload: Dict[str, Any]) -> Path:
    out = ensure_out(out_dir)
    path = out / COMPONENTS_JSON
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_components(out_dir: str | Path) -> Dict[str, Any]:
    """Load components.json from ``out`` or its parent (shared with net.npz)."""
    out = Path(out_dir)
    for cand in (out, out.parent):
        path = cand / COMPONENTS_JSON
        if path.is_file():
            return json.loads(path.read_text())
    raise FileNotFoundError(
        f"missing artifact {COMPONENTS_JSON} under {out} or parent {out.parent}"
    )


def save_ports(out_dir: str | Path, ports: PortSet, node_map: Dict[str, str]) -> None:
    out = ensure_out(out_dir)
    payload = {
        "pad_nodes": ports.pad_nodes,
        "pad_voltages": ports.pad_voltages,
        "cell_size": ports.cell_size,
        "nx": ports.nx,
        "ny": ports.ny,
        "bbox": list(ports.bbox),
        "top_layer": ports.top_layer,
        "bot_layer": ports.bot_layer,
        "vdd": ports.vdd,
        "vdd_layers": list(ports.vdd_layers),
        "metal_pitches": {str(k): float(v) for k, v in ports.metal_pitches.items()},
        "max_metal_pitch": float(ports.max_metal_pitch),
        "port_mode": getattr(ports, "port_mode", PORT_MODE_C4_GRID),
        "package_node": getattr(ports, "package_node", ""),
        "pad_attach_tiles": list(getattr(ports, "pad_attach_tiles", [])),
        "n_occupied_bumps": int(getattr(ports, "n_occupied_bumps", 0)),
        "n_open_pads": int(getattr(ports, "n_open_pads", 0)),
        "n_shared_tiles": int(getattr(ports, "n_shared_tiles", 0)),
        "pad_xy": [
            [float(x), float(y)] for x, y in getattr(ports, "pad_xy", [])
        ],
        "via_stub": str(getattr(ports, "via_stub", VIA_STUB_ZERO)),
        "pad_attach": [int(s) for s in ports.resolved_pad_attach()],
        "n_padless": int(ports.n_padless),
        "full_pads": bool(getattr(ports, "full_pads", False)),
        "cells": [
            {
                "ix": c.ix,
                "iy": c.iy,
                "node": c.node,
                "x": c.x,
                "y": c.y,
                "current": c.current,
                "members": c.members,
            }
            for c in ports.cells
        ],
    }
    (out / PORTS_JSON).write_text(json.dumps(payload, indent=2))
    (out / NODE_MAP_JSON).write_text(json.dumps(node_map, indent=2))


def load_ports(out_dir: str | Path) -> Tuple[PortSet, Dict[str, str]]:
    out = require(out_dir, PORTS_JSON, NODE_MAP_JSON)
    payload = json.loads((out / PORTS_JSON).read_text())
    cells = [
        GridCell(
            ix=int(c["ix"]),
            iy=int(c["iy"]),
            node=str(c["node"]),
            x=float(c["x"]),
            y=float(c["y"]),
            current=float(c["current"]),
            members=[str(m) for m in c.get("members", [])],
        )
        for c in payload["cells"]
    ]
    ports = PortSet(
        pad_nodes=[str(n) for n in payload["pad_nodes"]],
        pad_voltages=[float(v) for v in payload["pad_voltages"]],
        cells=cells,
        cell_size=float(payload["cell_size"]),
        nx=int(payload["nx"]),
        ny=int(payload["ny"]),
        bbox=tuple(float(x) for x in payload["bbox"]),  # type: ignore[arg-type]
        top_layer=int(payload.get("top_layer", -1)),
        bot_layer=int(payload.get("bot_layer", -1)),
        vdd=float(payload.get("vdd", 0.9)),
        vdd_layers=[int(x) for x in payload.get("vdd_layers", [])],
        metal_pitches={
            int(k): float(v) for k, v in payload.get("metal_pitches", {}).items()
        },
        max_metal_pitch=float(payload.get("max_metal_pitch", 0.0)),
        port_mode=_normalize_port_mode(
            str(payload.get("port_mode", PORT_MODE_C4_GRID))
        ),
        package_node=str(payload.get("package_node", "")),
        pad_attach_tiles=[str(t) for t in payload.get("pad_attach_tiles", [])],
        n_occupied_bumps=int(payload.get("n_occupied_bumps", 0)),
        n_open_pads=int(payload.get("n_open_pads", 0)),
        n_shared_tiles=int(payload.get("n_shared_tiles", 0)),
        pad_xy=[
            (float(xy[0]), float(xy[1])) for xy in payload.get("pad_xy", [])
        ],
        via_stub=str(payload.get("via_stub", VIA_STUB_ZERO)),
        pad_attach=[int(s) for s in payload.get("pad_attach", [])],
        full_pads=bool(payload.get("full_pads", False)),
    )
    node_map = {str(k): str(v) for k, v in json.loads((out / NODE_MAP_JSON).read_text()).items()}
    return ports, node_map


def save_gprime(
    out_dir: str | Path,
    Gprime: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    out = ensure_out(out_dir)
    np.save(out / GPRIME_NPY, np.asarray(Gprime, dtype=float))
    (out / SYSTEM_META).write_text(json.dumps(meta, indent=2))


def load_gprime(out_dir: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    out = require(out_dir, GPRIME_NPY, SYSTEM_META)
    Gprime = np.load(out / GPRIME_NPY)
    meta = json.loads((out / SYSTEM_META).read_text())
    return Gprime, meta


def save_pixel_model(
    out_dir: str | Path,
    fields: Dict[str, Any],
) -> None:
    """Persist Pixel-R topology fields (star_half_arm schema from PixelRModel.to_fields)."""
    out = ensure_out(out_dir)
    (out / PIXEL_MODEL_JSON).write_text(json.dumps(fields, indent=2))


def load_pixel_model_fields(out_dir: str | Path) -> Dict[str, Any]:
    out = require(out_dir, PIXEL_MODEL_JSON)
    return json.loads((out / PIXEL_MODEL_JSON).read_text())


def save_fit(
    out_dir: str | Path,
    pixel_r: Dict[str, float],
    lam_M: np.ndarray | None,
    lam_S: np.ndarray | None,
    Gprime: np.ndarray,
    Gs: np.ndarray,
    fit_meta: Dict[str, Any],
) -> None:
    out = ensure_out(out_dir)
    payload = dict(pixel_r)
    payload.update(fit_meta)
    (out / PIXEL_R_JSON).write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        out / SPECTRA_NPZ,
        lam_M=np.asarray([] if lam_M is None else lam_M, dtype=float),
        lam_S=np.asarray([] if lam_S is None else lam_S, dtype=float),
        Gprime=np.asarray(Gprime, dtype=float),
        Gs=np.asarray(Gs, dtype=float),
    )


def load_fit(out_dir: str | Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    out = require(out_dir, PIXEL_R_JSON, SPECTRA_NPZ)
    meta = json.loads((out / PIXEL_R_JSON).read_text())
    spectra = dict(np.load(out / SPECTRA_NPZ))
    return meta, spectra


def save_metrics(out_dir: str | Path, metrics: Dict[str, Any]) -> None:
    out = ensure_out(out_dir)
    (out / METRICS_JSON).write_text(json.dumps(metrics, indent=2))
