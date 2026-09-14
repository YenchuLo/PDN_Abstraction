"""OUT/ artifact helpers for my_flow dual-layer (Zhang) flow."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_SPEC = _HERE.parent.parent / "spec_flow" / "src"


def _load_spec_io():
    path = _SPEC / "io_artifacts.py"
    if str(_SPEC) not in sys.path:
        sys.path.insert(0, str(_SPEC))
    spec = importlib.util.spec_from_file_location("spec_flow_io_artifacts", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sio = _load_spec_io()

COMPONENTS_JSON = _sio.COMPONENTS_JSON
GPRIME_NPY = _sio.GPRIME_NPY
METRICS_JSON = _sio.METRICS_JSON
NET_META = _sio.NET_META
NET_NPZ = _sio.NET_NPZ
NODE_MAP_JSON = _sio.NODE_MAP_JSON
PORTS_JSON = _sio.PORTS_JSON
SPECTRA_NPZ = _sio.SPECTRA_NPZ
SPICE_DIR = _sio.SPICE_DIR
SYSTEM_META = _sio.SYSTEM_META
ensure_out = _sio.ensure_out
load_components = _sio.load_components
load_gprime = _sio.load_gprime
load_net = _sio.load_net
require = _sio.require
resolve_net_dir = _sio.resolve_net_dir
save_components = _sio.save_components
save_gprime = _sio.save_gprime
save_metrics = _sio.save_metrics
save_net = _sio.save_net

TRI_STAGGER_MODEL_JSON = "tri_stagger_model.json"
TRI_STAGGER_R_JSON = "tri_stagger_r.json"
TRI_STAGGER_PORTS_JSON = "tri_stagger_ports.json"
TRI_SQUARE_MODEL_JSON = "tri_square_model.json"
TRI_SQUARE_R_JSON = "tri_square_r.json"
TRI_SQUARE_PORTS_JSON = "tri_square_ports.json"
VOLTSPOT_MODEL_JSON = "voltspot_model.json"
VOLTSPOT_R_JSON = "voltspot_r.json"
VOLTSPOT_PORTS_JSON = "voltspot_ports.json"

from ports import GridCell  # noqa: E402
from ports_dual import DualPortSet  # noqa: E402


def save_ports(out_dir: str | Path, ports: DualPortSet, node_map: Dict[str, str]) -> None:
    out = ensure_out(out_dir)
    payload = {
        "kind": "dual_ports",
        "pad_nodes": ports.pad_nodes,
        "pad_voltages": ports.pad_voltages,
        "pad_xy": [[float(x), float(y)] for x, y in ports.pad_xy],
        "pitch_top": ports.pitch_top,
        "pitch_bot": ports.pitch_bot,
        "nx_top": ports.nx_top,
        "ny_top": ports.ny_top,
        "nx_bot": ports.nx_bot,
        "ny_bot": ports.ny_bot,
        "cell_size": ports.pitch_bot,  # legacy
        "nx": ports.nx_bot,
        "ny": ports.ny_bot,
        "bbox": list(ports.bbox),
        "top_layer": ports.top_layer,
        "bot_layer": ports.bot_layer,
        "vdd": ports.vdd,
        "vdd_layers": list(ports.vdd_layers),
        "metal_pitches": {str(k): float(v) for k, v in ports.metal_pitches.items()},
        "max_metal_pitch": float(ports.max_metal_pitch),
        "top_centers": [[float(x), float(y)] for x, y in ports.top_centers],
        "bot_centers": [[float(x), float(y)] for x, y in ports.bot_centers],
        "pad_mesh": [int(x) for x in ports.pad_mesh],
        "sink_mesh": [int(x) for x in ports.sink_mesh],
        "coarsen_k": int(ports.coarsen_k),
        "pad_cell_ids": [[int(a), int(b)] for a, b in ports.pad_cell_ids],
        "sink_cell_ids": [[int(a), int(b)] for a, b in ports.sink_cell_ids],
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


def load_ports(out_dir: str | Path) -> Tuple[DualPortSet, Dict[str, str]]:
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
    pad_xy = [(float(x), float(y)) for x, y in payload.get("pad_xy", [])]
    if not pad_xy:
        pad_xy = [(0.0, 0.0)] * len(payload["pad_nodes"])
    top_centers = [
        (float(x), float(y)) for x, y in payload.get("top_centers", [])
    ]
    bot_centers = [
        (float(x), float(y)) for x, y in payload.get("bot_centers", [])
    ]
    ports = DualPortSet(
        pad_nodes=[str(n) for n in payload["pad_nodes"]],
        pad_voltages=[float(v) for v in payload["pad_voltages"]],
        pad_xy=pad_xy,
        cells=cells,
        pitch_top=float(payload.get("pitch_top", payload.get("cell_size", 1.0))),
        pitch_bot=float(payload.get("pitch_bot", payload.get("cell_size", 1.0))),
        nx_top=int(payload.get("nx_top", payload.get("nx", 1))),
        ny_top=int(payload.get("ny_top", payload.get("ny", 1))),
        nx_bot=int(payload.get("nx_bot", payload.get("nx", 1))),
        ny_bot=int(payload.get("ny_bot", payload.get("ny", 1))),
        bbox=tuple(float(x) for x in payload["bbox"]),  # type: ignore[arg-type]
        top_layer=int(payload.get("top_layer", -1)),
        bot_layer=int(payload.get("bot_layer", -1)),
        vdd=float(payload.get("vdd", 0.9)),
        vdd_layers=[int(x) for x in payload.get("vdd_layers", [])],
        metal_pitches={
            int(k): float(v) for k, v in payload.get("metal_pitches", {}).items()
        },
        max_metal_pitch=float(payload.get("max_metal_pitch", 0.0)),
        top_centers=top_centers,
        bot_centers=bot_centers,
        pad_mesh=[int(x) for x in payload.get("pad_mesh", [])],
        sink_mesh=[int(x) for x in payload.get("sink_mesh", [])],
        coarsen_k=int(payload.get("coarsen_k", 1)),
        pad_cell_ids=[
            (int(a), int(b)) for a, b in payload.get("pad_cell_ids", [])
        ],
        sink_cell_ids=[
            (int(a), int(b)) for a, b in payload.get("sink_cell_ids", [])
        ],
        full_pads=bool(payload.get("full_pads", False)),
    )
    node_map = {
        str(k): str(v) for k, v in json.loads((out / NODE_MAP_JSON).read_text()).items()
    }
    return ports, node_map


def save_tri_stagger_model(out_dir: str | Path, fields: Dict[str, Any]) -> None:
    out = ensure_out(out_dir)
    (out / TRI_STAGGER_MODEL_JSON).write_text(json.dumps(fields, indent=2))


def load_tri_stagger_model_fields(out_dir: str | Path) -> Dict[str, Any]:
    out = Path(out_dir)
    for name in (TRI_STAGGER_MODEL_JSON, "stagger_model.json"):
        path = out / name
        if path.is_file():
            return json.loads(path.read_text())
    require(out_dir, TRI_STAGGER_MODEL_JSON)
    return {}


def save_tri_stagger_ports(
    out_dir: str | Path, ports: DualPortSet, node_map: Dict[str, str]
) -> None:
    """Write stagger port set without overwriting dual ports.json."""
    out = ensure_out(out_dir)
    payload = {
        "kind": "tri_stagger_ports",
        "pad_nodes": ports.pad_nodes,
        "pad_voltages": ports.pad_voltages,
        "pad_xy": [[float(x), float(y)] for x, y in ports.pad_xy],
        "pitch_top": ports.pitch_top,
        "pitch_bot": ports.pitch_bot,
        "nx_top": ports.nx_top,
        "ny_top": ports.ny_top,
        "nx_bot": ports.nx_bot,
        "ny_bot": ports.ny_bot,
        "bbox": list(ports.bbox),
        "top_layer": ports.top_layer,
        "bot_layer": ports.bot_layer,
        "vdd": ports.vdd,
        "vdd_layers": list(ports.vdd_layers),
        "pad_mesh": [int(x) for x in ports.pad_mesh],
        "sink_mesh": [int(x) for x in ports.sink_mesh],
        "coarsen_k": int(ports.coarsen_k),
        "pad_cell_ids": [[int(a), int(b)] for a, b in ports.pad_cell_ids],
        "sink_cell_ids": [[int(a), int(b)] for a, b in ports.sink_cell_ids],
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
        "n_pads": ports.n_pads,
        "n_sinks": ports.n_sinks,
    }
    (out / TRI_STAGGER_PORTS_JSON).write_text(json.dumps(payload, indent=2))
    # node_map already present from dual flow; keep a copy for clarity
    (out / "tri_stagger_node_map.json").write_text(json.dumps(node_map, indent=2))


def save_tri_stagger_fit(
    out_dir: str | Path,
    tri_stagger_r: Dict[str, Any],
    Gprime: np.ndarray,
    Gs: np.ndarray,
    fit_meta: Dict[str, Any],
) -> None:
    out = ensure_out(out_dir)
    payload = dict(tri_stagger_r)
    payload.update(fit_meta)
    (out / TRI_STAGGER_R_JSON).write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        out / "tri_stagger_spectra.npz",
        Gprime=np.asarray(Gprime, dtype=float),
        Gs=np.asarray(Gs, dtype=float),
    )


def save_tri_square_model(out_dir: str | Path, fields: Dict[str, Any]) -> None:
    out = ensure_out(out_dir)
    (out / TRI_SQUARE_MODEL_JSON).write_text(json.dumps(fields, indent=2))


def load_tri_square_model_fields(out_dir: str | Path) -> Dict[str, Any]:
    out = require(out_dir, TRI_SQUARE_MODEL_JSON)
    return json.loads((out / TRI_SQUARE_MODEL_JSON).read_text())


def save_tri_square_ports(
    out_dir: str | Path, ports: DualPortSet, node_map: Dict[str, str]
) -> None:
    """Write tri-square port set without overwriting dual ports.json."""
    out = ensure_out(out_dir)
    payload = {
        "kind": "tri_square_ports",
        "pad_nodes": ports.pad_nodes,
        "pad_voltages": ports.pad_voltages,
        "pad_xy": [[float(x), float(y)] for x, y in ports.pad_xy],
        "pitch_top": ports.pitch_top,
        "pitch_bot": ports.pitch_bot,
        "nx_top": ports.nx_top,
        "ny_top": ports.ny_top,
        "nx_bot": ports.nx_bot,
        "ny_bot": ports.ny_bot,
        "bbox": list(ports.bbox),
        "top_layer": ports.top_layer,
        "bot_layer": ports.bot_layer,
        "vdd": ports.vdd,
        "vdd_layers": list(ports.vdd_layers),
        "pad_mesh": [int(x) for x in ports.pad_mesh],
        "sink_mesh": [int(x) for x in ports.sink_mesh],
        "coarsen_k": int(ports.coarsen_k),
        "pad_cell_ids": [[int(a), int(b)] for a, b in ports.pad_cell_ids],
        "sink_cell_ids": [[int(a), int(b)] for a, b in ports.sink_cell_ids],
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
        "n_pads": ports.n_pads,
        "n_sinks": ports.n_sinks,
    }
    (out / TRI_SQUARE_PORTS_JSON).write_text(json.dumps(payload, indent=2))
    (out / "tri_square_node_map.json").write_text(json.dumps(node_map, indent=2))


def save_tri_square_fit(
    out_dir: str | Path,
    tri_square_r: Dict[str, Any],
    Gprime: np.ndarray,
    Gs: np.ndarray,
    fit_meta: Dict[str, Any],
) -> None:
    out = ensure_out(out_dir)
    payload = dict(tri_square_r)
    payload.update(fit_meta)
    (out / TRI_SQUARE_R_JSON).write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        out / "tri_square_spectra.npz",
        Gprime=np.asarray(Gprime, dtype=float),
        Gs=np.asarray(Gs, dtype=float),
    )


def save_voltspot_model(out_dir: str | Path, fields: Dict[str, Any]) -> None:
    out = ensure_out(out_dir)
    (out / VOLTSPOT_MODEL_JSON).write_text(json.dumps(fields, indent=2))


def load_voltspot_model_fields(out_dir: str | Path) -> Dict[str, Any]:
    out = require(out_dir, VOLTSPOT_MODEL_JSON)
    return json.loads((out / VOLTSPOT_MODEL_JSON).read_text())


def save_voltspot_ports(
    out_dir: str | Path, ports: DualPortSet, node_map: Dict[str, str]
) -> None:
    """Write VoltSpot port set without overwriting dual ports.json."""
    out = ensure_out(out_dir)
    payload = {
        "kind": "voltspot_ports",
        "pad_nodes": ports.pad_nodes,
        "pad_voltages": ports.pad_voltages,
        "pad_xy": [[float(x), float(y)] for x, y in ports.pad_xy],
        "pitch_top": ports.pitch_top,
        "pitch_bot": ports.pitch_bot,
        "nx_top": ports.nx_top,
        "ny_top": ports.ny_top,
        "nx_bot": ports.nx_bot,
        "ny_bot": ports.ny_bot,
        "bbox": list(ports.bbox),
        "top_layer": ports.top_layer,
        "bot_layer": ports.bot_layer,
        "vdd": ports.vdd,
        "vdd_layers": list(ports.vdd_layers),
        "pad_mesh": [int(x) for x in ports.pad_mesh],
        "sink_mesh": [int(x) for x in ports.sink_mesh],
        "coarsen_k": int(ports.coarsen_k),
        "pad_cell_ids": [[int(a), int(b)] for a, b in ports.pad_cell_ids],
        "sink_cell_ids": [[int(a), int(b)] for a, b in ports.sink_cell_ids],
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
        "n_pads": ports.n_pads,
        "n_sinks": ports.n_sinks,
        "grid_to_pad_ratio_target": 4.0,
    }
    (out / VOLTSPOT_PORTS_JSON).write_text(json.dumps(payload, indent=2))
    (out / "voltspot_node_map.json").write_text(json.dumps(node_map, indent=2))


def save_voltspot_fit(
    out_dir: str | Path,
    voltspot_r: Dict[str, Any],
    Gprime: np.ndarray,
    Gs: np.ndarray,
    fit_meta: Dict[str, Any],
) -> None:
    out = ensure_out(out_dir)
    payload = dict(voltspot_r)
    payload.update(fit_meta)
    (out / VOLTSPOT_R_JSON).write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        out / "voltspot_spectra.npz",
        Gprime=np.asarray(Gprime, dtype=float),
        Gs=np.asarray(Gs, dtype=float),
    )

