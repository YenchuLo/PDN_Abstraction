"""Emit SPICE decks: original, reduced G', tri-stagger model.

Supports ngspice and Cadence Spectre (+spice) via shared spec_flow dialect.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np

_HERE = Path(__file__).resolve().parent
_SPEC = _HERE.parent.parent / "spec_flow" / "src"
for _p in (str(_SPEC), str(_HERE)):
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_SPEC))
sys.path.insert(0, str(_HERE))


def _load_spec_spice_emit():
    path = _SPEC / "spice_emit.py"
    sys.path.remove(str(_HERE))
    sys.path.insert(0, str(_SPEC))
    spec = importlib.util.spec_from_file_location("spec_flow_spice_emit", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["spec_flow_spice_emit"] = mod
    spec.loader.exec_module(mod)
    if str(_HERE) in sys.path:
        sys.path.remove(str(_HERE))
    sys.path.insert(0, str(_HERE))
    return mod


_se = _load_spec_spice_emit()

ORIGINAL_SP = _se.ORIGINAL_SP
REDUCED_SP = _se.REDUCED_SP
SPICE_DIR = _se.SPICE_DIR
PORT_MAP_JSON = _se.PORT_MAP_JSON
SIMULATOR_JSON = _se.SIMULATOR_JSON
SIMULATORS = _se.SIMULATORS
emit_original = _se.emit_original
emit_reduced_gprime = _se.emit_reduced_gprime
pad_name = _se.pad_name
sink_name = _se.sink_name
save_port_map = _se.save_port_map
save_simulator = _se.save_simulator
load_simulator = _se.load_simulator
resolve_simulator = _se.resolve_simulator
spice_dir = _se.spice_dir
_canonical_order = _se._canonical_order
_control_print = _se._control_print
_analysis_print = _se._analysis_print
_stimuli_lines = _se._stimuli_lines

from ports_dual import DualPortSet  # noqa: E402
from spice_parser import SpiceNetlist  # noqa: E402


def _mid_name(i: int) -> str:
    return f"mid_{i}"


def _resistances(
    G: Optional[Union[float, Sequence[float], np.ndarray]],
    n: int,
    fallback_R: float,
) -> np.ndarray:
    if G is None:
        return np.full(n, float(fallback_R), dtype=float)
    arr = np.asarray(G, dtype=float).ravel()
    if arr.size == 1:
        g = float(arr[0])
        return np.full(n, 1.0 / g if g > 0 else fallback_R, dtype=float)
    if arr.size != n:
        raise ValueError(f"G length {arr.size} != n_edges {n}")
    out = np.empty(n, dtype=float)
    for i, g in enumerate(arr):
        out[i] = 1.0 / float(g) if g > 0 else fallback_R
    return out


def emit_tri_stagger(
    out_dir: str | Path,
    model: "TriStaggerModel",
    *,
    Rx: Optional[float] = None,
    Ry: Optional[float] = None,
    Rz: Optional[float] = None,
    alpha_x: float = 1.0,
    alpha_y: float = 1.0,
    alpha_via: float = 1.0,
    simulator: str = "ngspice",
) -> Path:
    """Tri-stagger SPICE: pad/sink vias at port XY + sparse mid Rx/Ry."""
    from tri_stagger import TriStaggerModel, stamp_tri_stagger_conductances  # local import

    sim = resolve_simulator(simulator)
    if not isinstance(model, TriStaggerModel):
        raise TypeError("emit_tri_stagger expects TriStaggerModel")
    if Rx is not None and Ry is not None and Rz is not None:
        stamp_tri_stagger_conductances(model, float(Rx), float(Ry), float(Rz))
        alpha_x = alpha_y = alpha_via = 1.0
    if (
        model.G_ew is None
        or model.G_ns is None
        or model.G_via_pad is None
        or model.G_via_sink is None
    ):
        raise ValueError("stagger model missing G_* vectors (pass Rx,Ry,Rz)")

    ports = model.ports
    R_ew = _resistances(alpha_x * np.asarray(model.G_ew, dtype=float), len(model.ew_edges), 1e-3)
    R_ns = _resistances(alpha_y * np.asarray(model.G_ns, dtype=float), len(model.ns_edges), 1e-3)
    R_vp = _resistances(
        alpha_via * np.asarray(model.G_via_pad, dtype=float), ports.n_pads, 1e-3
    )
    R_vs = _resistances(
        alpha_via * np.asarray(model.G_via_sink, dtype=float), ports.n_sinks, 1e-3
    )

    sd = spice_dir(out_dir)
    lines = [
        "* Staggered mid-layer PDN (vias at pad/sink XY, sparse rectilinear mid)",
        f"* Rx={float(model.Rx):.12g} Ry={float(model.Ry):.12g} Rz={float(model.Rz):.12g}",
        f"* n_mid={model.n_mid} n_pads={ports.n_pads} n_sinks={ports.n_sinks}",
        f"* simulator={sim}",
    ]
    rid = 0
    for e, (a, b) in enumerate(model.ew_edges):
        rid += 1
        lines.append(
            f"Rmew_{rid} {_mid_name(int(a))} {_mid_name(int(b))} {float(R_ew[e]):.12g}"
        )
    for e, (a, b) in enumerate(model.ns_edges):
        rid += 1
        lines.append(
            f"Rmns_{rid} {_mid_name(int(a))} {_mid_name(int(b))} {float(R_ns[e]):.12g}"
        )
    for k, mi in enumerate(model.pad_mid):
        rid += 1
        lines.append(
            f"Rvia_pad_{rid} {pad_name(k)} {_mid_name(int(mi))} {float(R_vp[k]):.12g}"
        )
    for j, mi in enumerate(model.sink_mid):
        rid += 1
        lines.append(
            f"Rvia_sink_{rid} {_mid_name(int(mi))} {sink_name(j)} {float(R_vs[j]):.12g}"
        )
    lines.append("")
    lines.extend(_stimuli_lines(ports.as_port_set(), use_aliases=True))
    lines.append("")
    volt = sd / "tri_stagger.volt"
    nodes = [(a, a) for a in _canonical_order(ports.as_port_set())]
    lines.extend(_control_print(nodes, volt, simulator=sim))
    lines.append(".end")
    lines.append("")
    path = sd / "tri_stagger.sp"
    path.write_text("\n".join(lines))
    return path


def emit_all_tri_stagger(
    out_dir: str | Path,
    net: SpiceNetlist,
    ports: DualPortSet,
    node_map: Dict[str, str],
    Gprime: np.ndarray,
    model: "TriStaggerModel",
    *,
    Rx: Optional[float] = None,
    Ry: Optional[float] = None,
    Rz: Optional[float] = None,
    alpha_x: float = 1.0,
    alpha_y: float = 1.0,
    alpha_via: float = 1.0,
    simulator: str = "ngspice",
) -> Dict[str, str]:
    sim = resolve_simulator(simulator)
    ps = ports.as_port_set()
    save_port_map(out_dir, ps)
    save_simulator(out_dir, sim)
    p1 = emit_original(out_dir, net, ps, node_map, simulator=sim)
    p2 = emit_reduced_gprime(out_dir, Gprime, ps, simulator=sim)
    p3 = emit_tri_stagger(
        out_dir,
        model,
        Rx=Rx,
        Ry=Ry,
        Rz=Rz,
        alpha_x=alpha_x,
        alpha_y=alpha_y,
        alpha_via=alpha_via,
        simulator=sim,
    )
    return {
        "original": str(p1),
        "reduced_gprime": str(p2),
        "tri_stagger": str(p3),
        "simulator": sim,
    }
