"""Emit SPICE decks: R-only original mesh, reduced G', and Pixel-R mesh.

Supports ngspice (.control / print redirects) and Cadence Spectre (+spice
dialect with .op / .print cards).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple

import numpy as np

from graph import (
    VIA_SHORT_G,
    clip_stamp_resistance,
    cross_layer_v0_pairs,
    galvanic_component_of,
)
from ports import VIA_STUB_RXRY, PortSet
from spice_parser import SpiceNetlist

SPICE_DIR = "spice"
ORIGINAL_SP = "original.sp"
REDUCED_SP = "reduced_gprime.sp"
PIXEL_SP = "pixel_r.sp"
PORT_MAP_JSON = "port_map.json"
SIMULATOR_JSON = "simulator.json"

SIMULATORS = ("ngspice", "spectre")
Simulator = Literal["ngspice", "spectre"]

G_THRESH = 1e-18


def resolve_simulator(value: Optional[str] = None) -> Simulator:
    """Resolve simulator from explicit arg, else env SPICE_SIMULATOR, else ngspice."""
    raw = (value if value is not None else os.environ.get("SPICE_SIMULATOR", "ngspice"))
    sim = str(raw).strip().lower()
    if sim not in SIMULATORS:
        raise ValueError(f"unknown simulator {raw!r}; expected one of {SIMULATORS}")
    return sim  # type: ignore[return-value]


def spice_dir(out_dir: str | Path) -> Path:
    d = Path(out_dir).resolve() / SPICE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def pad_name(k: int) -> str:
    return f"pad_{k}"


def sink_name(k: int) -> str:
    return f"sink_{k}"


def port_aliases(ports: PortSet) -> Dict[str, str]:
    """Canonical port alias -> original/remapped spice node name."""
    aliases: Dict[str, str] = {}
    for k, node in enumerate(ports.pad_nodes):
        aliases[pad_name(k)] = node
    for k, node in enumerate(ports.sink_nodes):
        aliases[sink_name(k)] = node
    return aliases


def _sink_spice_current(injection: float) -> float:
    """
    IBM-style I node 0 val draws val from node to gnd → nodal injection = -val.
    Given desired nodal injection, emit SPICE I value = -injection.
    """
    return float(-injection)


def _stimuli_lines(ports: PortSet, *, use_aliases: bool) -> List[str]:
    lines: List[str] = ["* C4 pad voltages and lumped sink currents"]
    for k, v in enumerate(ports.pad_voltages):
        node = pad_name(k) if use_aliases else ports.pad_nodes[k]
        lines.append(f"Vpad_{k} {node} 0 {float(v):.12g}")
    for k, c in enumerate(ports.cells):
        node = sink_name(k) if use_aliases else c.node
        i_sp = _sink_spice_current(c.current)
        lines.append(f"Isink_{k} {node} 0 {i_sp:.12g}")
    return lines


def _analysis_print_ngspice(
    nodes: Sequence[Tuple[str, str]], volt_path: Path
) -> List[str]:
    """ngspice .control: KLU OP, then one multi-node print to volt_path.

    A single ``print`` avoids hundreds of appends into a large SPARSE/KLU
    heap (ibmpg4 original mesh aborted mid-dump with per-node ``print >>``).
    """
    volt_path = Path(volt_path)
    args = " ".join(f"v({node})" for _, node in nodes)
    return [
        ".control",
        "option klu",
        "op",
        f"print {args} > {volt_path.as_posix()}",
        ".endc",
    ]


def _analysis_print_spectre(nodes: Sequence[Tuple[str, str]]) -> List[str]:
    """Spectre +spice cards: .op and .print (voltages recovered from +log)."""
    lines = [".op"]
    for _, node in nodes:
        lines.append(f".print v({node})")
    return lines


def _analysis_print(
    nodes: Sequence[Tuple[str, str]],
    volt_path: Path,
    *,
    simulator: str = "ngspice",
) -> List[str]:
    """
    nodes: list of (label, spice_node) to print.
    Labels recovered by pairing print order with port_map.
    """
    sim = resolve_simulator(simulator)
    if not nodes:
        raise ValueError("need at least one node to print")
    if sim == "spectre":
        return _analysis_print_spectre(nodes)
    return _analysis_print_ngspice(nodes, volt_path)


def _control_print(
    nodes: Sequence[Tuple[str, str]],
    volt_path: Path,
    *,
    simulator: str = "ngspice",
) -> List[str]:
    """Backward-compatible alias for _analysis_print."""
    return _analysis_print(nodes, volt_path, simulator=simulator)


def _is_gnd(name: str) -> bool:
    return name == "0" or name.lower() == "gnd"


def _emit_mapped_resistors(
    net: SpiceNetlist,
    node_map: Dict[str, str],
    keep_roots: Set[str],
) -> List[str]:
    """Stamp VDD-component resistors + near-ideal via shorts; skip GND mesh."""

    def root(n: str) -> str:
        if _is_gnd(n):
            return "0"
        return node_map.get(n, n)

    def keep_edge(a: str, b: str) -> bool:
        if a == b:
            return False
        if _is_gnd(a) and _is_gnd(b):
            return False
        if _is_gnd(a):
            return b in keep_roots
        if _is_gnd(b):
            return a in keep_roots
        return a in keep_roots and b in keep_roots

    lines: List[str] = [
        "* VDD R-network only (benchmark V/I discarded; GND stack omitted)",
    ]
    rid = 0
    for n1, n2, r in net.resistors:
        a, b = root(n1), root(n2)
        if not keep_edge(a, b):
            continue
        rc = clip_stamp_resistance(r)
        if rc is None:
            continue
        rid += 1
        lines.append(f"R_{rid} {a} {b} {float(rc):.12g}")

    r_via = 1.0 / VIA_SHORT_G
    lines.append("* cross-layer V=0 via shorts (near-ideal R)")
    for n1, n2 in cross_layer_v0_pairs(net):
        a, b = root(n1), root(n2)
        if not keep_edge(a, b):
            continue
        rid += 1
        lines.append(f"Rvia_{rid} {a} {b} {r_via:.12g}")
    return lines


def emit_original(
    out_dir: str | Path,
    net: SpiceNetlist,
    ports: PortSet,
    node_map: Dict[str, str],
    *,
    simulator: str = "ngspice",
) -> Path:
    """
    Emit VDD R-only PG mesh + grid-lump shorts + C4 pad V / lumped sink I.

    Does not .include the benchmark SPICE (avoids its V/I ports).
    """
    sim = resolve_simulator(simulator)
    sd = spice_dir(out_dir)
    volt = sd / "original.volt"
    aliases = port_aliases(ports)
    order = _canonical_order(ports)

    # Print canonical alias nodes, not physical original nodes.
    nodes = [(alias, alias) for alias in order]

    keep = galvanic_component_of(net, node_map, ports.port_nodes)

    alias_lines: List[str] = [
        "* canonical port alias shorts: pad_*/sink_* -> original spice nodes"
    ]
    aid = 0
    for alias in order:
        phys = aliases[alias]
        if alias == phys:
            continue
        aid += 1
        safe_alias = alias.replace("-", "_").replace(".", "_")
        alias_lines.append(f"Valias_{safe_alias}_{aid} {alias} {phys} 0")

    lump_lines: List[str] = [
        "* grid-lumping shorts (match stage02 apply_grid_lumping)",
    ]
    lid = 0
    for c in ports.cells:
        rep = c.node
        for mem in c.members:
            if mem == rep:
                continue
            lid += 1
            lump_lines.append(f"Vlump_{lid} {mem} {rep} 0")

    lines = [
        "* Multi-layer VDD PG R-mesh with C4-overlap pads and lumped sinks",
        "* Raw IBM V/I instances are not each a port; currents lumped per cell",
        f"* simulator={sim}",
        "",
        *_emit_mapped_resistors(net, node_map, keep),
        "",
        *lump_lines,
        "",
        *alias_lines,
        "",
        # Drive canonical alias nodes so original/reduced/dual use same port names.
        *_stimuli_lines(ports, use_aliases=True),
        "",
        *_analysis_print(nodes, volt, simulator=sim),
        ".end",
        "",
    ]
    path = sd / ORIGINAL_SP
    path.write_text("\n".join(lines))
    return path


def emit_reduced_gprime(
    out_dir: str | Path,
    Gprime: np.ndarray,
    ports: PortSet,
    *,
    g_thresh: float = G_THRESH,
    simulator: str = "ngspice",
) -> Path:
    """Stamp dense G' as resistors between pad_*/sink_* plus pad V / sink I."""
    sim = resolve_simulator(simulator)
    sd = spice_dir(out_dir)
    G = 0.5 * (np.asarray(Gprime, dtype=float) + np.asarray(Gprime, dtype=float).T)
    n = G.shape[0]
    if n != ports.n_ports:
        raise ValueError(f"Gprime size {n} != n_ports {ports.n_ports}")

    names = [pad_name(k) for k in range(ports.n_pads)] + [
        sink_name(k) for k in range(ports.n_sinks)
    ]
    lines = [
        "* Kron-reduced port-level equivalent of G'",
        f"* n_ports = {n}",
        f"* simulator={sim}",
    ]
    r_id = 0
    for i in range(n):
        for j in range(i + 1, n):
            g = -float(G[i, j])
            if g <= g_thresh:
                continue
            r_id += 1
            lines.append(f"Rg_{r_id} {names[i]} {names[j]} {1.0 / g:.12g}")
        # residual shunt to ground (Kron/reg may leave small row-sum)
        shunt = float(np.sum(G[i, :]))
        if shunt > g_thresh:
            r_id += 1
            lines.append(f"Rsh_{r_id} {names[i]} 0 {1.0 / shunt:.12g}")

    lines.append("")
    lines.extend(_stimuli_lines(ports, use_aliases=True))
    lines.append("")
    volt = sd / "reduced_gprime.volt"
    nodes = [(a, a) for a in _canonical_order(ports)]
    lines.extend(_analysis_print(nodes, volt, simulator=sim))
    lines.append(".end")
    lines.append("")

    path = sd / REDUCED_SP
    path.write_text("\n".join(lines))
    return path


def edge_name(k: int) -> str:
    return f"edge_{k}"


def bot_name(k: int) -> str:
    return f"bot_{k}"


def emit_pixel_r(
    out_dir: str | Path,
    ports: PortSet,
    pad_attach: Sequence[int],
    arms: Sequence[Tuple[int, int, str]],
    n_edge_nodes: int,
    Rx: float,
    Ry: float,
    Rz: float,
    *,
    simulator: str = "ngspice",
) -> Path:
    """
    Emit Spec star Pixel-R deck:

    pad_k -- [Rstubx / Rstuby] -- Rup=Rz -- sink_k -- Rdown=Rz -- bot_k
    sink_k -- Rx/Ry half-arms -- edge_* (shared or boundary)

    Stubs are omitted when ``ports.via_stub`` is ``zero`` (default).
    """
    sim = resolve_simulator(simulator)
    sd = spice_dir(out_dir)
    if Rx <= 0 or Ry <= 0 or Rz <= 0:
        raise ValueError("Rx, Ry, Rz must be positive")
    if n_edge_nodes < 0:
        raise ValueError("n_edge_nodes must be >= 0")

    stub_mode = str(getattr(ports, "via_stub", "zero"))
    lines = [
        "* Single-layer Pixel-R star (half-arm Rx/Ry, Rup/Rdown=Rz)",
        f"* Rx={Rx:.12g} Ry={Ry:.12g} Rz={Rz:.12g}",
        f"* via_stub={stub_mode}",
        f"* n_edge_nodes={int(n_edge_nodes)}",
        f"* simulator={sim}",
    ]
    r_id = 0
    for sink_k, edge_k, axis in arms:
        r_id += 1
        r_val = float(Rx) if str(axis) == "x" else float(Ry)
        prefix = "Rx" if str(axis) == "x" else "Ry"
        lines.append(
            f"{prefix}_{r_id} {sink_name(int(sink_k))} {edge_name(int(edge_k))} "
            f"{r_val:.12g}"
        )
    pitch = max(float(ports.cell_size), 1e-30)
    locs = ports.pad_locations()
    use_stub = stub_mode == VIA_STUB_RXRY
    for p, s in enumerate(pad_attach):
        s = int(s)
        via_top = pad_name(p)
        if use_stub and 0 <= p < len(locs) and 0 <= s < ports.n_sinks:
            px, py = locs[p]
            cx = float(ports.cells[s].x)
            cy = float(ports.cells[s].y)
            rx_s = float(Rx) * abs(px - cx) / pitch
            ry_s = float(Ry) * abs(py - cy) / pitch
            mid = f"via_mid_{p}"
            land = f"via_land_{p}"
            if rx_s > 1e-18:
                r_id += 1
                lines.append(f"Rstubx_{r_id} {via_top} {mid} {rx_s:.12g}")
                via_top = mid
            if ry_s > 1e-18:
                r_id += 1
                lines.append(f"Rstuby_{r_id} {via_top} {land} {ry_s:.12g}")
                via_top = land
        r_id += 1
        lines.append(
            f"Rup_{r_id} {via_top} {sink_name(s)} {float(Rz):.12g}"
        )
        r_id += 1
        lines.append(
            f"Rdown_{r_id} {sink_name(s)} {bot_name(s)} {float(Rz):.12g}"
        )

    lines.append("")
    lines.extend(_stimuli_lines(ports, use_aliases=True))
    lines.append("")
    volt = sd / "pixel_r.volt"
    nodes = [(a, a) for a in _canonical_order(ports)]
    lines.extend(_analysis_print(nodes, volt, simulator=sim))
    lines.append(".end")
    lines.append("")

    path = sd / PIXEL_SP
    path.write_text("\n".join(lines))
    return path


def _canonical_order(ports: PortSet) -> List[str]:
    return [pad_name(k) for k in range(ports.n_pads)] + [
        sink_name(k) for k in range(ports.n_sinks)
    ]


def save_port_map(out_dir: str | Path, ports: PortSet) -> Path:
    sd = spice_dir(out_dir)
    aliases = port_aliases(ports)
    order = _canonical_order(ports)
    payload = {
        "order": order,
        "alias_to_spice": aliases,
        "n_pads": ports.n_pads,
        "n_sinks": ports.n_sinks,
    }
    path = sd / PORT_MAP_JSON
    path.write_text(json.dumps(payload, indent=2))
    return path


def save_simulator(out_dir: str | Path, simulator: str = "ngspice") -> Path:
    sim = resolve_simulator(simulator)
    sd = spice_dir(out_dir)
    path = sd / SIMULATOR_JSON
    path.write_text(json.dumps({"simulator": sim}, indent=2))
    return path


def load_simulator(
    out_dir: str | Path, default: Optional[str] = None
) -> Simulator:
    """Read spice/simulator.json if present, else explicit default / env."""
    sd = Path(out_dir).resolve() / SPICE_DIR
    path = sd / SIMULATOR_JSON
    if path.is_file():
        data = json.loads(path.read_text())
        return resolve_simulator(str(data.get("simulator", "ngspice")))
    return resolve_simulator(default)


def emit_all(
    out_dir: str | Path,
    net: SpiceNetlist,
    ports: PortSet,
    node_map: Dict[str, str],
    Gprime: np.ndarray,
    pad_attach: Sequence[int],
    arms: Sequence[Tuple[int, int, str]],
    n_edge_nodes: int,
    Rx: float,
    Ry: float,
    Rz: float,
    *,
    simulator: str = "ngspice",
) -> Dict[str, str]:
    sim = resolve_simulator(simulator)
    save_port_map(out_dir, ports)
    save_simulator(out_dir, sim)
    p1 = emit_original(out_dir, net, ports, node_map, simulator=sim)
    p2 = emit_reduced_gprime(out_dir, Gprime, ports, simulator=sim)
    p3 = emit_pixel_r(
        out_dir,
        ports,
        pad_attach,
        arms,
        n_edge_nodes,
        Rx,
        Ry,
        Rz,
        simulator=sim,
    )
    return {
        "original": str(p1),
        "reduced_gprime": str(p2),
        "pixel_r": str(p3),
        "simulator": sim,
    }
