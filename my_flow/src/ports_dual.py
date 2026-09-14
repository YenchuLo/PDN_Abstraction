"""Ports for dual-layer flow: aligned common lattice (Zhang Fig. 4).

Both sheets share the same (nx, ny) grid. Pads attach to top(ix,iy);
sinks attach to bot(ix,iy) of the same cell. pitch_bot is the fine pitch;
pitch_top must equal pitch_bot or an integer multiple (coarsening factor).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from _bootstrap import ensure_paths

ensure_paths()

from graph import _root_of, vdd_layers, vdd_metal_pitches, vdd_roots  # noqa: E402
from ports import (  # noqa: E402
    DEFAULT_VDD,
    GridCell,
    PortSet,
    _assign_pads_by_containment,
    _assign_unique_nearest,
    _cell_index,
    _fill_padless_nearest_top,
    _is_gnd,
    _real_pad_sites,
    _vdd_layer_nodes,
    grid_counts,
    representative_vdd,
)
from spice_parser import SpiceNetlist, injection_at_nodes  # noqa: E402


@dataclass
class DualPortSet:
    """Pads (top) and sinks (bottom) on an aligned common lattice."""

    pad_nodes: List[str]
    pad_voltages: List[float]
    pad_xy: List[Tuple[float, float]]
    cells: List[GridCell]  # sinks on bottom grid
    pitch_top: float
    pitch_bot: float
    nx_top: int
    ny_top: int
    nx_bot: int
    ny_bot: int
    bbox: Tuple[float, float, float, float]
    top_layer: int = -1
    bot_layer: int = -1
    vdd: float = DEFAULT_VDD
    vdd_layers: List[int] = field(default_factory=list)
    metal_pitches: Dict[int, float] = field(default_factory=dict)
    max_metal_pitch: float = 0.0
    # Dual-layer mesh node centers (layout units) — top and bot share geometry
    top_centers: List[Tuple[float, float]] = field(default_factory=list)
    bot_centers: List[Tuple[float, float]] = field(default_factory=list)
    # pad local index -> top mesh node index; sink local -> bot mesh node index
    pad_mesh: List[int] = field(default_factory=list)
    sink_mesh: List[int] = field(default_factory=list)
    # coarsening: pitch_top / pitch_bot (integer >= 1)
    coarsen_k: int = 1
    # pad/sink cell ids on the fine (bot) grid
    pad_cell_ids: List[Tuple[int, int]] = field(default_factory=list)
    sink_cell_ids: List[Tuple[int, int]] = field(default_factory=list)
    # When True, every fine-grid cell has a voltage pad.
    full_pads: bool = False

    @property
    def cell_size(self) -> float:
        """Legacy alias: bottom pitch (sink grid)."""
        return self.pitch_bot

    @property
    def nx(self) -> int:
        return self.nx_bot

    @property
    def ny(self) -> int:
        return self.ny_bot

    @property
    def sink_nodes(self) -> List[str]:
        return [c.node for c in self.cells]

    @property
    def port_nodes(self) -> List[str]:
        return list(self.pad_nodes) + self.sink_nodes

    @property
    def n_pads(self) -> int:
        return len(self.pad_nodes)

    @property
    def n_sinks(self) -> int:
        return len(self.cells)

    @property
    def n_ports(self) -> int:
        return self.n_pads + self.n_sinks

    def lumped_sink_currents(self) -> np.ndarray:
        return np.array([c.current for c in self.cells], dtype=float)

    def as_port_set(self) -> PortSet:
        """View as spec_flow PortSet (cell_size = pitch_bot)."""
        sink_of = {(int(c.ix), int(c.iy)): i for i, c in enumerate(self.cells)}
        pad_attach: List[int] = []
        for cid in self.pad_cell_ids:
            key = (int(cid[0]), int(cid[1]))
            if key in sink_of:
                pad_attach.append(sink_of[key])
        if len(pad_attach) != len(self.pad_nodes):
            pad_attach = []
        return PortSet(
            pad_nodes=list(self.pad_nodes),
            pad_voltages=list(self.pad_voltages),
            cells=list(self.cells),
            cell_size=float(self.pitch_bot),
            nx=int(self.nx_bot),
            ny=int(self.ny_bot),
            bbox=self.bbox,
            top_layer=self.top_layer,
            bot_layer=self.bot_layer,
            vdd=self.vdd,
            vdd_layers=list(self.vdd_layers),
            metal_pitches=dict(self.metal_pitches),
            max_metal_pitch=float(self.max_metal_pitch),
            pad_xy=list(self.pad_xy),
            pad_attach=pad_attach,
            full_pads=bool(self.full_pads),
        )


def _mesh_centers(
    bbox: Tuple[float, float, float, float],
    pitch: float,
) -> Tuple[int, int, List[Tuple[float, float]], List[Tuple[int, int]]]:
    xmin, ymin, _, _ = bbox
    nx, ny = grid_counts(bbox, pitch)
    centers: List[Tuple[float, float]] = []
    ids: List[Tuple[int, int]] = []
    for iy in range(ny):
        for ix in range(nx):
            centers.append((xmin + (ix + 0.5) * pitch, ymin + (iy + 0.5) * pitch))
            ids.append((ix, iy))
    return nx, ny, centers, ids


def _validate_aligned_pitches(pitch_top: float, pitch_bot: float) -> int:
    """Return integer coarsening k = pitch_top / pitch_bot (>= 1)."""
    if pitch_top <= 0 or pitch_bot <= 0:
        raise ValueError("pitch_top and pitch_bot must be > 0")
    ratio = pitch_top / pitch_bot
    k = int(round(ratio))
    if k < 1:
        raise ValueError(
            f"pitch_top ({pitch_top}) must be >= pitch_bot ({pitch_bot})"
        )
    if abs(ratio - k) > 1e-6 * max(ratio, 1.0):
        raise ValueError(
            f"pitch_top/pitch_bot = {ratio:.6g} is not an integer; "
            f"aligned lattice requires pitch_top = k * pitch_bot for integer k >= 1"
        )
    return k


def suggest_smallest_legal_pitch(min_pitch: float) -> float:
    """Deprecated helper: 1.01 × min metal (not used by default auto pitch)."""
    if min_pitch <= 0:
        raise ValueError("min_pitch must be positive to suggest smallest legal pitch")
    return float(min_pitch) * 1.01


def suggest_legal_chip_pitch(
    net: SpiceNetlist,
    node_map: Dict[str, str],
    roots: Set[str],
    *,
    grid_to_pad_ratio: float = 1.0,
) -> Tuple[float, float, float]:
    """Match ``spec_flow.ports.suggest_legal_chip_pitch``.

    C4 lattice target, then clamp so pitch > max VDD metal stripe pitch.
    Returns ``(pitch, c4_lattice_target, max_metal_pitch)``.
    """
    from ports import suggest_legal_chip_pitch as _spec_suggest

    return _spec_suggest(
        net, node_map, roots, grid_to_pad_ratio=float(grid_to_pad_ratio)
    )


def suggest_pitch_from_pads(
    net: SpiceNetlist,
    node_map: Dict[str, str],
    roots: Set[str],
    *,
    grid_to_pad_ratio: float = 1.0,
    min_pitch: float = 0.0,
    max_pitch: float = 0.0,
) -> float:
    """
    Suggest mesh pitch (same auto rule as spec_flow Pixel-R).

    C4 lattice spacing (median unique-row/col gap), scaled by
    ``1/sqrt(grid_to_pad_ratio)``, then bump so pitch > max metal stripe
    pitch (``max_pitch``). ``min_pitch`` is unused for the clamp (kept for
    call-site compat).
    """
    del min_pitch
    pitch, _target, max_metal = suggest_legal_chip_pitch(
        net, node_map, roots, grid_to_pad_ratio=grid_to_pad_ratio
    )
    if max_pitch > 0 and max_metal <= 0:
        max_metal = float(max_pitch)
    if max_metal > 0 and pitch <= max_metal:
        pitch = float(max_metal) + max(1.0, abs(float(max_metal)) * 1e-6)
    return float(pitch)


def _fine_to_coarse(ix: int, iy: int, k: int) -> Tuple[int, int]:
    return ix // k, iy // k


def build_dual_ports(
    net: SpiceNetlist,
    pitch_top: float,
    pitch_bot: float,
    node_map: Optional[Dict[str, str]] = None,
    *,
    vdd: float = DEFAULT_VDD,
    roots: Optional[Set[str]] = None,
    full_grid_sinks: bool = False,
    full_pads: bool = False,
) -> DualPortSet:
    """
    Build aligned pad/sink ports for Zhang dual-layer.

    - Common fine lattice at ``pitch_bot``; top sheet may coarsen by integer k
      where ``pitch_top = k * pitch_bot``.
    - Pads: a fine-grid cell gets a voltage source iff a real C4 overlaps it
      (pad node / XY = that C4; at most one per cell). With ``full_pads``,
      padless cells get the unique nearest unused top-metal node and **all**
      pads share one common drive (representative of the real C4 voltages).
    - Sinks: currents lumped by fine pitch → unique nearest bottom-layer node
      on **every** fine-grid cell when ``full_grid_sinks`` (spec default);
      otherwise only nonzero-current cells
    - Pad of cell (ix,iy) → top mesh node; sink of (ix,iy) → bot mesh node
      (same cell indices on the fine grid; top index = coarse of fine when k>1)
    """
    pitch_top = float(pitch_top)
    pitch_bot = float(pitch_bot)
    k = _validate_aligned_pitches(pitch_top, pitch_bot)

    if node_map is None:
        node_map = {name: name for name in net.node_names}
    if roots is None:
        roots = vdd_roots(net, node_map)
    else:
        roots = set(roots)
    if not roots:
        raise ValueError("VDD component roots set is empty")

    layers = vdd_layers(net, node_map, roots)
    pitches = vdd_metal_pitches(net, node_map, roots)
    max_pitch = max(pitches.values()) if pitches else 0.0
    min_pitch = min(pitches.values()) if pitches else 0.0
    # Require pitch above the finest metal stripe pitch so each cell spans wires.
    # Default auto pitch matches the C4 lattice, then clamps > max metal.
    for label, p in (("pitch_top", pitch_top), ("pitch_bot", pitch_bot)):
        if min_pitch > 0 and p <= min_pitch:
            raise ValueError(
                f"{label} ({p}) must be > min VDD metal stripe pitch ({min_pitch})"
            )

    bbox = net.bbox
    top_layer, bot_layer, by_layer = _vdd_layer_nodes(net, node_map, roots, layers)
    nx_b, ny_b, bot_centers, bot_ids = _mesh_centers(bbox, pitch_bot)
    nx_t, ny_t, top_centers, top_ids = _mesh_centers(bbox, pitch_top)

    # Sanity: with integer k, fine and coarse grids must cover the same bbox.
    if nx_t * k < nx_b or ny_t * k < ny_b:
        # allow slight ceil mismatch; top must still cover fine cells
        pass

    top_id_to_idx = {cid: i for i, cid in enumerate(top_ids)}
    bot_id_to_idx = {cid: i for i, cid in enumerate(bot_ids)}
    bot_id_to_center = {cid: bot_centers[i] for i, cid in enumerate(bot_ids)}

    # --- pads: real C4 overlap with the fine grid (not every top-metal node) ---
    pad_sites = _real_pad_sites(net, node_map, roots)
    if not pad_sites:
        raise ValueError("no nonzero V→gnd pads found on this VDD component")
    c4_v = {name: volt for name, _x, _y, volt in pad_sites}
    c4_cands = [(name, x, y) for name, x, y, _v in pad_sites]
    pad_slots = _assign_pads_by_containment(
        c4_cands, bot_ids, bot_centers, bbox, pitch_bot, nx_b, ny_b
    )
    n_occ = sum(1 for p in pad_slots if p is not None)
    if n_occ == 0:
        raise ValueError(
            "no fine-grid cell overlaps a real C4 pad "
            f"(pitch_bot={pitch_bot}, n_c4={len(c4_cands)})"
        )
    used = {p[0] for p in pad_slots if p is not None}
    if full_pads:
        _fill_padless_nearest_top(
            pad_slots, bot_centers, by_layer[top_layer], used
        )
    pad_cell_ids: List[Tuple[int, int]] = []
    pad_nodes: List[str] = []
    pad_xy: List[Tuple[float, float]] = []
    pad_voltages: List[float] = []
    pad_mesh: List[int] = []
    pad_v_common = (
        representative_vdd(list(c4_v.values()), fallback=float(vdd))
        if full_pads
        else None
    )
    for cid, slot in zip(bot_ids, pad_slots):
        if slot is None:
            continue
        node, px, py = slot
        pad_cell_ids.append(cid)
        pad_nodes.append(node)
        pad_xy.append((float(px), float(py)))
        if pad_v_common is not None:
            pad_voltages.append(float(pad_v_common))
        else:
            pad_voltages.append(float(c4_v.get(node, vdd)))
        cx, cy = _fine_to_coarse(cid[0], cid[1], k)
        cx = min(max(cx, 0), nx_t - 1)
        cy = min(max(cy, 0), ny_t - 1)
        pad_mesh.append(top_id_to_idx[(cx, cy)])
    used = set(pad_nodes)

    # --- sinks: original I locations, lumped onto fine pitch cells ---
    inj = injection_at_nodes(net)
    current_buckets: Dict[Tuple[int, int], float] = {}
    # Weighted real injection XY per cell (for stub attach at real locations).
    inj_xy_sum: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    for node, cur in inj.items():
        if cur >= 0.0 or abs(cur) < 1e-30:
            continue
        if _root_of(node_map, node) not in roots:
            continue
        coord = net.coords.get(node)
        if coord is None:
            continue
        ix, iy = _cell_index(coord.x, coord.y, bbox, pitch_bot, nx_b, ny_b)
        cid = (ix, iy)
        current_buckets[cid] = current_buckets.get(cid, 0.0) + cur
        w = abs(float(cur))
        sx0, sy0, sw = inj_xy_sum.get(cid, (0.0, 0.0, 0.0))
        inj_xy_sum[cid] = (sx0 + w * float(coord.x), sy0 + w * float(coord.y), sw + w)

    # Real current sinks only (pad-forced zero-I sinks removed; stubs attach pads).
    # ``full_grid_sinks`` (uniform-current mode): every fine-grid cell is a sink.
    if full_grid_sinks:
        sink_cell_ids = list(bot_ids)
    else:
        sink_cell_ids = sorted(current_buckets.keys(), key=lambda t: (t[1], t[0]))
        if not sink_cell_ids:
            # Pad-only component: keep pad cells as zero-current sinks for topology.
            sink_cell_ids = list(pad_cell_ids) if pad_cell_ids else list(bot_ids)

    # Nearest bot node to *real* injection XY (or cell center if empty / full grid).
    sink_target_xy: List[Tuple[float, float]] = []
    for cid in sink_cell_ids:
        sx0, sy0, sw = inj_xy_sum.get(cid, (0.0, 0.0, 0.0))
        if sw > 0:
            sink_target_xy.append((sx0 / sw, sy0 / sw))
        else:
            sink_target_xy.append(bot_id_to_center[cid])
    sink_chosen = _assign_unique_nearest(by_layer[bot_layer], sink_target_xy, used)

    member_buckets: Dict[Tuple[int, int], List[str]] = {}
    for node, x, y in by_layer[bot_layer]:
        ix, iy = _cell_index(x, y, bbox, pitch_bot, nx_b, ny_b)
        member_buckets.setdefault((ix, iy), []).append(node)

    protected = set(pad_nodes) | {s[0] for s in sink_chosen}
    cells_out: List[GridCell] = []
    sink_mesh: List[int] = []
    for cid, (sink_node, _sx, _sy), (rx, ry) in zip(
        sink_cell_ids, sink_chosen, sink_target_xy
    ):
        ix, iy = cid
        members = [sink_node]
        for mem in member_buckets.get(cid, []):
            if mem == sink_node or mem not in protected:
                if mem not in members:
                    members.append(mem)
        cells_out.append(
            GridCell(
                ix=ix,
                iy=iy,
                node=sink_node,
                x=float(rx),
                y=float(ry),
                current=float(current_buckets.get(cid, 0.0)),
                members=members,
            )
        )
        sink_mesh.append(bot_id_to_idx[cid])

    return DualPortSet(
        pad_nodes=pad_nodes,
        pad_voltages=pad_voltages,
        pad_xy=pad_xy,
        cells=cells_out,
        pitch_top=pitch_top,
        pitch_bot=pitch_bot,
        nx_top=nx_t,
        ny_top=ny_t,
        nx_bot=nx_b,
        ny_bot=ny_b,
        bbox=bbox,
        top_layer=top_layer,
        bot_layer=bot_layer,
        vdd=representative_vdd(pad_voltages, fallback=float(vdd)),
        vdd_layers=list(layers),
        metal_pitches={int(k_): float(v) for k_, v in pitches.items()},
        max_metal_pitch=float(max_pitch),
        top_centers=top_centers,
        bot_centers=bot_centers,
        pad_mesh=pad_mesh,
        sink_mesh=sink_mesh,
        coarsen_k=k,
        pad_cell_ids=list(pad_cell_ids),
        sink_cell_ids=list(sink_cell_ids),
        full_pads=bool(full_pads),
    )


def nonzero_sink_indices(
    ports: DualPortSet, *, tol: float = 1e-30
) -> list[int]:
    """Indices of sink cells with |I| > tol (real current injections)."""
    return [i for i, c in enumerate(ports.cells) if abs(float(c.current)) > tol]


def can_tri_stagger_ports(ports: DualPortSet, *, tol: float = 1e-30) -> bool:
    """True if this dual port set has at least one nonzero-current sink."""
    return bool(nonzero_sink_indices(ports, tol=tol))


def tri_stagger_ports_from_dual(ports: DualPortSet) -> DualPortSet:
    """Return dual ports for stagger/square models (full sink grid preserved).

    Spec alignment: every chip-grid cell keeps a sink port (zero-``I`` allowed)
    so Kron ``G'`` and the abstract model share ``n_sinks = nx * ny``.

    Raises ``ValueError`` when every sink cell has zero current (pad-only
    island). Callers that process multi-component nets should skip those
    components — e.g. ibmpg3 ``comp1``.
    """
    if not can_tri_stagger_ports(ports):
        raise ValueError(
            "no nonzero current sinks for stagger ports "
            f"(n_pads={ports.n_pads}, n_sink_cells={len(ports.cells)}; "
            "pad-only / zero-load component — skip for stagger)"
        )
    # Identity copy: keep full lattice including zero-I sinks.
    return DualPortSet(
        pad_nodes=list(ports.pad_nodes),
        pad_voltages=list(ports.pad_voltages),
        pad_xy=list(ports.pad_xy),
        cells=list(ports.cells),
        pitch_top=ports.pitch_top,
        pitch_bot=ports.pitch_bot,
        nx_top=ports.nx_top,
        ny_top=ports.ny_top,
        nx_bot=ports.nx_bot,
        ny_bot=ports.ny_bot,
        bbox=ports.bbox,
        top_layer=ports.top_layer,
        bot_layer=ports.bot_layer,
        vdd=ports.vdd,
        vdd_layers=list(ports.vdd_layers),
        metal_pitches=dict(ports.metal_pitches),
        max_metal_pitch=float(ports.max_metal_pitch),
        top_centers=list(ports.top_centers),
        bot_centers=list(ports.bot_centers),
        pad_mesh=list(ports.pad_mesh),
        sink_mesh=list(ports.sink_mesh),
        coarsen_k=int(ports.coarsen_k),
        pad_cell_ids=list(ports.pad_cell_ids),
        sink_cell_ids=list(ports.sink_cell_ids),
        full_pads=bool(getattr(ports, "full_pads", False)),
    )


def remap_dual_ports(ports: DualPortSet, node_map: Dict[str, str]) -> DualPortSet:
    def m(name: str) -> str:
        return node_map.get(name, name)

    new_pads = [m(n) for n in ports.pad_nodes]
    new_cells = [
        GridCell(
            ix=c.ix,
            iy=c.iy,
            node=m(c.node),
            x=c.x,
            y=c.y,
            current=c.current,
            members=[m(x) for x in c.members] if c.members else [m(c.node)],
        )
        for c in ports.cells
    ]
    all_ports = new_pads + [c.node for c in new_cells]
    if len(all_ports) != len(set(all_ports)):
        raise ValueError("port remap produced duplicate nodes")
    if any(_is_gnd(n) for n in all_ports):
        raise ValueError("port remap mapped a port to ground")

    return DualPortSet(
        pad_nodes=new_pads,
        pad_voltages=list(ports.pad_voltages),
        pad_xy=list(ports.pad_xy),
        cells=new_cells,
        pitch_top=ports.pitch_top,
        pitch_bot=ports.pitch_bot,
        nx_top=ports.nx_top,
        ny_top=ports.ny_top,
        nx_bot=ports.nx_bot,
        ny_bot=ports.ny_bot,
        bbox=ports.bbox,
        top_layer=ports.top_layer,
        bot_layer=ports.bot_layer,
        vdd=ports.vdd,
        vdd_layers=list(ports.vdd_layers),
        metal_pitches=dict(ports.metal_pitches),
        max_metal_pitch=float(ports.max_metal_pitch),
        top_centers=list(ports.top_centers),
        bot_centers=list(ports.bot_centers),
        pad_mesh=list(ports.pad_mesh),
        sink_mesh=list(ports.sink_mesh),
        coarsen_k=int(ports.coarsen_k),
        pad_cell_ids=list(ports.pad_cell_ids),
        sink_cell_ids=list(ports.sink_cell_ids),
        full_pads=bool(getattr(ports, "full_pads", False)),
    )


def apply_dual_grid_lumping(
    node_map: Dict[str, str], ports: DualPortSet
) -> Dict[str, str]:
    """Ideal-short bottom-layer members in each sink cell to the sink rep."""
    out = dict(node_map)
    for c in ports.cells:
        rep = c.node
        for mem in c.members:
            if mem == rep:
                continue
            out[mem] = rep
    return out
