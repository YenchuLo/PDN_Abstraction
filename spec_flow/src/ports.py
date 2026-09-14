"""Port definition: C4-overlap voltage pads + per-grid lumped sinks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

# ``"conserve"`` = equal share of total sink current; float = per-sink draw (A).
UniformCurrentSpec = Union[str, float]

from graph import R_STAMP_MIN, _root_of, vdd_layers, vdd_metal_pitches, vdd_roots
from spice_parser import LAYER_TILE, LAYER_VDD, NodeCoord, SpiceNetlist, injection_at_nodes
from tsmc_region_grid import (
    DEFAULT_LATTICE,
    RegionLattice,
    bump_index_base,
    collect_region_vdd_ports,
    collect_vdd_bumps,
    parse_region_indices,
    vpad_name,
)

DEFAULT_VDD = 0.9
# Synthetic top layer for TSMC bump pads / tiles (sinks stay on LAYER_VDD).
LAYER_VIRTUAL_PAD = LAYER_TILE
# Near-short from each occupied bump pad into its mapped tile node.
# Floor matches the G-stamp window so pad–tile R does not explode G'.
VPAD_PACKAGE_R = R_STAMP_MIN
TSMC_PKG_NODE = "__tsmc_vdd_pkg__"
# IBM: real C4 pads on a chip-pitch lattice (sinks at cell centers).
PORT_MODE_C4_GRID = "c4_overlap_grid"
# Backward-compat alias for older artifacts / imports.
PORT_MODE_IMAGINARY = PORT_MODE_C4_GRID
PORT_MODE_IMAGINARY_LEGACY = "imaginary_grid"
PORT_MODE_TSMC_VIRTUAL = "tsmc_virtual_pixel_r"
# Pad→center via stub: short (default) or length-proportional Rx/Ry.
VIA_STUB_ZERO = "zero"
VIA_STUB_RXRY = "rxry"
VIA_STUB_MODES = (VIA_STUB_ZERO, VIA_STUB_RXRY)


@dataclass
class GridCell:
    ix: int
    iy: int
    # representative node in the multi-layer net (after short-merge mapping applied later)
    node: str
    x: float  # geometric cell-center X (Pixel-R lattice)
    y: float  # geometric cell-center Y (Pixel-R lattice)
    current: float  # total nodal injection (into circuit) for this cell
    members: List[str] = field(default_factory=list)  # bottom-layer nodes shorted to rep


@dataclass
class PortSet:
    """Ordered ports: pads first, then grid sink cells (row-major iy, ix)."""

    pad_nodes: List[str]
    pad_voltages: List[float]
    cells: List[GridCell]
    cell_size: float  # chip pitch / cell side length
    nx: int
    ny: int
    bbox: Tuple[float, float, float, float]
    top_layer: int = -1
    bot_layer: int = -1
    vdd: float = DEFAULT_VDD
    vdd_layers: List[int] = field(default_factory=list)
    metal_pitches: Dict[int, float] = field(default_factory=dict)
    max_metal_pitch: float = 0.0
    port_mode: str = PORT_MODE_C4_GRID
    package_node: str = ""
    # Per-pad tile attach for TSMC bump grid ("" = open-circuit / empty bump).
    pad_attach_tiles: List[str] = field(default_factory=list)
    n_occupied_bumps: int = 0
    n_open_pads: int = 0
    n_shared_tiles: int = 0
    # Real C4 XY (overlap site). Empty → treat as cell centers.
    pad_xy: List[Tuple[float, float]] = field(default_factory=list)
    # ``zero``: pad--Rz--sink as if co-located. ``rxry``: plus Rx/Ry L-bend stubs.
    via_stub: str = VIA_STUB_ZERO
    # Pad k attaches to sink index (Pixel-R Rz). Empty + n_pads==n_sinks → 1:1.
    # A cell has a voltage source iff a real C4 overlaps it; otherwise sink-only
    # (unless ``full_pads`` filled every cell from nearest top-metal).
    pad_attach: List[int] = field(default_factory=list)
    # When True, every grid cell has a voltage pad (C4 containment + nearest
    # unused top-metal fill for formerly padless cells).
    full_pads: bool = False

    @property
    def n(self) -> float:
        """Alias for cell_size (chip pitch)."""
        return self.cell_size

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
        """Length n_sinks: nodal injections into the circuit at sink ports."""
        return np.array([c.current for c in self.cells], dtype=float)

    def resolved_pad_attach(self) -> List[int]:
        """Sink index for each pad. Legacy ports with empty list are 1:1."""
        if self.pad_attach:
            att = [int(s) for s in self.pad_attach]
            if len(att) != self.n_pads:
                raise ValueError(
                    f"pad_attach length {len(att)} != n_pads {self.n_pads}"
                )
            return att
        if self.n_pads == self.n_sinks:
            return list(range(self.n_sinks))
        raise ValueError(
            f"pad_attach required when n_pads ({self.n_pads}) != n_sinks "
            f"({self.n_sinks})"
        )

    @property
    def n_padless(self) -> int:
        """Sink cells with no voltage pad (current only)."""
        occupied = {int(s) for s in self.resolved_pad_attach()}
        return int(self.n_sinks - len(occupied))

    def sink_has_pad(self) -> List[bool]:
        """Length n_sinks: True if a voltage pad attaches to that sink."""
        flags = [False] * self.n_sinks
        for s in self.resolved_pad_attach():
            if 0 <= int(s) < self.n_sinks:
                flags[int(s)] = True
        return flags

    def pad_locations(self) -> List[Tuple[float, float]]:
        """Real pad XY, or attached cell-center fallback when ``pad_xy`` is missing."""
        if len(self.pad_xy) == self.n_pads:
            return [(float(x), float(y)) for x, y in self.pad_xy]
        attach = self.resolved_pad_attach()
        return [
            (float(self.cells[s].x), float(self.cells[s].y)) for s in attach
        ]


def parse_uniform_current_arg(raw: Optional[str]) -> Optional[UniformCurrentSpec]:
    """
    Parse ``--uniform-current`` CLI / Tcl value.

    Returns ``None`` (off), ``\"conserve\"``, or a float per-sink draw in amps.
    Empty / ``off`` / ``0`` / ``false`` → off. Bare flag / ``conserve`` / ``on``
    → conserve total. A number → fixed per-sink draw (positive = sink draw).
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("", "0", "false", "no", "off", "none"):
        return None
    if s in ("conserve", "1", "true", "yes", "on"):
        return "conserve"
    try:
        return float(s)
    except ValueError as exc:
        raise ValueError(
            f"invalid --uniform-current {raw!r}; use 'conserve' or a number (amps)"
        ) from exc


def parse_via_stub_arg(raw: Optional[str]) -> str:
    """Parse ``--via-stub``: ``zero`` (default short) or ``rxry`` (∝ Rx, Ry)."""
    if raw is None:
        return VIA_STUB_ZERO
    s = str(raw).strip().lower()
    if s in ("", VIA_STUB_ZERO, "0", "off", "false", "no", "none", "short"):
        return VIA_STUB_ZERO
    if s in (VIA_STUB_RXRY, "nonzero", "non-zero", "on", "true", "yes", "1", "r"):
        return VIA_STUB_RXRY
    raise ValueError(f"via_stub must be 'zero' or 'rxry' (got {raw!r})")


def via_series_r(
    ports: PortSet,
    Rx: float,
    Ry: float,
    Rz: float,
    pad_k: int,
    sink_k: int,
) -> float:
    """Pad→sink series R: ``Rz`` plus optional L-bend stubs ``Rx|dx|/n + Ry|dy|/n``."""
    r = float(Rz)
    if str(getattr(ports, "via_stub", VIA_STUB_ZERO)) != VIA_STUB_RXRY:
        return max(r, 1e-30)
    pitch = max(float(ports.cell_size), 1e-30)
    locs = ports.pad_locations()
    if 0 <= pad_k < len(locs) and 0 <= sink_k < ports.n_sinks:
        px, py = locs[pad_k]
        cx = float(ports.cells[sink_k].x)
        cy = float(ports.cells[sink_k].y)
        r += float(Rx) * abs(px - cx) / pitch
        r += float(Ry) * abs(py - cy) / pitch
    return max(r, 1e-30)


def apply_uniform_sink_currents(
    cells: Sequence[GridCell],
    spec: UniformCurrentSpec,
) -> Dict[str, object]:
    """
    Set every sink cell to the same nodal injection (mutates ``cells``).

    - ``\"conserve\"``: ``I_each = sum(I) / N`` (preserves total and sign)
    - float ``I0 >= 0``: each sink draws ``I0`` A → stored as ``-|I0|``
    - float ``I0 < 0``: used as-is (nodal injection into the circuit)
    """
    if not cells:
        raise ValueError("no sink cells for uniform current")
    n = len(cells)
    total_before = float(sum(float(c.current) for c in cells))
    if isinstance(spec, str):
        if spec != "conserve":
            raise ValueError(f"unknown uniform-current mode: {spec!r}")
        i_each = total_before / float(n)
        mode = "conserve"
    else:
        i0 = float(spec)
        i_each = -abs(i0) if i0 >= 0.0 else i0
        mode = "fixed"
    for c in cells:
        c.current = float(i_each)
    return {
        "uniform_current_mode": mode,
        "i_each": float(i_each),
        "i_total_before": total_before,
        "i_total_after": float(i_each) * n,
        "n_sinks": n,
    }


def uniform_current_out_tag(spec: Optional[UniformCurrentSpec]) -> str:
    """Suffix fragment for flow OUT dirs (leading ``_`` when enabled)."""
    if spec is None:
        return ""
    if spec == "conserve":
        return "_uI"
    # Compact float tag, e.g. 1e-3 → uI1e-3
    return f"_uI{float(spec):.6g}".replace("+", "")


def _is_gnd(name: str) -> bool:
    return name == "0" or name.lower() == "gnd"


def representative_vdd(
    pad_voltages: Sequence[float],
    *,
    fallback: float = DEFAULT_VDD,
) -> float:
    """Common pad voltage when all agree; else median; else ``fallback``."""
    vals = [float(v) for v in pad_voltages]
    if not vals:
        return float(fallback)
    uniq = {round(v, 12) for v in vals}
    if len(uniq) == 1:
        return float(vals[0])
    arr = sorted(vals)
    mid = len(arr) // 2
    if len(arr) % 2 == 1:
        return float(arr[mid])
    return float(0.5 * (arr[mid - 1] + arr[mid]))


def grid_counts(
    bbox: Tuple[float, float, float, float],
    cell_size: float,
) -> Tuple[int, int]:
    """
    Cells along each axis from chip bbox and cell side length (chip pitch).

    Example: chiplet 1000×1000, cell_size=10 → 100×100 grid.
    """
    xmin, ymin, xmax, ymax = bbox
    if cell_size <= 0:
        raise ValueError("chip pitch / cell side length must be > 0")
    nx = max(1, int(math.ceil((xmax - xmin) / cell_size)))
    ny = max(1, int(math.ceil((ymax - ymin) / cell_size)))
    return nx, ny


def _cell_index(
    x: float,
    y: float,
    bbox: Tuple[float, float, float, float],
    cell_size: float,
    nx: int,
    ny: int,
) -> Tuple[int, int]:
    xmin, ymin, _, _ = bbox
    ix = int((x - xmin) / cell_size) if cell_size > 0 else 0
    iy = int((y - ymin) / cell_size) if cell_size > 0 else 0
    ix = min(max(ix, 0), nx - 1)
    iy = min(max(iy, 0), ny - 1)
    return ix, iy


def _root(node_map: Dict[str, str], name: str) -> str:
    if _is_gnd(name):
        return "0"
    return node_map.get(name, name)


def _real_pad_sites(
    net: SpiceNetlist,
    node_map: Dict[str, str],
    roots: Set[str],
) -> List[Tuple[str, float, float, float]]:
    """Real V→gnd pads on this component: (node_root, x, y, voltage)."""
    pads: List[Tuple[str, float, float, float]] = []
    seen: Set[str] = set()
    for n1, n2, v in net.voltages:
        if abs(v) < 1e-12:
            continue
        a_g = _is_gnd(n1)
        b_g = _is_gnd(n2)
        if a_g == b_g:
            continue
        live = n2 if a_g else n1
        root = _root(node_map, live)
        if root not in roots or root in seen:
            continue
        coord = net.coords.get(live) or net.coords.get(root)
        if coord is None:
            continue
        seen.add(root)
        pads.append((root, float(coord.x), float(coord.y), float(v)))
    pads.sort(key=lambda t: (t[2], t[1]))
    return pads


def _unique_sorted_coords(
    values: Sequence[float], *, merge_tol: float = 1.0
) -> List[float]:
    """Sort coordinates, merging values within ``merge_tol`` (layout units)."""
    vs = sorted(float(v) for v in values)
    if not vs:
        return []
    tol = max(float(merge_tol), 0.0)
    out = [vs[0]]
    for v in vs[1:]:
        if v - out[-1] > tol:
            out.append(v)
    return out


def _median_positive(vals: Sequence[float]) -> Optional[float]:
    xs = [float(v) for v in vals if v > 0.0]
    if not xs:
        return None
    xs.sort()
    mid = len(xs) // 2
    if len(xs) % 2 == 1:
        return float(xs[mid])
    return float(0.5 * (xs[mid - 1] + xs[mid]))


def axis_lattice_pitch(
    coords: Sequence[float], *, merge_tol: float = 1.0
) -> Optional[float]:
    """Median gap between adjacent unique coordinates on one axis."""
    uniq = _unique_sorted_coords(coords, merge_tol=merge_tol)
    if len(uniq) < 2:
        return None
    diffs = [uniq[i + 1] - uniq[i] for i in range(len(uniq) - 1)]
    return _median_positive(diffs)


def c4_lattice_pitch(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    merge_tol: float = 1.0,
) -> float:
    """
    C4 array pitch: the coarser of the two axis lattice pitches.

    Each axis uses the median adjacent gap of unique row/column coordinates
    (not pairwise average distance, and not ``sqrt(n_C4)`` packing).
    """
    px = axis_lattice_pitch(xs, merge_tol=merge_tol)
    py = axis_lattice_pitch(ys, merge_tol=merge_tol)
    cands = [p for p in (px, py) if p is not None]
    if not cands:
        raise ValueError("need at least two distinct C4 X or Y coordinates")
    return float(max(cands))


def suggest_pad_pitch(
    net: SpiceNetlist,
    node_map: Dict[str, str],
    roots: Set[str],
    *,
    grid_to_pad_ratio: float = 1.0,
) -> float:
    """
    C4-lattice auto pitch (no metal clamp).

    Target is the measured bump-array period (max of median Δx / Δy of unique
    C4 rows and columns). ``grid_to_pad_ratio`` > 1 requests a denser grid
    (``pitch / sqrt(ratio)``). Falls back to count-packing only if the C4s
    do not span two distinct coordinates on either axis.
    """
    pads = _real_pad_sites(net, node_map, roots)
    if not pads:
        raise ValueError("no pads to derive pitch from")
    ratio = float(grid_to_pad_ratio)
    if ratio <= 0.0:
        raise ValueError("grid_to_pad_ratio must be positive")
    xs = [p[1] for p in pads]
    ys = [p[2] for p in pads]
    try:
        p_c4 = c4_lattice_pitch(xs, ys)
    except ValueError:
        xmin, ymin, xmax, ymax = net.bbox
        side = max(xmax - xmin, ymax - ymin)
        n_target = max(2, int(math.ceil(math.sqrt(len(pads) * ratio))))
        return float(side / n_target)
    return float(p_c4 / math.sqrt(ratio))


def suggest_legal_chip_pitch(
    net: SpiceNetlist,
    node_map: Dict[str, str],
    roots: Set[str],
    *,
    grid_to_pad_ratio: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Legal Pixel-R pitch from the C4 lattice, clamped above max metal.

    ``chip_pitch = max(c4_lattice, max_metal + ε)``. Returns
    ``(pitch, c4_lattice_target, max_metal_pitch)``.
    """
    pitches = vdd_metal_pitches(net, node_map, roots)
    max_metal = max(pitches.values()) if pitches else 0.0
    target = suggest_pad_pitch(
        net, node_map, roots, grid_to_pad_ratio=grid_to_pad_ratio
    )
    if max_metal > 0.0 and target <= max_metal:
        pitch = float(max_metal) + max(1.0, abs(float(max_metal)) * 1e-6)
    else:
        pitch = float(target)
    return pitch, float(target), float(max_metal)



def _vdd_layer_nodes(
    net: SpiceNetlist,
    node_map: Dict[str, str],
    roots: Set[str],
    layers: Sequence[int],
) -> Tuple[int, int, Dict[int, List[Tuple[str, float, float]]]]:
    """
    Among VDD-component nodes, return (top_layer, bot_layer, layer->candidates).

    Each candidate is (root_name, x, y) for nodes originally on that layer.
    """
    by_layer: Dict[int, Dict[str, Tuple[float, float]]] = {int(L): {} for L in layers}
    layer_set = set(by_layer.keys())
    for name, coord in net.coords.items():
        if coord.layer not in layer_set:
            continue
        root = _root(node_map, name)
        if root not in roots or _is_gnd(root):
            continue
        layer_map = by_layer[coord.layer]
        if root not in layer_map:
            layer_map[root] = (coord.x, coord.y)

    nonempty = {L: nodes for L, nodes in by_layer.items() if nodes}
    if not nonempty:
        raise ValueError("no VDD-layer candidates with coordinates")

    sorted_L = sorted(nonempty.keys())
    bot_layer, top_layer = sorted_L[0], sorted_L[-1]
    candidates: Dict[int, List[Tuple[str, float, float]]] = {
        layer: [(n, xy[0], xy[1]) for n, xy in nodes.items()]
        for layer, nodes in nonempty.items()
    }
    return top_layer, bot_layer, candidates


def _assign_unique_nearest(
    candidates: Sequence[Tuple[str, float, float]],
    centers: Sequence[Tuple[float, float]],
    used: set,
) -> List[Tuple[str, float, float]]:
    """
    For each center, pick nearest unused candidate (greedy row-major order).
    Returns list of (node, x, y) aligned with centers.
    """
    avail = [c for c in candidates if c[0] not in used]
    if len(avail) < len(centers):
        raise ValueError(
            f"need {len(centers)} unique nodes but only {len(avail)} unused "
            f"candidates remain on this layer; use a coarser chip_pitch"
        )

    out: List[Tuple[str, float, float]] = []
    for cx, cy in centers:
        best_i = min(
            range(len(avail)),
            key=lambda i: (avail[i][1] - cx) ** 2 + (avail[i][2] - cy) ** 2,
        )
        chosen = avail.pop(best_i)
        used.add(chosen[0])
        out.append(chosen)
    return out


def _assign_pads_by_containment(
    candidates: Sequence[Tuple[str, float, float]],
    cell_ids: Sequence[Tuple[int, int]],
    centers: Sequence[Tuple[float, float]],
    bbox: Tuple[float, float, float, float],
    cell_size: float,
    nx: int,
    ny: int,
) -> List[Optional[Tuple[str, float, float]]]:
    """
    At most one pad per cell: the nearest candidate *inside* the cell.

    Cells with no candidate stay padless (sink-only). Candidates are real
    C4 / nonzero V→gnd sites (not every top-layer metal node).
    """
    buckets: Dict[Tuple[int, int], List[Tuple[str, float, float]]] = {}
    for node, x, y in candidates:
        ix, iy = _cell_index(x, y, bbox, cell_size, nx, ny)
        buckets.setdefault((ix, iy), []).append((node, float(x), float(y)))

    used: set = set()
    out: List[Optional[Tuple[str, float, float]]] = []
    for (ix, iy), (cx, cy) in zip(cell_ids, centers):
        cands = [c for c in buckets.get((ix, iy), []) if c[0] not in used]
        if not cands:
            out.append(None)
            continue
        best = min(cands, key=lambda t: (t[1] - cx) ** 2 + (t[2] - cy) ** 2)
        used.add(best[0])
        out.append(best)
    return out


def _fill_padless_nearest_top(
    pad_slots: List[Optional[Tuple[str, float, float]]],
    centers: Sequence[Tuple[float, float]],
    top_candidates: Sequence[Tuple[str, float, float]],
    used: set,
) -> None:
    """
    In-place: replace ``None`` pad slots with unique nearest unused top-metal nodes.

    Same greedy nearest rule as ``_assign_unique_nearest``, applied only to
    formerly padless cells (row-major order). Updates ``used`` as nodes are taken.
    """
    if len(pad_slots) != len(centers):
        raise ValueError(
            f"pad_slots ({len(pad_slots)}) and centers ({len(centers)}) length mismatch"
        )
    need = sum(1 for s in pad_slots if s is None)
    if need == 0:
        return
    avail = [c for c in top_candidates if c[0] not in used]
    if len(avail) < need:
        raise ValueError(
            f"full_pads: need {need} more unique top-metal nodes but only "
            f"{len(avail)} unused remain; use a coarser chip_pitch"
        )
    for k, slot in enumerate(pad_slots):
        if slot is not None:
            continue
        cx, cy = centers[k]
        best_i = min(
            range(len(avail)),
            key=lambda i: (avail[i][1] - cx) ** 2 + (avail[i][2] - cy) ** 2,
        )
        chosen = avail.pop(best_i)
        used.add(chosen[0])
        pad_slots[k] = chosen


def build_ports(
    net: SpiceNetlist,
    n: float,
    node_map: Optional[Dict[str, str]] = None,
    *,
    vdd: float = DEFAULT_VDD,
    roots: Optional[Set[str]] = None,
    via_stub: str = VIA_STUB_ZERO,
    full_pads: bool = False,
) -> PortSet:
    """
    Imaginary ports on a full pitch grid (one VDD galvanic component):

    - pad/sink geometry at geometric cell centers (Pixel-R viz / lattice XY)
    - electrical pad: real C4 (nonzero V→gnd) contained in the cell; cells
      with no overlapping C4 stay sink-only unless ``full_pads`` (then unique
      nearest unused top-metal node). At most one pad per cell
      (nearest C4 to the cell center when several overlap)
    - electrical sink / probe: unique nearest node on bottommost VDD layer
    - IBM instance currents (injection < 0) on this component lumped by (x, y)
    - every occupied pad driven at its IBM C4 voltage (``vdd`` is fallback /
      representative ``PortSet.vdd``); with ``full_pads``, **all** pads share
      one common drive (representative of the real C4 voltages)
    - ``via_stub``: ``zero`` (default) treats pad→center as a short; ``rxry``
      stamps L-bend stubs ``Rx·|dx|/n + Ry·|dy|/n`` in series with Rz

    If ``roots`` is omitted, uses the union of all VDD components (legacy single-rail).
    """
    cell_size = float(n)
    if cell_size <= 0:
        raise ValueError("chip pitch / cell side length n must be > 0")
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

    bbox = net.bbox
    xmin, ymin, _, _ = bbox
    nx, ny = grid_counts(bbox, cell_size)
    top_layer, bot_layer, by_layer = _vdd_layer_nodes(net, node_map, roots, layers)

    centers: List[Tuple[float, float]] = []
    cell_ids: List[Tuple[int, int]] = []
    for iy in range(ny):
        for ix in range(nx):
            cx = xmin + (ix + 0.5) * cell_size
            cy = ymin + (iy + 0.5) * cell_size
            centers.append((cx, cy))
            cell_ids.append((ix, iy))

    real_pads = _real_pad_sites(net, node_map, roots)
    c4_v = {name: float(volt) for name, _x, _y, volt in real_pads}
    c4_cands = [(name, x, y) for name, x, y, _v in real_pads]
    if not c4_cands:
        raise ValueError(
            "no real C4 / nonzero V→gnd pads on this component; "
            f"per-layer pitches={pitches}"
        )
    pad_slots = _assign_pads_by_containment(
        c4_cands,
        cell_ids,
        centers,
        bbox,
        cell_size,
        nx,
        ny,
    )
    n_occ = sum(1 for p in pad_slots if p is not None)
    if n_occ == 0:
        raise ValueError(
            "no grid cell overlaps a real C4 pad "
            f"(chip_pitch={cell_size}, n_c4={len(c4_cands)}); "
            f"per-layer pitches={pitches}"
        )
    used = {p[0] for p in pad_slots if p is not None}
    if full_pads:
        _fill_padless_nearest_top(
            pad_slots, centers, by_layer[top_layer], used
        )
    # When top == bot, pads and sinks share one pool (mutual exclusion).
    sink_chosen = _assign_unique_nearest(by_layer[bot_layer], centers, used)

    # Lump IBM instance currents by cell; only nodes on this component.
    inj = injection_at_nodes(net)
    current_buckets: Dict[Tuple[int, int], float] = {}
    for node, cur in inj.items():
        if cur >= 0.0 or abs(cur) < 1e-30:
            continue
        if _root_of(node_map, node) not in roots:
            continue
        coord = net.coords.get(node)
        if coord is None:
            continue
        ix, iy = _cell_index(coord.x, coord.y, bbox, cell_size, nx, ny)
        current_buckets[(ix, iy)] = current_buckets.get((ix, iy), 0.0) + cur

    # Bottom-layer members whose (x,y) fall in each cell (for ideal shorts).
    # Never short another cell's pad/sink port into this cell.
    member_buckets: Dict[Tuple[int, int], List[str]] = {}
    for node, x, y in by_layer[bot_layer]:
        ix, iy = _cell_index(x, y, bbox, cell_size, nx, ny)
        member_buckets.setdefault((ix, iy), []).append(node)

    pad_nodes: List[str] = []
    pad_xy: List[Tuple[float, float]] = []
    pad_voltages: List[float] = []
    pad_attach: List[int] = []
    # full_pads: one common drive for every cell (representative of real C4 Vs).
    pad_v_common = (
        representative_vdd(list(c4_v.values()), fallback=float(vdd))
        if full_pads
        else None
    )
    for k, slot in enumerate(pad_slots):
        if slot is None:
            continue
        pad_nodes.append(slot[0])
        pad_xy.append((float(slot[1]), float(slot[2])))
        if pad_v_common is not None:
            pad_voltages.append(float(pad_v_common))
        else:
            pad_voltages.append(float(c4_v.get(slot[0], vdd)))
        pad_attach.append(int(k))

    protected_ports = set(pad_nodes) | {s[0] for s in sink_chosen}

    cells_out: List[GridCell] = []
    for (ix, iy), (cx, cy), (sink_node, _sx, _sy) in zip(
        cell_ids, centers, sink_chosen
    ):
        raw_members = member_buckets.get((ix, iy), [])
        members = [sink_node]
        for mem in raw_members:
            if mem == sink_node or mem not in protected_ports:
                if mem not in members:
                    members.append(mem)
        cells_out.append(
            GridCell(
                ix=ix,
                iy=iy,
                node=sink_node,
                x=float(cx),
                y=float(cy),
                current=float(current_buckets.get((ix, iy), 0.0)),
                members=members,
            )
        )

    return PortSet(
        pad_nodes=pad_nodes,
        pad_voltages=pad_voltages,
        cells=cells_out,
        cell_size=cell_size,
        nx=nx,
        ny=ny,
        bbox=bbox,
        top_layer=top_layer,
        bot_layer=bot_layer,
        vdd=representative_vdd(pad_voltages, fallback=float(vdd)),
        vdd_layers=list(layers),
        metal_pitches={int(k): float(v) for k, v in pitches.items()},
        max_metal_pitch=float(max_pitch),
        port_mode=PORT_MODE_C4_GRID,
        pad_xy=pad_xy,
        via_stub=parse_via_stub_arg(via_stub),
        pad_attach=pad_attach,
        full_pads=bool(full_pads),
    )


def _package_voltage_node(net: SpiceNetlist) -> Optional[Tuple[str, float]]:
    """Return (live_node, voltage) for the first nonzero V→gnd source, if any."""
    for n1, n2, v in net.voltages:
        if abs(v) < 1e-30:
            continue
        a_g, b_g = _is_gnd(n1), _is_gnd(n2)
        if a_g == b_g:
            continue
        live = n2 if a_g else n1
        return live, float(v)
    return None


def _lattice_center_xy(net: SpiceNetlist) -> Tuple[float, float]:
    """Center of region-VDD_PORT micron bbox, or (0, 0) if none."""
    ports = collect_region_vdd_ports(net.node_names)
    if not ports:
        return 0.0, 0.0
    xs: List[float] = []
    ys: List[float] = []
    for name in ports.values():
        c = net.coords.get(name)
        if c is None:
            continue
        xs.append(c.x)
        ys.append(c.y)
    if not xs:
        return 0.0, 0.0
    return (min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5


def ensure_tsmc_package_node(
    net: SpiceNetlist,
    *,
    vdd: float = DEFAULT_VDD,
) -> Tuple[str, float]:
    """
    Read (or synthesize) the package DC voltage for pad stimuli.

    Prefers an existing nonzero V→gnd live node (e.g. ``VDD_in``). Otherwise
    synthesizes ``TSMC_PKG_NODE`` with a DC source at ``vdd``. Package shorts
    into the tile mesh are **not** used for bump-pad attach.
    """
    cx, cy = _lattice_center_xy(net)
    found = _package_voltage_node(net)
    if found is not None:
        node, volt = found
        if node not in net.coords:
            net.coords[node] = NodeCoord(layer=LAYER_VIRTUAL_PAD, x=cx, y=cy)
        net.node_names.add(node)
        return node, float(volt)

    node = TSMC_PKG_NODE
    net.node_names.add(node)
    net.coords[node] = NodeCoord(layer=LAYER_VIRTUAL_PAD, x=cx, y=cy)
    net.voltages.append((node, "0", float(vdd)))
    return node, float(vdd)


def _open_sink_name(ix: int, iy: int) -> str:
    return f"__open_sink_x_{ix}_y_{iy}"


def _fallback_tile_for_region(
    net: SpiceNetlist,
    ix: int,
    iy: int,
    sink: str,
) -> str:
    """
    When no BUMP comments exist, attach the pad to a co-indexed tile or the
    sink's non-region resistor neighbor (smoke / synthetic decks).
    """
    from tsmc_region_grid import parse_tile_indices

    for name in net.node_names:
        parsed = parse_tile_indices(name)
        if parsed is None:
            continue
        tx, ty, net_name = parsed
        if net_name == "VDD" and tx == ix and ty == iy:
            return name
    for n1, n2, _ in net.resistors:
        if n1 == sink and n2 != sink and not _is_gnd(n2):
            if parse_tile_indices(n2) is not None or "tile" in n2.lower():
                return n2
        if n2 == sink and n1 != sink and not _is_gnd(n1):
            if parse_tile_indices(n1) is not None or "tile" in n1.lower():
                return n1
    return ""


def build_tsmc_region_ports(
    net: SpiceNetlist,
    *,
    vdd: float = DEFAULT_VDD,
    lattice: Optional[RegionLattice] = None,
    package_node: Optional[str] = None,
    via_stub: str = VIA_STUB_ZERO,
) -> PortSet:
    """
    TSMC bump-lattice ports (real tile pads + region sinks).

    - Pad lattice sized from ``* BUMP_VDD_*`` occupancy (0-based TC1 or
      1-based smoke) covering at least the region-``VDD_PORT`` bbox.
    - Occupied bump → unique ``vpad_x_*_y_*`` near-shorted to that tile
      (injected later); empty bump → isolated open-circuit pad.
    - Sink = ``region_*_VDD_PORT`` at lattice microns (dummy sink if missing).
    - Currents lumped by region node identity from ``.isrc``.
    """
    region_ports = collect_region_vdd_ports(net.node_names)
    if not region_ports:
        raise ValueError("no region VDD_PORT nodes for TSMC bump-lattice ports")

    bumps = collect_vdd_bumps(net)
    base = bump_index_base(bumps)
    rixs = [c[0] for c in region_ports]
    riys = [c[1] for c in region_ports]
    nx_reg = max(rixs)
    ny_reg = max(riys)

    if base == 0:
        # Real TC1: bump (0..nx-1, 0..ny-1) ↔ region (bx+1, by+1).
        # Size from region bbox (43×33 on full TC1) and bump extent.
        nx = nx_reg
        ny = ny_reg
        if bumps:
            nx = max(nx, max(bx for bx, _ in bumps) + 1)
            ny = max(ny, max(by for _, by in bumps) + 1)
        bx0 = by0 = 0

        def region_key(bx: int, by: int) -> Tuple[int, int]:
            return (bx + 1, by + 1)

        def lattice_ix_iy(bx: int, by: int) -> Tuple[int, int]:
            return (bx + 1, by + 1)

    else:
        # Smoke / 1-based: bump indices match region indices.
        ix0_reg, iy0_reg = min(rixs), min(riys)
        nx = nx_reg - ix0_reg + 1
        ny = ny_reg - iy0_reg + 1
        if bumps:
            bx0 = min(min(bx for bx, _ in bumps), ix0_reg)
            by0 = min(min(by for _, by in bumps), iy0_reg)
            nx = max(nx, max(bx for bx, _ in bumps) - bx0 + 1)
            ny = max(ny, max(by for _, by in bumps) - by0 + 1)
        else:
            bx0, by0 = ix0_reg, iy0_reg

        def region_key(bx: int, by: int) -> Tuple[int, int]:
            return (bx, by)

        def lattice_ix_iy(bx: int, by: int) -> Tuple[int, int]:
            return (bx, by)

    cells_idx: List[Tuple[int, int]] = []
    for by in range(by0, by0 + ny):
        for bx in range(bx0, bx0 + nx):
            cells_idx.append((bx, by))

    lat = lattice or RegionLattice(
        origin_x=DEFAULT_LATTICE.origin_x,
        origin_y=DEFAULT_LATTICE.origin_y,
        pitch=DEFAULT_LATTICE.pitch,
        nx=max(nx_reg, 1),
        ny=max(ny_reg, 1),
    )
    micron_cells = [lattice_ix_iy(bx, by) for bx, by in cells_idx]
    bbox = lat.bbox_for_cells(micron_cells)
    pitch = float(lat.pitch)

    inj = injection_at_nodes(net)
    inj_by_suffix: Dict[str, float] = dict(inj)
    for node, cur in list(inj.items()):
        parsed = parse_region_indices(node.split(".")[-1])
        if parsed is None:
            continue
        ix, iy, net_name = parsed
        if net_name != "VDD":
            continue
        bare = region_ports.get((ix, iy))
        if bare and bare not in inj_by_suffix:
            inj_by_suffix[bare] = inj_by_suffix.get(bare, 0.0) + cur

    pad_nodes: List[str] = []
    pad_voltages: List[float] = []
    pad_attach: List[str] = []
    cells_out: List[GridCell] = []
    tile_hit: Dict[str, int] = {}
    use_fallback = not bumps

    for bx, by in cells_idx:
        lix, liy = lattice_ix_iy(bx, by)
        x, y = lat.index_to_xy(lix, liy)
        rkey = region_key(bx, by)
        sink = region_ports.get(rkey)
        if sink is None:
            sink = _open_sink_name(bx, by)
            net.node_names.add(sink)
        net.coords[sink] = NodeCoord(layer=LAYER_VDD, x=x, y=y)

        pad = vpad_name(bx, by)
        pad_nodes.append(pad)
        pad_voltages.append(float(vdd))
        tile = bumps.get((bx, by), "")
        if not tile and use_fallback and rkey in region_ports:
            tile = _fallback_tile_for_region(net, rkey[0], rkey[1], sink)
        pad_attach.append(tile)
        if tile:
            tile_hit[tile] = tile_hit.get(tile, 0) + 1

        cur = float(inj_by_suffix.get(sink, 0.0))
        cells_out.append(
            GridCell(
                ix=bx - bx0,
                iy=by - by0,
                node=sink,
                x=x,
                y=y,
                current=cur,
                members=[sink],
            )
        )

    n_occ = sum(1 for t in pad_attach if t)
    n_open = len(pad_attach) - n_occ
    n_shared = sum(1 for n in tile_hit.values() if n > 1)
    pkg = package_node or ""
    return PortSet(
        pad_nodes=pad_nodes,
        pad_voltages=pad_voltages,
        cells=cells_out,
        cell_size=pitch,
        nx=nx,
        ny=ny,
        bbox=bbox,
        top_layer=LAYER_VIRTUAL_PAD,
        bot_layer=LAYER_VDD,
        vdd=float(vdd),
        vdd_layers=[LAYER_VDD, LAYER_VIRTUAL_PAD],
        metal_pitches={},
        max_metal_pitch=0.0,
        port_mode=PORT_MODE_TSMC_VIRTUAL,
        package_node=str(pkg),
        pad_attach_tiles=pad_attach,
        n_occupied_bumps=n_occ,
        n_open_pads=n_open,
        n_shared_tiles=n_shared,
        pad_xy=[(float(c.x), float(c.y)) for c in cells_out],
        via_stub=parse_via_stub_arg(via_stub),
    )


def inject_virtual_pads(
    net: SpiceNetlist,
    ports: PortSet,
    *,
    package_node: str = "",
    r_package: float = VPAD_PACKAGE_R,
) -> int:
    """
    Register bump-lattice pad nodes; near-short occupied pads onto their tiles.

    Empty bump cells get an isolated pad node (no R) so Kron sees an
    open-circuit V source (zero G′ row/col). Package ``VDD_in`` is **not**
    shorted into the mesh.

    Returns the number of pad–tile resistors added.
    """
    if ports.port_mode != PORT_MODE_TSMC_VIRTUAL:
        return 0

    attach = list(ports.pad_attach_tiles)
    if len(attach) != len(ports.pad_nodes):
        # Legacy ports.json without attach list: treat all as open (no R).
        attach = [""] * len(ports.pad_nodes)

    n_added = 0
    for pad, cell, tile in zip(ports.pad_nodes, ports.cells, attach):
        net.node_names.add(pad)
        net.coords[pad] = NodeCoord(
            layer=LAYER_VIRTUAL_PAD, x=float(cell.x), y=float(cell.y)
        )
        if not tile:
            continue
        net.node_names.add(tile)
        if tile not in net.coords:
            net.coords[tile] = NodeCoord(
                layer=LAYER_VIRTUAL_PAD, x=float(cell.x), y=float(cell.y)
            )
        net.resistors.append((pad, tile, float(r_package)))
        n_added += 1
    _ = package_node  # kept for call-site compatibility; unused
    return n_added


def remap_ports(ports: PortSet, node_map: Dict[str, str]) -> PortSet:
    """Remap port node names through a short-merge / union-find map (preserve grid order)."""

    def m(name: str) -> str:
        return node_map.get(name, name)

    new_pads = [m(name) for name in ports.pad_nodes]
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
    new_attach = [m(t) if t else "" for t in ports.pad_attach_tiles]

    # Duplicate roots after merge are illegal for Kron port partitioning.
    all_ports = new_pads + [c.node for c in new_cells]
    if len(all_ports) != len(set(all_ports)):
        raise ValueError(
            "port remap produced duplicate nodes; check V=0 shorts vs grid assignment"
        )
    if any(_is_gnd(n) for n in all_ports):
        raise ValueError("port remap mapped a port to ground")

    return PortSet(
        pad_nodes=new_pads,
        pad_voltages=list(ports.pad_voltages),
        cells=new_cells,
        cell_size=ports.cell_size,
        nx=ports.nx,
        ny=ports.ny,
        bbox=ports.bbox,
        top_layer=ports.top_layer,
        bot_layer=ports.bot_layer,
        vdd=ports.vdd,
        vdd_layers=list(ports.vdd_layers),
        metal_pitches=dict(ports.metal_pitches),
        max_metal_pitch=float(ports.max_metal_pitch),
        port_mode=ports.port_mode,
        package_node=ports.package_node,
        pad_attach_tiles=new_attach,
        n_occupied_bumps=ports.n_occupied_bumps,
        n_open_pads=ports.n_open_pads,
        n_shared_tiles=ports.n_shared_tiles,
        pad_xy=[(float(x), float(y)) for x, y in ports.pad_xy],
        via_stub=str(getattr(ports, "via_stub", VIA_STUB_ZERO)),
        pad_attach=[int(s) for s in ports.resolved_pad_attach()],
        full_pads=bool(getattr(ports, "full_pads", False)),
    )


def apply_grid_lumping(node_map: Dict[str, str], ports: PortSet) -> Dict[str, str]:
    """
    Ideal-short all bottom-layer members in each grid cell to the sink representative.
    """
    out = dict(node_map)

    def root(name: str) -> str:
        seen = []
        while True:
            nxt = out.get(name, name)
            if nxt == "0":
                for s in seen:
                    out[s] = "0"
                return "0"
            if nxt == name:
                for s in seen:
                    out[s] = name
                return name
            seen.append(name)
            name = nxt

    for c in ports.cells:
        rep = root(c.node)
        if rep == "0":
            continue
        for mem in c.members:
            rm = root(mem)
            if rm == "0" or rm == rep:
                continue
            out[rm] = rep
            out[mem] = rep
        out[c.node] = rep
    keys = list(out.keys())
    for k in keys:
        out[k] = root(k)
    return out
