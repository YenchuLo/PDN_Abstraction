"""Graph construction: short-merge and sparse MNA conductance matrix."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import sparse

from spice_parser import SpiceNetlist, injection_at_nodes

# Near-ideal conductance for cross-layer V=0 (vias / stack shorts kept as 2 nodes).
VIA_SHORT_G = 1e9

# Stamp window for branch R. TSMC n-port decks include near-shorts (~1e-15)
# and dummy opens (~1e14+) that otherwise explode G / G' eigenvalues.
R_STAMP_MIN = 1e-6
R_STAMP_MAX = 1e12


def clip_stamp_resistance(
    r: float,
    *,
    r_min: float = R_STAMP_MIN,
    r_max: float = R_STAMP_MAX,
) -> Optional[float]:
    """Clip a branch resistance into ``[r_min, r_max]``.

    Returns ``None`` for non-finite or non-positive ``r`` (skip the stamp).
    """
    try:
        val = float(r)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val) or val <= 0.0:
        return None
    lo = float(r_min)
    hi = float(r_max)
    if lo <= 0.0 or hi < lo:
        raise ValueError(f"invalid stamp window [{lo}, {hi}]")
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: str) -> str:
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def mapping(self) -> Dict[str, str]:
        return {k: self.find(k) for k in self.parent}


def _is_gnd(name: str) -> bool:
    return name == "0" or name.lower() == "gnd"


def build_short_map(net: SpiceNetlist) -> Dict[str, str]:
    """
    Merge nodes connected by V=0 ideal shorts.

    - ``V n 0 0`` / ``V 0 n 0`` → map to ground.
    - Same-layer V=0 between two signal nodes → union.
    - Cross-layer V=0 (e.g. L1↔L3 vias, L0↔L2) are **not** unioned so
      layer-distinct ports remain; they are stamped as near-shorts in
      ``assemble_conductance`` / SPICE emit.
    """
    uf = UnionFind()
    for n in net.node_names:
        uf.add(n)
    grounded_nodes: List[str] = []
    for n1, n2, v in net.voltages:
        if abs(v) >= 1e-30:
            continue
        g1, g2 = _is_gnd(n1), _is_gnd(n2)
        if g1 and g2:
            continue
        if g1 or g2:
            grounded_nodes.append(n2 if g1 else n1)
            continue
        c1, c2 = net.coords.get(n1), net.coords.get(n2)
        if c1 is not None and c2 is not None and c1.layer != c2.layer:
            continue  # cross-layer: keep distinct nodes
        uf.union(n1, n2)

    grounded_roots = {uf.find(g) for g in grounded_nodes}
    mapping: Dict[str, str] = {}
    for n in list(uf.parent.keys()):
        root = uf.find(n)
        mapping[n] = "0" if root in grounded_roots else root
    for g in grounded_nodes:
        mapping[g] = "0"
    return mapping


def cross_layer_v0_pairs(net: SpiceNetlist) -> List[Tuple[str, str]]:
    """Non-ground V=0 shorts between different metal layers (ideal vias / stack)."""
    pairs: List[Tuple[str, str]] = []
    for n1, n2, v in net.voltages:
        if abs(v) >= 1e-30:
            continue
        if _is_gnd(n1) or _is_gnd(n2):
            continue
        c1, c2 = net.coords.get(n1), net.coords.get(n2)
        if c1 is None or c2 is None or c1.layer == c2.layer:
            continue
        pairs.append((n1, n2))
    return pairs


def _root_of(node_map: Dict[str, str], name: str) -> str:
    if _is_gnd(name):
        return "0"
    return node_map.get(name, name)


def vdd_roots(net: SpiceNetlist, node_map: Dict[str, str]) -> Set[str]:
    """
    Non-ground roots in all VDD galvanic components (union).

    Seeds: nodes with injection < 0 and nonzero-V terminals (net ID only).
    Edges: resistors + cross-layer V=0 pairs, after ``node_map`` roots.
    For per-rail work use ``vdd_connected_components``.
    """
    comps = vdd_connected_components(net, node_map)
    return set().union(*comps) if comps else set()


def _vdd_seeds(net: SpiceNetlist, node_map: Dict[str, str]) -> Set[str]:
    seeds: Set[str] = set()
    for node, cur in injection_at_nodes(net).items():
        if cur >= 0.0:
            continue
        r = _root_of(node_map, node)
        if not _is_gnd(r):
            seeds.add(r)
    for n1, n2, v in net.voltages:
        if abs(v) < 1e-30:
            continue
        for n in (n1, n2):
            if _is_gnd(n):
                continue
            r = _root_of(node_map, n)
            if not _is_gnd(r):
                seeds.add(r)
    if not seeds:
        raise ValueError("no VDD seeds (need inj<0 sinks and/or nonzero V sources)")
    return seeds


def _component_layers(
    net: SpiceNetlist, node_map: Dict[str, str], roots: Set[str]
) -> Set[int]:
    """Metal layers touched by ``roots``."""
    return {
        coord.layer
        for name, coord in net.coords.items()
        if _root_of(node_map, name) in roots
    }


def _component_layer_span(
    net: SpiceNetlist, node_map: Dict[str, str], roots: Set[str]
) -> Tuple[int, int]:
    """Return (topmost_layer, bottommost_layer) for roots; (-1,-1) if none."""
    layers = _component_layers(net, node_map, roots)
    if not layers:
        return -1, -1
    return max(layers), min(layers)


def vdd_connected_components(
    net: SpiceNetlist, node_map: Dict[str, str]
) -> List[Set[str]]:
    """
    VDD components keyed by **layer set**, topmost-layer first.

    Galvanic islands that share the same metal-layer set (e.g. several
    disconnected {1,3} meshes) are merged into one component. Distinct
    interleaved rails (different layer sets) stay separate.

    Ordering: max layer descending, then min layer descending, then size.
    """
    seeds = _vdd_seeds(net, node_map)
    adj = _galvanic_adj(net, node_map)

    # Grow the union of all seed-reachable nodes, then split into CCs.
    union: Set[str] = set()
    q: deque[str] = deque()
    for s in seeds:
        if s not in union:
            union.add(s)
            q.append(s)
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in union:
                union.add(v)
                q.append(v)

    islands: List[Set[str]] = []
    remaining = set(union)
    while remaining:
        start = next(iter(remaining))
        comp: Set[str] = set()
        q = deque([start])
        remaining.remove(start)
        comp.add(start)
        while q:
            u = q.popleft()
            for v in adj.get(u, ()):
                if v in remaining:
                    remaining.remove(v)
                    comp.add(v)
                    q.append(v)
        islands.append(comp)

    # Merge islands that occupy the same layer set.
    by_layers: Dict[frozenset, Set[str]] = {}
    for island in islands:
        key = frozenset(_component_layers(net, node_map, island))
        if key in by_layers:
            by_layers[key] |= island
        else:
            by_layers[key] = set(island)

    comps = list(by_layers.values())

    def sort_key(comp: Set[str]) -> Tuple[int, int, int]:
        top, bot = _component_layer_span(net, node_map, comp)
        return (top, bot, len(comp))

    comps.sort(key=sort_key, reverse=True)
    return comps


def _galvanic_adj(
    net: SpiceNetlist, node_map: Dict[str, str]
) -> Dict[str, Set[str]]:
    """Undirected adjacency over resistors + cross-layer V=0 (non-ground roots)."""
    adj: Dict[str, Set[str]] = defaultdict(set)

    def link(a: str, b: str) -> None:
        if _is_gnd(a) or _is_gnd(b) or a == b:
            return
        adj[a].add(b)
        adj[b].add(a)

    for n1, n2, _r in net.resistors:
        link(_root_of(node_map, n1), _root_of(node_map, n2))
    for n1, n2 in cross_layer_v0_pairs(net):
        link(_root_of(node_map, n1), _root_of(node_map, n2))
    return adj


def vss_roots(
    net: SpiceNetlist,
    node_map: Dict[str, str],
    vdd: Optional[Set[str]] = None,
) -> Set[str]:
    """
    Non-ground roots in the VSS/GND galvanic component(s).

    Seeds: layered nodes whose root is not in the VDD set and not ground.
    Edges: resistors + cross-layer V=0 pairs (same as ``vdd_roots``).
    Nodes shorted to ideal ground (``V n 0 0``) are excluded from the root set;
    their metal is still drawable via original coordinates (see ``pdn_geometry``).
    """
    if vdd is None:
        vdd = vdd_roots(net, node_map)

    seeds: Set[str] = set()
    for name in net.coords:
        r = _root_of(node_map, name)
        if _is_gnd(r) or r in vdd:
            continue
        seeds.add(r)

    adj = _galvanic_adj(net, node_map)
    seen: Set[str] = set()
    q: deque[str] = deque()
    for s in seeds:
        if s in vdd or _is_gnd(s):
            continue
        if s not in seen:
            seen.add(s)
            q.append(s)
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v in vdd or _is_gnd(v) or v in seen:
                continue
            seen.add(v)
            q.append(v)
    return seen


def galvanic_component_of(
    net: SpiceNetlist,
    node_map: Dict[str, str],
    seed_nodes: Sequence[str],
) -> Set[str]:
    """Connected component (via resistors + cross-layer V=0) containing seed nodes."""
    adj = _galvanic_adj(net, node_map)
    seeds = {_root_of(node_map, n) for n in seed_nodes if not _is_gnd(n)}
    seeds = {s for s in seeds if not _is_gnd(s)}
    if not seeds:
        raise ValueError("no non-ground seeds for galvanic_component_of")
    seen: Set[str] = set()
    q: deque[str] = deque()
    for s in seeds:
        if s not in seen:
            seen.add(s)
            q.append(s)
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def vdd_layers(net: SpiceNetlist, node_map: Dict[str, str], roots: Optional[Set[str]] = None) -> List[int]:
    """Sorted metal layers that appear on the VDD component."""
    if roots is None:
        roots = vdd_roots(net, node_map)
    layers = {
        coord.layer
        for name, coord in net.coords.items()
        if _root_of(node_map, name) in roots
    }
    if not layers:
        raise ValueError("VDD component has no layered coordinates")
    return sorted(layers)


def vss_layers(
    net: SpiceNetlist,
    node_map: Dict[str, str],
    vdd: Optional[Set[str]] = None,
) -> List[int]:
    """Sorted metal layers on non-VDD nodes (includes ground-merged GND stack)."""
    if vdd is None:
        vdd = vdd_roots(net, node_map)
    layers = {
        coord.layer
        for name, coord in net.coords.items()
        if _root_of(node_map, name) not in vdd
    }
    return sorted(layers)


def _max_track_gap(values: List[float]) -> float:
    """Largest consecutive gap among unique sorted track coordinates."""
    arr = np.asarray(sorted(set(float(v) for v in values)), dtype=float)
    if arr.size < 2:
        raise ValueError("need at least two track coordinates to infer pitch")
    return float(np.max(np.diff(arr)))


def metal_stripe_pitch(
    net: SpiceNetlist,
    layer: int,
    node_map: Optional[Dict[str, str]] = None,
    roots: Optional[Set[str]] = None,
) -> float:
    """
    Infer preferred-direction stripe pitch for one metal layer.

    Uses same-layer resistors: horizontal rails → pitch = max Δy between y-tracks;
    vertical rails → pitch = max Δx between x-tracks. Pitch is the largest
    consecutive preferred-direction track gap (including inter-band voids).
    """
    if node_map is None:
        node_map = {n: n for n in net.node_names}
    coords = net.coords
    horiz = 0
    vert = 0
    track_x: Set[float] = set()
    track_y: Set[float] = set()
    for n1, n2, _r in net.resistors:
        c1, c2 = coords.get(n1), coords.get(n2)
        if c1 is None or c2 is None or c1.layer != layer or c2.layer != layer:
            continue
        if roots is not None:
            if _root_of(node_map, n1) not in roots and _root_of(node_map, n2) not in roots:
                continue
        dx = abs(c1.x - c2.x)
        dy = abs(c1.y - c2.y)
        if dx < 1e-9 and dy > 1e-9:
            vert += 1
            track_x.add(c1.x)
        elif dy < 1e-9 and dx > 1e-9:
            horiz += 1
            track_y.add(c1.y)

    if vert == 0 and horiz == 0:
        # Fall back to all node coordinates on this layer in the VDD set
        xs, ys = [], []
        for name, coord in coords.items():
            if coord.layer != layer:
                continue
            if roots is not None and _root_of(node_map, name) not in roots:
                continue
            xs.append(coord.x)
            ys.append(coord.y)
        if len(set(xs)) >= len(set(ys)) and len(set(xs)) >= 2:
            return _max_track_gap(xs)
        if len(set(ys)) >= 2:
            return _max_track_gap(ys)
        raise ValueError(f"cannot infer metal pitch for layer {layer}")

    if vert >= horiz:
        return _max_track_gap(list(track_x))
    return _max_track_gap(list(track_y))


def vdd_metal_pitches(
    net: SpiceNetlist,
    node_map: Dict[str, str],
    roots: Optional[Set[str]] = None,
) -> Dict[int, float]:
    """Stripe pitch per VDD metal layer."""
    if roots is None:
        roots = vdd_roots(net, node_map)
    layers = vdd_layers(net, node_map, roots)
    return {
        int(L): float(metal_stripe_pitch(net, int(L), node_map=node_map, roots=roots))
        for L in layers
    }


@dataclass
class ConductanceSystem:
    """Sparse nodal conductance Laplacian (no ground row)."""

    G: sparse.csr_matrix
    nodes: List[str]
    index: Dict[str, int]
    node_map: Dict[str, str]
    r_clip: Dict[str, Any] = field(default_factory=dict)


def assemble_conductance(
    net: SpiceNetlist,
    node_map: Dict[str, str] | None = None,
    *,
    keep_roots: Optional[Set[str]] = None,
    via_g: float = VIA_SHORT_G,
    r_min: float = R_STAMP_MIN,
    r_max: float = R_STAMP_MAX,
) -> ConductanceSystem:
    """
    Build G with G_ii = sum g, G_ij = -g for resistors between merged nodes.
    Ground ("0") is excluded; resistors to ground become diagonal stamps.

    Branch resistances are clipped to ``[r_min, r_max]`` before ``g = 1/R``
    so TSMC-style near-shorts / dummy opens do not dominate the spectrum.

    If ``keep_roots`` is set (e.g. VDD component), only those non-ground roots
    are retained. Cross-layer V=0 pairs are stamped with conductance ``via_g``.
    """
    if node_map is None:
        node_map = {n: n for n in net.node_names}

    def root(n: str) -> str:
        return _root_of(node_map, n)

    if keep_roots is None:
        nodes_set = {root(n) for n in net.node_names if not _is_gnd(root(n))}
    else:
        nodes_set = {r for r in keep_roots if not _is_gnd(r)}

    nodes = sorted(nodes_set)
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    def stamp(i: int, j: int, g: float) -> None:
        rows.append(i)
        cols.append(j)
        data.append(g)

    def stamp_edge(a: str, b: str, g: float) -> None:
        if _is_gnd(a) and _is_gnd(b):
            return
        if a == b:
            return
        if _is_gnd(a):
            if b in index:
                jb = index[b]
                stamp(jb, jb, g)
            return
        if _is_gnd(b):
            if a in index:
                ja = index[a]
                stamp(ja, ja, g)
            return
        if a not in index or b not in index:
            return
        ia, ib = index[a], index[b]
        stamp(ia, ia, g)
        stamp(ib, ib, g)
        stamp(ia, ib, -g)
        stamp(ib, ia, -g)

    n_r = 0
    n_lo = 0
    n_hi = 0
    n_skip = 0
    raw_min = float("inf")
    raw_max = 0.0
    for n1, n2, r in net.resistors:
        n_r += 1
        try:
            rf = float(r)
        except (TypeError, ValueError):
            n_skip += 1
            continue
        if np.isfinite(rf) and rf > 0.0:
            if rf < raw_min:
                raw_min = rf
            if rf > raw_max:
                raw_max = rf
        rc = clip_stamp_resistance(rf, r_min=r_min, r_max=r_max)
        if rc is None:
            n_skip += 1
            continue
        if rf < r_min:
            n_lo += 1
        elif rf > r_max:
            n_hi += 1
        stamp_edge(root(n1), root(n2), 1.0 / rc)

    if via_g > 0.0:
        for n1, n2 in cross_layer_v0_pairs(net):
            stamp_edge(root(n1), root(n2), via_g)

    if not data:
        G = sparse.csr_matrix((n, n), dtype=float)
    else:
        G = sparse.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=float)
        G = G.tocsr()
        G.sum_duplicates()

    r_clip = {
        "r_min": float(r_min),
        "r_max": float(r_max),
        "n_resistors": n_r,
        "n_clipped_lo": n_lo,
        "n_clipped_hi": n_hi,
        "n_skipped": n_skip,
        "r_raw_min": None if raw_min == float("inf") else float(raw_min),
        "r_raw_max": None if raw_min == float("inf") else float(raw_max),
    }
    return ConductanceSystem(
        G=G, nodes=nodes, index=index, node_map=node_map, r_clip=r_clip
    )


def partition_ports(
    system: ConductanceSystem,
    port_nodes: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (port_indices, internal_indices) in system index space."""
    port_idx = []
    seen = set()
    for name in port_nodes:
        if name not in system.index:
            raise KeyError(f"port node {name!r} not in conductance system")
        i = system.index[name]
        if i in seen:
            raise ValueError(f"duplicate port index for {name!r}")
        seen.add(i)
        port_idx.append(i)
    port_idx_arr = np.asarray(port_idx, dtype=int)
    mask = np.ones(len(system.nodes), dtype=bool)
    mask[port_idx_arr] = False
    internal_idx = np.nonzero(mask)[0]
    return port_idx_arr, internal_idx
