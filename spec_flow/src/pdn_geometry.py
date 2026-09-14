"""Extract drawable PDN metal/via segments by VDD component / VSS and metal layer."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from graph import (
    _root_of,
    build_short_map,
    cross_layer_v0_pairs,
    vdd_connected_components,
    vdd_layers,
    vdd_roots,
    vss_layers,
)
from spice_parser import SpiceNetlist

Segment = Tuple[float, float, float, float]  # x1, y1, x2, y2
ViaSeg = Tuple[float, float, int, float, float, int]  # x1, y1, L1, x2, y2, L2


@dataclass
class NetGeometry:
    name: str
    layers: List[int] = field(default_factory=list)
    metal: Dict[int, List[Segment]] = field(default_factory=dict)
    vias: List[ViaSeg] = field(default_factory=list)

    @property
    def n_metal(self) -> int:
        return sum(len(v) for v in self.metal.values())


@dataclass
class PdnGeometry:
    vdd_components: List[NetGeometry]
    vss: NetGeometry
    bbox: Tuple[float, float, float, float]
    node_map: Dict[str, str] = field(default_factory=dict)
    vdd_root_set: Set[str] = field(default_factory=set)

    @property
    def vdd(self) -> NetGeometry:
        """First (topmost) VDD component — back-compat alias for ``vdd1``."""
        if not self.vdd_components:
            raise ValueError("no VDD components")
        return self.vdd_components[0]

    @property
    def n_vdd_components(self) -> int:
        return len(self.vdd_components)

    def net_geometry(self, name: str) -> NetGeometry:
        key = normalize_net_arg(name)
        if key == "vss":
            return self.vss
        for g in self.vdd_components:
            if g.name == key:
                return g
        raise KeyError(f"unknown net {name!r}")

    def iter_drawable_nets(self) -> Iterable[NetGeometry]:
        yield from self.vdd_components
        yield self.vss


def normalize_net_arg(name: str) -> str:
    """Map CLI/API net names: ``vdd`` → ``vdd1``; leave ``vddk`` / ``vss``."""
    n = name.strip().lower()
    if n == "vdd":
        return "vdd1"
    return n


def net_display_label(name: str) -> str:
    return normalize_net_arg(name).upper()


def _classify_root(
    root: str,
    comp_of_root: Dict[str, str],
    vdd_union: Set[str],
) -> Optional[str]:
    """Return ``vddk`` or ``vss`` for a mapped root name."""
    if root in comp_of_root:
        return comp_of_root[root]
    if root in vdd_union:
        return None
    return "vss"


def extract_pdn_geometry(
    net: SpiceNetlist,
    node_map: Optional[Dict[str, str]] = None,
) -> PdnGeometry:
    """
    Build per-net metal and via segments from a parsed IBM SPICE netlist.

    VDD rails are split into connected components (``vdd1`` = topmost, …).
    VSS: everything else with coordinates (including ground-merged GND stack).
    Metal edges: same-layer ``R``. Vias: cross-layer ``V=0`` and cross-layer ``R``.
    """
    if node_map is None:
        node_map = build_short_map(net)

    comps = vdd_connected_components(net, node_map)
    vdd_union = set().union(*comps) if comps else vdd_roots(net, node_map)
    vss_L = vss_layers(net, node_map, vdd_union)

    comp_of_root: Dict[str, str] = {}
    vdd_geoms: List[NetGeometry] = []
    vdd_metal: Dict[str, Dict[int, List[Segment]]] = {}
    vdd_vias: Dict[str, List[ViaSeg]] = {}
    for k, roots in enumerate(comps, start=1):
        name = f"vdd{k}"
        for r in roots:
            comp_of_root[r] = name
        layers = vdd_layers(net, node_map, roots)
        vdd_metal[name] = defaultdict(list)
        vdd_vias[name] = []
        vdd_geoms.append(
            NetGeometry(
                name=name,
                layers=list(layers),
                metal={},
                vias=[],
            )
        )

    vss_metal: Dict[int, List[Segment]] = defaultdict(list)
    vss_vias: List[ViaSeg] = []
    coords = net.coords

    def _endpoint_net(node: str) -> Optional[str]:
        return _classify_root(_root_of(node_map, node), comp_of_root, vdd_union)

    def _append_via(n1: str, n2: str) -> None:
        c1, c2 = coords.get(n1), coords.get(n2)
        if c1 is None or c2 is None or c1.layer == c2.layer:
            return
        net1 = _endpoint_net(n1)
        net2 = _endpoint_net(n2)
        if net1 is None or net1 != net2:
            return
        via = (c1.x, c1.y, c1.layer, c2.x, c2.y, c2.layer)
        if net1 == "vss":
            vss_vias.append(via)
        else:
            vdd_vias[net1].append(via)

    for n1, n2, _r in net.resistors:
        c1, c2 = coords.get(n1), coords.get(n2)
        if c1 is None or c2 is None:
            continue
        if c1.layer != c2.layer:
            _append_via(n1, n2)
            continue
        net1 = _endpoint_net(n1)
        net2 = _endpoint_net(n2)
        if net1 is None or net1 != net2:
            continue
        seg = (c1.x, c1.y, c2.x, c2.y)
        if net1 == "vss":
            vss_metal[c1.layer].append(seg)
        else:
            vdd_metal[net1][c1.layer].append(seg)

    for n1, n2 in cross_layer_v0_pairs(net):
        _append_via(n1, n2)

    for g in vdd_geoms:
        g.metal = {int(L): list(vdd_metal[g.name].get(int(L), [])) for L in g.layers}
        g.vias = list(vdd_vias[g.name])

    return PdnGeometry(
        vdd_components=vdd_geoms,
        vss=NetGeometry(
            name="vss",
            layers=list(vss_L),
            metal={int(L): vss_metal.get(int(L), []) for L in vss_L},
            vias=vss_vias,
        ),
        bbox=net.bbox,
        node_map=node_map,
        vdd_root_set=vdd_union,
    )


def segments_to_xy(segs: List[Segment]) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Flatten segments to Plotly line arrays with None breaks."""
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    for x1, y1, x2, y2 in segs:
        xs.extend([x1, x2, None])
        ys.extend([y1, y2, None])
    return xs, ys


def vias_to_xy(
    vias: List[ViaSeg],
    layer: Optional[int] = None,
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """
    XY projection of vias. If ``layer`` is set, keep vias that touch that layer.
    Coincident endpoints become a single marker point (x, y, None).
    """
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    for x1, y1, l1, x2, y2, l2 in vias:
        if layer is not None and layer != l1 and layer != l2:
            continue
        if abs(x1 - x2) < 1e-12 and abs(y1 - y2) < 1e-12:
            xs.extend([x1, None])
            ys.extend([y1, None])
        else:
            xs.extend([x1, x2, None])
            ys.extend([y1, y2, None])
    return xs, ys
