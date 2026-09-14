"""Lift 2D PDN geometry to 3D polylines with synthetic layer heights."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from pdn_geometry import PdnGeometry, Segment, ViaSeg

# (x1, y1, z1, x2, y2, z2)
Seg3D = Tuple[float, float, float, float, float, float]


@dataclass
class NetGeometry3D:
    name: str
    layers: List[int] = field(default_factory=list)
    metal: Dict[int, List[Seg3D]] = field(default_factory=dict)
    # Keep layer indices so per-layer via filtering stays exact.
    vias: List[ViaSeg] = field(default_factory=list)

    @property
    def n_metal(self) -> int:
        return sum(len(v) for v in self.metal.values())


@dataclass
class PdnGeometry3D:
    vdd_components: List[NetGeometry3D]
    vss: NetGeometry3D
    bbox: Tuple[float, float, float, float]
    z_pitch: float
    layer_z: Dict[int, float] = field(default_factory=dict)

    @property
    def vdd(self) -> NetGeometry3D:
        if not self.vdd_components:
            raise ValueError("no VDD components")
        return self.vdd_components[0]

    @property
    def n_vdd_components(self) -> int:
        return len(self.vdd_components)

    def net_geometry(self, name: str) -> NetGeometry3D:
        from pdn_geometry import normalize_net_arg

        key = normalize_net_arg(name)
        if key == "vss":
            return self.vss
        for g in self.vdd_components:
            if g.name == key:
                return g
        raise KeyError(f"unknown net {name!r}")

    @property
    def z_range(self) -> Tuple[float, float]:
        if not self.layer_z:
            return (0.0, self.z_pitch)
        zs = list(self.layer_z.values())
        return (min(zs), max(zs))


def default_z_pitch(bbox: Tuple[float, float, float, float]) -> float:
    """Synthetic vertical pitch from XY bbox diagonal / 20."""
    xmin, ymin, xmax, ymax = bbox
    diag = math.hypot(xmax - xmin, ymax - ymin)
    if diag <= 0.0:
        return 1.0
    return diag / 20.0


def layer_z_map(
    layers: Sequence[int],
    z_pitch: float,
) -> Dict[int, float]:
    """Map each distinct metal index to ``rank * z_pitch`` (sorted by layer)."""
    uniq = sorted({int(L) for L in layers})
    return {L: float(i) * float(z_pitch) for i, L in enumerate(uniq)}


def z_of(layer: int, layer_z: Dict[int, float], z_pitch: float) -> float:
    if layer in layer_z:
        return layer_z[layer]
    return float(layer) * float(z_pitch)


def _metal_to_3d(segs: List[Segment], z: float) -> List[Seg3D]:
    return [(x1, y1, z, x2, y2, z) for x1, y1, x2, y2 in segs]


def lift_pdn_geometry(
    geom: PdnGeometry,
    z_pitch: Optional[float] = None,
) -> PdnGeometry3D:
    """
    Convert 2D ``PdnGeometry`` into stacked 3D metal segments.

    ``z(layer)`` uses sorted unique layers across VDD+VSS so shared metal
    indices align between nets. Vias keep their layer indices for filtering.
    """
    if z_pitch is None:
        z_pitch = default_z_pitch(geom.bbox)
    if z_pitch <= 0.0:
        raise ValueError("z_pitch must be positive")

    all_layers: List[int] = []
    for g in geom.vdd_components:
        all_layers.extend(g.layers)
    all_layers.extend(geom.vss.layers)
    lz = layer_z_map(all_layers, z_pitch)

    def _lift_net(net_geom) -> NetGeometry3D:
        metal: Dict[int, List[Seg3D]] = {}
        for L in net_geom.layers:
            z = z_of(int(L), lz, z_pitch)
            metal[int(L)] = _metal_to_3d(net_geom.metal.get(int(L), []), z)
        return NetGeometry3D(
            name=net_geom.name,
            layers=list(net_geom.layers),
            metal=metal,
            vias=list(net_geom.vias),
        )

    return PdnGeometry3D(
        vdd_components=[_lift_net(g) for g in geom.vdd_components],
        vss=_lift_net(geom.vss),
        bbox=geom.bbox,
        z_pitch=float(z_pitch),
        layer_z=lz,
    )


def segments_to_xyz(
    segs: List[Seg3D],
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """Flatten 3D segments to Plotly line arrays with None breaks."""
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    zs: List[Optional[float]] = []
    for x1, y1, z1, x2, y2, z2 in segs:
        xs.extend([x1, x2, None])
        ys.extend([y1, y2, None])
        zs.extend([z1, z2, None])
    return xs, ys, zs


def vias_to_xyz(
    vias: List[ViaSeg],
    layer_z: Dict[int, float],
    z_pitch: float,
    layer: Optional[int] = None,
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """
    3D via polylines. If ``layer`` is set, keep vias that touch that layer.
    """
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    zs: List[Optional[float]] = []
    for x1, y1, l1, x2, y2, l2 in vias:
        if layer is not None and layer != l1 and layer != l2:
            continue
        z1 = z_of(int(l1), layer_z, z_pitch)
        z2 = z_of(int(l2), layer_z, z_pitch)
        xs.extend([x1, x2, None])
        ys.extend([y1, y2, None])
        zs.extend([z1, z2, None])
    return xs, ys, zs
