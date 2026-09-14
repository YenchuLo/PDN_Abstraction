"""Drawable 3D geometry for the Pixel-R star abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pdn_geometry_3d import Seg3D, z_of
from ports import VIA_STUB_RXRY, PortSet

Point3D = Tuple[float, float, float]


@dataclass
class PixelGeometry3D:
    """Pixel-R drawable primitives in the same XYZ space as ``PdnGeometry3D``."""

    pads: List[Point3D] = field(default_factory=list)
    sinks: List[Point3D] = field(default_factory=list)
    rx_edges: List[Seg3D] = field(default_factory=list)
    ry_edges: List[Seg3D] = field(default_factory=list)
    rz_edges: List[Seg3D] = field(default_factory=list)
    # Parallel to sinks / pads for hover text
    sink_meta: List[Dict[str, Any]] = field(default_factory=list)
    pad_meta: List[Dict[str, Any]] = field(default_factory=list)
    Rx: Optional[float] = None
    Ry: Optional[float] = None
    Rz: Optional[float] = None
    z_pad: float = 0.0
    z_sink: float = 0.0
    z_bot: float = 0.0

    @property
    def n_pads(self) -> int:
        return len(self.pads)

    @property
    def n_sinks(self) -> int:
        return len(self.sinks)

    @property
    def n_rx(self) -> int:
        return len(self.rx_edges)

    @property
    def n_ry(self) -> int:
        return len(self.ry_edges)

    @property
    def n_rz(self) -> int:
        return len(self.rz_edges)


def _plane_z(
    layer: int,
    layer_z: Dict[int, float],
    z_pitch: float,
    *,
    offset: float = 0.0,
) -> float:
    return z_of(int(layer), layer_z, z_pitch) + float(offset)


def _cell_center_xy(ports: PortSet, ix: int, iy: int) -> Tuple[float, float]:
    xmin, ymin, _, _ = ports.bbox
    pitch = float(ports.cell_size)
    return xmin + (ix + 0.5) * pitch, ymin + (iy + 0.5) * pitch


def build_pixel_geometry_3d(
    ports: PortSet,
    pad_attach: Sequence[int],
    ew_shared: Sequence[Sequence[int]],
    ns_shared: Sequence[Sequence[int]],
    layer_z: Dict[int, float],
    z_pitch: float,
    *,
    ew_boundary: Optional[Sequence[Sequence[Any]]] = None,
    ns_boundary: Optional[Sequence[Sequence[Any]]] = None,
    Rx: Optional[float] = None,
    Ry: Optional[float] = None,
    Rz: Optional[float] = None,
    pad_z_offset: Optional[float] = None,
    sink_z_offset: Optional[float] = None,
    bot_z_offset: Optional[float] = None,
) -> PixelGeometry3D:
    """
    Build Pixel-R star markers and half-arm edges at geometric cell centers.

    Pads / sinks sit at cell-center XY. Rx/Ry are center→edge-midpoint half-arms.
    Rz is drawn as Rup (pad→center) and Rdown (center→bot) vertical segments.
    """
    if not ports.cells:
        raise ValueError("no active sink cells for Pixel-R geometry")

    top = int(ports.top_layer) if ports.top_layer >= 0 else max(layer_z.keys(), default=0)
    bot = int(ports.bot_layer) if ports.bot_layer >= 0 else min(layer_z.keys(), default=0)
    dz = 0.05 * float(z_pitch)
    if pad_z_offset is None:
        pad_z_offset = dz
    if sink_z_offset is None:
        sink_z_offset = 0.0
    if bot_z_offset is None:
        bot_z_offset = -dz

    z_pad = _plane_z(top, layer_z, z_pitch, offset=pad_z_offset)
    z_sink = _plane_z(bot, layer_z, z_pitch, offset=sink_z_offset)
    # Bottom node slightly below the sink plane for visual separation
    z_bot = _plane_z(bot, layer_z, z_pitch, offset=bot_z_offset)
    if abs(z_bot - z_sink) < 1e-15:
        z_bot = z_sink - dz

    pitch = float(ports.cell_size)
    half = 0.5 * pitch
    cells = ports.cells
    centers: List[Tuple[float, float]] = [
        _cell_center_xy(ports, c.ix, c.iy) for c in cells
    ]

    sinks: List[Point3D] = []
    sink_meta: List[Dict[str, Any]] = []
    for k, c in enumerate(cells):
        cx, cy = centers[k]
        sinks.append((cx, cy, z_sink))
        sink_meta.append(
            {
                "ix": int(c.ix),
                "iy": int(c.iy),
                "current": float(c.current),
                "node": str(c.node),
                "k": k,
            }
        )

    pads: List[Point3D] = []
    pad_meta: List[Dict[str, Any]] = []
    pad_xy_draw: List[Tuple[float, float]] = []
    n_attach = min(len(pad_attach), len(ports.pad_nodes), len(cells))
    locs = ports.pad_locations()
    use_stub = str(getattr(ports, "via_stub", "")) == VIA_STUB_RXRY
    for k in range(n_attach):
        sink_k = int(pad_attach[k])
        if sink_k < 0 or sink_k >= len(cells):
            continue
        cx, cy = centers[sink_k]
        if use_stub and k < len(locs):
            px, py = locs[k]
        else:
            px, py = cx, cy
        pad_xy_draw.append((px, py))
        pads.append((px, py, z_pad))
        pad_meta.append(
            {
                "ix": int(cells[sink_k].ix),
                "iy": int(cells[sink_k].iy),
                "pad_node": str(ports.pad_nodes[k]),
                "sink_k": sink_k,
                "voltage": float(ports.pad_voltages[k])
                if k < len(ports.pad_voltages)
                else None,
            }
        )

    rx_edges: List[Seg3D] = []
    ry_edges: List[Seg3D] = []

    # Shared E–W: two half-arms meeting at midpoint
    for pair in ew_shared:
        a, b = int(pair[0]), int(pair[1])
        if not (0 <= a < len(cells) and 0 <= b < len(cells)):
            continue
        ax, ay = centers[a]
        bx, by = centers[b]
        mx, my = 0.5 * (ax + bx), 0.5 * (ay + by)
        rx_edges.append((ax, ay, z_sink, mx, my, z_sink))
        rx_edges.append((bx, by, z_sink, mx, my, z_sink))

    # Shared N–S
    for pair in ns_shared:
        a, b = int(pair[0]), int(pair[1])
        if not (0 <= a < len(cells) and 0 <= b < len(cells)):
            continue
        ax, ay = centers[a]
        bx, by = centers[b]
        mx, my = 0.5 * (ax + bx), 0.5 * (ay + by)
        ry_edges.append((ax, ay, z_sink, mx, my, z_sink))
        ry_edges.append((bx, by, z_sink, mx, my, z_sink))

    # Boundary dangling half-arms to cell-edge midpoints
    side_delta = {
        "W": (-half, 0.0),
        "E": (half, 0.0),
        "S": (0.0, -half),
        "N": (0.0, half),
    }
    for item in ew_boundary or []:
        sink_k, side = int(item[0]), str(item[1])
        if not (0 <= sink_k < len(cells)) or side not in ("W", "E"):
            continue
        cx, cy = centers[sink_k]
        dx, dy = side_delta[side]
        rx_edges.append((cx, cy, z_sink, cx + dx, cy + dy, z_sink))

    for item in ns_boundary or []:
        sink_k, side = int(item[0]), str(item[1])
        if not (0 <= sink_k < len(cells)) or side not in ("S", "N"):
            continue
        cx, cy = centers[sink_k]
        dx, dy = side_delta[side]
        ry_edges.append((cx, cy, z_sink, cx + dx, cy + dy, z_sink))

    # Rup + Rdown per attached pad/sink; optional L-bend stubs at pad Z
    rz_edges: List[Seg3D] = []
    for k in range(n_attach):
        sink_k = int(pad_attach[k])
        if sink_k < 0 or sink_k >= len(cells):
            continue
        cx, cy = centers[sink_k]
        px, py = pad_xy_draw[k] if k < len(pad_xy_draw) else (cx, cy)
        if use_stub and (abs(px - cx) > 1e-12 or abs(py - cy) > 1e-12):
            if abs(px - cx) > 1e-12:
                rx_edges.append((px, py, z_pad, cx, py, z_pad))
            if abs(py - cy) > 1e-12:
                ry_edges.append((cx, py, z_pad, cx, cy, z_pad))
        rz_edges.append((cx, cy, z_pad, cx, cy, z_sink))  # Rup
        rz_edges.append((cx, cy, z_sink, cx, cy, z_bot))  # Rdown

    return PixelGeometry3D(
        pads=pads,
        sinks=sinks,
        rx_edges=rx_edges,
        ry_edges=ry_edges,
        rz_edges=rz_edges,
        sink_meta=sink_meta,
        pad_meta=pad_meta,
        Rx=Rx,
        Ry=Ry,
        Rz=Rz,
        z_pad=z_pad,
        z_sink=z_sink,
        z_bot=z_bot,
    )


def points_to_xyz(
    pts: List[Point3D],
) -> Tuple[List[float], List[float], List[float]]:
    if not pts:
        return [], [], []
    xs, ys, zs = zip(*pts)
    return list(xs), list(ys), list(zs)
