"""Drawable 3D geometry for Zhang orthogonal dual-layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from _bootstrap import ensure_paths

ensure_paths()

from pdn_geometry_3d import Seg3D, z_of  # noqa: E402

Point3D = Tuple[float, float, float]


@dataclass
class DualGeometry3D:
    pads: List[Point3D] = field(default_factory=list)
    sinks: List[Point3D] = field(default_factory=list)
    rx_edges: List[Seg3D] = field(default_factory=list)  # top E–W (Rtop)
    ry_edges: List[Seg3D] = field(default_factory=list)  # bot N–S (Rbottom)
    rvia_edges: List[Seg3D] = field(default_factory=list)
    sink_meta: List[Dict[str, Any]] = field(default_factory=list)
    pad_meta: List[Dict[str, Any]] = field(default_factory=list)
    Rx: Optional[float] = None  # mean Rtop (diagnostic)
    Ry: Optional[float] = None  # mean Rbottom (diagnostic)
    Rvia: Optional[float] = None  # mean Rvia (diagnostic)
    # Per-edge R stats: {"min","median","max","mean","n"} for ew/ns/via
    R_edge_stats: Optional[Dict[str, Any]] = None
    # Optional named legend groups (tri-square: 6 R params). Each entry:
    # {"name", "kind", "color", "segs", "r_label"}
    legend_groups: Optional[List[Dict[str, Any]]] = None
    z_pad: float = 0.0
    z_sink: float = 0.0

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
    def n_rvia(self) -> int:
        return len(self.rvia_edges)

    @property
    def rz_edges(self) -> List[Seg3D]:
        return self.rvia_edges

    @property
    def n_rz(self) -> int:
        return self.n_rvia

    @property
    def Rz(self) -> Optional[float]:
        return self.Rvia


def build_dual_geometry_3d(
    *,
    top_centers: Sequence[Tuple[float, float]],
    bot_centers: Sequence[Tuple[float, float]],
    ew_top: Sequence[Sequence[int]],
    ns_bot: Sequence[Sequence[int]],
    vias: Sequence[Sequence[int]],
    pad_xy: Sequence[Tuple[float, float]],
    sink_xy: Sequence[Tuple[float, float]],
    pad_mesh: Sequence[int],
    sink_mesh: Sequence[int],
    layer_z: Dict[int, float],
    z_pitch: float,
    top_layer: int,
    bot_layer: int,
    Rx: Optional[float] = None,
    Ry: Optional[float] = None,
    Rvia: Optional[float] = None,
    R_edge_stats: Optional[Dict[str, Any]] = None,
) -> DualGeometry3D:
    dz = 0.05 * float(z_pitch)
    z_pad = z_of(int(top_layer), layer_z, z_pitch) + dz
    z_sink = z_of(int(bot_layer), layer_z, z_pitch)

    pads: List[Point3D] = []
    pad_meta: List[Dict[str, Any]] = []
    for k, ((x, y), ti) in enumerate(zip(pad_xy, pad_mesh)):
        # Draw pad at original xy, slightly above its mesh node
        pads.append((float(x), float(y), z_pad))
        tx, ty = top_centers[int(ti)]
        pad_meta.append({"k": k, "mesh": int(ti), "x": x, "y": y, "mx": tx, "my": ty})

    sinks: List[Point3D] = []
    sink_meta: List[Dict[str, Any]] = []
    for k, ((x, y), bj) in enumerate(zip(sink_xy, sink_mesh)):
        sinks.append((float(x), float(y), z_sink))
        bx, by = bot_centers[int(bj)]
        sink_meta.append({"k": k, "mesh": int(bj), "x": x, "y": y, "mx": bx, "my": by})

    rx_edges: List[Seg3D] = []
    for a, b in ew_top:
        ax, ay = top_centers[int(a)]
        bx, by = top_centers[int(b)]
        rx_edges.append((ax, ay, z_pad, bx, by, z_pad))

    ry_edges: List[Seg3D] = []
    for a, b in ns_bot:
        ax, ay = bot_centers[int(a)]
        bx, by = bot_centers[int(b)]
        ry_edges.append((ax, ay, z_sink, bx, by, z_sink))

    rvia_edges: List[Seg3D] = []
    for ti, bj in vias:
        tx, ty = top_centers[int(ti)]
        bx, by = bot_centers[int(bj)]
        rvia_edges.append((tx, ty, z_pad, bx, by, z_sink))

    return DualGeometry3D(
        pads=pads,
        sinks=sinks,
        rx_edges=rx_edges,
        ry_edges=ry_edges,
        rvia_edges=rvia_edges,
        sink_meta=sink_meta,
        pad_meta=pad_meta,
        Rx=Rx,
        Ry=Ry,
        Rvia=Rvia,
        R_edge_stats=R_edge_stats,
        z_pad=z_pad,
        z_sink=z_sink,
    )


def build_tri_geometry_3d(
    *,
    centers: Sequence[Tuple[float, float]],
    ew_mid: Sequence[Sequence[int]],
    ns_mid: Sequence[Sequence[int]],
    vias_tm: Sequence[Sequence[int]],
    vias_mb: Sequence[Sequence[int]],
    pad_xy: Sequence[Tuple[float, float]],
    sink_xy: Sequence[Tuple[float, float]],
    pad_mesh: Sequence[int],
    sink_mesh: Sequence[int],
    layer_z: Dict[int, float],
    z_pitch: float,
    top_layer: int,
    bot_layer: int,
    Rx: Optional[float] = None,
    Ry: Optional[float] = None,
    Rvia: Optional[float] = None,
    R_edge_stats: Optional[Dict[str, Any]] = None,
) -> DualGeometry3D:
    """Map tri mid-square topology onto DualGeometry3D drawable (reuse plotter).

    Mid square sits at mid-z; vias are top↔mid and mid↔bot segments.
    """
    dz = 0.05 * float(z_pitch)
    z_pad = z_of(int(top_layer), layer_z, z_pitch) + dz
    z_sink = z_of(int(bot_layer), layer_z, z_pitch)
    z_mid = 0.5 * (z_pad + z_sink)

    pads: List[Point3D] = []
    pad_meta: List[Dict[str, Any]] = []
    for k, ((x, y), ti) in enumerate(zip(pad_xy, pad_mesh)):
        pads.append((float(x), float(y), z_pad))
        tx, ty = centers[int(ti)]
        pad_meta.append({"k": k, "mesh": int(ti), "x": x, "y": y, "mx": tx, "my": ty})

    sinks: List[Point3D] = []
    sink_meta: List[Dict[str, Any]] = []
    for k, ((x, y), bj) in enumerate(zip(sink_xy, sink_mesh)):
        sinks.append((float(x), float(y), z_sink))
        bx, by = centers[int(bj)]
        sink_meta.append({"k": k, "mesh": int(bj), "x": x, "y": y, "mx": bx, "my": by})

    rx_edges: List[Seg3D] = []
    for a, b in ew_mid:
        ax, ay = centers[int(a)]
        bx, by = centers[int(b)]
        rx_edges.append((ax, ay, z_mid, bx, by, z_mid))

    ry_edges: List[Seg3D] = []
    for a, b in ns_mid:
        ax, ay = centers[int(a)]
        bx, by = centers[int(b)]
        ry_edges.append((ax, ay, z_mid, bx, by, z_mid))

    rvia_edges: List[Seg3D] = []
    for ti, mi in vias_tm:
        tx, ty = centers[int(ti)]
        mx, my = centers[int(mi)]
        rvia_edges.append((tx, ty, z_pad, mx, my, z_mid))
    for mi, bj in vias_mb:
        mx, my = centers[int(mi)]
        bx, by = centers[int(bj)]
        rvia_edges.append((mx, my, z_mid, bx, by, z_sink))

    return DualGeometry3D(
        pads=pads,
        sinks=sinks,
        rx_edges=rx_edges,
        ry_edges=ry_edges,
        rvia_edges=rvia_edges,
        sink_meta=sink_meta,
        pad_meta=pad_meta,
        Rx=Rx,
        Ry=Ry,
        Rvia=Rvia,
        R_edge_stats=R_edge_stats,
        z_pad=z_pad,
        z_sink=z_sink,
    )


def build_tri_stagger_geometry_3d(
    *,
    mid_xy: Sequence[Tuple[float, float]],
    ew_edges: Sequence[Sequence[int]],
    ns_edges: Sequence[Sequence[int]],
    pad_mid: Sequence[int],
    sink_mid: Sequence[int],
    pad_xy: Sequence[Tuple[float, float]],
    sink_xy: Sequence[Tuple[float, float]],
    layer_z: Dict[int, float],
    z_pitch: float,
    top_layer: int,
    bot_layer: int,
    Rx: Optional[float] = None,
    Ry: Optional[float] = None,
    Rvia: Optional[float] = None,
    R_edge_stats: Optional[Dict[str, Any]] = None,
) -> DualGeometry3D:
    """Drawable geometry for staggered mid-layer (vias at port XY)."""
    dz = 0.05 * float(z_pitch)
    z_pad = z_of(int(top_layer), layer_z, z_pitch) + dz
    z_sink = z_of(int(bot_layer), layer_z, z_pitch)
    z_mid = 0.5 * (z_pad + z_sink)

    pads: List[Point3D] = []
    pad_meta: List[Dict[str, Any]] = []
    for k, (x, y) in enumerate(pad_xy):
        pads.append((float(x), float(y), z_pad))
        mi = int(pad_mid[k]) if k < len(pad_mid) else k
        mx, my = mid_xy[mi] if mi < len(mid_xy) else (x, y)
        pad_meta.append(
            {"k": k, "mesh": mi, "x": x, "y": y, "mx": mx, "my": my}
        )

    sinks: List[Point3D] = []
    sink_meta: List[Dict[str, Any]] = []
    for k, (x, y) in enumerate(sink_xy):
        sinks.append((float(x), float(y), z_sink))
        mi = int(sink_mid[k]) if k < len(sink_mid) else k
        mx, my = mid_xy[mi] if mi < len(mid_xy) else (x, y)
        sink_meta.append(
            {"k": k, "mesh": mi, "x": x, "y": y, "mx": mx, "my": my}
        )

    rx_edges: List[Seg3D] = []
    for a, b in ew_edges:
        ax, ay = mid_xy[int(a)]
        bx, by = mid_xy[int(b)]
        rx_edges.append((ax, ay, z_mid, bx, by, z_mid))

    ry_edges: List[Seg3D] = []
    for a, b in ns_edges:
        ax, ay = mid_xy[int(a)]
        bx, by = mid_xy[int(b)]
        ry_edges.append((ax, ay, z_mid, bx, by, z_mid))

    rvia_edges: List[Seg3D] = []
    for k, (x, y) in enumerate(pad_xy):
        mi = int(pad_mid[k])
        mx, my = mid_xy[mi]
        # Vertical via from pad XY down to mid (same XY as pad landing).
        rvia_edges.append((float(x), float(y), z_pad, float(x), float(y), z_mid))
        # If mid node was offset (floor stub), add short mid hop visually via rx;
        # still draw via at pad XY. Optional link pad mid offset:
        if abs(mx - x) > 1e-9 or abs(my - y) > 1e-9:
            rvia_edges.append((float(x), float(y), z_mid, float(mx), float(my), z_mid))
    for k, (x, y) in enumerate(sink_xy):
        mi = int(sink_mid[k])
        mx, my = mid_xy[mi]
        rvia_edges.append((float(x), float(y), z_mid, float(x), float(y), z_sink))
        if abs(mx - x) > 1e-9 or abs(my - y) > 1e-9:
            rvia_edges.append((float(mx), float(my), z_mid, float(x), float(y), z_mid))

    return DualGeometry3D(
        pads=pads,
        sinks=sinks,
        rx_edges=rx_edges,
        ry_edges=ry_edges,
        rvia_edges=rvia_edges,
        sink_meta=sink_meta,
        pad_meta=pad_meta,
        Rx=Rx,
        Ry=Ry,
        Rvia=Rvia,
        R_edge_stats=R_edge_stats,
        z_pad=z_pad,
        z_sink=z_sink,
    )


def build_tri_square_geometry_3d(
    *,
    l2_xy: Sequence[Tuple[float, float]],
    l3_xy: Sequence[Tuple[float, float]],
    ew_u: Sequence[Sequence[int]],
    ns_u: Sequence[Sequence[int]],
    ew_l: Sequence[Sequence[int]],
    ns_l: Sequence[Sequence[int]],
    vias_ul: Sequence[Sequence[float]],
    pad_attach: Sequence[Sequence[Sequence[float]]],
    sink_attach: Sequence[Sequence[Sequence[float]]],
    pad_xy: Sequence[Tuple[float, float]],
    sink_xy: Sequence[Tuple[float, float]],
    layer_z: Dict[int, float],
    z_pitch: float,
    top_layer: int,
    bot_layer: int,
    Rx: Optional[float] = None,
    Ry: Optional[float] = None,
    Rvia: Optional[float] = None,
    R_edge_stats: Optional[Dict[str, Any]] = None,
) -> DualGeometry3D:
    """Drawable geometry for tri-square concept (bump → mid → bot → sink).

    Three clearly separated planes:
      z_pad  — bumps / voltage sources only (no grid)
      z_l2   — coarse mid square grid
      z_l3   — fine bot square grid + sinks

    Draws:
      - 1st via: vertical bump XY → mid-plane landing (same XY, usually inside a cell)
      - bilinear arms on the mid plane: landing → surrounding L2 nodes
      - 2nd via: vertical pillars from each L2 node down to the bot plane
      - sinks on the bot plane at real XY
    """
    z_top = z_of(int(top_layer), layer_z, z_pitch)
    z_bot = z_of(int(bot_layer), layer_z, z_pitch)
    # Force three well-separated planes even if metal z's are close.
    z_lo = min(float(z_top), float(z_bot))
    z_hi = max(float(z_top), float(z_bot))
    span = max(z_hi - z_lo, float(z_pitch), 1.0)
    z_pad = z_hi + 0.35 * span  # bumps clearly above mid grid
    z_l2 = z_hi - 0.05 * span  # mid grid
    z_l3 = z_lo  # bot grid + sinks
    z_sink = z_l3

    pads: List[Point3D] = []
    pad_meta: List[Dict[str, Any]] = []
    for k, (x, y) in enumerate(pad_xy):
        pads.append((float(x), float(y), z_pad))
        arms = pad_attach[k] if k < len(pad_attach) else []
        best = max(arms, key=lambda aw: float(aw[1]), default=None)
        mx, my = (x, y)
        mi = -1
        if best is not None:
            mi = int(best[0])
            if 0 <= mi < len(l2_xy):
                mx, my = l2_xy[mi]
        pad_meta.append(
            {"k": k, "mesh": mi, "x": x, "y": y, "mx": mx, "my": my}
        )

    sinks: List[Point3D] = []
    sink_meta: List[Dict[str, Any]] = []
    for k, (x, y) in enumerate(sink_xy):
        sinks.append((float(x), float(y), z_l3))
        arms = sink_attach[k] if k < len(sink_attach) else []
        best = max(arms, key=lambda aw: float(aw[1]), default=None)
        mx, my = (x, y)
        mi = -1
        if best is not None:
            mi = int(best[0])
            if 0 <= mi < len(l3_xy):
                mx, my = l3_xy[mi]
        sink_meta.append(
            {"k": k, "mesh": mi, "x": x, "y": y, "mx": mx, "my": my}
        )

    # Mid (coarse) at z_l2; bot (fine) at z_l3 — kept as separate legend groups.
    rx_u_edges: List[Seg3D] = []
    rx_l_edges: List[Seg3D] = []
    for a, b in ew_u:
        ax, ay = l2_xy[int(a)]
        bx, by = l2_xy[int(b)]
        rx_u_edges.append((ax, ay, z_l2, bx, by, z_l2))
    for a, b in ew_l:
        ax, ay = l3_xy[int(a)]
        bx, by = l3_xy[int(b)]
        rx_l_edges.append((ax, ay, z_l3, bx, by, z_l3))

    ry_u_edges: List[Seg3D] = []
    ry_l_edges: List[Seg3D] = []
    for a, b in ns_u:
        ax, ay = l2_xy[int(a)]
        bx, by = l2_xy[int(b)]
        ry_u_edges.append((ax, ay, z_l2, bx, by, z_l2))
    for a, b in ns_l:
        ax, ay = l3_xy[int(a)]
        bx, by = l3_xy[int(b)]
        ry_l_edges.append((ax, ay, z_l3, bx, by, z_l3))

    via1_edges: List[Seg3D] = []  # bump → mid (+ bilinear arms)
    via2_edges: List[Seg3D] = []  # mid → bot (+ sink attach)
    for k, (x, y) in enumerate(pad_xy):
        px, py = float(x), float(y)
        via1_edges.append((px, py, z_pad, px, py, z_l2))
        arms = pad_attach[k] if k < len(pad_attach) else []
        for entry in arms:
            idx = int(entry[0])
            if 0 <= idx < len(l2_xy):
                mx, my = l2_xy[idx]
                if abs(mx - px) > 1e-12 or abs(my - py) > 1e-12:
                    via1_edges.append((px, py, z_l2, float(mx), float(my), z_l2))
    seen_l2: set[int] = set()
    for row in vias_ul:
        u = int(row[0])
        if u in seen_l2 or not (0 <= u < len(l2_xy)):
            continue
        seen_l2.add(u)
        ux, uy = l2_xy[u]
        via2_edges.append((ux, uy, z_l2, ux, uy, z_l3))
    for k, (x, y) in enumerate(sink_xy):
        arms = sink_attach[k] if k < len(sink_attach) else []
        for entry in arms:
            idx = int(entry[0])
            if 0 <= idx < len(l3_xy):
                mx, my = l3_xy[idx]
                if abs(mx - x) > 1e-12 or abs(my - y) > 1e-12:
                    via2_edges.append(
                        (float(mx), float(my), z_l3, float(x), float(y), z_l3)
                    )

    rx_edges = rx_u_edges + rx_l_edges
    ry_edges = ry_u_edges + ry_l_edges
    rvia_edges = via1_edges + via2_edges

    def _r_label(keys: Sequence[str], fallback: Optional[float] = None) -> str:
        stats = R_edge_stats or {}
        for key in keys:
            s = stats.get(key) or {}
            if s.get("mean") is not None:
                return f"R={float(s['mean']):.4g}"
            if s.get("median") is not None:
                return f"R={float(s['median']):.4g}"
        if fallback is not None:
            return f"R={float(fallback):.4g}"
        return ""

    # Six R parameters: mid rx/ry, bot rx/ry, via1 (bump→mid), via2 (mid→bot).
    legend_groups: List[Dict[str, Any]] = [
        {
            "name": "Mid Rx",
            "kind": "rx_u",
            "color": "#0284c7",
            "segs": rx_u_edges,
            "r_label": _r_label(("ew_u", "rx_u"), None),
        },
        {
            "name": "Mid Ry",
            "kind": "ry_u",
            "color": "#7c3aed",
            "segs": ry_u_edges,
            "r_label": _r_label(("ns_u", "ry_u"), None),
        },
        {
            "name": "Bot Rx",
            "kind": "rx_l",
            "color": "#38bdf8",
            "segs": rx_l_edges,
            "r_label": _r_label(("ew_l", "rx_l"), None),
        },
        {
            "name": "Bot Ry",
            "kind": "ry_l",
            "color": "#a78bfa",
            "segs": ry_l_edges,
            "r_label": _r_label(("ns_l", "ry_l"), None),
        },
        {
            "name": "Via₁ bump→mid",
            "kind": "via_pad",
            "color": "#ea580c",
            "segs": via1_edges,
            "r_label": _r_label(("via_pad",), None),
        },
        {
            "name": "Via₂ mid→bot",
            "kind": "via_ul",
            "color": "#1e293b",
            "segs": via2_edges,
            "r_label": _r_label(("via_ul",), None),
        },
    ]

    return DualGeometry3D(
        pads=pads,
        sinks=sinks,
        rx_edges=rx_edges,
        ry_edges=ry_edges,
        rvia_edges=rvia_edges,
        sink_meta=sink_meta,
        pad_meta=pad_meta,
        Rx=Rx,
        Ry=Ry,
        Rvia=Rvia,
        R_edge_stats=R_edge_stats,
        legend_groups=legend_groups,
        z_pad=z_pad,
        z_sink=z_sink,
    )


def points_to_xyz(
    pts: List[Point3D],
) -> Tuple[List[float], List[float], List[float]]:
    if not pts:
        return [], [], []
    xs, ys, zs = zip(*pts)
    return list(xs), list(ys), list(zs)
