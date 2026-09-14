"""Nearest-grid stub attach (L-bend / floor stub) for real port XY.

Shared by stagger / tri-square. Stub lengths are stamped
later as R = Rx*(|dx|/pitch) or Ry*(|dy|/pitch).

``via_stub``:
  ``rxry`` (default) — landing at port XY + axis-aligned L-bend to nearest grid.
  ``zero`` — via lands on the nearest grid node (offset treated as a short).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]

VIA_STUB_ZERO = "zero"
VIA_STUB_RXRY = "rxry"
VIA_STUB_MODES = (VIA_STUB_ZERO, VIA_STUB_RXRY)


def parse_via_stub_arg(raw: Optional[str]) -> str:
    """Parse ``--via-stub``: ``rxry`` (default L-bend) or ``zero`` (snap)."""
    if raw is None:
        return VIA_STUB_RXRY
    s = str(raw).strip().lower()
    if s in ("", VIA_STUB_RXRY, "nonzero", "non-zero", "on", "true", "yes", "1", "r"):
        return VIA_STUB_RXRY
    if s in (VIA_STUB_ZERO, "0", "off", "false", "no", "none", "short"):
        return VIA_STUB_ZERO
    raise ValueError(f"via_stub must be 'zero' or 'rxry' (got {raw!r})")


@dataclass
class StubGraph:
    """Square (or rectangular) mesh + port landings + stub edges."""

    xy: List[Point] = field(default_factory=list)
    n_grid: int = 0
    nx: int = 0
    ny: int = 0
    ew_edges: List[Tuple[int, int]] = field(default_factory=list)
    ew_lengths: List[float] = field(default_factory=list)
    ns_edges: List[Tuple[int, int]] = field(default_factory=list)
    ns_lengths: List[float] = field(default_factory=list)
    pad_landing: List[int] = field(default_factory=list)
    sink_landing: List[int] = field(default_factory=list)
    pitch: float = 0.0
    via_stub: str = VIA_STUB_RXRY


def _xy_key(x: float, y: float, tol: float) -> Tuple[int, int]:
    return (int(round(x / tol)), int(round(y / tol)))


def build_stub_square_graph(
    pad_xy: Sequence[Point],
    sink_xy: Sequence[Point],
    pitch: float,
    *,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    grid_centers: Optional[Sequence[Point]] = None,
    nx: Optional[int] = None,
    ny: Optional[int] = None,
    eps_scale: float = 0.05,
    floor_scale: float = 0.25,
    square_mesh: bool = True,
    via_stub: str = VIA_STUB_RXRY,
) -> StubGraph:
    """Build mesh + pad/sink attach.

    ``via_stub=rxry`` (default): distinct landings at port XY with L-bend stubs.
    ``via_stub=zero``: via attaches to the nearest grid node (no extra stub R).

    If ``square_mesh`` is False, only stub edges are added (no full grid E–W/N–S);
    ``grid_centers`` must still be provided as the attach targets.
    """
    from ports_dual import _mesh_centers

    if not pad_xy and not sink_xy:
        raise ValueError("need at least one pad or sink")
    pitch = float(pitch)
    if pitch <= 0:
        raise ValueError("pitch must be positive")
    eps = max(pitch * float(eps_scale), 1e-9)
    l_floor = max(pitch * float(floor_scale), eps)

    if grid_centers is not None and nx is not None and ny is not None:
        centers = [(float(x), float(y)) for x, y in grid_centers]
        nx_g, ny_g = int(nx), int(ny)
        if len(centers) != nx_g * ny_g:
            raise ValueError(
                f"grid_centers length {len(centers)} != nx*ny={nx_g * ny_g}"
            )
    else:
        if bbox is None:
            pts = list(pad_xy) + list(sink_xy)
            if not pts:
                raise ValueError("need pad/sink XY or bbox/grid_centers")
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            pad = 0.5 * pitch
            bbox = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
        nx_g, ny_g, centers, _ids = _mesh_centers(bbox, pitch)

    n_grid = len(centers)
    xy: List[Point] = list(centers)
    key_to_idx: Dict[Tuple[int, int], int] = {
        _xy_key(x, y, eps): i for i, (x, y) in enumerate(xy)
    }

    def add_node(x: float, y: float, *, force_new: bool = False) -> int:
        if force_new:
            xy.append((float(x), float(y)))
            return len(xy) - 1
        k = _xy_key(x, y, eps)
        if k in key_to_idx:
            return key_to_idx[k]
        idx = len(xy)
        xy.append((float(x), float(y)))
        key_to_idx[k] = idx
        return idx

    ew_edges: List[Tuple[int, int]] = []
    ew_lengths: List[float] = []
    ns_edges: List[Tuple[int, int]] = []
    ns_lengths: List[float] = []
    seen_ew: set = set()
    seen_ns: set = set()

    def add_ew(a: int, b: int, length: float) -> None:
        if a == b or length <= 0:
            return
        key = (min(a, b), max(a, b))
        if key in seen_ew:
            return
        seen_ew.add(key)
        ew_edges.append((a, b))
        ew_lengths.append(float(length))

    def add_ns(a: int, b: int, length: float) -> None:
        if a == b or length <= 0:
            return
        key = (min(a, b), max(a, b))
        if key in seen_ns:
            return
        seen_ns.add(key)
        ns_edges.append((a, b))
        ns_lengths.append(float(length))

    if square_mesh:
        for iy in range(ny_g):
            for ix in range(nx_g - 1):
                a = iy * nx_g + ix
                b = a + 1
                ax, _ay = xy[a]
                bx, _by = xy[b]
                add_ew(a, b, max(abs(bx - ax), eps))
        for iy in range(ny_g - 1):
            for ix in range(nx_g):
                a = iy * nx_g + ix
                b = a + nx_g
                _ax, ay = xy[a]
                _bx, by = xy[b]
                add_ns(a, b, max(abs(by - ay), eps))

    # Bucket grid nodes for O(1)-ish nearest queries (dense meshes).
    _inv = 1.0 / pitch
    _buckets: Dict[Tuple[int, int], List[int]] = {}
    for i in range(n_grid):
        gx, gy = xy[i]
        bk = (int(math.floor(gx * _inv)), int(math.floor(gy * _inv)))
        _buckets.setdefault(bk, []).append(i)

    def nearest_grid(x: float, y: float) -> int:
        cx, cy = int(math.floor(x * _inv)), int(math.floor(y * _inv))
        best_i, best_d = 0, float("inf")
        found = False
        for rad in range(0, 8):
            for dx in range(-rad, rad + 1):
                for dy in range(-rad, rad + 1):
                    if rad > 0 and abs(dx) != rad and abs(dy) != rad:
                        continue
                    for i in _buckets.get((cx + dx, cy + dy), ()):
                        gx, gy = xy[i]
                        d = (gx - x) * (gx - x) + (gy - y) * (gy - y)
                        if d < best_d:
                            best_d = d
                            best_i = i
                            found = True
            if found:
                return best_i
        # Fallback: full scan (should be rare)
        for i in range(n_grid):
            gx, gy = xy[i]
            d = (gx - x) * (gx - x) + (gy - y) * (gy - y)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def connect_landing(landing: int, gidx: int) -> None:
        lx, ly = xy[landing]
        gx, gy = xy[gidx]
        dx, dy = abs(lx - gx), abs(ly - gy)
        if dx < eps and dy < eps:
            add_ew(landing, gidx, l_floor)
            return
        if dy < eps:
            add_ew(landing, gidx, max(dx, eps))
            return
        if dx < eps:
            add_ns(landing, gidx, max(dy, eps))
            return
        bend = add_node(lx, gy)
        add_ns(landing, bend, dy)
        add_ew(bend, gidx, dx)

    use_stub = parse_via_stub_arg(via_stub) == VIA_STUB_RXRY

    pad_landing: List[int] = []
    for x, y in pad_xy:
        gidx = nearest_grid(float(x), float(y))
        if use_stub:
            landing = add_node(float(x), float(y), force_new=True)
            pad_landing.append(landing)
            connect_landing(landing, gidx)
        else:
            pad_landing.append(gidx)

    sink_landing: List[int] = []
    for x, y in sink_xy:
        gidx = nearest_grid(float(x), float(y))
        if use_stub:
            landing = add_node(float(x), float(y), force_new=True)
            sink_landing.append(landing)
            connect_landing(landing, gidx)
        else:
            sink_landing.append(gidx)

    return StubGraph(
        xy=xy,
        n_grid=n_grid,
        nx=nx_g,
        ny=ny_g,
        ew_edges=ew_edges,
        ew_lengths=ew_lengths,
        ns_edges=ns_edges,
        ns_lengths=ns_lengths,
        pad_landing=pad_landing,
        sink_landing=sink_landing,
        pitch=pitch,
        via_stub=parse_via_stub_arg(via_stub),
    )


def conductances_from_rxryrz(
    ew_lengths: Sequence[float],
    ns_lengths: Sequence[float],
    pitch: float,
    Rx: float,
    Ry: float,
    Rz: float,
    n_via_pad: int,
    n_via_sink: int,
    *,
    floor_r: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stamp length-proportional lateral G and uniform via G from globals."""
    pitch = max(float(pitch), 1e-30)
    Rx = max(float(Rx), floor_r)
    Ry = max(float(Ry), floor_r)
    Rz = max(float(Rz), floor_r)
    lew = np.asarray(ew_lengths, dtype=float)
    lns = np.asarray(ns_lengths, dtype=float)
    R_ew = np.maximum(Rx * (lew / pitch), floor_r)
    R_ns = np.maximum(Ry * (lns / pitch), floor_r)
    G_ew = 1.0 / R_ew
    G_ns = 1.0 / R_ns
    G_via_pad = np.full(int(n_via_pad), 1.0 / Rz, dtype=float)
    G_via_sink = np.full(int(n_via_sink), 1.0 / Rz, dtype=float)
    return G_ew, G_ns, G_via_pad, G_via_sink
