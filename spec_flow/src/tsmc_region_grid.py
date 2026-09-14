"""TSMC region-port lattice: index → micron map and virtual Pixel-R detection.

Region ``x_*/y_*`` tokens in node names are **grid indices**, not microns.
TC1 default lattice (user-provided corners):

    x_1_y_1  → (30, 30)
    x_43_y_1 → (2550, 30)
    x_43_y_33 → (2550, 1950)
    x_1_y_33 → (30, 1950)

→ origin (30, 30), pitch 60, nx=43, ny=33.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from spice_parser import REGION_RE, SpiceNetlist, _normalize_node_token

# Default TC1 geometry (layout units / microns).
DEFAULT_ORIGIN_X = 30.0
DEFAULT_ORIGIN_Y = 30.0
DEFAULT_PITCH = 60.0
DEFAULT_NX = 43
DEFAULT_NY = 33

# Minimum filled rectangle size to take the virtual Pixel-R path (smoke decks
# with 1–2 region ports keep the legacy IBM-style port builder).
MIN_LATTICE_CELLS = 4

VPAD_RE = re.compile(
    r"^vpad_x_(?P<ix>\d+)_y_(?P<iy>\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RegionLattice:
    """Affine map from 1-based region indices to microns."""

    origin_x: float = DEFAULT_ORIGIN_X
    origin_y: float = DEFAULT_ORIGIN_Y
    pitch: float = DEFAULT_PITCH
    nx: int = DEFAULT_NX
    ny: int = DEFAULT_NY

    def index_to_xy(self, ix: int, iy: int) -> Tuple[float, float]:
        if ix < 1 or iy < 1:
            raise ValueError(f"region indices must be >= 1 (got ix={ix}, iy={iy})")
        return (
            float(self.origin_x + (ix - 1) * self.pitch),
            float(self.origin_y + (iy - 1) * self.pitch),
        )

    def bbox(self) -> Tuple[float, float, float, float]:
        """Axis-aligned bbox covering index cells [1..nx] × [1..ny]."""
        x0, y0 = self.index_to_xy(1, 1)
        x1, y1 = self.index_to_xy(self.nx, self.ny)
        return (x0, y0, x1, y1)

    def bbox_for_cells(
        self, cells: Sequence[Tuple[int, int]]
    ) -> Tuple[float, float, float, float]:
        if not cells:
            return self.bbox()
        xs: List[float] = []
        ys: List[float] = []
        for ix, iy in cells:
            x, y = self.index_to_xy(ix, iy)
            xs.append(x)
            ys.append(y)
        return (min(xs), min(ys), max(xs), max(ys))


DEFAULT_LATTICE = RegionLattice()


def parse_region_indices(name: str) -> Optional[Tuple[int, int, str]]:
    """
    Return ``(ix, iy, net)`` for a region port name, else None.

    ``net`` is ``VDD`` or ``VSS``.
    """
    name = _normalize_node_token(name)
    m = REGION_RE.match(name)
    if not m:
        return None
    return int(float(m.group("x"))), int(float(m.group("y"))), m.group("net").upper()


def parse_tile_indices(name: str) -> Optional[Tuple[int, int, str]]:
    """Return ``(ix, iy, net)`` from tile ``_x_*_y_*`` tokens, else None."""
    from spice_parser import LOOSE_TSMC_RE, TILE_RE

    name = _normalize_node_token(name)
    # Bare tile name, or hierarchical ``Xdie1.tile…`` (match on last segment /
    # loose ``_net_VDD_x_*_y_*``).
    base = name.split(".")[-1]
    m = TILE_RE.match(base)
    if m:
        return int(float(m.group("x"))), int(float(m.group("y"))), m.group("net").upper()
    m = LOOSE_TSMC_RE.search(name)
    if m and "tile" in name.lower():
        return int(float(m.group("x"))), int(float(m.group("y"))), m.group("net").upper()
    return None


def vpad_name(ix: int, iy: int) -> str:
    return f"vpad_x_{ix}_y_{iy}"


def parse_vpad_indices(name: str) -> Optional[Tuple[int, int]]:
    m = VPAD_RE.match(_normalize_node_token(name))
    if not m:
        return None
    return int(m.group("ix")), int(m.group("iy"))


def collect_region_vdd_ports(
    names: Iterable[str],
) -> Dict[Tuple[int, int], str]:
    """Map (ix, iy) → region VDD_PORT node name (first wins)."""
    out: Dict[Tuple[int, int], str] = {}
    for raw in names:
        name = _normalize_node_token(str(raw))
        parsed = parse_region_indices(name)
        if parsed is None:
            continue
        ix, iy, net = parsed
        if net != "VDD":
            continue
        if not name.upper().endswith("_VDD_PORT"):
            continue
        key = (ix, iy)
        if key not in out:
            out[key] = name
    return out


def collect_vdd_bumps(
    net: SpiceNetlist,
) -> Dict[Tuple[int, int], str]:
    """
    Map bump-grid ``(bx, by)`` → tile node for VDD sites (first wins per cell).

    Same tile may occupy several cells; use ``tiles_to_bumps`` for the inverse.
    """
    out: Dict[Tuple[int, int], str] = {}
    for site in net.bumps:
        if site.net.upper() != "VDD":
            continue
        key = (int(site.bx), int(site.by))
        if key not in out:
            out[key] = site.node
    return out


def tiles_to_bumps(
    bumps: Dict[Tuple[int, int], str],
) -> Dict[str, List[Tuple[int, int]]]:
    """Inverse of ``collect_vdd_bumps``: tile → list of ``(bx, by)`` cells."""
    inv: Dict[str, List[Tuple[int, int]]] = {}
    for key, tile in bumps.items():
        inv.setdefault(tile, []).append(key)
    for cells in inv.values():
        cells.sort(key=lambda t: (t[1], t[0]))
    return inv


def bump_index_base(bumps: Dict[Tuple[int, int], str]) -> int:
    """
    Return 0 if any bump index is 0-based (``bx==0`` or ``by==0``), else 1.

    Real TC1 uses 0..42 × 0..32; smoke decks use 1-based ``BUMP_VDD_1_1``.
    """
    if not bumps:
        return 1
    for bx, by in bumps:
        if bx == 0 or by == 0:
            return 0
    return 1


def is_filled_rectangle(cells: Sequence[Tuple[int, int]]) -> bool:
    """True if ``cells`` exactly fill the axis-aligned index bounding box."""
    if not cells:
        return False
    uniq = sorted(set(cells))
    ixs = [c[0] for c in uniq]
    iys = [c[1] for c in uniq]
    ix0, ix1 = min(ixs), max(ixs)
    iy0, iy1 = min(iys), max(iys)
    expect = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
    if expect != len(uniq):
        return False
    have: Set[Tuple[int, int]] = set(uniq)
    for iy in range(iy0, iy1 + 1):
        for ix in range(ix0, ix1 + 1):
            if (ix, iy) not in have:
                return False
    return True


def lattice_from_cells(
    cells: Sequence[Tuple[int, int]],
    *,
    base: RegionLattice = DEFAULT_LATTICE,
) -> RegionLattice:
    """
    Lattice sized to the index bounding box of ``cells``.

    Pitch / origin stay at TC1 defaults so ``index_to_xy`` matches the
    user-provided micron corners for 1-based indices.
    """
    if not cells:
        return base
    ixs = [c[0] for c in cells]
    iys = [c[1] for c in cells]
    return RegionLattice(
        origin_x=base.origin_x,
        origin_y=base.origin_y,
        pitch=base.pitch,
        nx=max(ixs),
        ny=max(iys),
    )


def is_virtual_pixel_r_lattice(
    net: SpiceNetlist,
    *,
    min_cells: int = MIN_LATTICE_CELLS,
) -> bool:
    """
    Detect decks that should use the virtual single-layer Pixel-R path.

    Requires a filled rectangle of region VDD_PORT cells with at least
    ``min_cells`` sites (default 4; the shipped 2×2 TC1 smoke deck qualifies).
    """
    ports = collect_region_vdd_ports(net.node_names)
    if len(ports) < min_cells:
        return False
    return is_filled_rectangle(list(ports.keys()))


def apply_region_micron_coords(
    net: SpiceNetlist,
    lattice: RegionLattice = DEFAULT_LATTICE,
    *,
    remap_tiles_without_bump: bool = True,
    bump_nodes: Optional[Set[str]] = None,
) -> int:
    """
    Overwrite region (and optionally tile) coords with lattice microns.

    Returns the number of nodes updated. Tile nodes listed in ``bump_nodes``
    keep their bump overrides.
    """
    from spice_parser import LAYER_VDD, LAYER_VSS, NodeCoord, TILE_RE

    bump_nodes = bump_nodes or set()
    n_updated = 0

    # Region ports: always remap when indices parse.
    for name in list(net.node_names):
        parsed = parse_region_indices(name)
        if parsed is None:
            continue
        ix, iy, net_name = parsed
        try:
            x, y = lattice.index_to_xy(ix, iy)
        except ValueError:
            continue
        layer = LAYER_VDD if net_name == "VDD" else LAYER_VSS
        net.coords[name] = NodeCoord(layer=layer, x=x, y=y)
        n_updated += 1

    if not remap_tiles_without_bump:
        return n_updated

    for name in list(net.node_names):
        if name in bump_nodes:
            continue
        base = name.split(".")[-1]
        if not (TILE_RE.match(base) or "tile" in name.lower()):
            continue
        parsed = parse_tile_indices(name)
        if parsed is None:
            continue
        ix, iy, net_name = parsed
        try:
            x, y = lattice.index_to_xy(ix, iy)
        except ValueError:
            continue
        from spice_parser import LAYER_TILE

        layer = LAYER_TILE if net_name == "VDD" else LAYER_VSS
        net.coords[name] = NodeCoord(layer=layer, x=x, y=y)
        n_updated += 1

    return n_updated
