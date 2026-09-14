"""Classify VDD metal layers by preferred wire orientation (Zhang Fig. 4).

x-directed (horizontal, dy≈0) layers collapse onto the dual-layer top sheet;
y-directed (vertical, dx≈0) layers collapse onto the bottom sheet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from _bootstrap import ensure_paths

ensure_paths()

from graph import _root_of, vdd_layers  # noqa: E402
from spice_parser import SpiceNetlist  # noqa: E402


@dataclass
class LayerOrientation:
    """Per-layer orientation and sheet assignment."""

    layer: int
    n_horiz: int  # same-layer R with dy≈0 (x-directed)
    n_vert: int  # same-layer R with dx≈0 (y-directed)
    orientation: str  # "x" | "y" | "unknown"
    sheet: str  # "top" (x) | "bot" (y) | "none"


def classify_layer_orientation(
    net: SpiceNetlist,
    layer: int,
    node_map: Optional[Dict[str, str]] = None,
    roots: Optional[Set[str]] = None,
) -> LayerOrientation:
    """Count same-layer resistor directions and pick majority orientation."""
    if node_map is None:
        node_map = {n: n for n in net.node_names}
    coords = net.coords
    horiz = 0
    vert = 0
    for n1, n2, _r in net.resistors:
        c1, c2 = coords.get(n1), coords.get(n2)
        if c1 is None or c2 is None:
            continue
        if c1.layer != layer or c2.layer != layer:
            continue
        if roots is not None:
            r1 = _root_of(node_map, n1)
            r2 = _root_of(node_map, n2)
            if r1 not in roots and r2 not in roots:
                continue
        dx = abs(c1.x - c2.x)
        dy = abs(c1.y - c2.y)
        if dx < 1e-9 and dy > 1e-9:
            vert += 1
        elif dy < 1e-9 and dx > 1e-9:
            horiz += 1

    if horiz == 0 and vert == 0:
        orient = "unknown"
        sheet = "none"
    elif horiz >= vert:
        orient = "x"
        sheet = "top"
    else:
        orient = "y"
        sheet = "bot"
    return LayerOrientation(
        layer=int(layer),
        n_horiz=int(horiz),
        n_vert=int(vert),
        orientation=orient,
        sheet=sheet,
    )


def classify_vdd_layers(
    net: SpiceNetlist,
    node_map: Dict[str, str],
    roots: Optional[Set[str]] = None,
    layers: Optional[Sequence[int]] = None,
) -> List[LayerOrientation]:
    """Classify every VDD metal layer; return list sorted by layer id."""
    if layers is None:
        layers = vdd_layers(net, node_map, roots)
    return [
        classify_layer_orientation(net, int(L), node_map=node_map, roots=roots)
        for L in layers
    ]


def sheet_layer_sets(
    orientations: Sequence[LayerOrientation],
) -> Tuple[List[int], List[int]]:
    """
    Return (x_layers → top sheet, y_layers → bot sheet).

    Layers with unknown orientation are omitted (no lateral contribution).
    """
    x_layers = [o.layer for o in orientations if o.sheet == "top"]
    y_layers = [o.layer for o in orientations if o.sheet == "bot"]
    return x_layers, y_layers


def orientation_summary(
    orientations: Sequence[LayerOrientation],
) -> Dict[str, object]:
    x_layers, y_layers = sheet_layer_sets(orientations)
    return {
        "layers": [
            {
                "layer": o.layer,
                "n_horiz": o.n_horiz,
                "n_vert": o.n_vert,
                "orientation": o.orientation,
                "sheet": o.sheet,
            }
            for o in orientations
        ],
        "x_layers": x_layers,
        "y_layers": y_layers,
    }
