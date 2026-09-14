"""Deprecated: tri-square no longer seeds from Zhang / via-stack extract."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Set

from layer_orient import LayerOrientation
from ports_dual import DualPortSet
from spice_parser import SpiceNetlist
from tri_square import TriSquareModel


def extract_tri_square_conductances(
    net: SpiceNetlist,
    ports: DualPortSet,
    model: TriSquareModel,
    node_map: Dict[str, str],
    roots: Set[str],
    *,
    orientations: Optional[Sequence[LayerOrientation]] = None,
    floor_g: float = 0.0,
) -> TriSquareModel:
    del net, ports, node_map, roots, orientations, floor_g
    meta = dict(model.extract_meta)
    meta["method"] = "global_Rx_Ry_Rz"
    meta["note"] = "Zhang/via-stack extract + bilinear attach retired"
    model.extract_meta = meta
    return model
