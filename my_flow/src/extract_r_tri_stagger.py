"""Deprecated: stagger no longer seeds from Zhang / via-stack extract.

Kept as a thin no-op so older imports do not break. Topology + global
Rx/Ry/Rz stamping live in ``tri_stagger``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Set

from _bootstrap import ensure_paths

ensure_paths()

from layer_orient import LayerOrientation  # noqa: E402
from ports_dual import DualPortSet  # noqa: E402
from spice_parser import SpiceNetlist  # noqa: E402
from tri_stagger import TriStaggerModel  # noqa: E402


def extract_tri_stagger_conductances(
    net: SpiceNetlist,
    ports: DualPortSet,
    model: TriStaggerModel,
    node_map: Dict[str, str],
    roots: Set[str],
    *,
    orientations: Optional[Sequence[LayerOrientation]] = None,
    floor_g: float = 0.0,
) -> TriStaggerModel:
    """No-op: conductances are stamped from fitted Rx, Ry, Rz."""
    del net, ports, node_map, roots, orientations, floor_g
    meta = dict(model.extract_meta)
    meta["method"] = "global_Rx_Ry_Rz"
    meta["note"] = "Zhang/via-stack extract retired; fit stamps G from Rx,Ry,Rz"
    model.extract_meta = meta
    return model
