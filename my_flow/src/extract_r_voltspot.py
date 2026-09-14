"""VoltSpot Eq. 2 branch-R extraction for the multi-layer virtual grid.

Per x-directed layer L:
  Z_wire ≈ mean(r/ℓ) * pitch          # wire resistance over one grid pitch
  Z_x(L) = (N_rows / N_metal_rows) * Z_wire
         = (ny / n_tracks) * mean_rpl * pitch

(Equivalent to paper Eq. 2 when Z_wire_full = mean_rpl * pitch*(N_cols-1).)

Y-directed layers contribute Z_y similarly. Layers of the same orientation
add in parallel on every matching grid edge. Vias are omitted.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from _bootstrap import ensure_paths

ensure_paths()

from extract_r import (  # noqa: E402
    FLOOR_G,
    FLOOR_R,
    _eq2_layer_stats,
    _same_layer_segments,
)
from layer_orient import LayerOrientation, classify_vdd_layers  # noqa: E402
from ports_dual import DualPortSet  # noqa: E402
from spice_parser import SpiceNetlist  # noqa: E402
from voltspot_model import VoltSpotModel  # noqa: E402


def extract_voltspot_conductances(
    net: SpiceNetlist,
    ports: DualPortSet,
    model: VoltSpotModel,
    node_map: Dict[str, str],
    roots: Set[str],
    *,
    orientations: Optional[Sequence[LayerOrientation]] = None,
    floor_g: float = FLOOR_G,
) -> VoltSpotModel:
    if orientations is None:
        orientations = classify_vdd_layers(net, node_map, roots, ports.vdd_layers)

    nx, ny = model.nx, model.ny
    pitch = float(model.pitch)
    g_ew_sum = 0.0
    g_ns_sum = 0.0
    layer_meta: List[Dict[str, Any]] = []

    for ori in orientations:
        L = int(ori.layer)
        want = ori.orientation
        if want not in ("x", "y"):
            layer_meta.append(
                {
                    "layer": L,
                    "orientation": want,
                    "skipped": True,
                    "reason": "unknown orientation",
                }
            )
            continue
        segs = _same_layer_segments(net, node_map, roots, [L], want=want)
        stats = _eq2_layer_stats(segs, want=want, nx=nx, ny=ny, pitch=pitch)
        r_branch = float(stats.get("R_branch", FLOOR_R))
        r_branch = max(r_branch, FLOOR_R)
        g_branch = 1.0 / r_branch
        entry = {
            "layer": L,
            "orientation": want,
            "skipped": False,
            "R_branch": r_branch,
            "G_branch": g_branch,
            "n_metal_tracks": stats.get("n_metal_tracks"),
            "mean_r_per_len": stats.get("mean_r_per_len"),
            "n_segs": stats.get("n_segs"),
            "formula": "Eq2: R = (N_grid_perp / N_metal_tracks) * mean(r/ℓ) * pitch",
        }
        layer_meta.append(entry)
        if want == "x":
            g_ew_sum += g_branch
        else:
            g_ns_sum += g_branch

    # Fallback if a direction has no metal: keep a tiny floor so G stays SPD-ish.
    if g_ew_sum <= 0:
        g_ew_sum = float(floor_g)
    if g_ns_sum <= 0:
        g_ns_sum = float(floor_g)

    n_ew = len(model.ew_edges)
    n_ns = len(model.ns_edges)
    G_ew = np.full(n_ew, g_ew_sum, dtype=float)
    G_ns = np.full(n_ns, g_ns_sum, dtype=float)
    R_ew = 1.0 / G_ew
    R_ns = 1.0 / G_ns
    R_ew_branch = float(1.0 / g_ew_sum)
    R_ns_branch = float(1.0 / g_ns_sum)

    meta: Dict[str, Any] = {
        "method": "voltspot_eq2_multilayer",
        "vias": "omitted",
        "stamp": (
            "single virtual square grid; per-layer Eq.2 R in parallel by "
            "orientation; near-ideal pad/sink attach; no interlayer vias"
        ),
        "formula": {
            "R_ew": "parallel_L∈x-layers  (ny / N_tracks_L) * mean(r/ℓ)_L * pitch",
            "R_ns": "parallel_L∈y-layers  (nx / N_tracks_L) * mean(r/ℓ)_L * pitch",
            "via": "omitted (VoltSpot validation claim)",
        },
        "layers": layer_meta,
        "R_ew_branch": R_ew_branch,
        "R_ns_branch": R_ns_branch,
        "summary_means": {
            "R_ew": R_ew_branch,
            "R_ns": R_ns_branch,
            "G_ew": float(g_ew_sum),
            "G_ns": float(g_ns_sum),
        },
        "grid_to_pad_ratio_target": 4.0,
        "nx": nx,
        "ny": ny,
        "pitch": pitch,
        "floor_g": float(floor_g),
        "floor_r": float(FLOOR_R),
    }

    return VoltSpotModel(
        ports=model.ports,
        nx=model.nx,
        ny=model.ny,
        pitch=model.pitch,
        ew_edges=list(model.ew_edges),
        ns_edges=list(model.ns_edges),
        pad_mesh=list(model.pad_mesh),
        sink_mesh=list(model.sink_mesh),
        G_ew=G_ew,
        G_ns=G_ns,
        R_ew=R_ew,
        R_ns=R_ns,
        R_ew_branch=R_ew_branch,
        R_ns_branch=R_ns_branch,
        extract_meta=meta,
    )
