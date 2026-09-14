"""Parametric single-layer Pixel-R star admittance G_S(Rx, Ry, Rz).

Each grid cell is an identical star unit cell (TSMC Spec):

- Top (pad / vsrc) -- Rup=Rz -- Center (sink / probe + Isink) -- Rdown=Rz -- Bottom
- Half-arms Rx / Ry from Center to shared (or boundary) edge midpoints

Fitted Rx, Ry are half-arm resistances: a neighbor path through a shared edge
node is 2·Rx or 2·Ry in series. Port admittance is the Schur complement of
internal edge + bottom nodes onto [pads | sinks].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from ports import PortSet, via_series_r

# Arm axis tags
AXIS_X = "x"
AXIS_Y = "y"


@dataclass
class PixelRModel:
    """
    Port ordering matches PortSet.port_nodes: pads first, then sinks.

    Topology (star_half_arm):
    - pad_p -- [optional Rx/Ry stubs] -- Rz -- sink_{pad_attach[p]} -- Rz -- bot
      (padless sinks have no Rz; current still lumped at the cell center)
    - sink_k -- Rx/Ry -- edge nodes (shared with neighbor or dangling at boundary)
    Stubs are off when ``ports.via_stub == "zero"`` (default).
    """

    ports: PortSet
    pad_attach: List[int]
    # Shared edges: (west_sink, east_sink) / (south_sink, north_sink)
    ew_shared: List[Tuple[int, int]] = field(default_factory=list)
    ns_shared: List[Tuple[int, int]] = field(default_factory=list)
    # Boundary dangling arms: (sink_k, side) side in {W,E} / {S,N}
    ew_boundary: List[Tuple[int, str]] = field(default_factory=list)
    ns_boundary: List[Tuple[int, str]] = field(default_factory=list)
    # Half-arm stamps after build: (sink_local, edge_local, axis)
    arms: List[Tuple[int, int, str]] = field(default_factory=list)
    n_edge_nodes: int = 0

    @property
    def n_ports(self) -> int:
        return self.ports.n_ports

    @property
    def n_pads(self) -> int:
        return self.ports.n_pads

    @property
    def n_sinks(self) -> int:
        return self.ports.n_sinks

    @property
    def ew_edges(self) -> List[Tuple[int, int]]:
        """Neighbor sink pairs (E–W), for backward-compatible summaries."""
        return [(a, b) for a, b in self.ew_shared]

    @property
    def ns_edges(self) -> List[Tuple[int, int]]:
        """Neighbor sink pairs (N–S), for backward-compatible summaries."""
        return [(a, b) for a, b in self.ns_shared]

    def to_fields(self) -> Dict[str, Any]:
        return {
            "topology": "star_half_arm",
            "pad_attach": [int(x) for x in self.pad_attach],
            "ew_shared": [[int(a), int(b)] for a, b in self.ew_shared],
            "ns_shared": [[int(a), int(b)] for a, b in self.ns_shared],
            "ew_boundary": [[int(k), str(side)] for k, side in self.ew_boundary],
            "ns_boundary": [[int(k), str(side)] for k, side in self.ns_boundary],
            "arms": [[int(s), int(e), str(ax)] for s, e, ax in self.arms],
            "n_edge_nodes": int(self.n_edge_nodes),
            # Convenience / legacy neighbor pairs
            "ew_edges": [[int(a), int(b)] for a, b in self.ew_shared],
            "ns_edges": [[int(a), int(b)] for a, b in self.ns_shared],
        }


def _build_star_topology(
    ports: PortSet,
) -> Tuple[
    List[Tuple[int, int]],
    List[Tuple[int, int]],
    List[Tuple[int, str]],
    List[Tuple[int, str]],
    List[Tuple[int, int, str]],
    int,
]:
    """
    Build shared + boundary edge nodes and half-arm list.

    Edge local indices are contiguous ``0 .. n_edge_nodes-1``.
    """
    loc = {(c.ix, c.iy): k for k, c in enumerate(ports.cells)}
    ew_shared: List[Tuple[int, int]] = []
    ns_shared: List[Tuple[int, int]] = []
    ew_boundary: List[Tuple[int, str]] = []
    ns_boundary: List[Tuple[int, str]] = []
    arms: List[Tuple[int, int, str]] = []
    edge_id = 0

    # Shared E–W edges (one node between west and east neighbors)
    for (ix, iy), k in loc.items():
        if (ix + 1, iy) in loc:
            ke = loc[(ix + 1, iy)]
            ew_shared.append((k, ke))
            arms.append((k, edge_id, AXIS_X))
            arms.append((ke, edge_id, AXIS_X))
            edge_id += 1

    # Shared N–S edges (south → north)
    for (ix, iy), k in loc.items():
        if (ix, iy + 1) in loc:
            kn = loc[(ix, iy + 1)]
            ns_shared.append((k, kn))
            arms.append((k, edge_id, AXIS_Y))
            arms.append((kn, edge_id, AXIS_Y))
            edge_id += 1

    # Dangling boundary half-arms so every pixel is identical
    for (ix, iy), k in loc.items():
        if (ix - 1, iy) not in loc:
            ew_boundary.append((k, "W"))
            arms.append((k, edge_id, AXIS_X))
            edge_id += 1
        if (ix + 1, iy) not in loc:
            ew_boundary.append((k, "E"))
            arms.append((k, edge_id, AXIS_X))
            edge_id += 1
        if (ix, iy - 1) not in loc:
            ns_boundary.append((k, "S"))
            arms.append((k, edge_id, AXIS_Y))
            edge_id += 1
        if (ix, iy + 1) not in loc:
            ns_boundary.append((k, "N"))
            arms.append((k, edge_id, AXIS_Y))
            edge_id += 1

    return ew_shared, ns_shared, ew_boundary, ns_boundary, arms, edge_id


def build_pixel_model(ports: PortSet) -> PixelRModel:
    cells = ports.cells
    if not cells:
        raise ValueError("no active sink cells for Pixel-R model")
    if ports.n_pads == 0:
        raise ValueError("need at least one voltage pad for Pixel-R")
    if ports.n_pads > ports.n_sinks:
        raise ValueError(
            f"n_pads ({ports.n_pads}) cannot exceed n_sinks ({ports.n_sinks})"
        )
    pad_attach = ports.resolved_pad_attach()
    if len(set(pad_attach)) != len(pad_attach):
        raise ValueError("each pad must attach to a unique sink cell")
    if any(s < 0 or s >= ports.n_sinks for s in pad_attach):
        raise ValueError("pad_attach sink index out of range")
    ew_shared, ns_shared, ew_boundary, ns_boundary, arms, n_edge = _build_star_topology(
        ports
    )
    return PixelRModel(
        ports=ports,
        pad_attach=pad_attach,
        ew_shared=ew_shared,
        ns_shared=ns_shared,
        ew_boundary=ew_boundary,
        ns_boundary=ns_boundary,
        arms=arms,
        n_edge_nodes=n_edge,
    )


def pixel_model_from_fields(ports: PortSet, fields: Dict[str, Any]) -> PixelRModel:
    """Reconstruct PixelRModel from pixel_model.json fields (+ ports)."""
    pad_attach = [int(x) for x in fields["pad_attach"]]
    topology = str(fields.get("topology", ""))

    if topology == "star_half_arm" and "arms" in fields:
        return PixelRModel(
            ports=ports,
            pad_attach=pad_attach,
            ew_shared=[(int(a), int(b)) for a, b in fields.get("ew_shared", [])],
            ns_shared=[(int(a), int(b)) for a, b in fields.get("ns_shared", [])],
            ew_boundary=[
                (int(k), str(side)) for k, side in fields.get("ew_boundary", [])
            ],
            ns_boundary=[
                (int(k), str(side)) for k, side in fields.get("ns_boundary", [])
            ],
            arms=[
                (int(s), int(e), str(ax)) for s, e, ax in fields.get("arms", [])
            ],
            n_edge_nodes=int(fields.get("n_edge_nodes", 0)),
        )

    # Legacy mesh artifacts: rebuild star from the grid (ignore old edge pairs).
    return build_pixel_model(ports)


def build_Gs(model: PixelRModel, Rx: float, Ry: float, Rz: float) -> np.ndarray:
    """
    Dense port admittance G_S(Rx, Ry, Rz).

    Exact port-level reduction of the star: two series half-arms of Rx (Ry)
    between neighbor sinks Schur to conductance 1/(2·Rx) (1/(2·Ry)); Rup=Rz
    couples pad↔sink (plus optional Rx/Ry stubs when ``via_stub=rxry``);
    dangling boundary arms and floating Rdown drop out.
    """
    if Rx <= 0 or Ry <= 0 or Rz <= 0:
        raise ValueError("Rx, Ry, Rz must be positive")

    n_p = model.n_pads
    n_s = model.n_sinks
    n = n_p + n_s
    G = np.zeros((n, n), dtype=float)

    # Shared half-arms → effective neighbor conductance 1/(2R)
    gx_eff, gy_eff = 0.5 / Rx, 0.5 / Ry

    def stamp(i: int, j: int, g: float) -> None:
        G[i, i] += g
        G[j, j] += g
        G[i, j] -= g
        G[j, i] -= g

    ew = model.ew_shared or model.ew_edges
    ns = model.ns_shared or model.ns_edges
    for a, b in ew:
        stamp(n_p + int(a), n_p + int(b), gx_eff)
    for a, b in ns:
        stamp(n_p + int(a), n_p + int(b), gy_eff)
    for p, s in enumerate(model.pad_attach):
        r_via = via_series_r(model.ports, Rx, Ry, Rz, p, int(s))
        stamp(p, n_p + int(s), 1.0 / r_via)

    return 0.5 * (G + G.T)


def neighbor_sink_conductance(Gs: np.ndarray, model: PixelRModel, a: int, b: int) -> float:
    """Return -G_S[sink_a, sink_b] (off-diagonal coupling magnitude)."""
    n_p = model.n_pads
    return float(-Gs[n_p + a, n_p + b])
