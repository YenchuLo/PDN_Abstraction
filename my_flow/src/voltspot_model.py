"""VoltSpot-style multi-layer virtual grid (DC / IR only).

Faithful to VoltSpot TR CS-2014-01 Sec. 3:
  - Regular square virtual grid (default ~4 grid nodes per pad)
  - Each physical metal layer → parallel R-branch on matching orientation
  - Branch R from Eq. 2 so total layer resistance matches the physical stack
  - Interlayer vias omitted
  - Pads / sinks attach to nearest grid nodes (near-ideal)

Topology (full stamped graph before Kron):
  [pads | grid | sinks]
  pad_k  -- G_attach --> grid[pad_mesh]
  sink_j -- G_attach --> grid[sink_mesh]
  grid E–W / N–S edges: parallel of per-layer Eq. 2 conductances
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from _bootstrap import ensure_paths

ensure_paths()

from kron import kron_reduce  # noqa: E402
from ports_dual import DualPortSet  # noqa: E402

VOLTSPOT_TOPOLOGY = "voltspot_eq2_multilayer"
ATTACH_G = 1.0e6  # near-ideal pad/sink → grid attach


@dataclass
class VoltSpotModel:
    ports: DualPortSet
    nx: int
    ny: int
    pitch: float
    ew_edges: List[Tuple[int, int]] = field(default_factory=list)
    ns_edges: List[Tuple[int, int]] = field(default_factory=list)
    pad_mesh: List[int] = field(default_factory=list)
    sink_mesh: List[int] = field(default_factory=list)
    # Uniform per-edge (parallel of layers) after extract
    G_ew: Optional[np.ndarray] = None
    G_ns: Optional[np.ndarray] = None
    R_ew: Optional[np.ndarray] = None
    R_ns: Optional[np.ndarray] = None
    R_ew_branch: float = 0.0  # scalar Eq.2 sum after parallel
    R_ns_branch: float = 0.0
    extract_meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_grid(self) -> int:
        return int(self.nx * self.ny)

    @property
    def n_pads(self) -> int:
        return self.ports.n_pads

    @property
    def n_sinks(self) -> int:
        return self.ports.n_sinks

    @property
    def n_ports(self) -> int:
        return self.ports.n_ports

    def to_fields(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "topology": VOLTSPOT_TOPOLOGY,
            "nx": int(self.nx),
            "ny": int(self.ny),
            "pitch": float(self.pitch),
            "n_grid": int(self.n_grid),
            "n_pads": int(self.n_pads),
            "n_sinks": int(self.n_sinks),
            "ew_edges": [[int(a), int(b)] for a, b in self.ew_edges],
            "ns_edges": [[int(a), int(b)] for a, b in self.ns_edges],
            "pad_mesh": [int(x) for x in self.pad_mesh],
            "sink_mesh": [int(x) for x in self.sink_mesh],
            "R_ew_branch": float(self.R_ew_branch),
            "R_ns_branch": float(self.R_ns_branch),
            "extract_meta": dict(self.extract_meta),
        }
        for key, arr in (
            ("G_ew", self.G_ew),
            ("G_ns", self.G_ns),
            ("R_ew", self.R_ew),
            ("R_ns", self.R_ns),
        ):
            if arr is not None:
                out[key] = [float(x) for x in np.asarray(arr, dtype=float)]
        return out


def _square_edges(nx: int, ny: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    ew: List[Tuple[int, int]] = []
    ns: List[Tuple[int, int]] = []
    for iy in range(ny):
        for ix in range(nx - 1):
            a = iy * nx + ix
            ew.append((a, a + 1))
    for iy in range(ny - 1):
        for ix in range(nx):
            a = iy * nx + ix
            ns.append((a, a + nx))
    return ew, ns


def build_voltspot_model(ports: DualPortSet) -> VoltSpotModel:
    """Build square virtual grid aligned to dual fine lattice (k=1)."""
    if ports.n_pads == 0 or ports.n_sinks == 0:
        raise ValueError("need at least one pad and one sink")
    if ports.nx_bot != ports.nx_top or ports.ny_bot != ports.ny_top:
        raise ValueError("VoltSpot v1 expects coarsen_k=1 (matching top/bot lattices)")
    nx, ny = int(ports.nx_bot), int(ports.ny_bot)
    ew, ns = _square_edges(nx, ny)
    return VoltSpotModel(
        ports=ports,
        nx=nx,
        ny=ny,
        pitch=float(ports.pitch_bot),
        ew_edges=ew,
        ns_edges=ns,
        pad_mesh=[int(x) for x in ports.pad_mesh],
        sink_mesh=[int(x) for x in ports.sink_mesh],
    )


def _as_gvec(
    value: Union[float, Sequence[float], np.ndarray],
    n: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 1:
        g = float(arr[0])
        if g <= 0:
            raise ValueError(f"{name} must be positive")
        return np.full(n, g, dtype=float)
    if arr.size != n:
        raise ValueError(f"{name}: expected length {n}, got {arr.size}")
    if np.any(arr <= 0):
        raise ValueError(f"{name} entries must be positive")
    return arr


def build_Gs_voltspot(
    model: VoltSpotModel,
    G_ew: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    G_ns: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    *,
    alpha_x: float = 1.0,
    alpha_y: float = 1.0,
    attach_g: float = ATTACH_G,
) -> np.ndarray:
    """Port admittance after Kron-eliminating grid internals.

    Node order: [pads | grid | sinks].
    """
    n_ew = len(model.ew_edges)
    n_ns = len(model.ns_edges)
    n_pad = model.n_pads
    n_sink = model.n_sinks
    n_grid = model.n_grid

    if G_ew is None:
        G_ew = model.G_ew
    if G_ns is None:
        G_ns = model.G_ns
    if G_ew is None or G_ns is None:
        raise ValueError("VoltSpot model missing extracted conductances")

    gew = float(alpha_x) * _as_gvec(G_ew, n_ew, "G_ew")
    gns = float(alpha_y) * _as_gvec(G_ns, n_ns, "G_ns")
    g_att = float(attach_g)
    if g_att <= 0:
        raise ValueError("attach_g must be positive")

    n = n_pad + n_grid + n_sink
    G = np.zeros((n, n), dtype=float)
    off_g = n_pad
    off_s = n_pad + n_grid

    def stamp(i: int, j: int, g: float) -> None:
        G[i, i] += g
        G[j, j] += g
        G[i, j] -= g
        G[j, i] -= g

    for e, (a, b) in enumerate(model.ew_edges):
        stamp(off_g + int(a), off_g + int(b), float(gew[e]))
    for e, (a, b) in enumerate(model.ns_edges):
        stamp(off_g + int(a), off_g + int(b), float(gns[e]))
    for k, gi in enumerate(model.pad_mesh):
        stamp(int(k), off_g + int(gi), g_att)
    for j, gi in enumerate(model.sink_mesh):
        stamp(off_g + int(gi), off_s + int(j), g_att)

    port_idx = list(range(n_pad)) + list(range(off_s, n))
    internal_idx = list(range(off_g, off_s))
    if not internal_idx:
        P = np.array(port_idx, dtype=int)
        return 0.5 * (G[np.ix_(P, P)] + G[np.ix_(P, P)].T)

    from scipy import sparse

    Gs = kron_reduce(
        sparse.csc_matrix(G),
        np.asarray(port_idx, dtype=int),
        np.asarray(internal_idx, dtype=int),
    )
    return 0.5 * (Gs + Gs.T)
