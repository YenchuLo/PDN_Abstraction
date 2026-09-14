"""Staggered mid-grid: real pad/sink XY + stubs + global Rx, Ry, Rz.

Topology (Pixel-R-style globals; real ports):
  Pad k  -- Rz -- Mid landing (pad XY)
  Sink j -- Rz -- Mid landing (sink XY)
  Mid: square grid at pitch_bot + length-proportional stubs
       R_ew = Rx * (|dx|/pitch),  R_ns = Ry * (|dy|/pitch)

Pad and sink landings are always distinct so pad→sink cannot be a pure via.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from _bootstrap import ensure_paths

ensure_paths()

from kron import kron_reduce  # noqa: E402
from ports_dual import DualPortSet  # noqa: E402
from stub_attach import (  # noqa: E402
    VIA_STUB_RXRY,
    build_stub_square_graph,
    conductances_from_rxryrz,
    parse_via_stub_arg,
)

Point = Tuple[float, float]

TRI_STAGGER_TOPOLOGY = "tri_stagger_mid_grid"


@dataclass
class TriStaggerModel:
    ports: DualPortSet
    mid_xy: List[Point] = field(default_factory=list)
    ew_edges: List[Tuple[int, int]] = field(default_factory=list)
    ns_edges: List[Tuple[int, int]] = field(default_factory=list)
    ew_lengths: List[float] = field(default_factory=list)
    ns_lengths: List[float] = field(default_factory=list)
    pad_mid: List[int] = field(default_factory=list)
    sink_mid: List[int] = field(default_factory=list)
    n_grid: int = 0
    nx_grid: int = 0
    ny_grid: int = 0
    # Optional last-stamped values (diagnostics)
    Rx: float = 0.0
    Ry: float = 0.0
    Rz: float = 0.0
    G_ew: Optional[np.ndarray] = None
    G_ns: Optional[np.ndarray] = None
    G_via_pad: Optional[np.ndarray] = None
    G_via_sink: Optional[np.ndarray] = None
    R_ew: Optional[np.ndarray] = None
    R_ns: Optional[np.ndarray] = None
    R_via_pad: Optional[np.ndarray] = None
    R_via_sink: Optional[np.ndarray] = None
    extract_meta: Dict[str, Any] = field(default_factory=dict)
    via_stub: str = VIA_STUB_RXRY

    @property
    def n_mid(self) -> int:
        return len(self.mid_xy)

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
            "topology": TRI_STAGGER_TOPOLOGY,
            "stamp": "global_Rx_Ry_Rz_length_proportional",
            "n_mid": int(self.n_mid),
            "n_grid": int(self.n_grid),
            "nx_grid": int(self.nx_grid),
            "ny_grid": int(self.ny_grid),
            "n_pads": int(self.n_pads),
            "n_sinks": int(self.n_sinks),
            "mid_xy": [[float(x), float(y)] for x, y in self.mid_xy],
            "ew_edges": [[int(a), int(b)] for a, b in self.ew_edges],
            "ns_edges": [[int(a), int(b)] for a, b in self.ns_edges],
            "ew_lengths": [float(x) for x in self.ew_lengths],
            "ns_lengths": [float(x) for x in self.ns_lengths],
            "pad_mid": [int(x) for x in self.pad_mid],
            "sink_mid": [int(x) for x in self.sink_mid],
            "Rx": float(self.Rx),
            "Ry": float(self.Ry),
            "Rz": float(self.Rz),
            "pad_xy": [[float(x), float(y)] for x, y in self.ports.pad_xy],
            "sink_xy": [[float(c.x), float(c.y)] for c in self.ports.cells],
            "pitch_bot": float(self.ports.pitch_bot),
            "via_stub": str(self.via_stub),
            "extract_meta": dict(self.extract_meta),
        }
        for key, arr in (
            ("G_ew", self.G_ew),
            ("G_ns", self.G_ns),
            ("G_via_pad", self.G_via_pad),
            ("G_via_sink", self.G_via_sink),
            ("R_ew", self.R_ew),
            ("R_ns", self.R_ns),
            ("R_via_pad", self.R_via_pad),
            ("R_via_sink", self.R_via_sink),
        ):
            if arr is not None:
                out[key] = [float(x) for x in arr]
        return out


def build_tri_stagger_mid_graph(
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
    via_stub: str = VIA_STUB_RXRY,
) -> Tuple[
    List[Point],
    List[int],
    List[int],
    List[Tuple[int, int]],
    List[float],
    List[Tuple[int, int]],
    List[float],
    int,
    int,
    int,
]:
    """Compatibility wrapper around :func:`build_stub_square_graph`."""
    g = build_stub_square_graph(
        pad_xy,
        sink_xy,
        pitch,
        bbox=bbox,
        grid_centers=grid_centers,
        nx=nx,
        ny=ny,
        eps_scale=eps_scale,
        floor_scale=floor_scale,
        square_mesh=True,
        via_stub=via_stub,
    )
    return (
        g.xy,
        g.pad_landing,
        g.sink_landing,
        g.ew_edges,
        g.ew_lengths,
        g.ns_edges,
        g.ns_lengths,
        g.n_grid,
        g.nx,
        g.ny,
    )


def build_tri_stagger_model(
    ports: DualPortSet, *, via_stub: str = VIA_STUB_RXRY
) -> TriStaggerModel:
    if ports.n_pads == 0 or ports.n_sinks == 0:
        raise ValueError("need at least one pad and one sink")
    stub_mode = parse_via_stub_arg(via_stub)
    sink_xy = [(float(c.x), float(c.y)) for c in ports.cells]
    grid_centers = ports.bot_centers or ports.top_centers or None
    nx = ports.nx_bot if grid_centers else None
    ny = ports.ny_bot if grid_centers else None
    if grid_centers and len(grid_centers) != ports.nx_bot * ports.ny_bot:
        grid_centers = None
        nx = ny = None
    (
        mid_xy,
        pad_mid,
        sink_mid,
        ew,
        ew_L,
        ns,
        ns_L,
        n_grid,
        nx_g,
        ny_g,
    ) = build_tri_stagger_mid_graph(
        ports.pad_xy,
        sink_xy,
        ports.pitch_bot,
        bbox=ports.bbox,
        grid_centers=grid_centers,
        nx=nx,
        ny=ny,
        via_stub=stub_mode,
    )
    return TriStaggerModel(
        ports=ports,
        mid_xy=mid_xy,
        ew_edges=ew,
        ns_edges=ns,
        ew_lengths=ew_L,
        ns_lengths=ns_L,
        pad_mid=pad_mid,
        sink_mid=sink_mid,
        n_grid=n_grid,
        nx_grid=nx_g,
        ny_grid=ny_g,
        via_stub=stub_mode,
        extract_meta={
            "method": "global_Rx_Ry_Rz",
            "stamp": "R_ew=Rx*|dx|/pitch; R_ns=Ry*|dy|/pitch; R_via=Rz",
            "via_stub": stub_mode,
            "formula": {
                "R_ew": "Rx * |dx| / pitch (grid + H stubs)"
                if stub_mode == VIA_STUB_RXRY
                else "Rx * |dx| / pitch (grid only)",
                "R_ns": "Ry * |dy| / pitch (grid + V stubs)"
                if stub_mode == VIA_STUB_RXRY
                else "Ry * |dy| / pitch (grid only)",
                "R_via_pad": "Rz",
                "R_via_sink": "Rz",
            },
        },
    )


def tri_stagger_model_from_fields(
    ports: DualPortSet, fields: Dict[str, Any]
) -> TriStaggerModel:
    topology = str(fields.get("topology", ""))
    if topology and topology not in (TRI_STAGGER_TOPOLOGY, "tri_stagger_mid"):
        raise ValueError(f"unexpected topology: {topology!r}")

    def _arr(key: str) -> Optional[np.ndarray]:
        v = fields.get(key)
        return np.asarray(v, dtype=float) if v is not None else None

    return TriStaggerModel(
        ports=ports,
        mid_xy=[(float(x), float(y)) for x, y in fields.get("mid_xy", [])],
        ew_edges=[(int(a), int(b)) for a, b in fields.get("ew_edges", [])],
        ns_edges=[(int(a), int(b)) for a, b in fields.get("ns_edges", [])],
        ew_lengths=[float(x) for x in fields.get("ew_lengths", [])],
        ns_lengths=[float(x) for x in fields.get("ns_lengths", [])],
        pad_mid=[int(x) for x in fields.get("pad_mid", [])],
        sink_mid=[int(x) for x in fields.get("sink_mid", [])],
        n_grid=int(fields.get("n_grid", 0)),
        nx_grid=int(fields.get("nx_grid", 0)),
        ny_grid=int(fields.get("ny_grid", 0)),
        Rx=float(fields.get("Rx", fields.get("rx0", 0.0))),
        Ry=float(fields.get("Ry", fields.get("ry0", 0.0))),
        Rz=float(fields.get("Rz", 0.0)),
        via_stub=parse_via_stub_arg(fields.get("via_stub", VIA_STUB_RXRY)),
        G_ew=_arr("G_ew"),
        G_ns=_arr("G_ns"),
        G_via_pad=_arr("G_via_pad"),
        G_via_sink=_arr("G_via_sink"),
        R_ew=_arr("R_ew"),
        R_ns=_arr("R_ns"),
        R_via_pad=_arr("R_via_pad"),
        R_via_sink=_arr("R_via_sink"),
        extract_meta=dict(fields.get("extract_meta", {})),
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


def stamp_tri_stagger_conductances(
    model: TriStaggerModel, Rx: float, Ry: float, Rz: float
) -> TriStaggerModel:
    """Fill model G/R arrays from global Rx, Ry, Rz (mutates and returns model)."""
    pitch = float(model.ports.pitch_bot)
    G_ew, G_ns, G_vp, G_vs = conductances_from_rxryrz(
        model.ew_lengths,
        model.ns_lengths,
        pitch,
        Rx,
        Ry,
        Rz,
        model.n_pads,
        model.n_sinks,
    )
    model.Rx = float(Rx)
    model.Ry = float(Ry)
    model.Rz = float(Rz)
    model.G_ew = G_ew
    model.G_ns = G_ns
    model.G_via_pad = G_vp
    model.G_via_sink = G_vs
    model.R_ew = 1.0 / G_ew
    model.R_ns = 1.0 / G_ns
    model.R_via_pad = np.full(model.n_pads, float(Rz))
    model.R_via_sink = np.full(model.n_sinks, float(Rz))
    return model


def build_Gs_tri_stagger(
    model: TriStaggerModel,
    G_ew: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    G_ns: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    G_via_pad: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    G_via_sink: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    *,
    Rx: Optional[float] = None,
    Ry: Optional[float] = None,
    Rz: Optional[float] = None,
    alpha_x: float = 1.0,
    alpha_y: float = 1.0,
    alpha_via: float = 1.0,
) -> np.ndarray:
    """Port admittance after Kron-eliminating mid nodes.

    Prefer ``Rx, Ry, Rz`` (Pixel-R-style). Legacy G_* / alpha_* still accepted.
    """
    if Rx is not None and Ry is not None and Rz is not None:
        stamp_tri_stagger_conductances(model, float(Rx), float(Ry), float(Rz))
        G_ew = model.G_ew
        G_ns = model.G_ns
        G_via_pad = model.G_via_pad
        G_via_sink = model.G_via_sink
        alpha_x = alpha_y = alpha_via = 1.0

    n_ew = len(model.ew_edges)
    n_ns = len(model.ns_edges)
    n_pad = model.n_pads
    n_sink = model.n_sinks
    n_mid = model.n_mid

    if G_ew is None:
        G_ew = model.G_ew
    if G_ns is None:
        G_ns = model.G_ns
    if G_via_pad is None:
        G_via_pad = model.G_via_pad
    if G_via_sink is None:
        G_via_sink = model.G_via_sink
    if G_ew is None or G_ns is None or G_via_pad is None or G_via_sink is None:
        raise ValueError("stagger model missing conductances (pass Rx,Ry,Rz)")

    gew = alpha_x * _as_gvec(G_ew, n_ew, "G_ew")
    gns = alpha_y * _as_gvec(G_ns, n_ns, "G_ns")
    gvp = alpha_via * _as_gvec(G_via_pad, n_pad, "G_via_pad")
    gvs = alpha_via * _as_gvec(G_via_sink, n_sink, "G_via_sink")

    n = n_pad + n_mid + n_sink
    G = np.zeros((n, n), dtype=float)
    off_m = n_pad
    off_s = n_pad + n_mid

    def stamp(i: int, j: int, g: float) -> None:
        G[i, i] += g
        G[j, j] += g
        G[i, j] -= g
        G[j, i] -= g

    for e, (a, b) in enumerate(model.ew_edges):
        stamp(off_m + int(a), off_m + int(b), float(gew[e]))
    for e, (a, b) in enumerate(model.ns_edges):
        stamp(off_m + int(a), off_m + int(b), float(gns[e]))
    for k, mi in enumerate(model.pad_mid):
        stamp(int(k), off_m + int(mi), float(gvp[k]))
    for j, mi in enumerate(model.sink_mid):
        stamp(off_m + int(mi), off_s + int(j), float(gvs[j]))

    port_idx = list(range(n_pad)) + list(range(off_s, n))
    internal_idx = list(range(off_m, off_s))
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
