"""Tri-square: L2 mid (= Pixel-R pitch) + denser L3 bot, stubs, 6 R params.

Topology:
  Pad -- Rz_pad -- L2 landing -- stubs on L2 @ pitch_u = pitch_bot (Pixel-R pitch)
  Each L2 grid node -- Rz_ul -- L3 landing (same XY) -- stubs on L3 @ pitch_l
       (stub R ∝ L3 Rx_l, Ry_l with pitch_l = pitch_u/k)
  Sink -- Rz_pad -- L3 landing -- stubs on L3 @ pitch_l

Six fit parameters:
  Rx_u, Ry_u  — L2 mid sheet (grid + pad stubs)
  Rx_l, Ry_l  — L3 bot sheet (grid + via/sink stubs)
  Rz_pad      — port vias (pad→L2 and sink→L3)
  Rz_ul       — interlayer vias (L2→L3)

Attach uses nearest-grid stubs (same as stagger), not bilinear.
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
    parse_via_stub_arg,
)

Point = Tuple[float, float]
TRI_SQUARE_TOPOLOGY = "tri_square_stub_6r"
# Older artifacts used a single (Rx, Ry, Rz) shared across sheets.
TRI_SQUARE_TOPOLOGY_LEGACY = "tri_square_stub_rxryrz"


@dataclass
class TriSquareModel:
    ports: DualPortSet
    pitch_u: float
    pitch_l: float
    coarsen_k: int
    nx_u: int
    ny_u: int
    nx_l: int
    ny_l: int
    l2_xy: List[Point] = field(default_factory=list)
    l3_xy: List[Point] = field(default_factory=list)
    ew_u: List[Tuple[int, int]] = field(default_factory=list)
    ew_u_L: List[float] = field(default_factory=list)
    ns_u: List[Tuple[int, int]] = field(default_factory=list)
    ns_u_L: List[float] = field(default_factory=list)
    ew_l: List[Tuple[int, int]] = field(default_factory=list)
    ew_l_L: List[float] = field(default_factory=list)
    ns_l: List[Tuple[int, int]] = field(default_factory=list)
    ns_l_L: List[float] = field(default_factory=list)
    # (l2_grid_idx, l3_landing_idx): vertical Rz; L3 landing stubs into L3 mesh
    vias_ul: List[Tuple[int, int]] = field(default_factory=list)
    via_landing_l3: List[int] = field(default_factory=list)
    pad_landing: List[int] = field(default_factory=list)
    sink_landing: List[int] = field(default_factory=list)
    n_l2_grid: int = 0
    n_l3_grid: int = 0
    Rx_u: float = 0.0
    Ry_u: float = 0.0
    Rx_l: float = 0.0
    Ry_l: float = 0.0
    Rz_pad: float = 0.0
    Rz_ul: float = 0.0
    extract_meta: Dict[str, Any] = field(default_factory=dict)
    via_stub: str = VIA_STUB_RXRY

    # Back-compat aliases (mid-sheet / pad-via)
    @property
    def Rx(self) -> float:
        return self.Rx_u

    @Rx.setter
    def Rx(self, v: float) -> None:
        self.Rx_u = float(v)

    @property
    def Ry(self) -> float:
        return self.Ry_u

    @Ry.setter
    def Ry(self, v: float) -> None:
        self.Ry_u = float(v)

    @property
    def Rz(self) -> float:
        return self.Rz_pad

    @Rz.setter
    def Rz(self, v: float) -> None:
        self.Rz_pad = float(v)

    @property
    def n_l2(self) -> int:
        return len(self.l2_xy)

    @property
    def n_l3(self) -> int:
        return len(self.l3_xy)

    @property
    def n_pads(self) -> int:
        return self.ports.n_pads

    @property
    def n_sinks(self) -> int:
        return self.ports.n_sinks

    def to_fields(self) -> Dict[str, Any]:
        return {
            "topology": TRI_SQUARE_TOPOLOGY,
            "stamp": "per_sheet_Rx_Ry_two_vias_stubs",
            "pitch_u": float(self.pitch_u),
            "pitch_l": float(self.pitch_l),
            "coarsen_k": int(self.coarsen_k),
            "nx_u": int(self.nx_u),
            "ny_u": int(self.ny_u),
            "nx_l": int(self.nx_l),
            "ny_l": int(self.ny_l),
            "n_l2": int(self.n_l2),
            "n_l3": int(self.n_l3),
            "n_l2_grid": int(self.n_l2_grid),
            "n_l3_grid": int(self.n_l3_grid),
            "l2_xy": [[float(x), float(y)] for x, y in self.l2_xy],
            "l3_xy": [[float(x), float(y)] for x, y in self.l3_xy],
            "ew_u": [[int(a), int(b)] for a, b in self.ew_u],
            "ew_u_L": [float(x) for x in self.ew_u_L],
            "ns_u": [[int(a), int(b)] for a, b in self.ns_u],
            "ns_u_L": [float(x) for x in self.ns_u_L],
            "ew_l": [[int(a), int(b)] for a, b in self.ew_l],
            "ew_l_L": [float(x) for x in self.ew_l_L],
            "ns_l": [[int(a), int(b)] for a, b in self.ns_l],
            "ns_l_L": [float(x) for x in self.ns_l_L],
            "vias_ul": [[int(a), int(b)] for a, b in self.vias_ul],
            "via_landing_l3": [int(x) for x in self.via_landing_l3],
            "pad_landing": [int(x) for x in self.pad_landing],
            "sink_landing": [int(x) for x in self.sink_landing],
            "Rx_u": float(self.Rx_u),
            "Ry_u": float(self.Ry_u),
            "Rx_l": float(self.Rx_l),
            "Ry_l": float(self.Ry_l),
            "Rz_pad": float(self.Rz_pad),
            "Rz_ul": float(self.Rz_ul),
            # Aliases for older readers
            "Rx": float(self.Rx_u),
            "Ry": float(self.Ry_u),
            "Rz": float(self.Rz_pad),
            "via_stub": str(self.via_stub),
            "extract_meta": dict(self.extract_meta),
        }


def build_tri_square_model(
    ports: DualPortSet, *, coarsen_k: int = 2, via_stub: str = VIA_STUB_RXRY
) -> TriSquareModel:
    if ports.n_pads == 0 or ports.n_sinks == 0:
        raise ValueError("need at least one pad and one sink")
    stub_mode = parse_via_stub_arg(via_stub)
    k = max(1, int(coarsen_k))
    # Mid L2 matches Pixel-R / port pitch; bot L3 is denser by k.
    pitch_u = float(ports.pitch_bot)
    pitch_l = pitch_u / float(k)
    if pitch_l <= 0:
        raise ValueError("pitch_l must be positive")
    sink_xy = [(float(c.x), float(c.y)) for c in ports.cells]

    # L2 mid (= Pixel-R pitch): pads attach with stubs (or snap if via_stub=zero)
    g2 = build_stub_square_graph(
        ports.pad_xy,
        [],
        pitch_u,
        bbox=ports.bbox,
        square_mesh=True,
        via_stub=stub_mode,
    )
    # L3 bot (denser): via landings at each L2 grid XY + sink landings.
    # Via landings stub into the L3 mesh with R ∝ Rx/Ry at pitch_l.
    l2_grid_xy = list(g2.xy[: g2.n_grid])
    g3 = build_stub_square_graph(
        l2_grid_xy,
        sink_xy,
        pitch_l,
        bbox=ports.bbox,
        square_mesh=True,
        via_stub=stub_mode,
    )
    via_landing_l3 = list(g3.pad_landing)
    if len(via_landing_l3) != g2.n_grid:
        raise RuntimeError(
            f"via landing count {len(via_landing_l3)} != n_l2_grid {g2.n_grid}"
        )
    vias_ul: List[Tuple[int, int]] = [
        (i, int(via_landing_l3[i])) for i in range(g2.n_grid)
    ]

    return TriSquareModel(
        ports=ports,
        pitch_u=pitch_u,
        pitch_l=pitch_l,
        coarsen_k=k,
        nx_u=g2.nx,
        ny_u=g2.ny,
        nx_l=g3.nx,
        ny_l=g3.ny,
        l2_xy=g2.xy,
        l3_xy=g3.xy,
        ew_u=g2.ew_edges,
        ew_u_L=g2.ew_lengths,
        ns_u=g2.ns_edges,
        ns_u_L=g2.ns_lengths,
        ew_l=g3.ew_edges,
        ew_l_L=g3.ew_lengths,
        ns_l=g3.ns_edges,
        ns_l_L=g3.ns_lengths,
        vias_ul=vias_ul,
        via_landing_l3=via_landing_l3,
        pad_landing=list(g2.pad_landing),
        sink_landing=list(g3.sink_landing),
        n_l2_grid=g2.n_grid,
        n_l3_grid=g3.n_grid,
        via_stub=stub_mode,
        extract_meta={
            "method": "per_sheet_Rx_Ry_two_vias",
            "stamp": "L2: Rx_u/Ry_u; L3: Rx_l/Ry_l; pad/sink via=Rz_pad; L2→L3=Rz_ul",
            "via_stub": stub_mode,
            "note": (
                "L2→L3: Rz_ul to landing @ L2 XY, then L3 stubs ∝ Rx_l/Ry_l"
                if stub_mode == VIA_STUB_RXRY
                else "L2→L3: Rz_ul to nearest L3 grid node (via_stub=zero)"
            ),
            "params": ["Rx_u", "Ry_u", "Rx_l", "Ry_l", "Rz_pad", "Rz_ul"],
        },
    )


def tri_square_model_from_fields(
    ports: DualPortSet, fields: Dict[str, Any]
) -> TriSquareModel:
    topology = str(fields.get("topology", ""))
    if topology and topology not in (
        TRI_SQUARE_TOPOLOGY,
        TRI_SQUARE_TOPOLOGY_LEGACY,
        "tri_square_bilinear",
    ):
        raise ValueError(f"unexpected topology: {topology!r}")
    if not fields.get("l2_xy") or not fields.get("via_landing_l3"):
        return build_tri_square_model(
            ports,
            coarsen_k=int(fields.get("coarsen_k", 2)),
            via_stub=parse_via_stub_arg(fields.get("via_stub", VIA_STUB_RXRY)),
        )
    rx = float(fields.get("Rx", 0.0))
    ry = float(fields.get("Ry", 0.0))
    rz = float(fields.get("Rz", 0.0))
    return TriSquareModel(
        ports=ports,
        pitch_u=float(fields["pitch_u"]),
        pitch_l=float(fields["pitch_l"]),
        coarsen_k=int(fields["coarsen_k"]),
        nx_u=int(fields["nx_u"]),
        ny_u=int(fields["ny_u"]),
        nx_l=int(fields["nx_l"]),
        ny_l=int(fields["ny_l"]),
        l2_xy=[(float(x), float(y)) for x, y in fields.get("l2_xy", [])],
        l3_xy=[(float(x), float(y)) for x, y in fields.get("l3_xy", [])],
        ew_u=[(int(a), int(b)) for a, b in fields.get("ew_u", [])],
        ew_u_L=[float(x) for x in fields.get("ew_u_L", [])],
        ns_u=[(int(a), int(b)) for a, b in fields.get("ns_u", [])],
        ns_u_L=[float(x) for x in fields.get("ns_u_L", [])],
        ew_l=[(int(a), int(b)) for a, b in fields.get("ew_l", [])],
        ew_l_L=[float(x) for x in fields.get("ew_l_L", [])],
        ns_l=[(int(a), int(b)) for a, b in fields.get("ns_l", [])],
        ns_l_L=[float(x) for x in fields.get("ns_l_L", [])],
        vias_ul=[(int(a), int(b)) for a, b in fields.get("vias_ul", [])],
        via_landing_l3=[int(x) for x in fields.get("via_landing_l3", [])],
        pad_landing=[int(x) for x in fields.get("pad_landing", [])],
        sink_landing=[int(x) for x in fields.get("sink_landing", [])],
        n_l2_grid=int(fields.get("n_l2_grid", 0)),
        n_l3_grid=int(fields.get("n_l3_grid", 0)),
        Rx_u=float(fields.get("Rx_u", rx)),
        Ry_u=float(fields.get("Ry_u", ry)),
        Rx_l=float(fields.get("Rx_l", rx)),
        Ry_l=float(fields.get("Ry_l", ry)),
        Rz_pad=float(fields.get("Rz_pad", rz)),
        Rz_ul=float(fields.get("Rz_ul", rz)),
        via_stub=parse_via_stub_arg(fields.get("via_stub", VIA_STUB_RXRY)),
        extract_meta=dict(fields.get("extract_meta", {})),
    )


def build_Gs_tri_square(
    model: TriSquareModel,
    *,
    Rx_u: Optional[float] = None,
    Ry_u: Optional[float] = None,
    Rx_l: Optional[float] = None,
    Ry_l: Optional[float] = None,
    Rz_pad: Optional[float] = None,
    Rz_ul: Optional[float] = None,
    # Legacy shared-globals kwargs (Rx=Rx_u=Rx_l, …)
    Rx: Optional[float] = None,
    Ry: Optional[float] = None,
    Rz: Optional[float] = None,
    **_legacy: Any,
) -> np.ndarray:
    """Node order: [pads | L2 | L3 | sinks].

    Parameters (6): Rx_u, Ry_u, Rx_l, Ry_l, Rz_pad, Rz_ul.
    Legacy ``Rx, Ry, Rz`` still accepted and applied to both sheets / both vias.
    """
    del _legacy
    if Rx is not None or Ry is not None or Rz is not None:
        if Rx_u is None:
            Rx_u = Rx
        if Ry_u is None:
            Ry_u = Ry
        if Rx_l is None:
            Rx_l = Rx
        if Ry_l is None:
            Ry_l = Ry
        if Rz_pad is None:
            Rz_pad = Rz
        if Rz_ul is None:
            Rz_ul = Rz
    if None in (Rx_u, Ry_u, Rx_l, Ry_l, Rz_pad, Rz_ul):
        raise TypeError(
            "build_Gs_tri_square needs Rx_u,Ry_u,Rx_l,Ry_l,Rz_pad,Rz_ul "
            "(or legacy Rx,Ry,Rz)"
        )
    Rx_u, Ry_u = float(Rx_u), float(Ry_u)
    Rx_l, Ry_l = float(Rx_l), float(Ry_l)
    Rz_pad, Rz_ul = float(Rz_pad), float(Rz_ul)
    model.Rx_u, model.Ry_u = Rx_u, Ry_u
    model.Rx_l, model.Ry_l = Rx_l, Ry_l
    model.Rz_pad, model.Rz_ul = Rz_pad, Rz_ul
    floor_r = 1e-12
    pu = max(model.pitch_u, 1e-30)
    pl = max(model.pitch_l, 1e-30)

    n_pad, n_sink = model.n_pads, model.n_sinks
    n2, n3 = model.n_l2, model.n_l3
    n = n_pad + n2 + n3 + n_sink
    G = np.zeros((n, n), dtype=float)
    off2 = n_pad
    off3 = n_pad + n2
    off_s = n_pad + n2 + n3

    def stamp(i: int, j: int, g: float) -> None:
        if g <= 0:
            return
        G[i, i] += g
        G[j, j] += g
        G[i, j] -= g
        G[j, i] -= g

    def g_lat(Runit: float, length: float, pitch: float) -> float:
        return 1.0 / max(Runit * (float(length) / pitch), floor_r)

    for (a, b), L in zip(model.ew_u, model.ew_u_L):
        stamp(off2 + int(a), off2 + int(b), g_lat(Rx_u, L, pu))
    for (a, b), L in zip(model.ns_u, model.ns_u_L):
        stamp(off2 + int(a), off2 + int(b), g_lat(Ry_u, L, pu))
    for (a, b), L in zip(model.ew_l, model.ew_l_L):
        stamp(off3 + int(a), off3 + int(b), g_lat(Rx_l, L, pl))
    for (a, b), L in zip(model.ns_l, model.ns_l_L):
        stamp(off3 + int(a), off3 + int(b), g_lat(Ry_l, L, pl))

    g_pad = 1.0 / max(Rz_pad, floor_r)
    g_ul = 1.0 / max(Rz_ul, floor_r)
    # Mid→bot: L2 grid -- Rz_ul -- L3 landing (stubs already in ew_l/ns_l)
    for a, b in model.vias_ul:
        stamp(off2 + int(a), off3 + int(b), g_ul)
    for k, li in enumerate(model.pad_landing):
        stamp(int(k), off2 + int(li), g_pad)
    for j, li in enumerate(model.sink_landing):
        stamp(off3 + int(li), off_s + int(j), g_pad)

    port_idx = list(range(n_pad)) + list(range(off_s, n))
    internal_idx = list(range(off2, off_s))
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


# Back-compat name used by older extract code
def bilinear_attach(*_a, **_k):
    raise RuntimeError("bilinear_attach retired; tri-square uses stub attach")
