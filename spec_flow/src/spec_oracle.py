"""Independent IBM port oracle for TSMC-spec compliance checks.

Re-derives C4 voltage pads and per-grid lumped current sinks from a parsed
SPICE deck + VDD component roots. Does **not** call ``build_ports`` /
``build_dual_ports`` / ``_assign_pads_by_containment``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from graph import _root_of, build_short_map, vdd_connected_components
from spice_parser import SpiceNetlist, injection_at_nodes, parse_spice


def _is_gnd(name: str) -> bool:
    return name == "0" or name.lower() == "gnd"


def oracle_grid_counts(
    bbox: Tuple[float, float, float, float],
    cell_size: float,
) -> Tuple[int, int]:
    """Cells along each axis (same rule as ports.grid_counts)."""
    xmin, ymin, xmax, ymax = bbox
    if cell_size <= 0:
        raise ValueError("chip pitch / cell side length must be > 0")
    nx = max(1, int(math.ceil((xmax - xmin) / cell_size)))
    ny = max(1, int(math.ceil((ymax - ymin) / cell_size)))
    return nx, ny


def oracle_cell_index(
    x: float,
    y: float,
    bbox: Tuple[float, float, float, float],
    cell_size: float,
    nx: int,
    ny: int,
) -> Tuple[int, int]:
    xmin, ymin, _, _ = bbox
    ix = int((x - xmin) / cell_size) if cell_size > 0 else 0
    iy = int((y - ymin) / cell_size) if cell_size > 0 else 0
    ix = min(max(ix, 0), nx - 1)
    iy = min(max(iy, 0), ny - 1)
    return ix, iy


@dataclass(frozen=True)
class OracleC4:
    """One nonzero V→gnd site on a VDD component."""

    node: str  # short-map root
    live: str  # original SPICE live terminal
    x: float
    y: float
    voltage: float


@dataclass(frozen=True)
class OracleInjection:
    """One IBM current injection (into circuit) at a non-ground node."""

    node: str
    x: float
    y: float
    current: float  # nodal injection (IBM I n 0 val → -val)


@dataclass
class OraclePad:
    """Chosen pad for one overlapping grid cell."""

    ix: int
    iy: int
    node: str
    x: float
    y: float
    voltage: float
    n_c4_in_cell: int
    extras: List[OracleC4] = field(default_factory=list)


@dataclass
class OracleCell:
    """One chip-grid cell (always present; current may be 0)."""

    ix: int
    iy: int
    cx: float
    cy: float
    current: float
    has_pad: bool
    pad: Optional[OraclePad] = None


@dataclass
class SpecOracle:
    """Ground-truth ports for one VDD component at a given pitch."""

    bbox: Tuple[float, float, float, float]
    cell_size: float
    nx: int
    ny: int
    c4s: List[OracleC4]
    injections: List[OracleInjection]
    cells: List[OracleCell]  # row-major iy, ix
    pads: List[OraclePad]
    n_c4_collapsed: int  # C4s not chosen when several share a cell
    i_total: float

    @property
    def n_pads(self) -> int:
        return len(self.pads)

    @property
    def n_sinks(self) -> int:
        return len(self.cells)

    @property
    def n_nonzero_current_cells(self) -> int:
        return sum(1 for c in self.cells if abs(c.current) > 1e-30)

    @property
    def occupied_cells(self) -> Set[Tuple[int, int]]:
        return {(p.ix, p.iy) for p in self.pads}

    def current_map(self) -> Dict[Tuple[int, int], float]:
        return {(c.ix, c.iy): float(c.current) for c in self.cells}

    def pad_node_set(self) -> Set[str]:
        return {p.node for p in self.pads}

    def to_summary(self) -> dict:
        return {
            "cell_size": self.cell_size,
            "nx": self.nx,
            "ny": self.ny,
            "n_c4": len(self.c4s),
            "n_injections": len(self.injections),
            "n_pads": self.n_pads,
            "n_sinks": self.n_sinks,
            "n_c4_collapsed": self.n_c4_collapsed,
            "i_total": self.i_total,
            "n_nonzero_current_cells": sum(
                1 for c in self.cells if abs(c.current) > 1e-30
            ),
        }


def collect_c4s(
    net: SpiceNetlist,
    node_map: Dict[str, str],
    roots: Set[str],
) -> List[OracleC4]:
    """Nonzero V→gnd pads whose live terminal is on ``roots``."""
    pads: List[OracleC4] = []
    seen: Set[str] = set()
    for n1, n2, v in net.voltages:
        if abs(v) < 1e-12:
            continue
        a_g, b_g = _is_gnd(n1), _is_gnd(n2)
        if a_g == b_g:
            continue
        live = n2 if a_g else n1
        root = _root_of(node_map, live)
        if root not in roots or root in seen:
            continue
        coord = net.coords.get(live) or net.coords.get(root)
        if coord is None:
            continue
        seen.add(root)
        pads.append(
            OracleC4(
                node=root,
                live=str(live),
                x=float(coord.x),
                y=float(coord.y),
                voltage=float(v),
            )
        )
    pads.sort(key=lambda t: (t.y, t.x))
    return pads


def collect_injections(
    net: SpiceNetlist,
    node_map: Dict[str, str],
    roots: Set[str],
) -> List[OracleInjection]:
    """Nodal injections < 0 on ``roots`` (IBM sinks)."""
    out: List[OracleInjection] = []
    inj = injection_at_nodes(net)
    for node, cur in inj.items():
        if cur >= 0.0 or abs(cur) < 1e-30:
            continue
        root = _root_of(node_map, node)
        if root not in roots:
            continue
        coord = net.coords.get(node) or net.coords.get(root)
        if coord is None:
            continue
        out.append(
            OracleInjection(
                node=str(node),
                x=float(coord.x),
                y=float(coord.y),
                current=float(cur),
            )
        )
    return out


def build_spec_oracle(
    net: SpiceNetlist,
    cell_size: float,
    *,
    node_map: Optional[Dict[str, str]] = None,
    roots: Optional[Set[str]] = None,
) -> SpecOracle:
    """
    Build independent ground-truth ports for one VDD component.

    - Pad iff ≥1 C4 XY falls in the cell (nearest C4 to cell center if several).
    - Current = sum of IBM injections whose XY falls in the cell.
    - One sink cell for every nx×ny grid slot (current may be 0).
    """
    cell_size = float(cell_size)
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0")
    if node_map is None:
        node_map = build_short_map(net)
    if roots is None:
        comps = vdd_connected_components(net, node_map)
        if not comps:
            raise ValueError("no VDD connected components")
        roots = comps[0]
    else:
        roots = set(roots)

    bbox = tuple(float(x) for x in net.bbox)  # type: ignore[assignment]
    xmin, ymin, _, _ = bbox
    nx, ny = oracle_grid_counts(bbox, cell_size)
    c4s = collect_c4s(net, node_map, roots)
    injections = collect_injections(net, node_map, roots)

    # Bucket C4s by cell
    c4_buckets: Dict[Tuple[int, int], List[OracleC4]] = {}
    for c4 in c4s:
        ix, iy = oracle_cell_index(c4.x, c4.y, bbox, cell_size, nx, ny)
        c4_buckets.setdefault((ix, iy), []).append(c4)

    # Lump currents by cell
    current_buckets: Dict[Tuple[int, int], float] = {}
    for inj in injections:
        ix, iy = oracle_cell_index(inj.x, inj.y, bbox, cell_size, nx, ny)
        current_buckets[(ix, iy)] = current_buckets.get((ix, iy), 0.0) + inj.current

    cells: List[OracleCell] = []
    pads: List[OraclePad] = []
    n_collapsed = 0
    for iy in range(ny):
        for ix in range(nx):
            cx = xmin + (ix + 0.5) * cell_size
            cy = ymin + (iy + 0.5) * cell_size
            bucket = c4_buckets.get((ix, iy), [])
            pad: Optional[OraclePad] = None
            if bucket:
                best = min(bucket, key=lambda t: (t.x - cx) ** 2 + (t.y - cy) ** 2)
                extras = [c for c in bucket if c.node != best.node]
                n_collapsed += len(extras)
                pad = OraclePad(
                    ix=ix,
                    iy=iy,
                    node=best.node,
                    x=best.x,
                    y=best.y,
                    voltage=best.voltage,
                    n_c4_in_cell=len(bucket),
                    extras=extras,
                )
                pads.append(pad)
            cells.append(
                OracleCell(
                    ix=ix,
                    iy=iy,
                    cx=float(cx),
                    cy=float(cy),
                    current=float(current_buckets.get((ix, iy), 0.0)),
                    has_pad=pad is not None,
                    pad=pad,
                )
            )

    i_total = float(sum(inj.current for inj in injections))
    return SpecOracle(
        bbox=bbox,  # type: ignore[arg-type]
        cell_size=cell_size,
        nx=nx,
        ny=ny,
        c4s=c4s,
        injections=injections,
        cells=cells,
        pads=pads,
        n_c4_collapsed=n_collapsed,
        i_total=i_total,
    )


def build_spec_oracle_from_spice(
    spice_path: str,
    cell_size: float,
    *,
    component_index: int = 1,
) -> Tuple[SpiceNetlist, Dict[str, str], Set[str], SpecOracle]:
    """Parse spice, pick VDD component (1-based, topmost-first), build oracle."""
    net = parse_spice(spice_path)
    node_map = build_short_map(net)
    comps = vdd_connected_components(net, node_map)
    if not comps:
        raise ValueError("no VDD connected components")
    if component_index < 1 or component_index > len(comps):
        raise ValueError(
            f"component_index {component_index} out of range 1..{len(comps)}"
        )
    roots = comps[component_index - 1]
    oracle = build_spec_oracle(net, cell_size, node_map=node_map, roots=roots)
    return net, node_map, roots, oracle


# ---------------------------------------------------------------------------
# Comparison helpers (flow ports.json / PortSet-like vs oracle)
# ---------------------------------------------------------------------------


@dataclass
class ClauseResult:
    clause: str
    status: str  # pass | fail | skip | intentional_deviation | info
    detail: str
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "clause": self.clause,
            "status": self.status,
            "detail": self.detail,
            "metrics": self.metrics,
        }


def _cell_key_from_flow_cells(
    cells: Sequence[dict],
) -> Dict[Tuple[int, int], dict]:
    return {(int(c["ix"]), int(c["iy"])): c for c in cells}


def compare_pad_occupancy(
    oracle: SpecOracle,
    *,
    pad_nodes: Sequence[str],
    pad_attach: Sequence[int],
    cells: Sequence[dict],
    node_map: Optional[Dict[str, str]] = None,
) -> ClauseResult:
    """A1: pads iff C4 overlap; pad nodes ⊆ real C4s on this net."""
    by_cell = _cell_key_from_flow_cells(cells)
    flow_occ: Set[Tuple[int, int]] = set()
    for s in pad_attach:
        s = int(s)
        if s < 0 or s >= len(cells):
            return ClauseResult(
                "A1_pad_occupancy",
                "fail",
                f"pad_attach index {s} out of range (n_cells={len(cells)})",
            )
        c = cells[s]
        flow_occ.add((int(c["ix"]), int(c["iy"])))

    oracle_occ = oracle.occupied_cells
    missing = sorted(oracle_occ - flow_occ)
    extra = sorted(flow_occ - oracle_occ)

    # Resolve pad nodes through short map if provided
    def root(n: str) -> str:
        if node_map is None:
            return n
        return _root_of(node_map, n)

    c4_roots = {c.node for c in oracle.c4s}
    c4_lives = {c.live for c in oracle.c4s}
    bad_pads = [
        n
        for n in pad_nodes
        if root(str(n)) not in c4_roots
        and str(n) not in c4_roots
        and str(n) not in c4_lives
    ]

    if missing or extra or bad_pads:
        return ClauseResult(
            "A1_pad_occupancy",
            "fail",
            (
                f"occupied mismatch: missing={missing[:8]}{'…' if len(missing)>8 else ''} "
                f"extra={extra[:8]}{'…' if len(extra)>8 else ''} "
                f"non_c4_pads={bad_pads[:8]}"
            ),
            {
                "n_oracle_pads": float(len(oracle_occ)),
                "n_flow_pads": float(len(flow_occ)),
                "n_missing": float(len(missing)),
                "n_extra": float(len(extra)),
                "n_non_c4_pads": float(len(bad_pads)),
            },
        )
    return ClauseResult(
        "A1_pad_occupancy",
        "pass",
        f"n_pads={len(flow_occ)} matches oracle C4-overlap cells; all pad nodes are C4s",
        {
            "n_pads": float(len(flow_occ)),
            "n_c4_collapsed": float(oracle.n_c4_collapsed),
        },
    )


def compare_current_lumping(
    oracle: SpecOracle,
    *,
    cells: Sequence[dict],
    uniform_current: Optional[object] = None,
    atol: float = 1e-9,
    rtol: float = 1e-6,
) -> ClauseResult:
    """A2: per-cell lumped I matches oracle (skip if uniform-current override)."""
    if uniform_current not in (None, "", 0, "0", "off", "false", "none"):
        return ClauseResult(
            "A2_current_lumping",
            "skip",
            f"uniform_current={uniform_current!r} overrides real IBM I; not scored",
        )

    by_cell = _cell_key_from_flow_cells(cells)
    oracle_map = oracle.current_map()
    # Every oracle cell must exist in flow for a full-grid compare; if flow is
    # sparse, only compare overlapping keys and flag missing nonzero cells.
    max_abs = 0.0
    n_mismatch = 0
    missing_nonzero = 0
    for key, i_ref in oracle_map.items():
        if key not in by_cell:
            if abs(i_ref) > 1e-30:
                missing_nonzero += 1
            continue
        i_flow = float(by_cell[key]["current"])
        tol = atol + rtol * abs(i_ref)
        err = abs(i_flow - i_ref)
        max_abs = max(max_abs, err)
        if err > tol:
            n_mismatch += 1

    # Extra flow cells not in oracle grid (should not happen if same nx/ny)
    extra_keys = set(by_cell) - set(oracle_map)
    i_flow_total = sum(float(c["current"]) for c in cells)
    i_res = abs(i_flow_total - oracle.i_total)

    if n_mismatch or missing_nonzero:
        return ClauseResult(
            "A2_current_lumping",
            "fail",
            (
                f"per-cell mismatches={n_mismatch}, "
                f"missing_nonzero_cells={missing_nonzero}, "
                f"max_abs_err={max_abs:.6g}, "
                f"total_residual={i_res:.6g}"
            ),
            {
                "n_mismatch": float(n_mismatch),
                "missing_nonzero": float(missing_nonzero),
                "max_abs_err": float(max_abs),
                "i_total_oracle": float(oracle.i_total),
                "i_total_flow": float(i_flow_total),
                "i_residual": float(i_res),
                "n_extra_keys": float(len(extra_keys)),
            },
        )
    return ClauseResult(
        "A2_current_lumping",
        "pass",
        f"per-cell I matches oracle; |ΣI_flow−ΣI_oracle|={i_res:.3g}",
        {
            "max_abs_err": float(max_abs),
            "i_total_oracle": float(oracle.i_total),
            "i_total_flow": float(i_flow_total),
            "i_residual": float(i_res),
        },
    )


def compare_full_sink_grid(
    oracle: SpecOracle,
    *,
    n_sinks: int,
    nx: int,
    ny: int,
    flow_name: str = "",
) -> ClauseResult:
    """A3: every chip-grid cell has one current-sink port."""
    expected = oracle.nx * oracle.ny
    if nx != oracle.nx or ny != oracle.ny:
        return ClauseResult(
            "A3_full_sink_grid",
            "fail",
            f"grid shape flow=({nx},{ny}) vs oracle=({oracle.nx},{oracle.ny})",
            {"n_sinks": float(n_sinks), "expected": float(expected)},
        )
    if n_sinks == expected:
        return ClauseResult(
            "A3_full_sink_grid",
            "pass",
            f"n_sinks={n_sinks} == nx*ny={expected}",
            {"n_sinks": float(n_sinks), "expected": float(expected)},
        )
    # my_flow intentional-ish research choice still fails the written spec
    status = "fail"
    note = ""
    if flow_name == "my_flow" and n_sinks < expected:
        note = " (my_flow drops zero-I cells — fails written spec)"
    return ClauseResult(
        "A3_full_sink_grid",
        status,
        f"n_sinks={n_sinks} != nx*ny={expected}{note}",
        {
            "n_sinks": float(n_sinks),
            "expected": float(expected),
            "n_padless_expected": float(
                sum(1 for c in oracle.cells if not c.has_pad)
            ),
            "n_zero_i_cells": float(
                sum(1 for c in oracle.cells if abs(c.current) < 1e-30)
            ),
        },
    )


def compare_pad_voltages(
    oracle: SpecOracle,
    *,
    pad_voltages: Sequence[float],
    pad_nodes: Sequence[str],
    node_map: Optional[Dict[str, str]] = None,
    atol: float = 1e-9,
) -> ClauseResult:
    """A5: pad drive voltages should match IBM C4 voltages (same net)."""
    c4_by_root = {c.node: c.voltage for c in oracle.c4s}
    ibm_vals = sorted({round(c.voltage, 12) for c in oracle.c4s})

    def root(n: str) -> str:
        if node_map is None:
            return str(n)
        return _root_of(node_map, str(n))

    n_mismatch = 0
    n_matched = 0
    n_unmatched_node = 0
    forced_constant: Optional[float] = None
    if pad_voltages:
        uniq = {round(float(v), 12) for v in pad_voltages}
        if len(uniq) == 1:
            forced_constant = next(iter(uniq))
    for i, node in enumerate(pad_nodes):
        if i >= len(pad_voltages):
            break
        v_flow = float(pad_voltages[i])
        r = root(node)
        v_ref = c4_by_root.get(r)
        if v_ref is None:
            n_unmatched_node += 1
            continue
        n_matched += 1
        if abs(v_flow - v_ref) > atol:
            n_mismatch += 1

    if n_matched == 0 and pad_nodes:
        return ClauseResult(
            "A5_pad_voltage",
            "fail",
            (
                f"no pad_nodes matched oracle C4s "
                f"(n_pads={len(pad_nodes)}, IBM voltages={ibm_vals})"
            ),
            {
                "n_unmatched_node": float(n_unmatched_node),
                "forced_constant": float(forced_constant)
                if forced_constant is not None
                else float("nan"),
            },
        )
    if n_mismatch == 0 and n_unmatched_node == 0:
        return ClauseResult(
            "A5_pad_voltage",
            "pass",
            f"pad voltages match IBM C4s ({ibm_vals})",
            {"n_pads": float(len(pad_nodes))},
        )
    detail = (
        f"{n_mismatch}/{n_matched} matched pads differ from IBM C4 voltage; "
        f"unmatched_nodes={n_unmatched_node}; IBM C4 voltages={ibm_vals}"
    )
    if forced_constant is not None and forced_constant not in {
        float(v) for v in ibm_vals
    }:
        detail += f"; flow forces constant {forced_constant}"
    return ClauseResult(
        "A5_pad_voltage",
        "fail",
        detail,
        {
            "n_mismatch": float(n_mismatch),
            "n_matched": float(n_matched),
            "n_unmatched_node": float(n_unmatched_node),
            "forced_constant": float(forced_constant)
            if forced_constant is not None
            else float("nan"),
            "ibm_voltage": float(ibm_vals[0]) if ibm_vals else float("nan"),
        },
    )


def compare_gprime_shape(
    *,
    n_pads: int,
    n_sinks: int,
    gprime_shape: Optional[Sequence[int]],
) -> ClauseResult:
    """B: G' is (n_ports × n_ports)."""
    n_ports = int(n_pads) + int(n_sinks)
    if gprime_shape is None:
        return ClauseResult(
            "B_kron_shape",
            "skip",
            "Gprime.npy / system_meta not present",
        )
    sh = [int(x) for x in gprime_shape]
    if sh == [n_ports, n_ports]:
        return ClauseResult(
            "B_kron_shape",
            "pass",
            f"G' shape {sh} == n_ports={n_ports}",
            {"n_ports": float(n_ports)},
        )
    return ClauseResult(
        "B_kron_shape",
        "fail",
        f"G' shape {sh} != ({n_ports},{n_ports})",
        {"n_ports": float(n_ports)},
    )


def compare_gs_pad_coupling(
    *,
    n_pads: int,
    n_sinks: int,
    pad_attach: Sequence[int],
    Gs,
    atol: float = 1e-12,
) -> ClauseResult:
    """C: pad–sink coupling only on attached (overlap) sinks."""
    import numpy as np

    if Gs is None:
        return ClauseResult("C_pad_sink_coupling", "skip", "G_S not available")
    Gs = np.asarray(Gs, dtype=float)
    n_ports = n_pads + n_sinks
    if Gs.shape != (n_ports, n_ports):
        return ClauseResult(
            "C_pad_sink_coupling",
            "fail",
            f"G_S shape {Gs.shape} != ({n_ports},{n_ports})",
        )
    has_pad = [False] * n_sinks
    for s in pad_attach:
        s = int(s)
        if 0 <= s < n_sinks:
            has_pad[s] = True
    bad_occupied = 0
    bad_padless = 0
    for s, ok in enumerate(has_pad):
        couplings = [abs(Gs[p, n_pads + s]) for p in range(n_pads)]
        mx = max(couplings) if couplings else 0.0
        if ok and mx <= atol:
            bad_occupied += 1
        if (not ok) and mx > atol:
            bad_padless += 1
    if bad_occupied or bad_padless:
        return ClauseResult(
            "C_pad_sink_coupling",
            "fail",
            (
                f"occupied_without_coupling={bad_occupied}, "
                f"padless_with_coupling={bad_padless}"
            ),
            {
                "bad_occupied": float(bad_occupied),
                "bad_padless": float(bad_padless),
            },
        )
    return ClauseResult(
        "C_pad_sink_coupling",
        "pass",
        "G_S pad–sink coupling iff pad_attach",
        {"n_pads": float(n_pads), "n_sinks": float(n_sinks)},
    )


def classify_fit_method(fit_method: Optional[str], model_class: Optional[str]) -> ClauseResult:
    """D: record eigen vs intentional deviations."""
    fm = (fit_method or "").lower()
    mc = (model_class or "").lower()
    if fm in ("eigen", "eigenvalue", "eigen_ls") and (
        not mc or "pixel" in mc or mc in ("star_half_arm", "uniform_pixel_r")
    ):
        return ClauseResult(
            "D_eigen_fit",
            "pass",
            f"spec default eigen LS (fit={fit_method}, model={model_class})",
        )
    if "ir" in fm or "feng" in fm or "local" in mc or "tri_" in mc or "stagger" in mc:
        return ClauseResult(
            "D_eigen_fit",
            "intentional_deviation",
            f"not spec eigen LS: fit={fit_method!r}, model={model_class!r}",
        )
    if not fm and not mc:
        return ClauseResult(
            "D_eigen_fit",
            "skip",
            "fit method / model class not recorded in OUT",
        )
    return ClauseResult(
        "D_eigen_fit",
        "info",
        f"fit={fit_method!r}, model={model_class!r}",
    )
