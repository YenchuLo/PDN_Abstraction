"""SPICE parser for IBM flat decks and TSMC hierarchical n-port models."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# IBM: n<layer>_<x>_<y>  (optional _X_ prefix)
NODE_RE = re.compile(
    r"^(?:_X_)?n(?P<layer>\d+)_(?P<x>-?\d+)_(?P<y>-?\d+)$",
    re.IGNORECASE,
)

# TSMC tile / region node names
TILE_RE = re.compile(
    r"^tile(?P<tile>\d+)_(?P<ix>\d+)_(?P<iy>\d+)_net_(?P<net>VDD|VSS)"
    r"_x_(?P<x>-?\d+(?:\.\d+)?)_y_(?P<y>-?\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)
REGION_RE = re.compile(
    r"^region_(?P<rid>\d+)_net_(?P<net>VDD|VSS)"
    r"_x_(?P<x>-?\d+(?:\.\d+)?)_y_(?P<y>-?\d+(?:\.\d+)?)"
    r"_(?P<net2>VDD|VSS)_PORT$",
    re.IGNORECASE,
)
LOOSE_TSMC_RE = re.compile(
    r"_net_(?P<net>VDD|VSS)_x_(?P<x>-?\d+(?:\.\d+)?)_y_(?P<y>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
BUMP_RE = re.compile(
    r"^\*\s*BUMP_(?P<net>VDD|VSS)_(?P<bx>\d+)_(?P<by>\d+)\s+"
    r"(?P<node>\S+)\s+(?P<net2>VDD|VSS)\s+"
    r"(?P<x>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
    r"(?P<y>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
FLOAT_RE = re.compile(
    r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$"
)
# SPICE engineering suffixes (Spectre/ngspice): 1.05m → 1.05e-3, 2k → 2000, …
_SPICE_SCALE = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "x": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "a": 1e-18,
}
SPICE_FLOAT_RE = re.compile(
    r"^(?P<num>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?P<suf>meg|[tgkxmunpfa])?$",
    re.IGNORECASE,
)
INCLUDE_RE = re.compile(
    r"^\.?(?:include|INCLUDE)\s+[\"']?([^\"']+)[\"']?\s*$"
)

# Synthetic layers for TSMC dual-rail nets (IBM uses numeric metal layers).
LAYER_VSS = 0
LAYER_VDD = 1
# Tile / bump-pad mesh sits above region VDD_PORT sinks (two-layer G_M).
LAYER_TILE = 2
DEFAULT_PORT_VDD = 0.9


@dataclass
class NodeCoord:
    layer: int
    x: float
    y: float


@dataclass
class BumpSite:
    """One ``* BUMP_{VDD|VSS}_{bx}_{by} <node> … <x> <y>`` occupancy record."""

    net: str  # VDD or VSS
    bx: int
    by: int
    node: str
    x: float
    y: float


@dataclass
class SpiceNetlist:
    """Parsed resistive DC netlist (IBM flat or flattened TSMC n-port)."""

    resistors: List[Tuple[str, str, float]] = field(default_factory=list)
    currents: List[Tuple[str, str, float]] = field(default_factory=list)
    # (node_pos, node_neg, voltage) — node_neg is usually "0"
    voltages: List[Tuple[str, str, float]] = field(default_factory=list)
    coords: Dict[str, NodeCoord] = field(default_factory=dict)
    node_names: set = field(default_factory=set)
    # All BUMP comment sites (same tile may appear at several (bx, by)).
    bumps: List[BumpSite] = field(default_factory=list)

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        xs = [c.x for c in self.coords.values()]
        ys = [c.y for c in self.coords.values()]
        if not xs:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class _SubcktDef:
    name: str
    pins: List[str]
    body: List[str]  # logical lines inside the subckt (no .SUBCKT/.ENDS)


def parse_node_coord(name: str) -> Optional[NodeCoord]:
    """Parse IBM or TSMC-style node name into (layer, x, y)."""
    name = _normalize_node_token(name)
    m = NODE_RE.match(name)
    if m:
        return NodeCoord(
            layer=int(m.group("layer")),
            x=float(m.group("x")),
            y=float(m.group("y")),
        )
    m = TILE_RE.match(name)
    if m:
        net = m.group("net").upper()
        return NodeCoord(
            layer=LAYER_TILE if net == "VDD" else LAYER_VSS,
            x=float(m.group("x")),
            y=float(m.group("y")),
        )
    m = REGION_RE.match(name)
    if m:
        net = m.group("net").upper()
        return NodeCoord(
            layer=LAYER_VDD if net == "VDD" else LAYER_VSS,
            x=float(m.group("x")),
            y=float(m.group("y")),
        )
    m = LOOSE_TSMC_RE.search(name)
    if m:
        net = m.group("net").upper()
        # Loose match: prefer tile layer when the name looks like a tile.
        layer = LAYER_VDD if net == "VDD" else LAYER_VSS
        if net == "VDD" and "tile" in name.lower():
            layer = LAYER_TILE
        return NodeCoord(
            layer=layer,
            x=float(m.group("x")),
            y=float(m.group("y")),
        )
    return None


def _normalize_node_token(tok: str) -> str:
    """Fix accidental spaces inside node names (e.g. 'region 637_net_...')."""
    t = tok.strip()
    if " " not in t:
        return t
    return re.sub(r"\s+", "_", t)


def _is_number(tok: str) -> bool:
    return bool(SPICE_FLOAT_RE.match(tok))


def parse_spice_float(tok: str) -> float:
    """Parse a SPICE float token, including engineering suffixes (``1.05m``)."""
    m = SPICE_FLOAT_RE.match(tok.strip())
    if not m:
        raise ValueError(f"not a SPICE float: {tok!r}")
    val = float(m.group("num"))
    suf = (m.group("suf") or "").lower()
    if suf:
        val *= _SPICE_SCALE[suf]
    return val


def _element_value_token(toks: Sequence[str]) -> Optional[str]:
    """
    Extract the numeric value token from an R/I/V element line.

    Supports:
    - ``R1 a b 1.2`` / ``I1 a b 1.05m``
    - Spectre/SPICE independent sources: ``VVDD VDD_in 0 DC 0.9``
    - Fallback: last numeric token on the line
    """
    if len(toks) < 4:
        return None
    # ``Vname n+ n- DC <val>`` (optional further tokens ignored)
    if len(toks) >= 5 and toks[3].upper() == "DC" and _is_number(toks[4]):
        return toks[4]
    if _is_number(toks[3]):
        return toks[3]
    for tok in reversed(toks[3:]):
        if _is_number(tok):
            return tok
    return None


def _is_gnd(name: str) -> bool:
    return name == "0" or name.lower() == "gnd"


def _register_node(
    net: SpiceNetlist,
    name: str,
    *,
    bump_xy: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    if _is_gnd(name):
        return
    name = _normalize_node_token(name)
    net.node_names.add(name)
    if name in net.coords:
        return
    if bump_xy and name in bump_xy:
        # Prefer physical bump coordinates; keep layer from name if possible.
        coord = parse_node_coord(name)
        layer = coord.layer if coord is not None else LAYER_VDD
        bx, by = bump_xy[name]
        net.coords[name] = NodeCoord(layer=layer, x=bx, y=by)
        return
    coord = parse_node_coord(name)
    if coord is not None:
        net.coords[name] = coord


def _merge_broken_region_tokens(tokens: Sequence[str]) -> List[str]:
    """Re-join ``region 637_net_...`` split tokens into one node name."""
    merged: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if (
            tok.lower() == "region"
            and i + 1 < len(tokens)
            and not _is_number(tokens[i + 1])
        ):
            merged.append(_normalize_node_token(tok + " " + tokens[i + 1]))
            i += 2
            continue
        merged.append(tok)
        i += 1
    return merged


def _iter_logical_lines(path: Path) -> Iterable[str]:
    """
    Yield logical lines with ``+`` continuations folded.

    Blank lines and ``*`` comments do **not** end a continued statement, so
    decks like::

        Xdie1 a b
        * section header
        + c d mystub

    fold to ``Xdie1 a b c d mystub`` (comment emitted on its own in between).
    """
    buf: List[str] = []
    with path.open("r", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n\r")
            stripped = line.lstrip()
            if not stripped:
                continue
            # Comments are their own logical lines; never join with ``+`` and
            # never flush an in-progress continued statement.
            if stripped.startswith("*"):
                yield stripped
                continue
            if stripped.startswith("+"):
                cont = stripped[1:].lstrip()
                if not cont:
                    continue
                if buf:
                    buf.append(cont)
                else:
                    # Orphan continuation — treat as start of a statement.
                    buf = [cont]
                continue
            if buf:
                yield " ".join(buf)
            buf = [stripped]
        if buf:
            yield " ".join(buf)


def _resolve_include(base_dir: Path, raw_path: str) -> Path:
    p = Path(raw_path.strip().strip("\"'"))
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _looks_like_statement_start(line: str) -> bool:
    """
    True if ``line`` begins a new SPICE statement (not a bare wrap fragment).

    ``region_*`` / ``tile*`` / ``VDD_in`` are nodes, not elements — even though
    ``region_…`` starts with the letter R.
    """
    toks = line.split()
    if not toks:
        return False
    t = toks[0]
    if t.startswith(".") or t.startswith("*"):
        return True
    if t.lower() in ("include",):
        return True
    base = t.split(".")[-1]
    if not base:
        return False
    b0 = base[0].upper()
    if b0 == "X":
        return True
    if b0 == "R":
        return not base.lower().startswith("region")
    if b0 == "I":
        return True
    if b0 == "V":
        low = base.lower()
        if low in ("vdd_in", "vss_in"):
            return False
        if low.startswith("vdd_") or low.startswith("vss_"):
            return False
        if "vdd_port" in low or "vss_port" in low or low.startswith("vpad"):
            return False
        return True
    return False


def _riv_line_complete(line: str) -> bool:
    """True if line is a complete R/I/V with a parseable value token."""
    toks = _merge_broken_region_tokens(line.split())
    if len(toks) < 4:
        return False
    base = toks[0].split(".")[-1]
    if not base or base[0].upper() not in "RIV":
        return False
    return _element_value_token(toks) is not None


def _fold_bare_continuations(lines: Sequence[str]) -> List[str]:
    """
    Join R/I/V statements wrapped onto the next line *without* a leading ``+``.

    Real TSMC decks often break long resistor lines mid-statement::

        R_128593 tile001_8_13_net_VDD_x_8_y_13
        region_657_net_VDD_x_20_y_31_VDD_PORT 3.307e+14
    """
    out: List[str] = []
    buf: Optional[str] = None
    for logical in lines:
        if logical.startswith("*"):
            # Comments do not end an incomplete R/I/V.
            out.append(logical)
            continue
        if buf is not None:
            if not _looks_like_statement_start(logical):
                buf = f"{buf} {logical}"
                if _riv_line_complete(buf):
                    out.append(buf)
                    buf = None
                continue
            out.append(buf)
            buf = None
        if (
            _looks_like_statement_start(logical)
            and logical.split()
            and logical.split()[0].split(".")[-1][:1].upper() in "RIV"
            and not _riv_line_complete(logical)
        ):
            buf = logical
            continue
        out.append(logical)
    if buf is not None:
        out.append(buf)
    return out


def _expand_file_lines(
    path: Path,
    *,
    seen: Optional[Set[Path]] = None,
) -> List[str]:
    """
    Read ``path``, fold ``+`` lines, and recursively inline include directives.

    ``.SUBCKT`` / ``.ENDS`` / ``X`` / ``R`` / ``I`` / ``V`` are left intact for
    a later pass. Analysis cards (``.PRINT``, ``.op``, …) are dropped.
    """
    path = Path(path).resolve()
    if seen is None:
        seen = set()
    if path in seen:
        raise ValueError(f"circular include detected: {path}")
    seen.add(path)

    out: List[str] = []
    base = path.parent
    logicals = _fold_bare_continuations(list(_iter_logical_lines(path)))
    for logical in logicals:
        if logical.startswith("*"):
            # Keep BUMP comment lines for coordinate overrides.
            if BUMP_RE.match(logical):
                out.append(logical)
            continue

        m = INCLUDE_RE.match(logical.strip())
        if m:
            inc = _resolve_include(base, m.group(1))
            if not inc.is_file():
                raise FileNotFoundError(f"include not found: {inc} (from {path})")
            out.extend(_expand_file_lines(inc, seen=seen))
            continue

        up = logical.lstrip().upper()
        # Drop analysis / control cards; keep .SUBCKT / .ENDS.
        if up.startswith(".") and not (
            up.startswith(".SUBCKT") or up.startswith(".ENDS")
        ):
            continue
        out.append(logical)
    return out


def _collect_subckts(lines: Sequence[str]) -> Tuple[Dict[str, _SubcktDef], List[str]]:
    """Split expanded lines into subckt defs + top-level residual lines."""
    defs: Dict[str, _SubcktDef] = {}
    top: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        up = line.lstrip().upper()
        if up.startswith(".SUBCKT"):
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"malformed .SUBCKT: {line}")
            name = parts[1]
            pins = [_normalize_node_token(p) for p in parts[2:]]
            body: List[str] = []
            i += 1
            while i < n and not lines[i].lstrip().upper().startswith(".ENDS"):
                body.append(lines[i])
                i += 1
            if i >= n:
                raise ValueError(f".SUBCKT {name} missing .ENDS")
            defs[name] = _SubcktDef(name=name, pins=pins, body=body)
            i += 1  # skip .ENDS
            continue
        top.append(line)
        i += 1
    return defs, top


# Package / top-rail nodes that short tile pins in without-package decks.
_PACKAGE_NODES = frozenset({"vdd_in", "vss_in"})


def _is_package_node(name: str) -> bool:
    return _normalize_node_token(name).lower() in _PACKAGE_NODES


def _is_tile_formal(name: str) -> bool:
    """True if a subckt pin name is a tile mesh node (not a region port)."""
    base = _normalize_node_token(name).split(".")[-1]
    if TILE_RE.match(base):
        return True
    return "tile" in base.lower() and LOOSE_TSMC_RE.search(base) is not None


def _rename_node(name: str, pin_map: Dict[str, str], prefix: str) -> str:
    if _is_gnd(name):
        return "0"
    name = _normalize_node_token(name)
    if name in pin_map:
        return pin_map[name]
    if not prefix:
        return name
    return f"{prefix}.{name}"


def _expand_x_instance(
    toks: Sequence[str],
    defs: Dict[str, _SubcktDef],
    *,
    inst_prefix: str,
) -> List[str]:
    """
    Expand ``Xname n1 n2 ... subckt_name`` into flat element lines.

    Internal (non-pin) nodes are prefixed with ``inst_prefix`` (instance name).

    Tile formals wired to package nodes (``VDD_in`` / ``VSS_in``) are **not**
    remapped so the die R-mesh keeps distinct tile nodes for bump-pad attach.
    """
    if len(toks) < 2:
        return []
    inst = toks[0]
    sub_name = toks[-1]
    actuals = [_normalize_node_token(t) for t in toks[1:-1]]
    if sub_name not in defs:
        raise KeyError(f"undefined subckt '{sub_name}' for instance {inst}")
    sub = defs[sub_name]
    if len(actuals) != len(sub.pins):
        raise ValueError(
            f"instance {inst}: subckt {sub_name} expects {len(sub.pins)} pins, "
            f"got {len(actuals)}"
        )
    pin_map: Dict[str, str] = {}
    for formal, actual in zip(sub.pins, actuals):
        # Keep tile mesh nodes when the top deck shorts them to package.
        # Map formal→formal so they are not prefixed as instance-locals.
        if _is_tile_formal(formal) and _is_package_node(actual):
            pin_map[formal] = formal
            continue
        pin_map[formal] = actual
    # Nested instances get a dotted prefix under this instance.
    prefix = inst if not inst_prefix else f"{inst_prefix}.{inst}"

    out: List[str] = []
    for bline in sub.body:
        if bline.startswith("*"):
            out.append(bline)
            continue
        btoks = _merge_broken_region_tokens(bline.split())
        if not btoks:
            continue
        kind = btoks[0][0].upper()
        if kind == "X":
            # Recursively expand nested X with hierarchical prefix.
            out.extend(
                _expand_x_instance(btoks, defs, inst_prefix=prefix)
            )
            continue
        if kind in ("R", "I", "V") and len(btoks) >= 4:
            n1 = _rename_node(btoks[1], pin_map, prefix)
            n2 = _rename_node(btoks[2], pin_map, prefix)
            rest = " ".join(btoks[3:])
            out.append(f"{prefix}.{btoks[0]} {n1} {n2} {rest}")
            continue
        # Unknown body line: skip
    return out


def _flatten_top(lines: Sequence[str], defs: Dict[str, _SubcktDef]) -> List[str]:
    """Expand top-level X instances; keep R/I/V and BUMP comments."""
    out: List[str] = []
    for line in lines:
        if line.startswith("*"):
            out.append(line)
            continue
        toks = _merge_broken_region_tokens(line.split())
        if not toks:
            continue
        if toks[0][0].upper() == "X":
            out.extend(_expand_x_instance(toks, defs, inst_prefix=""))
            continue
        out.append(line)
    return out


def _parse_elements(
    lines: Sequence[str],
    *,
    synthesize_port_v: bool = True,
) -> SpiceNetlist:
    # Late import avoids a circular dependency at module load.
    from tsmc_region_grid import (
        DEFAULT_LATTICE,
        apply_region_micron_coords,
        is_virtual_pixel_r_lattice,
        lattice_from_cells,
        collect_region_vdd_ports,
    )

    net = SpiceNetlist()
    bump_xy: Dict[str, Tuple[float, float]] = {}
    port_nodes: List[str] = []

    for line in lines:
        if line.startswith("*"):
            m = BUMP_RE.match(line)
            if m:
                node = _normalize_node_token(m.group("node"))
                bx = int(m.group("bx"))
                by = int(m.group("by"))
                x = float(m.group("x"))
                y = float(m.group("y"))
                net_name = m.group("net").upper()
                net.bumps.append(
                    BumpSite(net=net_name, bx=bx, by=by, node=node, x=x, y=y)
                )
                # First site wins for micron override on the tile node.
                if node not in bump_xy:
                    bump_xy[node] = (x, y)
            continue

        toks = _merge_broken_region_tokens(line.split())
        if len(toks) < 4:
            continue
        # Hierarchical names look like ``Xdie1.R_1`` — kind is the last segment.
        ename = toks[0]
        base = ename.split(".")[-1]
        if not base:
            continue
        kind = base[0].upper()
        n1 = _normalize_node_token(toks[1])
        n2 = _normalize_node_token(toks[2])
        val_tok = _element_value_token(toks)
        if val_tok is None:
            continue
        try:
            val = parse_spice_float(val_tok)
        except ValueError:
            continue

        if kind == "R":
            if val <= 0.0:
                continue
            net.resistors.append((n1, n2, val))
            _register_node(net, n1, bump_xy=bump_xy)
            _register_node(net, n2, bump_xy=bump_xy)
        elif kind == "I":
            net.currents.append((n1, n2, val))
            _register_node(net, n1, bump_xy=bump_xy)
            _register_node(net, n2, bump_xy=bump_xy)
        elif kind == "V":
            net.voltages.append((n1, n2, val))
            _register_node(net, n1, bump_xy=bump_xy)
            _register_node(net, n2, bump_xy=bump_xy)

        for n in (n1, n2):
            if n.upper().endswith("_VDD_PORT") or (
                REGION_RE.match(n) and n.upper().find("_VDD_") >= 0
            ):
                if n not in port_nodes:
                    port_nodes.append(n)

    # Apply bump overrides after element registration (in case BUMP follows R).
    for node, (bx, by) in bump_xy.items():
        if node in net.coords:
            c = net.coords[node]
            layer = c.layer
            # Prefer tile layer for VDD bump nodes.
            if any(b.node == node and b.net == "VDD" for b in net.bumps):
                layer = LAYER_TILE
            net.coords[node] = NodeCoord(layer=layer, x=bx, y=by)
        else:
            _register_node(net, node, bump_xy=bump_xy)

    use_virtual = is_virtual_pixel_r_lattice(net)
    if use_virtual:
        # Region (and tile-without-BUMP) indices → microns; skip neighbor snap.
        cells = list(collect_region_vdd_ports(net.node_names).keys())
        lattice = lattice_from_cells(cells, base=DEFAULT_LATTICE)
        apply_region_micron_coords(
            net,
            lattice,
            remap_tiles_without_bump=True,
            bump_nodes=set(bump_xy.keys()),
        )
    else:
        # Smoke / sparse decks: snap region ports to tile/bump neighbors.
        _relocate_ports_to_neighbors(net, port_nodes)

    # Without a package, top-level decks often have no V sources. Seed pad
    # sites from region VDD_PORT nodes so auto pitch / VDD discovery work.
    # Virtual Pixel-R lattices keep region ports as sinks; stage-02 creates
    # virtual top pads (and a package node if needed).
    has_v = any(abs(v) > 1e-30 for *_, v in net.voltages)
    if (
        synthesize_port_v
        and not use_virtual
        and not has_v
    ):
        for pn in port_nodes:
            if pn in net.node_names:
                net.voltages.append((pn, "0", DEFAULT_PORT_VDD))
                _register_node(net, pn, bump_xy=bump_xy)

    return net


def _relocate_ports_to_neighbors(
    net: SpiceNetlist, port_nodes: Sequence[str]
) -> None:
    """Move region ports to mean (x,y) of connected non-port neighbors."""
    if not port_nodes:
        return
    port_set = {_normalize_node_token(p) for p in port_nodes}
    nbrs: Dict[str, List[str]] = {p: [] for p in port_set}
    for n1, n2, _ in net.resistors:
        if n1 in port_set and n2 not in port_set:
            nbrs[n1].append(n2)
        if n2 in port_set and n1 not in port_set:
            nbrs[n2].append(n1)
    for pn, friends in nbrs.items():
        xs: List[float] = []
        ys: List[float] = []
        for f in friends:
            c = net.coords.get(f)
            if c is None:
                continue
            xs.append(c.x)
            ys.append(c.y)
        if not xs:
            # Drop misleading index coords so they do not pollute bbox.
            net.coords.pop(pn, None)
            continue
        # Keep region ports on the bottom VDD/VSS layer (not the tile layer).
        pu = pn.upper()
        if pu.endswith("_VSS_PORT") or "_NET_VSS_" in pu:
            layer = LAYER_VSS
        else:
            layer = LAYER_VDD
        net.coords[pn] = NodeCoord(
            layer=layer, x=sum(xs) / len(xs), y=sum(ys) / len(ys)
        )


def parse_spice(path: str | Path) -> SpiceNetlist:
    """
    Parse a SPICE deck into a flat resistive netlist.

    Supports:
    - IBM flat ``R`` / ``I`` / ``V`` decks (``nL_x_y`` nodes)
    - TSMC hierarchical n-port: ``include`` / ``.include``, ``.SUBCKT``,
      ``X`` instances, ``+`` continuations, tile/region node names, ``.isrc``
    """
    path = Path(path)
    expanded = _expand_file_lines(path)
    defs, top = _collect_subckts(expanded)
    flat = _flatten_top(top, defs) if defs else top
    # If the deck is only a subckt definition (no X), parse that body directly
    # so a lone ``.sp.subckt`` file still works.
    if not any(
        (not ln.startswith("*")) and ln.split() and ln.split()[0][0].upper() in "RIVX"
        for ln in flat
    ) and len(defs) == 1:
        only = next(iter(defs.values()))
        flat = list(only.body)
    return _parse_elements(flat)


def load_solution(path: str | Path) -> Dict[str, float]:
    """Load IBM .solution file: node_name voltage."""
    out: Dict[str, float] = {}
    path = Path(path)
    with path.open("r", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("*"):
                continue
            toks = line.split()
            if len(toks) < 2:
                continue
            try:
                out[toks[0]] = float(toks[1])
            except ValueError:
                continue
    return out


def injection_at_nodes(net: SpiceNetlist) -> Dict[str, float]:
    """
    Net current leaving the PDN into sinks at each non-ground node.

    Convention for IBM: `I n_sink 0 value` draws `value` from n_sink to ground,
    so the nodal current injected into the circuit at n_sink is -value.
    For `I 0 n_sink value`, injection at n_sink is +value.
    """
    inj: Dict[str, float] = {}
    for n1, n2, val in net.currents:
        if n1 != "0" and n1.lower() != "gnd":
            inj[n1] = inj.get(n1, 0.0) - val
        if n2 != "0" and n2.lower() != "gnd":
            inj[n2] = inj.get(n2, 0.0) + val
    return inj


def _node_layer(net: SpiceNetlist, name: str) -> Optional[int]:
    """Layer for a node from coords, else IBM/TSMC name parse."""
    c = net.coords.get(name)
    if c is not None:
        return int(c.layer)
    parsed = parse_node_coord(name)
    if parsed is None:
        return None
    return int(parsed.layer)


def _node_xy(net: SpiceNetlist, name: str) -> Optional[Tuple[float, float]]:
    """XY for a node from coords, else IBM/TSMC name parse."""
    c = net.coords.get(name)
    if c is not None:
        return float(c.x), float(c.y)
    parsed = parse_node_coord(name)
    if parsed is None:
        return None
    return float(parsed.x), float(parsed.y)


def parse_perturb_layer_arg(value: str | int) -> int | str:
    """Parse ``--perturb-layer``: metal int or ``all`` (current rail stack)."""
    if isinstance(value, int):
        return int(value)
    s = str(value).strip().lower()
    if s == "all":
        return "all"
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    raise ValueError(
        f"perturb_layer must be an integer metal layer or 'all' (got: {value})"
    )


def perturb_layer_resistors(
    net: SpiceNetlist,
    layer: int | str,
    amp: float,
    *,
    cell_size: float,
    seed: int = 0,
    layers: Optional[Sequence[int]] = None,
    roots: Optional[Set[str]] = None,
    node_map: Optional[Dict[str, str]] = None,
    min_factor: float = 1e-6,
) -> Dict[str, Any]:
    """
    Scale same-layer resistors by a **per-grid-cell** random factor.

    The lattice matches the port grid: ``net.bbox`` + ``cell_size``. One draw
    ``U ~ Uniform[-amp, amp]`` per cell ``(ix, iy)``; every matching same-layer R
    whose midpoint falls in that cell is multiplied by ``(1 + U)`` (clamped to
    at least ``min_factor``). Cross-layer vias are left unchanged.

    ``layer``:
      - ``int`` — only that metal
      - ``"all"`` — every metal in ``layers`` (the current VDD or VSS rail stack)

    When ``roots`` is set, only resistors whose endpoints map into ``roots``
    (via ``node_map``) are candidates — keeps VDD and VSS from mixing.

    Mutates ``net.resistors`` in place. Returns a small stats dict.
    """
    # Lazy import: ports imports spice_parser.
    from ports import _cell_index, grid_counts

    layer_spec = parse_perturb_layer_arg(layer)
    amp = float(amp)
    cell_size = float(cell_size)
    if amp < 0.0:
        raise ValueError(f"perturb amp must be >= 0 (got {amp})")
    if cell_size <= 0.0:
        raise ValueError(f"cell_size must be > 0 (got {cell_size})")
    if min_factor <= 0.0:
        raise ValueError("min_factor must be > 0")

    if layer_spec == "all":
        if not layers:
            raise ValueError(
                "perturb_layer='all' requires a non-empty layers= rail stack "
                "(VDD or VSS metals)"
            )
        target_layers = {int(L) for L in layers}
    else:
        target_layers = {int(layer_spec)}
        if layers is not None and int(layer_spec) not in {int(L) for L in layers}:
            raise ValueError(
                f"metal layer {layer_spec} is not in the rail stack {list(layers)}"
            )

    nmap = node_map if node_map is not None else {}

    def _root(n: str) -> str:
        if n == "0" or n.lower() == "gnd":
            return "0"
        return nmap.get(n, n)

    bbox = net.bbox
    nx, ny = grid_counts(bbox, cell_size)
    rng = np.random.default_rng(int(seed))
    # One factor per cell; draw all up front so unused cells still consume RNG
    # deterministically if we ever change touch logic.
    field = np.empty((ny, nx), dtype=float)
    for iy in range(ny):
        for ix in range(nx):
            u = float(rng.uniform(-amp, amp))
            field[iy, ix] = max(float(min_factor), 1.0 + u)

    new_rs: List[Tuple[str, str, float]] = []
    n_touch = 0
    factors: List[float] = []
    cells_used: Set[Tuple[int, int]] = set()
    layers_touched: Set[int] = set()

    for n1, n2, r in net.resistors:
        la = _node_layer(net, n1)
        lb = _node_layer(net, n2)
        if la is None or lb is None or la != lb or la not in target_layers:
            new_rs.append((n1, n2, r))
            continue
        if roots is not None:
            if _root(n1) not in roots or _root(n2) not in roots:
                new_rs.append((n1, n2, r))
                continue
        xy1 = _node_xy(net, n1)
        xy2 = _node_xy(net, n2)
        if xy1 is None or xy2 is None:
            new_rs.append((n1, n2, r))
            continue
        mx = 0.5 * (xy1[0] + xy2[0])
        my = 0.5 * (xy1[1] + xy2[1])
        ix, iy = _cell_index(mx, my, bbox, cell_size, nx, ny)
        fac = float(field[iy, ix])
        new_rs.append((n1, n2, float(r) * fac))
        n_touch += 1
        factors.append(fac)
        cells_used.add((ix, iy))
        layers_touched.add(int(la))

    if n_touch == 0:
        if layer_spec == "all":
            raise ValueError(
                f"no same-layer resistors found on rail layers {sorted(target_layers)} "
                f"to perturb"
            )
        raise ValueError(
            f"no same-layer resistors found on metal layer {layer_spec} to perturb"
        )

    net.resistors = new_rs
    arr = np.asarray(factors, dtype=float)
    return {
        "perturb_unit": "grid",
        "perturb_layer": layer_spec if layer_spec == "all" else int(layer_spec),
        "perturb_layers": sorted(layers_touched),
        "perturb_amp": amp,
        "perturb_seed": int(seed),
        "n_perturbed": n_touch,
        "n_cells_used": len(cells_used),
        "cell_size": cell_size,
        "nx": int(nx),
        "ny": int(ny),
        "factor_min": float(arr.min()),
        "factor_max": float(arr.max()),
        "factor_mean": float(arr.mean()),
    }


def perturb_layer_out_tag(layer: int | str, amp: float, seed: int) -> str:
    """OUT-dir suffix fragment when layer-R perturbation is enabled."""
    layer_spec = parse_perturb_layer_arg(layer)
    tag_l = "all" if layer_spec == "all" else str(int(layer_spec))
    return f"_pL{tag_l}a{float(amp):.6g}s{int(seed)}".replace("+", "")
