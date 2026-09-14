"""Unit tests for TSMC region lattice microns and virtual Pixel-R ports."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ports import (
    PORT_MODE_TSMC_VIRTUAL,
    build_tsmc_region_ports,
    ensure_tsmc_package_node,
    inject_virtual_pads,
)
from spice_parser import LAYER_VDD, NodeCoord, SpiceNetlist, parse_spice
from tsmc_region_grid import (
    DEFAULT_LATTICE,
    apply_region_micron_coords,
    collect_region_vdd_ports,
    is_filled_rectangle,
    is_virtual_pixel_r_lattice,
    parse_region_indices,
    vpad_name,
)

REPO = Path(__file__).resolve().parents[2]
SPEC = Path(__file__).resolve().parents[1] / "src"
SMOKE_SP = (
    REPO / "Benchmarks" / "TSMC" / "TC1" / "myALL_nportModel_without_package.sp"
)


def test_corner_microns():
    lat = DEFAULT_LATTICE
    assert lat.index_to_xy(1, 1) == (30.0, 30.0)
    assert lat.index_to_xy(43, 1) == (2550.0, 30.0)
    assert lat.index_to_xy(43, 33) == (2550.0, 1950.0)
    assert lat.index_to_xy(1, 33) == (30.0, 1950.0)
    assert lat.pitch == 60.0


def test_spice_float_suffixes():
    from spice_parser import parse_spice_float

    assert abs(parse_spice_float("1.0501918409m") - 1.0501918409e-3) < 1e-15
    assert abs(parse_spice_float("2k") - 2000.0) < 1e-12
    assert abs(parse_spice_float("1.0e-3") - 1.0e-3) < 1e-15


def test_parse_region_indices():
    name = "region_1_net_VDD_x_1_y_1_VDD_PORT"
    assert parse_region_indices(name) == (1, 1, "VDD")
    assert parse_region_indices("region_1420_net_VSS_x_1_y_1_VSS_PORT") == (
        1,
        1,
        "VSS",
    )


def test_filled_rectangle_and_lattice_detect():
    cells = [(1, 1), (2, 1), (1, 2), (2, 2)]
    assert is_filled_rectangle(cells)
    assert not is_filled_rectangle([(1, 1), (2, 1), (1, 2)])  # missing (2,2)

    net = SpiceNetlist()
    for ix, iy in cells:
        n = f"region_{ix}_net_VDD_x_{ix}_y_{iy}_VDD_PORT"
        net.node_names.add(n)
        net.coords[n] = NodeCoord(LAYER_VDD, float(ix), float(iy))
    assert is_virtual_pixel_r_lattice(net)

    # Smoke-sized (2 cells) must not take the virtual path.
    small = SpiceNetlist()
    for ix, iy in [(1, 1), (1, 2)]:
        n = f"region_{ix}_net_VDD_x_{ix}_y_{iy}_VDD_PORT"
        small.node_names.add(n)
    assert not is_virtual_pixel_r_lattice(small)


def test_apply_region_micron_coords():
    net = SpiceNetlist()
    n = "region_1_net_VDD_x_43_y_33_VDD_PORT"
    net.node_names.add(n)
    net.coords[n] = NodeCoord(LAYER_VDD, 43.0, 33.0)
    n_up = apply_region_micron_coords(net, DEFAULT_LATTICE, remap_tiles_without_bump=False)
    assert n_up == 1
    assert net.coords[n].x == 2550.0
    assert net.coords[n].y == 1950.0


def test_build_tsmc_region_ports_3x2():
    """Synthetic 3×2 lattice with tile mesh + isrc currents."""
    net = SpiceNetlist()
    # 3×2 region sinks
    sinks = {}
    rid = 0
    for iy in range(1, 3):
        for ix in range(1, 4):
            rid += 1
            sn = f"region_{rid}_net_VDD_x_{ix}_y_{iy}_VDD_PORT"
            sinks[(ix, iy)] = sn
            net.node_names.add(sn)
            net.coords[sn] = NodeCoord(LAYER_VDD, float(ix), float(iy))
            tile = f"tile001_{ix}_{iy}_net_VDD_x_{ix}_y_{iy}"
            net.node_names.add(tile)
            net.coords[tile] = NodeCoord(LAYER_VDD, float(ix), float(iy))
            net.resistors.append((tile, sn, 1.0e-2))
            net.currents.append((sn, "0", 1.0e-3))

    # Horizontal tile links
    for iy in range(1, 3):
        for ix in range(1, 3):
            a = f"tile001_{ix}_{iy}_net_VDD_x_{ix}_y_{iy}"
            b = f"tile001_{ix+1}_{iy}_net_VDD_x_{ix+1}_y_{iy}"
            net.resistors.append((a, b, 1.0))

    assert is_virtual_pixel_r_lattice(net)
    apply_region_micron_coords(net, DEFAULT_LATTICE, remap_tiles_without_bump=True)

    pkg, vdd = ensure_tsmc_package_node(net, vdd=0.9)
    ports = build_tsmc_region_ports(net, vdd=vdd, package_node=pkg)
    assert ports.port_mode == PORT_MODE_TSMC_VIRTUAL
    assert ports.n_pads == ports.n_sinks == 6
    assert ports.nx == 3 and ports.ny == 2
    assert ports.cell_size == 60.0
    # x_1_y_1 → (30,30); x_3_y_2 → (150, 90)
    c00 = next(c for c in ports.cells if c.ix == 0 and c.iy == 0)
    assert c00.x == 30.0 and c00.y == 30.0
    c21 = next(c for c in ports.cells if c.ix == 2 and c.iy == 1)
    assert c21.x == 150.0 and c21.y == 90.0
    assert ports.pad_nodes[0] == vpad_name(1, 1)
    assert all(abs(c.current + 1.0e-3) < 1e-15 for c in ports.cells)

    n_pads = inject_virtual_pads(net, ports, package_node=pkg)
    assert n_pads == 6
    assert pkg in net.node_names
    # Pads near-short to tiles, not to the package node.
    pad0 = ports.pad_nodes[0]
    tile0 = ports.pad_attach_tiles[0]
    assert tile0
    assert any(
        (a == pad0 and b == tile0) or (b == pad0 and a == tile0)
        for a, b, _ in net.resistors
    )
    assert not any(
        (a == pad0 and b == pkg) or (b == pad0 and a == pkg)
        for a, b, _ in net.resistors
    )


def _write_lattice_deck(dir_path: Path, nx: int = 3, ny: int = 2) -> Path:
    """Minimal hierarchical deck that triggers the virtual Pixel-R path."""
    pins = []
    body_r = []
    body_i = []
    rid = 0
    for iy in range(1, ny + 1):
        for ix in range(1, nx + 1):
            rid += 1
            sn = f"region_{rid}_net_VDD_x_{ix}_y_{iy}_VDD_PORT"
            tile = f"tile001_{ix}_{iy}_net_VDD_x_{ix}_y_{iy}"
            pins.append(sn)
            body_r.append(f"R_t{rid} {tile} {sn} 1e-2")
            body_i.append(f"I_{rid} {sn} 0 1.0e-3")
    for iy in range(1, ny + 1):
        for ix in range(1, nx):
            a = f"tile001_{ix}_{iy}_net_VDD_x_{ix}_y_{iy}"
            b = f"tile001_{ix+1}_{iy}_net_VDD_x_{ix+1}_y_{iy}"
            body_r.append(f"R_h{iy}_{ix} {a} {b} 1.0")

    sub = dir_path / "lat.sp.subckt"
    top = dir_path / "lat_top.sp"
    pin_line = " ".join(pins)
    sub.write_text(
        ".SUBCKT latModel "
        + pin_line
        + "\n"
        + "\n".join(body_r)
        + "\n"
        + "\n".join(body_i)
        + "\n.ENDS\n"
    )
    # Tie first pin to VDD_in as a crude package; rest are free region ports.
    actuals = ["VDD_in"] + pins[1:]
    # Subckt expects all region pins — map first to VDD_in, others keep names.
    # Simpler: leave all pins as region names and add VVDD on a tile via include body.
    top.write_text(
        f"include {sub.name}\n"
        f"VVDD VDD_in 0 DC 0.9\n"
        f"Xdie1 {' '.join(pins)} latModel\n"
        # Package short: VDD_in to first tile through a tiny R in a wrapper — attach by
        # also naming a top-level R from VDD_in into the first tile after flatten.
        # Use an extra resistor at top that references a hierarchical internal? Not flattened.
        # Instead put VDD_in as an extra subckt pin.
        "\n"
    )
    # Rebuild with VDD_in as first pin of subckt connected to first tile.
    first_tile = "tile001_1_1_net_VDD_x_1_y_1"
    sub.write_text(
        ".SUBCKT latModel VDD_in "
        + pin_line
        + "\n"
        + f"R_pkg VDD_in {first_tile} 1e-15\n"
        + "\n".join(body_r)
        + "\n"
        + "\n".join(body_i)
        + "\n.ENDS\n"
    )
    top.write_text(
        f"include {sub.name}\n"
        f"VVDD VDD_in 0 DC 0.9\n"
        f"Xdie1 VDD_in {' '.join(pins)} latModel\n"
    )
    return top


def test_stage01_02_virtual_lattice(tmp_path):
    top = _write_lattice_deck(tmp_path)
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(SPEC / "stage01_parse.py"), str(top), str(out)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    net = parse_spice(top)
    assert is_virtual_pixel_r_lattice(net)
    # Parser should have applied micron map.
    p = collect_region_vdd_ports(net.node_names)
    n11 = p[(1, 1)]
    assert net.coords[n11].x == 30.0
    assert net.coords[n11].y == 30.0

    proc2 = subprocess.run(
        [sys.executable, str(SPEC / "stage02_ports.py"), str(out)],
        capture_output=True,
        text=True,
    )
    assert proc2.returncode == 0, proc2.stderr + proc2.stdout
    ports = json.loads((out / "comp1" / "ports.json").read_text())
    assert ports["port_mode"] == PORT_MODE_TSMC_VIRTUAL
    assert len(ports["pad_nodes"]) == 6
    assert ports["cell_size"] == 60.0
    assert ports["pad_nodes"][0].startswith("vpad_")
    cells = {(c["ix"], c["iy"]): c for c in ports["cells"]}
    assert cells[(0, 0)]["x"] == 30.0
    assert cells[(0, 0)]["y"] == 30.0


@pytest.mark.skipif(not SMOKE_SP.is_file(), reason="TSMC TC1 smoke deck missing")
def test_smoke_deck_virtual_lattice():
    net = parse_spice(SMOKE_SP)
    assert is_virtual_pixel_r_lattice(net)
    assert net.coords["region_1_net_VDD_x_1_y_1_VDD_PORT"].x == 30.0
    assert net.coords["region_1_net_VDD_x_1_y_1_VDD_PORT"].y == 30.0
    assert any(n1 == "VDD_in" and abs(v - 0.9) < 1e-15 for n1, _n2, v in net.voltages)
