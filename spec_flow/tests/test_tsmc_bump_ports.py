"""TSMC bump-tile pads: keep tiles, many-to-one, open-circuit empty cells."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from graph import assemble_conductance, build_short_map, galvanic_component_of, partition_ports
from kron import kron_reduce
from ports import (
    build_tsmc_region_ports,
    ensure_tsmc_package_node,
    inject_virtual_pads,
)
from spice_parser import LAYER_TILE, LAYER_VDD, parse_spice
from tsmc_region_grid import bump_index_base, collect_vdd_bumps, vpad_name


def test_flatten_keeps_tile_formals_when_x_maps_to_vdd_in(tmp_path: Path):
    """Xdie1 … VDD_in … must not rewrite tile R terminals to VDD_in."""
    sub = tmp_path / "die.sp.subckt"
    top = tmp_path / "top.sp"
    tile = "tile001_1_1_net_VDD_x_1_y_1"
    sink = "region_1_net_VDD_x_1_y_1_VDD_PORT"
    sub.write_text(
        "\n".join(
            [
                f".SUBCKT dieModel {tile} {sink}",
                f"* BUMP_VDD_1_1 {tile} VDD 30.0 30.0",
                f"R_mesh {tile} {sink} 1.0e-2",
                ".ENDS",
                "",
            ]
        )
    )
    top.write_text(
        "\n".join(
            [
                f'include "{sub.name}"',
                "VVDD VDD_in 0 DC 0.9",
                f"Xdie1 VDD_in {sink} dieModel",
                "",
            ]
        )
    )
    net = parse_spice(top)
    assert any(tile in (a, b) for a, b, _ in net.resistors)
    assert not any(
        (a == "VDD_in" and b == sink) or (b == "VDD_in" and a == sink)
        for a, b, _ in net.resistors
    )
    assert len(net.bumps) == 1
    assert net.bumps[0].bx == 1 and net.bumps[0].node == tile
    assert net.coords[tile].layer == LAYER_TILE
    assert net.coords[sink].layer == LAYER_VDD


def test_many_to_one_bump_two_pads_same_tile(tmp_path: Path):
    """Two BUMP sites on one tile → two unique pads, both near-shorted to it."""
    deck = tmp_path / "many.sp"
    tile = "tile001_2_2_net_VDD_x_2_y_2"
    lines = [
        "VVDD VDD_in 0 DC 0.9",
        f"* BUMP_VDD_1_1 {tile} VDD 30.0 30.0",
        f"* BUMP_VDD_2_1 {tile} VDD 90.0 30.0",
    ]
    for iy in (1, 2):
        for ix in (1, 2):
            rid = (iy - 1) * 2 + ix
            sn = f"region_{rid}_net_VDD_x_{ix}_y_{iy}_VDD_PORT"
            t = f"tile001_{ix}_{iy}_net_VDD_x_{ix}_y_{iy}"
            lines.append(f"R_t{rid} {t} {sn} 1e-2")
            lines.append(f"I_{rid} {sn} 0 1e-3")
    lines.append(f"R_h {tile} tile001_1_1_net_VDD_x_1_y_1 1.0")
    deck.write_text("\n".join(lines) + "\n")
    net = parse_spice(deck)
    bumps = collect_vdd_bumps(net)
    assert bumps[(1, 1)] == tile
    assert bumps[(2, 1)] == tile
    assert len(net.bumps) == 2

    pkg, vdd = ensure_tsmc_package_node(net)
    ports = build_tsmc_region_ports(net, vdd=vdd, package_node=pkg)
    assert ports.n_pads == ports.n_sinks == 4
    assert ports.n_shared_tiles == 1
    assert ports.pad_attach_tiles.count(tile) == 2
    n = inject_virtual_pads(net, ports, package_node=pkg)
    assert n == ports.n_occupied_bumps
    # Both pads near-short to the shared tile (not VDD_in).
    for pad in (vpad_name(1, 1), vpad_name(2, 1)):
        assert any(
            (a == pad and b == tile) or (b == pad and a == tile)
            for a, b, _ in net.resistors
        )
        assert not any(
            (a == pad and b == "VDD_in") or (b == pad and a == "VDD_in")
            for a, b, _ in net.resistors
        )

    short_map = build_short_map(net)
    keep = galvanic_component_of(net, short_map, ports.port_nodes)
    system = assemble_conductance(net, short_map, keep_roots=keep)
    port_idx, internal_idx = partition_ports(system, ports.port_nodes)
    assert len(port_idx) == ports.n_ports
    Gprime = kron_reduce(system.G, port_idx, internal_idx)
    assert Gprime.shape == (ports.n_ports, ports.n_ports)


def test_sparse_open_circuit_pad(tmp_path: Path):
    """Missing bump → isolated pad with ~zero G′ row/col."""
    deck = tmp_path / "sparse.sp"
    # 2×2 regions; bumps only at (1,1) and (2,1) — missing (1,2) and (2,2).
    lines = ["VVDD VDD_in 0 DC 0.9"]
    for iy in (1, 2):
        for ix in (1, 2):
            rid = (iy - 1) * 2 + ix
            sn = f"region_{rid}_net_VDD_x_{ix}_y_{iy}_VDD_PORT"
            t = f"tile001_{ix}_{iy}_net_VDD_x_{ix}_y_{iy}"
            lines.append(f"R_t{rid} {t} {sn} 1e-2")
            lines.append(f"I_{rid} {sn} 0 1e-3")
    lines.append(
        "R_h tile001_1_1_net_VDD_x_1_y_1 tile001_2_1_net_VDD_x_2_y_1 1.0"
    )
    lines.append(
        "R_h2 tile001_1_2_net_VDD_x_1_y_2 tile001_2_2_net_VDD_x_2_y_2 1.0"
    )
    lines.append(
        "R_v tile001_1_1_net_VDD_x_1_y_1 tile001_1_2_net_VDD_x_1_y_2 1.0"
    )
    lines.append("* BUMP_VDD_1_1 tile001_1_1_net_VDD_x_1_y_1 VDD 30.0 30.0")
    lines.append("* BUMP_VDD_2_1 tile001_2_1_net_VDD_x_2_y_1 VDD 90.0 30.0")
    deck.write_text("\n".join(lines) + "\n")
    net = parse_spice(deck)
    pkg, vdd = ensure_tsmc_package_node(net)
    ports = build_tsmc_region_ports(net, vdd=vdd, package_node=pkg)
    assert ports.n_occupied_bumps == 2
    assert ports.n_open_pads == 2
    inject_virtual_pads(net, ports, package_node=pkg)

    short_map = build_short_map(net)
    keep = galvanic_component_of(net, short_map, ports.port_nodes)
    # Open pads are seeds even with no edges.
    open_pad = vpad_name(1, 2)
    assert open_pad in keep
    system = assemble_conductance(net, short_map, keep_roots=keep)
    port_idx, internal_idx = partition_ports(system, ports.port_nodes)
    Gprime = kron_reduce(system.G, port_idx, internal_idx)
    open_i = ports.pad_nodes.index(open_pad)
    assert float(np.max(np.abs(Gprime[open_i, :]))) < 1e-18
    assert float(np.max(np.abs(Gprime[:, open_i]))) < 1e-18
    # Occupied pads couple through the tile mesh.
    occ_i = ports.pad_nodes.index(vpad_name(1, 1))
    assert float(np.max(np.abs(Gprime[occ_i, :]))) > 1e-6


def test_zero_based_bump_aligns_to_region_plus_one(tmp_path: Path):
    """BUMP_VDD_0_0 maps to region x_1_y_1; lattice is 0-based."""
    deck = tmp_path / "zbase.sp"
    tile = "tile001_1_1_net_VDD_x_1_y_1"
    lines = [
        "VVDD VDD_in 0 DC 0.9",
        f"* BUMP_VDD_0_0 {tile} VDD 1.8 3.6",
        f"* BUMP_VDD_1_0 tile001_2_1_net_VDD_x_2_y_1 VDD 61.8 3.6",
    ]
    for iy in (1, 2):
        for ix in (1, 2):
            rid = (iy - 1) * 2 + ix
            sn = f"region_{rid}_net_VDD_x_{ix}_y_{iy}_VDD_PORT"
            t = f"tile001_{ix}_{iy}_net_VDD_x_{ix}_y_{iy}"
            lines.append(f"R_t{rid} {t} {sn} 1e-2")
            lines.append(f"I_{rid} {sn} 0 1e-3")
    lines.append(
        "R_h tile001_1_1_net_VDD_x_1_y_1 tile001_2_1_net_VDD_x_2_y_1 1.0"
    )
    deck.write_text("\n".join(lines) + "\n")
    net = parse_spice(deck)
    bumps = collect_vdd_bumps(net)
    assert bump_index_base(bumps) == 0
    assert bumps[(0, 0)] == tile

    pkg, vdd = ensure_tsmc_package_node(net)
    ports = build_tsmc_region_ports(net, vdd=vdd, package_node=pkg)
    # Region bbox is 2×2; bumps extend to bx=1 → nx>=2, ny>=2 (not forced 43×33).
    assert ports.nx == 2 and ports.ny == 2
    assert ports.pad_nodes[0] == vpad_name(0, 0)
    assert ports.pad_attach_tiles[0] == tile
    # Sink at bump (0,0) is region x_1_y_1.
    c00 = next(c for c in ports.cells if c.ix == 0 and c.iy == 0)
    assert c00.node.endswith("x_1_y_1_VDD_PORT")
    assert abs(c00.x - 30.0) < 1e-9 and abs(c00.y - 30.0) < 1e-9
