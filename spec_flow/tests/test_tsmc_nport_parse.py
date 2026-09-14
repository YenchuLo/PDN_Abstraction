"""Parse / stage-01 smoke for TSMC hierarchical n-port decks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from graph import build_short_map, vdd_connected_components
from spice_parser import parse_spice

REPO = Path(__file__).resolve().parents[2]
SPEC = Path(__file__).resolve().parents[1] / "src"
TSMC_SP = (
    REPO / "Benchmarks" / "TSMC" / "TC1" / "myALL_nportModel_without_package.sp"
)


def _write_many_plus_comment_deck(dir_path: Path) -> Path:
    """
    Mimic real TC1 style: long ``+`` pin lists, ``*`` section / BUMP comments
    interleaved with continuations, split R lines, ``.INCLUDE``, ``.PRINT``.
    """
    sub = dir_path / "many.sp.subckt"
    top = dir_path / "many_top.sp"
    # 2×2 region lattice (≥4 cells → virtual Pixel-R path) + package pin
    pins = [
        "tile001_1_1_net_VDD_x_1_y_1",
        "tile001_2_1_net_VDD_x_2_y_1",
        "region_1_net_VDD_x_1_y_1_VDD_PORT",
        "region_2_net_VDD_x_2_y_1_VDD_PORT",
        "region_3_net_VDD_x_1_y_2_VDD_PORT",
        "region_4_net_VDD_x_2_y_2_VDD_PORT",
    ]
    sub.write_text(
        "\n".join(
            [
                "* header comment before subckt",
                ".SUBCKT manyModel",
                f"+ {pins[0]} {pins[1]}",
                "* [POWER NETS] — comment between pin continuations",
                f"+ {pins[2]} {pins[3]}",
                f"+ {pins[4]} {pins[5]}",
                "",
                "* BUMP_VDD_1_1 tile001_1_1_net_VDD_x_1_y_1 VDD 30.0 30.0",
                "* BUMP_VDD_2_1 tile001_2_1_net_VDD_x_2_y_1 VDD 90.0 30.0",
                "* [GROUND NETS]",
                "R_1 tile001_1_1_net_VDD_x_1_y_1 tile001_2_1_net_VDD_x_2_y_1 1.0",
                "R_2 tile001_1_1_net_VDD_x_1_y_1",
                "+ region_1_net_VDD_x_1_y_1_VDD_PORT 1.0e-2",
                "R_3 tile001_2_1_net_VDD_x_2_y_1",
                "* note between R continuations",
                "+ region_2_net_VDD_x_2_y_1_VDD_PORT 1.0e-2",
                "R_4 tile001_1_1_net_VDD_x_1_y_1 region_3_net_VDD_x_1_y_2_VDD_PORT 1.0e-2",
                "R_5 tile001_2_1_net_VDD_x_2_y_1 region_4_net_VDD_x_2_y_2_VDD_PORT 1.0e-2",
                "I_1 region_1_net_VDD_x_1_y_1_VDD_PORT 0 1.05m",
                "I_2 region_2_net_VDD_x_2_y_1_VDD_PORT 0 1.05m",
                "I_3 region_3_net_VDD_x_1_y_2_VDD_PORT 0 1.05m",
                "I_4 region_4_net_VDD_x_2_y_2_VDD_PORT 0 1.05m",
                ".ENDS manyModel",
                "",
            ]
        )
    )
    top.write_text(
        "\n".join(
            [
                "* top-level banner",
                '.INCLUDE "./many.sp.subckt"',
                "VVDD VDD_in 0 DC 0.9",
                "Xdie1 VDD_in VDD_in",
                "* comment in the middle of X pin list",
                f"+ {pins[2]} {pins[3]}",
                f"+ {pins[4]} {pins[5]} manyModel",
                ".PRINT DC V(Xdie1.region_1_net_VDD_x_1_y_1_VDD_PORT)",
                ".DC",
                ".END",
                "",
            ]
        )
    )
    return top


def test_subckt_bumps_bare_r_wrap_and_isrc_include(tmp_path):
    """
    Real ``.sp.subckt`` shape: long ``+`` pins, ``* [POWER/GROUND]`` / ``* BUMP_``,
    R lines wrapped *without* ``+``, ``.INCLUDE`` of ``.isrc``, ``R_VSS_CONN_*``,
    ``.ENDS name``.
    """
    isrc = tmp_path / "myALL_nportModel.sp.isrc"
    sub = tmp_path / "myALL_nportModel.sp.subckt"
    top = tmp_path / "top.sp"
    isrc.write_text(
        "I_1_x_1_y_1 region_1_net_VDD_x_1_y_1_VDD_PORT "
        "region_5_net_VSS_x_1_y_1_VSS_PORT 1.0501918409m\n"
        "I_2_x_2_y_1 region_2_net_VDD_x_2_y_1_VDD_PORT "
        "region_6_net_VSS_x_2_y_1_VSS_PORT 1.0501918409m\n"
    )
    sub.write_text(
        "\n".join(
            [
                ".SUBCKT myALL_nportModel tile001_10_10_net_VDD_x_10_y_10",
                "+ tile001_10_11_net_VDD_x_10_y_11 tile001_10_12_net_VDD_x_10_y_12",
                "+ region_1_net_VDD_x_1_y_1_VDD_PORT region_2_net_VDD_x_2_y_1_VDD_PORT",
                "+ region_3_net_VDD_x_1_y_2_VDD_PORT region_4_net_VDD_x_2_y_2_VDD_PORT",
                "+region_5_net_VSS_x_1_y_1_VSS_PORT region_6_net_VSS_x_2_y_1_VSS_PORT",
                "+ region_7_net_VSS_x_1_y_2_VSS_PORT region_8_net_VSS_x_2_y_2_VSS_PORT",
                "* [POWER NETS]",
                "* BUMP_VDD_10_10 tile001_10_10_net_VDD_x_10_y_10 VDD 30.0 30.0",
                "* BUMP_VDD_10_11 tile001_10_11_net_VDD_x_10_y_11 VDD 30.0 90.0",
                "* [GROUND NETS]",
                "R_00001 tile001_10_10_net_VDD_x_10_y_10 "
                "tile001_10_11_net_VDD_x_10_y_11 2.110526192810e-01",
                # Bare wrap (no leading +) — must still parse as one R
                "R_128593 tile001_10_10_net_VDD_x_10_y_10",
                "region_1_net_VDD_x_1_y_1_VDD_PORT 3.307354305293e+14",
                "R_3746409 region_2_net_VDD_x_2_y_1_VDD_PORT",
                "region_1_net_VDD_x_1_y_1_VDD_PORT 2.451183376023e+02",
                '.INCLUDE "./myALL_nportModel.sp.isrc"',
                "R_VSS_CONN_0001 region_5_net_VSS_x_1_y_1_VSS_PORT "
                "tile001_10_12_net_VDD_x_10_y_12 1.000000000000e-15",
                ".ENDS myALL_nportModel",
                "",
            ]
        )
    )
    top.write_text(
        "\n".join(
            [
                "VVDD VDD_in 0 DC 0.9",
                "VVSS VSS_in 0 DC 0",
                '.INCLUDE "./myALL_nportModel.sp.subckt"',
                "Xdie1 VDD_in VDD_in VDD_in",
                "+ region_1_net_VDD_x_1_y_1_VDD_PORT region_2_net_VDD_x_2_y_1_VDD_PORT",
                "+ region_3_net_VDD_x_1_y_2_VDD_PORT region_4_net_VDD_x_2_y_2_VDD_PORT",
                "+ region_5_net_VSS_x_1_y_1_VSS_PORT region_6_net_VSS_x_2_y_1_VSS_PORT",
                "+ region_7_net_VSS_x_1_y_2_VSS_PORT region_8_net_VSS_x_2_y_2_VSS_PORT",
                "+ myALL_nportModel",
                ".END",
                "",
            ]
        )
    )
    net = parse_spice(top)
    vals = sorted(r[2] for r in net.resistors)
    assert len(net.currents) == 2
    assert abs(net.currents[0][2] - 1.0501918409e-3) < 1e-15
    # Bare-wrapped R's recovered (huge + mid values)
    assert any(abs(v - 3.307354305293e14) < 1e4 for v in vals)
    assert any(abs(v - 2.451183376023e2) < 1e-6 for v in vals)
    assert any(abs(v - 2.110526192810e-1) < 1e-12 for v in vals)
    assert any(abs(v - 1e-15) < 1e-20 for v in vals)
    # BUMP override applied to a tile pin (shorted to VDD_in → may be on VDD_in)
    assert len(net.voltages) == 2


def test_isrc_vdd_to_vss_one_sink_per_cell(tmp_path):
    """
    Real ``.isrc`` form: one I per region cell,

    ``I_*_x_*_y_*  <VDD_PORT>  <VSS_PORT>  1.0501918409m``

    Sink port for Pixel-R is the single VDD_PORT node; VSS is return only.
    """
    from ports import build_tsmc_region_ports, ensure_tsmc_package_node

    deck = tmp_path / "isrc_only.sp"
    lines = [
        "* isrc-style currents (included into a flat deck)",
        "VVDD VDD_in 0 DC 0.9",
    ]
    for iy in (1, 2):
        for ix in (1, 2):
            rid = (iy - 1) * 2 + ix
            vss_rid = 4 + rid
            vdd = f"region_{rid}_net_VDD_x_{ix}_y_{iy}_VDD_PORT"
            vss = f"region_{vss_rid}_net_VSS_x_{ix}_y_{iy}_VSS_PORT"
            tile = f"tile001_{ix}_{iy}_net_VDD_x_{ix}_y_{iy}"
            lines.append(f"R_{rid} VDD_in {tile} 1e-3")
            lines.append(f"R_s{rid} {tile} {vdd} 1e-2")
            lines.append(
                f"I_{rid}_x_{ix}_y_{iy} {vdd} {vss} 1.0501918409m"
            )
    deck.write_text("\n".join(lines) + "\n")
    net = parse_spice(deck)
    assert len(net.currents) == 4
    assert abs(net.currents[0][2] - 1.0501918409e-3) < 1e-15
    # Two-terminal: VDD_PORT → VSS_PORT (not to 0)
    assert net.currents[0][0].endswith("_VDD_PORT")
    assert net.currents[0][1].endswith("_VSS_PORT")

    pkg, vdd = ensure_tsmc_package_node(net, vdd=0.9)
    ports = build_tsmc_region_ports(net, vdd=vdd, package_node=pkg)
    assert ports.n_sinks == 4
    for c in ports.cells:
        assert c.members == [c.node]
        assert c.node.endswith("_VDD_PORT")
        # Nodal injection into circuit at VDD_PORT is -I
        assert abs(c.current + 1.0501918409e-3) < 1e-15


def test_real_top_deck_vsrc_include_xdie_print(tmp_path):
    """
    Real without-package top deck shape:

    VVDD / VVSS with ``DC``, ``.INCLUDE "…"``, long ``Xdie1`` pin list
    (many ``VDD_in`` / ``VSS_in`` / region ports), ``.PRINT`` / ``.DC`` / ``.END``.
    """
    sub = tmp_path / "myALL_nportModel.sp.subckt"
    top = tmp_path / "myALL_nportModel_without_package.sp"
    # Minimal filled 2×2 VDD region lattice + matching VSS ports + 2 package pins
    pins = [
        "bump_vdd",
        "bump_vss",
        "region_1_net_VDD_x_1_y_1_VDD_PORT",
        "region_2_net_VDD_x_2_y_1_VDD_PORT",
        "region_3_net_VDD_x_1_y_2_VDD_PORT",
        "region_4_net_VDD_x_2_y_2_VDD_PORT",
        "region_5_net_VSS_x_1_y_1_VSS_PORT",
        "region_6_net_VSS_x_2_y_1_VSS_PORT",
        "region_7_net_VSS_x_1_y_2_VSS_PORT",
        "region_8_net_VSS_x_2_y_2_VSS_PORT",
    ]
    sub.write_text(
        "\n".join(
            [
                f".SUBCKT myALL_nportModel {' '.join(pins)}",
                "R_1 bump_vdd region_1_net_VDD_x_1_y_1_VDD_PORT 1",
                "R_2 bump_vdd region_2_net_VDD_x_2_y_1_VDD_PORT 1",
                "R_3 bump_vdd region_3_net_VDD_x_1_y_2_VDD_PORT 1",
                "R_4 bump_vdd region_4_net_VDD_x_2_y_2_VDD_PORT 1",
                "R_5 bump_vss region_5_net_VSS_x_1_y_1_VSS_PORT 1e-15",
                "I_1 region_1_net_VDD_x_1_y_1_VDD_PORT region_5_net_VSS_x_1_y_1_VSS_PORT 1.05m",
                "I_2 region_2_net_VDD_x_2_y_1_VDD_PORT region_6_net_VSS_x_2_y_1_VSS_PORT 1.05m",
                "I_3 region_3_net_VDD_x_1_y_2_VDD_PORT region_7_net_VSS_x_1_y_2_VSS_PORT 1.05m",
                "I_4 region_4_net_VDD_x_2_y_2_VDD_PORT region_8_net_VSS_x_2_y_2_VSS_PORT 1.05m",
                ".ENDS",
                "",
            ]
        )
    )
    top.write_text(
        "\n".join(
            [
                "VVDD             VDD_in    0         DC  0.9",
                "VVSS             VSS_in    0         DC  0",
                "",
                '.INCLUDE "./myALL_nportModel.sp.subckt"',
                "",
                "Xdie1 VDD_in VSS_in",
                "+ region_1_net_VDD_x_1_y_1_VDD_PORT region_2_net_VDD_x_2_y_1_VDD_PORT",
                "+ region_3_net_VDD_x_1_y_2_VDD_PORT region_4_net_VDD_x_2_y_2_VDD_PORT",
                "+ region_5_net_VSS_x_1_y_1_VSS_PORT region_6_net_VSS_x_2_y_1_VSS_PORT",
                "+ region_7_net_VSS_x_1_y_2_VSS_PORT region_8_net_VSS_x_2_y_2_VSS_PORT",
                "+ myALL_nportModel",
                "",
                ".PRINT DC V(Xdie1.region_1_net_VDD_x_1_y_1_VDD_PORT)",
                ".PRINT DC V(Xdie1.region_2838_net_VSS_x_43_y_33_VSS_PORT)",
                ".DC",
                ".PRINT DC I(VVDD) I(VVSS)",
                ".END",
                "",
            ]
        )
    )
    net = parse_spice(top)
    assert len(net.voltages) == 2
    vmap = {n1: v for n1, _n2, v in net.voltages}
    assert abs(vmap["VDD_in"] - 0.9) < 1e-15
    assert abs(vmap["VSS_in"] - 0.0) < 1e-15
    assert len(net.currents) == 4
    assert abs(net.currents[0][2] - 1.05e-3) < 1e-15
    # Package pin shorted to VDD_in
    assert any(a == "VDD_in" or b == "VDD_in" for a, b, _ in net.resistors)
    assert any(n.endswith("_VDD_PORT") for n in net.node_names)
    assert any(n.endswith("_VSS_PORT") for n in net.node_names)


def test_plus_continuations_and_star_comments(tmp_path):
    """Long ``+`` lists and ``*`` comments must not break flatten/parse."""
    top = _write_many_plus_comment_deck(tmp_path)
    net = parse_spice(top)
    # 5 R's + package is not present inside subckt; X maps first two pins → VDD_in
    assert len(net.resistors) == 5
    assert len(net.currents) == 4
    assert len(net.voltages) == 1
    assert abs(net.voltages[0][2] - 0.9) < 1e-15
    # Engineering suffix on I
    assert abs(net.currents[0][2] - 1.05e-3) < 1e-15
    # BUMP override kept (tile at micron coords from comment)
    assert "tile001_1_1_net_VDD_x_1_y_1" in net.coords or any(
        "tile001_1_1" in n for n in net.coords
    )
    # Region ports present after pin mapping
    assert any(n.endswith("_VDD_PORT") for n in net.node_names)
    # .PRINT / .DC / .END dropped — no crash, no bogus elements
    assert all(not n.upper().startswith(".PRINT") for n in net.node_names)


@pytest.mark.skipif(not TSMC_SP.is_file(), reason="TSMC TC1 smoke deck missing")
def test_tsmc_nport_flatten_parse():
    from tsmc_region_grid import is_virtual_pixel_r_lattice

    net = parse_spice(TSMC_SP)
    assert len(net.resistors) == 13
    assert len(net.currents) == 4
    assert len(net.voltages) == 2
    assert net.bbox == (30.0, 30.0, 90.0, 90.0)
    assert is_virtual_pixel_r_lattice(net)
    # Top-level VVDD / VVSS (DC)
    vmap = {n1: v for n1, _n2, v in net.voltages}
    assert abs(vmap["VDD_in"] - 0.9) < 1e-15
    assert abs(vmap["VSS_in"] - 0.0) < 1e-15
    assert any(n.endswith("_VDD_PORT") for n in net.node_names)
    assert any(n.endswith("_VSS_PORT") for n in net.node_names)
    # Region lattice microns
    assert net.coords["region_1_net_VDD_x_1_y_1_VDD_PORT"].x == 30.0
    assert net.coords["region_4_net_VDD_x_2_y_2_VDD_PORT"].y == 90.0
    # isrc engineering suffix
    assert abs(net.currents[0][2] - 1.0501918409e-3) < 1e-15

    node_map = build_short_map(net)
    comps = vdd_connected_components(net, node_map)
    assert len(comps) >= 1


@pytest.mark.skipif(not TSMC_SP.is_file(), reason="TSMC TC1 smoke deck missing")
def test_tsmc_nport_stage01(tmp_path):
    out = tmp_path / "tsmc_tc1"
    proc = subprocess.run(
        [sys.executable, str(SPEC / "stage01_parse.py"), str(TSMC_SP), str(out)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    meta = json.loads((out / "net_meta.json").read_text())
    assert meta["n_resistors"] == 13
    assert meta["n_currents"] == 4
    assert meta["n_voltages"] == 2
