"""Unit tests for VDD/VSS PDN geometry extraction and Plotly figure build."""

from __future__ import annotations

from pathlib import Path

import pytest

from graph import (
    build_short_map,
    vdd_connected_components,
    vdd_roots,
    vss_layers,
    vss_roots,
)
from pdn_geometry import extract_pdn_geometry
from pdn_plot import build_figure
from spice_parser import SpiceNetlist, parse_node_coord


def _node(name: str):
    c = parse_node_coord(name)
    assert c is not None
    return name, c


def _tiny_dual_net() -> SpiceNetlist:
    """
    Tiny dual-rail PDN:
      VDD: M1 horizontal + M3 horizontal, via M1↔M3, pad V and sink I
      VSS: M0 horizontal + M2 horizontal, via M0↔M2, one node tied to 0
    """
    net = SpiceNetlist()
    names = [
        "n1_0_0",
        "n1_10_0",
        "n3_0_0",
        "n3_10_0",
        "n0_0_5",
        "n0_10_5",
        "n2_0_5",
        "n2_10_5",
    ]
    for n in names:
        name, c = _node(n)
        net.node_names.add(name)
        net.coords[name] = c

    # VDD metals
    net.resistors.append(("n1_0_0", "n1_10_0", 1.0))
    net.resistors.append(("n3_0_0", "n3_10_0", 1.0))
    # VSS metals
    net.resistors.append(("n0_0_5", "n0_10_5", 1.0))
    net.resistors.append(("n2_0_5", "n2_10_5", 1.0))
    # vias
    net.voltages.append(("n1_0_0", "n3_0_0", 0.0))
    net.voltages.append(("n0_0_5", "n2_0_5", 0.0))
    # VDD pad + sink injection
    net.voltages.append(("n3_10_0", "0", 0.9))
    net.currents.append(("n1_10_0", "0", -0.1))
    # one VSS node shorted to ideal ground
    net.voltages.append(("n0_10_5", "0", 0.0))
    return net


def test_vss_roots_and_layers_on_tiny():
    net = _tiny_dual_net()
    node_map = build_short_map(net)
    vdd = vdd_roots(net, node_map)
    assert "n1_0_0" in vdd or node_map.get("n1_0_0", "n1_0_0") in vdd

    vss = vss_roots(net, node_map, vdd)
    # n2 stack may remain as non-ground roots; n0_10_5 maps to 0
    layers = vss_layers(net, node_map, vdd)
    assert layers == [0, 2]


def test_extract_pdn_geometry_counts():
    net = _tiny_dual_net()
    geom = extract_pdn_geometry(net)

    assert geom.n_vdd_components == 1
    assert geom.vdd_components[0].name == "vdd1"
    assert geom.vdd.layers == [1, 3]
    assert geom.vss.layers == [0, 2]
    assert geom.vdd.n_metal == 2
    assert geom.vss.n_metal == 2
    assert len(geom.vdd.metal[1]) == 1
    assert len(geom.vdd.metal[3]) == 1
    assert len(geom.vss.metal[0]) == 1
    assert len(geom.vss.metal[2]) == 1
    assert len(geom.vdd.vias) == 1
    assert len(geom.vss.vias) == 1


def _tiny_two_vdd() -> SpiceNetlist:
    """Two disconnected VDD rails (L3+L1 and L4+L2) plus one VSS."""
    net = SpiceNetlist()
    names = [
        "n1_0_0",
        "n1_10_0",
        "n3_0_0",
        "n3_10_0",
        "n2_0_20",
        "n2_10_20",
        "n4_0_20",
        "n4_10_20",
        "n0_0_5",
        "n0_10_5",
    ]
    for n in names:
        name, c = _node(n)
        net.node_names.add(name)
        net.coords[name] = c
    # Rail A: L1–L3
    net.resistors.append(("n1_0_0", "n1_10_0", 1.0))
    net.resistors.append(("n3_0_0", "n3_10_0", 1.0))
    net.voltages.append(("n1_0_0", "n3_0_0", 0.0))
    net.voltages.append(("n3_10_0", "0", 0.9))
    net.currents.append(("n1_10_0", "0", -0.1))
    # Rail B: L2–L4 (topmost = 4 → vdd1)
    net.resistors.append(("n2_0_20", "n2_10_20", 1.0))
    net.resistors.append(("n4_0_20", "n4_10_20", 1.0))
    net.voltages.append(("n2_0_20", "n4_0_20", 0.0))
    net.voltages.append(("n4_10_20", "0", 0.9))
    net.currents.append(("n2_10_20", "0", -0.1))
    # VSS
    net.resistors.append(("n0_0_5", "n0_10_5", 1.0))
    net.voltages.append(("n0_10_5", "0", 0.0))
    return net


def test_vdd_connected_components_order_and_geometry():
    from graph import _component_layer_span

    net = _tiny_two_vdd()
    node_map = build_short_map(net)
    comps = vdd_connected_components(net, node_map)
    assert len(comps) == 2
    # Topmost-first: rail with L4 before rail with L3
    tops = [_component_layer_span(net, node_map, c)[0] for c in comps]
    assert tops == [4, 3]

    geom = extract_pdn_geometry(net)
    assert geom.n_vdd_components == 2
    assert [g.name for g in geom.vdd_components] == ["vdd1", "vdd2"]
    assert geom.vdd_components[0].layers == [2, 4]
    assert geom.vdd_components[1].layers == [1, 3]
    assert geom.vdd_components[0].n_metal == 2
    assert geom.vdd_components[1].n_metal == 2


def _tiny_same_layer_islands() -> SpiceNetlist:
    """Two galvanic islands on the same layer set {1,3} — must merge to one comp."""
    net = SpiceNetlist()
    names = [
        "n1_0_0",
        "n1_10_0",
        "n3_0_0",
        "n3_10_0",
        "n1_100_0",
        "n1_110_0",
        "n3_100_0",
        "n3_110_0",
    ]
    for n in names:
        name, c = _node(n)
        net.node_names.add(name)
        net.coords[name] = c
    # Island A
    net.resistors.append(("n1_0_0", "n1_10_0", 1.0))
    net.resistors.append(("n3_0_0", "n3_10_0", 1.0))
    net.voltages.append(("n1_0_0", "n3_0_0", 0.0))
    net.voltages.append(("n3_10_0", "0", 0.9))
    net.currents.append(("n1_10_0", "0", -0.1))
    # Island B (same layers, no path to A)
    net.resistors.append(("n1_100_0", "n1_110_0", 1.0))
    net.resistors.append(("n3_100_0", "n3_110_0", 1.0))
    net.voltages.append(("n1_100_0", "n3_100_0", 0.0))
    net.voltages.append(("n3_110_0", "0", 0.9))
    net.currents.append(("n1_110_0", "0", -0.1))
    return net


def test_same_layer_islands_merged():
    net = _tiny_same_layer_islands()
    node_map = build_short_map(net)
    comps = vdd_connected_components(net, node_map)
    assert len(comps) == 1
    geom = extract_pdn_geometry(net)
    assert geom.n_vdd_components == 1
    assert geom.vdd.layers == [1, 3]
    assert geom.vdd.n_metal == 4


def test_cross_layer_resistor_counted_as_via():
    """IBM TC2–TC4 style: vias are cross-layer R, not V=0."""
    net = SpiceNetlist()
    for n in ("n1_0_0", "n1_10_0", "n3_0_0", "n3_10_0"):
        name, c = _node(n)
        net.node_names.add(name)
        net.coords[name] = c
    net.resistors.append(("n1_0_0", "n1_10_0", 1.0))
    net.resistors.append(("n3_0_0", "n3_10_0", 1.0))
    net.resistors.append(("n1_0_0", "n3_0_0", 0.03))  # via
    net.voltages.append(("n3_10_0", "0", 0.9))
    net.currents.append(("n1_10_0", "0", -0.1))
    geom = extract_pdn_geometry(net)
    assert geom.vdd.n_metal == 2
    assert len(geom.vdd.vias) == 1


def test_build_figure_smoke():
    pytest.importorskip("plotly")
    net = _tiny_dual_net()
    geom = extract_pdn_geometry(net)
    fig = build_figure(geom, default_net="vdd", default_layer=None)
    assert len(fig.data) > 0
    titles = [a.text for a in fig.layout.annotations if a.text and "PDN layout" in a.text]
    assert titles and "VDD1" in titles[0]
    # Non-negative data should not pad the axis into negative ticks.
    assert fig.layout.xaxis.range[0] >= 0.0
    assert fig.layout.yaxis.range[0] >= 0.0
    # Active via trace should be legend-toggleable.
    via_tr = next(tr for tr in fig.data if tr.name == "Vias" and tr.visible)
    assert via_tr.showlegend is True


def test_tc1_net_npz_if_present():
    out = Path(__file__).resolve().parents[1] / "outputs" / "ibmpg1_n500"
    if not (out / "net.npz").is_file():
        pytest.skip("ibmpg1_n500/net.npz not present")
    from io_artifacts import load_net

    net = load_net(out)
    geom = extract_pdn_geometry(net)
    assert 1 in geom.vdd.layers and 3 in geom.vdd.layers
    assert geom.vdd.n_metal > 0
    # GND stack present on TC1
    assert geom.vss.n_metal > 0
    assert set(geom.vss.layers) & {0, 2}
