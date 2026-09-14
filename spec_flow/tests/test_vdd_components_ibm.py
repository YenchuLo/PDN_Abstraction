"""IBM multi-rail VDD connected-component counts (skip if spice missing)."""

from __future__ import annotations

from pathlib import Path

import pytest

from graph import _component_layer_span, build_short_map, vdd_connected_components
from spice_parser import parse_spice

REPO = Path(__file__).resolve().parents[2]
TC3 = REPO / "Benchmarks" / "IBM" / "TC3" / "ibmpg3.spice"
TC5 = REPO / "Benchmarks" / "IBM" / "TC5" / "ibmpg5.spice"


@pytest.mark.skipif(not TC3.is_file(), reason="IBM TC3 spice not present")
def test_ibmpg3_two_vdd_components_topmost_first():
    net = parse_spice(TC3)
    node_map = build_short_map(net)
    comps = vdd_connected_components(net, node_map)
    assert len(comps) == 2
    tops = [_component_layer_span(net, node_map, c)[0] for c in comps]
    assert tops == [14, 13]


@pytest.mark.skipif(not TC5.is_file(), reason="IBM TC5 spice not present")
def test_ibmpg5_four_vdd_components_topmost_first():
    net = parse_spice(TC5)
    node_map = build_short_map(net)
    comps = vdd_connected_components(net, node_map)
    assert len(comps) == 4
    tops = [_component_layer_span(net, node_map, c)[0] for c in comps]
    assert tops == [14, 13, 12, 11]


@pytest.mark.skipif(
    not (REPO / "Benchmarks" / "IBM" / "TC1" / "ibmpg1.spice").is_file(),
    reason="IBM TC1 spice not present",
)
def test_ibmpg1_same_layer_islands_merged():
    tc1 = REPO / "Benchmarks" / "IBM" / "TC1" / "ibmpg1.spice"
    net = parse_spice(tc1)
    node_map = build_short_map(net)
    comps = vdd_connected_components(net, node_map)
    assert len(comps) == 1
    top, bot = _component_layer_span(net, node_map, comps[0])
    assert (bot, top) == (1, 3)
