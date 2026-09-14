"""IBM ibmpg2: local_flow localized G_S pad–sink coupling vs oracle ports."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TC2 = REPO / "Benchmarks" / "IBM" / "TC2" / "ibmpg2.spice"
LOCAL_OUT = REPO / "local_flow" / "outputs" / "ibmpg2_nauto_spec_audit" / "comp1"
LOCAL_SRC = REPO / "local_flow" / "src"
SPEC_SRC = REPO / "spec_flow" / "src"


@pytest.fixture(scope="module")
def _paths():
    for p in (str(LOCAL_SRC), str(SPEC_SRC)):
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(SPEC_SRC))
    sys.path.insert(0, str(LOCAL_SRC))


@pytest.mark.skipif(not TC2.is_file(), reason="IBM TC2 spice not present")
def test_local_flow_ports_match_oracle_via_build_ports(_paths):
    """local_flow reuses spec_flow build_ports — same oracle expectations."""
    from graph import build_short_map, vdd_connected_components
    from ports import build_ports
    from spec_oracle import (
        build_spec_oracle,
        compare_current_lumping,
        compare_full_sink_grid,
        compare_pad_occupancy,
    )
    from spice_parser import parse_spice

    net = parse_spice(TC2)
    node_map = build_short_map(net)
    roots = vdd_connected_components(net, node_map)[0]
    pitch = 927.0
    oracle = build_spec_oracle(net, pitch, node_map=node_map, roots=roots)
    ports = build_ports(net, pitch, node_map=node_map, roots=roots)
    cells = [
        {"ix": c.ix, "iy": c.iy, "node": c.node, "current": c.current}
        for c in ports.cells
    ]
    assert (
        compare_pad_occupancy(
            oracle,
            pad_nodes=ports.pad_nodes,
            pad_attach=ports.resolved_pad_attach(),
            cells=cells,
            node_map=node_map,
        ).status
        == "pass"
    )
    assert compare_current_lumping(oracle, cells=cells).status == "pass"
    assert (
        compare_full_sink_grid(
            oracle, n_sinks=ports.n_sinks, nx=ports.nx, ny=ports.ny
        ).status
        == "pass"
    )


@pytest.mark.skipif(not TC2.is_file(), reason="IBM TC2 spice not present")
def test_localized_gs_coupling_follows_pad_attach(_paths):
    from graph import build_short_map, vdd_connected_components
    from localized_pixel import build_Gs, build_localized_model
    from ports import build_ports
    from spec_oracle import compare_gs_pad_coupling
    from spice_parser import parse_spice

    net = parse_spice(TC2)
    node_map = build_short_map(net)
    roots = vdd_connected_components(net, node_map)[0]
    ports = build_ports(net, 927.0, node_map=node_map, roots=roots)
    model = build_localized_model(ports, block_size=0)  # global = uniform
    n = ports.n_sinks
    Rx = [1.0] * n
    Ry = [2.0] * n
    Rz = [0.5] * n
    Gs = build_Gs(model, Rx, Ry, Rz)
    r = compare_gs_pad_coupling(
        n_pads=ports.n_pads,
        n_sinks=ports.n_sinks,
        pad_attach=ports.resolved_pad_attach(),
        Gs=Gs,
    )
    assert r.status == "pass", r.detail
    assert Gs.shape == (ports.n_ports, ports.n_ports)
    assert n == ports.nx * ports.ny


@pytest.mark.skipif(
    not TC2.is_file() or not LOCAL_OUT.is_dir(),
    reason="IBM TC2 or local_flow non-uI OUT missing",
)
def test_check_spec_ibm_cli_local_flow(_paths):
    sys.path.insert(0, str(SPEC_SRC))
    from check_spec_ibm import score_comp

    summary = score_comp(TC2, LOCAL_OUT, flow="local_flow")
    by = {c["clause"]: c for c in summary["clauses"]}
    assert by["A1_pad_occupancy"]["status"] == "pass"
    assert by["A2_current_lumping"]["status"] == "pass"
    assert by["A3_full_sink_grid"]["status"] == "pass"
    assert by["A5_pad_voltage"]["status"] == "pass"
    assert by["C_model_class"]["status"] == "intentional_deviation"
    assert by["D_eigen_fit"]["status"] == "intentional_deviation"
