"""IBM ibmpg2: my_flow dual ports vs independent spec oracle."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TC2 = REPO / "Benchmarks" / "IBM" / "TC2" / "ibmpg2.spice"
MY_OUT = REPO / "my_flow" / "outputs" / "ibmpg2_auto_k1_spec_audit" / "comp1"
SPEC_SRC = REPO / "spec_flow" / "src"
MY_SRC = REPO / "my_flow" / "src"


@pytest.fixture(scope="module")
def _paths():
    # my_flow bootstrap: my_src then spec_src
    for p in (str(MY_SRC), str(SPEC_SRC)):
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(SPEC_SRC))
    sys.path.insert(0, str(MY_SRC))


@pytest.mark.skipif(not TC2.is_file(), reason="IBM TC2 spice not present")
def test_dual_ports_c4_overlap_and_full_grid(_paths):
    from graph import build_short_map, vdd_connected_components
    from ports_dual import build_dual_ports, tri_stagger_ports_from_dual
    from spec_oracle import (
        build_spec_oracle,
        compare_current_lumping,
        compare_full_sink_grid,
        compare_pad_occupancy,
        compare_pad_voltages,
    )
    from spice_parser import parse_spice

    net = parse_spice(TC2)
    node_map = build_short_map(net)
    roots = vdd_connected_components(net, node_map)[0]
    pitch = 927.0
    oracle = build_spec_oracle(net, pitch, node_map=node_map, roots=roots)
    dual = build_dual_ports(
        net, pitch, pitch, node_map=node_map, roots=roots, full_grid_sinks=True
    )

    occ_cells = [
        {"ix": int(a), "iy": int(b), "node": "?", "current": 0.0}
        for a, b in dual.pad_cell_ids
    ]
    r1 = compare_pad_occupancy(
        oracle,
        pad_nodes=dual.pad_nodes,
        pad_attach=list(range(len(occ_cells))),
        cells=occ_cells,
        node_map=node_map,
    )
    assert r1.status == "pass", r1.detail

    cells = [
        {"ix": c.ix, "iy": c.iy, "node": c.node, "current": c.current}
        for c in dual.cells
    ]
    r2 = compare_current_lumping(oracle, cells=cells)
    assert r2.status == "pass", r2.detail

    r3 = compare_full_sink_grid(
        oracle,
        n_sinks=dual.n_sinks,
        nx=dual.nx_bot,
        ny=dual.ny_bot,
        flow_name="my_flow",
    )
    assert r3.status == "pass", r3.detail
    assert dual.n_sinks == oracle.nx * oracle.ny

    stagger = tri_stagger_ports_from_dual(dual)
    assert stagger.n_sinks == dual.n_sinks

    r5 = compare_pad_voltages(
        oracle, pad_voltages=dual.pad_voltages, pad_nodes=dual.pad_nodes
    )
    assert r5.status == "pass", r5.detail


@pytest.mark.skipif(not TC2.is_file(), reason="IBM TC2 spice not present")
def test_dual_ports_full_grid_at_fine_pitch(_paths):
    """Finer pitch still keeps one sink per cell (spec A3)."""
    from graph import build_short_map, vdd_connected_components
    from ports_dual import build_dual_ports, tri_stagger_ports_from_dual
    from spec_oracle import build_spec_oracle, compare_full_sink_grid
    from spice_parser import parse_spice

    net = parse_spice(TC2)
    node_map = build_short_map(net)
    roots = vdd_connected_components(net, node_map)[0]
    pitch = 231.5
    oracle = build_spec_oracle(net, pitch, node_map=node_map, roots=roots)
    dual = build_dual_ports(
        net, pitch, pitch, node_map=node_map, roots=roots, full_grid_sinks=True
    )
    assert oracle.n_nonzero_current_cells < oracle.nx * oracle.ny
    assert dual.n_sinks == dual.nx_bot * dual.ny_bot
    r3 = compare_full_sink_grid(
        oracle,
        n_sinks=dual.n_sinks,
        nx=dual.nx_bot,
        ny=dual.ny_bot,
        flow_name="my_flow",
    )
    assert r3.status == "pass", r3.detail
    assert tri_stagger_ports_from_dual(dual).n_sinks == dual.n_sinks


@pytest.mark.skipif(
    not TC2.is_file() or not MY_OUT.is_dir(),
    reason="IBM TC2 or my_flow OUT missing",
)
def test_check_spec_ibm_cli_my_flow(_paths):
    sys.path.insert(0, str(SPEC_SRC))
    from check_spec_ibm import score_comp

    summary = score_comp(TC2, MY_OUT, flow="my_flow")
    by = {c["clause"]: c for c in summary["clauses"]}
    assert by["A1_pad_occupancy"]["status"] == "pass", by["A1_pad_occupancy"]
    assert by["A3_full_sink_grid"]["status"] == "pass"
    assert by["A5_pad_voltage"]["status"] == "pass"
    assert by["C_model_class"]["status"] == "intentional_deviation"
    assert by["D_eigen_fit"]["status"] == "intentional_deviation"
