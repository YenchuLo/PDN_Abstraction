"""IBM ibmpg2: independent spec oracle vs build_ports / Pixel-R G_S."""

from __future__ import annotations

from pathlib import Path

import pytest

from graph import build_short_map, vdd_connected_components
from pixel_r import build_Gs, build_pixel_model
from ports import build_ports
from spec_oracle import (
    build_spec_oracle,
    compare_current_lumping,
    compare_full_sink_grid,
    compare_gs_pad_coupling,
    compare_pad_occupancy,
    compare_pad_voltages,
)
from spice_parser import parse_spice

REPO = Path(__file__).resolve().parents[2]
TC2 = REPO / "Benchmarks" / "IBM" / "TC2" / "ibmpg2.spice"
SPEC_OUT = REPO / "spec_flow" / "outputs" / "ibmpg2_nauto_spec_audit" / "comp1"


@pytest.fixture(scope="module")
def ibmpg2():
    if not TC2.is_file():
        pytest.skip("IBM TC2 spice not present")
    net = parse_spice(TC2)
    node_map = build_short_map(net)
    roots = vdd_connected_components(net, node_map)[0]
    return net, node_map, roots


@pytest.mark.skipif(not TC2.is_file(), reason="IBM TC2 spice not present")
def test_oracle_c4s_are_nonzero_v_to_gnd(ibmpg2):
    net, node_map, roots = ibmpg2
    oracle = build_spec_oracle(net, 927.0, node_map=node_map, roots=roots)
    assert oracle.n_pads > 0
    assert len(oracle.c4s) >= oracle.n_pads
    # No V=0 via mistaken as pad
    for c4 in oracle.c4s:
        assert abs(c4.voltage) > 1e-12
    # Current conservation
    assert abs(sum(c.current for c in oracle.cells) - oracle.i_total) < 1e-9
    assert oracle.n_sinks == oracle.nx * oracle.ny


@pytest.mark.skipif(not TC2.is_file(), reason="IBM TC2 spice not present")
def test_build_ports_matches_oracle_occupancy_and_current(ibmpg2):
    net, node_map, roots = ibmpg2
    pitch = 927.0
    oracle = build_spec_oracle(net, pitch, node_map=node_map, roots=roots)
    ports = build_ports(net, pitch, node_map=node_map, roots=roots)

    cells = [
        {
            "ix": c.ix,
            "iy": c.iy,
            "node": c.node,
            "current": c.current,
        }
        for c in ports.cells
    ]
    r1 = compare_pad_occupancy(
        oracle,
        pad_nodes=ports.pad_nodes,
        pad_attach=ports.resolved_pad_attach(),
        cells=cells,
        node_map=node_map,
    )
    assert r1.status == "pass", r1.detail

    r2 = compare_current_lumping(oracle, cells=cells)
    assert r2.status == "pass", r2.detail

    r3 = compare_full_sink_grid(
        oracle, n_sinks=ports.n_sinks, nx=ports.nx, ny=ports.ny
    )
    assert r3.status == "pass", r3.detail

    # Known fix: pad voltages follow IBM C4 values (not DEFAULT_VDD).
    r5 = compare_pad_voltages(
        oracle, pad_voltages=ports.pad_voltages, pad_nodes=ports.pad_nodes
    )
    assert r5.status == "pass", r5.detail
    assert abs(ports.vdd - ports.pad_voltages[0]) < 1e-12


@pytest.mark.skipif(not TC2.is_file(), reason="IBM TC2 spice not present")
def test_pixel_r_gs_coupling_only_on_overlap(ibmpg2):
    net, node_map, roots = ibmpg2
    ports = build_ports(net, 927.0, node_map=node_map, roots=roots)
    model = build_pixel_model(ports)
    Gs = build_Gs(model, 1.0, 2.0, 0.5)
    r = compare_gs_pad_coupling(
        n_pads=ports.n_pads,
        n_sinks=ports.n_sinks,
        pad_attach=model.pad_attach,
        Gs=Gs,
    )
    assert r.status == "pass", r.detail


@pytest.mark.skipif(
    not TC2.is_file() or not SPEC_OUT.is_dir(),
    reason="IBM TC2 or spec_flow OUT missing",
)
def test_check_spec_ibm_cli_on_existing_out():
    from check_spec_ibm import score_comp

    summary = score_comp(TC2, SPEC_OUT, flow="spec_flow")
    by = {c["clause"]: c for c in summary["clauses"]}
    assert by["A1_pad_occupancy"]["status"] == "pass"
    assert by["A2_current_lumping"]["status"] == "pass"
    assert by["A3_full_sink_grid"]["status"] == "pass"
    assert by["A5_pad_voltage"]["status"] == "pass"
    assert by["B_kron_shape"]["status"] in ("pass", "skip")
    # Fresh audit OUT uses eigen fit (spec default)
    assert by["D_eigen_fit"]["status"] == "pass"
