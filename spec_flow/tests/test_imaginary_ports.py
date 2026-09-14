"""Unit tests for C4-overlap / full-grid port construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from graph import build_short_map, vdd_connected_components
from ports import build_ports, grid_counts
from pixel_r import build_pixel_model
from spice_emit import emit_original
from spice_parser import parse_spice

REPO = Path(__file__).resolve().parents[2]
TC1 = REPO / "Benchmarks" / "IBM" / "TC1" / "ibmpg1.spice"


@pytest.mark.skipif(not TC1.is_file(), reason="IBM TC1 spice not present")
def test_imaginary_ports_full_grid_unique(tmp_path):
    net = parse_spice(TC1)
    node_map = build_short_map(net)
    xmin, ymin, xmax, ymax = net.bbox
    pitch = max(xmax - xmin, ymax - ymin) / 4.0
    # Use the topmost VDD component (same as flow comp1), not the union of all rails.
    roots = vdd_connected_components(net, node_map)[0]
    ports = build_ports(net, pitch, node_map=node_map, roots=roots)

    nx, ny = grid_counts(net.bbox, pitch)
    assert ports.nx == nx and ports.ny == ny
    assert ports.n_sinks == nx * ny
    assert 1 <= ports.n_pads <= ports.n_sinks
    assert len(set(ports.pad_nodes)) == ports.n_pads
    assert len(set(ports.sink_nodes)) == ports.n_sinks
    assert set(ports.pad_nodes).isdisjoint(set(ports.sink_nodes))
    # ibmpg1 C4s are 1.8 V — pad voltages follow the deck, not DEFAULT_VDD.
    assert ports.pad_voltages
    assert all(abs(v - 1.8) < 1e-12 for v in ports.pad_voltages)
    # Geometry is at geometric cell centers (electrical attach stays nearest-node).
    xmin, ymin, _, _ = net.bbox
    for c in ports.cells:
        cx = xmin + (c.ix + 0.5) * pitch
        cy = ymin + (c.iy + 0.5) * pitch
        assert c.x == cx and c.y == cy
    assert abs(ports.vdd - 1.8) < 1e-12
    assert ports.vdd_layers == [1, 3]
    assert ports.bot_layer == 1
    assert ports.top_layer == 3
    assert ports.max_metal_pitch > 0
    assert pitch > ports.max_metal_pitch
    assert set(ports.metal_pitches.keys()) == {1, 3}
    # Largest-gap inference: M1 max gap > fine modal (~33); M3 inter-band ~1.8k
    assert ports.metal_pitches[1] > 100
    assert ports.metal_pitches[3] > 1000
    assert ports.max_metal_pitch == ports.metal_pitches[3]

    # C4 overlap at any pitch: some cells may be sink-only.
    fine = build_ports(
        net, 0.5 * ports.max_metal_pitch, node_map=node_map, roots=roots
    )
    assert fine.n_sinks == fine.nx * fine.ny
    assert 1 <= fine.n_pads <= fine.n_sinks
    assert len(fine.pad_attach) == fine.n_pads
    assert len(set(fine.pad_nodes)) == fine.n_pads
    assert set(fine.pad_nodes).isdisjoint(set(fine.sink_nodes))

    model = build_pixel_model(ports)
    assert model.pad_attach == ports.resolved_pad_attach()
    assert len(model.pad_attach) == ports.n_pads
    assert model.to_fields()["topology"] == "star_half_arm"
    assert len(model.ew_shared) + len(model.ns_shared) > 0
    assert len(model.arms) > 0
    assert model.n_edge_nodes > 0

    path = emit_original(tmp_path, net, ports, node_map)
    text = path.read_text()
    assert ".include" not in text
    assert "Vpad_0" in text
    assert "Isink_0" in text
    assert "VDD" in text or "vdd" in text.lower() or "VDD R-network" in text
    assert " 1.8" in text or " 1.80000000000" in text
    # No benchmark-style leftover include of ibmpg
    assert "ibmpg" not in text.lower()
    # GND-stack layer-0 node names should not appear as R endpoints if omitted
    assert "Rvia_" in text


def test_grid_counts_example():
    nx, ny = grid_counts((0.0, 0.0, 1000.0, 1000.0), 10.0)
    assert (nx, ny) == (100, 100)
