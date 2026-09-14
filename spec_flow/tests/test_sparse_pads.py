"""Pads only on grid cells that overlap a real C4."""

from __future__ import annotations

import numpy as np
import pytest

from graph import build_short_map, vdd_connected_components, vdd_metal_pitches
from grid_reff import compute_grid_reff
from io_artifacts import load_ports, save_ports
from pixel_r import build_Gs, build_pixel_model
from ports import GridCell, PortSet, build_ports, grid_counts
from spice_parser import NodeCoord, SpiceNetlist


def _add(net: SpiceNetlist, name: str, layer: int, x: float, y: float) -> str:
    net.node_names.add(name)
    net.coords[name] = NodeCoord(layer, float(x), float(y))
    return name


def _striped_top_net() -> SpiceNetlist:
    """40×40 die: M3 vertical stripes at x=5 and x=25 (pitch 20); M1 5-unit mesh."""
    net = SpiceNetlist()
    top_xs = (5, 25)
    top_ys = (0, 10, 20, 30, 40)
    for x in top_xs:
        names = [_add(net, f"n3_{x}_{y}", 3, x, y) for y in top_ys]
        for a, b in zip(names, names[1:]):
            net.resistors.append((a, b, 1.0))

    bot_xy = list(range(0, 41, 5))
    for y in bot_xy:
        for x in bot_xy:
            _add(net, f"n1_{x}_{y}", 1, x, y)
    for y in bot_xy:
        xs = [f"n1_{x}_{y}" for x in bot_xy]
        for a, b in zip(xs, xs[1:]):
            net.resistors.append((a, b, 1.0))
    for x in bot_xy:
        ys = [f"n1_{x}_{y}" for y in bot_xy]
        for a, b in zip(ys, ys[1:]):
            net.resistors.append((a, b, 1.0))

    for x in top_xs:
        for y in top_ys:
            net.resistors.append((f"n3_{x}_{y}", f"n1_{x}_{y}", 0.1))

    # Real C4s: 2D subset of the Y-stripes (not every top-metal node).
    for x, y in ((5, 0), (5, 20), (25, 0), (25, 20)):
        net.voltages.append((f"n3_{x}_{y}", "0", 0.9))
    net.currents.append(("n1_10_10", "0", 1e-3))
    return net


def test_sparse_pads_when_pitch_below_top_metal():
    net = _striped_top_net()
    node_map = build_short_map(net)
    roots = vdd_connected_components(net, node_map)[0]
    pitches = vdd_metal_pitches(net, node_map, roots)
    assert pitches[3] == pytest.approx(20.0)
    assert pitches[1] == pytest.approx(5.0)

    ports = build_ports(net, 10.0, node_map=node_map, roots=roots)
    nx, ny = grid_counts(net.bbox, 10.0)
    assert (nx, ny) == (4, 4)
    assert ports.n_sinks == 16
    assert ports.n_pads < ports.n_sinks
    assert ports.n_padless == ports.n_sinks - ports.n_pads
    assert ports.n_pads == 4  # C4s at (ix,iy) in {(0,0),(0,2),(2,0),(2,2)}
    occupied = {(ports.cells[s].ix, ports.cells[s].iy) for s in ports.pad_attach}
    assert occupied == {(0, 0), (0, 2), (2, 0), (2, 2)}
    assert set(ports.pad_nodes) == {
        "n3_5_0",
        "n3_5_20",
        "n3_25_0",
        "n3_25_20",
    }
    occupied_ix = {ix for ix, _iy in occupied}
    occupied_iy = {iy for _ix, iy in occupied}
    assert occupied_ix == {0, 2}
    assert occupied_iy == {0, 2}  # not full Y-columns of top metal
    assert all(ports.sink_has_pad()[s] for s in ports.pad_attach)
    assert ports.sink_has_pad().count(False) == 12

    model = build_pixel_model(ports)
    assert model.pad_attach == ports.pad_attach
    Gs = build_Gs(model, 1.0, 2.0, 0.5)
    assert Gs.shape == (ports.n_ports, ports.n_ports)
    n_p = ports.n_pads
    has = ports.sink_has_pad()
    for s, ok in enumerate(has):
        couplings = [abs(Gs[p, n_p + s]) for p in range(n_p)]
        if ok:
            assert max(couplings) > 1e-9
        else:
            assert max(couplings) < 1e-12

    data = compute_grid_reff(Gs, ports)
    for s, ok in enumerate(has):
        if ok:
            assert data["r_z"][s] == pytest.approx(0.5)
        else:
            assert np.isnan(data["r_z"][s])


def test_coarse_pitch_still_c4_overlap_only():
    """Pitch > max metal still places pads only on C4-overlapping cells."""
    net = _striped_top_net()
    node_map = build_short_map(net)
    roots = vdd_connected_components(net, node_map)[0]
    ports = build_ports(net, 25.0, node_map=node_map, roots=roots)
    assert ports.cell_size > ports.max_metal_pitch
    assert ports.n_sinks == ports.nx * ports.ny == 4
    # C4s at x={5,25} y={0,20} collapse into two cells at pitch 25.
    assert ports.n_pads == 2
    assert ports.n_padless == 2
    occupied = {(ports.cells[s].ix, ports.cells[s].iy) for s in ports.pad_attach}
    assert occupied == {(0, 0), (1, 0)}


def test_full_pads_fills_every_cell_with_nearest_top():
    """--full-pads: C4 cells keep real C4s; padless get unique nearest top-metal."""
    # Densify top Y so unused top-metal ≥ padless cells (12 of 16 at pitch 10).
    net = _striped_top_net()
    for x in (5, 25):
        prev = None
        for y in range(0, 41, 5):
            name = f"n3_{x}_{y}"
            if name not in net.node_names:
                _add(net, name, 3, float(x), float(y))
                net.resistors.append((name, f"n1_{x}_{y}", 0.1))
            if prev is not None and (prev, name, 1.0) not in net.resistors:
                net.resistors.append((prev, name, 1.0))
            prev = name

    node_map = build_short_map(net)
    roots = vdd_connected_components(net, node_map)[0]

    sparse = build_ports(net, 10.0, node_map=node_map, roots=roots)
    assert sparse.n_pads < sparse.n_sinks
    assert sparse.full_pads is False

    ports = build_ports(
        net, 10.0, node_map=node_map, roots=roots, full_pads=True
    )
    assert ports.full_pads is True
    assert ports.n_pads == ports.n_sinks == 16
    assert ports.n_padless == 0
    assert ports.pad_attach == list(range(16))
    assert len(set(ports.pad_nodes)) == 16

    # C4-overlapping cells still use the real C4 pad nodes.
    c4_cells = {(0, 0), (0, 2), (2, 0), (2, 2)}
    c4_nodes = {"n3_5_0", "n3_5_20", "n3_25_0", "n3_25_20"}
    for s, cell in enumerate(ports.cells):
        key = (cell.ix, cell.iy)
        node = ports.pad_nodes[s]
        assert node.startswith("n3_")
        if key in c4_cells:
            assert node in c4_nodes
        else:
            assert node not in c4_nodes
        # full_pads: every pad driven at the same representative C4 voltage.
        assert ports.pad_voltages[s] == pytest.approx(0.9)
    assert len(set(np.round(ports.pad_voltages, 12))) == 1
    assert ports.vdd == pytest.approx(0.9)

    model = build_pixel_model(ports)
    assert model.pad_attach == list(range(16))
    Gs = build_Gs(model, 1.0, 2.0, 0.5)
    n_p = ports.n_pads
    for s in range(ports.n_sinks):
        assert abs(Gs[s, n_p + s]) > 1e-9


def test_sparse_ports_roundtrip_json(tmp_path):
    cells = [
        GridCell(ix=0, iy=0, node="s0", x=5.0, y=5.0, current=-0.1, members=["s0"]),
        GridCell(ix=1, iy=0, node="s1", x=15.0, y=5.0, current=-0.1, members=["s1"]),
    ]
    ports = PortSet(
        pad_nodes=["p0"],
        pad_voltages=[0.9],
        cells=cells,
        cell_size=10.0,
        nx=2,
        ny=1,
        bbox=(0.0, 0.0, 20.0, 10.0),
        pad_xy=[(5.0, 5.0)],
        pad_attach=[0],
        max_metal_pitch=20.0,
        metal_pitches={1: 4.0, 3: 20.0},
        full_pads=False,
    )
    save_ports(tmp_path, ports, {"s0": "s0", "s1": "s1", "p0": "p0"})
    loaded, _ = load_ports(tmp_path)
    assert loaded.n_pads == 1
    assert loaded.n_sinks == 2
    assert loaded.pad_attach == [0]
    assert loaded.n_padless == 1
    assert loaded.sink_has_pad() == [True, False]
    assert loaded.full_pads is False
