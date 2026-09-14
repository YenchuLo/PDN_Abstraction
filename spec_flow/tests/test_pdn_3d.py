"""Unit tests for 3D PDN geometry lift, Pixel-R drawable, and figure build."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdn_geometry import extract_pdn_geometry
from pdn_geometry_3d import (
    default_z_pitch,
    layer_z_map,
    lift_pdn_geometry,
    vias_to_xyz,
    z_of,
)
from pdn_plot_3d import build_figure_3d, write_figure_png_3d
from pixel_geometry_3d import build_pixel_geometry_3d
from pixel_r import build_pixel_model
from ports import GridCell, PortSet
from spice_parser import SpiceNetlist, parse_node_coord


def _node(name: str):
    c = parse_node_coord(name)
    assert c is not None
    return name, c


def _tiny_dual_net() -> SpiceNetlist:
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
    net.resistors.append(("n1_0_0", "n1_10_0", 1.0))
    net.resistors.append(("n3_0_0", "n3_10_0", 1.0))
    net.resistors.append(("n0_0_5", "n0_10_5", 1.0))
    net.resistors.append(("n2_0_5", "n2_10_5", 1.0))
    net.voltages.append(("n1_0_0", "n3_0_0", 0.0))
    net.voltages.append(("n0_0_5", "n2_0_5", 0.0))
    net.voltages.append(("n3_10_0", "0", 0.9))
    net.currents.append(("n1_10_0", "0", -0.1))
    net.voltages.append(("n0_10_5", "0", 0.0))
    return net


def test_layer_z_mapping_and_via_heights():
    layers = [0, 1, 2, 3]
    z_pitch = 10.0
    lz = layer_z_map(layers, z_pitch)
    assert lz[0] == 0.0
    assert lz[1] == 10.0
    assert lz[3] == 30.0
    assert z_of(2, lz, z_pitch) == 20.0

    vias = [(0.0, 0.0, 1, 0.0, 0.0, 3)]
    xs, ys, zs = vias_to_xyz(vias, lz, z_pitch)
    assert xs[:2] == [0.0, 0.0]
    assert zs[0] == 10.0
    assert zs[1] == 30.0
    assert zs[0] != zs[1]


def test_lift_pdn_geometry_on_tiny():
    net = _tiny_dual_net()
    geom2d = extract_pdn_geometry(net)
    geom3d = lift_pdn_geometry(geom2d, z_pitch=5.0)

    assert geom3d.z_pitch == 5.0
    assert geom3d.vdd.n_metal == 2
    assert geom3d.vss.n_metal == 2
    # VDD M1 and M3 at different z
    m1 = geom3d.vdd.metal[1][0]
    m3 = geom3d.vdd.metal[3][0]
    assert m1[2] == m1[5]  # horizontal metal
    assert m3[2] == m3[5]
    assert m1[2] != m3[2]


def test_default_z_pitch_positive():
    assert default_z_pitch((0.0, 0.0, 100.0, 0.0)) > 0.0
    assert default_z_pitch((0.0, 0.0, 0.0, 0.0)) == 1.0


def _tiny_ports() -> PortSet:
    """2x2 sink grid with matching pads."""
    cells = [
        GridCell(ix=0, iy=0, node="n1_0_0", x=0.0, y=0.0, current=-0.1),
        GridCell(ix=1, iy=0, node="n1_10_0", x=10.0, y=0.0, current=-0.1),
        GridCell(ix=0, iy=1, node="n1_0_10", x=0.0, y=10.0, current=-0.1),
        GridCell(ix=1, iy=1, node="n1_10_10", x=10.0, y=10.0, current=-0.1),
    ]
    return PortSet(
        pad_nodes=["n3_0_0", "n3_10_0", "n3_0_10", "n3_10_10"],
        pad_voltages=[0.9, 0.9, 0.9, 0.9],
        cells=cells,
        cell_size=10.0,
        nx=2,
        ny=2,
        bbox=(0.0, 0.0, 10.0, 10.0),
        top_layer=3,
        bot_layer=1,
        vdd_layers=[1, 3],
    )


def test_pixel_geometry_edge_counts():
    ports = _tiny_ports()
    model = build_pixel_model(ports)
    lz = layer_z_map([1, 3], z_pitch=5.0)
    pix = build_pixel_geometry_3d(
        ports,
        model.pad_attach,
        model.ew_shared,
        model.ns_shared,
        lz,
        5.0,
        ew_boundary=model.ew_boundary,
        ns_boundary=model.ns_boundary,
        Rx=1.0,
        Ry=2.0,
        Rz=3.0,
    )
    assert pix.n_pads == 4
    assert pix.n_sinks == 4
    # 2x2 star: 2 shared EW × 2 half-arms + 4 boundary = 8 Rx segments
    # same for Ry; Rup+Rdown → 8 Rz segments
    assert pix.n_rx == 2 * len(model.ew_shared) + len(model.ew_boundary)
    assert pix.n_ry == 2 * len(model.ns_shared) + len(model.ns_boundary)
    assert pix.n_rz == 2 * len(model.pad_attach)
    assert pix.n_rx == 8
    assert pix.n_ry == 8
    assert pix.n_rz == 8
    assert pix.z_pad > pix.z_sink
    # Markers at geometric cell centers
    assert pix.sinks[0][:2] == (5.0, 5.0)
    # Rz vertical at same xy
    x1, y1, z1, x2, y2, z2 = pix.rz_edges[0]
    assert x1 == x2 and y1 == y2
    assert z1 != z2


def test_build_figure_3d_smoke():
    pytest.importorskip("plotly")
    net = _tiny_dual_net()
    geom2d = extract_pdn_geometry(net)
    geom3d = lift_pdn_geometry(geom2d, z_pitch=5.0)
    ports = _tiny_ports()
    model = build_pixel_model(ports)
    pix = build_pixel_geometry_3d(
        ports,
        model.pad_attach,
        model.ew_shared,
        model.ns_shared,
        geom3d.layer_z,
        geom3d.z_pitch,
        ew_boundary=model.ew_boundary,
        ns_boundary=model.ns_boundary,
        Rx=1e-3,
        Ry=1e-3,
        Rz=1e-6,
    )
    fig = build_figure_3d(
        geom3d,
        pix,
        default_mode="both",
        default_net="vdd",
        default_layer=None,
    )
    assert len(fig.data) > 0
    titles = [
        a.text for a in fig.layout.annotations if a.text and "PDN 3D" in a.text
    ]
    assert titles and "Original | Pixel-R" in titles[0]
    # Side-by-side scenes + separate legends
    assert fig.layout.scene is not None
    assert fig.layout.scene2 is not None
    assert fig.layout.legend2 is not None
    scenes = {getattr(tr, "scene", "scene") for tr in fig.data}
    assert "scene" in scenes and "scene2" in scenes
    # Pixel legend traces present; sinks larger than legacy size-3 markers
    names = {tr.name for tr in fig.data}
    assert "Pads" in names
    assert "Sinks" in names
    assert "Rx edges" in names
    assert "Rz vias" in names
    sink = next(tr for tr in fig.data if tr.name == "Sinks")
    assert sink.marker.size >= 8
    # Mode + View menus
    assert len(fig.layout.updatemenus) == 2


def test_build_figure_3d_original_only():
    pytest.importorskip("plotly")
    net = _tiny_dual_net()
    geom3d = lift_pdn_geometry(extract_pdn_geometry(net), z_pitch=5.0)
    fig = build_figure_3d(geom3d, None, default_mode="both", default_net="vdd")
    # Falls back to original when no pixel
    titles = [
        a.text for a in fig.layout.annotations if a.text and "PDN 3D" in a.text
    ]
    assert titles and "Original" in titles[0]
    assert len(fig.layout.updatemenus[0].buttons) == 1


def test_write_figure_png_3d_matplotlib(tmp_path, monkeypatch):
    pytest.importorskip("plotly")
    pytest.importorskip("matplotlib")
    net = _tiny_dual_net()
    geom3d = lift_pdn_geometry(extract_pdn_geometry(net), z_pitch=5.0)
    fig = build_figure_3d(geom3d, None, default_mode="original", default_net="vdd")

    def _boom(*_a, **_k):
        raise AssertionError("fig.write_image (kaleido) must not be called")

    monkeypatch.setattr(fig, "write_image", _boom, raising=False)
    out = write_figure_png_3d(fig, tmp_path / "pdn_view_3d.png", width=640, height=360, scale=1.0)
    assert out is not None
    assert out.is_file()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_ibmpg1_n500_smoke_if_present():
    pytest.importorskip("plotly")
    out = Path(__file__).resolve().parents[1] / "outputs" / "ibmpg1_n500"
    if not (out / "net.npz").is_file():
        pytest.skip("ibmpg1_n500/net.npz not present")
    from io_artifacts import load_net, load_pixel_model_fields, load_ports
    import json

    net = load_net(out)
    geom3d = lift_pdn_geometry(extract_pdn_geometry(net))
    ports, _ = load_ports(out)
    fields = load_pixel_model_fields(out)
    fit = {}
    if (out / "pixel_r.json").is_file():
        fit = json.loads((out / "pixel_r.json").read_text())
    ew_shared = fields.get("ew_shared", fields.get("ew_edges", []))
    ns_shared = fields.get("ns_shared", fields.get("ns_edges", []))
    pix = build_pixel_geometry_3d(
        ports,
        fields["pad_attach"],
        ew_shared,
        ns_shared,
        geom3d.layer_z,
        geom3d.z_pitch,
        ew_boundary=fields.get("ew_boundary", []),
        ns_boundary=fields.get("ns_boundary", []),
        Rx=fit.get("Rx"),
        Ry=fit.get("Ry"),
        Rz=fit.get("Rz"),
    )
    assert pix.n_rx == 2 * len(ew_shared) + len(fields.get("ew_boundary", []))
    assert pix.n_ry == 2 * len(ns_shared) + len(fields.get("ns_boundary", []))
    assert pix.n_rz == 2 * len(fields["pad_attach"])
    fig = build_figure_3d(geom3d, pix, default_mode="both", default_net="vdd")
    assert len(fig.data) > 0
    # Count visible traces in Both/VDD/all
    n_vis = sum(1 for tr in fig.data if tr.visible)
    assert n_vis > 0
