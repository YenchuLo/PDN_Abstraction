"""Via-stub flag: zero (short) vs Rx/Ry-proportional L-bend."""

from __future__ import annotations

import pytest

from io_artifacts import load_ports, save_ports
from pixel_r import build_Gs, build_pixel_model
from ports import (
    DEFAULT_VDD,
    VIA_STUB_RXRY,
    VIA_STUB_ZERO,
    GridCell,
    PortSet,
    parse_via_stub_arg,
    via_series_r,
)
from spice_emit import emit_pixel_r


def _ports(*, via_stub: str = VIA_STUB_ZERO, offset: bool = False) -> PortSet:
    cells = [
        GridCell(ix=0, iy=0, node="s0", x=5.0, y=5.0, current=-0.1, members=["s0"]),
        GridCell(ix=1, iy=0, node="s1", x=15.0, y=5.0, current=-0.1, members=["s1"]),
    ]
    pad_xy = [(8.0, 7.0), (16.0, 5.0)] if offset else [(5.0, 5.0), (15.0, 5.0)]
    return PortSet(
        pad_nodes=["p0", "p1"],
        pad_voltages=[DEFAULT_VDD, DEFAULT_VDD],
        cells=cells,
        cell_size=10.0,
        nx=2,
        ny=1,
        bbox=(0.0, 0.0, 20.0, 10.0),
        pad_xy=pad_xy,
        via_stub=via_stub,
    )


def test_parse_via_stub_arg():
    assert parse_via_stub_arg(None) == VIA_STUB_ZERO
    assert parse_via_stub_arg("zero") == VIA_STUB_ZERO
    assert parse_via_stub_arg("RXRY") == VIA_STUB_RXRY
    assert parse_via_stub_arg("nonzero") == VIA_STUB_RXRY
    with pytest.raises(ValueError, match="via_stub"):
        parse_via_stub_arg("bilinear")


def test_via_series_r_zero_ignores_offset():
    ports = _ports(via_stub=VIA_STUB_ZERO, offset=True)
    assert via_series_r(ports, 2.0, 4.0, 0.5, 0, 0) == pytest.approx(0.5)


def test_via_series_r_rxry_is_length_proportional():
    ports = _ports(via_stub=VIA_STUB_RXRY, offset=True)
    # pad0 at (8,7), center (5,5), pitch 10 → |dx|/n=0.3, |dy|/n=0.2
    assert via_series_r(ports, 2.0, 4.0, 0.5, 0, 0) == pytest.approx(
        0.5 + 2.0 * 0.3 + 4.0 * 0.2
    )
    # pad1 at (16,5), center (15,5) → |dx|/n=0.1, |dy|=0
    assert via_series_r(ports, 2.0, 4.0, 0.5, 1, 1) == pytest.approx(0.5 + 2.0 * 0.1)


def test_build_gs_zero_matches_aligned():
    aligned = _ports(via_stub=VIA_STUB_RXRY, offset=False)
    offset_short = _ports(via_stub=VIA_STUB_ZERO, offset=True)
    Gs_a = build_Gs(build_pixel_model(aligned), 1.0, 2.0, 0.4)
    Gs_z = build_Gs(build_pixel_model(offset_short), 1.0, 2.0, 0.4)
    assert Gs_a == pytest.approx(Gs_z)


def test_build_gs_rxry_weakens_via():
    ports = _ports(via_stub=VIA_STUB_RXRY, offset=True)
    Gs = build_Gs(build_pixel_model(ports), 2.0, 4.0, 0.5)
    n_p = ports.n_pads
    g0 = -Gs[0, n_p + 0]
    g1 = -Gs[1, n_p + 1]
    assert g0 == pytest.approx(1.0 / (0.5 + 2.0 * 0.3 + 4.0 * 0.2))
    assert g1 == pytest.approx(1.0 / (0.5 + 2.0 * 0.1))
    assert g0 < g1 < 1.0 / 0.5


def test_save_load_preserves_pad_xy_and_via_stub(tmp_path):
    ports = _ports(via_stub=VIA_STUB_RXRY, offset=True)
    save_ports(tmp_path, ports, {})
    loaded, _ = load_ports(tmp_path)
    assert loaded.via_stub == VIA_STUB_RXRY
    assert loaded.pad_xy == [(8.0, 7.0), (16.0, 5.0)]


def test_emit_pixel_r_writes_stub_resistors(tmp_path):
    ports = _ports(via_stub=VIA_STUB_RXRY, offset=True)
    model = build_pixel_model(ports)
    path = emit_pixel_r(
        tmp_path,
        ports,
        model.pad_attach,
        model.arms,
        model.n_edge_nodes,
        Rx=2.0,
        Ry=4.0,
        Rz=0.5,
    )
    text = path.read_text()
    assert "via_stub=rxry" in text
    assert "Rstubx_" in text
    assert "Rstuby_" in text
    assert "0.6" in text  # Rx * 0.3
    assert "0.8" in text  # Ry * 0.2


def test_emit_pixel_r_zero_has_no_stubs(tmp_path):
    ports = _ports(via_stub=VIA_STUB_ZERO, offset=True)
    model = build_pixel_model(ports)
    path = emit_pixel_r(
        tmp_path,
        ports,
        model.pad_attach,
        model.arms,
        model.n_edge_nodes,
        Rx=2.0,
        Ry=4.0,
        Rz=0.5,
    )
    text = path.read_text()
    assert "via_stub=zero" in text
    assert "Rstubx_" not in text
    assert "Rstuby_" not in text
