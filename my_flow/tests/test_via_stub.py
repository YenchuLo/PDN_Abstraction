"""--via-stub rxry (L-bend) vs zero (snap to nearest grid)."""

from __future__ import annotations

import pytest

from stub_attach import (
    VIA_STUB_RXRY,
    VIA_STUB_ZERO,
    build_stub_square_graph,
    parse_via_stub_arg,
)


def test_parse_via_stub_arg_defaults_rxry():
    assert parse_via_stub_arg(None) == VIA_STUB_RXRY
    assert parse_via_stub_arg("rxry") == VIA_STUB_RXRY
    assert parse_via_stub_arg("zero") == VIA_STUB_ZERO
    with pytest.raises(ValueError, match="via_stub"):
        parse_via_stub_arg("bilinear")


def test_rxry_adds_landing_off_grid():
    g = build_stub_square_graph(
        pad_xy=[(3.0, 3.0)],
        sink_xy=[(7.0, 7.0)],
        pitch=10.0,
        bbox=(0.0, 0.0, 10.0, 10.0),
        via_stub=VIA_STUB_RXRY,
    )
    assert g.via_stub == VIA_STUB_RXRY
    assert g.pad_landing[0] >= g.n_grid
    assert g.sink_landing[0] >= g.n_grid


def test_zero_snaps_to_grid():
    g = build_stub_square_graph(
        pad_xy=[(3.0, 3.0)],
        sink_xy=[(7.0, 7.0)],
        pitch=10.0,
        bbox=(0.0, 0.0, 10.0, 10.0),
        via_stub=VIA_STUB_ZERO,
    )
    assert g.via_stub == VIA_STUB_ZERO
    assert g.pad_landing[0] < g.n_grid
    assert g.sink_landing[0] < g.n_grid
    assert len(g.xy) == g.n_grid
