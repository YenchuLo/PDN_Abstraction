"""Uniform sink-current helper tests."""

from __future__ import annotations

import pytest

from ports import (
    GridCell,
    apply_uniform_sink_currents,
    parse_uniform_current_arg,
    uniform_current_out_tag,
)


def _cells(currents: list[float]) -> list[GridCell]:
    return [
        GridCell(ix=i, iy=0, node=f"n{i}", x=float(i), y=0.0, current=c)
        for i, c in enumerate(currents)
    ]


def test_parse_uniform_current_arg():
    assert parse_uniform_current_arg(None) is None
    assert parse_uniform_current_arg("off") is None
    assert parse_uniform_current_arg("0") is None
    assert parse_uniform_current_arg("conserve") == "conserve"
    assert parse_uniform_current_arg("ON") == "conserve"
    assert parse_uniform_current_arg("1e-3") == pytest.approx(1e-3)
    with pytest.raises(ValueError, match="invalid"):
        parse_uniform_current_arg("nope")


def test_apply_uniform_conserve_preserves_total():
    cells = _cells([-1.0, -3.0, 0.0, 0.0])
    meta = apply_uniform_sink_currents(cells, "conserve")
    assert meta["uniform_current_mode"] == "conserve"
    assert meta["i_each"] == pytest.approx(-1.0)
    assert all(c.current == pytest.approx(-1.0) for c in cells)
    assert meta["i_total_after"] == pytest.approx(-4.0)


def test_apply_uniform_fixed_draw():
    cells = _cells([-1.0, 0.0])
    meta = apply_uniform_sink_currents(cells, 2e-3)
    assert meta["uniform_current_mode"] == "fixed"
    assert meta["i_each"] == pytest.approx(-2e-3)
    assert all(c.current == pytest.approx(-2e-3) for c in cells)


def test_uniform_current_out_tag():
    assert uniform_current_out_tag(None) == ""
    assert uniform_current_out_tag("conserve") == "_uI"
    assert uniform_current_out_tag(1e-3) == "_uI0.001"
