"""C4-lattice auto pitch (default chip_pitch=0)."""

from __future__ import annotations

import math

import pytest

from ports import axis_lattice_pitch, c4_lattice_pitch, suggest_pad_pitch


def test_axis_lattice_pitch_regular():
    ys = [220.0 + 1112.0 * i for i in range(13)]
    assert axis_lattice_pitch(ys) == pytest.approx(1112.0)


def test_axis_lattice_pitch_merges_near_duplicates():
    xs = [0.0, 0.4, 100.0, 100.2, 200.0]
    assert axis_lattice_pitch(xs, merge_tol=1.0) == pytest.approx(100.0)


def test_c4_lattice_pitch_uses_coarser_axis():
    # ibmpg4-like: regular Y=1112, staggered X (186 / 925).
    ys = [220.0 + 1112.0 * i for i in range(13)]
    xs: list[float] = []
    x = 220.0
    for i in range(15):
        xs.append(x)
        x += 186.0 if i % 2 == 0 else 925.0
    pitch = c4_lattice_pitch(xs, ys)
    assert pitch == pytest.approx(1112.0)


def test_c4_lattice_pitch_needs_two_coordinates():
    with pytest.raises(ValueError, match="distinct"):
        c4_lattice_pitch([10.0, 10.4], [5.0, 5.2], merge_tol=1.0)


@pytest.mark.parametrize("ratio, scale", [(1.0, 1.0), (4.0, 0.5)])
def test_suggest_pad_pitch_scales_by_sqrt_ratio(ratio, scale, monkeypatch):
    pads = [
        (f"p{i}_{j}", float(100 * i), float(200 * j), 1.8)
        for i in range(4)
        for j in range(3)
    ]

    def _fake_pads(_net, _node_map, _roots):
        return pads

    monkeypatch.setattr("ports._real_pad_sites", _fake_pads)
    got = suggest_pad_pitch(object(), {}, set(), grid_to_pad_ratio=ratio)
    assert got == pytest.approx(200.0 * scale)


def test_suggest_pad_pitch_count_fallback(monkeypatch):
    pads = [("p0", 10.0, 20.0, 1.8)]

    class _Net:
        bbox = (0.0, 0.0, 100.0, 80.0)

    monkeypatch.setattr("ports._real_pad_sites", lambda *_a, **_k: pads)
    got = suggest_pad_pitch(_Net(), {}, set(), grid_to_pad_ratio=1.0)
    assert got == pytest.approx(100.0 / 2)
    assert math.isfinite(got)
