"""Unit tests for localized Pixel-R partitions, stamps, and IR fit."""

from __future__ import annotations

import numpy as np
import pytest

from correlate import build_stimuli
from fit_ir import fit_localized_ir
from localized_pixel import build_Gs, build_Gs_from_regions, build_localized_model
from partitions import build_partition, expand_region_params
from pixel_r import build_Gs as build_Gs_uniform
from pixel_r import build_pixel_model
from ports import (  # noqa: E402
    DEFAULT_VDD,
    VIA_STUB_RXRY,
    GridCell,
    PortSet,
    apply_uniform_sink_currents,
)


def _tiny_ports(nx: int = 2, ny: int = 2) -> PortSet:
    cells = []
    pads = []
    for iy in range(ny):
        for ix in range(nx):
            pads.append(f"p{ix}_{iy}")
            cells.append(
                GridCell(
                    ix=ix,
                    iy=iy,
                    node=f"s{ix}_{iy}",
                    x=5.0 + 10.0 * ix,
                    y=5.0 + 10.0 * iy,
                    current=-0.1,
                    members=[f"s{ix}_{iy}"],
                )
            )
    return PortSet(
        pad_nodes=pads,
        pad_voltages=[DEFAULT_VDD] * (nx * ny),
        cells=cells,
        cell_size=10.0,
        nx=nx,
        ny=ny,
        bbox=(0.0, 0.0, 10.0 * nx, 10.0 * ny),
        top_layer=3,
        bot_layer=1,
        vdd=DEFAULT_VDD,
        vdd_layers=[1, 3],
        metal_pitches={1: 4.0, 3: 10.0},
        max_metal_pitch=10.0,
    )


def test_partition_k0_global():
    ports = _tiny_ports(4, 4)
    part = build_partition(ports, 0)
    assert part.n_regions == 1
    assert part.cell_to_region == [0] * 16


def test_partition_k1_per_cell():
    ports = _tiny_ports(3, 2)
    part = build_partition(ports, 1)
    assert part.n_regions == 6
    assert sorted(part.cell_to_region) == list(range(6))


def test_partition_k2_with_remainder():
    # 3x3 with K=2 → blocks at (0,0),(2,0),(0,2),(2,2)
    ports = _tiny_ports(3, 3)
    part = build_partition(ports, 2)
    assert part.n_regions == 4
    # corner cell (2,2) alone in its block
    k_corner = next(k for k, c in enumerate(ports.cells) if c.ix == 2 and c.iy == 2)
    rid = part.cell_to_region[k_corner]
    assert part.region_cells[rid] == [k_corner]


def test_series_arm_stamp_matches_uniform_when_equal():
    ports = _tiny_ports(2, 2)
    model = build_localized_model(ports, 0)
    Rx = np.full(4, 1.0)
    Ry = np.full(4, 2.0)
    Rz = np.full(4, 0.5)
    Gs_loc = build_Gs(model, Rx, Ry, Rz)
    Gs_uni = build_Gs_uniform(model.star, 1.0, 2.0, 0.5)
    assert Gs_loc.shape == (8, 8)
    assert np.allclose(Gs_loc, Gs_uni)
    assert np.allclose(Gs_loc, Gs_loc.T)


def test_series_arm_asymmetric():
    ports = _tiny_ports(2, 1)
    model = build_localized_model(ports, 1)
    Rx = np.array([1.0, 3.0])
    Ry = np.array([1.0, 1.0])
    Rz = np.array([0.5, 0.5])
    Gs = build_Gs(model, Rx, Ry, Rz)
    n_p = 2
    # EW between sinks 0 and 1: g = 1/(1+3) = 0.25
    assert Gs[n_p + 0, n_p + 1] == pytest.approx(-0.25)


def test_k0_uniform_recovers_one_triple():
    ports = _tiny_ports(2, 2)
    model = build_localized_model(ports, 0)
    # Synthetic G' from a known uniform Pixel-R
    star = build_pixel_model(ports)
    Gprime = build_Gs_uniform(star, 0.7, 1.4, 0.3)
    fit = fit_localized_ir(Gprime, model, ports, seed=0, max_feng_iters=5)
    assert model.n_regions == 1
    assert fit.Rx_r.size == 1
    assert fit.relative_ir_error < 0.05
    assert abs(fit.Rx_r[0] - 0.7) / 0.7 < 0.2
    assert abs(fit.Ry_r[0] - 1.4) / 1.4 < 0.2
    assert abs(fit.Rz_r[0] - 0.3) / 0.3 < 0.2


def test_two_region_recovers_distinct_r():
    """Left block soft, right block stiff on a 4×2 lattice (non-degenerate series)."""
    ports = _tiny_ports(4, 2)
    model = build_localized_model(ports, 2)  # two 2×2 regions side by side
    assert model.n_regions == 2
    # Soft left (ix 0–1), stiff right (ix 2–3)
    Rx_true = np.array(
        [0.5 if c.ix < 2 else 2.0 for c in ports.cells], dtype=float
    )
    Ry_true = np.ones(ports.n_sinks, dtype=float)
    Rz_true = np.full(ports.n_sinks, 0.4, dtype=float)
    Gprime = build_Gs(model, Rx_true, Ry_true, Rz_true)
    fit = fit_localized_ir(Gprime, model, ports, seed=0, max_feng_iters=15)
    assert fit.relative_ir_error < 0.05
    soft_rx = float(fit.Rx_r[0])
    stiff_rx = float(fit.Rx_r[1])
    # Regions ordered by (by, bx) → left block first, right second
    assert soft_rx < stiff_rx
    assert stiff_rx / soft_rx > 1.5


def test_expand_region_params():
    ports = _tiny_ports(2, 2)
    part = build_partition(ports, 2)
    assert part.n_regions == 1  # 2x2 with K=2 → one block
    Rx, Ry, Rz = expand_region_params(part, [1.5], [2.5], [0.25])
    assert np.allclose(Rx, 1.5)
    assert np.allclose(Ry, 2.5)
    assert np.allclose(Rz, 0.25)


def test_uniform_current_is_original_ir_stimulus():
    ports = _tiny_ports(2, 2)
    ports.cells[0].current = -1.0
    ports.cells[1].current = -3.0
    ports.cells[2].current = 0.0
    ports.cells[3].current = 0.0
    meta = apply_uniform_sink_currents(ports.cells, "conserve")
    assert meta["uniform_current_mode"] == "conserve"
    assert meta["i_each"] == pytest.approx(-1.0)
    name, I = build_stimuli(ports, seed=0)[0]
    assert name == "original"
    assert np.allclose(I, -1.0)
    star = build_pixel_model(ports)
    Gprime = build_Gs_uniform(star, 0.8, 1.2, 0.4)
    model = build_localized_model(ports, 0)
    fit = fit_localized_ir(Gprime, model, ports, seed=0, max_feng_iters=5)
    assert fit.relative_ir_error < 0.05


def test_localized_via_stub_rxry_series():
    ports = _tiny_ports(2, 1)
    ports.via_stub = VIA_STUB_RXRY
    ports.pad_xy = [(8.0, 5.0), (15.0, 5.0)]
    model = build_localized_model(ports, 1)
    Rx = np.array([1.0, 1.0])
    Ry = np.array([2.0, 2.0])
    Rz = np.array([0.5, 0.5])
    Gs = build_Gs(model, Rx, Ry, Rz)
    n_p = 2
    # pad0: |dx|/n = 0.3 → R = 0.5 + 1.0*0.3
    assert Gs[0, n_p + 0] == pytest.approx(-1.0 / 0.8)
    # pad1: aligned → R = 0.5
    assert Gs[1, n_p + 1] == pytest.approx(-1.0 / 0.5)


def test_sparse_pads_gs_shape_and_no_rz_on_padless():
    """2×2 sinks, pads only on (0,0) and (1,1); padless cells still have I."""
    cells = []
    for iy in range(2):
        for ix in range(2):
            cells.append(
                GridCell(
                    ix=ix,
                    iy=iy,
                    node=f"s{ix}_{iy}",
                    x=5.0 + 10.0 * ix,
                    y=5.0 + 10.0 * iy,
                    current=-0.1,
                    members=[f"s{ix}_{iy}"],
                )
            )
    ports = PortSet(
        pad_nodes=["p0_0", "p1_1"],
        pad_voltages=[DEFAULT_VDD, DEFAULT_VDD],
        cells=cells,
        cell_size=10.0,
        nx=2,
        ny=2,
        bbox=(0.0, 0.0, 20.0, 20.0),
        pad_xy=[(5.0, 5.0), (15.0, 15.0)],
        pad_attach=[0, 3],
        max_metal_pitch=20.0,
        metal_pitches={1: 4.0, 3: 20.0},
    )
    assert ports.n_padless == 2
    model = build_localized_model(ports, 1)
    assert model.n_pads == 2 and model.n_sinks == 4
    Rx = np.ones(4)
    Ry = np.ones(4)
    Rz = np.full(4, 0.5)
    Gs = build_Gs(model, Rx, Ry, Rz)
    assert Gs.shape == (6, 6)
    n_p = 2
    assert Gs[0, n_p + 0] == pytest.approx(-1.0 / 0.5)
    assert Gs[1, n_p + 3] == pytest.approx(-1.0 / 0.5)
    assert Gs[0, n_p + 1] == pytest.approx(0.0)
    assert Gs[1, n_p + 1] == pytest.approx(0.0)
    # Lateral path still couples padless sink 1 to occupied sink 0
    assert Gs[n_p + 0, n_p + 1] < 0.0

    star = build_pixel_model(ports)
    Gprime = build_Gs_uniform(star, 0.8, 1.2, 0.4)
    fit = fit_localized_ir(Gprime, model, ports, seed=0, max_feng_iters=8)
    assert fit.relative_ir_error < 0.08


def test_localized_eigen_k0_recovers_uniform():
    from fit_eigen_local import fit_localized_eigen

    ports = _tiny_ports(2, 2)
    star = build_pixel_model(ports)
    Gprime = build_Gs_uniform(star, 0.7, 1.4, 0.3)
    model = build_localized_model(ports, 0)
    fit = fit_localized_eigen(Gprime, model, max_nfev=80)
    assert model.n_regions == 1
    assert fit.relative_spectral_error < 0.05
    assert abs(fit.Rx_r[0] - 0.7) / 0.7 < 0.2
    assert abs(fit.Ry_r[0] - 1.4) / 1.4 < 0.2
    assert abs(fit.Rz_r[0] - 0.3) / 0.3 < 0.2
