"""Grid-level Kron-branch R from G' (Pixel-R stencil, not pads-grounded Reff)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from grid_reff import compute_grid_reff, kron_branch_r, two_terminal_r, write_grid_reff
from pixel_r import build_Gs, build_pixel_model
from ports import DEFAULT_VDD, GridCell, PortSet


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
                    x=float(ix),
                    y=float(iy),
                    current=-0.1,
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


def test_kron_branch_matches_pixel_r_params():
    ports = _tiny_ports(3, 2)
    Rx, Ry, Rz = 1.0, 2.0, 0.5
    G = build_Gs(build_pixel_model(ports), Rx, Ry, Rz)
    data = compute_grid_reff(G, ports)
    assert np.allclose(data["r_z"], Rz)
    assert np.allclose(data["r_ew"], 2.0 * Rx)
    assert np.allclose(data["r_ns"], 2.0 * Ry)
    assert data["r_ew"].size == 4
    assert data["r_ns"].size == 3
    assert np.isnan(data["r_ew_map"][:, -1]).all()
    assert np.isnan(data["r_ns_map"][-1, :]).all()
    # Pads-grounded neighbor Reff is *not* 2 Rx (via short dominates).
    assert np.median(data["r_ew_grounded"]) < 1.5 * Rx
    # Combined driving-point is smaller than Rz (lateral spreading).
    assert np.max(data["r_pad_sink"]) < Rz


def test_reff_scales_inverse_with_conductance():
    ports = _tiny_ports(2, 2)
    model = build_pixel_model(ports)
    G = build_Gs(model, 0.8, 1.2, 0.4)
    d1 = compute_grid_reff(G, ports)
    d2 = compute_grid_reff(2.0 * G, ports)
    assert np.allclose(d2["r_z"], 0.5 * d1["r_z"], rtol=1e-8)
    assert np.allclose(d2["r_ew"], 0.5 * d1["r_ew"], rtol=1e-8)
    assert np.allclose(d2["r_ns"], 0.5 * d1["r_ns"], rtol=1e-8)


def test_two_terminal_symmetric():
    z = np.array([[2.0, 0.5], [0.5, 3.0]])
    assert two_terminal_r(z, 0, 1) == pytest.approx(4.0)
    assert two_terminal_r(z, 0, 1) == two_terminal_r(z, 1, 0)


def test_kron_branch_nan_if_uncoupled():
    G = np.eye(2)
    assert np.isnan(kron_branch_r(G, 0, 1))


def test_write_grid_reff_artifacts(tmp_path):
    ports = _tiny_ports(2, 2)
    model = build_pixel_model(ports)
    G = build_Gs(model, 1.0, 1.0, 0.5)
    paths = write_grid_reff(tmp_path, G, ports)
    assert (tmp_path / "grid_reff.npz").is_file()
    assert (tmp_path / "grid_reff.json").is_file()
    assert "png" in paths
    assert "dp_png" in paths
    assert (tmp_path / "grid_reff_maps.png").is_file()
    assert (tmp_path / "grid_reff_dp.png").is_file()
    payload = json.loads((tmp_path / "grid_reff.json").read_text())
    assert payload["kind"] == "kron_branch"
    assert payload["stats"]["R_z"]["n"] == 4
    assert payload["stats"]["R_z"]["median"] == pytest.approx(0.5)
