"""Synthetic mesh: Kron reduction + Pixel-R eigen fit sanity checks."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from fit_eigen import fit_pixel_r
from fit_ir import fit_pixel_r_ir
from kron import grounded_eigh, kron_reduce
from pixel_r import build_Gs, build_pixel_model, neighbor_sink_conductance
from ports import GridCell, PortSet


def _ladder_G(n_ports: int = 4, n_internal: int = 6, g: float = 1.0) -> tuple:
    """Simple chain: ports at ends interleaved with internals — use block mesh."""
    n = n_ports + n_internal
    G = sparse.lil_matrix((n, n), dtype=float)
    for i in range(n - 1):
        G[i, i] += g
        G[i + 1, i + 1] += g
        G[i, i + 1] -= g
        G[i + 1, i] -= g
    G[n - 1, n - 1] += 0.1
    port_idx = np.arange(n_ports)
    internal_idx = np.arange(n_ports, n)
    return G.tocsr(), port_idx, internal_idx


def test_kron_matches_dense_schur():
    G, port_idx, internal_idx = _ladder_G()
    Gp = kron_reduce(G, port_idx, internal_idx, reg=0.0)

    Gd = G.toarray()
    P, I = port_idx, internal_idx
    Gpp = Gd[np.ix_(P, P)]
    Gpi = Gd[np.ix_(P, I)]
    Gii = Gd[np.ix_(I, I)]
    Gii = Gii + np.eye(len(I)) * 1e-12
    ref = Gpp - Gpi @ np.linalg.solve(Gii, Gpi.T)
    ref = 0.5 * (ref + ref.T)
    assert np.allclose(Gp, ref, rtol=1e-5, atol=1e-6)


def test_star_half_arm_schur_neighbor_coupling():
    """Two series half-arms Rx Schur to conductance 1/(2·Rx) between sinks."""
    n = 2
    n_cells = n * n
    cells = [
        GridCell(
            ix=i % n,
            iy=i // n,
            node=f"s{i}",
            x=float(i % n),
            y=float(i // n),
            current=0.0,
        )
        for i in range(n_cells)
    ]
    ports = PortSet(
        pad_nodes=[f"p{k}" for k in range(n_cells)],
        pad_voltages=[0.9] * n_cells,
        cells=cells,
        cell_size=1.0,
        nx=n,
        ny=n,
        bbox=(0.0, 0.0, 2.0, 2.0),
    )
    model = build_pixel_model(ports)
    assert model.to_fields()["topology"] == "star_half_arm"
    assert len(model.ew_shared) == 2
    assert len(model.ns_shared) == 2
    # 4 cells × 4 arms, but shared edges counted once each → 8 shared half-arms
    # + 8 boundary dangling = 16 arms; n_edge = 2+2 shared + 8 boundary = 12
    assert model.n_edge_nodes == 12
    assert len(model.arms) == 16

    Rx_t, Ry_t, Rz_t = 2.0, 4.0, 5.0
    Gs = build_Gs(model, Rx_t, Ry_t, Rz_t)
    assert Gs.shape == (2 * n_cells, 2 * n_cells)

    a, b = model.ew_shared[0]
    g_ab = neighbor_sink_conductance(Gs, model, a, b)
    assert np.isclose(g_ab, 1.0 / (2.0 * Rx_t), rtol=1e-9, atol=1e-12)

    a, b = model.ns_shared[0]
    g_ab = neighbor_sink_conductance(Gs, model, a, b)
    assert np.isclose(g_ab, 1.0 / (2.0 * Ry_t), rtol=1e-9, atol=1e-12)

    # Pad–sink coupling remains 1/Rz (Rdown floats out)
    for p, s in enumerate(model.pad_attach):
        assert np.isclose(-Gs[p, n_cells + s], 1.0 / Rz_t, rtol=1e-9, atol=1e-12)


def test_pixel_r_recovery_on_self():
    """
    Eigen-fit on a Pixel-R-generated G'_M: true θ has ~0 residual;
    optimizer from a wrong start reaches a low spectral residual.
    """
    n = 4
    n_cells = n * n
    cells = [
        GridCell(
            ix=i % n,
            iy=i // n,
            node=f"s{i}",
            x=float(i % n),
            y=float(i // n),
            current=0.0,
        )
        for i in range(n_cells)
    ]
    pad_nodes = [f"p{k}" for k in range(n_cells)]
    ports = PortSet(
        pad_nodes=pad_nodes,
        pad_voltages=[1.0] * n_cells,
        cells=cells,
        cell_size=1.0,
        nx=n,
        ny=n,
        bbox=(0.0, 0.0, 4.0, 4.0),
    )
    model = build_pixel_model(ports)
    assert model.pad_attach == list(range(n_cells))
    assert len(model.arms) > 0

    Rx_t, Ry_t, Rz_t = 2.0, 3.0, 5.0
    Gtrue = build_Gs(model, Rx_t, Ry_t, Rz_t)
    Gtrue = Gtrue + np.eye(Gtrue.shape[0]) * 1e-9

    truth = fit_pixel_r(Gtrue, model, x0=(Rx_t, Ry_t, Rz_t))
    assert truth.relative_spectral_error < 1e-6

    fit = fit_pixel_r(Gtrue, model, x0=(1.0, 1.0, 1.0))
    assert fit.Rx > 0 and fit.Ry > 0 and fit.Rz > 0
    assert fit.relative_spectral_error < 0.05
    assert np.isclose(fit.Rx, Rx_t, rtol=0.05)
    assert np.isclose(fit.Ry, Ry_t, rtol=0.05)
    assert np.isclose(fit.Rz, Rz_t, rtol=0.05)


def test_pixel_r_ir_recovery_on_self():
    """
    IR-fit on a Pixel-R-generated G'_M: true θ has ~0 residual;
    optimizer from a wrong start recovers Rx, Ry, Rz.
    """
    n = 4
    n_cells = n * n
    cells = [
        GridCell(
            ix=i % n,
            iy=i // n,
            node=f"s{i}",
            x=float(i % n),
            y=float(i // n),
            current=0.01 * (1 + (i % 3)),
        )
        for i in range(n_cells)
    ]
    ports = PortSet(
        pad_nodes=[f"p{k}" for k in range(n_cells)],
        pad_voltages=[1.0] * n_cells,
        cells=cells,
        cell_size=1.0,
        nx=n,
        ny=n,
        bbox=(0.0, 0.0, 4.0, 4.0),
    )
    model = build_pixel_model(ports)

    Rx_t, Ry_t, Rz_t = 2.0, 3.0, 5.0
    Gtrue = build_Gs(model, Rx_t, Ry_t, Rz_t)
    Gtrue = Gtrue + np.eye(Gtrue.shape[0]) * 1e-9

    truth = fit_pixel_r_ir(Gtrue, model, ports, x0=(Rx_t, Ry_t, Rz_t))
    assert truth.relative_ir_error < 1e-6

    fit = fit_pixel_r_ir(Gtrue, model, ports, x0=(1.0, 1.0, 1.0))
    assert fit.Rx > 0 and fit.Ry > 0 and fit.Rz > 0
    assert fit.relative_ir_error < 0.05
    assert np.isclose(fit.Rx, Rx_t, rtol=0.05)
    assert np.isclose(fit.Ry, Ry_t, rtol=0.05)
    assert np.isclose(fit.Rz, Rz_t, rtol=0.05)


def test_grounded_eigh_symmetric():
    A = np.array([[2.0, -1.0], [-1.0, 2.0]])
    w, Q = grounded_eigh(A, shift=0.0)
    assert np.allclose(Q @ np.diag(w) @ Q.T, A, atol=1e-10)


def test_clip_stamp_resistance_window():
    from graph import R_STAMP_MAX, R_STAMP_MIN, clip_stamp_resistance

    assert clip_stamp_resistance(1.0) == 1.0
    assert clip_stamp_resistance(1e-15) == R_STAMP_MIN
    assert clip_stamp_resistance(3.307354305293e14) == R_STAMP_MAX
    assert clip_stamp_resistance(0.0) is None
    assert clip_stamp_resistance(-4.0) is None
    assert clip_stamp_resistance(float("inf")) is None
    assert clip_stamp_resistance(float("nan")) is None


def test_assemble_clips_extreme_resistors():
    """TSMC-style 1e-15 shorts / 1e14 dummy opens must not explode G."""
    from graph import R_STAMP_MAX, R_STAMP_MIN, assemble_conductance
    from spice_parser import NodeCoord, SpiceNetlist

    net = SpiceNetlist()
    net.node_names = {"a", "b", "c"}
    net.coords = {
        "a": NodeCoord(1, 0.0, 0.0),
        "b": NodeCoord(1, 1.0, 0.0),
        "c": NodeCoord(1, 2.0, 0.0),
    }
    net.resistors = [
        ("a", "b", 1.0e-15),
        ("b", "c", 3.307354305293e14),
        ("a", "c", 2.0),
    ]
    system = assemble_conductance(net)
    G = system.G.toarray()
    ia, ib, ic = system.index["a"], system.index["b"], system.index["c"]
    g_lo = 1.0 / R_STAMP_MIN
    g_hi = 1.0 / R_STAMP_MAX
    assert np.isclose(-G[ia, ib], g_lo, rtol=1e-12)
    assert np.isclose(-G[ib, ic], g_hi, rtol=1e-12)
    assert np.isclose(-G[ia, ic], 0.5, rtol=1e-12)
    assert system.r_clip["n_clipped_lo"] == 1
    assert system.r_clip["n_clipped_hi"] == 1
    assert system.r_clip["n_skipped"] == 0
    assert system.r_clip["r_raw_min"] == 1.0e-15
    assert system.r_clip["r_raw_max"] == 3.307354305293e14
