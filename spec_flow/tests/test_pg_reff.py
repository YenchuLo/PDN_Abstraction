"""Unreduced-mesh driving-point R_eff with real V sources grounded."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from graph import ConductanceSystem
from pg_reff import compute_mesh_reff, real_vdd_source_nodes


def test_real_vdd_source_nodes_skips_zero_and_gnd():
    node_map = {"a": "a", "b": "b", "x": "a"}
    voltages = [
        ("a", "0", 1.8),
        ("0", "b", 1.8),
        ("x", "0", 1.8),
        ("c", "0", 0.0),
        ("p", "q", 1.8),
    ]
    assert real_vdd_source_nodes(voltages, node_map) == ["a", "b"]


def test_series_two_ohm_driving_point():
    nodes = ["pad", "mid", "sink"]
    index = {n: i for i, n in enumerate(nodes)}
    G = np.array(
        [
            [1.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 1.0],
        ]
    )
    system = ConductanceSystem(
        G=sparse.csr_matrix(G),
        nodes=nodes,
        index=index,
        node_map={n: n for n in nodes},
    )
    r = compute_mesh_reff(system, ["sink", "mid", "pad"], ["pad"], reg=0.0)
    assert r[0] == pytest.approx(2.0, rel=1e-9)
    assert r[1] == pytest.approx(1.0, rel=1e-9)
    assert np.isnan(r[2])


def test_parallel_pads_halve_package():
    nodes = ["p0", "p1", "sink"]
    index = {n: i for i, n in enumerate(nodes)}
    g = 1.0 / 0.25
    G = np.zeros((3, 3))
    for p in (0, 1):
        G[p, p] += g
        G[2, 2] += g
        G[p, 2] -= g
        G[2, p] -= g
    system = ConductanceSystem(
        G=sparse.csr_matrix(G),
        nodes=nodes,
        index=index,
        node_map={n: n for n in nodes},
    )
    r = compute_mesh_reff(system, ["sink"], ["p0", "p1"], reg=0.0)
    assert r[0] == pytest.approx(0.125, rel=1e-9)
