"""Grid-cell same-layer resistor perturbation tests."""

from __future__ import annotations

import pytest

from spice_parser import NodeCoord, SpiceNetlist, perturb_layer_resistors


def _toy_net() -> SpiceNetlist:
    """
    Two cells of size 10:
      cell (0,0): n1@0 and n1@8 (same cell), n3@0 (same cell)
      cell (1,0): n1@15
    Plus a via at (0,0).
    """
    net = SpiceNetlist()
    for name, layer, x, y in [
        ("n1_0_0", 1, 0.0, 0.0),
        ("n1_8_0", 1, 8.0, 0.0),
        ("n1_15_0", 1, 15.0, 0.0),
        ("n1_20_0", 1, 20.0, 0.0),
        ("n3_0_0", 3, 0.0, 0.0),
        ("n3_8_0", 3, 8.0, 0.0),
    ]:
        net.coords[name] = NodeCoord(layer, x, y)
        net.node_names.add(name)
    net.resistors = [
        ("n1_0_0", "n1_8_0", 1.0),  # cell (0,0), L1
        ("n1_15_0", "n1_20_0", 2.0),  # cell (1,0), L1
        ("n3_0_0", "n3_8_0", 3.0),  # cell (0,0), L3
        ("n1_0_0", "n3_0_0", 0.01),  # via
    ]
    return net


def test_same_cell_shares_factor_different_cells_differ():
    net = _toy_net()
    before = [r[2] for r in net.resistors]
    meta = perturb_layer_resistors(
        net, layer=1, amp=0.5, seed=7, cell_size=10.0, layers=[1, 3]
    )
    assert meta["perturb_unit"] == "grid"
    assert meta["perturb_layer"] == 1
    assert meta["n_perturbed"] == 2
    assert meta["n_cells_used"] == 2
    # Via and L3 unchanged.
    assert net.resistors[2][2] == before[2]
    assert net.resistors[3][2] == before[3]
    fac0 = net.resistors[0][2] / before[0]
    fac1 = net.resistors[1][2] / before[1]
    assert 0.5 - 1e-12 <= fac0 <= 1.5 + 1e-12
    assert 0.5 - 1e-12 <= fac1 <= 1.5 + 1e-12
    assert fac0 != fac1


def test_all_layers_same_cell_share_factor():
    net = _toy_net()
    before = [r[2] for r in net.resistors]
    meta = perturb_layer_resistors(
        net, layer="all", amp=0.4, seed=3, cell_size=10.0, layers=[1, 3]
    )
    assert meta["perturb_layer"] == "all"
    assert meta["n_perturbed"] == 3
    assert sorted(meta["perturb_layers"]) == [1, 3]
    # L1 and L3 in cell (0,0) share the same factor.
    fac_l1 = net.resistors[0][2] / before[0]
    fac_l3 = net.resistors[2][2] / before[2]
    assert abs(fac_l1 - fac_l3) < 1e-12
    # Via unchanged.
    assert net.resistors[3][2] == before[3]
    # Other cell L1 may differ.
    fac_other = net.resistors[1][2] / before[1]
    assert fac_other != fac_l1


def test_via_untouched_and_reproducible():
    a = _toy_net()
    b = _toy_net()
    perturb_layer_resistors(a, 1, 0.3, seed=42, cell_size=10.0, layers=[1, 3])
    perturb_layer_resistors(b, 1, 0.3, seed=42, cell_size=10.0, layers=[1, 3])
    assert [r[2] for r in a.resistors] == [r[2] for r in b.resistors]
    # Via still 0.01
    assert a.resistors[3][2] == 0.01


def test_roots_restrict_to_rail():
    net = _toy_net()
    before = list(net.resistors)
    # Only allow the L1 cell-(0,0) nodes as roots → second L1 R untouched.
    roots = {"n1_0_0", "n1_8_0"}
    meta = perturb_layer_resistors(
        net,
        layer=1,
        amp=0.5,
        seed=1,
        cell_size=10.0,
        layers=[1, 3],
        roots=roots,
    )
    assert meta["n_perturbed"] == 1
    assert net.resistors[0][2] != before[0][2]
    assert net.resistors[1][2] == before[1][2]


def test_missing_layer_errors():
    net = _toy_net()
    with pytest.raises(ValueError, match="no same-layer"):
        perturb_layer_resistors(
            net, layer=9, amp=0.1, seed=0, cell_size=10.0, layers=[1, 3, 9]
        )


def test_all_requires_layers():
    net = _toy_net()
    with pytest.raises(ValueError, match="requires a non-empty layers"):
        perturb_layer_resistors(net, layer="all", amp=0.1, seed=0, cell_size=10.0)
