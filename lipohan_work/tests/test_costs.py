"""Synthetic recovery: when G'_M is itself a Pixel-R star, every cost should land near truth."""

from __future__ import annotations

import numpy as np

from _bootstrap import ensure_paths

ensure_paths()

from costs import FitContext, PRIORITY_COSTS, STABILIZED_COSTS  # noqa: E402
from evaluate import evaluate  # noqa: E402
from fit_costs import fit_cost  # noqa: E402
from run_cost_sweep import _synthetic_bundle  # noqa: E402
from synthetic_cases import (  # noqa: E402
    build_cases,
    build_boundary_ensemble,
    build_current_ensemble,
    evaluate_boundary_ensemble,
    evaluate_current_ensemble,
    perturb_laplacian_conductances,
)


def test_priority_costs_recover_true_star():
    ports, model, Gprime = _synthetic_bundle()
    ctx = FitContext(Gprime=Gprime, model=model, ports=ports, seed=0, rq_basis="mixed")
    true = np.array([2.0, 4.0, 5.0])
    for cost in PRIORITY_COSTS:
        fit = fit_cost(ctx, cost, max_nfev=200)
        got = np.array([fit.Rx, fit.Ry, fit.Rz])
        rel = np.max(np.abs(np.log10(got) - np.log10(true)))
        assert rel < 0.05, f"{cost} recovered {got} (log10 rel {rel})"
        ev = evaluate(ctx, fit.Gs)
        assert ev["e_ir_original"] < 1e-3, cost
        assert ev["J_inf"] < 0.05, cost


def test_frobenius_zero_at_truth():
    ports, model, Gprime = _synthetic_bundle()
    ctx = FitContext(Gprime=Gprime, model=model, ports=ports, seed=0)
    from pixel_r import build_Gs

    Gs = build_Gs(model, 2.0, 4.0, 5.0)
    ev = evaluate(ctx, Gs)
    assert ev["J_F"] < 1e-18
    assert ev["e_ir_original"] < 1e-12


def test_training_stimulus_override():
    ports, model, Gprime = _synthetic_bundle()
    original = ports.lumped_sink_currents()
    ctx = FitContext(
        Gprime=Gprime,
        model=model,
        ports=ports,
        training_stimuli=[("original_only", original)],
    )
    assert ctx.stimulus_names == ["original_only"]
    assert len(ctx.I_sinks) == 1
    assert ctx.target_v.shape == (ports.n_sinks,)


def test_synthetic_cases_are_symmetric_laplacians():
    cases = build_cases(n=5, seed=0)
    assert [case.name for case in cases] == [
        "exact_global",
        "smooth_local",
        "random_local",
        "long_range",
        "combined",
    ]
    exact = cases[0].Gprime
    for case in cases:
        assert np.allclose(case.Gprime, case.Gprime.T)
        assert np.allclose(np.sum(case.Gprime, axis=1), 0.0, atol=1e-10)
        assert case.Gprime.shape == exact.shape
    assert not np.allclose(cases[-1].Gprime, exact)


def test_heldout_evaluation_is_exact_for_identical_model():
    case = build_cases(n=4, seed=0)[0]
    stimuli = build_current_ensemble(case.ports, seed=71, n_physical=2, n_hotspot=2)
    result = evaluate_current_ensemble(
        case.Gprime, case.Gprime, case.ports, stimuli
    )
    assert result["n_stimuli"] == len(stimuli)
    assert result["e_ir_worst"] < 1e-12
    assert result["within_20pct"] == 1.0
    boundaries = build_boundary_ensemble(case.ports, seed=81)
    boundary_result = evaluate_boundary_ensemble(
        case.Gprime, case.Gprime, case.ports, boundaries
    )
    assert boundary_result["n_stimuli"] == 12
    assert boundary_result["e_ir_worst"] < 1e-12
    assert boundary_result["within_20pct"] == 1.0


def test_stabilized_relative_costs_recover_true_star():
    ports, model, Gprime = _synthetic_bundle()
    ctx = FitContext(Gprime=Gprime, model=model, ports=ports, seed=0)
    true = np.array([2.0, 4.0, 5.0])
    for cost in STABILIZED_COSTS:
        fit = fit_cost(ctx, cost, max_nfev=250)
        got = np.array([fit.Rx, fit.Ry, fit.Rz])
        rel = np.max(np.abs(np.log10(got) - np.log10(true)))
        assert rel < 0.05, f"{cost} recovered {got} (log10 rel {rel})"
        assert evaluate(ctx, fit.Gs)["e_ir_original"] < 1e-3, cost


def test_conductance_noise_preserves_laplacian():
    case = build_cases(n=4, seed=0)[0]
    noisy = perturb_laplacian_conductances(case.Gprime, sigma=0.1, seed=17)
    assert np.allclose(noisy, noisy.T)
    assert np.allclose(np.sum(noisy, axis=1), 0.0, atol=1e-10)
    assert not np.allclose(noisy, case.Gprime)


if __name__ == "__main__":
    test_frobenius_zero_at_truth()
    print("test_frobenius_zero_at_truth ok")
    test_training_stimulus_override()
    print("test_training_stimulus_override ok")
    test_synthetic_cases_are_symmetric_laplacians()
    print("test_synthetic_cases_are_symmetric_laplacians ok")
    test_heldout_evaluation_is_exact_for_identical_model()
    print("test_heldout_evaluation_is_exact_for_identical_model ok")
    test_stabilized_relative_costs_recover_true_star()
    print("test_stabilized_relative_costs_recover_true_star ok")
    test_conductance_noise_preserves_laplacian()
    print("test_conductance_noise_preserves_laplacian ok")
    test_priority_costs_recover_true_star()
    print("test_priority_costs_recover_true_star ok")
