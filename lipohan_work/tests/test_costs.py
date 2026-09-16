"""Synthetic recovery: when G'_M is itself a Pixel-R star, every cost should land near truth."""

from __future__ import annotations

import numpy as np

from _bootstrap import ensure_paths

ensure_paths()

from costs import FitContext, PRIORITY_COSTS  # noqa: E402
from evaluate import evaluate  # noqa: E402
from fit_costs import fit_cost  # noqa: E402
from run_cost_sweep import _synthetic_bundle  # noqa: E402


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


if __name__ == "__main__":
    test_frobenius_zero_at_truth()
    print("test_frobenius_zero_at_truth ok")
    test_priority_costs_recover_true_star()
    print("test_priority_costs_recover_true_star ok")
