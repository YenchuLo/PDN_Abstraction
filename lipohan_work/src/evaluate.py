"""Evaluate every named metric for a fitted G_S, independent of the training cost."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from _bootstrap import ensure_paths

ensure_paths()

from correlate import e_ir  # noqa: E402
from costs import EPS, FitContext, _rayleigh, _reff_matrix, residual, scalar_loss  # noqa: E402
from fit_eigen import proj_eigs, relative_spectral_error  # noqa: E402


def evaluate(ctx: FitContext, Gs: np.ndarray) -> Dict[str, Any]:
    Gm = ctx.Gprime
    lam_S = proj_eigs(ctx.Q, Gs)
    drops_s = ctx.ir_drops(Gs)
    e_by: Dict[str, float] = {}
    rel_max_by: Dict[str, float] = {}
    within20_by: Dict[str, float] = {}
    for name, d_m, d_s in zip(ctx.stimulus_names, ctx.drops_m, drops_s):
        e_by[name] = float(e_ir(d_s, d_m))
        rel = (d_s - d_m) / (np.abs(d_m) + EPS)
        rel_max_by[name] = float(np.max(np.abs(rel)))
        within20_by[name] = float(np.mean(np.abs(rel) <= 0.20))

    concat_m = ctx.target_v
    concat_s = np.concatenate(drops_s)
    rel_all = (concat_s - concat_m) / (np.abs(concat_m) + EPS)
    vdd = float(ctx.vref) if abs(ctx.vref) > EPS else 1.0
    ir_pct_m = 100.0 * concat_m / vdd
    ir_pct_s = 100.0 * concat_s / vdd

    J_F = float(np.sum(np.square(Gs - Gm)))
    J_F_rel = J_F / (float(np.sum(np.square(Gm))) + EPS)
    R_S = _reff_matrix(Gs)
    i, j = ctx.tri
    J_R = float(np.sum(np.square((R_S - ctx.reff_M)[i, j])))

    return {
        "J_lambda": float(np.sum(np.square(ctx.lam_M - lam_S))),
        "J_lambda_rel": float(np.sum(np.square((lam_S - ctx.lam_M) / (np.abs(ctx.lam_M) + EPS)))),
        "relative_spectral_error": float(relative_spectral_error(ctx.lam_M, lam_S)),
        "J_F": J_F,
        "J_F_rel": J_F_rel,
        "J_W": float(np.sum(np.square(ctx.W * (Gs - Gm)))),
        "J_RQ": float(np.sum(np.square(_rayleigh(Gs, ctx.X_rq) - ctx.rq_target))),
        "J_RQ_rel": float(
            np.sum(
                np.square(_rayleigh(Gs - Gm, ctx.X_rq) / (ctx.rq_target + EPS))
            )
        ),
        "J_V": float(np.sum(np.square(concat_s - concat_m))),
        "J_V_rel": float(np.sum(np.square(rel_all))),
        "J_inf": float(np.max(np.abs(rel_all))),
        "J_p4": float(np.mean(np.abs(concat_s - concat_m) ** 4) ** 0.25),
        "J_p8": float(np.mean(np.abs(concat_s - concat_m) ** 8) ** 0.125),
        "J_R": J_R,
        "J_2": float(np.linalg.norm(Gs - Gm, ord=2)),
        "e_ir_by_stimulus": e_by,
        "e_ir_original": e_by.get("original", float("nan")),
        "e_worst": float(max(e_by.values()) if e_by else float("nan")),
        "rel_max_by_stimulus": rel_max_by,
        "frac_within_20pct_by_stimulus": within20_by,
        "frac_within_20pct_original": within20_by.get("original", float("nan")),
        "frac_within_20pct_all": float(np.mean(np.abs(rel_all) <= 0.20)),
        "ir_pct_original_ref": (100.0 * ctx.drops_m[0] / vdd).tolist()
        if ctx.drops_m
        else [],
        "ir_pct_original_pred": (100.0 * drops_s[0] / vdd).tolist() if drops_s else [],
        "n_rq_vectors": int(ctx.X_rq.shape[1]),
        "n_ports": int(Gm.shape[0]),
        "n_pads": int(ctx.ports.n_pads),
        "n_sinks": int(ctx.ports.n_sinks),
        "vdd": vdd,
        "ir_pct_concat_ref_mean": float(np.mean(ir_pct_m)),
        "ir_pct_concat_pred_mean": float(np.mean(ir_pct_s)),
        "minimax_loss": float(scalar_loss(ctx, "minimax", Gs)),
        "eigen_residual_vec_norm2": float(np.sum(np.square(residual(ctx, "eigen", Gs)))),
    }


def metrics_row(cost: str, Rx: float, Ry: float, Rz: float, ev: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "cost": cost,
        "Rx": float(Rx),
        "Ry": float(Ry),
        "Rz": float(Rz),
        "J_lambda": ev["J_lambda"],
        "J_F": ev["J_F"],
        "J_F_rel": ev["J_F_rel"],
        "J_RQ": ev["J_RQ"],
        "J_V": ev["J_V"],
        "J_V_rel": ev["J_V_rel"],
        "J_inf": ev["J_inf"],
        "J_2": ev["J_2"],
        "J_R": ev["J_R"],
        "e_ir_original": ev["e_ir_original"],
        "e_worst": ev["e_worst"],
        "frac_within_20pct_original": ev["frac_within_20pct_original"],
        "relative_spectral_error": ev["relative_spectral_error"],
    }
    for name, val in ev["e_ir_by_stimulus"].items():
        row[f"e_ir_{name}"] = val
    return row
