#!/usr/bin/env python3
"""Run a reproducible synthetic study of Pixel-R fitting objectives."""

from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from _bootstrap import ensure_paths

ensure_paths()

from correlate import build_stimuli  # noqa: E402
from costs import FitContext, parse_cost_list  # noqa: E402
from evaluate import evaluate  # noqa: E402
from fit_costs import fit_cost  # noqa: E402
from synthetic_cases import (  # noqa: E402
    SyntheticCase,
    build_boundary_ensemble,
    build_cases,
    build_current_ensemble,
    evaluate_boundary_ensemble,
    evaluate_current_ensemble,
    perturb_laplacian_conductances,
)

HERE = Path(__file__).resolve().parents[1]
DEFAULT_OUT = HERE / "outputs" / "synthetic_study"


def _fit_record(
    case: SyntheticCase,
    *,
    study: str,
    variant: str,
    cost: str,
    context: FitContext,
    heldout,
    boundary,
    max_nfev: int,
    x0: Tuple[float, float, float] | None = None,
    warm_start: bool = True,
    evaluation_gprime: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Fit once and return a flat summary plus nested diagnostic details."""
    label = f"{case.name}/{study}/{variant}/{cost}"
    print(f"[fit] {label}", flush=True)
    try:
        fit = fit_cost(
            context,
            cost,
            x0=x0,
            max_nfev=max_nfev,
            warm_start=warm_start,
        )
        train = evaluate(context, fit.Gs)
        reference_gprime = (
            case.Gprime if evaluation_gprime is None else evaluation_gprime
        )
        test = evaluate_current_ensemble(
            reference_gprime, fit.Gs, case.ports, heldout
        )
        boundary_test = evaluate_boundary_ensemble(
            reference_gprime, fit.Gs, case.ports, boundary
        )
        true = case.truth_params
        parameter_log_error = float(
            np.max(
                np.abs(
                    np.log10([fit.Rx, fit.Ry, fit.Rz])
                    - np.log10([true["Rx"], true["Ry"], true["Rz"]])
                )
            )
        )
        record: Dict[str, Any] = {
            "scenario": case.name,
            "study": study,
            "variant": variant,
            "cost": fit.cost,
            "Rx": fit.Rx,
            "Ry": fit.Ry,
            "Rz": fit.Rz,
            "parameter_log10_max_error": parameter_log_error,
            "training_loss": fit.training_loss,
            "fit_success": fit.success,
            "fit_message": fit.message,
            "nfev": fit.nfev,
            "elapsed_s": fit.elapsed_s,
            "warm_start_from": fit.warm_start_from,
            "n_training_stimuli": len(context.I_sinks),
            "training_stimuli": list(context.stimulus_names),
            "train_e_ir_original": train["e_ir_original"],
            "train_e_worst": train["e_worst"],
            "train_J_inf": train["J_inf"],
            "train_J_inf_floor": train["J_inf_floor"],
            "train_J_lambda": train["J_lambda"],
            "train_J_F": train["J_F"],
            "train_J_RQ": train["J_RQ"],
            "train_J_V": train["J_V"],
            "train_relative_spectral_error": train["relative_spectral_error"],
            "heldout_e_ir_mean": test["e_ir_mean"],
            "heldout_e_ir_median": test["e_ir_median"],
            "heldout_e_ir_p95": test["e_ir_p95"],
            "heldout_e_ir_worst": test["e_ir_worst"],
            "heldout_stable_rel_p95": test["stable_rel_p95"],
            "heldout_stable_rel_max": test["stable_rel_max"],
            "heldout_within_20pct": test["within_20pct"],
            "heldout_family_e_ir_mean": test["family_e_ir_mean"],
            "heldout_by_stimulus": test["by_stimulus"],
            "boundary_e_ir_mean": boundary_test["e_ir_mean"],
            "boundary_e_ir_p95": boundary_test["e_ir_p95"],
            "boundary_e_ir_worst": boundary_test["e_ir_worst"],
            "boundary_stable_rel_p95": boundary_test["stable_rel_p95"],
            "boundary_within_20pct": boundary_test["within_20pct"],
            "boundary_by_stimulus": boundary_test["by_stimulus"],
        }
        print(
            "  "
            f"success={fit.success} nfev={fit.nfev} "
            f"train={record['train_e_worst']:.4g} "
            f"heldout_mean={record['heldout_e_ir_mean']:.4g} "
            f"heldout_worst={record['heldout_e_ir_worst']:.4g}",
            flush=True,
        )
        return record
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
        return {
            "scenario": case.name,
            "study": study,
            "variant": variant,
            "cost": cost,
            "fit_success": False,
            "fit_message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _main_cost_sweep(
    cases: Sequence[SyntheticCase],
    costs: Sequence[str],
    *,
    seed: int,
    heldout,
    boundary,
    max_nfev: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for case in cases:
        context = FitContext(
            Gprime=case.Gprime,
            model=case.model,
            ports=case.ports,
            seed=seed,
            rq_basis="mixed",
        )
        for cost in costs:
            records.append(
                _fit_record(
                    case,
                    study="cost_sweep",
                    variant="default",
                    cost=cost,
                    context=context,
                    heldout=heldout,
                    boundary=boundary,
                    max_nfev=max_nfev,
                )
            )
    return records


def _ablation_studies(
    case: SyntheticCase,
    *,
    seed: int,
    heldout,
    boundary,
    max_nfev: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    standard = build_stimuli(case.ports, seed=seed)
    diverse = build_current_ensemble(
        case.ports,
        seed=seed + 303,
        n_physical=4,
        n_hotspot=4,
        n_signed=2,
    )
    current_sets = {
        "original_only": standard[:1],
        "original_random": standard[:2],
        "standard_four": standard,
        "diverse_fifteen": standard[:1] + diverse,
    }
    for variant, stimuli in current_sets.items():
        context = FitContext(
            case.Gprime,
            case.model,
            case.ports,
            seed=seed,
            training_stimuli=stimuli,
        )
        records.append(
            _fit_record(
                case,
                study="current_training",
                variant=variant,
                cost="voltage",
                context=context,
                heldout=heldout,
                boundary=boundary,
                max_nfev=max_nfev,
            )
        )

    for basis in ("eigen", "random", "current", "mixed"):
        context = FitContext(
            case.Gprime,
            case.model,
            case.ports,
            seed=seed,
            rq_basis=basis,
            n_rq_eigen=case.Gprime.shape[0],
            n_rq_rand=32,
        )
        records.append(
            _fit_record(
                case,
                study="rayleigh_basis",
                variant=basis,
                cost="rayleigh",
                context=context,
                heldout=heldout,
                boundary=boundary,
                max_nfev=max_nfev,
            )
        )

    for p_norm in (2.0, 4.0, 8.0, 16.0):
        context = FitContext(
            case.Gprime,
            case.model,
            case.ports,
            seed=seed,
            p_norm=p_norm,
        )
        records.append(
            _fit_record(
                case,
                study="pnorm",
                variant=f"p={p_norm:g}",
                cost="pnorm",
                context=context,
                heldout=heldout,
                boundary=boundary,
                max_nfev=max_nfev,
            )
        )

    hybrid_weights = {
        "pure_eigen": (1.0, 0.0, 0.0),
        "pure_rayleigh": (0.0, 1.0, 0.0),
        "pure_voltage": (0.0, 0.0, 1.0),
        "equal": (1.0, 1.0, 1.0),
        "voltage_x10": (1.0, 1.0, 10.0),
    }
    for variant, weights in hybrid_weights.items():
        context = FitContext(
            case.Gprime,
            case.model,
            case.ports,
            seed=seed,
            hybrid_alpha=weights[0],
            hybrid_beta=weights[1],
            hybrid_gamma=weights[2],
        )
        records.append(
            _fit_record(
                case,
                study="hybrid_weights",
                variant=variant,
                cost="hybrid",
                context=context,
                heldout=heldout,
                boundary=boundary,
                max_nfev=max_nfev,
            )
        )

    matrix_weights = {
        "uniform": (1.0, 1.0, 1.0, 1.0),
        "default_local": (5.0, 3.0, 3.0, 0.05),
        "strict_stencil": (5.0, 3.0, 3.0, 0.0),
    }
    for variant, weights in matrix_weights.items():
        context = FitContext(
            case.Gprime,
            case.model,
            case.ports,
            seed=seed,
            w_diag=weights[0],
            w_nbr=weights[1],
            w_via=weights[2],
            w_far=weights[3],
        )
        records.append(
            _fit_record(
                case,
                study="matrix_weights",
                variant=variant,
                cost="weighted",
                context=context,
                heldout=heldout,
                boundary=boundary,
                max_nfev=max_nfev,
            )
        )

    for fit_seed in range(seed, seed + 5):
        for cost in ("voltage", "rayleigh", "hybrid"):
            context = FitContext(
                case.Gprime,
                case.model,
                case.ports,
                seed=fit_seed,
            )
            records.append(
                _fit_record(
                    case,
                    study="seed_stability",
                    variant=f"seed={fit_seed}",
                    cost=cost,
                    context=context,
                    heldout=heldout,
                    boundary=boundary,
                    max_nfev=max_nfev,
                )
            )

    context = FitContext(case.Gprime, case.model, case.ports, seed=seed)
    minimax_starts: Dict[str, Tuple[float, float, float]] = {
        "nominal_scale": (context.r_scale, context.r_scale, context.r_scale)
    }
    for start_cost in ("eigen", "frobenius", "rayleigh", "voltage", "weighted"):
        start_fit = fit_cost(context, start_cost, max_nfev=max_nfev)
        minimax_starts[start_cost] = (start_fit.Rx, start_fit.Ry, start_fit.Rz)
    for variant, x0 in minimax_starts.items():
        records.append(
            _fit_record(
                case,
                study="minimax_multistart",
                variant=variant,
                cost="minimax_floor",
                context=context,
                heldout=heldout,
                boundary=boundary,
                max_nfev=max(500, max_nfev),
                x0=x0,
                warm_start=False,
            )
        )
    return records


def _build_heldout(ports, *, seed: int, repeats: int):
    patterns = []
    for offset in range(repeats):
        current_seed = seed + offset
        for name, current in build_current_ensemble(ports, seed=current_seed):
            patterns.append((f"seed{current_seed}_{name}", current))
    return patterns


def _grid_size_study(
    *,
    sizes: Sequence[int],
    seed: int,
    heldout_seed: int,
    heldout_repeats: int,
    max_nfev: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    representative_costs = (
        "eigen",
        "frobenius",
        "weighted",
        "rayleigh_rel_floor",
        "voltage",
        "hybrid",
    )
    for size in sizes:
        base = build_cases(size, seed)[-1]
        case = SyntheticCase(
            name=f"combined_n{size}",
            description=f"Combined mismatch on a {size}x{size} sink grid.",
            ports=base.ports,
            model=base.model,
            Gprime=base.Gprime,
            truth_params=base.truth_params,
        )
        heldout = _build_heldout(
            case.ports,
            seed=heldout_seed,
            repeats=heldout_repeats,
        )
        boundary = build_boundary_ensemble(
            case.ports,
            seed=heldout_seed + 1000,
        )
        context = FitContext(case.Gprime, case.model, case.ports, seed=seed)
        for cost in representative_costs:
            records.append(
                _fit_record(
                    case,
                    study="grid_size",
                    variant=f"n={size}",
                    cost=cost,
                    context=context,
                    heldout=heldout,
                    boundary=boundary,
                    max_nfev=max_nfev,
                )
            )
    return records


def _noise_study(
    *,
    seed: int,
    heldout_seed: int,
    heldout_repeats: int,
    max_nfev: int,
) -> List[Dict[str, Any]]:
    clean = build_cases(5, seed)[0]
    heldout = _build_heldout(
        clean.ports,
        seed=heldout_seed,
        repeats=heldout_repeats,
    )
    boundary = build_boundary_ensemble(clean.ports, seed=heldout_seed + 1000)
    representative_costs = (
        "eigen",
        "frobenius",
        "weighted",
        "rayleigh_rel_floor",
        "voltage",
        "hybrid",
    )
    records: List[Dict[str, Any]] = []
    for sigma in (0.01, 0.05, 0.10, 0.25):
        noisy_gprime = perturb_laplacian_conductances(
            clean.Gprime,
            sigma=sigma,
            seed=seed + 1701,
        )
        noisy = SyntheticCase(
            name=f"noise_{sigma:.2f}",
            description=f"Exact global truth observed with sigma={sigma:.2f} edge noise.",
            ports=clean.ports,
            model=clean.model,
            Gprime=noisy_gprime,
            truth_params=clean.truth_params,
        )
        context = FitContext(noisy.Gprime, noisy.model, noisy.ports, seed=seed)
        for cost in representative_costs:
            record = _fit_record(
                noisy,
                study="conductance_noise",
                variant=f"sigma={sigma:.2f}",
                cost=cost,
                context=context,
                heldout=heldout,
                boundary=boundary,
                max_nfev=max_nfev,
                evaluation_gprime=clean.Gprime,
            )
            record["noise_sigma"] = sigma
            records.append(record)
    return records


def _serializable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    return value


def _write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    preferred = [
        "scenario",
        "study",
        "variant",
        "cost",
        "fit_success",
        "Rx",
        "Ry",
        "Rz",
        "train_e_ir_original",
        "train_e_worst",
        "heldout_e_ir_mean",
        "heldout_e_ir_p95",
        "heldout_e_ir_worst",
        "heldout_stable_rel_p95",
        "heldout_within_20pct",
        "boundary_e_ir_mean",
        "boundary_e_ir_p95",
        "boundary_e_ir_worst",
        "boundary_within_20pct",
        "nfev",
        "elapsed_s",
    ]
    keys = sorted({key for record in records for key in record})
    columns = preferred + [key for key in keys if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = {}
            for key in columns:
                value = record.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(_serializable(value), separators=(",", ":"))
                row[key] = value
            writer.writerow(row)


def _successful(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("fit_success") and "heldout_e_ir_mean" in record
    ]


def _plot_results(records: Sequence[Dict[str, Any]], out: Path) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: List[str] = []
    main = _successful(
        record for record in records if record.get("study") == "cost_sweep"
    )
    scenarios = list(dict.fromkeys(str(record["scenario"]) for record in main))
    costs = list(dict.fromkeys(str(record["cost"]) for record in main))
    lookup = {
        (str(record["scenario"]), str(record["cost"])): float(
            record["heldout_e_ir_mean"]
        )
        for record in main
    }
    matrix = np.full((len(scenarios), len(costs)), np.nan)
    for row, scenario in enumerate(scenarios):
        for col, cost in enumerate(costs):
            matrix[row, col] = lookup.get((scenario, cost), np.nan)
    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(costs)), 5.5))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(costs)), costs, rotation=35, ha="right")
    ax.set_yticks(range(len(scenarios)), scenarios)
    ax.set_title("Held-out mean relative IR error")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if np.isfinite(matrix[row, col]):
                ax.text(
                    col,
                    row,
                    f"{100.0 * matrix[row, col]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if matrix[row, col] > np.nanmedian(matrix) else "black",
                )
    fig.colorbar(image, ax=ax, label="relative error")
    fig.tight_layout()
    path = out / "heldout_cost_heatmap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    combined = [record for record in main if record["scenario"] == "combined"]
    combined.sort(key=lambda record: float(record["heldout_e_ir_mean"]))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = [str(record["cost"]) for record in combined]
    values = [100.0 * float(record["heldout_e_ir_mean"]) for record in combined]
    ax.bar(np.arange(len(labels)), values, color="#2f6f8f")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    ax.set_ylabel("held-out mean e_IR (%)")
    ax.set_title("Combined mismatch: all fitting objectives")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / "combined_cost_ranking.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    ablation = _successful(
        record for record in records if record.get("study") != "cost_sweep"
    )
    studies = list(dict.fromkeys(str(record["study"]) for record in ablation))
    for study in studies:
        rows = [record for record in ablation if record["study"] == study]
        rows.sort(key=lambda record: float(record["heldout_e_ir_mean"]), reverse=True)
        labels = [f"{record['variant']} / {record['cost']}" for record in rows]
        values = [100.0 * float(record["heldout_e_ir_mean"]) for record in rows]
        positions = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(10, max(3.5, 0.38 * len(rows))))
        bars = ax.barh(positions, values, color="#6b8e23")
        ax.set_yticks(positions, labels)
        ax.set_xlabel("held-out mean e_IR (%)")
        ax.set_title(study.replace("_", " ").title())
        ax.grid(True, axis="x", alpha=0.3)
        right = max(values, default=1.0)
        ax.set_xlim(0.0, max(right * 1.15, 1.0))
        for bar, value in zip(bars, values):
            ax.text(
                value + max(right * 0.01, 0.02),
                bar.get_y() + 0.5 * bar.get_height(),
                f"{value:.2f}%",
                va="center",
                fontsize=8,
            )
        fig.tight_layout()
        path = out / f"ablation_{study}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for scenario in scenarios:
        rows = [record for record in main if record["scenario"] == scenario]
        ax.scatter(
            [100.0 * float(record["train_e_worst"]) for record in rows],
            [100.0 * float(record["heldout_e_ir_mean"]) for record in rows],
            label=scenario,
            alpha=0.8,
        )
    ax.set_xlabel("training worst e_IR (%)")
    ax.set_ylabel("held-out mean e_IR (%)")
    ax.set_title("Training metric versus held-out generalization")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "training_vs_heldout.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))
    return written


def _format_percent(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{100.0 * number:.3f}%"


def _table(records: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "| Cost / variant | Rx | Ry | Rz | Train worst | Held-out mean | Held-out p95 | Held-out worst | Boundary mean | Within 20% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        label = str(record.get("cost", ""))
        if record.get("variant") not in (None, "", "default"):
            label = f"{record['variant']} / {label}"
        if not record.get("fit_success"):
            lines.append(f"| {label} | colspan | | | FAILED | | | | |")
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"{float(record['Rx']):.5g}",
                    f"{float(record['Ry']):.5g}",
                    f"{float(record['Rz']):.5g}",
                    _format_percent(record["train_e_worst"]),
                    _format_percent(record["heldout_e_ir_mean"]),
                    _format_percent(record["heldout_e_ir_p95"]),
                    _format_percent(record["heldout_e_ir_worst"]),
                    _format_percent(record["boundary_e_ir_mean"]),
                    _format_percent(record["heldout_within_20pct"]),
                ]
            )
            + " |"
        )
    return lines


def _write_report(
    path: Path,
    cases: Sequence[SyntheticCase],
    records: Sequence[Dict[str, Any]],
    *,
    costs: Sequence[str],
    seed: int,
    heldout_seed: int,
    heldout_repeats: int,
    n_heldout: int,
    n_boundary: int,
    plots: Sequence[str],
) -> None:
    main = [record for record in records if record.get("study") == "cost_sweep"]
    lines = [
        "# Synthetic Pixel-R Cost Study",
        "",
        "## Method",
        "",
        f"- Grid: {cases[0].ports.nx}x{cases[0].ports.ny}, "
        f"{cases[0].ports.n_pads} sparse pads + {cases[0].ports.n_sinks} sinks.",
        f"- Fit seed: `{seed}`; held-out seeds: `{heldout_seed}` through "
        f"`{heldout_seed + heldout_repeats - 1}`.",
        f"- Main objectives: `{', '.join(costs)}`.",
        "- Every fit uses the same global three-parameter Pixel-R model.",
        "- Default training currents: original, Gaussian random, center-localized, and striped.",
        f"- Held-out set: {n_heldout} independent physical, hotspot, smooth, and signed currents.",
        f"- Boundary set: {n_boundary} tests with nonuniform pad voltages (2% VDD amplitude) and independent currents.",
        "- Primary test metric: mean relative L2 IR error over held-out currents.",
        "- Stable relative-tail metrics use a 2% reference-drop floor to avoid division by near-zero IR.",
        "- Stabilized relative objectives use the same voltage floor; eigen/Rayleigh denominators use a 1e-6 scale floor.",
        "",
        "## Truth scenarios",
        "",
    ]
    for case in cases:
        lines.append(f"- **{case.name}:** {case.description}")

    lines.extend(["", "## Main cost sweep", ""])
    for case in cases:
        rows = [record for record in main if record["scenario"] == case.name]
        good = _successful(rows)
        good.sort(key=lambda record: float(record["heldout_e_ir_mean"]))
        failed = [record for record in rows if not record.get("fit_success")]
        lines.append(f"### {case.name}")
        lines.append("")
        lines.extend(_table(good + failed))
        lines.append("")

    lines.extend(["## Ablation studies on combined mismatch", ""])
    for study in (
        "current_training",
        "rayleigh_basis",
        "pnorm",
        "hybrid_weights",
        "matrix_weights",
        "seed_stability",
        "minimax_multistart",
        "grid_size",
        "conductance_noise",
    ):
        rows = [record for record in records if record.get("study") == study]
        if not rows:
            continue
        lines.append(f"### {study.replace('_', ' ').title()}")
        lines.append("")
        lines.extend(_table(rows))
        lines.append("")

    failures = [record for record in records if not record.get("fit_success")]
    lines.extend(
        [
            "## Execution summary",
            "",
            f"- Fits attempted: {len(records)}",
            f"- Fits marked successful: {len(records) - len(failures)}",
            f"- Failed/non-converged fits: {len(failures)}",
            f"- Plots: {len(plots)}",
        ]
    )
    if failures:
        lines.append("")
        for record in failures:
            lines.append(
                f"- `{record.get('scenario')}/{record.get('study')}/"
                f"{record.get('variant')}/{record.get('cost')}`: "
                f"{record.get('fit_message')}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_study(
    *,
    out: Path,
    grid_size: int,
    costs: Sequence[str],
    seed: int,
    heldout_seed: int,
    heldout_repeats: int,
    max_nfev: int,
    case_names: Sequence[str] | None = None,
    run_ablations: bool = True,
) -> Dict[str, Any]:
    cases = build_cases(grid_size, seed)
    if case_names:
        wanted = set(case_names)
        cases = [case for case in cases if case.name in wanted]
        missing = wanted - {case.name for case in cases}
        if missing:
            raise ValueError(f"unknown case(s): {sorted(missing)}")
    out.mkdir(parents=True, exist_ok=True)
    heldout = _build_heldout(
        cases[0].ports,
        seed=heldout_seed,
        repeats=heldout_repeats,
    )
    boundary = build_boundary_ensemble(
        cases[0].ports,
        seed=heldout_seed + 1000,
    )
    records = _main_cost_sweep(
        cases,
        costs,
        seed=seed,
        heldout=heldout,
        boundary=boundary,
        max_nfev=max_nfev,
    )
    if run_ablations:
        combined = next((case for case in cases if case.name == "combined"), None)
        if combined is not None:
            records.extend(
                _ablation_studies(
                    combined,
                    seed=seed,
                    heldout=heldout,
                    boundary=boundary,
                    max_nfev=max_nfev,
                )
            )
        records.extend(
            _grid_size_study(
                sizes=(3, 5, 7),
                seed=seed,
                heldout_seed=heldout_seed,
                heldout_repeats=heldout_repeats,
                max_nfev=max_nfev,
            )
        )
        records.extend(
            _noise_study(
                seed=seed,
                heldout_seed=heldout_seed,
                heldout_repeats=heldout_repeats,
                max_nfev=max_nfev,
            )
        )

    plots = _plot_results(records, out)
    payload = {
        "method": {
            "grid_size": grid_size,
            "seed": seed,
            "heldout_seed": heldout_seed,
            "heldout_repeats": heldout_repeats,
            "n_heldout": len(heldout),
            "n_boundary": len(boundary),
            "costs": list(costs),
            "max_nfev": max_nfev,
            "training_default": ["original", "random", "localized", "striped"],
        },
        "cases": [
            {"name": case.name, "description": case.description}
            for case in cases
        ],
        "records": records,
        "plots": plots,
    }
    (out / "study_results.json").write_text(
        json.dumps(_serializable(payload), indent=2), encoding="utf-8"
    )
    _write_csv(out / "study_results.csv", records)
    _write_report(
        out / "REPORT.md",
        cases,
        records,
        costs=costs,
        seed=seed,
        heldout_seed=heldout_seed,
        heldout_repeats=heldout_repeats,
        n_heldout=len(heldout),
        n_boundary=len(boundary),
        plots=plots,
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Comprehensive synthetic Pixel-R cost study")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--costs", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heldout-seed", type=int, default=9001)
    parser.add_argument("--heldout-repeats", type=int, default=5)
    parser.add_argument("--max-nfev", type=int, default=200)
    parser.add_argument(
        "--cases",
        default="",
        help="Comma-separated subset of exact_global,smooth_local,random_local,long_range,combined",
    )
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run priority costs on exact_global and combined without ablations",
    )
    args = parser.parse_args(argv)
    costs = parse_cost_list("priority" if args.quick else args.costs)
    case_names = [name.strip() for name in args.cases.split(",") if name.strip()]
    if args.quick and not case_names:
        case_names = ["exact_global", "combined"]
    payload = run_study(
        out=args.out,
        grid_size=args.grid_size,
        costs=costs,
        seed=args.seed,
        heldout_seed=args.heldout_seed,
        heldout_repeats=max(1, args.heldout_repeats),
        max_nfev=args.max_nfev,
        case_names=case_names or None,
        run_ablations=not (args.no_ablations or args.quick),
    )
    failures = [record for record in payload["records"] if not record.get("fit_success")]
    print(
        json.dumps(
            {
                "out": str(args.out),
                "fits": len(payload["records"]),
                "failed": len(failures),
                "report": str(args.out / "REPORT.md"),
            },
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())