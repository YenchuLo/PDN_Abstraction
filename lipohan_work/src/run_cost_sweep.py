#!/usr/bin/env python3
"""Sweep Pixel-R cost functions on a Kron-reduced component (or a synthetic star).

Examples (from repo root):

  # stages 01–04 then the five priority costs on ibmpg2
  python lipohan_work/src/run_cost_sweep.py --spice Benchmarks/IBM/TC2/ibmpg2.spice

  # reuse an existing spec_flow / lipohan_work OUT
  python lipohan_work/src/run_cost_sweep.py --comp lipohan_work/outputs/ibmpg2_nauto/comp1

  # closed-form Pixel-R star (no IBM deck)
  python lipohan_work/src/run_cost_sweep.py --synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from _bootstrap import ensure_paths

ensure_paths()

from costs import FitContext, PRIORITY_COSTS, parse_cost_list  # noqa: E402
from evaluate import evaluate, metrics_row  # noqa: E402
from fit_costs import fit_cost  # noqa: E402
from io_artifacts import (  # noqa: E402
    load_gprime,
    load_pixel_model_fields,
    load_ports,
    save_fit,
    save_metrics,
)
from pixel_r import pixel_model_from_fields  # noqa: E402
from plot_compare import plot_sweep  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SPICE = REPO / "Benchmarks" / "IBM" / "TC2" / "ibmpg2.spice"


def _run_frontend(spice: Path, out: Path, chip_pitch: float) -> Path:
    from stage01_parse import main as s01
    from stage02_ports import main as s02
    from stage03_assemble_kron import main as s03
    from stage04_pixel_model import main as s04

    out.mkdir(parents=True, exist_ok=True)
    rc = s01([str(spice), str(out)])
    if rc:
        raise SystemExit(rc)
    argv2 = [str(out)]
    if chip_pitch > 0:
        argv2.append(str(chip_pitch))
    rc = s02(argv2)
    if rc:
        raise SystemExit(rc)
    comps = sorted(p for p in out.glob("comp*") if p.is_dir())
    if not comps:
        raise FileNotFoundError(f"no compN/ under {out}")
    for comp in comps:
        rc = s03([str(comp)])
        if rc:
            raise SystemExit(rc)
        rc = s04([str(comp)])
        if rc:
            raise SystemExit(rc)
    return comps[0]


def _synthetic_bundle() -> Tuple[Any, Any, Any]:
    """3x3 identical Pixel-R star used as both truth and model class."""
    from pixel_r import build_Gs, build_pixel_model
    from ports import GridCell, PortSet

    n = 3
    vdd = 1.8
    cells = []
    pad_nodes = []
    pad_xy = []
    for iy in range(n):
        for ix in range(n):
            k = iy * n + ix
            cells.append(
                GridCell(
                    ix=ix,
                    iy=iy,
                    node=f"s{k}",
                    x=float(ix),
                    y=float(iy),
                    current=-1.0 if (ix, iy) == (1, 1) else -0.1,
                )
            )
            pad_nodes.append(f"p{k}")
            pad_xy.append((float(ix), float(iy)))
    ports = PortSet(
        pad_nodes=pad_nodes,
        pad_voltages=[vdd] * (n * n),
        cells=cells,
        cell_size=1.0,
        nx=n,
        ny=n,
        bbox=(0.0, 0.0, float(n), float(n)),
        pad_xy=pad_xy,
        pad_attach=list(range(n * n)),
        vdd=vdd,
    )
    model = build_pixel_model(ports)
    Gprime = build_Gs(model, 2.0, 4.0, 5.0)
    return ports, model, Gprime


def _load_comp(comp: Path):
    ports, _ = load_ports(comp)
    Gprime, meta = load_gprime(comp)
    fields = load_pixel_model_fields(comp)
    model = pixel_model_from_fields(ports, fields)
    return ports, model, Gprime, meta


def sweep(
    *,
    ports,
    model,
    Gprime,
    costs: Sequence[str],
    seed: int,
    out_dir: Path,
    rq_basis: str,
    p_norm: float,
    hybrid: Tuple[float, float, float],
    max_nfev: int,
) -> Dict[str, Any]:
    ctx = FitContext(
        Gprime=Gprime,
        model=model,
        ports=ports,
        seed=seed,
        p_norm=p_norm,
        hybrid_alpha=hybrid[0],
        hybrid_beta=hybrid[1],
        hybrid_gamma=hybrid[2],
        rq_basis=rq_basis,
    )
    fits: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for cost in costs:
        print(f"[fit] {cost} …", flush=True)
        fit = fit_cost(ctx, cost, max_nfev=max_nfev)
        ev = evaluate(ctx, fit.Gs)
        row = metrics_row(cost, fit.Rx, fit.Ry, fit.Rz, ev)
        row["ir_pct_original_ref"] = ev["ir_pct_original_ref"]
        row["ir_pct_original_pred"] = ev["ir_pct_original_pred"]
        row["training_loss"] = fit.training_loss
        row["fit_success"] = fit.success
        row["fit_message"] = fit.message
        row["nfev"] = fit.nfev
        row["elapsed_s"] = fit.elapsed_s
        row["warm_start_from"] = fit.warm_start_from
        rows.append(row)
        cost_dir = out_dir / f"fit_{cost}"
        cost_dir.mkdir(parents=True, exist_ok=True)
        save_fit(
            cost_dir,
            {"Rx": fit.Rx, "Ry": fit.Ry, "Rz": fit.Rz},
            ctx.lam_M,
            None,
            ctx.Gprime,
            fit.Gs,
            {
                "fit_method": cost,
                "training_loss": fit.training_loss,
                "fit_success": fit.success,
                "fit_message": fit.message,
                "nfev": fit.nfev,
                "elapsed_s": fit.elapsed_s,
                "warm_start_from": fit.warm_start_from,
                "seed": seed,
            },
        )
        save_metrics(cost_dir, {**row, **{k: v for k, v in ev.items() if k not in row}})
        fits.append(
            {
                "cost": cost,
                "Rx": fit.Rx,
                "Ry": fit.Ry,
                "Rz": fit.Rz,
                "elapsed_s": fit.elapsed_s,
                "success": fit.success,
            }
        )
        print(
            json.dumps(
                {
                    "cost": cost,
                    "Rx": fit.Rx,
                    "Ry": fit.Ry,
                    "Rz": fit.Rz,
                    "e_ir_original": row["e_ir_original"],
                    "J_inf": row["J_inf"],
                    "frac_within_20pct_original": row["frac_within_20pct_original"],
                    "elapsed_s": fit.elapsed_s,
                },
                indent=2,
            ),
            flush=True,
        )

    plots = plot_sweep(rows, out_dir)
    # JSON-friendly rows (drop huge IR vectors from the summary table copy)
    table = []
    for r in rows:
        item = {k: v for k, v in r.items() if k not in ("ir_pct_original_ref", "ir_pct_original_pred")}
        table.append(item)
    summary = {
        "costs": list(costs),
        "seed": seed,
        "rq_basis": rq_basis,
        "p_norm": p_norm,
        "hybrid": {"alpha": hybrid[0], "beta": hybrid[1], "gamma": hybrid[2]},
        "n_ports": int(Gprime.shape[0]),
        "n_pads": int(ports.n_pads),
        "n_sinks": int(ports.n_sinks),
        "fits": fits,
        "table": table,
        "plots": plots,
    }
    (out_dir / "comparison.json").write_text(json.dumps(summary, indent=2))
    _write_csv(out_dir / "comparison.csv", table)
    return summary


def _write_csv(path: Path, table: Sequence[Dict[str, Any]]) -> None:
    if not table:
        return
    keys = [
        "cost",
        "Rx",
        "Ry",
        "Rz",
        "e_ir_original",
        "e_worst",
        "J_inf",
        "frac_within_20pct_original",
        "J_lambda",
        "J_F",
        "J_RQ",
        "J_V",
        "J_2",
        "relative_spectral_error",
        "elapsed_s",
        "fit_success",
    ]
    extra = [k for k in table[0].keys() if k not in keys]
    cols = keys + extra
    lines = [",".join(cols)]
    for row in table:
        vals = []
        for k in cols:
            v = row.get(k, "")
            if isinstance(v, float):
                vals.append(f"{v:.8g}")
            else:
                vals.append(str(v).replace(",", ";"))
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Pixel-R cost-function sweep")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--comp", type=Path, help="Existing compN/ with Gprime + pixel_model")
    src.add_argument("--spice", type=Path, help="IBM/TSMC SPICE; run spec_flow 01–04 first")
    src.add_argument("--synthetic", action="store_true", help="3×3 Pixel-R identity test")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Sweep output directory (default under lipohan_work/outputs/)",
    )
    p.add_argument(
        "--costs",
        default="priority",
        help="Comma list, 'priority' (default), or 'all'",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chip-pitch", type=float, default=0.0, help="0 = auto (spec_flow)")
    p.add_argument("--rq-basis", choices=("eigen", "random", "mixed", "current"), default="mixed")
    p.add_argument("--p-norm", type=float, default=8.0)
    p.add_argument("--hybrid-alpha", type=float, default=1.0)
    p.add_argument("--hybrid-beta", type=float, default=1.0)
    p.add_argument("--hybrid-gamma", type=float, default=1.0)
    p.add_argument("--max-nfev", type=int, default=300)
    args = p.parse_args(argv)

    costs = parse_cost_list(args.costs)
    work_root = Path(__file__).resolve().parents[1] / "outputs"

    meta: Dict[str, Any] = {}
    if args.synthetic:
        ports, model, Gprime = _synthetic_bundle()
        out_dir = args.out or (work_root / "synthetic_3x3")
        tag = "synthetic"
    elif args.comp:
        ports, model, Gprime, meta = _load_comp(args.comp)
        out_dir = args.out or (work_root / f"{args.comp.parent.name}_{args.comp.name}_costs")
        tag = str(args.comp)
    else:
        spice = args.spice or DEFAULT_SPICE
        if not spice.is_file():
            print(f"error: spice not found: {spice}", file=sys.stderr)
            return 2
        front = args.out or (work_root / f"{spice.stem}_nauto")
        print(f"[frontend] parse/ports/kron/pixel → {front}", flush=True)
        comp = _run_frontend(spice, front, args.chip_pitch)
        ports, model, Gprime, meta = _load_comp(comp)
        out_dir = front / "cost_sweep"
        tag = str(comp)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "tag": tag,
                "out": str(out_dir),
                "costs": costs,
                "n_ports": int(Gprime.shape[0]),
                "n_pads": int(ports.n_pads),
                "n_sinks": int(ports.n_sinks),
                "frontend_meta": meta,
            },
            indent=2,
            default=str,
        ),
        flush=True,
    )
    summary = sweep(
        ports=ports,
        model=model,
        Gprime=Gprime,
        costs=costs,
        seed=args.seed,
        out_dir=out_dir,
        rq_basis=args.rq_basis,
        p_norm=args.p_norm,
        hybrid=(args.hybrid_alpha, args.hybrid_beta, args.hybrid_gamma),
        max_nfev=args.max_nfev,
    )
    print(json.dumps({"out": str(out_dir), "plots": summary["plots"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
