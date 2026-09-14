#!/usr/bin/env python3
"""Stage 08: correlate IR drops from original / reduced / Pixel-R SPICE solves."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from correlate_spice import correlate_three, plot_spice_ir
from io_artifacts import load_fit, load_ports, save_metrics
from spice_emit import load_simulator
from spice_run import load_voltages


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Correlate SPICE IR across three nets")
    p.add_argument("out", type=Path, help="Output / work directory")
    p.add_argument(
        "seed",
        type=int,
        help="RNG seed (reserved; v1 uses lumped original currents only)",
    )
    args = p.parse_args(argv)

    ports, _node_map = load_ports(args.out)
    fit_meta, _spectra = load_fit(args.out)
    voltages = load_voltages(args.out)
    report = correlate_three(voltages, ports)
    ir_source = load_simulator(args.out)

    metrics = {
        "cell_size": ports.cell_size,
        "nx": ports.nx,
        "ny": ports.ny,
        "bbox": list(ports.bbox),
        "Rx": fit_meta.get("Rx"),
        "Ry": fit_meta.get("Ry"),
        "Rz": fit_meta.get("Rz"),
        "residual": fit_meta.get("residual"),
        "relative_spectral_error": fit_meta.get("relative_spectral_error"),
        "relative_ir_error": fit_meta.get("relative_ir_error"),
        "fit_method": fit_meta.get("fit_method", "eigen"),
        "fit_success": fit_meta.get("fit_success"),
        "seed": args.seed,
        "ir_source": ir_source,
        "e_worst": report["e_worst"],
        "e_ir_original_vs_pixel_r": report["e_ir_original_vs_pixel_r"],
        "e_ir_original_vs_reduced": report["e_ir_original_vs_reduced"],
        "e_ir_reduced_vs_pixel_r": report["e_ir_reduced_vs_pixel_r"],
        "pearson_original_vs_pixel_r": report["pearson_original_vs_pixel_r"],
        "ir_pairs": report["ir_pairs"],
    }

    plots: list[str] = []
    try:
        plots = plot_spice_ir(report, ports, args.out)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: plots skipped: {exc}", file=sys.stderr)
    if getattr(ports, "port_mode", "") == "tsmc_virtual_pixel_r":
        try:
            from plot_tsmc_virtual_grid import plot_virtual_grid

            plots = list(plots) + plot_virtual_grid(ports, args.out)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: virtual-grid plots skipped: {exc}", file=sys.stderr)
    metrics["plots"] = plots
    save_metrics(args.out, metrics)

    print(
        json.dumps(
            {
                "stage": "08_correlate_ir",
                "out": str(args.out),
                "e_worst": metrics["e_worst"],
                "e_ir_original_vs_pixel_r": metrics["e_ir_original_vs_pixel_r"],
                "e_ir_original_vs_reduced": metrics["e_ir_original_vs_reduced"],
                "e_ir_reduced_vs_pixel_r": metrics["e_ir_reduced_vs_pixel_r"],
                "plots": plots,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
