#!/usr/bin/env python3
"""Stage 05: IR-fit localized Pixel-R (uniform init + Feng block τ)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from _bootstrap import ensure_paths

ensure_paths()

from fit_ir import fit_localized_ir  # noqa: E402
from io_artifacts import load_gprime, load_ports  # noqa: E402
from localized_pixel import localized_model_from_fields  # noqa: E402

MODEL_JSON = "localized_model.json"
R_JSON = "localized_r.json"
SPECTRA_NPZ = "localized_spectra.npz"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fit localized Pixel-R by IR matching")
    p.add_argument("out", type=Path, help="Component OUT directory")
    p.add_argument("seed", type=int, nargs="?", default=0)
    args = p.parse_args(argv)

    ports, _ = load_ports(args.out)
    Gprime, _ = load_gprime(args.out)
    fields = json.loads((args.out / MODEL_JSON).read_text())
    model = localized_model_from_fields(ports, fields)
    fit = fit_localized_ir(Gprime, model, ports, seed=args.seed)

    payload = {
        "fit_method": "ir_feng_localized",
        "block_size": model.partition.block_size,
        "n_regions": model.n_regions,
        "Rx_r": fit.Rx_r.tolist(),
        "Ry_r": fit.Ry_r.tolist(),
        "Rz_r": fit.Rz_r.tolist(),
        "Rx_cell": fit.Rx_cell.tolist(),
        "Ry_cell": fit.Ry_cell.tolist(),
        "Rz_cell": fit.Rz_cell.tolist(),
        "Rx_uniform": fit.Rx_uniform,
        "Ry_uniform": fit.Ry_uniform,
        "Rz_uniform": fit.Rz_uniform,
        "residual": fit.residual,
        "relative_ir_error": fit.relative_ir_error,
        "residual_uniform": fit.residual_uniform,
        "relative_ir_error_uniform": fit.relative_ir_error_uniform,
        "e_worst": fit.e_worst,
        "e_ir_by_stimulus": fit.e_ir_by_stimulus,
        "e_ir_by_stimulus_uniform": fit.e_ir_by_stimulus_uniform,
        "n_feng_iters": fit.n_feng_iters,
        "polished": fit.polished,
        "fit_success": fit.success,
        "fit_message": fit.message,
        "stimulus_names": fit.stimulus_names,
        "seed": args.seed,
    }
    (args.out / R_JSON).write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        args.out / SPECTRA_NPZ,
        Gprime=np.asarray(Gprime, dtype=float),
        Gs=np.asarray(fit.Gs, dtype=float),
        Gs_uniform=np.asarray(fit.Gs_uniform, dtype=float),
        Rx_cell=fit.Rx_cell,
        Ry_cell=fit.Ry_cell,
        Rz_cell=fit.Rz_cell,
    )
    fields.update(
        {
            "Rx_r": fit.Rx_r.tolist(),
            "Ry_r": fit.Ry_r.tolist(),
            "Rz_r": fit.Rz_r.tolist(),
            "Rx_uniform": fit.Rx_uniform,
            "Ry_uniform": fit.Ry_uniform,
            "Rz_uniform": fit.Rz_uniform,
        }
    )
    (args.out / MODEL_JSON).write_text(json.dumps(fields, indent=2))
    print(
        json.dumps(
            {
                "stage": "05_fit_ir",
                "out": str(args.out),
                "n_regions": model.n_regions,
                "relative_ir_error": fit.relative_ir_error,
                "relative_ir_error_uniform": fit.relative_ir_error_uniform,
                "e_ir_by_stimulus": fit.e_ir_by_stimulus,
                "n_feng_iters": fit.n_feng_iters,
                "polished": fit.polished,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
