#!/usr/bin/env python3
"""Stage 05 (alt): multi-stimulus mixed-BC IR fit → Pixel-R Rx, Ry, Rz."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_ir import fit_pixel_r_ir
from io_artifacts import load_gprime, load_pixel_model_fields, load_ports, save_fit
from pixel_r import pixel_model_from_fields


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fit Pixel-R by mixed-BC IR matching")
    p.add_argument("out", type=Path, help="Output / work directory")
    p.add_argument(
        "seed",
        type=int,
        nargs="?",
        default=0,
        help="RNG seed for IR fit stimuli (default 0)",
    )
    args = p.parse_args(argv)

    ports, _node_map = load_ports(args.out)
    Gprime, _meta = load_gprime(args.out)
    fields = load_pixel_model_fields(args.out)
    model = pixel_model_from_fields(ports, fields)
    fit = fit_pixel_r_ir(Gprime, model, ports, seed=args.seed)
    pixel_r = {"Rx": fit.Rx, "Ry": fit.Ry, "Rz": fit.Rz}
    fit_meta = {
        "fit_method": "ir",
        "residual": fit.residual,
        "relative_ir_error": fit.relative_ir_error,
        "fit_success": fit.success,
        "fit_message": fit.message,
        "stimulus_names": fit.stimulus_names,
        "seed": args.seed,
    }
    save_fit(args.out, pixel_r, None, None, Gprime, fit.Gs, fit_meta)
    print(
        json.dumps(
            {"stage": "05_fit_ir", "out": str(args.out), **pixel_r, **fit_meta},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
