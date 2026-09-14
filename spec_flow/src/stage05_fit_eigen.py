#!/usr/bin/env python3
"""Stage 05: eigenvalue LS fit → Pixel-R Rx, Ry, Rz."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_eigen import fit_pixel_r
from io_artifacts import load_gprime, load_pixel_model_fields, load_ports, save_fit
from pixel_r import pixel_model_from_fields


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fit Pixel-R by eigenvalue matching")
    p.add_argument("out", type=Path, help="Output / work directory")
    args = p.parse_args(argv)

    ports, _node_map = load_ports(args.out)
    Gprime, _meta = load_gprime(args.out)
    fields = load_pixel_model_fields(args.out)
    model = pixel_model_from_fields(ports, fields)
    fit = fit_pixel_r(Gprime, model)
    pixel_r = {"Rx": fit.Rx, "Ry": fit.Ry, "Rz": fit.Rz}
    fit_meta = {
        "residual": fit.residual,
        "relative_spectral_error": fit.relative_spectral_error,
        "fit_success": fit.success,
        "fit_message": fit.message,
    }
    save_fit(args.out, pixel_r, fit.lam_M, fit.lam_S, Gprime, fit.Gs, fit_meta)
    print(
        json.dumps(
            {"stage": "05_fit_eigen", "out": str(args.out), **pixel_r, **fit_meta},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
