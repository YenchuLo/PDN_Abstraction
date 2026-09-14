#!/usr/bin/env python3
"""Compare projected spectra of IR-fit Gs vs eigen-fit Gs.

Post-hoc evaluation (not an inverse of the IR objective):

  G' = Q diag(lam_M) Q^T
  lam_S = diag(Q^T G_S Q)

Given an IR-fitted G_S and (optionally) an eigen-fitted G_S, report relative
spectral error / Pearson vs lam_M and write overlay plots.

Usage:
  python spec_flow/src/compare_eigen_ir.py \\
      spec_flow/outputs/ibmpg2_nauto/comp1 \\
      spec_flow/outputs/ibmpg2_nauto_ir/comp1

  python spec_flow/src/compare_eigen_ir.py --refit-eigen \\
      spec_flow/outputs/ibmpg2_nauto_ir/comp1
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_eigen import fit_pixel_r  # noqa: E402
from io_artifacts import load_fit, load_pixel_model_fields, load_ports  # noqa: E402
from kron import grounded_eigh  # noqa: E402
from pixel_r import pixel_model_from_fields  # noqa: E402
from spectra_compare import (  # noqa: E402
    compare_gs_spectra,
    plot_eigen_ir_compare,
)


def _r_params(meta: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in ("Rx", "Ry", "Rz", "Rx_u", "Ry_u", "Rx_l", "Ry_l", "Rz_pad", "Rz_ul"):
        if k in meta and meta[k] is not None:
            try:
                out[k] = float(meta[k])
            except (TypeError, ValueError):
                pass
    return out


def _load_gs_pair(
    eigen_dir: Path, ir_dir: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float], Dict[str, float], float]:
    meta_e, sp_e = load_fit(eigen_dir)
    meta_i, sp_i = load_fit(ir_dir)
    Gprime_e = np.asarray(sp_e["Gprime"], dtype=float)
    Gprime_i = np.asarray(sp_i["Gprime"], dtype=float)
    if Gprime_e.shape != Gprime_i.shape:
        raise ValueError(
            f"G' shape mismatch: eigen {Gprime_e.shape} vs IR {Gprime_i.shape}"
        )
    gdiff = float(np.max(np.abs(Gprime_e - Gprime_i)))
    Gs_e = np.asarray(sp_e["Gs"], dtype=float)
    Gs_i = np.asarray(sp_i["Gs"], dtype=float)
    lam_M = np.asarray(sp_e.get("lam_M", []), dtype=float)
    Gprime = Gprime_e if lam_M.size else Gprime_i
    return Gprime, Gs_i, Gs_e, _r_params(meta_i), _r_params(meta_e), gdiff


def _write_artifacts(
    dest: Path,
    prefix: str,
    result,
    *,
    Gprime: np.ndarray,
    Gs_ir: np.ndarray,
    Gs_eigen: Optional[np.ndarray],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    written = plot_eigen_ir_compare(result, dest, prefix=prefix)
    result.written = written
    payload["plots"] = written
    json_path = dest / f"{prefix}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        dest / f"{prefix}_spectra.npz",
        lam_M=result.lam_M,
        lam_S_ir=result.lam_S_ir,
        lam_S_eigen=(
            result.lam_S_eigen
            if result.lam_S_eigen is not None
            else np.asarray([], dtype=float)
        ),
        Gprime=Gprime,
        Gs_ir=Gs_ir,
        Gs_eigen=(
            Gs_eigen if Gs_eigen is not None else np.asarray([], dtype=float)
        ),
    )
    payload["json"] = str(json_path)
    payload["spectra_npz"] = str(dest / f"{prefix}_spectra.npz")
    return payload


def compare_dirs(
    eigen_dir: Path,
    ir_dir: Path,
    *,
    out_dir: Optional[Path] = None,
    prefix: str = "eigen_ir_compare",
) -> Dict[str, Any]:
    Gprime, Gs_ir, Gs_eigen, params_ir, params_eigen, gdiff = _load_gs_pair(
        Path(eigen_dir), Path(ir_dir)
    )
    _, sp_e = load_fit(eigen_dir)
    lam_M_saved = np.asarray(sp_e.get("lam_M", []), dtype=float)
    lam_M = lam_M_saved if lam_M_saved.size else None
    Q = None
    if lam_M is not None:
        _, Q = grounded_eigh(Gprime)

    result = compare_gs_spectra(
        Gprime,
        Gs_ir,
        Gs_eigen=Gs_eigen,
        params_ir=params_ir,
        params_eigen=params_eigen,
        lam_M=lam_M,
        Q=Q,
    )
    result.gprime_max_abs_diff = gdiff

    dest = Path(out_dir) if out_dir is not None else Path(ir_dir)
    payload = {
        "eigen_dir": str(Path(eigen_dir).resolve()),
        "ir_dir": str(Path(ir_dir).resolve()),
        "n": result.n,
        "gprime_max_abs_diff": result.gprime_max_abs_diff,
        "ir": asdict(result.ir),
        "eigen": None if result.eigen is None else asdict(result.eigen),
    }
    return _write_artifacts(
        dest,
        prefix,
        result,
        Gprime=Gprime,
        Gs_ir=Gs_ir,
        Gs_eigen=Gs_eigen,
        payload=payload,
    )


def compare_ir_dir_with_refit(
    ir_dir: Path,
    *,
    out_dir: Optional[Path] = None,
    prefix: str = "eigen_ir_compare",
) -> Dict[str, Any]:
    """Evaluate IR Gs spectrum and re-run Pixel-R eigen LS on the same model."""
    ir_dir = Path(ir_dir)
    meta_i, sp_i = load_fit(ir_dir)
    Gprime = np.asarray(sp_i["Gprime"], dtype=float)
    Gs_ir = np.asarray(sp_i["Gs"], dtype=float)
    ports, _ = load_ports(ir_dir)
    fields = load_pixel_model_fields(ir_dir)
    model = pixel_model_from_fields(ports, fields)
    fit_e = fit_pixel_r(Gprime, model)

    result = compare_gs_spectra(
        Gprime,
        Gs_ir,
        Gs_eigen=fit_e.Gs,
        params_ir=_r_params(meta_i),
        params_eigen={"Rx": fit_e.Rx, "Ry": fit_e.Ry, "Rz": fit_e.Rz},
        lam_M=fit_e.lam_M,
        Q=fit_e.Q,
    )
    dest = Path(out_dir) if out_dir is not None else ir_dir
    payload = {
        "ir_dir": str(ir_dir.resolve()),
        "eigen_source": "refit_pixel_r",
        "n": result.n,
        "gprime_max_abs_diff": 0.0,
        "ir": asdict(result.ir),
        "eigen": None if result.eigen is None else asdict(result.eigen),
        "eigen_fit_success": fit_e.success,
        "eigen_fit_message": fit_e.message,
    }
    return _write_artifacts(
        dest,
        prefix,
        result,
        Gprime=Gprime,
        Gs_ir=Gs_ir,
        Gs_eigen=fit_e.Gs,
        payload=payload,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Back-project IR-fit Gs eigenvalues and compare with eigen LS "
            "(Pixel-R / spec_flow)."
        )
    )
    p.add_argument(
        "eigen_or_ir",
        type=Path,
        help="Eigen-fit comp dir, or IR-fit comp dir when using --refit-eigen",
    )
    p.add_argument(
        "ir_dir",
        type=Path,
        nargs="?",
        default=None,
        help="IR-fit comp dir (omit with --refit-eigen)",
    )
    p.add_argument(
        "--refit-eigen",
        action="store_true",
        help="Only pass IR dir; re-run eigen LS on the same Pixel-R model",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory for plots/JSON (default: IR comp dir)",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="eigen_ir_compare",
        help="Output filename prefix (default: eigen_ir_compare)",
    )
    args = p.parse_args(argv)

    if args.refit_eigen:
        payload = compare_ir_dir_with_refit(
            args.eigen_or_ir, out_dir=args.out, prefix=args.prefix
        )
    else:
        if args.ir_dir is None:
            p.error("ir_dir is required unless --refit-eigen is set")
        payload = compare_dirs(
            args.eigen_or_ir,
            args.ir_dir,
            out_dir=args.out,
            prefix=args.prefix,
        )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
