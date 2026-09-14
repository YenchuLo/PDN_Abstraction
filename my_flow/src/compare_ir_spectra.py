#!/usr/bin/env python3
"""Back-project IR-fit Gs eigenvalues; optionally compare to eigen LS.

For my_flow models (tri_stagger / tri_square; voltspot: back-project only):

  1. Load G', Gs from <model>_spectra.npz (IR fit).
  2. Compute lam_M, Q = grounded_eigh(G') and
     lam_S_IR = diag(Q.T @ Gs_IR @ Q).
  3. With --refit-eigen, re-run eigenvalue LS on the same model class and
     overlay lam_S_eigen.

Usage:
  python my_flow/src/compare_ir_spectra.py OUT/comp1 --model tri_stagger
  python my_flow/src/compare_ir_spectra.py OUT/comp1 --model tri_square --refit-eigen
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from _bootstrap import ensure_paths

ensure_paths()

from fit_eigen import fit_log_r_params_eigen  # noqa: E402
from fit_ir_tri_square import TRI_SQUARE_PARAM_NAMES  # noqa: E402
from io_artifacts import (  # noqa: E402
    load_ports,
    load_tri_square_model_fields,
    load_tri_stagger_model_fields,
)
from ports_dual import DualPortSet, tri_stagger_ports_from_dual  # noqa: E402
from spectra_compare import compare_gs_spectra, plot_eigen_ir_compare  # noqa: E402
from tri_square import build_Gs_tri_square, tri_square_model_from_fields  # noqa: E402
from tri_stagger import build_Gs_tri_stagger, tri_stagger_model_from_fields  # noqa: E402

MODEL_CFG = {
    "tri_stagger": {
        "spectra": "tri_stagger_spectra.npz",
        "r_json": "tri_stagger_r.json",
        "param_names": ["Rx", "Ry", "Rz"],
        "prefix": "eigen_ir_compare_tri_stagger",
        "can_refit": True,
    },
    "tri_square": {
        "spectra": "tri_square_spectra.npz",
        "r_json": "tri_square_r.json",
        "param_names": list(TRI_SQUARE_PARAM_NAMES),
        "prefix": "eigen_ir_compare_tri_square",
        "can_refit": True,
    },
    "voltspot": {
        "spectra": "voltspot_spectra.npz",
        "r_json": "voltspot_r.json",
        "param_names": ["alpha_x", "alpha_y"],
        "prefix": "eigen_ir_compare_voltspot",
        "can_refit": False,
    },
}


def _r_params(meta: Dict[str, Any], names: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in names:
        if k in meta and meta[k] is not None:
            out[k] = float(meta[k])
    # Common aliases for stagger / voltspot
    aliases = {
        "Rx": ("alpha_x", "rx0"),
        "Ry": ("alpha_y", "ry0"),
        "Rz": ("alpha_via",),
        "alpha_x": ("Rx",),
        "alpha_y": ("Ry",),
    }
    for k in names:
        if k in out:
            continue
        for alt in aliases.get(k, ()):
            if alt in meta and meta[alt] is not None:
                out[k] = float(meta[alt])
                break
    return out


def _load_spectra(out: Path, model: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    cfg = MODEL_CFG[model]
    sp_path = out / cfg["spectra"]
    r_path = out / cfg["r_json"]
    if not sp_path.is_file():
        raise FileNotFoundError(f"missing {sp_path}")
    if not r_path.is_file():
        raise FileNotFoundError(f"missing {r_path}")
    sp = np.load(sp_path)
    if "Gprime" not in sp or "Gs" not in sp:
        raise KeyError(f"{sp_path} must contain Gprime and Gs")
    meta = json.loads(r_path.read_text())
    return (
        np.asarray(sp["Gprime"], dtype=float),
        np.asarray(sp["Gs"], dtype=float),
        meta,
    )


def _model_ports(out: Path) -> DualPortSet:
    dual_ports, _node_map = load_ports(out)
    return tri_stagger_ports_from_dual(dual_ports)


def _build_gs_fn(
    out: Path, model: str
) -> Tuple[Callable[..., np.ndarray], List[str]]:
    ports = _model_ports(out)
    if model == "tri_stagger":
        fields = load_tri_stagger_model_fields(out)
        m = tri_stagger_model_from_fields(ports, fields)

        def build_Gs(Rx: float, Ry: float, Rz: float) -> np.ndarray:
            return build_Gs_tri_stagger(m, Rx=Rx, Ry=Ry, Rz=Rz)

        return build_Gs, list(MODEL_CFG[model]["param_names"])

    if model == "tri_square":
        fields = load_tri_square_model_fields(out)
        m = tri_square_model_from_fields(ports, fields)

        def build_Gs(
            Rx_u: float,
            Ry_u: float,
            Rx_l: float,
            Ry_l: float,
            Rz_pad: float,
            Rz_ul: float,
        ) -> np.ndarray:
            return build_Gs_tri_square(
                m,
                Rx_u=Rx_u,
                Ry_u=Ry_u,
                Rx_l=Rx_l,
                Ry_l=Ry_l,
                Rz_pad=Rz_pad,
                Rz_ul=Rz_ul,
            )

        return build_Gs, list(MODEL_CFG[model]["param_names"])

    raise ValueError(f"eigen refit not supported for model {model!r}")


def run_compare(
    out: Path,
    model: str,
    *,
    refit_eigen: bool = False,
    out_dir: Optional[Path] = None,
    max_nfev: int = 300,
) -> Dict[str, Any]:
    out = Path(out)
    cfg = MODEL_CFG[model]
    Gprime, Gs_ir, meta_ir = _load_spectra(out, model)
    params_ir = _r_params(meta_ir, cfg["param_names"])

    Gs_eigen = None
    params_eigen: Dict[str, float] = {}
    eigen_meta: Dict[str, Any] = {}
    lam_M = None
    Q = None

    if refit_eigen:
        if not cfg["can_refit"]:
            raise ValueError(
                f"--refit-eigen is not supported for model={model!r} "
                "(back-projection of IR Gs still works without this flag)"
            )
        build_Gs, names = _build_gs_fn(out, model)
        x0 = (
            [params_ir[n] for n in names]
            if all(n in params_ir for n in names)
            else None
        )
        fit_e = fit_log_r_params_eigen(
            Gprime,
            build_Gs,
            names,
            x0=x0,
            max_nfev=max_nfev,
        )
        Gs_eigen = fit_e.Gs
        params_eigen = dict(fit_e.params)
        lam_M = fit_e.lam_M
        Q = fit_e.Q
        eigen_meta = {
            "eigen_source": "refit",
            "eigen_fit_success": fit_e.success,
            "eigen_fit_message": fit_e.message,
        }

    result = compare_gs_spectra(
        Gprime,
        Gs_ir,
        Gs_eigen=Gs_eigen,
        params_ir=params_ir,
        params_eigen=params_eigen,
        lam_M=lam_M,
        Q=Q,
    )
    dest = Path(out_dir) if out_dir is not None else out
    prefix = str(cfg["prefix"])
    written = plot_eigen_ir_compare(result, dest, prefix=prefix)
    result.written = written

    payload: Dict[str, Any] = {
        "out": str(out.resolve()),
        "model": model,
        "refit_eigen": bool(refit_eigen),
        "n": result.n,
        "ir": asdict(result.ir),
        "eigen": None if result.eigen is None else asdict(result.eigen),
        "plots": written,
        **eigen_meta,
    }
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Back-project IR-fit eigenvalues; optionally refit eigen LS"
    )
    p.add_argument("out", type=Path, help="Component OUT (e.g. .../comp1)")
    p.add_argument(
        "--model",
        choices=sorted(MODEL_CFG.keys()),
        default="tri_stagger",
        help="Which IR-fit spectra to evaluate (default: tri_stagger)",
    )
    p.add_argument(
        "--refit-eigen",
        action="store_true",
        help="Also run eigenvalue LS on the same model and overlay spectra",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for plots/JSON (default: component OUT)",
    )
    p.add_argument("--max-nfev", type=int, default=300)
    args = p.parse_args(argv)

    payload = run_compare(
        args.out,
        args.model,
        refit_eigen=args.refit_eigen,
        out_dir=args.out_dir,
        max_nfev=args.max_nfev,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
