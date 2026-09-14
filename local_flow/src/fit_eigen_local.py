"""Localized Pixel-R eigenvalue LS (same objective as spec_flow, per-region R)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from _bootstrap import ensure_paths

ensure_paths()

from fit_eigen import fit_log_r_params_eigen, fit_pixel_r  # noqa: E402
from localized_pixel import LocalizedPixelModel, build_Gs_from_regions  # noqa: E402
from partitions import expand_region_params  # noqa: E402


@dataclass
class FitLocalizedEigenResult:
    Rx_r: np.ndarray
    Ry_r: np.ndarray
    Rz_r: np.ndarray
    Rx_cell: np.ndarray
    Ry_cell: np.ndarray
    Rz_cell: np.ndarray
    Rx_uniform: float
    Ry_uniform: float
    Rz_uniform: float
    residual: float
    relative_spectral_error: float
    residual_uniform: float
    relative_spectral_error_uniform: float
    Gs: np.ndarray
    Gs_uniform: np.ndarray
    success: bool
    message: str


def fit_localized_eigen(
    Gprime: np.ndarray,
    model: LocalizedPixelModel,
    *,
    max_nfev: int | None = None,
) -> FitLocalizedEigenResult:
    """min ||λ_M - diag(Qᵀ G_S Q)||² over per-region (Rx, Ry, Rz)."""
    uni = fit_pixel_r(Gprime, model.star)
    nr = int(model.n_regions)
    x0 = np.concatenate(
        [
            np.full(nr, uni.Rx, dtype=float),
            np.full(nr, uni.Ry, dtype=float),
            np.full(nr, uni.Rz, dtype=float),
        ]
    )
    names: List[str] = (
        [f"Rx_{i}" for i in range(nr)]
        + [f"Ry_{i}" for i in range(nr)]
        + [f"Rz_{i}" for i in range(nr)]
    )
    nfev = int(max_nfev) if max_nfev is not None else max(400, 8 * nr)

    def _build(*vals: float) -> np.ndarray:
        v = np.asarray(vals, dtype=float)
        return build_Gs_from_regions(model, v[:nr], v[nr : 2 * nr], v[2 * nr :])

    fit = fit_log_r_params_eigen(
        Gprime,
        _build,
        names,
        x0=x0,
        max_nfev=nfev,
    )
    Rx_r = np.array([fit.params[f"Rx_{i}"] for i in range(nr)], dtype=float)
    Ry_r = np.array([fit.params[f"Ry_{i}"] for i in range(nr)], dtype=float)
    Rz_r = np.array([fit.params[f"Rz_{i}"] for i in range(nr)], dtype=float)
    Rx_c, Ry_c, Rz_c = expand_region_params(model.partition, Rx_r, Ry_r, Rz_r)
    Gs_uni = build_Gs_from_regions(
        model,
        np.full(nr, uni.Rx),
        np.full(nr, uni.Ry),
        np.full(nr, uni.Rz),
    )
    return FitLocalizedEigenResult(
        Rx_r=Rx_r,
        Ry_r=Ry_r,
        Rz_r=Rz_r,
        Rx_cell=Rx_c,
        Ry_cell=Ry_c,
        Rz_cell=Rz_c,
        Rx_uniform=float(uni.Rx),
        Ry_uniform=float(uni.Ry),
        Rz_uniform=float(uni.Rz),
        residual=float(fit.residual),
        relative_spectral_error=float(fit.relative_spectral_error),
        residual_uniform=float(uni.residual),
        relative_spectral_error_uniform=float(uni.relative_spectral_error),
        Gs=np.asarray(fit.Gs, dtype=float),
        Gs_uniform=np.asarray(Gs_uni, dtype=float),
        success=bool(fit.success),
        message=str(fit.message),
    )
