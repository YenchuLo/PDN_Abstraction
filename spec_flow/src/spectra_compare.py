"""Post-hoc projected-spectrum comparison for IR-fit vs eigen-fit Gs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from fit_eigen import proj_eigs, relative_spectral_error
from kron import grounded_eigh


@dataclass
class SpectrumSide:
    label: str
    relative_spectral_error: float
    residual: float
    pearson: float
    params: Dict[str, float]


@dataclass
class CompareResult:
    n: int
    gprime_max_abs_diff: float
    lam_M: np.ndarray
    lam_S_ir: np.ndarray
    lam_S_eigen: Optional[np.ndarray]
    ir: SpectrumSide
    eigen: Optional[SpectrumSide]
    written: List[str]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _side(
    label: str, lam_M: np.ndarray, lam_S: np.ndarray, params: Dict[str, float]
) -> SpectrumSide:
    resid = float(np.sum((lam_M - lam_S) ** 2))
    return SpectrumSide(
        label=label,
        relative_spectral_error=relative_spectral_error(lam_M, lam_S),
        residual=resid,
        pearson=_pearson(lam_M, lam_S),
        params=params,
    )


def evaluate_projected_spectrum(
    Gprime: np.ndarray,
    Gs: np.ndarray,
    *,
    lam_M: Optional[np.ndarray] = None,
    Q: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lam_M, Q, lam_S=diag(Q.T @ Gs @ Q))."""
    if lam_M is None or Q is None:
        lam_M, Q = grounded_eigh(Gprime)
    else:
        lam_M = np.asarray(lam_M, dtype=float)
        Q = np.asarray(Q, dtype=float)
    lam_S = proj_eigs(Q, np.asarray(Gs, dtype=float))
    return lam_M, Q, lam_S


def compare_gs_spectra(
    Gprime: np.ndarray,
    Gs_ir: np.ndarray,
    *,
    Gs_eigen: Optional[np.ndarray] = None,
    params_ir: Optional[Dict[str, float]] = None,
    params_eigen: Optional[Dict[str, float]] = None,
    lam_M: Optional[np.ndarray] = None,
    Q: Optional[np.ndarray] = None,
) -> CompareResult:
    lam_M, Q, lam_S_ir = evaluate_projected_spectrum(
        Gprime, Gs_ir, lam_M=lam_M, Q=Q
    )
    ir = _side("ir", lam_M, lam_S_ir, dict(params_ir or {}))

    lam_S_eigen: Optional[np.ndarray] = None
    eigen: Optional[SpectrumSide] = None
    if Gs_eigen is not None:
        lam_S_eigen = proj_eigs(Q, np.asarray(Gs_eigen, dtype=float))
        eigen = _side("eigen", lam_M, lam_S_eigen, dict(params_eigen or {}))

    return CompareResult(
        n=int(lam_M.size),
        gprime_max_abs_diff=0.0,
        lam_M=lam_M,
        lam_S_ir=lam_S_ir,
        lam_S_eigen=lam_S_eigen,
        ir=ir,
        eigen=eigen,
        written=[],
    )


def plot_eigen_ir_compare(
    result: CompareResult,
    out_dir: Path,
    *,
    prefix: str = "eigen_ir_compare",
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    lam_M = result.lam_M
    lam_ir = result.lam_S_ir
    lam_eig = result.lam_S_eigen

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(
        lam_M,
        lam_ir,
        s=14,
        alpha=0.75,
        c="#c44e52",
        label=f"IR  (rel={result.ir.relative_spectral_error:.3g})",
    )
    if lam_eig is not None and result.eigen is not None:
        ax.scatter(
            lam_M,
            lam_eig,
            s=14,
            alpha=0.65,
            c="#4c72b0",
            label=f"eigen (rel={result.eigen.relative_spectral_error:.3g})",
        )
    lo = float(min(lam_M.min(), lam_ir.min()))
    hi = float(max(lam_M.max(), lam_ir.max()))
    if lam_eig is not None:
        lo = min(lo, float(lam_eig.min()))
        hi = max(hi, float(lam_eig.max()))
    if lo == hi:
        lo, hi = lo - 1e-6, hi + 1e-6
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(r"$\lambda^M$ (from $G'$)")
    ax.set_ylabel(r"$\lambda^S=\mathrm{diag}(Q^\top G_S Q)$")
    ax.set_title("Projected spectrum: IR-fit vs eigen-fit")
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    p = out_dir / f"{prefix}_scatter.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p))

    order = np.argsort(lam_M)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(lam_M[order], "k-", lw=1.5, label=r"$\lambda^M$")
    ax.plot(lam_ir[order], color="#c44e52", lw=1.2, label=r"$\lambda^{S,\mathrm{IR}}$")
    if lam_eig is not None:
        ax.plot(
            lam_eig[order],
            color="#4c72b0",
            lw=1.2,
            label=r"$\lambda^{S,\mathrm{eigen}}$",
        )
    ax.set_xlabel(r"mode index (sorted by $\lambda^M$)")
    ax.set_ylabel("eigenvalue")
    ax.set_title("Sorted projected eigenvalues")
    ax.legend(loc="best", fontsize=9)
    ax.set_yscale("symlog", linthresh=1e-12)
    p = out_dir / f"{prefix}_sorted.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    denom = np.abs(lam_M[order]) + 1e-30
    ax.semilogy(
        np.abs(lam_ir[order] - lam_M[order]) / denom,
        color="#c44e52",
        lw=1.2,
        label="IR",
    )
    if lam_eig is not None:
        ax.semilogy(
            np.abs(lam_eig[order] - lam_M[order]) / denom,
            color="#4c72b0",
            lw=1.2,
            label="eigen",
        )
    ax.set_xlabel(r"mode index (sorted by $\lambda^M$)")
    ax.set_ylabel(r"$|\lambda^S-\lambda^M| / |\lambda^M|$")
    ax.set_title("Per-mode relative spectral error")
    ax.legend(loc="best", fontsize=9)
    p = out_dir / f"{prefix}_relerr.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p))

    return written
