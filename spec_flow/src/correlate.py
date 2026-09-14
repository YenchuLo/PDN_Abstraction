"""Correlation metrics between abstracted G_S and real G'_M."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from ports import PortSet


@dataclass
class StimulusResult:
    name: str
    e_ir: float
    pearson_r: float


@dataclass
class CorrelationReport:
    eigen_residual: float
    relative_spectral_error: float
    pearson_eigen: float
    e_ir_by_stimulus: List[StimulusResult] = field(default_factory=list)
    e_worst: float = 0.0
    pearson_ir_concat: float = 0.0

    def to_dict(self) -> dict:
        return {
            "eigen_residual": self.eigen_residual,
            "relative_spectral_error": self.relative_spectral_error,
            "pearson_eigen": self.pearson_eigen,
            "e_worst": self.e_worst,
            "pearson_ir_concat": self.pearson_ir_concat,
            "e_ir_by_stimulus": [asdict(s) for s in self.e_ir_by_stimulus],
        }


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _factor_sink_block(G: np.ndarray, n_pads: int, reg: float = 1e-9):
    """Factor Gss once for repeated mixed-BC solves."""
    n_s = G.shape[0] - n_pads
    if n_s == 0:
        return None, None
    Gss = G[n_pads:, n_pads:] + np.eye(n_s, dtype=float) * reg
    Gsp = G[n_pads:, :n_pads]
    try:
        lu = lu_factor(Gss, check_finite=False)
    except np.linalg.LinAlgError:
        lu = None
    return lu, Gsp


def solve_mixed_bc(
    G: np.ndarray,
    n_pads: int,
    V_pad: np.ndarray,
    I_sink: np.ndarray,
    *,
    reg: float = 1e-9,
    _lu=None,
    _Gsp: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Pads voltage-fixed, sinks current-driven.

    Partition G = [Gpp Gps; Gsp Gss].
    With Vp fixed and Is given:
      Gss Vs = Is - Gsp Vp
    Returns full port voltage vector [Vp; Vs].
    """
    n_p = n_pads
    n_s = G.shape[0] - n_p
    if n_s == 0:
        return np.asarray(V_pad, dtype=float).copy()

    Gsp = _Gsp if _Gsp is not None else G[n_p:, :n_p]
    rhs = np.asarray(I_sink, dtype=float) - Gsp @ np.asarray(V_pad, dtype=float)
    if _lu is not None:
        Vs = lu_solve(_lu, rhs, check_finite=False)
    else:
        Gss = G[n_p:, n_p:] + np.eye(n_s, dtype=float) * reg
        try:
            Vs = np.linalg.solve(Gss, rhs)
        except np.linalg.LinAlgError:
            Vs = np.linalg.lstsq(Gss, rhs, rcond=None)[0]
    return np.concatenate([np.asarray(V_pad, dtype=float), Vs])


def e_ir(V_s: np.ndarray, V_m: np.ndarray) -> float:
    V_s = np.asarray(V_s, dtype=float)
    V_m = np.asarray(V_m, dtype=float)
    denom = np.linalg.norm(V_m)
    if denom < 1e-30:
        return float(np.linalg.norm(V_s - V_m))
    return float(np.linalg.norm(V_s - V_m) / denom)


def build_stimuli(
    ports: PortSet,
    *,
    seed: int = 0,
) -> List[Tuple[str, np.ndarray]]:
    """Return list of (name, I_sink) with length n_sinks."""
    rng = np.random.default_rng(seed)
    n_s = ports.n_sinks
    if n_s == 0:
        return []

    original = ports.lumped_sink_currents()
    stimuli = [("original", original)]
    stimuli.append(("random", rng.normal(0.0, 1.0, size=n_s)))

    localized = np.zeros(n_s)
    localized[n_s // 2] = 1.0
    stimuli.append(("localized", localized))

    striped = np.zeros(n_s)
    for k, c in enumerate(ports.cells):
        striped[k] = 1.0 if (c.ix % 2 == 0) else -1.0
    stimuli.append(("striped", striped))
    return stimuli


def correlate(
    Gprime: np.ndarray,
    Gs: np.ndarray,
    ports: PortSet,
    *,
    lam_M: Optional[np.ndarray] = None,
    lam_S: Optional[np.ndarray] = None,
    eigen_residual: Optional[float] = None,
    relative_spectral_error: Optional[float] = None,
    seed: int = 0,
) -> CorrelationReport:
    if lam_M is not None and lam_S is not None:
        pearson_eigen = _pearson(lam_M, lam_S)
        if eigen_residual is None:
            eigen_residual = float(np.sum((lam_M - lam_S) ** 2))
        if relative_spectral_error is None:
            denom = float(np.sum(lam_M**2)) + 1e-30
            relative_spectral_error = float(np.sqrt(eigen_residual / denom))
    else:
        pearson_eigen = float("nan")
        eigen_residual = float("nan") if eigen_residual is None else eigen_residual
        relative_spectral_error = (
            float("nan") if relative_spectral_error is None else relative_spectral_error
        )

    V_pad = np.asarray(ports.pad_voltages, dtype=float)
    if ports.n_pads == 0:
        raise ValueError("need at least one pad for mixed-BC correlation")

    n_p = ports.n_pads
    lu_m, Gsp_m = _factor_sink_block(Gprime, n_p)
    lu_s, Gsp_s = _factor_sink_block(Gs, n_p)

    results: List[StimulusResult] = []
    all_m: List[np.ndarray] = []
    all_s: List[np.ndarray] = []

    for name, I_sink in build_stimuli(ports, seed=seed):
        Vm = solve_mixed_bc(Gprime, n_p, V_pad, I_sink, _lu=lu_m, _Gsp=Gsp_m)
        Vs = solve_mixed_bc(Gs, n_p, V_pad, I_sink, _lu=lu_s, _Gsp=Gsp_s)
        vref = float(np.mean(V_pad))
        drop_m = vref - Vm[n_p:]
        drop_s = vref - Vs[n_p:]
        err = e_ir(drop_s, drop_m)
        r = _pearson(drop_m, drop_s)
        results.append(StimulusResult(name=name, e_ir=err, pearson_r=r))
        all_m.append(drop_m)
        all_s.append(drop_s)

    e_worst = max((r.e_ir for r in results), default=0.0)
    if all_m:
        pearson_concat = _pearson(np.concatenate(all_m), np.concatenate(all_s))
    else:
        pearson_concat = float("nan")

    return CorrelationReport(
        eigen_residual=float(eigen_residual),
        relative_spectral_error=float(relative_spectral_error),
        pearson_eigen=pearson_eigen,
        e_ir_by_stimulus=results,
        e_worst=e_worst,
        pearson_ir_concat=pearson_concat,
    )


def plot_correlation(
    report: CorrelationReport,
    lam_M: np.ndarray,
    lam_S: np.ndarray,
    Gprime: np.ndarray,
    Gs: np.ndarray,
    ports: PortSet,
    out_dir: str | Path,
    *,
    seed: int = 0,
) -> List[str]:
    """Write scatter / eigen / IR comparison plots. Returns written paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(lam_M, lam_S, s=12, alpha=0.7)
    lims = [min(lam_M.min(), lam_S.min()), max(lam_M.max(), lam_S.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel(r"$\lambda^M$")
    ax.set_ylabel(r"$\lambda^S$ (projected)")
    ax.set_title(f"Eigenvalues (r={report.pearson_eigen:.4f})")
    ax.set_aspect("equal", adjustable="box")
    p = out_dir / "eigen_scatter.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p))

    V_pad = np.asarray(ports.pad_voltages, dtype=float)
    stimuli = dict(build_stimuli(ports, seed=seed))
    if "original" in stimuli:
        I = stimuli["original"]
        lu_m, Gsp_m = _factor_sink_block(Gprime, ports.n_pads)
        lu_s, Gsp_s = _factor_sink_block(Gs, ports.n_pads)
        Vm = solve_mixed_bc(Gprime, ports.n_pads, V_pad, I, _lu=lu_m, _Gsp=Gsp_m)
        Vs = solve_mixed_bc(Gs, ports.n_pads, V_pad, I, _lu=lu_s, _Gsp=Gsp_s)
        n_p = ports.n_pads
        vref = float(np.mean(V_pad))
        drop_m = vref - Vm[n_p:]
        drop_s = vref - Vs[n_p:]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(drop_m, drop_s, s=12, alpha=0.7)
        lo = min(drop_m.min(), drop_s.min())
        hi = max(drop_m.max(), drop_s.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("Real IR drop (G'M)")
        ax.set_ylabel("Abstracted IR drop (Gs)")
        ax.set_title(f"IR original (e={report.e_ir_by_stimulus[0].e_ir:.4g})")
        p = out_dir / "ir_scatter_original.png"
        fig.tight_layout()
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(str(p))

        err_map = np.full((ports.ny, ports.nx), np.nan)
        for k, c in enumerate(ports.cells):
            err_map[c.iy, c.ix] = abs(drop_s[k] - drop_m[k])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(err_map, origin="lower", cmap="magma")
        fig.colorbar(im, ax=ax, label="|ΔIR|")
        ax.set_title("Sink IR-drop error map")
        ax.set_xlabel("ix")
        ax.set_ylabel("iy")
        p = out_dir / "ir_error_heatmap.png"
        fig.tight_layout()
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(str(p))

    return written
