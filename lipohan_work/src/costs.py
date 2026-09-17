"""Cost-function residuals for Pixel-R (Rx, Ry, Rz) fitting.

Training costs (priority five): eigen, frobenius, rayleigh, voltage, minimax.
Everything else is available as a named cost *and* as an evaluation metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from _bootstrap import ensure_paths

ensure_paths()

from correlate import _factor_sink_block, build_stimuli, solve_mixed_bc  # noqa: E402
from fit_eigen import proj_eigs  # noqa: E402
from kron import grounded_eigh  # noqa: E402
from pixel_r import PixelRModel, build_Gs  # noqa: E402
from ports import PortSet  # noqa: E402

EPS = 1e-12

# Names used by --costs / reports. Aliases map to the canonical name.
PRIORITY_COSTS = ("eigen", "frobenius", "rayleigh", "voltage", "minimax")

COST_ALIASES: Dict[str, str] = {
    "j_lambda": "eigen",
    "j_l": "eigen",
    "lambda": "eigen",
    "j_lambda_rel": "eigen_rel",
    "j_lambda_rel_floor": "eigen_rel_floor",
    "j_f": "frobenius",
    "f": "frobenius",
    "j_f_rel": "frobenius_rel",
    "j_w": "weighted",
    "weighted_frobenius": "weighted",
    "j_rq": "rayleigh",
    "rq": "rayleigh",
    "j_rq_rel": "rayleigh_rel",
    "j_rq_rel_floor": "rayleigh_rel_floor",
    "j_v": "voltage",
    "v": "voltage",
    "j_v_rel": "voltage_rel",
    "j_v_rel_floor": "voltage_rel_floor",
    "j_inf": "minimax",
    "j_infty": "minimax",
    "inf": "minimax",
    "infty": "minimax",
    "j_inf_floor": "minimax_floor",
    "j_p": "pnorm",
    "j_r": "reff",
    "effective_resistance": "reff",
    "j_2": "spectral",
    "spectral_norm": "spectral",
}

VECTOR_COSTS = {
    "eigen",
    "eigen_rel",
    "eigen_rel_floor",
    "frobenius",
    "frobenius_rel",
    "weighted",
    "rayleigh",
    "rayleigh_rel",
    "rayleigh_rel_floor",
    "voltage",
    "voltage_rel",
    "voltage_rel_floor",
    "pnorm",
    "reff",
    "hybrid",
}
SCALAR_COSTS = {"minimax", "minimax_floor", "pnorm_rel_floor", "spectral"}
STABILIZED_COSTS = (
    "eigen_rel_floor",
    "rayleigh_rel_floor",
    "voltage_rel_floor",
    "minimax_floor",
    "pnorm_rel_floor",
)


def canonical_cost(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "_")
    key = key.replace("λ", "lambda").replace(",", "")
    return COST_ALIASES.get(key, key)


def parse_cost_list(raw: str | Sequence[str] | None) -> List[str]:
    if raw is None:
        return list(PRIORITY_COSTS)
    if isinstance(raw, str):
        parts = [p for p in raw.replace(";", ",").split(",") if p.strip()]
    else:
        parts = [str(p) for p in raw]
    if not parts or parts == ["priority"]:
        return list(PRIORITY_COSTS)
    if parts == ["all"]:
        return list(PRIORITY_COSTS) + [
            "eigen_rel",
            "eigen_rel_floor",
            "frobenius_rel",
            "weighted",
            "rayleigh_rel",
            "rayleigh_rel_floor",
            "voltage_rel",
            "voltage_rel_floor",
            "pnorm",
            "pnorm_rel_floor",
            "reff",
            "spectral",
            "hybrid",
            "minimax_floor",
        ]
    out: List[str] = []
    seen = set()
    for p in parts:
        c = canonical_cost(p)
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


@dataclass
class FitContext:
    Gprime: np.ndarray
    model: PixelRModel
    ports: PortSet
    seed: int = 0
    training_stimuli: Optional[Sequence[Tuple[str, np.ndarray]]] = None
    p_norm: float = 8.0
    hybrid_alpha: float = 1.0
    hybrid_beta: float = 1.0
    hybrid_gamma: float = 1.0
    rq_basis: str = "mixed"  # eigen | random | mixed
    n_rq_rand: int = 32
    n_rq_eigen: int = 32
    w_diag: float = 5.0
    w_nbr: float = 3.0
    w_via: float = 3.0
    w_far: float = 0.05
    voltage_rel_floor_fraction: float = 0.02
    spectral_rel_floor_fraction: float = 1e-6
    lam_M: np.ndarray = field(init=False)
    Q: np.ndarray = field(init=False)
    V_pad: np.ndarray = field(init=False)
    vref: float = field(init=False)
    lu_m: object = field(init=False)
    Gsp_m: np.ndarray = field(init=False)
    stimulus_names: List[str] = field(init=False)
    I_sinks: List[np.ndarray] = field(init=False)
    drops_m: List[np.ndarray] = field(init=False)
    target_v: np.ndarray = field(init=False)
    voltage_rel_denom: np.ndarray = field(init=False)
    X_rq: np.ndarray = field(init=False)
    rq_target: np.ndarray = field(init=False)
    rq_rel_denom: np.ndarray = field(init=False)
    eigen_rel_denom: np.ndarray = field(init=False)
    W: np.ndarray = field(init=False)
    reff_M: np.ndarray = field(init=False)
    tri: Tuple[np.ndarray, np.ndarray] = field(init=False)
    r_scale: float = field(init=False)

    def __post_init__(self) -> None:
        G = np.asarray(self.Gprime, dtype=float)
        self.Gprime = 0.5 * (G + G.T)
        self.lam_M, self.Q = grounded_eigh(self.Gprime)
        eigen_floor = max(
            self.spectral_rel_floor_fraction * float(np.max(np.abs(self.lam_M))),
            EPS,
        )
        self.eigen_rel_denom = np.abs(self.lam_M) + eigen_floor
        g_scale = max(float(np.median(np.diag(self.Gprime))), 1e-6)
        self.r_scale = 1.0 / g_scale
        self.V_pad = np.asarray(self.ports.pad_voltages, dtype=float)
        self.vref = float(np.mean(self.V_pad))
        n_p = self.ports.n_pads
        self.lu_m, self.Gsp_m = _factor_sink_block(self.Gprime, n_p)
        if self.training_stimuli is None:
            stimuli = build_stimuli(self.ports, seed=self.seed)
        else:
            stimuli = [
                (str(name), np.asarray(current, dtype=float).copy())
                for name, current in self.training_stimuli
            ]
        if not stimuli:
            raise ValueError("training_stimuli must contain at least one current pattern")
        for name, current in stimuli:
            if np.asarray(current).shape != (self.ports.n_sinks,):
                raise ValueError(
                    f"stimulus {name!r} shape {np.asarray(current).shape} != "
                    f"({self.ports.n_sinks},)"
                )
        self.stimulus_names = [n for n, _ in stimuli]
        self.I_sinks = [np.asarray(I, dtype=float) for _, I in stimuli]
        self.drops_m = []
        for I in self.I_sinks:
            Vm = solve_mixed_bc(
                self.Gprime, n_p, self.V_pad, I, _lu=self.lu_m, _Gsp=self.Gsp_m
            )
            self.drops_m.append(self.vref - Vm[n_p:])
        self.target_v = np.concatenate(self.drops_m)
        voltage_denominators = []
        for drop in self.drops_m:
            floor = max(
                self.voltage_rel_floor_fraction * float(np.max(np.abs(drop))),
                EPS,
            )
            voltage_denominators.append(np.abs(drop) + floor)
        self.voltage_rel_denom = np.concatenate(voltage_denominators)
        self.X_rq = _rq_basis(
            self.Q,
            self.Gprime,
            self.I_sinks,
            n_pads=n_p,
            basis=self.rq_basis,
            n_rand=self.n_rq_rand,
            n_eigen=self.n_rq_eigen,
            seed=self.seed,
        )
        self.rq_target = _rayleigh(self.Gprime, self.X_rq)
        rq_floor = max(
            self.spectral_rel_floor_fraction
            * float(np.max(np.abs(self.rq_target))),
            EPS,
        )
        self.rq_rel_denom = np.abs(self.rq_target) + rq_floor
        self.W = _weight_matrix(
            self.model,
            self.Gprime.shape[0],
            w_diag=self.w_diag,
            w_nbr=self.w_nbr,
            w_via=self.w_via,
            w_far=self.w_far,
        )
        self.reff_M = _reff_matrix(self.Gprime)
        n = self.Gprime.shape[0]
        self.tri = np.triu_indices(n, k=1)

    def build_Gs(self, Rx: float, Ry: float, Rz: float) -> np.ndarray:
        return build_Gs(self.model, float(Rx), float(Ry), float(Rz))

    def build_Gs_log(self, log_theta: np.ndarray) -> np.ndarray:
        Rx, Ry, Rz = (10.0 ** np.asarray(log_theta, dtype=float)).tolist()
        return self.build_Gs(Rx, Ry, Rz)

    def ir_drops(self, Gs: np.ndarray) -> List[np.ndarray]:
        n_p = self.ports.n_pads
        lu_s, Gsp_s = _factor_sink_block(Gs, n_p)
        out: List[np.ndarray] = []
        for I in self.I_sinks:
            Vs = solve_mixed_bc(Gs, n_p, self.V_pad, I, _lu=lu_s, _Gsp=Gsp_s)
            out.append(self.vref - Vs[n_p:])
        return out


def _rq_basis(
    Q: np.ndarray,
    G: np.ndarray,
    I_sinks: Sequence[np.ndarray],
    *,
    n_pads: int,
    basis: str,
    n_rand: int,
    n_eigen: int,
    seed: int,
) -> np.ndarray:
    """Columns are unit test directions in port space."""
    n = Q.shape[0]
    cols: List[np.ndarray] = []
    mode = str(basis).lower()
    if mode in ("eigen", "mixed"):
        k = min(int(n_eigen), n)
        idx = np.unique(np.linspace(0, n - 1, k, dtype=int))
        cols.append(Q[:, idx])
    if mode in ("random", "mixed"):
        rng = np.random.default_rng(seed + 17)
        k = min(int(n_rand), n)
        A = rng.normal(size=(n, k))
        q, _ = np.linalg.qr(A)
        cols.append(q)
    if mode in ("current", "mixed"):
        for I in I_sinks:
            v = np.zeros(n, dtype=float)
            v[n_pads:] = np.asarray(I, dtype=float)
            nrm = np.linalg.norm(v)
            if nrm > EPS:
                cols.append((v / nrm)[:, None])
    if not cols:
        cols.append(Q)
    X = np.concatenate(cols, axis=1)
    # Drop near-duplicates / zeros
    keep: List[int] = []
    for j in range(X.shape[1]):
        col = X[:, j]
        nrm = np.linalg.norm(col)
        if nrm < EPS:
            continue
        col = col / nrm
        X[:, j] = col
        if not keep:
            keep.append(j)
            continue
        overlaps = np.abs(X[:, keep].T @ col)
        if float(np.max(overlaps)) < 0.999:
            keep.append(j)
    return X[:, keep]


def _rayleigh(G: np.ndarray, X: np.ndarray) -> np.ndarray:
    """x_k^T G x_k / x_k^T x_k for unit columns of X."""
    return np.sum(X * (G @ X), axis=0)


def _weight_matrix(
    model: PixelRModel,
    n: int,
    *,
    w_diag: float,
    w_nbr: float,
    w_via: float,
    w_far: float,
) -> np.ndarray:
    W = np.full((n, n), float(w_far), dtype=float)
    np.fill_diagonal(W, float(w_diag))
    n_p = model.n_pads
    for a, b in model.ew_shared or model.ew_edges:
        i, j = n_p + int(a), n_p + int(b)
        W[i, j] = W[j, i] = float(w_nbr)
    for a, b in model.ns_shared or model.ns_edges:
        i, j = n_p + int(a), n_p + int(b)
        W[i, j] = W[j, i] = float(w_nbr)
    for p, s in enumerate(model.pad_attach):
        i, j = int(p), n_p + int(s)
        W[i, j] = W[j, i] = float(w_via)
    return W


def _laplacian_pinv(G: np.ndarray, *, shift: float = 1e-10) -> np.ndarray:
    n = G.shape[0]
    if n <= 1:
        return np.zeros_like(G, dtype=float)
    g = 0.5 * (G + G.T)
    g = g[:-1, :-1] + np.eye(n - 1, dtype=float) * shift
    inv = np.linalg.inv(g)
    P = np.zeros((n, n), dtype=float)
    P[:-1, :-1] = inv
    return P


def _reff_matrix(G: np.ndarray) -> np.ndarray:
    P = _laplacian_pinv(G)
    d = np.diag(P)
    return d[:, None] + d[None, :] - P - P.T


def residual(ctx: FitContext, cost: str, Gs: np.ndarray) -> np.ndarray:
    """Vector residual whose squared 2-norm is the training loss (or a monotone of it)."""
    c = canonical_cost(cost)
    Gm = ctx.Gprime
    if c == "eigen":
        return ctx.lam_M - proj_eigs(ctx.Q, Gs)
    if c == "eigen_rel":
        lam_S = proj_eigs(ctx.Q, Gs)
        return (lam_S - ctx.lam_M) / (np.abs(ctx.lam_M) + EPS)
    if c == "eigen_rel_floor":
        lam_S = proj_eigs(ctx.Q, Gs)
        return (lam_S - ctx.lam_M) / ctx.eigen_rel_denom
    if c == "frobenius":
        return (Gs - Gm).ravel()
    if c == "frobenius_rel":
        denom = float(np.linalg.norm(Gm, ord="fro")) + EPS
        return (Gs - Gm).ravel() / denom
    if c == "weighted":
        return (ctx.W * (Gs - Gm)).ravel()
    if c == "rayleigh":
        return _rayleigh(Gs, ctx.X_rq) - ctx.rq_target
    if c == "rayleigh_rel":
        num = _rayleigh(Gs - Gm, ctx.X_rq)
        return num / (ctx.rq_target + EPS)
    if c == "rayleigh_rel_floor":
        num = _rayleigh(Gs - Gm, ctx.X_rq)
        return num / ctx.rq_rel_denom
    if c == "voltage":
        drops = ctx.ir_drops(Gs)
        return np.concatenate(drops) - ctx.target_v
    if c == "voltage_rel":
        drops = np.concatenate(ctx.ir_drops(Gs))
        return (drops - ctx.target_v) / (np.abs(ctx.target_v) + EPS)
    if c == "voltage_rel_floor":
        drops = np.concatenate(ctx.ir_drops(Gs))
        return (drops - ctx.target_v) / ctx.voltage_rel_denom
    if c == "pnorm":
        e = np.concatenate(ctx.ir_drops(Gs)) - ctx.target_v
        p = float(ctx.p_norm)
        # ||r||_2^2 = sum |e|^p  (up to the usual 1/N later at report time)
        return np.sign(e) * np.power(np.abs(e) + EPS, 0.5 * p)
    if c == "pnorm_rel_floor":
        e = (np.concatenate(ctx.ir_drops(Gs)) - ctx.target_v) / ctx.voltage_rel_denom
        p = float(ctx.p_norm)
        return np.sign(e) * np.power(np.abs(e) + EPS, 0.5 * p)
    if c == "reff":
        R_S = _reff_matrix(Gs)
        i, j = ctx.tri
        return (R_S - ctx.reff_M)[i, j]
    if c == "hybrid":
        r_l = (ctx.lam_M - proj_eigs(ctx.Q, Gs)) / (np.linalg.norm(ctx.lam_M) + EPS)
        r_q = (_rayleigh(Gs, ctx.X_rq) - ctx.rq_target) / (
            np.linalg.norm(ctx.rq_target) + EPS
        )
        drops = np.concatenate(ctx.ir_drops(Gs))
        r_v = (drops - ctx.target_v) / (np.linalg.norm(ctx.target_v) + EPS)
        return np.concatenate(
            [
                np.sqrt(max(ctx.hybrid_alpha, 0.0)) * r_l,
                np.sqrt(max(ctx.hybrid_beta, 0.0)) * r_q,
                np.sqrt(max(ctx.hybrid_gamma, 0.0)) * r_v,
            ]
        )
    raise ValueError(f"cost {cost!r} is scalar or unknown; use scalar_loss()")


def scalar_loss(ctx: FitContext, cost: str, Gs: np.ndarray) -> float:
    c = canonical_cost(cost)
    if c == "minimax":
        rel = residual(ctx, "voltage_rel", Gs)
        return float(np.max(np.abs(rel)))
    if c == "minimax_floor":
        rel = residual(ctx, "voltage_rel_floor", Gs)
        return float(np.max(np.abs(rel)))
    if c == "pnorm_rel_floor":
        rel = residual(ctx, "voltage_rel_floor", Gs)
        return float(np.mean(np.abs(rel) ** float(ctx.p_norm)) ** (1.0 / float(ctx.p_norm)))
    if c == "spectral":
        return float(np.linalg.norm(Gs - ctx.Gprime, ord=2))
    r = residual(ctx, c, Gs)
    return float(np.sum(np.square(r)))


ResidualFn = Callable[[np.ndarray], np.ndarray]
ScalarFn = Callable[[np.ndarray], float]


def make_residual_fn(ctx: FitContext, cost: str) -> ResidualFn:
    c = canonical_cost(cost)
    if c not in VECTOR_COSTS:
        raise ValueError(f"{c} is not a vector cost")

    def fun(log_theta: np.ndarray) -> np.ndarray:
        return residual(ctx, c, ctx.build_Gs_log(log_theta))

    return fun


def make_scalar_fn(ctx: FitContext, cost: str) -> ScalarFn:
    c = canonical_cost(cost)

    def fun(log_theta: np.ndarray) -> float:
        return scalar_loss(ctx, c, ctx.build_Gs_log(log_theta))

    return fun
