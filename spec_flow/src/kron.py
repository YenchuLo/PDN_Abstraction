"""Sparse Kron / Schur port reduction."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve


def kron_reduce(
    G: sparse.spmatrix,
    port_idx: np.ndarray,
    internal_idx: np.ndarray,
    *,
    reg: float = 1e-12,
    rhs_batch: Optional[int] = None,
) -> np.ndarray:
    """
    Port Schur complement (Kron reduction):

        G' = G_PP - G_PI G_II^{-1} G_IP

    Uses one SuperLU factorization of G_II. The multi-RHS solve is done in
    column batches so peak memory stays ~O(n_i · batch) instead of O(n_i · n_p),
    which is required for large IBM PG port counts.
    """
    G = G.tocsc()
    n_p = len(port_idx)
    n_i = len(internal_idx)
    if n_p == 0:
        raise ValueError("no ports for Kron reduction")

    Gp = G[port_idx[:, None], port_idx].toarray()
    if n_i == 0:
        return 0.5 * (Gp + Gp.T)

    Gc = G[port_idx[:, None], internal_idx].tocsr()  # (n_p, n_i) sparse
    Gi = G[internal_idx[:, None], internal_idx].tocsc()
    if reg > 0:
        Gi = Gi + sparse.eye(n_i, format="csc") * reg

    # Choose batch so n_i * batch * 8 bytes stays under ~2 GiB (cap at n_p).
    if rhs_batch is None:
        target_bytes = 2 * 1024**3
        rhs_batch = max(1, min(n_p, int(target_bytes / max(8 * n_i, 1))))
    rhs_batch = max(1, int(rhs_batch))

    try:
        lu = splu(Gi)
        solve = lu.solve
    except Exception:
        def solve(rhs: np.ndarray) -> np.ndarray:  # type: ignore[misc]
            X = spsolve(Gi, rhs)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return X

    Gprime = np.array(Gp, dtype=float, copy=True)
    for start in range(0, n_p, rhs_batch):
        end = min(start + rhs_batch, n_p)
        # Columns of G_IP = rows of Gc → RHS shape (n_i, batch)
        rhs = Gc[start:end, :].T
        if sparse.issparse(rhs):
            rhs = rhs.toarray()
        rhs = np.asarray(rhs, dtype=float, order="F")
        X = solve(rhs)  # (n_i, batch)
        # Gprime[:, start:end] -= Gc @ X  (Gc stays sparse)
        Gprime[:, start:end] -= Gc @ X

    return 0.5 * (Gprime + Gprime.T)


def grounded_eigh(G: np.ndarray, *, shift: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full symmetric eigendecomposition with a small shift to lift the Laplacian
    nullspace. Spec fitting needs the full spectrum, so this stays dense eigh.
    """
    A = np.asarray(G, dtype=float)
    A = 0.5 * (A + A.T) + np.eye(A.shape[0], dtype=float) * shift
    w, Q = np.linalg.eigh(A, UPLO="L")
    return w, Q
