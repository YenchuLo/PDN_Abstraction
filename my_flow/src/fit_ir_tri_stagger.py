"""IR fit for staggered mid-layer model (global Rx, Ry, Rz)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from _bootstrap import ensure_paths

ensure_paths()

from fit_rxryrz import FitRxRyRzResult, fit_rxryrz_ir  # noqa: E402
from ports_dual import DualPortSet  # noqa: E402
from tri_stagger import TriStaggerModel, build_Gs_tri_stagger  # noqa: E402


@dataclass
class FitTriStaggerResult:
    Rx: float
    Ry: float
    Rz: float
    relative_ir_error: float
    e_worst: float
    e_ir_by_stimulus: Dict[str, float]
    heldout_e_worst: float
    heldout_e_ir_by_stimulus: Dict[str, float]
    Gs: np.ndarray
    e_ir_x0: float
    polish: bool
    success: bool
    message: str

    # Back-compat aliases (old α JSON keys)
    @property
    def alpha_x(self) -> float:
        return self.Rx

    @property
    def alpha_y(self) -> float:
        return self.Ry

    @property
    def alpha_via(self) -> float:
        return self.Rz

    @property
    def e_ir_extract_only(self) -> float:
        return self.e_ir_x0

    @property
    def e_ir_after_alpha(self) -> float:
        return self.e_worst


def fit_tri_stagger_ir(
    Gprime: np.ndarray,
    model: TriStaggerModel,
    ports: DualPortSet,
    *,
    seed: int = 0,
    heldout_seed: int = 1,
    polish: bool = True,
    bounds: Optional[Tuple[float, float]] = None,
    via_bounds: Optional[Tuple[float, float]] = None,
) -> FitTriStaggerResult:
    del polish, via_bounds

    def build_Gs(Rx: float, Ry: float, Rz: float) -> np.ndarray:
        return build_Gs_tri_stagger(model, Rx=Rx, Ry=Ry, Rz=Rz)

    fit: FitRxRyRzResult = fit_rxryrz_ir(
        Gprime,
        ports,
        build_Gs,
        seed=seed,
        heldout_seed=heldout_seed,
        bounds=bounds,
    )
    return FitTriStaggerResult(
        Rx=fit.Rx,
        Ry=fit.Ry,
        Rz=fit.Rz,
        relative_ir_error=fit.relative_ir_error,
        e_worst=fit.e_worst,
        e_ir_by_stimulus=fit.e_ir_by_stimulus,
        heldout_e_worst=fit.heldout_e_worst,
        heldout_e_ir_by_stimulus=fit.heldout_e_ir_by_stimulus,
        Gs=fit.Gs,
        e_ir_x0=fit.e_ir_x0,
        polish=True,
        success=fit.success,
        message=fit.message,
    )
