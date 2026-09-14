"""IR fit for tri-square: 6 R params (per-sheet Rx/Ry + two vias)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from _bootstrap import ensure_paths

ensure_paths()

from fit_rxryrz import FitLogRParamsResult, fit_log_r_params_ir  # noqa: E402
from ports_dual import DualPortSet  # noqa: E402
from tri_square import TriSquareModel, build_Gs_tri_square  # noqa: E402

TRI_SQUARE_PARAM_NAMES = ["Rx_u", "Ry_u", "Rx_l", "Ry_l", "Rz_pad", "Rz_ul"]


@dataclass
class FitTriSquareResult:
    Rx_u: float
    Ry_u: float
    Rx_l: float
    Ry_l: float
    Rz_pad: float
    Rz_ul: float
    relative_ir_error: float
    e_worst: float
    e_ir_by_stimulus: Dict[str, float]
    heldout_e_worst: float
    heldout_e_ir_by_stimulus: Dict[str, float]
    Gs: any
    e_ir_x0: float
    polish: bool
    success: bool
    message: str

    # Back-compat aliases
    @property
    def Rx(self) -> float:
        return self.Rx_u

    @property
    def Ry(self) -> float:
        return self.Ry_u

    @property
    def Rz(self) -> float:
        return self.Rz_pad

    @property
    def alpha_xu(self) -> float:
        return self.Rx_u

    @property
    def alpha_yu(self) -> float:
        return self.Ry_u

    @property
    def alpha_xl(self) -> float:
        return self.Rx_l

    @property
    def alpha_yl(self) -> float:
        return self.Ry_l

    @property
    def alpha_via(self) -> float:
        return self.Rz_ul

    @property
    def e_ir_extract_only(self) -> float:
        return self.e_ir_x0

    @property
    def e_ir_after_alpha(self) -> float:
        return self.e_worst


def fit_tri_square_ir(
    Gprime,
    model: TriSquareModel,
    ports: DualPortSet,
    *,
    seed: int = 0,
    heldout_seed: int = 1,
    polish: bool = True,
    bounds: Optional[Tuple[float, float]] = None,
    via_bounds: Optional[Tuple[float, float]] = None,
) -> FitTriSquareResult:
    del polish, via_bounds

    def build_Gs(Rx_u, Ry_u, Rx_l, Ry_l, Rz_pad, Rz_ul):
        return build_Gs_tri_square(
            model,
            Rx_u=Rx_u,
            Ry_u=Ry_u,
            Rx_l=Rx_l,
            Ry_l=Ry_l,
            Rz_pad=Rz_pad,
            Rz_ul=Rz_ul,
        )

    fit: FitLogRParamsResult = fit_log_r_params_ir(
        Gprime,
        ports,
        build_Gs,
        TRI_SQUARE_PARAM_NAMES,
        seed=seed,
        heldout_seed=heldout_seed,
        bounds=bounds,
        max_nfev=500,
    )
    p = fit.params
    return FitTriSquareResult(
        Rx_u=p["Rx_u"],
        Ry_u=p["Ry_u"],
        Rx_l=p["Rx_l"],
        Ry_l=p["Ry_l"],
        Rz_pad=p["Rz_pad"],
        Rz_ul=p["Rz_ul"],
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
