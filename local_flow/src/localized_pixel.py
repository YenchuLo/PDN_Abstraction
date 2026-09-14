"""Localized Pixel-R: same star topology, per-cell (Rx, Ry, Rz).

Shared EW/NS edge between cells a,b uses series half-arms:
  g = 1 / (R_axis[a] + R_axis[b])
Pad–sink: g = 1 / via_series_r (Rz[k], plus Rx/Ry stubs if via_stub=rxry).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np

from _bootstrap import ensure_paths

ensure_paths()

from partitions import (  # noqa: E402
    RegionPartition,
    build_partition,
    expand_region_params,
    partition_from_fields,
)
from pixel_r import PixelRModel, build_pixel_model, pixel_model_from_fields  # noqa: E402
from ports import PortSet, via_series_r  # noqa: E402

LOCALIZED_TOPOLOGY = "localized_star_half_arm"


@dataclass
class LocalizedPixelModel:
    """Star topology + region partition; R values live outside (fit stage)."""

    ports: PortSet
    star: PixelRModel
    partition: RegionPartition

    @property
    def n_pads(self) -> int:
        return self.star.n_pads

    @property
    def n_sinks(self) -> int:
        return self.star.n_sinks

    @property
    def n_ports(self) -> int:
        return self.star.n_ports

    @property
    def n_regions(self) -> int:
        return self.partition.n_regions

    def to_fields(self) -> Dict[str, Any]:
        fields = self.star.to_fields()
        fields["topology"] = LOCALIZED_TOPOLOGY
        fields["partition"] = self.partition.to_fields()
        return fields


def build_localized_model(ports: PortSet, block_size: int) -> LocalizedPixelModel:
    star = build_pixel_model(ports)
    part = build_partition(ports, block_size)
    return LocalizedPixelModel(ports=ports, star=star, partition=part)


def localized_model_from_fields(
    ports: PortSet, fields: Dict[str, Any]
) -> LocalizedPixelModel:
    star_fields = dict(fields)
    star_fields["topology"] = "star_half_arm"
    star = pixel_model_from_fields(ports, star_fields)
    part = partition_from_fields(fields["partition"])
    return LocalizedPixelModel(ports=ports, star=star, partition=part)


def build_Gs(
    model: LocalizedPixelModel,
    Rx: Sequence[float],
    Ry: Sequence[float],
    Rz: Sequence[float],
) -> np.ndarray:
    """
    Dense port admittance with per-cell half-arm resistances.

    ``Rx``, ``Ry``, ``Rz`` are length ``n_sinks`` (per-cell).
    """
    star = model.star
    n_p = star.n_pads
    n_s = star.n_sinks
    Rx_a = np.asarray(Rx, dtype=float).ravel()
    Ry_a = np.asarray(Ry, dtype=float).ravel()
    Rz_a = np.asarray(Rz, dtype=float).ravel()
    if Rx_a.size != n_s or Ry_a.size != n_s or Rz_a.size != n_s:
        raise ValueError(
            f"per-cell R length mismatch: Rx={Rx_a.size} Ry={Ry_a.size} "
            f"Rz={Rz_a.size} vs n_sinks={n_s}"
        )
    if np.any(Rx_a <= 0) or np.any(Ry_a <= 0) or np.any(Rz_a <= 0):
        raise ValueError("Rx, Ry, Rz must be positive")

    n = n_p + n_s
    G = np.zeros((n, n), dtype=float)

    def stamp(i: int, j: int, g: float) -> None:
        G[i, i] += g
        G[j, j] += g
        G[i, j] -= g
        G[j, i] -= g

    ew = star.ew_shared or star.ew_edges
    ns = star.ns_shared or star.ns_edges
    for a, b in ew:
        a_i, b_i = int(a), int(b)
        stamp(n_p + a_i, n_p + b_i, 1.0 / (Rx_a[a_i] + Rx_a[b_i]))
    for a, b in ns:
        a_i, b_i = int(a), int(b)
        stamp(n_p + a_i, n_p + b_i, 1.0 / (Ry_a[a_i] + Ry_a[b_i]))
    for p, s in enumerate(star.pad_attach):
        s_i = int(s)
        r_via = via_series_r(
            model.ports, float(Rx_a[s_i]), float(Ry_a[s_i]), float(Rz_a[s_i]), p, s_i
        )
        stamp(p, n_p + s_i, 1.0 / r_via)

    return 0.5 * (G + G.T)


def build_Gs_from_regions(
    model: LocalizedPixelModel,
    Rx_r: Sequence[float],
    Ry_r: Sequence[float],
    Rz_r: Sequence[float],
) -> np.ndarray:
    Rx, Ry, Rz = expand_region_params(model.partition, Rx_r, Ry_r, Rz_r)
    return build_Gs(model, Rx, Ry, Rz)
