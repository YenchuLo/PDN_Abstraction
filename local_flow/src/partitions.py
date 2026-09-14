"""Pad-lattice region partitions for localized Pixel-R.

``block_size`` (CLI --block K):
  K=0  → one global region (uniform sanity)
  K=1  → one region per cell
  K>=2 → K×K pad blocks; remainder cells form smaller edge blocks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from _bootstrap import ensure_paths

ensure_paths()

from ports import PortSet  # noqa: E402


@dataclass
class RegionPartition:
    """Maps each sink/pad cell index to a region id."""

    block_size: int
    n_cells: int
    n_regions: int
    cell_to_region: List[int]
    region_cells: List[List[int]] = field(default_factory=list)
    region_bbox: List[Tuple[int, int, int, int]] = field(default_factory=list)

    def to_fields(self) -> Dict[str, Any]:
        return {
            "block_size": int(self.block_size),
            "n_cells": int(self.n_cells),
            "n_regions": int(self.n_regions),
            "cell_to_region": [int(x) for x in self.cell_to_region],
            "region_cells": [[int(c) for c in cells] for cells in self.region_cells],
            "region_bbox": [
                [int(a), int(b), int(c), int(d)] for a, b, c, d in self.region_bbox
            ],
        }


def partition_from_fields(fields: Dict[str, Any]) -> RegionPartition:
    return RegionPartition(
        block_size=int(fields["block_size"]),
        n_cells=int(fields["n_cells"]),
        n_regions=int(fields["n_regions"]),
        cell_to_region=[int(x) for x in fields["cell_to_region"]],
        region_cells=[[int(c) for c in cells] for cells in fields.get("region_cells", [])],
        region_bbox=[
            (int(a), int(b), int(c), int(d))
            for a, b, c, d in fields.get("region_bbox", [])
        ],
    )


def build_partition(ports: PortSet, block_size: int) -> RegionPartition:
    """Partition the (ix, iy) pad lattice into regions of size ``block_size``.

    ``block_size == 0`` means a single global region.
    """
    n = ports.n_sinks
    if n == 0:
        raise ValueError("need at least one sink cell")
    if block_size < 0:
        raise ValueError(f"block_size must be >= 0, got {block_size}")

    cells = ports.cells
    if block_size == 0:
        return RegionPartition(
            block_size=0,
            n_cells=n,
            n_regions=1,
            cell_to_region=[0] * n,
            region_cells=[list(range(n))],
            region_bbox=[(0, 0, ports.nx - 1, ports.ny - 1)],
        )

    K = int(block_size)
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for k, c in enumerate(cells):
        bx = (c.ix // K) * K
        by = (c.iy // K) * K
        buckets.setdefault((bx, by), []).append(k)

    keys = sorted(buckets.keys(), key=lambda t: (t[1], t[0]))
    cell_to_region = [0] * n
    region_cells: List[List[int]] = []
    region_bbox: List[Tuple[int, int, int, int]] = []
    for rid, key in enumerate(keys):
        members = sorted(buckets[key])
        region_cells.append(members)
        for k in members:
            cell_to_region[k] = rid
        ixs = [cells[k].ix for k in members]
        iys = [cells[k].iy for k in members]
        region_bbox.append((min(ixs), min(iys), max(ixs), max(iys)))

    return RegionPartition(
        block_size=K,
        n_cells=n,
        n_regions=len(region_cells),
        cell_to_region=cell_to_region,
        region_cells=region_cells,
        region_bbox=region_bbox,
    )


def expand_region_params(
    partition: RegionPartition,
    Rx_r: Sequence[float],
    Ry_r: Sequence[float],
    Rz_r: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Broadcast per-region R to per-cell arrays."""
    n = partition.n_cells
    nr = partition.n_regions
    if len(Rx_r) != nr or len(Ry_r) != nr or len(Rz_r) != nr:
        raise ValueError(
            f"region R length mismatch: got {len(Rx_r)},{len(Ry_r)},{len(Rz_r)} "
            f"vs n_regions={nr}"
        )
    Rx = np.empty(n, dtype=float)
    Ry = np.empty(n, dtype=float)
    Rz = np.empty(n, dtype=float)
    for k, rid in enumerate(partition.cell_to_region):
        Rx[k] = float(Rx_r[rid])
        Ry[k] = float(Ry_r[rid])
        Rz[k] = float(Rz_r[rid])
    return Rx, Ry, Rz
