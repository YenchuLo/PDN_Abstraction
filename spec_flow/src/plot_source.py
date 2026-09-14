#!/usr/bin/env python3
"""Spatial magnitude maps of Pixel-R / dual-port voltage pads and current sinks.

Reads ``ports.json`` (lumped Vpad / Isink attached to the reduced network), not
raw SPICE instance lists. Writes two PNGs:

  - voltage_sources_spatial.png  — pad voltage magnitude
  - current_sinks_spatial.png    — sink |I| magnitude
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def _resolve_net_dir(out_dir: Path) -> Path:
    """Directory containing ``net.npz`` (``out`` or its parent for ``compk/``)."""
    out = Path(out_dir)
    if (out / "net.npz").is_file():
        return out
    parent = out.parent
    if (parent / "net.npz").is_file():
        return parent
    return out


def _find_comp_dirs(out_dir: Path, comp: Optional[int]) -> List[Path]:
    """Return component dirs that contain ports.json."""
    out = Path(out_dir)
    if (out / "ports.json").is_file():
        return [out]
    if comp is not None:
        d = out / f"comp{comp}"
        if not (d / "ports.json").is_file():
            raise FileNotFoundError(f"missing ports.json under {d}")
        return [d]
    comps = sorted(
        p for p in out.glob("comp*") if p.is_dir() and (p / "ports.json").is_file()
    )
    if not comps:
        raise FileNotFoundError(
            f"no ports.json under {out} or {out}/comp*/ "
            "(run stage02 / ports stage first)"
        )
    return comps


def _cell_center_xy(
    bbox: Tuple[float, float, float, float],
    cell_size: float,
    ix: int,
    iy: int,
) -> Tuple[float, float]:
    """Geometric mid-point of grid cell (ix, iy), matching pixel_geometry_3d."""
    xmin, ymin, _, _ = bbox
    pitch = float(cell_size)
    return xmin + (ix + 0.5) * pitch, ymin + (iy + 0.5) * pitch


def _centers_from_indices(
    bbox: Tuple[float, float, float, float],
    cell_size: float,
    ix: np.ndarray,
    iy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i, j in zip(ix, iy):
        if int(i) < 0 or int(j) < 0:
            xs.append(np.nan)
            ys.append(np.nan)
        else:
            cx, cy = _cell_center_xy(bbox, cell_size, int(i), int(j))
            xs.append(cx)
            ys.append(cy)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _load_port_sources(comp_dir: Path) -> dict:
    """Load pad / sink magnitudes; place both at geometric cell centers."""
    payload = json.loads((comp_dir / "ports.json").read_text())
    bbox = tuple(float(x) for x in payload["bbox"])
    nx = int(payload["nx"])
    ny = int(payload["ny"])
    cell_size = float(payload.get("cell_size") or payload.get("pitch_bot") or 0.0)
    if cell_size <= 0:
        raise ValueError(f"ports.json under {comp_dir} missing positive cell_size")

    pad_v = np.asarray(payload["pad_voltages"], dtype=float)
    cells = payload["cells"]
    si = np.asarray([float(c["current"]) for c in cells], dtype=float)
    six = np.asarray([int(c["ix"]) for c in cells], dtype=int)
    siy = np.asarray([int(c["iy"]) for c in cells], dtype=int)

    # Pad lattice indices: explicit pad_cell_ids, else 1:1 with sinks.
    pad_cell_ids = payload.get("pad_cell_ids")
    if pad_cell_ids and len(pad_cell_ids) == len(pad_v):
        pix = np.asarray([int(p[0]) for p in pad_cell_ids], dtype=int)
        piy = np.asarray([int(p[1]) for p in pad_cell_ids], dtype=int)
    elif len(pad_v) == len(six):
        pix, piy = six.copy(), siy.copy()
    else:
        pix = np.full(len(pad_v), -1, dtype=int)
        piy = np.full(len(pad_v), -1, dtype=int)

    px, py = _centers_from_indices(bbox, cell_size, pix, piy)
    sx, sy = _centers_from_indices(bbox, cell_size, six, siy)

    return {
        "pad_x": px,
        "pad_y": py,
        "pad_v": pad_v,
        "pad_ix": pix,
        "pad_iy": piy,
        "sink_x": sx,
        "sink_y": sy,
        "sink_i": si,
        "sink_ix": six,
        "sink_iy": siy,
        "bbox": bbox,
        "nx": nx,
        "ny": ny,
    }


def _style_axes(ax, xmin, ymin, xmax, ymax, title: str) -> None:
    ax.set_aspect("equal", adjustable="box")
    pad_x = 0.02 * (xmax - xmin + 1.0)
    pad_y = 0.02 * (ymax - ymin + 1.0)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, linewidth=0.5)


def _grid_map(
    ix: np.ndarray,
    iy: np.ndarray,
    values: np.ndarray,
    nx: int,
    ny: int,
) -> np.ndarray:
    """Place port values on an ny×nx lattice (NaN where no port)."""
    grid = np.full((ny, nx), np.nan, dtype=float)
    for i, j, v in zip(ix, iy, values):
        if 0 <= int(i) < nx and 0 <= int(j) < ny:
            grid[int(j), int(i)] = float(v)
    return grid


def _imshow_grid(ax, grid: np.ndarray, bbox, cmap: str):
    xmin, ymin, xmax, ymax = bbox
    masked = np.ma.masked_invalid(grid)
    return ax.imshow(
        masked,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",
    )


def plot_voltage_sources(
    x: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    pad_ix: np.ndarray,
    pad_iy: np.ndarray,
    bbox: Tuple[float, float, float, float],
    nx: int,
    ny: int,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xmin, ymin, xmax, ymax = bbox
    mag = np.abs(v)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)

    ax = axes[0]
    sc = ax.scatter(
        x,
        y,
        c=mag,
        s=48,
        cmap="plasma",
        edgecolors="k",
        linewidths=0.35,
        zorder=3,
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("|V| (port)")
    _style_axes(
        ax,
        xmin,
        ymin,
        xmax,
        ymax,
        f"Voltage pads (scatter by |V|)\nn={len(x)}  "
        f"min={float(np.nanmin(mag)):.4g}  max={float(np.nanmax(mag)):.4g}",
    )

    ax = axes[1]
    if np.any(pad_ix >= 0):
        grid = _grid_map(pad_ix, pad_iy, mag, nx, ny)
        im = _imshow_grid(ax, grid, bbox, "plasma")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("|V| / cell")
        n_active = int(np.sum(~np.isnan(grid)))
        _style_axes(
            ax,
            xmin,
            ymin,
            xmax,
            ymax,
            f"Voltage-pad magnitude grid ({nx}×{ny}, {n_active} pads)",
        )
    else:
        sc = ax.scatter(
            x,
            y,
            c=mag,
            s=64,
            cmap="plasma",
            marker="^",
            edgecolors="k",
            linewidths=0.35,
            zorder=3,
        )
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("|V| (port)")
        _style_axes(
            ax,
            xmin,
            ymin,
            xmax,
            ymax,
            f"Voltage pads (no lattice indices, n={len(x)})",
        )

    fig.suptitle("Pixel-R / port network voltage sources", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_current_sinks(
    x: np.ndarray,
    y: np.ndarray,
    i_val: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    bbox: Tuple[float, float, float, float],
    nx: int,
    ny: int,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mag = np.abs(i_val)
    xmin, ymin, xmax, ymax = bbox
    tot = float(mag.sum())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)

    ax = axes[0]
    sc = ax.scatter(
        x,
        y,
        c=mag,
        s=np.clip(20.0 * mag / (np.median(mag) + 1e-30), 12.0, 90.0),
        cmap="viridis",
        edgecolors="k",
        linewidths=0.3,
        zorder=3,
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("|I| (port, A)")
    _style_axes(
        ax,
        xmin,
        ymin,
        xmax,
        ymax,
        f"Current sinks (scatter by |I|)\nn={len(x)}  "
        f"Sigma|I|={tot:.4g}",
    )

    ax = axes[1]
    grid = _grid_map(ix, iy, mag, nx, ny)
    im = _imshow_grid(ax, grid, bbox, "viridis")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("|I| / cell (A)")
    _style_axes(
        ax,
        xmin,
        ymin,
        xmax,
        ymax,
        f"Current-sink magnitude grid ({nx}×{ny})\nSigma|I|={tot:.4g}",
    )

    fig.suptitle("Pixel-R / port network current sinks", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "out_dir",
        type=Path,
        help="Flow OUT/ or compN/ with ports.json "
        "(e.g. spec_flow/outputs/ibmpg2_nauto_ir)",
    )
    p.add_argument(
        "--comp",
        type=int,
        default=None,
        help="Component index when out_dir is the flow root (default: all comps)",
    )
    p.add_argument(
        "--png-dir",
        type=Path,
        default=None,
        help="Directory for PNGs (default: each comp dir with ports.json)",
    )
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    net_dir = _resolve_net_dir(out)
    comp_dirs = _find_comp_dirs(out, args.comp)

    for comp_dir in comp_dirs:
        src = _load_port_sources(comp_dir)
        if args.png_dir is not None:
            png_dir = args.png_dir
        elif (out / "ports.json").is_file():
            png_dir = comp_dir
        else:
            # Flow OUT/: keep PNGs next to net.npz (previous plot_source location).
            png_dir = net_dir
        # Prefix when several comps share one png dir.
        prefix = f"{comp_dir.name}_" if len(comp_dirs) > 1 else ""
        v_png = png_dir / f"{prefix}voltage_sources_spatial.png"
        i_png = png_dir / f"{prefix}current_sinks_spatial.png"

        plot_voltage_sources(
            src["pad_x"],
            src["pad_y"],
            src["pad_v"],
            src["pad_ix"],
            src["pad_iy"],
            src["bbox"],
            src["nx"],
            src["ny"],
            v_png,
        )
        plot_current_sinks(
            src["sink_x"],
            src["sink_y"],
            src["sink_i"],
            src["sink_ix"],
            src["sink_iy"],
            src["bbox"],
            src["nx"],
            src["ny"],
            i_png,
        )

        pv, si = src["pad_v"], src["sink_i"]
        print(f"wrote {v_png}")
        print(f"wrote {i_png}")
        print(
            f"summary [{comp_dir.name}]: V={len(pv)} "
            f"(min|V|={float(np.nanmin(np.abs(pv))):.6g}, "
            f"max|V|={float(np.nanmax(np.abs(pv))):.6g}); "
            f"I={len(si)} grid={src['nx']}x{src['ny']}; "
            f"Sigma|I|={float(np.abs(si).sum()):.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
