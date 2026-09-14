#!/usr/bin/env python3
"""Plot TSMC virtual Pixel-R region lattice (pads / sinks / currents)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_artifacts import load_ports  # noqa: E402
from ports import PORT_MODE_TSMC_VIRTUAL, PortSet  # noqa: E402


def plot_virtual_grid(ports: PortSet, out_dir: str | Path) -> List[str]:
    """
    Write PNGs for the virtual region lattice:

    - ``tsmc_virtual_sinks_current.png`` — sink sites colored by |I|
    - ``tsmc_virtual_pads.png`` — co-located virtual top pads
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    xs = np.array([c.x for c in ports.cells], dtype=float)
    ys = np.array([c.y for c in ports.cells], dtype=float)
    cur = np.array([c.current for c in ports.cells], dtype=float)
    mag = np.abs(cur)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        xs,
        ys,
        c=mag,
        s=36,
        cmap="viridis",
        edgecolors="k",
        linewidths=0.3,
    )
    fig.colorbar(sc, ax=ax, label="|I_sink| (A)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"TSMC virtual Pixel-R sinks ({ports.nx}×{ports.ny}, pitch={ports.cell_size:g})"
    )
    ax.grid(True, alpha=0.3)
    path = out / "tsmc_virtual_sinks_current.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        xs,
        ys,
        c="tab:red",
        s=28,
        marker="^",
        label="virtual pad (top V)",
        zorder=3,
    )
    ax.scatter(
        xs,
        ys,
        c="tab:blue",
        s=18,
        marker="o",
        label="region VDD_PORT sink",
        zorder=2,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("TSMC virtual Pixel-R pad / sink lattice")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    path = out / "tsmc_virtual_pads.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    return written


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Plot TSMC virtual Pixel-R region grid")
    p.add_argument(
        "out",
        type=Path,
        help="Component OUT dir with ports.json (e.g. .../comp1)",
    )
    args = p.parse_args(argv)
    ports, _ = load_ports(args.out)
    if ports.port_mode != PORT_MODE_TSMC_VIRTUAL:
        print(
            f"warning: port_mode={ports.port_mode!r} (expected {PORT_MODE_TSMC_VIRTUAL})",
            file=sys.stderr,
        )
    paths = plot_virtual_grid(ports, args.out)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
