#!/usr/bin/env python3
"""Stage 02: C4-overlap pads and full-grid lumped sinks per VDD component."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence, Set, Union

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph import (
    _component_layer_span,
    build_short_map,
    vdd_connected_components,
    vdd_layers,
)
from io_artifacts import (
    NET_META,
    ensure_out,
    load_net,
    save_components,
    save_net,
    save_ports,
)
from ports import (
    DEFAULT_VDD,
    PORT_MODE_TSMC_VIRTUAL,
    VIA_STUB_ZERO,
    UniformCurrentSpec,
    apply_grid_lumping,
    apply_uniform_sink_currents,
    build_ports,
    build_tsmc_region_ports,
    ensure_tsmc_package_node,
    inject_virtual_pads,
    parse_uniform_current_arg,
    remap_ports,
    suggest_legal_chip_pitch,
)
from spice_parser import parse_perturb_layer_arg, perturb_layer_resistors
from tsmc_region_grid import is_virtual_pixel_r_lattice

PerturbLayerSpec = Union[int, str]


def _parse_perturb_layer_cli(raw: Optional[str]) -> Optional[PerturbLayerSpec]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    return parse_perturb_layer_arg(s)


def _apply_grid_perturb(
    out: Path,
    net,
    *,
    layer: PerturbLayerSpec,
    amp: float,
    seed: int,
    cell_size: float,
    layers: Sequence[int],
    roots: Set[str],
    node_map: dict,
    spice_path: str,
) -> dict:
    perturb_meta = perturb_layer_resistors(
        net,
        layer,
        amp,
        cell_size=cell_size,
        seed=seed,
        layers=layers,
        roots=roots,
        node_map=node_map,
    )
    save_net(out, net, spice_path)
    meta_path = out / NET_META
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    meta.update(perturb_meta)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"perturb-layer {perturb_meta['perturb_layer']} unit=grid "
        f"amp={perturb_meta['perturb_amp']} seed={perturb_meta['perturb_seed']} "
        f"cell={perturb_meta['cell_size']:.6g} "
        f"({perturb_meta['nx']}x{perturb_meta['ny']}): "
        f"n={perturb_meta['n_perturbed']} cells_used={perturb_meta['n_cells_used']} "
        f"layers={perturb_meta['perturb_layers']} "
        f"factor=[{perturb_meta['factor_min']:.4g}, {perturb_meta['factor_max']:.4g}] "
        f"mean={perturb_meta['factor_mean']:.4g}"
    )
    return perturb_meta


def _write_pad_occupancy(comp_dir: Path, ports) -> Optional[str]:
    """Grid map: cells with a voltage pad vs sink-only cells."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    occ = np.full((ports.ny, ports.nx), np.nan, dtype=float)
    has = ports.sink_has_pad()
    for k, c in enumerate(ports.cells):
        occ[c.iy, c.ix] = 1.0 if has[k] else 0.0
    path = Path(comp_dir) / "pad_occupancy.png"
    cmap = ListedColormap(["#c8c8c8", "#2a6f97"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax = axes[0]
    im = ax.imshow(occ, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
    cbar.ax.set_yticklabels(["sink-only", "has pad"])
    ax.set_title(f"Pad occupancy  {ports.n_pads}/{ports.n_sinks} cells")
    ax.set_xlabel("ix")
    ax.set_ylabel("iy")

    ax = axes[1]
    xmin, ymin, xmax, ymax = ports.bbox
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    for k, c in enumerate(ports.cells):
        color = "#2a6f97" if has[k] else "#888888"
        marker = "s" if has[k] else "x"
        ax.plot(c.x, c.y, marker, color=color, ms=5 if has[k] else 6, zorder=2)
    locs = ports.pad_locations()
    if locs:
        px, py = zip(*locs)
        ax.scatter(
            px, py, s=18, c="#d9480f", marker="^", zorder=3,
            label="C4 pad",
        )
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"pitch={ports.cell_size:.4g}  max metal={ports.max_metal_pitch:.4g}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.suptitle("Imaginary-grid voltage pads vs current sinks", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _run_tsmc_virtual(
    out: Path,
    net,
    spice_path: str,
    *,
    uniform_spec: UniformCurrentSpec | None = None,
    perturb_layer: Optional[PerturbLayerSpec] = None,
    via_stub: str = VIA_STUB_ZERO,
) -> int:
    """Virtual Pixel-R ports on the TSMC region lattice (single component)."""
    if perturb_layer is not None:
        print(
            "error: --perturb-layer is not supported for TSMC virtual Pixel-R "
            "lattices (IBM multi-track meshes only)",
            file=sys.stderr,
        )
        return 2

    pkg, vdd = ensure_tsmc_package_node(net, vdd=DEFAULT_VDD)
    ports = build_tsmc_region_ports(
        net, vdd=vdd, package_node=pkg, via_stub=via_stub
    )
    uniform_meta: dict = {}
    if uniform_spec is not None:
        try:
            uniform_meta = apply_uniform_sink_currents(ports.cells, uniform_spec)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(
            f"uniform-current mode={uniform_meta['uniform_current_mode']} "
            f"i_each={uniform_meta['i_each']:.6g} "
            f"(n_sinks={uniform_meta['n_sinks']}, "
            f"i_total_before={uniform_meta['i_total_before']:.6g})"
        )
    n_inj = inject_virtual_pads(net, ports, package_node=pkg)
    # Persist augmented net (virtual pads + package R) for Kron / emit.
    save_net(out, net, spice_path)

    short_map = build_short_map(net)
    comps = vdd_connected_components(net, short_map)
    if not comps:
        print(
            "error: no VDD connected component after virtual-pad inject",
            file=sys.stderr,
        )
        return 2

    sink_roots = {short_map.get(c.node, c.node) for c in ports.cells}
    roots = comps[0]
    for cand in comps:
        if sink_roots & cand:
            roots = cand
            break

    comp_dir = out / "comp1"
    ensure_out(comp_dir)
    node_map = dict(short_map)
    ports_k = remap_ports(ports, node_map)
    save_ports(comp_dir, ports_k, node_map)

    try:
        from plot_tsmc_virtual_grid import plot_virtual_grid

        plots = plot_virtual_grid(ports_k, comp_dir)
        print(f"virtual-grid plots: {plots}")
    except Exception as exc:  # noqa: BLE001
        print(f"warning: virtual-grid plots skipped: {exc}", file=sys.stderr)

    top, bot = _component_layer_span(net, short_map, roots)
    layers = vdd_layers(net, short_map, roots)
    entry = {
        "index": 1,
        "path": "comp1",
        "n_nodes": len(roots),
        "top_layer": int(top),
        "bot_layer": int(bot),
        "vdd_layers": list(layers),
        "n_pads": ports_k.n_pads,
        "n_sinks": ports_k.n_sinks,
        "n_ports": ports_k.n_ports,
        "nx": ports_k.nx,
        "ny": ports_k.ny,
        "chip_pitch": ports_k.cell_size,
        "max_metal_pitch": ports_k.max_metal_pitch,
        "grid_to_pad_ratio": None,
        "pad_pitch_target": None,
        "port_mode": PORT_MODE_TSMC_VIRTUAL,
        "package_node": pkg,
        "n_virtual_pad_r": n_inj,
        "n_occupied_bumps": ports_k.n_occupied_bumps,
        "n_open_pads": ports_k.n_open_pads,
        "n_shared_tiles": ports_k.n_shared_tiles,
        "via_stub": ports_k.via_stub,
        "uniform_current": (
            "conserve" if uniform_spec == "conserve" else uniform_spec
        ),
        **uniform_meta,
    }
    summary = {
        "stage": "02_ports",
        "component_index": 1,
        "out": str(comp_dir),
        **{key: entry[key] for key in entry if key != "path"},
        "vdd": ports_k.vdd,
        "metal_pitches": {str(kk): v for kk, v in ports_k.metal_pitches.items()},
    }
    print(json.dumps(summary, indent=2))

    save_components(
        out,
        {
            "n_components": 1,
            "order": "tsmc_virtual_pixel_r",
            "components": [entry],
        },
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build C4-overlap pads and full-grid sinks for each VDD component"
    )
    p.add_argument("out", type=Path, help="Output / work directory (flow OUT root)")
    p.add_argument(
        "chip_pitch",
        type=float,
        nargs="?",
        default=0.0,
        help=(
            "Grid cell side length (layout units). "
            "0 = auto: C4 lattice pitch (median unique-row/col spacing), "
            "clamped to chip_pitch > max VDD metal stripe pitch. "
            "A cell gets a voltage pad iff a real C4 / nonzero V→gnd site "
            "overlaps it (sink-only otherwise), at any pitch. "
            "Pass --full-pads to also attach nearest unused top-metal nodes "
            "to formerly padless cells."
        ),
    )
    p.add_argument(
        "--grid-to-pad-ratio",
        type=float,
        default=1.0,
        help=(
            "Divide C4 lattice pitch by sqrt(ratio) (default 1.0 = match bump "
            "spacing). Ignored if chip_pitch > 0."
        ),
    )
    p.add_argument(
        "--uniform-current",
        nargs="?",
        const="conserve",
        default=None,
        metavar="MODE",
        help=(
            "Same current on every grid sink. "
            "Omit value or pass 'conserve' to share total I equally; "
            "pass a number for fixed per-sink draw in amps (stored as -|I|)."
        ),
    )
    p.add_argument(
        "--perturb-layer",
        default=None,
        metavar="L",
        help=(
            "Grid-cell R perturbation: metal layer int, or 'all' for every metal "
            "in the current VDD rail stack. One U per cell; vias unchanged."
        ),
    )
    p.add_argument(
        "--perturb-amp",
        type=float,
        default=0.2,
        metavar="A",
        help="Half-width of uniform relative perturbation (default 0.2 = ±20%%)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for --perturb-layer (default 0)",
    )
    p.add_argument(
        "--full-pads",
        action="store_true",
        help=(
            "Assign a voltage pad to every grid cell: keep C4-overlap pads, "
            "fill padless cells with the unique nearest unused top-metal VDD node, "
            "and drive all pads at one common voltage (representative of real C4s)."
        ),
    )
    args = p.parse_args(argv)

    pitch_arg = float(args.chip_pitch)
    if pitch_arg < 0:
        print("error: chip_pitch must be >= 0 (0 = auto)", file=sys.stderr)
        return 2
    grid_to_pad_ratio = float(args.grid_to_pad_ratio)
    if grid_to_pad_ratio <= 0:
        print("error: --grid-to-pad-ratio must be positive", file=sys.stderr)
        return 2
    try:
        uniform_spec = parse_uniform_current_arg(args.uniform_current)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        perturb_layer = _parse_perturb_layer_cli(args.perturb_layer)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if perturb_layer is not None and float(args.perturb_amp) < 0:
        print("error: --perturb-amp must be >= 0", file=sys.stderr)
        return 2
    via_stub = VIA_STUB_ZERO
    full_pads = bool(args.full_pads)

    out = ensure_out(args.out)
    net = load_net(out)
    meta_path = out / NET_META
    spice_path = str(out)
    if meta_path.is_file():
        try:
            spice_path = str(
                json.loads(meta_path.read_text()).get("spice_path", spice_path)
            )
        except json.JSONDecodeError:
            pass

    if is_virtual_pixel_r_lattice(net):
        print(
            f"TSMC region lattice detected: virtual Pixel-R ports "
            f"(pitch={ports_pitch_hint(net)})"
        )
        return _run_tsmc_virtual(
            out,
            net,
            spice_path,
            uniform_spec=uniform_spec,
            perturb_layer=perturb_layer,
            via_stub=via_stub,
        )

    short_map = build_short_map(net)
    comps = vdd_connected_components(net, short_map)
    n_comp = len(comps)
    print(
        f"Found {n_comp} VDD connected component(s); "
        f"running topmost-first as comp1..comp{n_comp}"
    )
    if n_comp == 0:
        print("error: no VDD connected components", file=sys.stderr)
        return 2

    # Resolve pitch + apply grid perturb once (shared lattice) before ports.
    node_map0 = dict(short_map)
    roots0 = comps[0]
    chip_pitch = pitch_arg
    if chip_pitch <= 0:
        try:
            chip_pitch, _, _ = suggest_legal_chip_pitch(
                net,
                node_map0,
                roots0,
                grid_to_pad_ratio=grid_to_pad_ratio,
            )
        except ValueError as exc:
            print(f"error: auto pitch failed: {exc}", file=sys.stderr)
            return 2

    if perturb_layer is not None:
        all_roots: Set[str] = set()
        all_layers: Set[int] = set()
        for roots in comps:
            all_roots |= set(roots)
            all_layers.update(int(L) for L in vdd_layers(net, short_map, roots))
        try:
            _apply_grid_perturb(
                out,
                net,
                layer=perturb_layer,
                amp=float(args.perturb_amp),
                seed=int(args.seed),
                cell_size=float(chip_pitch),
                layers=sorted(all_layers),
                roots=all_roots,
                node_map=node_map0,
                spice_path=spice_path,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    index: list[dict] = []
    for k, roots in enumerate(comps, start=1):
        comp_dir = out / f"comp{k}"
        ensure_out(comp_dir)
        node_map = dict(short_map)

        chip_pitch_k = pitch_arg
        target_pad = None
        if chip_pitch_k <= 0:
            try:
                chip_pitch_k, target_pad, max_metal = suggest_legal_chip_pitch(
                    net,
                    node_map,
                    roots,
                    grid_to_pad_ratio=grid_to_pad_ratio,
                )
            except ValueError as exc:
                print(f"error: comp{k}: auto pitch failed: {exc}", file=sys.stderr)
                return 2
            print(
                f"comp{k}: auto chip_pitch={chip_pitch_k:.6g} "
                f"(c4-lattice={target_pad:.6g}, max_metal={max_metal:.6g}; "
                f"legal = max(c4, max_metal+1))"
            )

        try:
            ports = build_ports(
                net,
                n=chip_pitch_k,
                node_map=node_map,
                vdd=DEFAULT_VDD,
                roots=roots,
                via_stub=via_stub,
                full_pads=full_pads,
            )
        except ValueError as exc:
            print(f"error: comp{k}: {exc}", file=sys.stderr)
            return 2
        uniform_meta: dict = {}
        if uniform_spec is not None:
            try:
                uniform_meta = apply_uniform_sink_currents(ports.cells, uniform_spec)
            except ValueError as exc:
                print(f"error: comp{k}: {exc}", file=sys.stderr)
                return 2
            print(
                f"comp{k}: uniform-current mode={uniform_meta['uniform_current_mode']} "
                f"i_each={uniform_meta['i_each']:.6g} "
                f"(n_sinks={uniform_meta['n_sinks']}, "
                f"i_total_before={uniform_meta['i_total_before']:.6g})"
            )
        node_map = apply_grid_lumping(node_map, ports)
        ports = remap_ports(ports, node_map)
        save_ports(comp_dir, ports, node_map)
        try:
            occ_png = _write_pad_occupancy(comp_dir, ports)
            if occ_png:
                print(f"comp{k}: pad occupancy {occ_png}")
        except Exception as exc:  # noqa: BLE001
            print(f"warning: pad occupancy plot skipped: {exc}", file=sys.stderr)
        print(
            f"comp{k}: C4 pads n_pads={ports.n_pads}/{ports.n_sinks} "
            f"(chip_pitch={ports.cell_size:.6g}, "
            f"max_metal={ports.max_metal_pitch:.6g}; "
            f"{ports.n_padless} sink-only cells)"
        )

        top, bot = _component_layer_span(net, short_map, roots)
        layers = vdd_layers(net, short_map, roots)
        entry = {
            "index": k,
            "path": f"comp{k}",
            "n_nodes": len(roots),
            "top_layer": int(top),
            "bot_layer": int(bot),
            "vdd_layers": list(layers),
            "n_pads": ports.n_pads,
            "n_sinks": ports.n_sinks,
            "n_padless": int(ports.n_padless),
            "n_ports": ports.n_ports,
            "nx": ports.nx,
            "ny": ports.ny,
            "chip_pitch": ports.cell_size,
            "max_metal_pitch": ports.max_metal_pitch,
            "grid_to_pad_ratio": grid_to_pad_ratio if pitch_arg <= 0 else None,
            "pad_pitch_target": target_pad,
            "port_mode": ports.port_mode,
            "via_stub": ports.via_stub,
            "full_pads": bool(getattr(ports, "full_pads", False)),
            "uniform_current": (
                "conserve" if uniform_spec == "conserve" else uniform_spec
            ),
            **uniform_meta,
        }
        index.append(entry)
        summary = {
            "stage": "02_ports",
            "component_index": k,
            "out": str(comp_dir),
            **{key: entry[key] for key in entry if key != "path"},
            "vdd": ports.vdd,
            "metal_pitches": {str(kk): v for kk, v in ports.metal_pitches.items()},
        }
        print(json.dumps(summary, indent=2))

    save_components(
        out,
        {
            "n_components": n_comp,
            "order": "topmost_layer_first",
            "components": index,
        },
    )
    return 0


def ports_pitch_hint(net) -> str:
    from tsmc_region_grid import DEFAULT_LATTICE, collect_region_vdd_ports, lattice_from_cells

    cells = list(collect_region_vdd_ports(net.node_names).keys())
    lat = lattice_from_cells(cells, base=DEFAULT_LATTICE)
    return f"{lat.pitch:g}"


if __name__ == "__main__":
    raise SystemExit(main())
