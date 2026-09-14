#!/usr/bin/env python3
"""Stage 02: aligned dual-layer ports (common lattice, integer coarsening)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence, Set, Union

from _bootstrap import ensure_paths

ensure_paths()

from graph import (  # noqa: E402
    _component_layer_span,
    build_short_map,
    vdd_connected_components,
    vdd_layers,
    vdd_metal_pitches,
)
from io_artifacts import (  # noqa: E402
    NET_META,
    ensure_out,
    load_net,
    save_components,
    save_net,
    save_ports,
)
from ports import (  # noqa: E402
    apply_uniform_sink_currents,
    parse_uniform_current_arg,
)
from ports_dual import (  # noqa: E402
    DEFAULT_VDD,
    apply_dual_grid_lumping,
    build_dual_ports,
    remap_dual_ports,
    suggest_legal_chip_pitch,
)
from spice_parser import parse_perturb_layer_arg, perturb_layer_resistors  # noqa: E402
from tsmc_region_grid import is_virtual_pixel_r_lattice  # noqa: E402

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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build aligned dual-layer ports (pitch_top = k * pitch_bot)"
    )
    p.add_argument("out", type=Path, help="Flow OUT root")
    p.add_argument(
        "pitch_bot",
        type=float,
        nargs="?",
        default=0.0,
        help="Fine mesh pitch (0 = auto: same as spec_flow — C4 lattice, > max metal)",
    )
    p.add_argument(
        "coarsen_k",
        type=int,
        nargs="?",
        default=1,
        help="Integer coarsening: pitch_top = k * pitch_bot (default 1)",
    )
    p.add_argument(
        "--grid-to-pad-ratio",
        type=float,
        default=1.0,
        help=(
            "Divide C4 lattice pitch by sqrt(ratio) (default 1.0 = match bump "
            "spacing). Then clamp to > max VDD metal stripe pitch (spec_flow rule). "
            "Ignored when pitch_bot > 0."
        ),
    )
    p.add_argument(
        "--uniform-current",
        nargs="?",
        const="conserve",
        default=None,
        metavar="MODE",
        help=(
            "Same current on every fine-grid sink (expands sinks to full bot grid). "
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
            "Assign a voltage pad to every fine-grid cell: keep C4-overlap pads, "
            "fill padless cells with the unique nearest unused top-metal VDD node, "
            "and drive all pads at one common voltage (representative of real C4s)."
        ),
    )
    args = p.parse_args(argv)
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

    out = ensure_out(args.out)
    net = load_net(out)
    if is_virtual_pixel_r_lattice(net):
        print(
            "error: my_flow dual-layer ports do not support TSMC virtual Pixel-R "
            "lattices (shorted VDD_in / region ports). Use:\n"
            "  tclsh spec_flow/run_flow.tcl tsmc1\n"
            "Tri-stagger / tri-square need IBM-style multi-track metal meshes.",
            file=sys.stderr,
        )
        return 2
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

    pitch_bot_arg = float(args.pitch_bot)
    k = max(1, int(args.coarsen_k))
    grid_to_pad_ratio = float(args.grid_to_pad_ratio)
    if grid_to_pad_ratio <= 0:
        print("error: --grid-to-pad-ratio must be positive", file=sys.stderr)
        return 2

    meta_path = out / NET_META
    spice_path = str(out)
    if meta_path.is_file():
        try:
            spice_path = str(
                json.loads(meta_path.read_text()).get("spice_path", spice_path)
            )
        except json.JSONDecodeError:
            pass

    node_map0 = dict(short_map)
    roots0 = comps[0]
    pb0 = pitch_bot_arg
    if pb0 <= 0:
        try:
            pb0, _, _ = suggest_legal_chip_pitch(
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
                cell_size=float(pb0),
                layers=sorted(all_layers),
                roots=all_roots,
                node_map=node_map0,
                spice_path=spice_path,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    index: list[dict] = []
    for ki, roots in enumerate(comps, start=1):
        comp_dir = out / f"comp{ki}"
        ensure_out(comp_dir)
        node_map = dict(short_map)

        pb = pitch_bot_arg
        if pb <= 0:
            pitches = vdd_metal_pitches(net, node_map, roots)
            min_pitch = min(pitches.values()) if pitches else 0.0
            max_pitch = max(pitches.values()) if pitches else 0.0
            try:
                pb, target_pad, max_metal = suggest_legal_chip_pitch(
                    net,
                    node_map,
                    roots,
                    grid_to_pad_ratio=grid_to_pad_ratio,
                )
                if max_pitch <= 0:
                    max_pitch = max_metal
            except ValueError as exc:
                print(f"error: comp{ki}: auto pitch failed: {exc}", file=sys.stderr)
                return 2
            print(
                f"comp{ki}: auto pitch_bot={pb:.6g} "
                f"(c4-lattice={target_pad:.6g}, max_metal={max_pitch:.6g}; "
                f"min_metal={min_pitch:.6g}; legal = max(c4, max_metal+1))"
            )
        else:
            target_pad = None
        pt = k * pb

        try:
            ports = build_dual_ports(
                net,
                pitch_top=pt,
                pitch_bot=pb,
                node_map=node_map,
                vdd=DEFAULT_VDD,
                roots=roots,
                full_grid_sinks=True,
                full_pads=bool(args.full_pads),
            )
        except ValueError as exc:
            print(f"error: comp{ki}: {exc}", file=sys.stderr)
            return 2
        uniform_meta: dict = {}
        if uniform_spec is not None:
            try:
                uniform_meta = apply_uniform_sink_currents(ports.cells, uniform_spec)
            except ValueError as exc:
                print(f"error: comp{ki}: {exc}", file=sys.stderr)
                return 2
            print(
                f"comp{ki}: uniform-current mode={uniform_meta['uniform_current_mode']} "
                f"i_each={uniform_meta['i_each']:.6g} "
                f"(n_sinks={uniform_meta['n_sinks']}, "
                f"i_total_before={uniform_meta['i_total_before']:.6g})"
            )
        node_map = apply_dual_grid_lumping(node_map, ports)
        ports = remap_dual_ports(ports, node_map)
        save_ports(comp_dir, ports, node_map)
        n_grid = int(ports.nx_bot) * int(ports.ny_bot)
        n_padless = max(0, n_grid - int(ports.n_pads))
        print(
            f"comp{ki}: C4 pads n_pads={ports.n_pads}/{n_grid} "
            f"(pitch_bot={ports.pitch_bot:.6g}; {n_padless} cells without C4)"
        )

        top, bot = _component_layer_span(net, short_map, roots)
        layers = vdd_layers(net, short_map, roots)
        entry = {
            "index": ki,
            "path": f"comp{ki}",
            "n_nodes": len(roots),
            "top_layer": int(top),
            "bot_layer": int(bot),
            "vdd_layers": list(layers),
            "n_pads": ports.n_pads,
            "n_sinks": ports.n_sinks,
            "n_ports": ports.n_ports,
            "pitch_top": ports.pitch_top,
            "pitch_bot": ports.pitch_bot,
            "coarsen_k": ports.coarsen_k,
            "grid_to_pad_ratio": grid_to_pad_ratio if pitch_bot_arg <= 0 else None,
            "nx_top": ports.nx_top,
            "ny_top": ports.ny_top,
            "nx_bot": ports.nx_bot,
            "ny_bot": ports.ny_bot,
            "max_metal_pitch": ports.max_metal_pitch,
            "full_pads": bool(getattr(ports, "full_pads", False)),
            "uniform_current": (
                "conserve" if uniform_spec == "conserve" else uniform_spec
            ),
            **uniform_meta,
        }
        index.append(entry)
        print(
            json.dumps(
                {
                    "stage": "02_ports",
                    "component_index": ki,
                    "out": str(comp_dir),
                    **{key: entry[key] for key in entry if key != "path"},
                    "vdd": ports.vdd,
                    "metal_pitches": {
                        str(kk): v for kk, v in ports.metal_pitches.items()
                    },
                },
                indent=2,
            )
        )

    save_components(
        out,
        {
            "n_components": n_comp,
            "order": "topmost_layer_first",
            "components": index,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
