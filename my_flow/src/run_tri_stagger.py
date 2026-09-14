#!/usr/bin/env python3
"""Build staggered mid-grid model; fit global Rx, Ry, Rz (Pixel-R style).

Vias at pad/sink XY attach to a square mid grid via stubs (R ∝ Rx/Ry).
Reuses dual-flow OUT/compN, drops pad-forced sinks, rebuilds G'.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import ensure_paths

ensure_paths()

from fit_ir_tri_stagger import fit_tri_stagger_ir  # noqa: E402
from graph import (  # noqa: E402
    assemble_conductance,
    galvanic_component_of,
    partition_ports,
)
from io_artifacts import (  # noqa: E402
    load_net,
    load_ports,
    save_tri_stagger_fit,
    save_tri_stagger_model,
    save_tri_stagger_ports,
)
from kron import kron_reduce  # noqa: E402
from ports_dual import can_tri_stagger_ports, tri_stagger_ports_from_dual  # noqa: E402
from plot_ir_compare import plot_ir_for_model_run  # noqa: E402
from spice_emit import emit_all_tri_stagger  # noqa: E402
from stub_attach import parse_via_stub_arg  # noqa: E402
from tri_stagger import build_tri_stagger_model  # noqa: E402


def _rebuild_gprime(net, ports, node_map):
    keep = galvanic_component_of(net, node_map, ports.port_nodes)
    system = assemble_conductance(net, node_map, keep_roots=keep)
    port_idx, internal_idx = partition_ports(system, ports.port_nodes)
    Gprime = kron_reduce(system.G, port_idx, internal_idx)
    return Gprime, keep


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Staggered mid-layer + IR fit of global Rx, Ry, Rz"
    )
    p.add_argument("out", type=Path, help="Component OUT (e.g. .../ibmpg2_auto_k1/comp1)")
    p.add_argument("seed", type=int, nargs="?", default=0)
    p.add_argument("--spice", action="store_true", help="Also emit ngspice decks")
    p.add_argument("--timeout", type=int, default=21600)
    p.add_argument(
        "--via-stub",
        default="rxry",
        metavar="MODE",
        help=(
            "Pad/sink attach: 'rxry' (default, L-bend ∝ Rx/Ry) or 'zero' "
            "(via on nearest mid-grid node)."
        ),
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail (exit 2) on pad-only / zero-current components instead of skip",
    )
    args = p.parse_args(argv)

    dual_ports, node_map = load_ports(args.out)
    net = load_net(args.out)
    if not can_tri_stagger_ports(dual_ports):
        msg = (
            "no nonzero current sinks for stagger ports "
            f"(n_pads={dual_ports.n_pads}, n_sink_cells={dual_ports.n_sinks}; "
            "pad-only / zero-load component — skip for stagger)"
        )
        out = {
            "stage": "tri_stagger_mid_grid",
            "out": str(args.out),
            "skipped": True,
            "reason": msg,
            "n_pads": dual_ports.n_pads,
            "n_sinks": dual_ports.n_sinks,
        }
        print(json.dumps(out, indent=2))
        if args.strict:
            print(f"error: {msg}", file=sys.stderr)
            return 2
        return 0
    ports = tri_stagger_ports_from_dual(dual_ports)
    save_tri_stagger_ports(args.out, ports, node_map)

    Gprime, _roots = _rebuild_gprime(net, ports, node_map)

    try:
        stub_mode = parse_via_stub_arg(args.via_stub)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    model = build_tri_stagger_model(ports, via_stub=stub_mode)
    fields = model.to_fields()
    save_tri_stagger_model(args.out, fields)

    fit = fit_tri_stagger_ir(Gprime, model, ports, seed=args.seed, polish=True)
    stagger_r = {
        "Rx": fit.Rx,
        "Ry": fit.Ry,
        "Rz": fit.Rz,
        "stamp": "global_Rx_Ry_Rz_length_proportional",
        "topology": "tri_stagger_mid_grid",
        "via_stub": stub_mode,
    }
    fit_meta = {
        "fit_method": "ir_rxryrz",
        "polish": True,
        "relative_ir_error": fit.relative_ir_error,
        "e_worst": fit.e_worst,
        "e_ir_x0": fit.e_ir_x0,
        "e_ir_by_stimulus": fit.e_ir_by_stimulus,
        "heldout_e_worst": fit.heldout_e_worst,
        "heldout_e_ir_by_stimulus": fit.heldout_e_ir_by_stimulus,
        "fit_success": fit.success,
        "fit_message": fit.message,
        "seed": args.seed,
        "extract_meta": model.extract_meta,
        "n_mid": model.n_mid,
        "n_grid": model.n_grid,
        "nx_grid": model.nx_grid,
        "ny_grid": model.ny_grid,
        "n_pads": model.n_pads,
        "n_sinks": model.n_sinks,
        "n_ew": len(model.ew_edges),
        "n_ns": len(model.ns_edges),
        "via_stub": stub_mode,
        "Gprime_shape": list(Gprime.shape),
    }
    save_tri_stagger_fit(args.out, stagger_r, Gprime, fit.Gs, fit_meta)

    plot_info = plot_ir_for_model_run(
        args.out,
        ports,
        Gprime,
        fit.Gs,
        model_key="tri_stagger",
        seed=args.seed,
        spice_model_stem="tri_stagger",
    )

    spice_info = None
    if args.spice:
        paths = emit_all_tri_stagger(
            args.out,
            net,
            ports,
            node_map,
            Gprime,
            model,
            Rx=fit.Rx,
            Ry=fit.Ry,
            Rz=fit.Rz,
        )
        spice_info = {"decks": paths, "note": "MNA-only; decks written"}

    out = {
        "stage": "tri_stagger_mid_grid",
        "out": str(args.out),
        "topology": "tri_stagger_mid_grid",
        "relative_ir_error": fit.relative_ir_error,
        "e_ir_x0": fit.e_ir_x0,
        "e_ir_by_stimulus": fit.e_ir_by_stimulus,
        "heldout_e_worst": fit.heldout_e_worst,
        "Rx": fit.Rx,
        "Ry": fit.Ry,
        "Rz": fit.Rz,
        "n_pads": model.n_pads,
        "n_sinks": model.n_sinks,
        "n_mid": model.n_mid,
        "n_grid": model.n_grid,
        "nx_grid": model.nx_grid,
        "ny_grid": model.ny_grid,
        "n_ew": len(model.ew_edges),
        "n_ns": len(model.ns_edges),
        "polish": True,
        "plots": plot_info.get("plots"),
        "plot_report": {
            "e_ir_reduced_vs_model": (plot_info.get("mna") or {}).get(
                "e_ir_reduced_vs_model"
            ),
            "original_vs_reduced": plot_info.get("original_vs_reduced"),
        },
        "spice": spice_info,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
