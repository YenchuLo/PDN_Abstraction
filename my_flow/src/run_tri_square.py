#!/usr/bin/env python3
"""Build tri-square L2/L3 stubs model + fit 6 R params (per-sheet Rx/Ry + 2 vias)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import ensure_paths

ensure_paths()

from fit_ir_tri_square import fit_tri_square_ir  # noqa: E402
from graph import (  # noqa: E402
    assemble_conductance,
    galvanic_component_of,
    partition_ports,
)
from io_artifacts import (  # noqa: E402
    load_net,
    load_ports,
    save_tri_square_fit,
    save_tri_square_model,
    save_tri_square_ports,
)
from kron import kron_reduce  # noqa: E402
from plot_ir_compare import plot_ir_for_model_run  # noqa: E402
from ports_dual import can_tri_stagger_ports, tri_stagger_ports_from_dual  # noqa: E402
from stub_attach import parse_via_stub_arg  # noqa: E402
from tri_square import TRI_SQUARE_TOPOLOGY, build_tri_square_model  # noqa: E402


def _rebuild_gprime(net, ports, node_map):
    keep = galvanic_component_of(net, node_map, ports.port_nodes)
    system = assemble_conductance(net, node_map, keep_roots=keep)
    port_idx, internal_idx = partition_ports(system, ports.port_nodes)
    Gprime = kron_reduce(system.G, port_idx, internal_idx)
    return Gprime, keep


def _mean_slot(value: float) -> dict:
    v = float(value)
    return {"mean": v, "median": v, "min": v, "max": v, "n": 1}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Tri-square L2/L3 stubs + IR fit of "
            "Rx_u,Ry_u,Rx_l,Ry_l,Rz_pad,Rz_ul"
        )
    )
    p.add_argument("out", type=Path, help="Component OUT (e.g. .../ibmpg2_auto_k1/comp1)")
    p.add_argument("seed", type=int, nargs="?", default=0)
    p.add_argument(
        "--k",
        type=int,
        default=2,
        help="L2/L3: pitch_u = pitch_bot (Pixel-R), pitch_l = pitch_u/k denser (default 2)",
    )
    p.add_argument(
        "--via-stub",
        default="rxry",
        metavar="MODE",
        help=(
            "Pad/sink/L2→L3 attach: 'rxry' (default, L-bend ∝ Rx/Ry) or 'zero' "
            "(via on nearest grid node)."
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
            "no nonzero current sinks for tri-square ports "
            f"(n_pads={dual_ports.n_pads}, n_sink_cells={dual_ports.n_sinks})"
        )
        out = {
            "stage": "tri_square",
            "out": str(args.out),
            "skipped": True,
            "reason": msg,
        }
        print(json.dumps(out, indent=2))
        if args.strict:
            print(f"error: {msg}", file=sys.stderr)
            return 2
        return 0

    ports = tri_stagger_ports_from_dual(dual_ports)
    save_tri_square_ports(args.out, ports, node_map)

    print(
        f"ports: n_pads={ports.n_pads} n_sinks={ports.n_sinks} "
        f"pitch_bot={ports.pitch_bot:.6g} k={args.k}",
        flush=True,
    )
    Gprime, _roots = _rebuild_gprime(net, ports, node_map)
    print(f"Gprime shape={tuple(Gprime.shape)}", flush=True)

    try:
        stub_mode = parse_via_stub_arg(args.via_stub)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    model = build_tri_square_model(ports, coarsen_k=int(args.k), via_stub=stub_mode)
    print(
        f"grids: L2 {model.nx_u}x{model.ny_u} pitch_u={model.pitch_u:.6g}; "
        f"L3 {model.nx_l}x{model.ny_l} pitch_l={model.pitch_l:.6g}; "
        f"via_arms={len(model.vias_ul)} (n_l2={model.n_l2} n_l3={model.n_l3})",
        flush=True,
    )
    fields = model.to_fields()
    save_tri_square_model(args.out, fields)

    print("fitting Rx_u,Ry_u,Rx_l,Ry_l,Rz_pad,Rz_ul ...", flush=True)
    fit = fit_tri_square_ir(Gprime, model, ports, seed=args.seed, polish=True)
    summary_means = {
        "R_ew_u": fit.Rx_u,
        "R_ns_u": fit.Ry_u,
        "R_ew_l": fit.Rx_l,
        "R_ns_l": fit.Ry_l,
        "R_via_pad": fit.Rz_pad,
        "R_via_ul": fit.Rz_ul,
    }
    R_edge_stats = {
        "ew_u": _mean_slot(fit.Rx_u),
        "ns_u": _mean_slot(fit.Ry_u),
        "ew_l": _mean_slot(fit.Rx_l),
        "ns_l": _mean_slot(fit.Ry_l),
        "via_pad": _mean_slot(fit.Rz_pad),
        "via_ul": _mean_slot(fit.Rz_ul),
    }
    tri_r = {
        "Rx_u": fit.Rx_u,
        "Ry_u": fit.Ry_u,
        "Rx_l": fit.Rx_l,
        "Ry_l": fit.Ry_l,
        "Rz_pad": fit.Rz_pad,
        "Rz_ul": fit.Rz_ul,
        # Aliases
        "Rx": fit.Rx_u,
        "Ry": fit.Ry_u,
        "Rz": fit.Rz_pad,
        "stamp": "per_sheet_Rx_Ry_two_vias_stubs",
        "topology": TRI_SQUARE_TOPOLOGY,
        "via_stub": stub_mode,
        "pitch_u": model.pitch_u,
        "pitch_l": model.pitch_l,
        "coarsen_k": model.coarsen_k,
        "summary_means": summary_means,
        "R_edge_stats": R_edge_stats,
    }
    fit_meta = {
        "fit_method": "ir_6r",
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
        "n_l2": model.n_l2,
        "n_l3": model.n_l3,
        "nx_u": model.nx_u,
        "ny_u": model.ny_u,
        "nx_l": model.nx_l,
        "ny_l": model.ny_l,
        "n_pads": model.n_pads,
        "n_sinks": model.n_sinks,
        "n_ew_u": len(model.ew_u),
        "n_ns_u": len(model.ns_u),
        "n_ew_l": len(model.ew_l),
        "n_ns_l": len(model.ns_l),
        "n_vias_ul": len(model.vias_ul),
        "via_stub": stub_mode,
        "Gprime_shape": list(Gprime.shape),
        "param_names": [
            "Rx_u",
            "Ry_u",
            "Rx_l",
            "Ry_l",
            "Rz_pad",
            "Rz_ul",
        ],
    }
    save_tri_square_fit(args.out, tri_r, Gprime, fit.Gs, fit_meta)

    fields.update(
        {
            "Rx_u": fit.Rx_u,
            "Ry_u": fit.Ry_u,
            "Rx_l": fit.Rx_l,
            "Ry_l": fit.Ry_l,
            "Rz_pad": fit.Rz_pad,
            "Rz_ul": fit.Rz_ul,
            "Rx": fit.Rx_u,
            "Ry": fit.Ry_u,
            "Rz": fit.Rz_pad,
            "R_edge_stats": R_edge_stats,
            "summary_means": summary_means,
        }
    )
    save_tri_square_model(args.out, fields)

    plot_info = plot_ir_for_model_run(
        args.out,
        ports,
        Gprime,
        fit.Gs,
        model_key="tri_square",
        seed=args.seed,
    )

    out = {
        "stage": "tri_square",
        "out": str(args.out),
        "topology": TRI_SQUARE_TOPOLOGY,
        "relative_ir_error": fit.relative_ir_error,
        "e_ir_x0": fit.e_ir_x0,
        "e_ir_by_stimulus": fit.e_ir_by_stimulus,
        "heldout_e_worst": fit.heldout_e_worst,
        "Rx_u": fit.Rx_u,
        "Ry_u": fit.Ry_u,
        "Rx_l": fit.Rx_l,
        "Ry_l": fit.Ry_l,
        "Rz_pad": fit.Rz_pad,
        "Rz_ul": fit.Rz_ul,
        "pitch_u": model.pitch_u,
        "pitch_l": model.pitch_l,
        "coarsen_k": model.coarsen_k,
        "n_pads": model.n_pads,
        "n_sinks": model.n_sinks,
        "n_l2": model.n_l2,
        "n_l3": model.n_l3,
        "fit_message": fit.message,
        "plots": plot_info.get("plots"),
        "plot_report": {
            "e_ir_reduced_vs_model": (plot_info.get("mna") or {}).get(
                "e_ir_reduced_vs_model"
            ),
            "original_vs_reduced": plot_info.get("original_vs_reduced"),
        },
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
