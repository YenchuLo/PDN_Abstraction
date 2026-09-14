#!/usr/bin/env python3
"""VoltSpot-style Eq. 2 multi-layer grid on IBM PG (separate from the 4 models).

Builds its own OUT (default ibmpg2_voltspot) with grid-to-pad ratio 4:1,
extracts parallel per-layer branch R (vias omitted), evaluates sink IR vs Kron G'.

By default reports extract-only error (paper style). Pass --polish for optional
α_x / α_y cleanup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import ensure_paths

ensure_paths()

from extract_r_voltspot import extract_voltspot_conductances  # noqa: E402
from fit_ir_voltspot import fit_voltspot_ir  # noqa: E402
from graph import (  # noqa: E402
    assemble_conductance,
    galvanic_component_of,
    partition_ports,
)
from io_artifacts import (  # noqa: E402
    load_net,
    load_ports,
    save_voltspot_fit,
    save_voltspot_model,
    save_voltspot_ports,
)
from kron import kron_reduce  # noqa: E402
from layer_orient import classify_vdd_layers  # noqa: E402
from ports_dual import can_tri_stagger_ports, tri_stagger_ports_from_dual  # noqa: E402
from voltspot_model import build_voltspot_model  # noqa: E402

SRC = Path(__file__).resolve().parent
HERE = SRC.parent
ROOT = HERE.parent
DEFAULT_SPICE = ROOT / "Benchmarks/IBM/TC2/ibmpg2.spice"
DEFAULT_OUT = HERE / "outputs/ibmpg2_voltspot"


def _rebuild_gprime(net, ports, node_map):
    keep = galvanic_component_of(net, node_map, ports.port_nodes)
    system = assemble_conductance(net, node_map, keep_roots=keep)
    port_idx, internal_idx = partition_ports(system, ports.port_nodes)
    Gprime = kron_reduce(system.G, port_idx, internal_idx)
    return Gprime, keep


def _ensure_flow(
    spice: Path,
    out_root: Path,
    *,
    grid_to_pad_ratio: float,
    py: str,
) -> Path:
    """Parse + ports + return comp1 path. Reuses existing artifacts if present."""
    import subprocess

    out_root.mkdir(parents=True, exist_ok=True)
    net_npz = out_root / "net.npz"
    if not net_npz.is_file():
        subprocess.check_call(
            [py, str(SRC / "stage01_parse.py"), str(spice), str(out_root)]
        )
    comps = out_root / "components.json"
    ports_ok = False
    if comps.is_file():
        d = json.loads(comps.read_text())
        for c in d.get("components", []):
            ratio = c.get("grid_to_pad_ratio")
            if ratio is not None and abs(float(ratio) - grid_to_pad_ratio) < 1e-9:
                ports_ok = (out_root / c["path"] / "ports.json").is_file()
                break
    if not ports_ok:
        subprocess.check_call(
            [
                py,
                str(SRC / "stage02_ports.py"),
                str(out_root),
                "0",
                "1",
                "--grid-to-pad-ratio",
                str(grid_to_pad_ratio),
            ]
        )
    return out_root / "comp1"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="VoltSpot-style Eq.2 multi-layer IR eval (not one of the 4 models)"
    )
    p.add_argument(
        "--spice",
        type=Path,
        default=DEFAULT_SPICE,
        help=f"IBM SPICE (default: {DEFAULT_SPICE})",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Flow OUT root (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--comp",
        type=Path,
        default=None,
        help="Reuse an existing dual-flow OUT/compN (skips parse/ports)",
    )
    p.add_argument("seed", type=int, nargs="?", default=0)
    p.add_argument(
        "--grid-to-pad-ratio",
        type=float,
        default=4.0,
        help="VoltSpot default 4:1 (ignored with --comp)",
    )
    p.add_argument(
        "--polish",
        action="store_true",
        help="Optional α_x/α_y polish (paper validation is extract-only)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail on pad-only / zero-current components",
    )
    args = p.parse_args(argv)

    py = sys.executable
    if args.comp is not None:
        comp_dir = args.comp
    else:
        comp_dir = _ensure_flow(
            args.spice,
            args.out,
            grid_to_pad_ratio=float(args.grid_to_pad_ratio),
            py=py,
        )

    dual_ports, node_map = load_ports(comp_dir)
    net = load_net(comp_dir)
    if not can_tri_stagger_ports(dual_ports):
        msg = (
            "no nonzero current sinks for VoltSpot ports "
            f"(n_pads={dual_ports.n_pads}, n_sink_cells={dual_ports.n_sinks})"
        )
        out = {
            "stage": "voltspot_eq2_multilayer",
            "out": str(comp_dir),
            "skipped": True,
            "reason": msg,
        }
        print(json.dumps(out, indent=2))
        if args.strict:
            print(f"error: {msg}", file=sys.stderr)
            return 2
        return 0

    # Current-only sinks (same port hygiene as stagger / tri-square).
    ports = tri_stagger_ports_from_dual(dual_ports)
    save_voltspot_ports(comp_dir, ports, node_map)

    print(
        f"ports: n_pads={ports.n_pads} n_sinks={ports.n_sinks} "
        f"grid={ports.nx_bot}x{ports.ny_bot} pitch={ports.pitch_bot:.6g} "
        f"ratio={args.grid_to_pad_ratio:g}",
        flush=True,
    )
    Gprime, roots = _rebuild_gprime(net, ports, node_map)
    print(f"Gprime shape={tuple(Gprime.shape)}", flush=True)

    model = build_voltspot_model(ports)
    orientations = classify_vdd_layers(net, node_map, roots, ports.vdd_layers)
    model = extract_voltspot_conductances(
        net, ports, model, node_map, roots, orientations=orientations
    )
    fields = model.to_fields()
    fields["stamp"] = model.extract_meta.get("stamp")
    fields["formula"] = model.extract_meta.get("formula")
    fields["summary_means"] = model.extract_meta.get("summary_means")
    fields["layers"] = model.extract_meta.get("layers")
    save_voltspot_model(comp_dir, fields)

    fit = fit_voltspot_ir(
        Gprime, model, ports, seed=args.seed, polish=bool(args.polish)
    )
    voltspot_r = {
        "alpha_x": fit.alpha_x,
        "alpha_y": fit.alpha_y,
        "stamp": "voltspot_eq2_multilayer",
        "topology": "voltspot_eq2_multilayer",
        "R_ew_branch": model.R_ew_branch,
        "R_ns_branch": model.R_ns_branch,
        "summary_means": model.extract_meta.get("summary_means"),
        "formula": model.extract_meta.get("formula"),
        "vias": "omitted",
        "grid_to_pad_ratio_target": float(args.grid_to_pad_ratio),
        "nx": model.nx,
        "ny": model.ny,
        "pitch": model.pitch,
    }
    fit_meta = {
        "fit_method": "ir_alpha_bounded" if fit.polish else "extract_only",
        "polish": fit.polish,
        "relative_ir_error": fit.relative_ir_error,
        "e_worst": fit.e_worst,
        "e_ir_pre_fit": fit.e_ir_extract_only,
        "e_ir_extract_only": fit.e_ir_extract_only,
        "e_ir_after_alpha": fit.e_ir_after_alpha,
        "e_ir_by_stimulus": fit.e_ir_by_stimulus,
        "heldout_e_worst": fit.heldout_e_worst,
        "heldout_e_ir_by_stimulus": fit.heldout_e_ir_by_stimulus,
        "fit_success": fit.success,
        "fit_message": fit.message,
        "seed": args.seed,
        "extract_meta": model.extract_meta,
        "n_grid": model.n_grid,
        "n_ew": len(model.ew_edges),
        "n_ns": len(model.ns_edges),
        "note": (
            "VoltSpot-style path; not registered among the four my_flow "
            "abstraction models (dual/tri/stagger/tri_square)."
        ),
    }
    save_voltspot_fit(comp_dir, voltspot_r, Gprime, fit.Gs, fit_meta)

    out = {
        "stage": "voltspot_eq2_multilayer",
        "out": str(comp_dir),
        "topology": "voltspot_eq2_multilayer",
        "relative_ir_error": fit.relative_ir_error,
        "e_ir_pre_fit": fit.e_ir_extract_only,
        "e_ir_by_stimulus": fit.e_ir_by_stimulus,
        "heldout_e_worst": fit.heldout_e_worst,
        "alpha_x": fit.alpha_x,
        "alpha_y": fit.alpha_y,
        "polish": fit.polish,
        "R_ew_branch": model.R_ew_branch,
        "R_ns_branch": model.R_ns_branch,
        "nx": model.nx,
        "ny": model.ny,
        "pitch": model.pitch,
        "grid_to_pad_ratio": float(args.grid_to_pad_ratio),
        "n_pads": ports.n_pads,
        "n_sinks": ports.n_sinks,
        "layers": model.extract_meta.get("layers"),
        "summary_means": model.extract_meta.get("summary_means"),
        "vias": "omitted",
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
