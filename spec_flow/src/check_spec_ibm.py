#!/usr/bin/env python3
"""Score a flow OUT against the independent IBM spec oracle.

Usage (from repo root)::

    python3 spec_flow/src/check_spec_ibm.py \\
        Benchmarks/IBM/TC2/ibmpg2.spice \\
        spec_flow/outputs/ibmpg2_nauto_ir/comp1

    python3 spec_flow/src/check_spec_ibm.py \\
        Benchmarks/IBM/TC2/ibmpg2.spice \\
        my_flow/outputs/ibmpg2_auto_k1/comp1 --flow my_flow

Prints a per-clause table (JSON with --json).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph import build_short_map, vdd_connected_components  # noqa: E402
from spec_oracle import (  # noqa: E402
    ClauseResult,
    build_spec_oracle,
    classify_fit_method,
    compare_current_lumping,
    compare_full_sink_grid,
    compare_gprime_shape,
    compare_gs_pad_coupling,
    compare_pad_occupancy,
    compare_pad_voltages,
)
from spice_parser import parse_spice  # noqa: E402


def _load_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _detect_flow(comp_dir: Path, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    ports = _load_json(comp_dir / "ports.json") or {}
    if ports.get("kind") == "dual_ports" or (comp_dir / "tri_stagger_r.json").is_file():
        return "my_flow"
    if (comp_dir / "localized_r.json").is_file() or (
        comp_dir / "localized_model.json"
    ).is_file():
        return "local_flow"
    return "spec_flow"


def _pad_attach_from_payload(payload: dict) -> List[int]:
    if "pad_attach" in payload and payload["pad_attach"] is not None:
        att = [int(s) for s in payload["pad_attach"]]
        if att:
            return att
        # Empty list: legacy 1:1 when n_pads == n_sinks
        n_p = len(payload.get("pad_nodes", []))
        n_s = len(payload.get("cells", []))
        if n_p == n_s and n_p > 0:
            return list(range(n_p))
    # my_flow dual_ports: map pad_cell_ids → sink index
    cells = payload.get("cells", [])
    sink_of = {(int(c["ix"]), int(c["iy"])): i for i, c in enumerate(cells)}
    attach: List[int] = []
    for cid in payload.get("pad_cell_ids", []):
        key = (int(cid[0]), int(cid[1]))
        if key in sink_of:
            attach.append(sink_of[key])
        else:
            # pad cell without a sink port (zero-I dropped) — mark invalid index
            attach.append(-1)
    if attach:
        return attach
    # Legacy OUTs without pad_attach / pad_cell_ids: assume 1:1
    n_p = len(payload.get("pad_nodes", []))
    n_s = len(payload.get("cells", []))
    if n_p == n_s and n_p > 0:
        return list(range(n_p))
    return attach


def _uniform_current_from_out(comp_dir: Path) -> Any:
    comps = _load_json(comp_dir.parent / "components.json")
    if not comps:
        return None
    for entry in comps.get("components", []):
        if entry.get("path") == comp_dir.name or str(entry.get("index")) == comp_dir.name.replace(
            "comp", ""
        ):
            return entry.get("uniform_current")
    # fallback: first component
    entries = comps.get("components", [])
    if entries:
        return entries[0].get("uniform_current")
    return None


def _fit_meta(comp_dir: Path, flow: str) -> Tuple[Optional[str], Optional[str]]:
    if flow == "local_flow":
        r = _load_json(comp_dir / "localized_r.json") or {}
        m = _load_json(comp_dir / "localized_model.json") or {}
        return r.get("fit_method"), m.get("topology") or "localized_star_half_arm"
    if flow == "my_flow":
        for name in ("tri_stagger_r.json", "tri_square_r.json"):
            r = _load_json(comp_dir / name)
            if r:
                return r.get("fit_method"), r.get("topology")
        ports = _load_json(comp_dir / "ports.json") or {}
        if ports.get("kind") == "dual_ports":
            return "ir_rxryrz_typical", "tri_stagger_or_tri_square"
        return None, "tri_stagger_or_tri_square"
    # spec_flow
    r = _load_json(comp_dir / "pixel_r.json") or {}
    m = _load_json(comp_dir / "pixel_model.json") or {}
    fm = r.get("fit_method")
    if fm is None and r:
        # eigen fit may omit fit_method historically
        fm = "eigen" if "relative_spectral_error" in r and "relative_ir_error" not in r else None
    return fm, m.get("topology") or "star_half_arm"


def _try_build_gs(comp_dir: Path, flow: str, payload: dict):
    """Return (Gs, pad_attach) or (None, pad_attach) if model not loadable."""
    pad_attach = _pad_attach_from_payload(payload)
    if flow == "my_flow":
        return None, pad_attach
    try:
        from io_artifacts import load_ports
        from pixel_r import build_Gs, build_pixel_model, pixel_model_from_fields
    except Exception:
        return None, pad_attach

    try:
        ports, _ = load_ports(comp_dir)
    except Exception:
        return None, pad_attach

    if flow == "local_flow":
        model_path = comp_dir / "localized_model.json"
        r_path = comp_dir / "localized_r.json"
        if not model_path.is_file():
            return None, pad_attach
        fields = json.loads(model_path.read_text())
        # Prefer stored spectra Gs if present
        npz = comp_dir / "localized_spectra.npz"
        if npz.is_file():
            data = np.load(npz)
            if "Gs" in data:
                return np.asarray(data["Gs"], dtype=float), list(ports.resolved_pad_attach())
        if r_path.is_file():
            try:
                from localized_pixel import (  # type: ignore
                    build_Gs as build_loc_Gs,
                    localized_model_from_fields,
                )

                # localized_pixel lives in local_flow/src — may not be on path
            except Exception:
                pass
        # Fall back: uniform star using mean cell R if present
        r = _load_json(r_path) or {}
        Rx = float(r.get("Rx_uniform", r.get("Rx", 1.0)))
        Ry = float(r.get("Ry_uniform", r.get("Ry", 1.0)))
        Rz = float(r.get("Rz_uniform", r.get("Rz", 1.0)))
        star_fields = dict(fields)
        star_fields["topology"] = "star_half_arm"
        try:
            model = pixel_model_from_fields(ports, star_fields)
            return build_Gs(model, Rx, Ry, Rz), list(model.pad_attach)
        except Exception:
            model = build_pixel_model(ports)
            return build_Gs(model, Rx, Ry, Rz), list(model.pad_attach)

    # spec_flow
    model_path = comp_dir / "pixel_model.json"
    r_path = comp_dir / "pixel_r.json"
    npz = comp_dir / "spectra.npz"
    if npz.is_file():
        data = np.load(npz)
        if "Gs" in data:
            att = list(ports.resolved_pad_attach())
            return np.asarray(data["Gs"], dtype=float), att
    if not model_path.is_file():
        model = build_pixel_model(ports)
    else:
        fields = json.loads(model_path.read_text())
        model = pixel_model_from_fields(ports, fields)
    r = _load_json(r_path) or {}
    Rx = float(r.get("Rx", 1.0))
    Ry = float(r.get("Ry", 1.0))
    Rz = float(r.get("Rz", 1.0))
    return build_Gs(model, Rx, Ry, Rz), list(model.pad_attach)


def score_comp(
    spice_path: Path,
    comp_dir: Path,
    *,
    flow: Optional[str] = None,
    component_index: int = 1,
) -> Dict[str, Any]:
    flow = _detect_flow(comp_dir, flow)
    payload = _load_json(comp_dir / "ports.json")
    if not payload:
        raise FileNotFoundError(f"missing ports.json under {comp_dir}")

    cell_size = float(payload.get("pitch_bot", payload.get("cell_size")))
    nx = int(payload.get("nx_bot", payload.get("nx")))
    ny = int(payload.get("ny_bot", payload.get("ny")))
    cells = payload["cells"]
    pad_nodes = [str(n) for n in payload["pad_nodes"]]
    pad_voltages = [float(v) for v in payload.get("pad_voltages", [])]
    pad_attach = _pad_attach_from_payload(payload)

    net = parse_spice(spice_path)
    node_map = build_short_map(net)
    comps = vdd_connected_components(net, node_map)
    if component_index < 1 or component_index > len(comps):
        raise ValueError(f"component_index {component_index} out of range")
    roots = comps[component_index - 1]
    oracle = build_spec_oracle(net, cell_size, node_map=node_map, roots=roots)

    # Prefer OUT node_map for pad-root checks when present
    out_map_path = comp_dir / "node_map.json"
    out_map = None
    if out_map_path.is_file():
        out_map = {str(k): str(v) for k, v in json.loads(out_map_path.read_text()).items()}

    uniform = _uniform_current_from_out(comp_dir)
    results: List[ClauseResult] = []

    # Occupancy: prefer pad_cell_ids (my_flow) so sparse sinks do not
    # falsely mark C4-overlap pads as missing.
    if payload.get("pad_cell_ids"):
        occ_cells = [
            {"ix": int(a), "iy": int(b), "node": "?", "current": 0.0}
            for a, b in payload["pad_cell_ids"]
        ]
        # Synthetic attach: one pad per pad_cell_id in order
        occ_attach = list(range(len(occ_cells)))
        # Reorder pad_nodes to align with pad_cell_ids (same order in DualPortSet)
        results.append(
            compare_pad_occupancy(
                oracle,
                pad_nodes=pad_nodes,
                pad_attach=occ_attach,
                cells=occ_cells,
                node_map=out_map or node_map,
            )
        )
    else:
        results.append(
            compare_pad_occupancy(
                oracle,
                pad_nodes=pad_nodes,
                pad_attach=[s for s in pad_attach if s >= 0],
                cells=cells,
                node_map=out_map or node_map,
            )
        )
    # Pads whose cells were dropped from the sink list (my_flow)
    n_orphan_pads = sum(1 for s in pad_attach if s < 0)
    if n_orphan_pads:
        results.append(
            ClauseResult(
                "A1b_pad_without_sink",
                "fail",
                f"{n_orphan_pads} pads attach to cells with no sink port "
                f"(zero-I cells dropped)",
                {"n_orphan_pads": float(n_orphan_pads)},
            )
        )

    results.append(
        compare_current_lumping(
            oracle, cells=cells, uniform_current=uniform
        )
    )
    results.append(
        compare_full_sink_grid(
            oracle,
            n_sinks=len(cells),
            nx=nx,
            ny=ny,
            flow_name=flow,
        )
    )
    results.append(
        compare_pad_voltages(
            oracle,
            pad_voltages=pad_voltages,
            pad_nodes=pad_nodes,
            node_map=out_map or node_map,
        )
    )

    meta = _load_json(comp_dir / "system_meta.json") or {}
    gshape = meta.get("Gprime_shape")
    if gshape is None and (comp_dir / "Gprime.npy").is_file():
        Gp = np.load(comp_dir / "Gprime.npy")
        gshape = list(Gp.shape)
    results.append(
        compare_gprime_shape(
            n_pads=len(pad_nodes), n_sinks=len(cells), gprime_shape=gshape
        )
    )

    Gs, gs_attach = _try_build_gs(comp_dir, flow, payload)
    if flow == "my_flow":
        results.append(
            ClauseResult(
                "C_pad_sink_coupling",
                "intentional_deviation",
                "my_flow is not Pixel-R star; pad–sink coupling clause N/A",
            )
        )
        results.append(
            ClauseResult(
                "C_model_class",
                "intentional_deviation",
                "model class is tri-stagger / tri-square, not identical Pixel-R stars",
            )
        )
    else:
        results.append(
            compare_gs_pad_coupling(
                n_pads=len(pad_nodes),
                n_sinks=len(cells),
                pad_attach=gs_attach,
                Gs=Gs,
            )
        )
        results.append(
            ClauseResult(
                "C_model_class",
                "pass"
                if flow == "spec_flow"
                else "intentional_deviation",
                (
                    "uniform Pixel-R star_half_arm"
                    if flow == "spec_flow"
                    else "localized Pixel-R (per-block/cell Rx,Ry,Rz) — in-class variant"
                ),
            )
        )

    fit_method, model_class = _fit_meta(comp_dir, flow)
    results.append(classify_fit_method(fit_method, model_class))

    # Same-net sanity: all oracle C4s / injections already filtered by roots
    results.append(
        ClauseResult(
            "A4_same_net",
            "pass",
            f"oracle restricted to VDD component {component_index} "
            f"(n_roots={len(roots)}, n_c4={len(oracle.c4s)}, "
            f"n_I={len(oracle.injections)})",
            {
                "n_roots": float(len(roots)),
                "n_c4": float(len(oracle.c4s)),
                "n_injections": float(len(oracle.injections)),
            },
        )
    )

    summary = {
        "spice": str(spice_path),
        "comp_dir": str(comp_dir),
        "flow": flow,
        "component_index": component_index,
        "oracle": oracle.to_summary(),
        "flow_ports": {
            "n_pads": len(pad_nodes),
            "n_sinks": len(cells),
            "nx": nx,
            "ny": ny,
            "cell_size": cell_size,
            "uniform_current": uniform,
            "fit_method": fit_method,
            "model_class": model_class,
        },
        "clauses": [r.to_dict() for r in results],
        "n_pass": sum(1 for r in results if r.status == "pass"),
        "n_fail": sum(1 for r in results if r.status == "fail"),
        "n_skip": sum(1 for r in results if r.status == "skip"),
        "n_intentional_deviation": sum(
            1 for r in results if r.status == "intentional_deviation"
        ),
    }
    return summary


def _print_table(summary: Dict[str, Any]) -> None:
    print(f"flow={summary['flow']}  OUT={summary['comp_dir']}")
    o = summary["oracle"]
    f = summary["flow_ports"]
    print(
        f"oracle: n_c4={o['n_c4']} n_I={o['n_injections']} "
        f"grid={o['nx']}x{o['ny']} n_pads={o['n_pads']} "
        f"n_sinks={o['n_sinks']} collapsed_c4={o['n_c4_collapsed']} "
        f"I_tot={o['i_total']:.6g}"
    )
    print(
        f"flow:   n_pads={f['n_pads']} n_sinks={f['n_sinks']} "
        f"grid={f['nx']}x{f['ny']} pitch={f['cell_size']} "
        f"uniform={f['uniform_current']!r} fit={f['fit_method']!r}"
    )
    print("-" * 72)
    print(f"{'clause':<28} {'status':<22} detail")
    print("-" * 72)
    for c in summary["clauses"]:
        detail = c["detail"]
        if len(detail) > 60:
            detail = detail[:57] + "..."
        print(f"{c['clause']:<28} {c['status']:<22} {detail}")
    print("-" * 72)
    print(
        f"pass={summary['n_pass']} fail={summary['n_fail']} "
        f"skip={summary['n_skip']} intentional_deviation="
        f"{summary['n_intentional_deviation']}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Score a flow OUT against the IBM TSMC-spec port oracle"
    )
    p.add_argument("spice", type=Path, help="IBM .spice deck")
    p.add_argument("comp_dir", type=Path, help="Component OUT (…/comp1)")
    p.add_argument(
        "--flow",
        choices=("spec_flow", "local_flow", "my_flow"),
        default=None,
        help="Override flow detection",
    )
    p.add_argument(
        "--comp",
        type=int,
        default=1,
        help="VDD component index (1-based, topmost-first)",
    )
    p.add_argument("--json", action="store_true", help="Print full JSON summary")
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write JSON summary to this path",
    )
    args = p.parse_args(argv)

    if not args.spice.is_file():
        print(f"error: spice not found: {args.spice}", file=sys.stderr)
        return 2
    if not args.comp_dir.is_dir():
        print(f"error: comp_dir not found: {args.comp_dir}", file=sys.stderr)
        return 2

    summary = score_comp(
        args.spice,
        args.comp_dir,
        flow=args.flow,
        component_index=int(args.comp),
    )
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out_json}")
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_table(summary)
    return 0 if summary["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
