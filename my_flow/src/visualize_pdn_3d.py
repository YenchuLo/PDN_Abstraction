#!/usr/bin/env python3
"""Interactive Plotly 3D PDN visualizer (original left + selectable abstraction)."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from _bootstrap import ensure_paths

ensure_paths()

from dual_geometry_3d import (  # noqa: E402
    DualGeometry3D,
    build_tri_stagger_geometry_3d,
    build_tri_square_geometry_3d,
)
from io_artifacts import (  # noqa: E402
    COMPONENTS_JSON,
    NET_NPZ,
    PORTS_JSON,
    TRI_STAGGER_MODEL_JSON,
    TRI_STAGGER_R_JSON,
    TRI_SQUARE_MODEL_JSON,
    TRI_SQUARE_R_JSON,
    load_components,
    load_net,
    load_ports,
    load_tri_stagger_model_fields,
    load_tri_square_model_fields,
)
from pdn_geometry import extract_pdn_geometry, net_display_label, normalize_net_arg  # noqa: E402
from pdn_geometry_3d import default_z_pitch, lift_pdn_geometry  # noqa: E402
from pdn_plot_3d import (  # noqa: E402
    MODEL_LABELS,
    MODEL_ORDER,
    build_figure_3d,
    figure_config_3d,
    write_figure_html_3d,
)
from spice_parser import parse_spice  # noqa: E402

# model_id -> { net_key -> DualGeometry3D }
ModelBundle = Dict[str, Dict[str, DualGeometry3D]]


def load_netlist(path: Union[str, Path]):
    """Load from a ``.spice`` file or a flow ``OUT/`` directory containing ``net.npz``."""
    path = Path(path)
    if path.is_dir():
        return load_net(path)
    if path.suffix.lower() in {".sp", ".spice"} or path.is_file():
        if path.name == NET_NPZ:
            return load_net(path.parent)
        return parse_spice(path)
    raise FileNotFoundError(f"expected .spice file or OUT dir with {NET_NPZ}: {path}")


def _parse_layer(text: str) -> Optional[int]:
    t = text.strip().lower()
    if t in {"all", "*", "total"}:
        return None
    return int(t)


def _fill_port_meta(geom: DualGeometry3D, ports) -> DualGeometry3D:
    for k, m in enumerate(geom.pad_meta):
        m["ix"] = k
        m["iy"] = 0
        m["pad_node"] = ports.pad_nodes[k] if k < len(ports.pad_nodes) else ""
        m["voltage"] = ports.pad_voltages[k] if k < len(ports.pad_voltages) else None
    for k, m in enumerate(geom.sink_meta):
        if k < len(ports.cells):
            c = ports.cells[k]
            m["ix"] = c.ix
            m["iy"] = c.iy
            m["current"] = c.current
            m["node"] = c.node
        else:
            m["ix"] = k
            m["iy"] = 0
            m["current"] = 0.0
            m["node"] = ""
    return geom


def _r_stats_from_fit(
    fit: dict, *, ew_keys, ns_keys, via_keys
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[dict]]:
    sm = fit.get("summary_means") or {}
    Rx = Ry = Rvia = None
    for k in ew_keys:
        if sm.get(k) is not None:
            Rx = float(sm[k]) or None
            break
    for k in ns_keys:
        if sm.get(k) is not None:
            Ry = float(sm[k]) or None
            break
    via_vals = [float(sm[k]) for k in via_keys if sm.get(k) is not None]
    if via_vals:
        Rvia = sum(via_vals) / len(via_vals)
    R_edge_stats = fit.get("R_edge_stats") or (fit.get("extract_meta") or {}).get(
        "R_edge_stats"
    )
    return Rx, Ry, Rvia, R_edge_stats


def _try_load_tri_stagger_only(
    out_dir: Path, layer_z: dict, z_pitch: float
) -> Optional[DualGeometry3D]:
    has_model = (out_dir / TRI_STAGGER_MODEL_JSON).is_file() or (
        out_dir / "stagger_model.json"
    ).is_file()
    if not (out_dir / PORTS_JSON).is_file() or not has_model:
        return None
    dual_ports, _ = load_ports(out_dir)
    fields = load_tri_stagger_model_fields(out_dir)
    pad_xy = [(float(x), float(y)) for x, y in fields.get("pad_xy", dual_ports.pad_xy)]
    sink_xy = [(float(x), float(y)) for x, y in fields.get("sink_xy", [])]
    if not sink_xy:
        sink_xy = [(c.x, c.y) for c in dual_ports.cells]
    ports = dual_ports
    try:
        from ports_dual import tri_stagger_ports_from_dual

        if (out_dir / "tri_stagger_ports.json").is_file() or (
            out_dir / "stagger_ports.json"
        ).is_file():
            ports = tri_stagger_ports_from_dual(dual_ports)
    except Exception:
        ports = dual_ports
    Rx = Ry = Rvia = None
    R_edge_stats = None
    r_json = out_dir / TRI_STAGGER_R_JSON
    if not r_json.is_file():
        r_json = out_dir / "stagger_r.json"
    if r_json.is_file():
        fit = json.loads(r_json.read_text())
        Rx, Ry, Rvia, R_edge_stats = _r_stats_from_fit(
            fit,
            ew_keys=("R_ew",),
            ns_keys=("R_ns",),
            via_keys=("R_via_pad", "R_via_sink"),
        )
    if R_edge_stats is None:
        R_edge_stats = fields.get("R_edge_stats") or (
            fields.get("extract_meta") or {}
        ).get("R_edge_stats")
    if R_edge_stats:
        R_edge_stats = {
            "ew": R_edge_stats.get("ew", R_edge_stats.get("ew_mid")),
            "ns": R_edge_stats.get("ns", R_edge_stats.get("ns_mid")),
            "via": R_edge_stats.get("via_pad", R_edge_stats.get("via")),
        }
    geom = build_tri_stagger_geometry_3d(
        mid_xy=[(float(x), float(y)) for x, y in fields.get("mid_xy", [])],
        ew_edges=fields.get("ew_edges", []),
        ns_edges=fields.get("ns_edges", []),
        pad_mid=fields.get("pad_mid", []),
        sink_mid=fields.get("sink_mid", []),
        pad_xy=pad_xy,
        sink_xy=sink_xy,
        layer_z=layer_z,
        z_pitch=z_pitch,
        top_layer=dual_ports.top_layer,
        bot_layer=dual_ports.bot_layer,
        Rx=Rx,
        Ry=Ry,
        Rvia=Rvia,
        R_edge_stats=R_edge_stats,
    )
    return _fill_port_meta(geom, ports)


def _try_load_tri_square_only(
    out_dir: Path, layer_z: dict, z_pitch: float
) -> Optional[DualGeometry3D]:
    if not (out_dir / PORTS_JSON).is_file() or not (
        out_dir / TRI_SQUARE_MODEL_JSON
    ).is_file():
        return None
    dual_ports, _ = load_ports(out_dir)
    fields = load_tri_square_model_fields(out_dir)
    ports = dual_ports
    try:
        from ports_dual import tri_stagger_ports_from_dual

        if (out_dir / "tri_square_ports.json").is_file():
            ports = tri_stagger_ports_from_dual(dual_ports)
    except Exception:
        ports = dual_ports
    Rx = Ry = Rvia = None
    R_edge_stats = None
    if (out_dir / TRI_SQUARE_R_JSON).is_file():
        fit = json.loads((out_dir / TRI_SQUARE_R_JSON).read_text())
        sm = fit.get("summary_means") or {}
        # Keep all 6 R params (mid rx/ry, bot rx/ry, via1, via2).
        if sm.get("R_ew_u") is not None:
            Rx = float(sm["R_ew_u"])
        if sm.get("R_ns_u") is not None:
            Ry = float(sm["R_ns_u"])
        if sm.get("R_via_pad") is not None:
            Rvia = float(sm["R_via_pad"])
        R_edge_stats = fit.get("R_edge_stats") or (
            fit.get("extract_meta") or {}
        ).get("R_edge_stats")
        if isinstance(R_edge_stats, dict):
            R_edge_stats = dict(R_edge_stats)
            # Ensure mean keys exist for legend labels.
            for src, dst in (
                ("R_ew_u", "ew_u"),
                ("R_ns_u", "ns_u"),
                ("R_ew_l", "ew_l"),
                ("R_ns_l", "ns_l"),
                ("R_via_pad", "via_pad"),
                ("R_via_ul", "via_ul"),
            ):
                if sm.get(src) is not None:
                    slot = dict(R_edge_stats.get(dst) or {})
                    slot.setdefault("mean", float(sm[src]))
                    R_edge_stats[dst] = slot
    if R_edge_stats is None:
        R_edge_stats = fields.get("R_edge_stats") or (
            fields.get("extract_meta") or {}
        ).get("R_edge_stats")
    pad_xy = list(ports.pad_xy)
    sink_xy = [(c.x, c.y) for c in ports.cells]
    geom = build_tri_square_geometry_3d(
        l2_xy=[(float(x), float(y)) for x, y in fields.get("l2_xy", [])],
        l3_xy=[(float(x), float(y)) for x, y in fields.get("l3_xy", [])],
        ew_u=fields.get("ew_u", []),
        ns_u=fields.get("ns_u", []),
        ew_l=fields.get("ew_l", []),
        ns_l=fields.get("ns_l", []),
        vias_ul=fields.get("vias_ul", []),
        pad_attach=fields.get("pad_attach", []),
        sink_attach=fields.get("sink_attach", []),
        pad_xy=pad_xy,
        sink_xy=sink_xy,
        layer_z=layer_z,
        z_pitch=z_pitch,
        top_layer=dual_ports.top_layer,
        bot_layer=dual_ports.bot_layer,
        Rx=Rx,
        Ry=Ry,
        Rvia=Rvia,
        R_edge_stats=R_edge_stats,
    )
    return _fill_port_meta(geom, ports)


_LOADERS = {
    "tri_stagger": _try_load_tri_stagger_only,
    "tri_square": _try_load_tri_square_only,
}


def _resolve_flow_root(path: Path) -> Path:
    if (path / NET_NPZ).is_file() or (path / COMPONENTS_JSON).is_file():
        return path
    if path.name.startswith("comp") and (
        (path.parent / NET_NPZ).is_file() or (path.parent / COMPONENTS_JSON).is_file()
    ):
        return path.parent
    return path


def _load_models_for_comp(
    comp_dir: Path,
    layer_z: dict,
    z_pitch: float,
    net_key: str,
) -> Tuple[ModelBundle, list[str]]:
    models: ModelBundle = {}
    warns: list[str] = []
    for mid in MODEL_ORDER:
        loader = _LOADERS[mid]
        try:
            geom = loader(comp_dir, layer_z, z_pitch)
        except Exception as exc:  # noqa: BLE001 — keep other models loading
            warns.append(f"{mid}: {exc}")
            continue
        if geom is not None:
            models.setdefault(mid, {})[net_key] = geom
    if not models:
        warns.append(
            f"no abstraction models under {comp_dir} "
            f"(need one of {TRI_STAGGER_MODEL_JSON}, {TRI_SQUARE_MODEL_JSON})"
        )
    return models, warns


def _merge_bundles(dst: ModelBundle, src: ModelBundle) -> None:
    for mid, by_net in src.items():
        dst.setdefault(mid, {}).update(by_net)


def _load_models_for_out(
    out_dir: Path,
    layer_z: dict,
    z_pitch: float,
    *,
    only_comp: Optional[int] = None,
) -> Tuple[ModelBundle, list[str]]:
    models: ModelBundle = {}
    warns: list[str] = []
    root = _resolve_flow_root(out_dir)

    if (out_dir / PORTS_JSON).is_file():
        name = out_dir.name
        idx = None
        if name.startswith("comp") and name[4:].isdigit():
            idx = int(name[4:])
        if only_comp is not None and idx is not None and idx != only_comp:
            return models, warns
        key = f"vdd{idx}" if idx is not None else "vdd1"
        bundle, w = _load_models_for_comp(out_dir, layer_z, z_pitch, key)
        _merge_bundles(models, bundle)
        warns.extend(w)
        return models, warns

    if (root / COMPONENTS_JSON).is_file():
        idx = load_components(root)
        for entry in idx.get("components", []):
            k = int(entry["index"])
            if only_comp is not None and k != only_comp:
                continue
            comp_dir = root / entry["path"]
            bundle, w = _load_models_for_comp(
                comp_dir, layer_z, z_pitch, f"vdd{k}"
            )
            _merge_bundles(models, bundle)
            warns.extend(w)
        return models, warns

    bundle, w = _load_models_for_comp(out_dir, layer_z, z_pitch, "vdd1")
    _merge_bundles(models, bundle)
    warns.extend(w)
    return models, warns


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Visualize IBM PG metal geometry in 3D: original stack (left) and "
            "selectable abstraction model (right)"
        )
    )
    p.add_argument(
        "input",
        help="IBM .spice path or flow OUT directory (with net.npz)",
    )
    p.add_argument(
        "--model",
        choices=list(MODEL_ORDER),
        default=None,
        help="initial abstraction model (default: first available)",
    )
    p.add_argument(
        "--mode",
        choices=("both", "original", "dual", "pixel"),
        default="both",
        help="legacy; split view is used whenever any abstraction is present",
    )
    p.add_argument(
        "--net",
        default="vdd1",
        help="initial net view: vdd1..vddN or vss (default: vdd1; 'vdd' = vdd1)",
    )
    p.add_argument(
        "--comp",
        type=int,
        default=None,
        help="initial VDD component index (1-based); sets --net to vddK",
    )
    p.add_argument(
        "--layer",
        default="all",
        help="initial layer: 'all' or integer metal index (default: all)",
    )
    p.add_argument(
        "--z-pitch",
        type=float,
        default=None,
        help="synthetic layer height pitch (default: bbox diagonal / 20)",
    )
    p.add_argument(
        "--html",
        type=Path,
        default=None,
        help="output HTML path (default: OUT/pdn_view_3d.html)",
    )
    p.add_argument(
        "--browser",
        action="store_true",
        help="open the HTML in a browser after writing (default: off)",
    )
    p.add_argument(
        "--no-vias",
        action="store_true",
        help="hide original vias in the initial view",
    )
    p.add_argument(
        "--png",
        type=Path,
        default=None,
        help="PNG snapshot path (default: same stem as --html)",
    )
    p.add_argument(
        "--no-png",
        action="store_true",
        help="skip writing a static PNG snapshot beside the HTML",
    )
    args = p.parse_args(argv)

    src = Path(args.input)
    net = load_netlist(src)
    geom2d = extract_pdn_geometry(net)
    z_pitch = args.z_pitch if args.z_pitch is not None else default_z_pitch(geom2d.bbox)
    geom3d = lift_pdn_geometry(geom2d, z_pitch=z_pitch)
    layer = _parse_layer(args.layer)

    print(
        f"Found {geom3d.n_vdd_components} VDD connected component(s); "
        f"views are VDD1..VDD{geom3d.n_vdd_components} (topmost-first) + VSS"
    )

    if args.comp is not None:
        if args.comp < 1 or args.comp > geom3d.n_vdd_components:
            print(
                f"error: --comp {args.comp} out of range "
                f"[1, {geom3d.n_vdd_components}]",
                file=sys.stderr,
            )
            return 2
        net_name = f"vdd{args.comp}"
    else:
        net_name = normalize_net_arg(args.net)

    models: ModelBundle = {}
    if src.is_dir() or (src.is_file() and src.name == NET_NPZ):
        out_dir = src if src.is_dir() else src.parent
        models, warns = _load_models_for_out(
            out_dir,
            geom3d.layer_z,
            geom3d.z_pitch,
            only_comp=args.comp,
        )
        for w in warns:
            print(f"warning: {w}", file=sys.stderr)
    else:
        print(
            "warning: abstraction overlays require a completed flow OUT directory; "
            "rendering Original only",
            file=sys.stderr,
        )

    available = [m for m in MODEL_ORDER if m in models]
    if args.model is not None and args.model not in models:
        print(
            f"error: --model {args.model} not available "
            f"(found: {available or 'none'})",
            file=sys.stderr,
        )
        return 2
    default_model = args.model or (available[0] if available else None)

    fig = build_figure_3d(
        geom3d,
        models=models,
        default_model=default_model,
        default_net=net_name,
        default_layer=layer,
        show_vias=not args.no_vias,
    )

    if args.html is not None:
        html_path = args.html
    elif src.is_dir():
        html_path = src / "pdn_view_3d.html"
    else:
        html_path = Path.cwd() / f"{src.stem}_pdn_3d.html"

    html_path = html_path.resolve()
    png_out = write_figure_html_3d(
        fig,
        html_path,
        config=figure_config_3d(),
        png=not args.no_png,
        png_path=args.png,
    )

    print(f"wrote {html_path}")
    if png_out is not None:
        print(f"wrote {png_out}")
    for g in geom3d.vdd_components:
        print(
            f"{net_display_label(g.name)} layers={g.layers} "
            f"metals={g.n_metal} vias={len(g.vias)}  z_pitch={geom3d.z_pitch:.6g}"
        )
    print(
        f"VSS layers={geom3d.vss.layers} metals={geom3d.vss.n_metal} "
        f"vias={len(geom3d.vss.vias)}"
    )
    if models:
        print("Abstractions loaded:")
        for mid in MODEL_ORDER:
            if mid not in models:
                continue
            for key, geom in sorted(models[mid].items()):
                print(
                    f"  {MODEL_LABELS.get(mid, mid)} / {net_display_label(key)}: "
                    f"pads={geom.n_pads} sinks={geom.n_sinks} "
                    f"ew={geom.n_rx} ns={geom.n_ry} via={geom.n_rvia}"
                )
        missing = [MODEL_LABELS[m] for m in MODEL_ORDER if m not in models]
        if missing:
            print(f"  (not present: {', '.join(missing)})")
    else:
        print("Abstractions: (none loaded)")

    if args.browser:
        webbrowser.open(html_path.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
