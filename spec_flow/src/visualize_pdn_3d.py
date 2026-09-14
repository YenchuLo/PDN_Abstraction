#!/usr/bin/env python3
"""Interactive Plotly 3D PDN visualizer (original stack + Pixel-R overlay)."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from io_artifacts import (
    COMPONENTS_JSON,
    NET_NPZ,
    PIXEL_MODEL_JSON,
    PIXEL_R_JSON,
    PORTS_JSON,
    load_components,
    load_net,
    load_pixel_model_fields,
    load_ports,
)
from pdn_geometry import extract_pdn_geometry, net_display_label, normalize_net_arg
from pdn_geometry_3d import default_z_pitch, lift_pdn_geometry
from pdn_plot_3d import build_figure_3d, figure_config_3d, write_figure_png_3d
from pixel_geometry_3d import PixelGeometry3D, build_pixel_geometry_3d
from spice_parser import parse_spice


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


def _try_load_pixel(
    out_dir: Path,
    layer_z: dict,
    z_pitch: float,
) -> Tuple[Optional[PixelGeometry3D], Optional[str]]:
    """
    Load Pixel-R drawable from OUT / component artifacts.

    Returns (geometry, warning_message). warning_message is set when Pixel-R
    cannot be loaded.
    """
    model_path = out_dir / PIXEL_MODEL_JSON
    localized_path = out_dir / "localized_model.json"
    if not (out_dir / PORTS_JSON).is_file() or not (
        model_path.is_file() or localized_path.is_file()
    ):
        return None, (
            f"Pixel-R overlay requires {PORTS_JSON} and "
            f"{PIXEL_MODEL_JSON} (or localized_model.json) "
            f"under {out_dir}; rendering Original only"
        )

    ports, _node_map = load_ports(out_dir)
    if model_path.is_file():
        fields = load_pixel_model_fields(out_dir)
    else:
        fields = json.loads(localized_path.read_text())
    pad_attach = [int(x) for x in fields.get("pad_attach", [])]
    ew_shared = fields.get("ew_shared", fields.get("ew_edges", []))
    ns_shared = fields.get("ns_shared", fields.get("ns_edges", []))
    ew_boundary = fields.get("ew_boundary", [])
    ns_boundary = fields.get("ns_boundary", [])

    Rx = Ry = Rz = None
    fit_path = out_dir / PIXEL_R_JSON
    if not fit_path.is_file():
        loc_r = out_dir / "localized_r.json"
        if loc_r.is_file():
            fit_path = loc_r
    if fit_path.is_file():
        fit = json.loads(fit_path.read_text())
        if "Rx" in fit:
            Rx = float(fit["Rx"])
        elif "Rx_uniform" in fit:
            Rx = float(fit["Rx_uniform"])
        if "Ry" in fit:
            Ry = float(fit["Ry"])
        elif "Ry_uniform" in fit:
            Ry = float(fit["Ry_uniform"])
        if "Rz" in fit:
            Rz = float(fit["Rz"])
        elif "Rz_uniform" in fit:
            Rz = float(fit["Rz_uniform"])

    geom = build_pixel_geometry_3d(
        ports,
        pad_attach,
        ew_shared,
        ns_shared,
        layer_z,
        z_pitch,
        ew_boundary=ew_boundary,
        ns_boundary=ns_boundary,
        Rx=Rx,
        Ry=Ry,
        Rz=Rz,
    )
    return geom, None


def _resolve_flow_root(path: Path) -> Path:
    """Directory that holds net.npz / components.json."""
    if (path / NET_NPZ).is_file() or (path / COMPONENTS_JSON).is_file():
        return path
    if path.name.startswith("comp") and (
        (path.parent / NET_NPZ).is_file() or (path.parent / COMPONENTS_JSON).is_file()
    ):
        return path.parent
    return path


def _load_pixels_for_out(
    out_dir: Path,
    layer_z: dict,
    z_pitch: float,
    *,
    only_comp: Optional[int] = None,
) -> Tuple[Dict[str, PixelGeometry3D], list[str]]:
    """Load per-component Pixel-R maps keyed by ``vddk``."""
    pixels: Dict[str, PixelGeometry3D] = {}
    warns: list[str] = []
    root = _resolve_flow_root(out_dir)

    # Direct component directory (…/compk) with ports at this level.
    if (out_dir / PORTS_JSON).is_file():
        name = out_dir.name
        idx = None
        if name.startswith("comp") and name[4:].isdigit():
            idx = int(name[4:])
        if only_comp is not None and idx is not None and idx != only_comp:
            return pixels, warns
        pix, warn = _try_load_pixel(out_dir, layer_z, z_pitch)
        if pix is not None:
            key = f"vdd{idx}" if idx is not None else "vdd1"
            pixels[key] = pix
        elif warn:
            warns.append(warn)
        return pixels, warns

    # Flow OUT root with components.json
    if (root / COMPONENTS_JSON).is_file():
        idx = load_components(root)
        for entry in idx.get("components", []):
            k = int(entry["index"])
            if only_comp is not None and k != only_comp:
                continue
            comp_dir = root / entry["path"]
            pix, warn = _try_load_pixel(comp_dir, layer_z, z_pitch)
            if pix is not None:
                pixels[f"vdd{k}"] = pix
            elif warn:
                warns.append(f"comp{k}: {warn}")
        return pixels, warns

    # Legacy flat OUT (pre-multi-comp)
    pix, warn = _try_load_pixel(out_dir, layer_z, z_pitch)
    if pix is not None:
        pixels["vdd1"] = pix
    elif warn:
        warns.append(warn)
    return pixels, warns


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Visualize IBM PG metal geometry in 3D (original stack + optional "
            "Pixel-R abstraction overlay)"
        )
    )
    p.add_argument(
        "input",
        help="IBM .spice path or flow OUT directory (with net.npz)",
    )
    p.add_argument(
        "--mode",
        choices=("both", "original", "pixel"),
        default="both",
        help="initial view mode (default: both; falls back to original if no Pixel-R)",
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
        help="output HTML path (default: <stem>_pdn_3d.html or OUT/pdn_view_3d.html)",
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

    pixels: Dict[str, PixelGeometry3D] = {}
    mode = args.mode
    if src.is_dir() or (src.is_file() and src.name == NET_NPZ):
        out_dir = src if src.is_dir() else src.parent
        pixels, warns = _load_pixels_for_out(
            out_dir,
            geom3d.layer_z,
            geom3d.z_pitch,
            only_comp=args.comp,
        )
        for w in warns:
            print(f"warning: {w}", file=sys.stderr)
        if not pixels and mode in ("pixel", "both"):
            mode = "original"
    else:
        print(
            "warning: Pixel-R overlay requires a completed flow OUT directory; "
            "rendering Original only",
            file=sys.stderr,
        )
        if mode in ("pixel", "both"):
            mode = "original"

    if not pixels and args.mode == "pixel":
        print(
            "error: --mode pixel requested but Pixel-R artifacts are unavailable",
            file=sys.stderr,
        )
        return 2

    fig = build_figure_3d(
        geom3d,
        pixels=pixels,
        default_mode=mode,
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
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(html_path),
        include_plotlyjs=True,
        full_html=True,
        config=figure_config_3d(),
    )
    png_out = None
    if not args.no_png:
        png_path = (
            Path(args.png).resolve()
            if args.png is not None
            else html_path.with_suffix(".png")
        )
        png_out = write_figure_png_3d(fig, png_path)

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
    if pixels:
        for key, pixel in sorted(pixels.items()):
            print(
                f"Pixel-R {net_display_label(key)}: pads={pixel.n_pads} "
                f"sinks={pixel.n_sinks} Rx_edges={pixel.n_rx} Ry_edges={pixel.n_ry} "
                f"Rz={pixel.n_rz} Rx={pixel.Rx} Ry={pixel.Ry} Rz={pixel.Rz}"
            )
    else:
        print("Pixel-R: (not loaded)")

    if args.browser:
        webbrowser.open(html_path.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
