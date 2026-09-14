#!/usr/bin/env python3
"""Interactive Plotly PDN visualizer (VDD1..VDDN/VSS, per-layer / total XY views)."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from typing import Optional, Union

from io_artifacts import NET_NPZ, load_net
from pdn_geometry import extract_pdn_geometry, net_display_label, normalize_net_arg
from pdn_plot import build_figure, figure_config
from spice_parser import parse_spice


def load_netlist(path: Union[str, Path]):
    """Load from a ``.spice`` file or a flow ``OUT/`` directory containing ``net.npz``."""
    path = Path(path)
    if path.is_dir():
        return load_net(path)
    if path.suffix.lower() in {".sp", ".spice"} or path.is_file():
        # Prefer net.npz if user pointed at a file named oddly; otherwise parse.
        if path.name == NET_NPZ:
            return load_net(path.parent)
        return parse_spice(path)
    raise FileNotFoundError(f"expected .spice file or OUT dir with {NET_NPZ}: {path}")


def _parse_layer(text: str) -> Optional[int]:
    t = text.strip().lower()
    if t in {"all", "*", "total"}:
        return None
    return int(t)


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description="Visualize IBM PG metal geometry (VDD components / VSS) in Plotly"
    )
    p.add_argument(
        "input",
        help="IBM .spice path or flow OUT directory (with net.npz)",
    )
    p.add_argument(
        "--net",
        default="vdd1",
        help="initial net view: vdd1..vddN or vss (default: vdd1; 'vdd' = vdd1)",
    )
    p.add_argument(
        "--layer",
        default="all",
        help="initial layer: 'all' or integer metal index (default: all)",
    )
    p.add_argument(
        "--html",
        type=Path,
        default=None,
        help="output HTML path (default: <stem>_pdn.html in CWD or OUT/)",
    )
    p.add_argument(
        "--browser",
        action="store_true",
        help="open the HTML in a browser after writing (default: off)",
    )
    p.add_argument(
        "--no-vias",
        action="store_true",
        help="hide vias in the initial view",
    )
    args = p.parse_args(argv)

    src = Path(args.input)
    net = load_netlist(src)
    geom = extract_pdn_geometry(net)
    layer = _parse_layer(args.layer)
    net_name = normalize_net_arg(args.net)

    print(
        f"Found {geom.n_vdd_components} VDD connected component(s); "
        f"views are VDD1..VDD{geom.n_vdd_components} (topmost-first) + VSS"
    )

    fig = build_figure(
        geom,
        default_net=net_name,
        default_layer=layer,
        show_vias=not args.no_vias,
    )

    if args.html is not None:
        html_path = args.html
    elif src.is_dir():
        html_path = src / "pdn_view.html"
    else:
        html_path = Path.cwd() / f"{src.stem}_pdn.html"

    html_path = html_path.resolve()
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(html_path),
        include_plotlyjs=True,
        full_html=True,
        config=figure_config(),
    )

    print(f"wrote {html_path}")
    for g in geom.vdd_components:
        print(
            f"{net_display_label(g.name)} layers={g.layers} "
            f"metals={g.n_metal} vias={len(g.vias)}"
        )
    print(
        f"VSS layers={geom.vss.layers} metals={geom.vss.n_metal} vias={len(geom.vss.vias)}"
    )

    if args.browser:
        webbrowser.open(html_path.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
