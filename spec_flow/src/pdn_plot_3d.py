"""Plotly interactive 3D viewer: original PDN stack + Pixel-R abstraction."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

from pdn_geometry import net_display_label, normalize_net_arg
from pdn_geometry_3d import (
    PdnGeometry3D,
    segments_to_xyz,
    vias_to_xyz,
)
from pixel_geometry_3d import PixelGeometry3D, points_to_xyz

_LAYER_COLORS = [
    "#0284c7",
    "#d97706",
    "#059669",
    "#dc2626",
    "#7c3aed",
    "#0d9488",
    "#ea580c",
    "#475569",
]
_VIA_COLOR = "#334155"
_PAD_COLOR = "#d97706"
_SINK_COLOR = "#0d9488"
_RX_COLOR = "#0284c7"
_RY_COLOR = "#7c3aed"
_RZ_COLOR = "#1e293b"
_SEGMENT_WARN = 200_000

_FONT = "IBM Plex Sans, Segoe UI, Helvetica Neue, Arial, sans-serif"
_BG_PAPER = "#f8fafc"
_BG_PLOT = "#ffffff"
_GRID = "#e2e8f0"
_AXIS = "#64748b"
_TEXT = "#0f172a"
_ACCENT = "#0f766e"

Mode = str  # "original" | "pixel" | "both"


def _layer_color(layer: int) -> str:
    return _LAYER_COLORS[int(layer) % len(_LAYER_COLORS)]


def _axis_range(lo: float, hi: float) -> List[float]:
    if hi < lo:
        lo, hi = hi, lo
    span = hi - lo
    pad = max(span * 0.012, 1.0) if span > 0 else 1.0
    r0, r1 = lo - pad, hi + pad
    if lo >= 0.0:
        r0 = max(0.0, r0)
    if hi <= 0.0:
        r1 = min(0.0, r1)
    if r0 == r1:
        r1 = r0 + 1.0
    return [r0, r1]


def _iter_nets(geom: PdnGeometry3D):
    yield from geom.vdd_components
    yield geom.vss


def _view_choices(geom: PdnGeometry3D) -> List[Tuple[str, str, Optional[int]]]:
    choices: List[Tuple[str, str, Optional[int]]] = []
    for net_geom in _iter_nets(geom):
        label = net_display_label(net_geom.name)
        choices.append((f"{label}  ·  All layers", net_geom.name, None))
        for L in net_geom.layers:
            choices.append((f"{label}  ·  Metal {L}", net_geom.name, L))
    return choices


def _active_index(
    choices: List[Tuple[str, str, Optional[int]]],
    net: str,
    layer: Optional[int],
) -> int:
    for i, (_, n, L) in enumerate(choices):
        if n == net and L == layer:
            return i
    return 0


def _mode_index(mode: str) -> int:
    return {"original": 0, "pixel": 1, "both": 2}.get(mode, 2)


def _title_text(mode: str, net: str, layer: Optional[int]) -> str:
    layer_s = "all layers" if layer is None else f"metal {layer}"
    mode_s = {
        "original": "Original",
        "pixel": "Pixel-R",
        "both": "Original | Pixel-R",
    }.get(mode, mode)
    if mode == "pixel":
        return f"PDN 3D  ·  {mode_s}  ·  {net_display_label(net)}"
    return f"PDN 3D  ·  {mode_s}  ·  {net_display_label(net)}  ·  {layer_s}"


def _pixel_for_net(
    pixels: Optional[Dict[str, PixelGeometry3D]], net: str
) -> Optional[PixelGeometry3D]:
    if not pixels:
        return None
    return pixels.get(normalize_net_arg(net))


def _subtitle(
    geom: PdnGeometry3D,
    pixels: Optional[Dict[str, PixelGeometry3D]],
    mode: str,
    net: str,
    layer: Optional[int],
) -> str:
    parts: List[str] = []
    if mode in ("original", "both"):
        g = geom.net_geometry(net)
        if layer is None:
            n_metal = g.n_metal
            n_vias = len(g.vias)
            layer_s = ", ".join(f"M{L}" for L in g.layers) or "—"
        else:
            n_metal = len(g.metal.get(layer, []))
            n_vias = sum(
                1 for _x1, _y1, l1, _x2, _y2, l2 in g.vias if l1 == layer or l2 == layer
            )
            layer_s = f"M{layer}"
        parts.append(f"{n_metal:,} metal  ·  {n_vias:,} vias  ·  {layer_s}")
    pixel = _pixel_for_net(pixels, net)
    if pixel is not None and mode in ("pixel", "both"):
        r_bits = []
        if pixel.Rx is not None:
            r_bits.append(f"Rx={pixel.Rx:.3g}")
        if pixel.Ry is not None:
            r_bits.append(f"Ry={pixel.Ry:.3g}")
        if pixel.Rz is not None:
            r_bits.append(f"Rz={pixel.Rz:.3g}")
        pix = (
            f"Pixel-R  {pixel.n_pads} pads  ·  {pixel.n_sinks} sinks  ·  "
            f"{pixel.n_rx} Rx  ·  {pixel.n_ry} Ry  ·  {pixel.n_rz} Rz"
        )
        if r_bits:
            pix += "  ·  " + ", ".join(r_bits)
        parts.append(pix)
    if not parts:
        parts.append("no geometry")
    parts.append("orbit to rotate  ·  legend toggles traces")
    return "  ·  ".join(parts)


def _original_visibility(
    m: Dict[str, Any],
    net: str,
    layer: Optional[int],
    show_vias: bool,
) -> bool:
    """Whether a single original-family meta should be visible."""
    if m["net"] != net:
        return False
    if m["kind"] == "via":
        if not show_vias:
            return False
        if layer is None:
            return m["layer"] is None
        return m["layer"] == layer
    if layer is None:
        return True
    return m["layer"] == layer


def _visibility(
    metas: List[Dict[str, Any]],
    mode: str,
    net: str,
    layer: Optional[int],
    show_vias: bool,
) -> List[bool]:
    show_orig = mode in ("original", "both")
    show_pix = mode in ("pixel", "both")
    vis: List[bool] = []
    for m in metas:
        if m["family"] == "original":
            if not show_orig:
                vis.append(False)
            else:
                vis.append(_original_visibility(m, net, layer, show_vias))
        else:
            # Pixel overlay is per VDD component (net key).
            vis.append(show_pix and m.get("net") == net)
    return vis


def _legend_flags(
    metas: List[Dict[str, Any]],
    mode: str,
    net: str,
    layer: Optional[int],
    show_vias: bool,
) -> List[bool]:
    show_orig = mode in ("original", "both")
    show_pix = mode in ("pixel", "both")
    flags: List[bool] = []
    for m in metas:
        if m["family"] == "original":
            if not show_orig or m["net"] != net:
                flags.append(False)
                continue
            if m["kind"] == "metal":
                flags.append(layer is None or m["layer"] == layer)
            elif m["kind"] == "via":
                if not show_vias:
                    flags.append(False)
                elif layer is None:
                    flags.append(m["layer"] is None)
                else:
                    flags.append(m["layer"] == layer)
            else:
                flags.append(False)
        else:
            flags.append(
                show_pix and m.get("net") == net and m.get("legend", False)
            )
    return flags


def _metal_trace_3d(
    layer: int,
    segs,
    net: str,
    *,
    scene: str = "scene",
    legend: str = "legend",
):
    import plotly.graph_objects as go

    xs, ys, zs = segments_to_xyz(list(segs))
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        name=f"M{layer}",
        line=dict(color=_layer_color(layer), width=3),
        hovertemplate=f"{net_display_label(net)} · M{layer}<extra></extra>",
        legendgroup=f"{net}_metal_{layer}",
        showlegend=True,
        scene=scene,
        legend=legend,
    )


def _via_trace_3d(
    net: str,
    xs,
    ys,
    zs,
    *,
    showlegend: bool,
    scene: str = "scene",
    legend: str = "legend",
):
    import plotly.graph_objects as go

    n_pts = sum(1 for v in xs if v is not None)
    if n_pts == 0:
        xs, ys, zs = [None], [None], [None]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        name="Vias",
        line=dict(color=_VIA_COLOR, width=2),
        hovertemplate=f"{net_display_label(net)} · via<extra></extra>",
        legendgroup=f"{net}_vias",
        showlegend=showlegend,
        opacity=0.75,
        scene=scene,
        legend=legend,
    )


def _pixel_r_str(pixel: PixelGeometry3D) -> str:
    bits = []
    if pixel.Rx is not None:
        bits.append(f"Rx={pixel.Rx:.4g}")
    if pixel.Ry is not None:
        bits.append(f"Ry={pixel.Ry:.4g}")
    if pixel.Rz is not None:
        bits.append(f"Rz={pixel.Rz:.4g}")
    return ", ".join(bits) if bits else "Pixel-R"


def _add_pixel_traces(
    traces: list,
    metas: List[Dict[str, Any]],
    pixel: PixelGeometry3D,
    net: str,
    *,
    scene: str = "scene",
    legend: str = "legend",
) -> None:
    import plotly.graph_objects as go

    rlabel = _pixel_r_str(pixel)
    net_key = normalize_net_arg(net)
    lg = f"{net_key}_pixel"

    # Pads
    px, py, pz = points_to_xyz(pixel.pads)
    if not px:
        px, py, pz = [None], [None], [None]
    pad_text = [
        (
            f"Pad ({m['ix']},{m['iy']})<br>"
            f"node={m['pad_node']}<br>"
            f"V={m['voltage']}<br>{rlabel}"
        )
        for m in pixel.pad_meta
    ] or [None]
    traces.append(
        go.Scatter3d(
            x=px,
            y=py,
            z=pz,
            mode="markers",
            name="Pads",
            marker=dict(
                color=_PAD_COLOR,
                size=7,
                symbol="diamond",
                opacity=1.0,
                line=dict(width=1, color="#ffffff"),
            ),
            text=pad_text,
            hovertemplate="%{text}<extra></extra>",
            legendgroup=f"{lg}_pads",
            showlegend=True,
            scene=scene,
            legend=legend,
        )
    )
    metas.append(
        {"family": "pixel", "net": net_key, "kind": "pads", "legend": True}
    )

    # Sinks — larger + outlined so they read at the bottom z-plane
    sx, sy, sz = points_to_xyz(pixel.sinks)
    if not sx:
        sx, sy, sz = [None], [None], [None]
    sink_text = [
        (
            f"Sink ({m['ix']},{m['iy']})<br>"
            f"I={m['current']:.4g}<br>"
            f"node={m['node']}<br>{rlabel}"
        )
        for m in pixel.sink_meta
    ] or [None]
    traces.append(
        go.Scatter3d(
            x=sx,
            y=sy,
            z=sz,
            mode="markers",
            name="Sinks",
            marker=dict(
                color=_SINK_COLOR,
                size=9,
                symbol="circle",
                opacity=1.0,
                line=dict(width=1.5, color="#042f2e"),
            ),
            text=sink_text,
            hovertemplate="%{text}<extra></extra>",
            legendgroup=f"{lg}_sinks",
            showlegend=True,
            scene=scene,
            legend=legend,
        )
    )
    metas.append(
        {"family": "pixel", "net": net_key, "kind": "sinks", "legend": True}
    )

    def _edge_trace(name: str, segs, color: str, kind: str):
        xs, ys, zs = segments_to_xyz(list(segs))
        if not xs:
            xs, ys, zs = [None], [None], [None]
        traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                name=name,
                line=dict(color=color, width=4),
                hovertemplate=f"{name}<br>{rlabel}<extra></extra>",
                legendgroup=f"{lg}_{kind}",
                showlegend=True,
                scene=scene,
                legend=legend,
            )
        )
        metas.append(
            {"family": "pixel", "net": net_key, "kind": kind, "legend": True}
        )

    _edge_trace("Rx edges", pixel.rx_edges, _RX_COLOR, "rx")
    _edge_trace("Ry edges", pixel.ry_edges, _RY_COLOR, "ry")
    _edge_trace("Rz vias", pixel.rz_edges, _RZ_COLOR, "rz")


def _chrome_annotations(
    geom: PdnGeometry3D,
    pixels: Optional[Dict[str, PixelGeometry3D]],
    mode: str,
    net: str,
    layer: Optional[int],
    *,
    dual: bool = False,
) -> List[Dict[str, Any]]:
    anns: List[Dict[str, Any]] = [
        dict(
            text="Mode",
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.325,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=11, color=_ACCENT, family=_FONT),
        ),
        dict(
            text="View",
            xref="paper",
            yref="paper",
            x=0.22,
            y=1.325,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=11, color=_ACCENT, family=_FONT),
        ),
        dict(
            text=_title_text(mode, net, layer),
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.14,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=18, color=_TEXT, family=_FONT),
        ),
        dict(
            text=_subtitle(geom, pixels, mode, net, layer),
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.06,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=11.5, color=_AXIS, family=_FONT),
        ),
    ]
    if dual:
        anns.extend(
            [
                dict(
                    text="Original",
                    xref="paper",
                    yref="paper",
                    x=0.22,
                    y=1.0,
                    xanchor="center",
                    yanchor="bottom",
                    showarrow=False,
                    font=dict(size=13, color=_TEXT, family=_FONT),
                ),
                dict(
                    text="Pixel-R",
                    xref="paper",
                    yref="paper",
                    x=0.78,
                    y=1.0,
                    xanchor="center",
                    yanchor="bottom",
                    showarrow=False,
                    font=dict(size=13, color=_TEXT, family=_FONT),
                ),
            ]
        )
    return anns


def _scene_axis_kwargs(title: str, rng: List[float]) -> Dict[str, Any]:
    return dict(
        title=title,
        range=rng,
        backgroundcolor=_BG_PLOT,
        gridcolor=_GRID,
        showbackground=True,
        tickfont=dict(size=11, color=_AXIS, family=_FONT),
    )


def _scene_layout(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    zmin: float,
    zmax: float,
    *,
    domain_x: Optional[List[float]] = None,
) -> Dict[str, Any]:
    scene: Dict[str, Any] = dict(
        xaxis=_scene_axis_kwargs("x (layout units)", _axis_range(xmin, xmax)),
        yaxis=_scene_axis_kwargs("y (layout units)", _axis_range(ymin, ymax)),
        zaxis=_scene_axis_kwargs("z (layer height)", _axis_range(zmin, zmax)),
        aspectmode="data",
        bgcolor=_BG_PAPER,
    )
    if domain_x is not None:
        scene["domain"] = dict(x=domain_x, y=[0.0, 0.95])
    return scene


def _legend_layout(
    *,
    x: float,
    xanchor: str = "left",
    title: str = "Legend",
) -> Dict[str, Any]:
    return dict(
        orientation="v",
        yanchor="top",
        y=0.95,
        xanchor=xanchor,
        x=x,
        bgcolor="rgba(255,255,255,0.96)",
        bordercolor="#e2e8f0",
        borderwidth=1,
        font=dict(size=11, family=_FONT, color=_TEXT),
        itemsizing="constant",
        tracegroupgap=4,
        title=dict(
            text=title,
            font=dict(size=11, color=_AXIS, family=_FONT),
        ),
    )


def build_figure_3d(
    geom: PdnGeometry3D,
    pixel: Optional[PixelGeometry3D] = None,
    *,
    pixels: Optional[Dict[str, PixelGeometry3D]] = None,
    default_mode: str = "both",
    default_net: str = "vdd1",
    default_layer: Optional[int] = None,
    show_vias: bool = True,
):
    """
    Interactive Plotly 3D figure.

    Controls:
    - Mode dropdown: Original | Pixel-R | Both
    - View dropdown: VDD1..VDDN/VSS × All | each metal layer

    When Pixel-R data is present, Original (left) and Pixel-R (right) use
    side-by-side scenes with separate legends. A single ``pixel`` argument is
    treated as ``vdd1`` for back-compat.
    """
    import plotly.graph_objects as go

    pix_map: Dict[str, PixelGeometry3D] = {}
    if pixels:
        pix_map.update({normalize_net_arg(k): v for k, v in pixels.items()})
    if pixel is not None:
        pix_map.setdefault("vdd1", pixel)
    has_pixel = bool(pix_map)
    dual = has_pixel

    default_net = normalize_net_arg(default_net)
    valid = {g.name for g in _iter_nets(geom)}
    if default_net not in valid:
        raise ValueError(
            f"default_net must be one of {sorted(valid)} (got {default_net!r})"
        )
    default_mode = default_mode.lower()
    if default_mode not in ("original", "pixel", "both"):
        raise ValueError("default_mode must be 'original', 'pixel', or 'both'")
    if not has_pixel and default_mode in ("pixel", "both"):
        default_mode = "original"
    if (
        default_mode in ("pixel", "both")
        and _pixel_for_net(pix_map, default_net) is None
        and default_net != "vss"
    ):
        # Selected VDD has no Pixel-R; still allow original / other comps.
        if default_mode == "pixel":
            default_mode = "original"

    total_segs = sum(g.n_metal for g in _iter_nets(geom))
    if total_segs > _SEGMENT_WARN:
        warnings.warn(
            f"PDN has {total_segs} metal segments; 3D rendering may be slow",
            stacklevel=2,
        )

    traces = []
    metas: List[Dict[str, Any]] = []
    orig_scene = "scene"
    orig_legend = "legend"
    pix_scene = "scene2" if dual else "scene"
    pix_legend = "legend2" if dual else "legend"

    for net_geom in _iter_nets(geom):
        net_name = net_geom.name
        for L in net_geom.layers:
            segs = net_geom.metal.get(L, [])
            traces.append(
                _metal_trace_3d(
                    L, segs, net_name, scene=orig_scene, legend=orig_legend
                )
            )
            metas.append(
                {"family": "original", "net": net_name, "layer": L, "kind": "metal"}
            )

        xs_all, ys_all, zs_all = vias_to_xyz(
            net_geom.vias, geom.layer_z, geom.z_pitch, layer=None
        )
        traces.append(
            _via_trace_3d(
                net_name,
                xs_all,
                ys_all,
                zs_all,
                showlegend=False,
                scene=orig_scene,
                legend=orig_legend,
            )
        )
        metas.append(
            {"family": "original", "net": net_name, "layer": None, "kind": "via"}
        )
        for L in net_geom.layers:
            xs_L, ys_L, zs_L = vias_to_xyz(
                net_geom.vias, geom.layer_z, geom.z_pitch, layer=L
            )
            traces.append(
                _via_trace_3d(
                    net_name,
                    xs_L,
                    ys_L,
                    zs_L,
                    showlegend=False,
                    scene=orig_scene,
                    legend=orig_legend,
                )
            )
            metas.append(
                {"family": "original", "net": net_name, "layer": L, "kind": "via"}
            )

    for net_name, pix in pix_map.items():
        _add_pixel_traces(
            traces,
            metas,
            pix,
            net_name,
            scene=pix_scene,
            legend=pix_legend,
        )

    init_vis = _visibility(metas, default_mode, default_net, default_layer, show_vias)
    init_leg = _legend_flags(metas, default_mode, default_net, default_layer, show_vias)
    for tr, v, leg in zip(traces, init_vis, init_leg):
        tr.visible = v
        tr.showlegend = leg

    # Mode menu: uses CLI default net/layer for original filtering.
    mode_buttons = []
    mode_order = (
        ["original", "pixel", "both"] if has_pixel else ["original"]
    )
    mode_labels = {
        "original": "Original",
        "pixel": "Pixel-R",
        "both": "Both",
    }
    for mode in mode_order:
        vis = _visibility(metas, mode, default_net, default_layer, show_vias)
        leg = _legend_flags(metas, mode, default_net, default_layer, show_vias)
        mode_buttons.append(
            dict(
                label=mode_labels[mode],
                method="update",
                args=[
                    {"visible": vis, "showlegend": leg},
                    {
                        "annotations": _chrome_annotations(
                            geom,
                            pix_map,
                            mode,
                            default_net,
                            default_layer,
                            dual=dual,
                        )
                    },
                ],
            )
        )

    # View menu: applies Both (or Original if no pixel) + selected net/layer.
    view_mode = "both" if has_pixel else "original"
    choices = _view_choices(geom)
    view_buttons = []
    for label, net, layer in choices:
        vis = _visibility(metas, view_mode, net, layer, show_vias)
        leg = _legend_flags(metas, view_mode, net, layer, show_vias)
        view_buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": vis, "showlegend": leg},
                    {
                        "annotations": _chrome_annotations(
                            geom, pix_map, view_mode, net, layer, dual=dual
                        )
                    },
                ],
            )
        )

    xmin, ymin, xmax, ymax = geom.bbox
    zmin, zmax = geom.z_range
    for pix in pix_map.values():
        zmin = min(zmin, pix.z_sink, pix.z_pad)
        zmax = max(zmax, pix.z_sink, pix.z_pad)

    fig = go.Figure(data=traces)
    layout_kwargs: Dict[str, Any] = dict(
        title=None,
        paper_bgcolor=_BG_PAPER,
        font=dict(family=_FONT, color=_TEXT, size=13),
        margin=dict(l=40, r=40 if dual else 140, t=188 if dual else 168, b=40),
        hovermode="closest",
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.0,
                y=1.30,
                xanchor="left",
                yanchor="top",
                buttons=mode_buttons,
                showactive=True,
                active=_mode_index(default_mode) if has_pixel else 0,
                bgcolor="#ffffff",
                bordercolor="#94a3b8",
                borderwidth=1,
                font=dict(size=12.5, color=_TEXT, family=_FONT),
                pad=dict(l=12, r=12, t=8, b=8),
            ),
            dict(
                type="dropdown",
                direction="down",
                x=0.22,
                y=1.30,
                xanchor="left",
                yanchor="top",
                buttons=view_buttons,
                showactive=True,
                active=_active_index(choices, default_net, default_layer),
                bgcolor="#ffffff",
                bordercolor="#94a3b8",
                borderwidth=1,
                font=dict(size=12.5, color=_TEXT, family=_FONT),
                pad=dict(l=12, r=12, t=8, b=8),
            ),
        ],
        annotations=_chrome_annotations(
            geom, pix_map, default_mode, default_net, default_layer, dual=dual
        ),
    )
    if dual:
        layout_kwargs["legend"] = _legend_layout(
            x=0.455, xanchor="right", title="Original"
        )
        layout_kwargs["legend2"] = _legend_layout(
            x=1.0, xanchor="left", title="Pixel-R"
        )
        layout_kwargs["scene"] = _scene_layout(
            xmin, ymin, xmax, ymax, zmin, zmax, domain_x=[0.0, 0.45]
        )
        layout_kwargs["scene2"] = _scene_layout(
            xmin, ymin, xmax, ymax, zmin, zmax, domain_x=[0.55, 1.0]
        )
    else:
        layout_kwargs["legend"] = _legend_layout(x=1.01, title="Legend")
        layout_kwargs["scene"] = _scene_layout(
            xmin, ymin, xmax, ymax, zmin, zmax
        )
    fig.update_layout(**layout_kwargs)
    return fig


def figure_config_3d() -> Dict[str, Any]:
    return dict(
        displaylogo=False,
        toImageButtonOptions=dict(
            format="png",
            filename="pdn_view_3d",
            scale=2,
        ),
    )


def write_figure_png_3d(
    fig,
    png_path,
    *,
    width: int = 1600,
    height: int = 900,
    scale: float = 2.0,
):
    """Write a static PNG snapshot via matplotlib (no kaleido)."""
    from plotly_mpl_png import write_plotly_3d_png

    return write_plotly_3d_png(
        fig, png_path, width=width, height=height, scale=scale
    )































