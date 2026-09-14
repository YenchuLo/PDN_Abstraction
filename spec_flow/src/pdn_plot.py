"""Plotly interactive XY viewer for PDN metal layers (VDD / VSS)."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pdn_geometry import (
    PdnGeometry,
    net_display_label,
    normalize_net_arg,
    segments_to_xy,
    vias_to_xy,
)

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
_SEGMENT_WARN = 200_000

_FONT = "IBM Plex Sans, Segoe UI, Helvetica Neue, Arial, sans-serif"
_BG_PAPER = "#f8fafc"
_BG_PLOT = "#ffffff"
_GRID = "#e2e8f0"
_AXIS = "#64748b"
_TEXT = "#0f172a"
_ACCENT = "#0f766e"


def _layer_color(layer: int) -> str:
    return _LAYER_COLORS[int(layer) % len(_LAYER_COLORS)]


def _scatter_cls(n_pts: int):
    import plotly.graph_objects as go

    return go.Scattergl if n_pts >= 8_000 else go.Scatter


def _metal_trace(
    layer: int, segs: Sequence[Tuple[float, float, float, float]], net: str
):
    xs, ys = segments_to_xy(list(segs))
    n_pts = sum(1 for v in xs if v is not None)
    cls = _scatter_cls(n_pts)
    return cls(
        x=xs,
        y=ys,
        mode="lines",
        name=f"M{layer}",
        line=dict(color=_layer_color(layer), width=1.35),
        hovertemplate=f"{net_display_label(net)} · M{layer}<extra></extra>",
        legendgroup=f"{net}_metal_{layer}",
        showlegend=True,
    )


def _via_trace(
    net: str,
    xs: List[Optional[float]],
    ys: List[Optional[float]],
    *,
    showlegend: bool,
):
    n_pts = sum(1 for v in xs if v is not None)
    cls = _scatter_cls(max(n_pts, 1))
    # Empty via set: keep a legend-capable trace (no visible marks).
    if n_pts == 0:
        xs, ys = [None], [None]
    return cls(
        x=xs,
        y=ys,
        mode="markers",
        name="Vias",
        marker=dict(
            color=_VIA_COLOR,
            size=5,
            symbol="circle",
            opacity=0.65,
            line=dict(width=0),
        ),
        hovertemplate=f"{net_display_label(net)} · via<extra></extra>",
        legendgroup=f"{net}_vias",
        showlegend=showlegend,
    )


def _visibility(
    metas: List[Dict[str, Any]],
    net: str,
    layer: Optional[int],
    show_vias: bool,
) -> List[bool]:
    """``layer is None`` means all layers of ``net``."""
    vis: List[bool] = []
    for m in metas:
        if m["net"] != net:
            vis.append(False)
            continue
        if m["kind"] == "via":
            if not show_vias:
                vis.append(False)
            elif layer is None:
                vis.append(m["layer"] is None)
            else:
                vis.append(m["layer"] == layer)
            continue
        if layer is None:
            vis.append(True)
        else:
            vis.append(m["layer"] == layer)
    return vis


def _legend_flags(
    metas: List[Dict[str, Any]],
    net: str,
    layer: Optional[int],
    show_vias: bool,
) -> List[bool]:
    """
    Only the active view's traces appear in the legend (one Vias entry).
    Hidden nets/layers keep showlegend=False so the legend stays tidy.
    """
    flags: List[bool] = []
    for m in metas:
        if m["net"] != net:
            flags.append(False)
            continue
        if m["kind"] == "metal":
            flags.append(layer is None or m["layer"] == layer)
            continue
        # vias: single legend entry for the active via trace
        if not show_vias:
            flags.append(False)
        elif layer is None:
            flags.append(m["layer"] is None)
        else:
            flags.append(m["layer"] == layer)
    return flags


def _view_choices(geom: PdnGeometry) -> List[Tuple[str, str, Optional[int]]]:
    choices: List[Tuple[str, str, Optional[int]]] = []
    for net_geom in geom.iter_drawable_nets():
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


def _title_text(net: str, layer: Optional[int]) -> str:
    layer_s = "all layers" if layer is None else f"metal {layer}"
    return f"PDN layout  ·  {net_display_label(net)}  ·  {layer_s}"


def _subtitle(geom: PdnGeometry, net: str, layer: Optional[int] = None) -> str:
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
    return (
        f"{n_metal:,} metal segments  ·  {n_vias:,} vias  ·  "
        f"{layer_s}  ·  click legend to toggle layers / vias"
    )


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


def _axis_style(range_xy: List[float]) -> Dict[str, Any]:
    return dict(
        range=range_xy,
        autorange=False,
        constrain="domain",
        showgrid=True,
        gridcolor=_GRID,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor="#cbd5e1",
        linewidth=1,
        ticks="outside",
        tickcolor="#cbd5e1",
        tickfont=dict(size=11, color=_AXIS, family=_FONT),
        title_font=dict(size=12, color=_AXIS, family=_FONT),
        tickformat=",.0f",
        separatethousands=True,
        exponentformat="none",
        showexponent="none",
        mirror=False,
        fixedrange=False,
    )


def build_figure(
    geom: PdnGeometry,
    *,
    default_net: str = "vdd1",
    default_layer: Optional[int] = None,
    show_vias: bool = True,
):
    """
    Interactive Plotly figure (top-down XY).

    Control: View dropdown (VDD1..VDDN/VSS × All | each metal layer), placed
    above the title. Legend on the right lists visible metals + Vias.
    """
    import plotly.graph_objects as go

    default_net = normalize_net_arg(default_net)
    valid = {g.name for g in geom.iter_drawable_nets()}
    if default_net not in valid:
        raise ValueError(
            f"default_net must be one of {sorted(valid)} (got {default_net!r})"
        )

    total_segs = sum(g.n_metal for g in geom.iter_drawable_nets())
    if total_segs > _SEGMENT_WARN:
        warnings.warn(
            f"PDN has {total_segs} metal segments; rendering may be slow",
            stacklevel=2,
        )

    traces = []
    metas: List[Dict[str, Any]] = []

    for net_geom in geom.iter_drawable_nets():
        net_name = net_geom.name
        for L in net_geom.layers:
            segs = net_geom.metal.get(L, [])
            traces.append(_metal_trace(L, segs, net_name))
            metas.append({"net": net_name, "layer": L, "kind": "metal"})

        xs_all, ys_all = vias_to_xy(net_geom.vias, layer=None)
        traces.append(
            _via_trace(net_name, xs_all, ys_all, showlegend=False)
        )
        metas.append({"net": net_name, "layer": None, "kind": "via"})
        for L in net_geom.layers:
            xs_L, ys_L = vias_to_xy(net_geom.vias, layer=L)
            traces.append(
                _via_trace(net_name, xs_L, ys_L, showlegend=False)
            )
            metas.append({"net": net_name, "layer": L, "kind": "via"})

    init_vis = _visibility(metas, default_net, default_layer, show_vias)
    init_leg = _legend_flags(metas, default_net, default_layer, show_vias)
    for tr, v, leg in zip(traces, init_vis, init_leg):
        tr.visible = v
        tr.showlegend = leg

    choices = _view_choices(geom)
    view_buttons = []
    for label, net, layer in choices:
        vis = _visibility(metas, net, layer, show_vias)
        leg = _legend_flags(metas, net, layer, show_vias)
        view_buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": vis, "showlegend": leg},
                    {"annotations": _chrome_annotations(geom, net, layer)},
                ],
            )
        )

    xmin, ymin, xmax, ymax = geom.bbox
    xaxis = _axis_style(_axis_range(xmin, xmax))
    yaxis = _axis_style(_axis_range(ymin, ymax))
    yaxis.update(scaleanchor="x", scaleratio=1)

    fig = go.Figure(data=traces)
    fig.update_layout(
        # Title lives in annotations (layout.title.y cannot exceed 1).
        title=None,
        paper_bgcolor=_BG_PAPER,
        plot_bgcolor=_BG_PLOT,
        font=dict(family=_FONT, color=_TEXT, size=13),
        xaxis=xaxis,
        yaxis=yaxis,
        xaxis_title=dict(text="x (layout units)", standoff=8),
        yaxis_title=dict(text="y (layout units)", standoff=8),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(size=11, family=_FONT, color=_TEXT),
            itemsizing="constant",
            tracegroupgap=4,
            title=dict(
                text="Legend",
                font=dict(size=11, color=_AXIS, family=_FONT),
            ),
        ),
        # View dropdown (top) → title → subtitle → plot; legend on the right.
        margin=dict(l=80, r=120, t=168, b=64),
        hovermode="closest",
        dragmode="pan",
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.0,
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
        annotations=_chrome_annotations(geom, default_net, default_layer),
    )
    return fig


def _chrome_annotations(
    geom: PdnGeometry, net: str, layer: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Top chrome: View label, title, subtitle (paper y may exceed 1)."""
    return [
        dict(
            text="View",
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
            text=_title_text(net, layer),
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
            text=_subtitle(geom, net, layer),
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


def figure_config() -> Dict[str, Any]:
    return dict(
        scrollZoom=True,
        displaylogo=False,
        modeBarButtonsToRemove=[
            "lasso2d",
            "select2d",
            "autoScale2d",
        ],
        toImageButtonOptions=dict(
            format="png",
            filename="pdn_view",
            scale=2,
        ),
    )
