"""Plotly interactive 3D viewer: original PDN (left) + selectable abstraction (right)."""

from __future__ import annotations


import sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
_SPEC = _HERE.parent.parent / "spec_flow" / "src"
if str(_SPEC) not in sys.path:
    sys.path.insert(0, str(_SPEC))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


import warnings
from typing import Any, Dict, List, Optional, Tuple

from pdn_geometry import net_display_label, normalize_net_arg
from pdn_geometry_3d import (
    PdnGeometry3D,
    segments_to_xyz,
    vias_to_xyz,
)
from dual_geometry_3d import DualGeometry3D, points_to_xyz

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

# Stable dropdown order for my_flow abstraction models.
# Active abstraction models in the 3D viewer (others kept loadable but hidden).
MODEL_ORDER = ("tri_stagger", "tri_square")
MODEL_LABELS = {
    "tri_stagger": "Tri-stagger",
    "tri_square": "Tri-square",
}


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


def _model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def _title_text(
    net: str, layer: Optional[int], *, model_id: Optional[str] = None
) -> str:
    layer_s = "all layers" if layer is None else f"metal {layer}"
    if model_id:
        return (
            f"PDN 3D  ·  Original | {_model_label(model_id)}  ·  "
            f"{net_display_label(net)}  ·  {layer_s}"
        )
    return f"PDN 3D  ·  Original  ·  {net_display_label(net)}  ·  {layer_s}"


def _geom_for_model_net(
    models: Optional[Dict[str, Dict[str, DualGeometry3D]]],
    model_id: Optional[str],
    net: str,
) -> Optional[DualGeometry3D]:
    if not models or not model_id:
        return None
    by_net = models.get(model_id) or {}
    return by_net.get(normalize_net_arg(net))


def _subtitle(
    geom: PdnGeometry3D,
    models: Optional[Dict[str, Dict[str, DualGeometry3D]]],
    model_id: Optional[str],
    net: str,
    layer: Optional[int],
) -> str:
    parts: List[str] = []
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
    parts.append(f"Original {n_metal:,} metal  ·  {n_vias:,} vias  ·  {layer_s}")
    pixel = _geom_for_model_net(models, model_id, net)
    if pixel is not None and model_id:
        r_bits = []
        stats = getattr(pixel, "R_edge_stats", None) or {}
        if any(k in stats for k in ("ew_u", "ew_l", "via_pad", "via_ul")):
            for label, key in (
                ("Mid_Rx", "ew_u"),
                ("Mid_Ry", "ns_u"),
                ("Bot_Rx", "ew_l"),
                ("Bot_Ry", "ns_l"),
                ("Via1", "via_pad"),
                ("Via2", "via_ul"),
            ):
                s = stats.get(key) or {}
                val = s.get("mean", s.get("median"))
                if val is not None:
                    r_bits.append(f"{label}={float(val):.3g}")
        else:
            for label, key, mean in (
                ("R_ew", "ew", pixel.Rx),
                ("R_ns", "ns", pixel.Ry),
                ("R_via", "via", pixel.Rvia),
            ):
                s = stats.get(key) or {}
                if s and s.get("n"):
                    r_bits.append(f"{label} med={s['median']:.3g}")
                elif mean is not None:
                    r_bits.append(f"mean_{label}={mean:.3g}")
        pix = (
            f"{_model_label(model_id)}  {pixel.n_pads} pads  ·  "
            f"{pixel.n_sinks} sinks  ·  {pixel.n_rx} ew  ·  "
            f"{pixel.n_ry} ns  ·  {pixel.n_rvia} via"
        )
        if r_bits:
            pix += "  ·  " + ", ".join(r_bits)
        parts.append(pix)
    elif model_id:
        parts.append(f"{_model_label(model_id)}: no geometry for this net")
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
    net: str,
    layer: Optional[int],
    show_vias: bool,
    *,
    model_id: Optional[str] = None,
) -> List[bool]:
    vis: List[bool] = []
    for m in metas:
        if m["family"] == "original":
            vis.append(_original_visibility(m, net, layer, show_vias))
        else:
            vis.append(
                bool(model_id)
                and m.get("model") == model_id
                and m.get("net") == net
            )
    return vis


def _legend_flags(
    metas: List[Dict[str, Any]],
    net: str,
    layer: Optional[int],
    show_vias: bool,
    *,
    model_id: Optional[str] = None,
) -> List[bool]:
    flags: List[bool] = []
    for m in metas:
        if m["family"] == "original":
            if m["net"] != net:
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
                bool(model_id)
                and m.get("model") == model_id
                and m.get("net") == net
                and m.get("legend", False)
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


def _pixel_r_str(pixel: DualGeometry3D) -> str:
    bits = []
    stats = getattr(pixel, "R_edge_stats", None) or {}

    def _fmt(label: str, key: str, mean: Optional[float] = None) -> None:
        s = stats.get(key) or {}
        if s.get("mean") is not None:
            bits.append(f"{label}={float(s['mean']):.4g}")
        elif s and s.get("n"):
            bits.append(
                f"{label}=[{s['min']:.3g}/{s['median']:.3g}/{s['max']:.3g}]"
            )
        elif mean is not None:
            bits.append(f"{label}={mean:.4g}")

    # Tri-square: 6 parameters (mid rx/ry, bot rx/ry, two vias).
    if any(k in stats for k in ("ew_u", "ew_l", "via_pad", "via_ul")):
        _fmt("Mid_Rx", "ew_u")
        _fmt("Mid_Ry", "ns_u")
        _fmt("Bot_Rx", "ew_l")
        _fmt("Bot_Ry", "ns_l")
        _fmt("Via1", "via_pad")
        _fmt("Via2", "via_ul")
        return ", ".join(bits) if bits else "tri-square (6 R)"

    _fmt("R_ew", "ew", pixel.Rx)
    _fmt("R_ns", "ns", pixel.Ry)
    _fmt("R_via", "via", pixel.Rvia)
    return ", ".join(bits) if bits else "per-edge Dual-Layer"


def _add_pixel_traces(
    traces: list,
    metas: List[Dict[str, Any]],
    pixel: DualGeometry3D,
    net: str,
    *,
    model_id: str,
    scene: str = "scene",
    legend: str = "legend",
) -> None:
    import plotly.graph_objects as go

    rlabel = _pixel_r_str(pixel)
    net_key = normalize_net_arg(net)
    lg = f"{model_id}_{net_key}"

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
        {
            "family": "abstract",
            "model": model_id,
            "net": net_key,
            "kind": "pads",
            "legend": True,
        }
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
        {
            "family": "abstract",
            "model": model_id,
            "net": net_key,
            "kind": "sinks",
            "legend": True,
        }
    )

    def _edge_trace(
        name: str, segs, color: str, kind: str, *, hover_r: Optional[str] = None
    ):
        xs, ys, zs = segments_to_xyz(list(segs))
        if not xs:
            xs, ys, zs = [None], [None], [None]
        hover = hover_r or rlabel
        traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                name=name,
                line=dict(color=color, width=4),
                hovertemplate=f"{name}<br>{hover}<extra></extra>",
                legendgroup=f"{lg}_{kind}",
                showlegend=True,
                scene=scene,
                legend=legend,
            )
        )
        metas.append(
            {
                "family": "abstract",
                "model": model_id,
                "net": net_key,
                "kind": kind,
                "legend": True,
            }
        )

    groups = getattr(pixel, "legend_groups", None) or None
    if groups:
        for g in groups:
            gname = str(g.get("name", "edge"))
            glabel = str(g.get("r_label") or "")
            hover = f"{gname}" + (f"<br>{glabel}" if glabel else f"<br>{rlabel}")
            _edge_trace(
                gname,
                g.get("segs") or [],
                str(g.get("color") or _RZ_COLOR),
                str(g.get("kind") or "edge"),
                hover_r=hover,
            )
    else:
        _edge_trace("E-W", pixel.rx_edges, _RX_COLOR, "rx")
        _edge_trace("N-S", pixel.ry_edges, _RY_COLOR, "ry")
        _edge_trace("Via / attach", pixel.rvia_edges, _RZ_COLOR, "rvia")


def _chrome_annotations(
    geom: PdnGeometry3D,
    models: Optional[Dict[str, Dict[str, DualGeometry3D]]],
    model_id: Optional[str],
    net: str,
    layer: Optional[int],
    *,
    dual: bool = False,
) -> List[Dict[str, Any]]:
    # Title/subtitle sit below the external HTML toolbar (Abstraction / View).
    anns: List[Dict[str, Any]] = [
        dict(
            text=_title_text(net, layer, model_id=model_id if dual else None),
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.12,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=18, color=_TEXT, family=_FONT),
        ),
        dict(
            text=_subtitle(geom, models, model_id, net, layer),
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.04,
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
                    y=0.98,
                    xanchor="center",
                    yanchor="bottom",
                    showarrow=False,
                    font=dict(size=13, color=_TEXT, family=_FONT),
                ),
                dict(
                    text=_model_label(model_id) if model_id else "Abstraction",
                    xref="paper",
                    yref="paper",
                    x=0.78,
                    y=0.98,
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
    pixel: Optional[DualGeometry3D] = None,
    *,
    pixels: Optional[Dict[str, DualGeometry3D]] = None,
    models: Optional[Dict[str, Dict[str, DualGeometry3D]]] = None,
    default_model: Optional[str] = None,
    default_mode: str = "both",
    default_net: str = "vdd1",
    default_layer: Optional[int] = None,
    show_vias: bool = True,
):
    """
    Interactive Plotly 3D figure.

    Controls (when abstractions are present):
    - Abstraction dropdown: Dual-Layer | Tri mid-square | Tri-stagger | Tri-square
    - View dropdown: VDD1..VDDN/VSS × All | each metal layer

    Layout: Original (left scene) | selected abstraction (right scene).

    ``models`` maps ``model_id -> {net_name -> DualGeometry3D}``.
    Legacy ``pixels`` / ``pixel`` are treated as a single ``dual`` model.
    ``default_mode`` is accepted for back-compat and ignored when models exist.
    """
    import plotly.graph_objects as go

    del default_mode  # replaced by abstraction selector

    model_map: Dict[str, Dict[str, DualGeometry3D]] = {}
    if models:
        for mid, by_net in models.items():
            model_map[str(mid)] = {
                normalize_net_arg(k): v for k, v in by_net.items()
            }
    legacy: Dict[str, DualGeometry3D] = {}
    if pixels:
        legacy.update({normalize_net_arg(k): v for k, v in pixels.items()})
    if pixel is not None:
        legacy.setdefault("vdd1", pixel)
    if legacy and "dual" not in model_map:
        model_map["dual"] = legacy

    available = [m for m in MODEL_ORDER if m in model_map and model_map[m]]
    for mid in model_map:
        if mid not in available and model_map[mid]:
            available.append(mid)
    dual = bool(available)
    if default_model is None:
        default_model = available[0] if available else None
    elif default_model not in available:
        default_model = available[0] if available else None

    default_net = normalize_net_arg(default_net)
    valid = {g.name for g in _iter_nets(geom)}
    if default_net not in valid:
        raise ValueError(
            f"default_net must be one of {sorted(valid)} (got {default_net!r})"
        )

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

    for mid, by_net in model_map.items():
        for net_name, pix in by_net.items():
            _add_pixel_traces(
                traces,
                metas,
                pix,
                net_name,
                model_id=mid,
                scene=pix_scene,
                legend=pix_legend,
            )

    init_vis = _visibility(
        metas, default_net, default_layer, show_vias, model_id=default_model
    )
    init_leg = _legend_flags(
        metas, default_net, default_layer, show_vias, model_id=default_model
    )
    for tr, v, leg in zip(traces, init_vis, init_leg):
        tr.visible = v
        tr.showlegend = leg

    xmin, ymin, xmax, ymax = geom.bbox
    zmin, zmax = geom.z_range
    for by_net in model_map.values():
        for pix in by_net.values():
            zmin = min(zmin, pix.z_sink, pix.z_pad)
            zmax = max(zmax, pix.z_sink, pix.z_pad)

    def _split_layout(model_id: Optional[str], net: str, layer: Optional[int]) -> Dict[str, Any]:
        layout: Dict[str, Any] = {
            "annotations": _chrome_annotations(
                geom, model_map, model_id, net, layer, dual=dual
            )
        }
        if not dual:
            return layout
        right_title = _model_label(model_id) if model_id else "Abstraction"
        layout["scene"] = _scene_layout(
            xmin, ymin, xmax, ymax, zmin, zmax, domain_x=[0.0, 0.45]
        )
        layout["scene2"] = _scene_layout(
            xmin, ymin, xmax, ymax, zmin, zmax, domain_x=[0.55, 1.0]
        )
        layout["legend"] = _legend_layout(
            x=0.455, xanchor="right", title="Original"
        )
        layout["legend2"] = _legend_layout(
            x=1.0, xanchor="left", title=right_title
        )
        layout["margin"] = dict(l=40, r=40, t=160, b=40)
        return layout

    choices = _view_choices(geom)
    active_model = 0
    if default_model in available:
        active_model = available.index(default_model)
    active_view = _active_index(choices, default_net, default_layer)

    # Native HTML <select> controls (injected by write_figure_html_3d) drive
    # the (model × view) table — Plotly updatemenus with method=skip do not
    # reliably fire plotly_buttonclicked across CDN builds.
    model_options = [
        {"value": i, "label": _model_label(mid)} for i, mid in enumerate(available)
    ]
    view_options = [
        {"value": i, "label": label} for i, (label, _n, _L) in enumerate(choices)
    ]
    sync_table: Dict[str, Any] = {}
    for mi, mid in enumerate(available or [None]):
        for vi, (_label, net, layer) in enumerate(choices):
            key = f"{mi}:{vi}"
            sync_table[key] = {
                "visible": _visibility(
                    metas, net, layer, show_vias, model_id=mid
                ),
                "showlegend": _legend_flags(
                    metas, net, layer, show_vias, model_id=mid
                ),
                "layout": _split_layout(mid, net, layer),
            }

    fig = go.Figure(data=traces)
    layout_kwargs: Dict[str, Any] = dict(
        title=None,
        paper_bgcolor=_BG_PAPER,
        font=dict(family=_FONT, color=_TEXT, size=13),
        margin=dict(l=40, r=40 if dual else 140, t=160 if dual else 120, b=40),
        hovermode="closest",
        annotations=_chrome_annotations(
            geom, model_map, default_model, default_net, default_layer, dual=dual
        ),
    )
    if dual:
        layout_kwargs.update(
            _split_layout(default_model, default_net, default_layer)
        )
        # Extra top margin so title clears the HTML toolbar.
        layout_kwargs["margin"] = dict(l=40, r=40, t=160, b=40)
    else:
        layout_kwargs["legend"] = _legend_layout(x=1.01, title="Legend")
        layout_kwargs["scene"] = _scene_layout(
            xmin, ymin, xmax, ymax, zmin, zmax
        )
    fig.update_layout(**layout_kwargs)
    fig._pdn_sync = {  # type: ignore[attr-defined]
        "available": available,
        "active_model": active_model,
        "active_view": active_view,
        "model_options": model_options,
        "view_options": view_options,
        "table": sync_table,
        "dual": dual,
    }
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
) -> Optional[_Path]:
    """Write a static PNG snapshot via matplotlib (no kaleido)."""
    from plotly_mpl_png import write_plotly_3d_png

    out = write_plotly_3d_png(
        fig, png_path, width=width, height=height, scale=scale
    )
    return _Path(out) if out is not None else None


def write_figure_html_3d(
    fig,
    html_path,
    *,
    config: Optional[Dict[str, Any]] = None,
    png: bool = True,
    png_path: Optional[_Path] = None,
) -> Optional[_Path]:
    """Write Plotly HTML and inject working Abstraction / View <select> controls.

    When ``png`` is True (default), also writes a matplotlib PNG snapshot beside
    the HTML (or to ``png_path`` if given). Returns the PNG path when written.
    """
    import json
    from pathlib import Path

    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(html_path),
        include_plotlyjs=True,
        full_html=True,
        config=config or figure_config_3d(),
    )
    png_out: Optional[Path] = None
    if png:
        target = Path(png_path) if png_path is not None else html_path.with_suffix(".png")
        png_out = write_figure_png_3d(fig, target)
    sync = getattr(fig, "_pdn_sync", None)
    if not sync or not sync.get("dual"):
        return png_out

    def _opts(options: List[Dict[str, Any]], active: int) -> str:
        parts = []
        for opt in options:
            sel = " selected" if int(opt["value"]) == int(active) else ""
            label = (
                str(opt["label"])
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            parts.append(
                f'<option value="{int(opt["value"])}"{sel}>{label}</option>'
            )
        return "\n".join(parts)

    model_opts = _opts(sync.get("model_options") or [], sync["active_model"])
    view_opts = _opts(sync.get("view_options") or [], sync["active_view"])
    payload = json.dumps(
        {
            "active_model": int(sync["active_model"]),
            "active_view": int(sync["active_view"]),
            "table": sync["table"],
        }
    )
    toolbar = f"""
<style>
  #pdn-toolbar {{
    font-family: {_FONT};
    background: {_BG_PAPER};
    color: {_TEXT};
    padding: 10px 16px 6px 16px;
    display: flex;
    flex-wrap: wrap;
    gap: 18px 28px;
    align-items: flex-end;
    border-bottom: 1px solid {_GRID};
  }}
  #pdn-toolbar .pdn-field {{
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}
  #pdn-toolbar label {{
    font-size: 11px;
    font-weight: 600;
    color: {_ACCENT};
    letter-spacing: 0.02em;
  }}
  #pdn-toolbar select {{
    font-size: 13px;
    color: {_TEXT};
    background: #ffffff;
    border: 1px solid #94a3b8;
    border-radius: 6px;
    padding: 7px 12px;
    min-width: 200px;
    cursor: pointer;
  }}
  #pdn-toolbar select:focus {{
    outline: 2px solid {_ACCENT};
    outline-offset: 1px;
  }}
</style>
<div id="pdn-toolbar">
  <div class="pdn-field">
    <label for="pdn-model">Abstraction</label>
    <select id="pdn-model">{model_opts}</select>
  </div>
  <div class="pdn-field">
    <label for="pdn-view">View</label>
    <select id="pdn-view">{view_opts}</select>
  </div>
</div>
<script>
(function() {{
  var payload = {payload};
  function gd() {{
    return document.querySelector('.plotly-graph-div');
  }}
  function apply() {{
    var g = gd();
    var modelSel = document.getElementById('pdn-model');
    var viewSel = document.getElementById('pdn-view');
    if (!g || !g.data || !modelSel || !viewSel) return;
    var model = parseInt(modelSel.value, 10);
    var view = parseInt(viewSel.value, 10);
    var st = payload.table[String(model) + ':' + String(view)];
    if (!st) return;
    Plotly.update(g, {{visible: st.visible, showlegend: st.showlegend}}, st.layout);
  }}
  function bind() {{
    var g = gd();
    var modelSel = document.getElementById('pdn-model');
    var viewSel = document.getElementById('pdn-view');
    if (!g || !window.Plotly || !modelSel || !viewSel) {{
      setTimeout(bind, 50);
      return;
    }}
    modelSel.addEventListener('change', apply);
    viewSel.addEventListener('change', apply);
  }}
  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', bind);
  }} else {{
    bind();
  }}
}})();
</script>
"""
    text = html_path.read_text(encoding="utf-8")
    # Insert toolbar at the start of <body> so it sits above the plot.
    lower = text.lower()
    body_idx = lower.find("<body")
    if body_idx >= 0:
        gt = text.find(">", body_idx)
        if gt >= 0:
            text = text[: gt + 1] + "\n" + toolbar + "\n" + text[gt + 1 :]
        else:
            text = toolbar + text
    elif "</body>" in text:
        text = text.replace("</body>", toolbar + "\n</body>", 1)
    else:
        text = toolbar + text
    html_path.write_text(text, encoding="utf-8")
    return png_out
