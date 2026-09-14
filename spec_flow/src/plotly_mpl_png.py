"""Matplotlib PNG snapshot of a Plotly 3D figure (no kaleido)."""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float, float]
Segment = Tuple[Point, Point]

_MAX_SEGS = 120_000
_MARKER_MAP = {
    "diamond": "D",
    "diamond-open": "D",
    "square": "s",
    "square-open": "s",
    "cross": "x",
    "x": "x",
    "circle-open": "o",
}


def _plain_title(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    return s.strip()


def _finite(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _xyz(tr: Any) -> Tuple[Sequence[Any], Sequence[Any], Sequence[Any]]:
    xs = getattr(tr, "x", None) or ()
    ys = getattr(tr, "y", None) or ()
    zs = getattr(tr, "z", None) or ()
    return xs, ys, zs


def _polylines_to_segments(
    xs: Iterable[Any], ys: Iterable[Any], zs: Iterable[Any]
) -> List[Segment]:
    segs: List[Segment] = []
    pts: List[Point] = []
    for x, y, z in zip(xs, ys, zs):
        fx, fy, fz = _finite(x), _finite(y), _finite(z)
        if fx is None or fy is None or fz is None:
            if len(pts) >= 2:
                segs.extend(zip(pts[:-1], pts[1:]))
            pts = []
            continue
        pts.append((fx, fy, fz))
    if len(pts) >= 2:
        segs.extend(zip(pts[:-1], pts[1:]))
    if len(segs) > _MAX_SEGS:
        step = max(1, len(segs) // _MAX_SEGS)
        segs = segs[::step]
    return segs


def _marker_points(
    xs: Iterable[Any], ys: Iterable[Any], zs: Iterable[Any]
) -> Tuple[List[float], List[float], List[float]]:
    ox: List[float] = []
    oy: List[float] = []
    oz: List[float] = []
    for x, y, z in zip(xs, ys, zs):
        fx, fy, fz = _finite(x), _finite(y), _finite(z)
        if fx is None or fy is None or fz is None:
            continue
        ox.append(fx)
        oy.append(fy)
        oz.append(fz)
    return ox, oy, oz


def _visible(tr: Any) -> bool:
    vis = getattr(tr, "visible", True)
    return vis not in (False, "legendonly")


def _scene_name(tr: Any) -> str:
    scene = getattr(tr, "scene", None) or "scene"
    if not isinstance(scene, str):
        scene = getattr(scene, "plotly_name", None) or "scene"
    return scene if scene else "scene"


def _mode(tr: Any) -> str:
    return str(getattr(tr, "mode", None) or "lines")


def _line_style(tr: Any) -> Tuple[str, float, float]:
    line = getattr(tr, "line", None)
    color = "#334155"
    width = 1.5
    if line is not None:
        color = getattr(line, "color", None) or color
        try:
            width = float(getattr(line, "width", None) or width)
        except (TypeError, ValueError):
            pass
    if isinstance(color, (list, tuple)):
        color = color[0] if color else "#334155"
    opacity = getattr(tr, "opacity", None)
    try:
        alpha = float(opacity) if opacity is not None else 1.0
    except (TypeError, ValueError):
        alpha = 1.0
    return str(color), max(0.25, width / 2.5), min(max(alpha, 0.05), 1.0)


def _marker_style(tr: Any) -> Tuple[str, str, float, float]:
    marker = getattr(tr, "marker", None)
    color = "#0f172a"
    size = 6.0
    symbol = "o"
    if marker is not None:
        color = getattr(marker, "color", None) or color
        try:
            size = float(getattr(marker, "size", None) or size)
        except (TypeError, ValueError):
            pass
        raw = str(getattr(marker, "symbol", None) or "circle")
        symbol = _MARKER_MAP.get(raw, "o")
    if isinstance(color, (list, tuple)):
        color = color[0] if color else "#0f172a"
    opacity = getattr(tr, "opacity", None)
    try:
        alpha = float(opacity) if opacity is not None else 1.0
    except (TypeError, ValueError):
        alpha = 1.0
    return str(color), symbol, max(8.0, size * 2.2), min(max(alpha, 0.05), 1.0)


def _axis_limits(scene: Any) -> Optional[Tuple[List[float], List[float], List[float]]]:
    if scene is None:
        return None

    def _range(axis: Any) -> Optional[List[float]]:
        if axis is None:
            return None
        rng = getattr(axis, "range", None)
        if rng is None or len(rng) < 2:
            return None
        a, b = _finite(rng[0]), _finite(rng[1])
        if a is None or b is None:
            return None
        return [a, b]

    xr = _range(getattr(scene, "xaxis", None))
    yr = _range(getattr(scene, "yaxis", None))
    zr = _range(getattr(scene, "zaxis", None))
    if xr is None or yr is None or zr is None:
        return None
    return xr, yr, zr


def _apply_limits(ax: Any, scene: Any, traces: List[Any]) -> None:
    lims = _axis_limits(scene)
    if lims is not None:
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        ax.set_zlim(*lims[2])
        dx = abs(lims[0][1] - lims[0][0]) or 1.0
        dy = abs(lims[1][1] - lims[1][0]) or 1.0
        dz = abs(lims[2][1] - lims[2][0]) or 1.0
        try:
            ax.set_box_aspect((dx, dy, dz))
        except Exception:  # noqa: BLE001
            pass
        return
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for tr in traces:
        tx, ty, tz = _xyz(tr)
        px, py, pz = _marker_points(tx, ty, tz)
        xs.extend(px)
        ys.extend(py)
        zs.extend(pz)
    if not xs:
        return
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))
    ax.set_zlim(min(zs), max(zs))


def _draw_traces(ax: Any, traces: List[Any]) -> None:
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    legend_used = set()
    for tr in traces:
        mode = _mode(tr)
        name = str(getattr(tr, "name", None) or "")
        show = bool(getattr(tr, "showlegend", False)) and name and name not in legend_used
        xs, ys, zs = _xyz(tr)
        if "lines" in mode:
            segs = _polylines_to_segments(xs, ys, zs)
            if segs:
                color, lw, alpha = _line_style(tr)
                lc = Line3DCollection(
                    segs, colors=color, linewidths=lw, alpha=alpha
                )
                ax.add_collection3d(lc)
                if show:
                    ax.plot([], [], [], color=color, lw=lw, alpha=alpha, label=name)
                    legend_used.add(name)
                    show = False
        if "markers" in mode:
            mx, my, mz = _marker_points(xs, ys, zs)
            if mx:
                color, symbol, size, alpha = _marker_style(tr)
                ax.scatter(
                    mx,
                    my,
                    mz,
                    c=color,
                    marker=symbol,
                    s=size,
                    alpha=alpha,
                    depthshade=False,
                    label=name if show else None,
                )
                if show:
                    legend_used.add(name)


def write_plotly_3d_png(
    fig: Any,
    png_path,
    *,
    width: int = 1600,
    height: int = 900,
    scale: float = 2.0,
) -> Optional[Path]:
    """Render visible Scatter3d traces with matplotlib Agg (no kaleido)."""
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(
            f"warning: could not write PNG snapshot {png_path} (matplotlib): {exc}",
            file=sys.stderr,
        )
        return None

    by_scene: dict[str, List[Any]] = {"scene": [], "scene2": []}
    for tr in getattr(fig, "data", ()) or ():
        if not _visible(tr):
            continue
        kind = str(getattr(tr, "type", None) or "").lower()
        if kind and kind != "scatter3d":
            continue
        scene = _scene_name(tr)
        by_scene.setdefault(scene, []).append(tr)

    dual = bool(by_scene.get("scene2"))
    dpi = 100.0
    fig_w = max(4.0, float(width) * float(scale) / dpi)
    fig_h = max(3.0, float(height) * float(scale) / dpi)
    mpl_fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#f8fafc")
    layout = getattr(fig, "layout", None)
    title = _plain_title(getattr(getattr(layout, "title", None), "text", None))
    if dual:
        ax_l = mpl_fig.add_subplot(121, projection="3d", facecolor="#ffffff")
        ax_r = mpl_fig.add_subplot(122, projection="3d", facecolor="#ffffff")
        _draw_traces(ax_l, by_scene.get("scene") or [])
        _draw_traces(ax_r, by_scene.get("scene2") or [])
        _apply_limits(ax_l, getattr(layout, "scene", None), by_scene.get("scene") or [])
        _apply_limits(
            ax_r, getattr(layout, "scene2", None), by_scene.get("scene2") or []
        )
        ax_l.set_xlabel("x")
        ax_l.set_ylabel("y")
        ax_l.set_zlabel("z")
        ax_r.set_xlabel("x")
        ax_r.set_ylabel("y")
        ax_r.set_zlabel("z")
        if ax_l.get_legend_handles_labels()[0]:
            ax_l.legend(loc="upper left", fontsize=8)
        if ax_r.get_legend_handles_labels()[0]:
            ax_r.legend(loc="upper left", fontsize=8)
    else:
        ax = mpl_fig.add_subplot(111, projection="3d", facecolor="#ffffff")
        traces = by_scene.get("scene") or []
        _draw_traces(ax, traces)
        _apply_limits(ax, getattr(layout, "scene", None), traces)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", fontsize=8)
    if title:
        mpl_fig.suptitle(title, fontsize=12, color="#0f172a")
    mpl_fig.tight_layout()
    try:
        mpl_fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="#f8fafc")
    except Exception as exc:  # noqa: BLE001
        plt.close(mpl_fig)
        print(
            f"warning: could not write PNG snapshot {png_path} (matplotlib): {exc}",
            file=sys.stderr,
        )
        return None
    plt.close(mpl_fig)
    return png_path
