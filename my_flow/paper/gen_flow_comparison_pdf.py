#!/usr/bin/env python3
"""Generate PDF: flowcharts + comparison of pixel-r / tri-stagger / tri-square."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    Flowable,
)

import json
import os
import subprocess
import sys
import tempfile

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    plt = None  # type: ignore
    FancyBboxPatch = None  # type: ignore

_ANACONDA_PY = Path("/home/yenchulo/anaconda3/bin/python")

OUT_PDF = Path(__file__).resolve().parent / "PDN_Abstraction_Flow_Comparison.pdf"

# Unicode-capable fonts (Helvetica lacks ‖, λ, α, arrows, etc.)
_FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("DejaVu", str(_FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(_FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(_FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(
    TTFont("DejaVu-BoldOblique", str(_FONT_DIR / "DejaVuSans-BoldOblique.ttf"))
)
pdfmetrics.registerFontFamily(
    "DejaVu",
    normal="DejaVu",
    bold="DejaVu-Bold",
    italic="DejaVu-Oblique",
    boldItalic="DejaVu-BoldOblique",
)
FONT = "DejaVu"
FONT_B = "DejaVu-Bold"
FONT_I = "DejaVu-Oblique"

# Palette
C_TITLE = colors.HexColor("#1a2332")
C_ACCENT = colors.HexColor("#0d5c63")
C_PIXEL = colors.HexColor("#1e4d8c")
C_STAGGER = colors.HexColor("#8b4513")
C_TRISQ = colors.HexColor("#2d6a4f")
C_SHARED = colors.HexColor("#4a5568")
C_LIGHT = colors.HexColor("#f0f4f8")
C_BOX = colors.HexColor("#e8eef5")
C_ARROW = colors.HexColor("#334155")
C_IMPR = colors.HexColor("#166534")
C_WARN = colors.HexColor("#9a3412")
C_FORMULA_BG = colors.HexColor("#f8fafc")
C_FORMULA_BD = colors.HexColor("#cbd5e1")


class FlowChart(Flowable):
    """Vertical flowchart of labeled boxes with arrows."""

    def __init__(
        self,
        steps: list[str],
        width: float = 6.5 * inch,
        box_h: float = 0.38 * inch,
        gap: float = 0.18 * inch,
        fill: colors.Color = C_BOX,
        stroke: colors.Color = C_ACCENT,
        title: str | None = None,
        note: str | None = None,
    ):
        super().__init__()
        self.steps = steps
        self.box_w = width * 0.72
        self.width = width
        self.box_h = box_h
        self.gap = gap
        self.fill = fill
        self.stroke = stroke
        self.title = title
        self.note = note
        n = len(steps)
        self._title_h = 0.28 * inch if title else 0
        self._note_h = 0.22 * inch if note else 0
        self.height = (
            self._title_h
            + n * box_h
            + max(0, n - 1) * gap
            + self._note_h
            + 0.08 * inch
        )

    def draw(self):
        c = self.canv
        y = self.height - 0.04 * inch
        if self.title:
            c.setFillColor(self.stroke)
            c.setFont(FONT_B, 11)
            c.drawCentredString(self.width / 2, y - 14, self.title)
            y -= self._title_h
        x0 = (self.width - self.box_w) / 2
        for i, text in enumerate(self.steps):
            y -= self.box_h
            c.setStrokeColor(self.stroke)
            c.setFillColor(self.fill)
            c.setLineWidth(1.4)
            c.roundRect(x0, y, self.box_w, self.box_h, 6, fill=1, stroke=1)
            c.setFillColor(C_TITLE)
            c.setFont(FONT, 8.5)
            # wrap long labels
            max_chars = 72
            if len(text) > max_chars:
                # split at " — " or mid
                parts = text.split(" — ", 1)
                if len(parts) == 2 and len(parts[0]) < max_chars:
                    c.drawCentredString(
                        self.width / 2, y + self.box_h / 2 + 4, parts[0]
                    )
                    c.setFont(FONT, 7.5)
                    c.setFillColor(C_SHARED)
                    c.drawCentredString(
                        self.width / 2, y + self.box_h / 2 - 8, parts[1]
                    )
                else:
                    c.setFont(FONT, 7.5)
                    c.drawCentredString(
                        self.width / 2, y + self.box_h / 2 - 3, text[:90]
                    )
            else:
                c.drawCentredString(
                    self.width / 2, y + self.box_h / 2 - 3, text
                )
            if i < len(self.steps) - 1:
                # arrow
                ay0 = y
                ay1 = y - self.gap
                mid_x = self.width / 2
                c.setStrokeColor(C_ARROW)
                c.setFillColor(C_ARROW)
                c.setLineWidth(1.2)
                c.line(mid_x, ay0, mid_x, ay1 + 4)
                path = c.beginPath()
                path.moveTo(mid_x, ay1 + 1)
                path.lineTo(mid_x - 4, ay1 + 7)
                path.lineTo(mid_x + 4, ay1 + 7)
                path.close()
                c.drawPath(path, fill=1, stroke=0)
                y -= self.gap
        if self.note:
            c.setFillColor(C_SHARED)
            c.setFont(FONT_I, 7.5)
            c.drawCentredString(self.width / 2, 4, self.note)


class DualColumnFlow(Flowable):
    """Two side-by-side mini flowcharts (for shared prefix vs branch)."""

    def __init__(
        self,
        left_title: str,
        left_steps: list[str],
        right_title: str,
        right_steps: list[str],
        width: float = 7.0 * inch,
        left_color: colors.Color = C_PIXEL,
        right_color: colors.Color = C_STAGGER,
    ):
        super().__init__()
        self.left_title = left_title
        self.left_steps = left_steps
        self.right_title = right_title
        self.right_steps = right_steps
        self.width = width
        self.left_color = left_color
        self.right_color = right_color
        self.col_w = width * 0.46
        self.box_h = 0.32 * inch
        self.gap = 0.12 * inch
        n = max(len(left_steps), len(right_steps))
        self.height = 0.3 * inch + n * self.box_h + max(0, n - 1) * self.gap + 0.1 * inch

    def _draw_col(self, c, x_off, title, steps, stroke):
        y = self.height - 0.08 * inch
        c.setFillColor(stroke)
        c.setFont(FONT_B, 9)
        c.drawCentredString(x_off + self.col_w / 2, y - 12, title)
        y -= 0.28 * inch
        box_w = self.col_w * 0.92
        x0 = x_off + (self.col_w - box_w) / 2
        for i, text in enumerate(steps):
            y -= self.box_h
            c.setStrokeColor(stroke)
            c.setFillColor(C_LIGHT)
            c.setLineWidth(1.2)
            c.roundRect(x0, y, box_w, self.box_h, 4, fill=1, stroke=1)
            c.setFillColor(C_TITLE)
            c.setFont(FONT, 7)
            c.drawCentredString(x_off + self.col_w / 2, y + self.box_h / 2 - 2.5, text)
            if i < len(steps) - 1:
                mid = x_off + self.col_w / 2
                c.setStrokeColor(C_ARROW)
                c.setFillColor(C_ARROW)
                c.setLineWidth(1.0)
                c.line(mid, y, mid, y - self.gap + 3)
                path = c.beginPath()
                path.moveTo(mid, y - self.gap + 0.5)
                path.lineTo(mid - 3, y - self.gap + 5.5)
                path.lineTo(mid + 3, y - self.gap + 5.5)
                path.close()
                c.drawPath(path, fill=1, stroke=0)
                y -= self.gap

    def draw(self):
        c = self.canv
        gap = self.width - 2 * self.col_w
        self._draw_col(c, 0, self.left_title, self.left_steps, self.left_color)
        self._draw_col(
            c, self.col_w + gap, self.right_title, self.right_steps, self.right_color
        )


_FORMULA_DIR = Path(__file__).resolve().parent / "_formula_cache"


def _hex(c: colors.Color) -> str:
    return f"#{int(c.red * 255):02x}{int(c.green * 255):02x}{int(c.blue * 255):02x}"


_MPL_HELPER = r"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

cfg = json.load(open(sys.argv[1], encoding="utf-8"))
path = cfg["path"]
title = cfg["title"]
latex_lines = cfg["latex_lines"]
note = cfg.get("note")
accent_hex = cfg["accent"]
n = len(latex_lines) + (1 if note else 0)
fig_h = 0.72 + 0.48 * n
fig, ax = plt.subplots(figsize=(7.4, fig_h), dpi=180)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
fig.patch.set_facecolor("white")
ax.add_patch(FancyBboxPatch((0.01, 0.04), 0.98, 0.92,
    boxstyle="round,pad=0.02,rounding_size=0.04", linewidth=1.6,
    edgecolor=accent_hex, facecolor="#f8fafc", transform=ax.transAxes, clip_on=False))
ax.add_patch(FancyBboxPatch((0.01, 0.04), 0.012, 0.92,
    boxstyle="round,pad=0.0,rounding_size=0.0", linewidth=0,
    facecolor=accent_hex, transform=ax.transAxes, clip_on=False))
y = 0.82
ax.text(0.055, y, title, transform=ax.transAxes, fontsize=11, fontweight="bold",
        color=accent_hex, va="center", ha="left")
y -= 0.30
for latex in latex_lines:
    ax.text(0.055, y, f"${latex}$", transform=ax.transAxes, fontsize=13.5,
            color="#1a2332", va="center", ha="left")
    y -= 0.28
if note:
    ax.text(0.055, y, note, transform=ax.transAxes, fontsize=9,
            color="#4a5568", va="center", ha="left", style="italic")
fig.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor="white")
print(fig_h)
"""


def _render_formula_png(
    path: Path,
    title: str,
    latex_lines: list[str],
    *,
    note: str | None,
    accent_hex: str,
) -> float:
    """Return figure height in inches."""
    if _HAS_MPL:
        n = len(latex_lines) + (1 if note else 0)
        fig_h = 0.72 + 0.48 * n
        fig, ax = plt.subplots(figsize=(7.4, fig_h), dpi=180)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.patch.set_facecolor("white")
        ax.add_patch(
            FancyBboxPatch(
                (0.01, 0.04),
                0.98,
                0.92,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                linewidth=1.6,
                edgecolor=accent_hex,
                facecolor="#f8fafc",
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (0.01, 0.04),
                0.012,
                0.92,
                boxstyle="round,pad=0.0,rounding_size=0.0",
                linewidth=0,
                facecolor=accent_hex,
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        y = 0.82
        ax.text(
            0.055,
            y,
            title,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color=accent_hex,
            va="center",
            ha="left",
        )
        y -= 0.30
        for latex in latex_lines:
            ax.text(
                0.055,
                y,
                f"${latex}$",
                transform=ax.transAxes,
                fontsize=13.5,
                color="#1a2332",
                va="center",
                ha="left",
            )
            y -= 0.28
        if note:
            ax.text(
                0.055,
                y,
                note,
                transform=ax.transAxes,
                fontsize=9,
                color="#4a5568",
                va="center",
                ha="left",
                style="italic",
            )
        fig.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        plt.close(fig)
        return fig_h

    if _ANACONDA_PY.is_file():
        cfg = {
            "path": str(path),
            "title": title,
            "latex_lines": latex_lines,
            "note": note,
            "accent": accent_hex,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(cfg, f)
            cfg_path = f.name
        helper = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8")
        helper.write(_MPL_HELPER)
        helper.close()
        try:
            out = subprocess.check_output(
                [str(_ANACONDA_PY), helper.name, cfg_path],
                text=True,
            ).strip()
            return float(out.splitlines()[-1])
        finally:
            os.unlink(cfg_path)
            os.unlink(helper.name)

    raise RuntimeError("matplotlib not available (install it or use anaconda python)")


def render_formula_card(
    name: str,
    title: str,
    latex_lines: list[str],
    *,
    note: str | None = None,
    accent: colors.Color = C_ACCENT,
    width: float = 6.8 * inch,
) -> Flowable:
    """Render a neat math card (matplotlib mathtext) and return an Image flowable."""
    _FORMULA_DIR.mkdir(exist_ok=True)
    path = _FORMULA_DIR / f"{name}.png"
    try:
        fig_h = _render_formula_png(
            path, title, latex_lines, note=note, accent_hex=_hex(accent)
        )
    except Exception as exc:
        print(f"[warn] formula render failed for {name}: {exc}", file=sys.stderr)
        return _UnicodeFormulaCard(
            title, latex_lines + ([note] if note else []), width, accent
        )
    img = Image(str(path), width=width, height=width * (fig_h / 7.4) * 0.95)
    img.hAlign = "CENTER"
    return img


class _UnicodeFormulaCard(Flowable):
    """Fallback shaded card when matplotlib is unavailable."""

    def __init__(
        self,
        title: str,
        lines: list[str],
        width: float = 6.8 * inch,
        accent: colors.Color = C_ACCENT,
    ):
        super().__init__()
        self.title = title
        self.lines = lines
        self.width = width
        self.accent = accent
        self._pad = 8
        self._line_h = 14
        self.height = self._pad + 12 + 4 + len(lines) * self._line_h + self._pad

    def draw(self):
        c = self.canv
        c.setStrokeColor(self.accent)
        c.setFillColor(C_FORMULA_BG)
        c.setLineWidth(1.2)
        c.roundRect(0, 0, self.width, self.height, 5, fill=1, stroke=1)
        c.setFillColor(self.accent)
        c.rect(0, 0, 3.5, self.height, fill=1, stroke=0)
        y = self.height - self._pad - 10
        c.setFillColor(self.accent)
        c.setFont(FONT_B, 8)
        c.drawString(12, y, self.title)
        y -= 16
        c.setFillColor(C_TITLE)
        c.setFont(FONT, 8.5)
        for line in self.lines:
            c.drawString(12, y, line)
            y -= self._line_h


class TopologySketch(Flowable):
    """Simple schematic of each model's topology."""

    def __init__(self, kind: str, width: float = 6.8 * inch, height: float | None = None):
        super().__init__()
        self.kind = kind
        self.width = width
        if height is not None:
            self.height = height
        elif kind == "pixel":
            self.height = 2.35 * inch
        elif kind == "tri_stagger":
            self.height = 2.75 * inch
        else:
            self.height = 2.85 * inch

    def draw(self):
        c = self.canv
        c.setStrokeColor(C_SHARED)
        c.setFillColor(C_LIGHT)
        c.setLineWidth(0.8)
        c.roundRect(0, 0, self.width, self.height, 6, fill=1, stroke=1)
        if self.kind == "pixel":
            self._pixel(c)
        elif self.kind == "tri_stagger":
            self._stagger(c)
        else:
            self._trisquare(c)

    def _text(self, c, x, y, text, size=7, color=C_TITLE, bold=False, align="left"):
        c.setFillColor(color)
        c.setFont(FONT_B if bold else FONT, size)
        if align == "center":
            c.drawCentredString(x, y, text)
        elif align == "right":
            c.drawRightString(x, y, text)
        else:
            c.drawString(x, y, text)

    def _node(self, c, x, y, r, fill):
        c.setFillColor(fill)
        c.setStrokeColor(C_TITLE)
        c.setLineWidth(1.0)
        c.circle(x, y, r, fill=1, stroke=1)

    def _wire(self, c, x1, y1, x2, y2, color=C_PIXEL):
        c.setStrokeColor(color)
        c.setLineWidth(1.6)
        c.line(x1, y1, x2, y2)

    def _pixel(self, c):
        self._text(
            c,
            self.width / 2,
            self.height - 14,
            "Pixel-R star unit cell (identical per grid cell)",
            9,
            C_PIXEL,
            bold=True,
            align="center",
        )
        mid_r, cen_r = 5.0, 7.5

        # ── Left: side view (vertical stack) ──────────────────────────
        sx = 100.0
        top_y, cy, bot_y = 128.0, 72.0, 24.0
        self._text(c, sx, 148, "Side view (Z)", 7.5, C_SHARED, bold=True, align="center")
        self._wire(c, sx, top_y - 7, sx, cy + cen_r, C_PIXEL)
        self._wire(c, sx, cy - cen_r, sx, bot_y + 6, C_PIXEL)
        self._text(c, sx + 12, (top_y + cy) / 2 - 2, "Rz (Rup)", 7, C_PIXEL, bold=True)
        self._text(c, sx + 12, (cy + bot_y) / 2 - 2, "Rz (Rdown)", 7, C_PIXEL, bold=True)
        self._node(c, sx, top_y, 7, colors.HexColor("#93c5fd"))
        self._text(c, sx, top_y + 11, "Top / pad", 6.5, C_TITLE, align="center")
        self._node(c, sx, cy, cen_r, colors.HexColor("#fca5a5"))
        self._text(c, sx - 14, cy - 2, "Center / sink", 6.5, C_TITLE, align="right")
        self._node(c, sx, bot_y, 6, colors.HexColor("#cbd5e1"))
        self._text(c, sx, bot_y - 12, "Bottom (internal)", 6.5, C_TITLE, align="center")

        # divider
        c.setStrokeColor(colors.HexColor("#94a3b8"))
        c.setDash(2, 2)
        c.setLineWidth(0.8)
        c.line(185, 14, 185, 150)
        c.setDash()

        # ── Right: plan view (orthogonal cross) ───────────────────────
        px, py = 305.0, 72.0
        arm = 48.0
        self._text(c, px, 148, "Plan view (X–Y)", 7.5, C_SHARED, bold=True, align="center")
        # arms
        self._wire(c, px - cen_r, py, px - arm + mid_r, py, C_PIXEL)
        self._wire(c, px + cen_r, py, px + arm - mid_r, py, C_PIXEL)
        self._wire(c, px, py + cen_r, px, py + arm - mid_r, C_PIXEL)
        self._wire(c, px, py - cen_r, px, py - arm + mid_r, C_PIXEL)
        # Rx / Ry are half-arm values; neighbor path = two in series = 2·Rx / 2·Ry
        self._text(c, px - arm / 2, py + 9, "Rx", 7, C_PIXEL, bold=True, align="center")
        self._text(c, px + arm / 2, py + 9, "Rx", 7, C_PIXEL, bold=True, align="center")
        self._text(c, px + 12, py + arm / 2 + 2, "Ry", 7, C_PIXEL, bold=True)
        self._text(c, px + 12, py - arm / 2 - 6, "Ry", 7, C_PIXEL, bold=True)
        # nodes
        self._node(c, px, py, cen_r, colors.HexColor("#fca5a5"))
        self._node(c, px - arm, py, mid_r, colors.HexColor("#e2e8f0"))
        self._text(c, px - arm, py - 14, "W mid", 6, C_SHARED, align="center")
        self._node(c, px + arm, py, mid_r, colors.HexColor("#e2e8f0"))
        self._text(c, px + arm, py - 14, "E mid", 6, C_SHARED, align="center")
        self._node(c, px, py + arm, mid_r, colors.HexColor("#e2e8f0"))
        self._text(c, px, py + arm + 10, "N mid", 6, C_SHARED, align="center")
        self._node(c, px, py - arm, mid_r, colors.HexColor("#e2e8f0"))
        self._text(c, px, py - arm - 12, "S mid", 6, C_SHARED, align="center")

        # notes (far right, clear of cross)
        self._text(c, 385, 120, "pink = Center / sink", 6.5, C_SHARED)
        self._text(c, 385, 104, "Neighbor path = 2·Rx or 2·Ry", 6.5, C_PIXEL)
        self._text(c, 385, 88, "G_S = Schur(edge + bottom)", 6.5, C_PIXEL)
        self._text(c, 385, 72, "Fit globals: Rx, Ry, Rz", 6.5, C_PIXEL)

    def _stagger(self, c):
        h = self.height
        # Title band (keep all drawing below this)
        self._text(
            c,
            self.width / 2,
            h - 16,
            "Tri-stagger topology",
            9,
            C_STAGGER,
            bold=True,
            align="center",
        )
        self._text(
            c,
            self.width / 2,
            h - 30,
            "pad/sink vias → distinct mid landings + square mid grid + stubs",
            7,
            C_SHARED,
            align="center",
        )

        y_mid = 0.42 * h
        grid_top, grid_bot = y_mid + 28, y_mid - 28
        c.setStrokeColor(colors.HexColor("#d6d3d1"))
        c.setLineWidth(0.6)
        for i in range(5):
            x = 55 + i * 48
            c.line(x, grid_bot, x, grid_top)
        for j in range(3):
            y = grid_bot + j * ((grid_top - grid_bot) / 2)
            c.line(55, y, 55 + 4 * 48, y)
        self._text(
            c, 150, grid_bot - 14, "mid square grid (pitch_bot)", 6.5, C_SHARED, align="center"
        )

        # pad above mid (must stay below title band ≈ h-38)
        pad_y = min(h - 48, grid_top + 36)
        self._node(c, 115, pad_y, 6, colors.HexColor("#93c5fd"))
        self._text(c, 115, pad_y + 11, "pad XY", 6.5, C_TITLE, align="center")
        self._wire(c, 115, pad_y - 6, 115, y_mid + 12, C_STAGGER)
        self._text(
            c,
            108,
            (pad_y + y_mid) / 2 - 2,
            "Rz",
            6.5,
            C_STAGGER,
            bold=True,
            align="right",
        )
        self._node(c, 115, y_mid + 8, 5, colors.HexColor("#fdba74"))
        self._text(c, 115 - 16, y_mid + 5, "landing", 6, C_SHARED, align="right")
        c.setStrokeColor(C_STAGGER)
        c.setDash(2, 2)
        c.setLineWidth(1.3)
        c.line(115, y_mid + 3, 155, y_mid)
        c.setDash()
        self._text(c, 138, y_mid + 12, "stub", 6, C_STAGGER, align="center")

        # sink below mid
        sink_y = max(18.0, grid_bot - 32)
        self._node(c, 220, sink_y, 6, colors.HexColor("#fca5a5"))
        self._text(c, 220, sink_y - 12, "sink XY", 6.5, C_TITLE, align="center")
        self._wire(c, 220, sink_y + 6, 220, y_mid - 12, C_STAGGER)
        self._text(
            c,
            228,
            (sink_y + y_mid) / 2 - 2,
            "Rz",
            6.5,
            C_STAGGER,
            bold=True,
        )
        self._node(c, 220, y_mid - 8, 5, colors.HexColor("#fdba74"))
        self._text(c, 238, y_mid - 11, "landing", 6, C_SHARED)
        c.setStrokeColor(C_STAGGER)
        c.setDash(2, 2)
        c.line(220, y_mid - 3, 190, y_mid)
        c.setDash()

        # Notes column — right side, below title band, clear of grid
        nx = 310
        ny0 = h - 55
        for i, line in enumerate(
            [
                "R_ew = Rx · |L|/pitch",
                "R_ns = Ry · |L|/pitch",
                "pad ≠ sink landing",
                "(no pure via short)",
                "Fit: Rx, Ry, Rz",
            ]
        ):
            self._text(c, nx, ny0 - i * 14, line, 7.5, C_STAGGER)

    def _trisquare(self, c):
        h = self.height
        self._text(
            c,
            self.width / 2,
            h - 16,
            "Tri-square stub stack (6 R: per-sheet Rx,Ry + 2 vias)",
            9,
            C_TRISQ,
            bold=True,
            align="center",
        )
        self._text(
            c,
            self.width / 2,
            h - 30,
            "L1 pads → L2 (coarse) → L3 (fine) → sinks",
            7,
            C_SHARED,
            align="center",
        )

        # Layer centers from top content area downward (all inside box)
        top = h - 50
        gap = 42.0
        layers = [
            (top, "L1 pads (via Rz_pad + stubs)", colors.HexColor("#93c5fd")),
            (top - gap, "L2 mid @ pitch_bot (= Pixel-R)", colors.HexColor("#86efac")),
            (top - 2 * gap, "L3 bot @ pitch_bot/k (denser)", colors.HexColor("#bbf7d0")),
            (top - 3 * gap, "sinks (via Rz_pad + stubs)", colors.HexColor("#fca5a5")),
        ]
        box_h = 20.0
        box_w = 240.0
        x0 = 36.0
        for y, lab, col in layers:
            c.setFillColor(col)
            c.setStrokeColor(C_TITLE)
            c.setLineWidth(0.9)
            c.roundRect(x0, y - box_h / 2, box_w, box_h, 3, fill=1, stroke=1)
            c.setFillColor(C_TITLE)
            c.setFont(FONT, 7.5)
            c.drawString(x0 + 8, y - 3, lab)

        # Via connectors between layers
        c.setStrokeColor(C_TRISQ)
        c.setLineWidth(1.4)
        for x in (x0 + 40, x0 + box_w / 2, x0 + box_w - 40):
            for i in range(3):
                y_hi = layers[i][0] - box_h / 2
                y_lo = layers[i + 1][0] + box_h / 2
                c.line(x, y_hi, x, y_lo)

        # Notes: one per via gap + two summary lines stacked below with fixed spacing
        nx = 300
        self._text(
            c, nx, top - gap / 2 - 2, "1st via @ pad XY → stub → L2", 7, C_TRISQ, bold=True
        )
        self._text(
            c,
            nx,
            top - 1.5 * gap - 2,
            "2nd via @ L2 XY → L3 landing + stubs",
            7,
            C_TRISQ,
            bold=True,
        )
        self._text(
            c, nx, top - 2.5 * gap - 2, "default k=2 (L3 denser than L2)", 7, C_TRISQ
        )
        # Keep these two well below the sinks layer, 16pt apart
        y_sum = max(28.0, layers[-1][0] - box_h / 2 - 18)
        self._text(c, nx, y_sum, "Rx_u,Ry_u on L2; Rx_l,Ry_l on L3", 7, C_TRISQ)
        self._text(
            c,
            nx,
            y_sum - 16,
            "Fit: 6R (Rz_pad, Rz_ul) IR LS",
            7,
            C_TRISQ,
            bold=True,
        )


def _styles():
    ss = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "T",
            parent=ss["Title"],
            fontSize=18,
            textColor=C_TITLE,
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName=FONT_B,
        ),
        "subtitle": ParagraphStyle(
            "ST",
            parent=ss["Normal"],
            fontSize=10,
            textColor=C_SHARED,
            alignment=TA_CENTER,
            spaceAfter=14,
            fontName=FONT,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=ss["Heading1"],
            fontSize=13,
            textColor=C_ACCENT,
            spaceBefore=10,
            spaceAfter=8,
            fontName=FONT_B,
            borderPadding=3,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=ss["Heading2"],
            fontSize=11,
            textColor=C_TITLE,
            spaceBefore=8,
            spaceAfter=5,
            fontName=FONT_B,
        ),
        "body": ParagraphStyle(
            "B",
            parent=ss["Normal"],
            fontSize=9,
            leading=12,
            textColor=C_TITLE,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            fontName=FONT,
        ),
        "bullet": ParagraphStyle(
            "Bu",
            parent=ss["Normal"],
            fontSize=8.5,
            leading=11,
            textColor=C_TITLE,
            leftIndent=12,
            spaceAfter=3,
            fontName=FONT,
        ),
        "impr": ParagraphStyle(
            "Im",
            parent=ss["Normal"],
            fontSize=8.5,
            leading=11,
            textColor=C_IMPR,
            leftIndent=12,
            spaceAfter=3,
            fontName=FONT,
        ),
        "cell": ParagraphStyle(
            "Cell",
            parent=ss["Normal"],
            fontSize=7.5,
            leading=9.5,
            textColor=C_TITLE,
            fontName=FONT,
        ),
        "cell_h": ParagraphStyle(
            "CellH",
            parent=ss["Normal"],
            fontSize=8,
            leading=10,
            textColor=colors.white,
            fontName=FONT_B,
            alignment=TA_CENTER,
        ),
        "footer": ParagraphStyle(
            "F",
            parent=ss["Normal"],
            fontSize=7.5,
            textColor=C_SHARED,
            alignment=TA_CENTER,
            fontName=FONT,
        ),
        "caption": ParagraphStyle(
            "Cap",
            parent=ss["Normal"],
            fontSize=7.5,
            textColor=C_SHARED,
            alignment=TA_CENTER,
            spaceBefore=2,
            spaceAfter=8,
            fontName=FONT_I,
        ),
    }
    return styles


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def _cmp_table(styles) -> Table:
    cell = styles["cell"]
    ch = styles["cell_h"]
    headers = [
        _p("Aspect", ch),
        _p("Pixel-R<br/>(spec_flow)", ch),
        _p("Tri-stagger<br/>(my_flow)", ch),
        _p("Tri-square<br/>(my_flow)", ch),
    ]
    rows_raw = [
        [
            "Role / lineage",
            "TSMC Spec baseline star abstraction",
            "my_flow: real ports + stubs + global Rx,Ry,Rz (Pixel-R-style fit)",
            "Same ports/IR objective; L2/L3 + 6 R params",
        ],
        [
            "Topology tag",
            "star_half_arm",
            "tri_stagger_mid_grid",
            "tri_square_stub_6r",
        ],
        [
            "Metal / sheets",
            "Single abstract star grid (not physical metals)",
            "One mid square sheet + pad/sink landings + stubs",
            "L2 coarse + L3 fine (per-sheet Rx,Ry)",
        ],
        [
            "Pitch (auto)",
            "Pad-array target, clamp &gt; max metal",
            "Same rule as Pixel-R (shared for fair model compare)",
            "Same as stagger / Pixel-R",
        ],
        [
            "Port model",
            "Imaginary full chip-pitch grid; n_pads = n_sinks = nx×ny",
            "Real pad XY + real nonzero-current sinks (may n_p ≠ n_s)",
            "Same as stagger (reuse tri_stagger_ports_from_dual)",
        ],
        [
            "Lateral R",
            "Global half-arms Rx, Ry (identical cells)",
            "R = Rx·|L|/pitch (EW), Ry·|L|/pitch (NS) on mid + stubs",
            "L2: Rx_u/Ry_u; L3: Rx_l/Ry_l (each ·|L|/pitch_sheet)",
        ],
        [
            "Vertical / via",
            "Global Rz (Rup = Rdown = Rz); no stubs",
            "Pad/sink: Rz then stubs; no sheet–sheet via",
            "Rz_pad (pad/sink); Rz_ul (L2→L3) then L3 stubs",
        ],
        [
            "Port / via attach",
            "Pad & sink at same cell center (aligned)",
            "port—Rz—landing—stubs→mid grid",
            "port—Rz_pad—landing—stubs; L2—Rz_ul—L3 landing—stubs",
        ],
        [
            "R extract",
            "None — fit R directly from Kron G′",
            "None — fit R directly (Zhang extract retired)",
            "None — fit R directly (Zhang extract retired)",
        ],
        [
            "Fit variables",
            "Rx, Ry, Rz (3 globals)",
            "Rx, Ry, Rz (3 globals)",
            "Rx_u,Ry_u,Rx_l,Ry_l,Rz_pad,Rz_ul (6)",
        ],
        [
            "Fitting objective",
            "Default: eigen LS ||λ_M − diag(Qᵀ G_S Q)||²; alt: IR voltage match",
            "Same as Pixel-R IR: Σ||drop_S − drop_M||² (fit_rxryrz)",
            "Same Pixel-R IR objective (6-param TRF)",
        ],
        [
            "Fitting method",
            "scipy least_squares TRF; log10(R); max_nfev ≈ 300",
            "Same TRF on log10(Rx,Ry,Rz) as Pixel-R IR",
            "TRF on log10 of 6 R params (fit_log_r_params_ir)",
        ],
        [
            "BC / residual",
            "Eigen: spectrum of G′; IR: pads V-fixed, sinks I-driven",
            "Mixed BC; absolute drop residual (all stimuli)",
            "Same mixed-BC IR residual as stagger / Pixel-R IR",
        ],
        [
            "SPICE validation",
            "Always emit + solve original / reduced / pixel_r",
            "Optional --spice; MNA IR plots always",
            "MNA IR plots always",
        ],
        [
            "Entry point",
            "tclsh spec_flow/run_flow.tcl [case] [eigen|ir]",
            "run_tri_stagger.py &lt;comp&gt; [seed] [--spice]",
            "run_tri_square.py &lt;comp&gt; [seed] [--k 2]",
        ],
        [
            "Key artifacts",
            "pixel_model.json, pixel_r.json, spice/pixel_r.sp",
            "tri_stagger_model.json, tri_stagger_r.json (Rx,Ry,Rz)",
            "tri_square_model.json, tri_square_r.json (6R)",
        ],
    ]
    data = [headers]
    for row in rows_raw:
        data.append([_p(c, cell) for c in row])

    col_w = [1.15 * inch, 2.0 * inch, 2.05 * inch, 2.05 * inch]
    t = Table(data, colWidths=col_w, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), C_ACCENT),
        ("BACKGROUND", (1, 0), (1, 0), C_PIXEL),
        ("BACKGROUND", (2, 0), (2, 0), C_STAGGER),
        ("BACKGROUND", (3, 0), (3, 0), C_TRISQ),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#e2e8f0")),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(
                ("BACKGROUND", (1, i), (-1, i), colors.HexColor("#f8fafc"))
            )
    t.setStyle(TableStyle(style_cmds))
    return t


def build_pdf(path: Path) -> None:
    styles = _styles()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title="PDN Abstraction Flow Comparison",
        author="TSMC_PDN_Abstraction",
    )
    story: list = []

    # ── Cover / overview ──────────────────────────────────────────────
    story.append(_p("PDN Abstraction Flow Comparison", styles["title"]))
    story.append(
        _p(
            "Pixel-R (spec_flow) · Tri-stagger · Tri-square (my_flow)<br/>"
            "Focus: model topology differences (pitch + IR objective matched)",
            styles["subtitle"],
        )
    )
    story.append(_p("1. Overview", styles["h1"]))
    story.append(
        _p(
            "Both packages start from IBM power-grid SPICE, discover VDD connected "
            "components, assemble a sparse mesh conductance <b>G<sub>M</sub></b>, and "
            "Kron-reduce internals to a dense port matrix <b>G′</b>. They then stamp a "
            "compact R-model and fit parameters so the model matches Kron IR (or, for "
            "Pixel-R by default, the port spectrum). The three models below are the "
            "primary comparison arms for reduced-vs-model IR plots.",
            styles["body"],
        )
    )
    story.append(
        _p(
            "<b>Comparison setup (for now):</b> my_flow uses the <b>same auto pitch rule</b> "
            "as Pixel-R (C4 lattice, clamped &gt; max metal) and the <b>same IR objective</b> "
            "(absolute multi-stimulus drop match). Differences to study are "
            "<b>model topology and port attach</b>: Pixel-R star on imaginary grid → "
            "<b>tri-stagger</b> (real ports + stubs + mid square; 3R) → "
            "<b>tri-square</b> (L2/L3 hierarchy + stubs; 6R).",
            styles["body"],
        )
    )
    story.append(_p("my_flow attach rules (ports &amp; vias)", styles["h2"]))
    story.append(
        _p(
            "Sink XY = current-weighted average of real SPICE injections in each "
            "<font face='Courier' size='8'>pitch_bot</font> cell. Pad XY = mean of "
            "real V sources in the cell. Then:",
            styles["body"],
        )
    )
    for line in [
        "<b>Voltage pads &amp; current sinks (my_flow):</b> "
        "<font face='Courier' size='8'>port — Rz(/Rz_pad) — landing @ real XY — stubs "
        "(∝ sheet Rx/Ry) — nearest mesh node</font>.",
        "<b>Tri-stagger:</b> only those pad/sink vias (no separate sheet-to-sheet via stack).",
        "<b>Tri-square:</b> pads → L2 (stubs ∝ Rx_u/Ry_u); sinks → L3 (stubs ∝ Rx_l/Ry_l); "
        "mid→bot: <font face='Courier' size='8'>L2 grid — Rz_ul — L3 landing @ L2 XY — "
        "stubs on L3 (∝ Rx_l/Ry_l, pitch_l)</font>.",
        "<b>Pixel-R:</b> imaginary aligned pad/sink at cell centers; no stub attach "
        "(half-arm star cells).",
    ]:
        story.append(_p(f"• {line}", styles["bullet"]))

    # Shared pipeline
    story.append(_p("2. Shared front-end (all models)", styles["h1"]))
    story.append(
        FlowChart(
            [
                "Parse IBM SPICE (R / I / V) → net.npz",
                "Discover VDD components; build pad + sink ports (compN/)",
                "Assemble sparse G_M; Kron-reduce → G′ (Gprime.npy)",
                "Build abstract R-model topology on those ports",
                "Fit parameters (eigen or IR) → stamp G_S",
                "Validate: MNA and/or SPICE .op → IR correlation plots",
            ],
            fill=C_LIGHT,
            stroke=C_SHARED,
            title="Common pipeline skeleton",
            note="IR definition: IR_k = mean(V_pads) − V_sink,k   |   Mixed BC: pads V-fixed, sinks I-driven",
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    # ── Pixel-R ───────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(_p("3. Pixel-R flow (spec_flow)", styles["h1"]))
    story.append(
        _p(
            "Baseline TSMC Spec abstraction: a <b>single-layer grid of identical stars</b>. "
            "No geometry extract — three globals <b>(R<sub>x</sub>, R<sub>y</sub>, R<sub>z</sub>)</b> "
            "are fit so the Schur port admittance G<sub>S</sub> matches Kron G′.",
            styles["body"],
        )
    )
    story.append(
        FlowChart(
            [
                "stage01_parse — IBM flat SPICE → OUT/",
                "stage02_ports — VDD CCs; imaginary chip-pitch pads/sinks",
                "stage03_assemble_kron — G_M → Kron G′",
                "stage04_pixel_model — star topology (half-arm Rx/Ry, Rup/Rdown)",
                "stage05_fit_eigen (default)  OR  stage05_fit_ir",
                "stage06_emit_spice — original / reduced_gprime / pixel_r.sp",
                "stage07_spice_solve — ngspice or Spectre .op",
                "stage08_correlate_ir — sink IR metrics + scatter/spatial PNGs",
            ],
            fill=colors.HexColor("#dbeafe"),
            stroke=C_PIXEL,
            title="Pixel-R generation flow",
            note="Driver: tclsh spec_flow/run_flow.tcl [ibmpgN] [eigen|ir]   |   chip_pitch > max metal pitch",
            box_h=0.32 * inch,
            gap=0.12 * inch,
        )
    )
    story.append(PageBreak())
    story.append(_p("3b. Pixel-R model & fitting", styles["h1"]))
    story.append(TopologySketch("pixel"))
    story.append(
        _p(
            "Figure: Pixel-R star. Left = vertical pad–sink–bottom stack; right = in-plane half-arms. "
            "Neighbor sinks couple through two series half-arms (effective g = 1/(2R<sub>x</sub>) or 1/(2R<sub>y</sub>)).",
            styles["caption"],
        )
    )
    story.append(_p("Fitting details", styles["h2"]))
    story.append(
        KeepTogether(
            [
                render_formula_card(
                    "pixel_eigen",
                    "Eigen objective (default, TSMC spec)",
                    [
                        r"\min_{R_x,R_y,R_z}\ \left\| \lambda_M - \mathrm{diag}(Q^{T} G_S Q) \right\|^2",
                        r"G' = Q\,\mathrm{diag}(\lambda_M)\,Q^{T}",
                    ],
                    note="Grounded eigh of Kron G′  (match projected spectrum of G_S to λ_M)",
                    accent=C_PIXEL,
                ),
                Spacer(1, 0.05 * inch),
                render_formula_card(
                    "pixel_ir",
                    "IR objective (alternative)",
                    [
                        r"\min_{R_x,R_y,R_z}\ \sum_j \left\| V_S(I_j) - V_M(I_j) \right\|^2",
                    ],
                    note="On sink IR drops; multi-stimulus mixed BC (pads V-fixed, sinks I-driven)",
                    accent=C_PIXEL,
                ),
                Spacer(1, 0.05 * inch),
                _p(
                    "• <b>Optimizer:</b> scipy.optimize.least_squares, method=TRF, variables in log10(R), "
                    "bounds ~ [10<sup>−4</sup>, 10<sup>4</sup>] × r_scale.",
                    styles["bullet"],
                ),
                _p(
                    "• <b>Limitation addressed by my_flow models:</b> not metal-faithful; aligned "
                    "pad↔sink; three globals only — hard to capture heterogeneous / multi-layer anisotropy.",
                    styles["bullet"],
                ),
            ]
        )
    )

    # ── Stagger ───────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(_p("4. Tri-stagger flow (my_flow)", styles["h1"]))
    story.append(
        _p(
            "Reuses my_flow <b>OUT/compN</b> (parse → ports → Kron already done). Builds a "
            "<b>square mid grid</b> at pitch<sub>bot</sub>. Attach rule: "
            "<font face='Courier' size='8'>pad/sink — Rz — landing @ real XY — stubs "
            "(∝ Rx/Ry) — nearest mid node</font>. Sink XY is the current-weighted injection "
            "average in the cell. No sheet-to-sheet via beyond those pad/sink vias "
            "(so pad→sink cannot be a pure vertical short).",
            styles["body"],
        )
    )
    story.append(
        FlowChart(
            [
                "Prerequisite: my_flow stages 01–03 (ports + G′ in compN/)",
                "Filter to nonzero-current sinks → tri_stagger_ports.json; rebuild G′",
                "Build mid square @ pitch_bot + pad/sink landings + stubs (tri_stagger)",
                "Fit global Rx, Ry, Rz by multi-stimulus IR LS (fit_rxryrz / fit_ir_tri_stagger)",
                "Stamp R = Rx·|L|/pitch, Ry·|L|/pitch; vias = Rz",
                "MNA IR plots (scatter / spatial / error heatmap)",
                "[optional --spice] emit original / reduced / tri_stagger.sp",
            ],
            fill=colors.HexColor("#ffedd5"),
            stroke=C_STAGGER,
            title="Tri-stagger generation flow",
            note=None,
            box_h=0.34 * inch,
            gap=0.13 * inch,
        )
    )
    story.append(
        _p(
            "Driver: <font face='Courier' size='8'>python3 my_flow/src/run_tri_stagger.py "
            "&lt;compN&gt; [seed] [--spice]</font>  ·  pad-only comps skipped",
            styles["caption"],
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    story.append(TopologySketch("tri_stagger"))
    story.append(
        _p(
            "Figure: staggered vias. Distinct mid landings + length-proportional stubs "
            "force current through the mid sheet.",
            styles["caption"],
        )
    )
    story.append(_p("Fitting details", styles["h2"]))
    story.append(
        KeepTogether(
            [
                render_formula_card(
                    "stagger_obj",
                    "IR fit objective (= spec_flow Pixel-R IR)",
                    [
                        r"\min_{R_x,R_y,R_z}\ \sum_j \left\| \mathrm{drop}_S(I_j)-\mathrm{drop}_M(I_j) \right\|^2",
                    ],
                    note="Absolute voltage drops; all stimuli (incl. random); mixed BC",
                    accent=C_STAGGER,
                ),
                Spacer(1, 0.05 * inch),
                render_formula_card(
                    "stagger_opt",
                    "Parameters & optimizer",
                    [
                        r"R_{\mathrm{ew}}=R_x\cdot\frac{|L|}{\mathrm{pitch}},\;"
                        r"R_{\mathrm{ns}}=R_y\cdot\frac{|L|}{\mathrm{pitch}},\;"
                        r"R_{\mathrm{via}}=R_z",
                    ],
                    note="TRF on log10(Rx,Ry,Rz); bounds from median(diag G′); same as fit_ir.py",
                    accent=C_STAGGER,
                ),
            ]
        )
    )
    story.append(Spacer(1, 0.06 * inch))
    story.append(_p("What changes vs Pixel-R (model focus)", styles["h2"]))
    for line in [
        "Pitch + IR objective matched to Pixel-R IR — isolate topology.",
        "Ports: port—Rz—landing—stubs→mesh (vs imaginary aligned ports).",
        "Full square mid sheet (E–W and N–S) vs Pixel-R abstract star cells.",
        "Distinct landings so pad→sink cannot be a pure via short.",
        "Same globals Rx, Ry, Rz; stub R ∝ Rx/Ry, via = Rz.",
    ]:
        story.append(_p(f"• {line}", styles["impr"]))

    # ── Tri-square ────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(_p("5. Tri-square flow (my_flow)", styles["h1"]))
    story.append(
        _p(
            "Same my_flow <b>OUT/compN</b> prerequisite and port filtering as tri-stagger, but stamps "
            "<b>two</b> square sheets with <b>six</b> R parameters: mid L2 at the "
            "<b>Pixel-R pitch</b> (pitch<sub>bot</sub>) and denser bot L3 "
            "(pitch<sub>l</sub> = pitch<sub>bot</sub>/k). "
            "<b>Pads:</b> <font face='Courier' size='8'>V — Rz_pad — L2 landing — stubs (∝ Rx_u/Ry_u) → L2</font>. "
            "<b>Sinks:</b> <font face='Courier' size='8'>I — Rz_pad — L3 landing — stubs (∝ Rx_l/Ry_l) → L3</font>. "
            "<b>Mid→bot vias:</b> <font face='Courier' size='8'>L2 grid — Rz_ul — L3 landing @ L2 XY "
            "— stubs (∝ Rx_l/Ry_l, pitch_l) → L3 grid</font>.",
            styles["body"],
        )
    )
    story.append(
        FlowChart(
            [
                "Prerequisite: my_flow OUT/compN (same as tri-stagger)",
                "Filter nonzero sinks; rebuild G′ (shared helper)",
                "Build L2 mid @ pitch_bot (= Pixel-R) and denser L3 @ pitch_bot/k (stubs)",
                "Vias: pad/sink→Rz_pad; L2→L3 landing@XY + L3 stubs (Rz_ul)",
                "Fit 6R: Rx_u,Ry_u,Rx_l,Ry_l,Rz_pad,Rz_ul (fit_ir_tri_square)",
                "MNA IR plots (…_tri_square.png)",
            ],
            fill=colors.HexColor("#dcfce7"),
            stroke=C_TRISQ,
            title="Tri-square generation flow",
            note=None,
            box_h=0.34 * inch,
            gap=0.13 * inch,
        )
    )
    story.append(
        _p(
            "Driver: <font face='Courier' size='8'>python3 my_flow/src/run_tri_square.py "
            "&lt;compN&gt; [seed] [--k 2]</font>  ·  default k=2 (L3 denser; pitch_l = pitch_bot/k)",
            styles["caption"],
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    story.append(TopologySketch("trisquare"))
    story.append(
        _p(
            "Figure: dual-pitch sheets. L2 @ pitch_bot (= Pixel-R), L3 denser. "
            "Six R params: per-sheet Rx/Ry plus Rz_pad (ports) and Rz_ul (L2→L3).",
            styles["caption"],
        )
    )
    story.append(_p("Fitting details", styles["h2"]))
    story.append(
        KeepTogether(
            [
                render_formula_card(
                    "trisq_obj",
                    "IR fit objective (= spec_flow Pixel-R IR)",
                    [
                        r"\min_{R}\ \sum_j \left\| \mathrm{drop}_S(I_j)-\mathrm{drop}_M(I_j) \right\|^2",
                    ],
                    note="Same absolute-drop residual as stagger / Pixel-R IR",
                    accent=C_TRISQ,
                ),
                Spacer(1, 0.05 * inch),
                render_formula_card(
                    "trisq_opt",
                    "Parameters & optimizer (6)",
                    [
                        r"R_x^{u},R_y^{u},\;R_x^{l},R_y^{l},\;R_z^{\mathrm{pad}},R_z^{\mathrm{ul}}",
                    ],
                    note="TRF on log10(R); L2 vs L3 differ by pitch and independent Rx/Ry",
                    accent=C_TRISQ,
                ),
            ]
        )
    )
    story.append(Spacer(1, 0.06 * inch))
    story.append(_p("Improvements vs tri-stagger", styles["h2"]))
    for line in [
        "Two sheets: L2 mid @ Pixel-R pitch + denser L3 (pitch_bot/k).",
        "Six R params: Rx_u,Ry_u,Rx_l,Ry_l,Rz_pad,Rz_ul (not shared 3-globals).",
        "Pads/sinks: port—Rz_pad—landing—stubs→sheet.",
        "Mid→bot vias: L2—Rz_ul—L3 landing@XY—stubs→L3 grid (not pure via).",
        "Bilinear attach retired.",
    ]:
        story.append(_p(f"• {line}", styles["impr"]))

    # ── Side-by-side comparison ───────────────────────────────────────
    story.append(PageBreak())
    story.append(_p("6. Side-by-side comparison", styles["h1"]))
    story.append(
        _p(
            "Pitch policy and IR fitting are matched to Pixel-R IR. The remaining "
            "differences are <b>model structure</b>: star vs mid square vs L2/L3, and "
            "imaginary aligned ports vs real XY + stubs.",
            styles["body"],
        )
    )
    story.append(_cmp_table(styles))
    story.append(Spacer(1, 0.12 * inch))

    # ── Improvement summary ───────────────────────────────────────────
    story.append(_p("7. What each step improves", styles["h1"]))
    story.append(
        DualColumnFlow(
            "Pixel-R → Tri-stagger",
            [
                "Star half-arm cells",
                "Imaginary aligned ports",
                "Same pitch + Rx,Ry,Rz IR",
                "No stub attach",
            ],
            "Tri-stagger → Tri-square",
            [
                "One mid sheet + stubs",
                "Same pitch + IR fit",
                "Nearest-node stubs",
                "Single via stack to mid",
            ],
            left_color=C_PIXEL,
            right_color=C_STAGGER,
        )
    )
    story.append(Spacer(1, 0.06 * inch))
    story.append(
        _p(
            "<b>Pixel-R → Tri-stagger:</b> change topology/ports (star → mid square + real stubs); "
            "keep pitch + IR objective. "
            "<b>Tri-stagger → Tri-square:</b> add L2/L3 hierarchy and expand to 6 R params.",
            styles["body"],
        )
    )

    # Compact improvement table
    impr_headers = [
        _p("From → To", styles["cell_h"]),
        _p("Primary improvement", styles["cell_h"]),
        _p("Fitting / model change", styles["cell_h"]),
    ]
    impr_rows = [
        [
            "Pixel-R → Tri-stagger",
            "Mid square + real ports + stubs (model change)",
            "Pitch + IR objective held fixed (= Pixel-R IR)",
        ],
        [
            "Tri-stagger → Tri-square",
            "L2/L3 hierarchy; per-sheet Rx/Ry + 2 vias",
            "6R fit (Rx_u,Ry_u,Rx_l,Ry_l,Rz_pad,Rz_ul); same IR objective",
        ],
        [
            "Shared validation",
            "Correlate sink IR vs Kron G′ (and vs original SPICE when solved)",
            "e_IR = ||ΔV_sink|| / ||V_ref||; plots under each compN/",
        ],
    ]
    data2 = [impr_headers] + [[_p(x, styles["cell"]) for x in r] for r in impr_rows]
    t2 = Table(data2, colWidths=[1.4 * inch, 2.9 * inch, 2.9 * inch])
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), C_ACCENT),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#dbeafe")),
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#dcfce7")),
                ("BACKGROUND", (0, 3), (-1, 3), C_LIGHT),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    story.append(t2)

    story.append(Spacer(1, 0.18 * inch))
    story.append(_p("8. How to reproduce the comparison plots", styles["h1"]))
    story.append(
        _p(
            "Fair compare for <b>model</b> diffs: same auto pitch (&gt; max metal) and same IR "
            "objective (<font face='Courier' size='8'>fit_rxryrz</font> = Pixel-R IR).<br/><br/>"
            "1. my_flow (stages 01–03 + both models): "
            "<font face='Courier' size='8'>tclsh my_flow/run_flow.tcl ibmpg2</font><br/>"
            "2. Or per-comp after stages 01–03: "
            "<font face='Courier' size='8'>python3 my_flow/src/run_tri_stagger.py "
            "my_flow/outputs/ibmpg2_auto_k1/comp1 0</font> · "
            "<font face='Courier' size='8'>python3 my_flow/src/run_tri_square.py "
            "…/comp1 0 --k 2</font><br/>"
            "3. Pixel-R IR arm: <font face='Courier' size='8'>tclsh spec_flow/run_flow.tcl ibmpg2 ir</font><br/>"
            "Compare <font face='Courier' size='8'>ir_scatter_reduced_vs_*.png</font> under each "
            "<font face='Courier' size='8'>compN/</font> (ibmpg2–6).",
            styles["body"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))
    story.append(
        _p(
            "Sources: spec_flow/src/{pixel_r,fit_eigen,fit_ir,stage0*}.py · "
            "my_flow/src/{stub_attach,fit_rxryrz,tri_stagger,tri_square,run_*}.py · READMEs",
            styles["footer"],
        )
    )

    doc.build(story)
    print(f"Wrote {path}")


if __name__ == "__main__":
    build_pdf(OUT_PDF)
