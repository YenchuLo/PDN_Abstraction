"""IR scatter + sink-error heatmaps for tri-stagger / tri-square runs.

Primary plots (MNA, matches fit metric):
  - reduced (G') vs model (Gs) — IR % of Vdd with ±20% bands + spatial maps
  - |ΔIR| heatmap model vs reduced

Also:
  - original vs model when ngspice .volt files exist (IR % + spatial)
  - original vs reduced when ngspice .volt files exist (volt scatter)

Dual-layer / mid-square plots are intentionally not produced here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from _bootstrap import ensure_paths

ensure_paths()

from correlate import _factor_sink_block, build_stimuli, e_ir, solve_mixed_bc  # noqa: E402
from ports_dual import DualPortSet  # noqa: E402

MODEL_TAGS = {
    "tri_stagger": {
        "label": "Tri-stagger",
        "spectra": "tri_stagger_spectra.npz",
        "scatter_reduced": "ir_scatter_reduced_vs_tri_stagger.png",
        "scatter_original": "ir_scatter_original_vs_tri_stagger.png",
        "spatial_reduced": "ir_spatial_reduced_vs_tri_stagger.png",
        "spatial_original": "ir_spatial_original_vs_tri_stagger.png",
        "heatmap": "ir_error_heatmap_tri_stagger.png",
        # legacy alias kept for older viewers
        "heatmap_legacy": "ir_error_heatmap.png",
    },
    "tri_square": {
        "label": "Tri-square",
        "spectra": "tri_square_spectra.npz",
        "scatter_reduced": "ir_scatter_reduced_vs_tri_square.png",
        "scatter_original": "ir_scatter_original_vs_tri_square.png",
        "spatial_reduced": "ir_spatial_reduced_vs_tri_square.png",
        "spatial_original": "ir_spatial_original_vs_tri_square.png",
        "heatmap": "ir_error_heatmap_tri_square.png",
    },
}


def _vdd_ref(ports: DualPortSet) -> float:
    vdd = float(getattr(ports, "vdd", 0.0) or 0.0)
    if vdd > 0.0:
        return vdd
    pads = np.asarray(ports.pad_voltages, dtype=float)
    mean_pad = float(np.mean(pads)) if pads.size else 0.0
    if mean_pad <= 0.0:
        raise ValueError("cannot resolve Vdd from ports.vdd or pad_voltages")
    return mean_pad


def _ir_pct(drop: np.ndarray, vdd: float) -> np.ndarray:
    return np.asarray(drop, dtype=float) / float(vdd) * 100.0


def scatter_ir(
    ref: np.ndarray,
    pred: np.ndarray,
    title: str,
    path: Path,
) -> str:
    """Simple volt scatter (y=x only). Used for original-vs-reduced."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ref, pred, s=12, alpha=0.7)
    lo = float(min(ref.min(), pred.min()))
    hi = float(max(ref.max(), pred.max()))
    if lo == hi:
        lo, hi = lo - 1e-6, hi + 1e-6
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Reference IR drop")
    ax.set_ylabel("Predicted IR drop")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def scatter_ir_pct(
    ref_pct: np.ndarray,
    pred_pct: np.ndarray,
    title: str,
    path: Path,
) -> str:
    """Correlation scatter in IR % with ideal and ±20% relative error lines.

    x = feasibility / model (pred), y = real / reference (ref).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = np.asarray(ref_pct, dtype=float)
    pred = np.asarray(pred_pct, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pred, ref, alpha=0.7, label="IR Drop (%)")

    lo = float(min(ref.min(), pred.min()))
    hi = float(max(ref.max(), pred.max()))
    if lo == hi:
        lo, hi = lo - 1e-6, hi + 1e-6
    x_vals = [lo, hi]
    ax.plot(x_vals, x_vals, "k--", label="Ideal (y = x)")
    ax.plot(x_vals, [1.2 * x for x in x_vals], "r--", label="+20% Error")
    ax.plot(x_vals, [0.8 * x for x in x_vals], "b--", label="-20% Error")

    ax.set_xlabel("Feasibility_IR_Drop (%)")
    ax.set_ylabel("Real PDN IR Drop (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _spatial_ir_side_by_side(
    ports: DualPortSet,
    drop_ref_pct: np.ndarray,
    drop_pred_pct: np.ndarray,
    titles: Sequence[str],
    path: Path,
) -> str:
    """Side-by-side spatial IR Drop (%) maps at physical cell (x, y)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = ports.as_port_set().cells
    xs = np.asarray([c.x for c in cells], dtype=float)
    ys = np.asarray([c.y for c in cells], dtype=float)
    ref = np.asarray(drop_ref_pct, dtype=float)
    pred = np.asarray(drop_pred_pct, dtype=float)
    if len(xs) != len(ref) or len(xs) != len(pred):
        raise ValueError(
            f"spatial plot size mismatch: cells={len(xs)} ref={len(ref)} pred={len(pred)}"
        )

    min_val = float(min(ref.min(), pred.min()))
    max_val = float(max(ref.max(), pred.max()))
    if min_val == max_val:
        min_val, max_val = min_val - 1e-6, max_val + 1e-6

    point_size = 60
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1.scatter(
        xs,
        ys,
        c=ref,
        cmap="jet",
        vmin=min_val,
        vmax=max_val,
        s=point_size,
        alpha=0.8,
        edgecolors="none",
    )
    ax1.set_title(titles[0], fontsize=14, fontweight="bold")
    ax1.set_xlabel("X Coordinate", fontsize=12)
    ax1.set_ylabel("Y Coordinate", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)

    sc2 = ax2.scatter(
        xs,
        ys,
        c=pred,
        cmap="jet",
        vmin=min_val,
        vmax=max_val,
        s=point_size,
        alpha=0.8,
        edgecolors="none",
    )
    ax2.set_title(titles[1], fontsize=14, fontweight="bold")
    ax2.set_xlabel("X Coordinate", fontsize=12)
    ax2.set_ylabel("Y Coordinate", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax1.set_aspect("equal", adjustable="box")
    ax2.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(sc2, ax=[ax1, ax2], orientation="vertical", fraction=0.03, pad=0.05)
    cbar.set_label("IR Drop (%)", rotation=270, labelpad=15, fontsize=12)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _heatmap(
    err_map: np.ndarray,
    title: str,
    cbar_label: str,
    path: Path,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(err_map, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title)
    ax.set_xlabel("ix")
    ax.set_ylabel("iy")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def sink_ir_from_Gs(
    Gprime: np.ndarray,
    Gs: np.ndarray,
    ports: DualPortSet,
    *,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Original-stimulus sink IR drops from reduced G' and model Gs."""
    stimuli = dict(build_stimuli(ports.as_port_set(), seed=seed))
    if "original" not in stimuli:
        raise ValueError("need original sink stimulus")
    I = stimuli["original"]
    n_p = ports.n_pads
    V_pad = np.asarray(ports.pad_voltages, dtype=float)
    vref = float(np.mean(V_pad))
    lu_m, Gsp_m = _factor_sink_block(Gprime, n_p)
    lu_s, Gsp_s = _factor_sink_block(Gs, n_p)
    Vm = solve_mixed_bc(Gprime, n_p, V_pad, I, _lu=lu_m, _Gsp=Gsp_m)
    Vs = solve_mixed_bc(Gs, n_p, V_pad, I, _lu=lu_s, _Gsp=Gsp_s)
    drop_r = vref - Vm[n_p:]
    drop_s = vref - Vs[n_p:]
    return drop_r, drop_s, float(e_ir(drop_s, drop_r))


def plot_model_vs_reduced(
    out_dir: str | Path,
    ports: DualPortSet,
    Gprime: np.ndarray,
    Gs: np.ndarray,
    *,
    model_key: str,
    seed: int = 0,
) -> Dict[str, object]:
    """Write reduced-vs-model % scatter, spatial maps, and error heatmap (MNA)."""
    if model_key not in MODEL_TAGS:
        raise ValueError(f"unknown model_key {model_key!r}; expected one of {list(MODEL_TAGS)}")
    tag = MODEL_TAGS[model_key]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    drop_r, drop_s, e_red = sink_ir_from_Gs(Gprime, Gs, ports, seed=seed)
    vdd = _vdd_ref(ports)
    ref_pct = _ir_pct(drop_r, vdd)
    pred_pct = _ir_pct(drop_s, vdd)

    written: List[str] = []
    written.append(
        scatter_ir_pct(
            ref_pct,
            pred_pct,
            f"Reduced vs {tag['label']} (e={e_red:.4g}) +-20% Error Lines",
            out / tag["scatter_reduced"],
        )
    )
    written.append(
        _spatial_ir_side_by_side(
            ports,
            ref_pct,
            pred_pct,
            (
                "Reduced VDD IR Drop Distribution (%)",
                f"{tag['label']} VDD IR Drop Distribution (%)",
            ),
            out / tag["spatial_reduced"],
        )
    )

    ps = ports.as_port_set()
    err_map = np.full((ps.ny, ps.nx), np.nan)
    for k, c in enumerate(ps.cells):
        err_map[c.iy, c.ix] = abs(float(drop_s[k]) - float(drop_r[k]))
    written.append(
        _heatmap(
            err_map,
            f"Sink IR error ({tag['label']} vs reduced)",
            f"|ΔIR| {tag['label']} vs reduced",
            out / tag["heatmap"],
        )
    )
    if "heatmap_legacy" in tag:
        written.append(
            _heatmap(
                err_map,
                f"Sink IR error ({tag['label']} vs reduced)",
                f"|ΔIR| {tag['label']} vs reduced",
                out / tag["heatmap_legacy"],
            )
        )

    return {
        "model": model_key,
        "e_ir_reduced_vs_model": e_red,
        "plots": written,
        "source": "mna_gprime_gs",
    }


def _try_spice_original_vs_reduced(
    out_dir: Path,
    ports: DualPortSet,
) -> Optional[Dict[str, object]]:
    """Plot original vs reduced from SPICE volts if present and aligned."""
    from correlate_spice import pair_metrics, sink_ir_drop  # local import
    from spice_emit import PORT_MAP_JSON, load_simulator, spice_dir
    from spice_run import parse_volt_file

    sd = spice_dir(out_dir)
    port_map_path = sd / PORT_MAP_JSON
    ov = sd / "original.volt"
    rv = sd / "reduced_gprime.volt"
    if not (port_map_path.is_file() and ov.is_file() and rv.is_file()):
        return None
    port_map = json.loads(port_map_path.read_text())
    order = list(port_map["order"])
    n_p, n_s = ports.n_pads, ports.n_sinks
    if len(order) != n_p + n_s:
        return None
    try:
        vo = parse_volt_file(ov, order)
        vr = parse_volt_file(rv, order)
    except Exception:
        return None
    drop_o = sink_ir_drop(vo, n_p, n_s)
    drop_r = sink_ir_drop(vr, n_p, n_s)
    m = pair_metrics("original_vs_reduced", drop_r, drop_o)
    path = scatter_ir(
        drop_o,
        drop_r,
        f"Original vs Reduced (e={m['e_ir']:.4g})",
        out_dir / "ir_scatter_original_vs_reduced.png",
    )
    return {
        "e_ir_original_vs_reduced": m["e_ir"],
        "pearson_original_vs_reduced": m.get("pearson_r"),
        "plots": [path],
        "source": load_simulator(out_dir),
    }


def _try_spice_original_vs_model(
    out_dir: Path,
    ports: DualPortSet,
    *,
    model_key: str,
    model_volt_stem: str,
) -> Optional[Dict[str, object]]:
    from correlate_spice import pair_metrics, sink_ir_drop
    from spice_emit import PORT_MAP_JSON, load_simulator, spice_dir
    from spice_run import parse_volt_file

    tag = MODEL_TAGS[model_key]
    sd = spice_dir(out_dir)
    port_map_path = sd / PORT_MAP_JSON
    ov = sd / "original.volt"
    mv = sd / f"{model_volt_stem}.volt"
    if not (port_map_path.is_file() and ov.is_file() and mv.is_file()):
        return None
    port_map = json.loads(port_map_path.read_text())
    order = list(port_map["order"])
    n_p, n_s = ports.n_pads, ports.n_sinks
    if len(order) != n_p + n_s:
        return None
    try:
        vo = parse_volt_file(ov, order)
        vm = parse_volt_file(mv, order)
    except Exception:
        return None
    drop_o = sink_ir_drop(vo, n_p, n_s)
    drop_m = sink_ir_drop(vm, n_p, n_s)
    m = pair_metrics(f"original_vs_{model_key}", drop_m, drop_o)

    vdd = _vdd_ref(ports)
    ref_pct = _ir_pct(drop_o, vdd)
    pred_pct = _ir_pct(drop_m, vdd)
    written: List[str] = [
        scatter_ir_pct(
            ref_pct,
            pred_pct,
            f"Original vs {tag['label']} (e={m['e_ir']:.4g}) +-20% Error Lines",
            out_dir / tag["scatter_original"],
        ),
        _spatial_ir_side_by_side(
            ports,
            ref_pct,
            pred_pct,
            (
                "Signoff VDD IR Drop Distribution (%)",
                f"{tag['label']} VDD IR Drop Distribution (%)",
            ),
            out_dir / tag["spatial_original"],
        ),
    ]
    return {
        "e_ir_original_vs_model": m["e_ir"],
        "plots": written,
        "source": load_simulator(out_dir),
    }


def plot_ir_for_model_run(
    out_dir: str | Path,
    ports: DualPortSet,
    Gprime: np.ndarray,
    Gs: np.ndarray,
    *,
    model_key: str,
    seed: int = 0,
    spice_model_stem: Optional[str] = None,
) -> Dict[str, object]:
    """Full plot suite for one stagger / tri-square run."""
    out = Path(out_dir)
    report: Dict[str, object] = {"model": model_key, "plots": []}
    mna = plot_model_vs_reduced(
        out, ports, Gprime, Gs, model_key=model_key, seed=seed
    )
    report["mna"] = mna
    report["plots"] = list(mna["plots"])  # type: ignore[arg-type]

    ovr = _try_spice_original_vs_reduced(out, ports)
    if ovr is not None:
        report["original_vs_reduced"] = ovr
        report["plots"] = list(report["plots"]) + list(ovr["plots"])  # type: ignore[operator]
    else:
        # At ports, Kron G' is exact for the real PDN — emit MNA identity check
        # only if spice is missing, using G' vs G' is useless; skip with note.
        report["original_vs_reduced"] = {
            "skipped": True,
            "reason": "ngspice original.volt / reduced_gprime.volt not available",
        }

    stem = spice_model_stem
    if stem is None and model_key == "tri_stagger":
        stem = "tri_stagger"
    if stem is not None:
        ovm = _try_spice_original_vs_model(
            out, ports, model_key=model_key, model_volt_stem=stem
        )
        if ovm is not None:
            report["original_vs_model"] = ovm
            report["plots"] = list(report["plots"]) + list(ovm["plots"])  # type: ignore[operator]

    return report


def plot_from_saved_spectra(
    out_dir: str | Path,
    ports: DualPortSet,
    *,
    model_key: str,
    seed: int = 0,
) -> Dict[str, object]:
    """Load Gs/G' from model spectra.npz and plot."""
    out = Path(out_dir)
    spectra = out / MODEL_TAGS[model_key]["spectra"]
    if not spectra.is_file():
        raise FileNotFoundError(spectra)
    data = np.load(spectra)
    return plot_ir_for_model_run(
        out,
        ports,
        np.asarray(data["Gprime"], dtype=float),
        np.asarray(data["Gs"], dtype=float),
        model_key=model_key,
        seed=seed,
    )
