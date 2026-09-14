#!/usr/bin/env python3
"""Batch tri-stagger on ibmpg3–6: fit, ngspice IR plots, 3D HTML."""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

from _bootstrap import ensure_paths

ensure_paths()

from correlate_spice import pair_metrics, sink_ir_drop  # noqa: E402
from io_artifacts import load_ports  # noqa: E402
from ports_dual import can_tri_stagger_ports, tri_stagger_ports_from_dual  # noqa: E402
from spice_emit import PORT_MAP_JSON, load_simulator, spice_dir  # noqa: E402
from spice_run import parse_volt_file, run_deck  # noqa: E402

SRC = Path(__file__).resolve().parent
HERE = SRC.parent
ROOT = HERE.parent
SEED = 0
CASES = [3, 4, 5, 6]


def _py() -> str:
    import os

    return os.environ.get("MY_FLOW_PYTHON") or sys.executable


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def spice_and_plots(comp_out: Path) -> dict:
    from spice_run import parse_spectre_log, write_compat_volt

    sd = spice_dir(comp_out)
    sim = load_simulator(comp_out)
    port_map = json.loads((sd / PORT_MAP_JSON).read_text())
    order = list(port_map["order"])
    decks = {
        "original": ("original.sp", "original.volt"),
        "reduced_gprime": ("reduced_gprime.sp", "reduced_gprime.volt"),
        "tri_stagger": ("tri_stagger.sp", "tri_stagger.volt"),
    }
    voltages = {}
    for key, (deck_name, volt_name) in decks.items():
        print(f"  {sim} {deck_name} ...", flush=True)
        log_path = run_deck(sd / deck_name, simulator=sim, timeout_s=21600.0)
        volt_path = sd / volt_name
        if sim == "spectre":
            voltages[key] = parse_spectre_log(log_path, order)
            write_compat_volt(volt_path, order, voltages[key])
        else:
            voltages[key] = parse_volt_file(volt_path, order)
        (sd / f"{key}.volt.json").write_text(json.dumps(voltages[key], indent=2))

    dual_ports, _ = load_ports(comp_out)
    ports = tri_stagger_ports_from_dual(dual_ports)
    ps = ports.as_port_set()
    n_p, n_s = ps.n_pads, ps.n_sinks
    drops = {
        key: sink_ir_drop(voltages[key], n_p, n_s)
        for key in ("original", "reduced_gprime", "tri_stagger")
    }
    pairs = [
        pair_metrics("original_vs_tri_stagger", drops["tri_stagger"], drops["original"]),
        pair_metrics("original_vs_reduced", drops["reduced_gprime"], drops["original"]),
        pair_metrics(
            "reduced_vs_tri_stagger", drops["tri_stagger"], drops["reduced_gprime"]
        ),
    ]
    report = {
        "e_worst": max(p["e_ir"] for p in pairs),
        "e_ir_original_vs_tri_stagger": pairs[0]["e_ir"],
        "e_ir_original_vs_reduced": pairs[1]["e_ir"],
        "e_ir_reduced_vs_tri_stagger": pairs[2]["e_ir"],
        "pearson_original_vs_tri_stagger": pairs[0]["pearson_r"],
        "ir_pairs": pairs,
        "drops": {k: v.tolist() for k, v in drops.items()},
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drop_o = np.asarray(report["drops"]["original"], dtype=float)
    drop_r = np.asarray(report["drops"]["reduced_gprime"], dtype=float)
    drop_s = np.asarray(report["drops"]["tri_stagger"], dtype=float)
    written: list[str] = []

    def scatter(ref, pred, title, fname):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(ref, pred, s=12, alpha=0.7)
        lo = float(min(ref.min(), pred.min()))
        hi = float(max(ref.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("Reference IR drop")
        ax.set_ylabel("Predicted IR drop")
        ax.set_title(title)
        p = comp_out / fname
        fig.tight_layout()
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(str(p))

    scatter(
        drop_o,
        drop_s,
        f"Original vs Tri-stagger (e={report['e_ir_original_vs_tri_stagger']:.4g})",
        "ir_scatter_original_vs_tri_stagger.png",
    )
    scatter(
        drop_o,
        drop_r,
        f"Original vs Reduced (e={report['e_ir_original_vs_reduced']:.4g})",
        "ir_scatter_original_vs_reduced.png",
    )
    scatter(
        drop_r,
        drop_s,
        f"Reduced vs Tri-stagger (e={report['e_ir_reduced_vs_tri_stagger']:.4g})",
        "ir_scatter_reduced_vs_tri_stagger.png",
    )

    err_map = np.full((ps.ny, ps.nx), np.nan)
    for k, c in enumerate(ps.cells):
        err_map[c.iy, c.ix] = abs(drop_s[k] - drop_r[k])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(err_map, origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax, label="|ΔIR| Tri-stagger vs reduced")
    ax.set_title("Sink IR-drop error map")
    ax.set_xlabel("ix")
    ax.set_ylabel("iy")
    for fname in ("ir_error_heatmap_tri_stagger.png", "ir_error_heatmap.png"):
        p = comp_out / fname
        fig.tight_layout()
        fig.savefig(p, dpi=140)
        written.append(str(p))
    plt.close(fig)

    metrics = {
        "topology": "tri_stagger_mid_grid",
        "ir_source": load_simulator(comp_out),
        "e_worst": report["e_worst"],
        "e_ir_original_vs_tri_stagger": report["e_ir_original_vs_tri_stagger"],
        "e_ir_original_vs_reduced": report["e_ir_original_vs_reduced"],
        "e_ir_reduced_vs_tri_stagger": report["e_ir_reduced_vs_tri_stagger"],
        "pearson_original_vs_tri_stagger": report["pearson_original_vs_tri_stagger"],
        "ir_pairs": report["ir_pairs"],
        "plots": written,
        "n_pads": n_p,
        "n_sinks": n_s,
        "nx": ps.nx,
        "ny": ps.ny,
        "pitch_bot": ports.pitch_bot,
    }
    sr = comp_out / "tri_stagger_r.json"
    if sr.is_file():
        fit = json.loads(sr.read_text())
        for k in (
            "alpha_x",
            "alpha_y",
            "alpha_via",
            "relative_ir_error",
            "e_ir_by_stimulus",
            "rx0",
            "ry0",
        ):
            if k in fit:
                metrics[k] = fit[k]
    (comp_out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def run_case(tc: int) -> list[dict]:
    import shutil

    py = _py()
    spice = ROOT / f"Benchmarks/IBM/TC{tc}/ibmpg{tc}.spice"
    out = HERE / f"outputs/ibmpg{tc}_auto_k1"
    print(f"\n======== ibmpg{tc} ========", flush=True)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    _run([py, str(SRC / "stage01_parse.py"), str(spice), str(out)])
    _run(
        [
            py,
            str(SRC / "stage02_ports.py"),
            str(out),
            "0",
            "1",
            "--grid-to-pad-ratio",
            "1.0",
        ]
    )
    comps = json.loads((out / "components.json").read_text())["components"]
    rows: list[dict] = []
    for c in comps:
        comp = c["path"]
        cout = out / comp
        # Skip pad-only / zero-current components (stagger needs real sinks).
        dual_ports, _ = load_ports(cout)
        if not can_tri_stagger_ports(dual_ports):
            print(
                f"---- {comp} skip: pad-only / zero-current "
                f"(n_pads={dual_ports.n_pads}, n_sinks={dual_ports.n_sinks}) ----",
                flush=True,
            )
            continue

        print(f"---- {comp} stagger ----", flush=True)
        _run(
            [
                py,
                str(SRC / "run_tri_stagger.py"),
                str(cout),
                str(SEED),
                "--spice",
            ]
        )
        print(f"---- {comp} spice+plots ----", flush=True)
        m = spice_and_plots(cout)
        m["case"] = out.name
        m["comp"] = comp
        rows.append(m)
        print(
            f"  e_ir={m['e_ir_original_vs_tri_stagger']:.4f} "
            f"pearson={m['pearson_original_vs_tri_stagger']:.4f}",
            flush=True,
        )

    if not rows:
        raise RuntimeError(f"{out.name}: no components with nonzero sinks")

    html = out / "pdn_view_3d_tri_stagger.html"
    print(f"---- visualize {html.name} ----", flush=True)
    _run(
        [
            py,
            str(SRC / "visualize_pdn_3d.py"),
            str(out),
            "--mode",
            "dual",
            "--html",
            str(html),
        ]
    )
    return rows


def main() -> int:
    all_rows: list[dict] = []
    failures: list[dict] = []
    for tc in CASES:
        try:
            all_rows.extend(run_case(tc))
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR ibmpg{tc}: {exc}", flush=True)
            traceback.print_exc()
            failures.append({"case": f"ibmpg{tc}", "error": str(exc)})

    summary_path = HERE / "outputs/validation_summary_stagger_ibmpg3_6.json"
    summary_path.write_text(
        json.dumps({"rows": all_rows, "failures": failures}, indent=2)
    )
    print("\n======== SUMMARY ========", flush=True)
    print(
        f"{'case':22s} {'comp':6s} {'e_ir':>8s} {'pearson':>8s} "
        f"{'ax':>7s} {'ay':>7s} {'av':>8s}",
        flush=True,
    )
    for r in all_rows:
        print(
            f"{r['case']:22s} {r['comp']:6s} {r['e_ir_original_vs_tri_stagger']:8.4f} "
            f"{r['pearson_original_vs_tri_stagger']:8.4f} "
            f"{r.get('alpha_x', float('nan')):7.3f} "
            f"{r.get('alpha_y', float('nan')):7.3f} "
            f"{r.get('alpha_via', float('nan')):8.4f}",
            flush=True,
        )
    print(f"wrote {summary_path}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
