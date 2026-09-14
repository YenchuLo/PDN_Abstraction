"""Smoke test: run staged flow on IBM TC1 with a coarse chip pitch."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ports import grid_counts
from spice_parser import parse_spice

REPO = Path(__file__).resolve().parents[2]
SPEC = Path(__file__).resolve().parents[1] / "src"
TC1 = REPO / "Benchmarks" / "IBM" / "TC1" / "ibmpg1.spice"

has_ngspice = shutil.which("ngspice") is not None


@pytest.mark.skipif(not TC1.is_file(), reason="IBM TC1 spice not present")
@pytest.mark.skipif(not has_ngspice, reason="ngspice not on PATH")
def test_ibm_tc1_staged_smoke(tmp_path):
    xmin, ymin, xmax, ymax = parse_spice(TC1).bbox
    chip_pitch = max(xmax - xmin, ymax - ymin) / 4.0
    out = tmp_path / "tc1_smoke"

    stages = [
        [sys.executable, str(SPEC / "stage01_parse.py"), str(TC1), str(out)],
        [sys.executable, str(SPEC / "stage02_ports.py"), str(out), str(chip_pitch)],
    ]
    for cmd in stages:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr + proc.stdout

    comps = json.loads((out / "components.json").read_text())
    # TC1 may have several galvanic islands, but they share layers {1,3} → one comp.
    assert comps["n_components"] == 1
    assert comps["components"][0]["path"] == "comp1"
    nx, ny = grid_counts((xmin, ymin, xmax, ymax), chip_pitch)

    for entry in comps["components"]:
        comp = out / entry["path"]
        for stage in (
            "stage03_assemble_kron.py",
            "stage04_pixel_model.py",
            "stage05_fit_eigen.py",
            "stage06_emit_spice.py",
            "stage07_spice_solve.py",
        ):
            proc = subprocess.run(
                [sys.executable, str(SPEC / stage), str(comp)],
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, proc.stderr + proc.stdout
        proc = subprocess.run(
            [sys.executable, str(SPEC / "stage08_correlate_ir.py"), str(comp), "0"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr + proc.stdout

        ports = json.loads((comp / "ports.json").read_text())
        assert ports["nx"] == nx and ports["ny"] == ny
        assert len(ports["cells"]) == nx * ny
        assert 1 <= len(ports["pad_nodes"]) <= nx * ny
        assert len(ports["pad_attach"]) == len(ports["pad_nodes"])
        assert abs(float(ports["vdd"]) - 1.8) < 1e-12
        assert all(abs(float(v) - 1.8) < 1e-12 for v in ports["pad_voltages"])
        assert ports.get("port_mode") in (
            "c4_overlap_grid",
            "imaginary_grid",
            None,
        )
        assert ports["vdd_layers"] == [1, 3]
        assert ports["bot_layer"] == 1
        assert ports["top_layer"] == 3
        assert float(ports["max_metal_pitch"]) > 0
        assert chip_pitch > float(ports["max_metal_pitch"])

        original_sp = (comp / "spice" / "original.sp").read_text()
        assert ".include" not in original_sp
        assert "ibmpg" not in original_sp.lower()
        assert "Rvia_" in original_sp

        pixel_model = json.loads((comp / "pixel_model.json").read_text())
        assert pixel_model["pad_attach"] == ports["pad_attach"]
        assert len(pixel_model["pad_attach"]) == len(ports["pad_nodes"])
        assert pixel_model.get("topology") == "star_half_arm"
        assert int(pixel_model.get("n_edge_nodes", 0)) > 0
        assert len(pixel_model.get("arms", [])) > 0

        pixel_r = json.loads((comp / "pixel_r.json").read_text())
        metrics = json.loads((comp / "metrics.json").read_text())
        assert pixel_r["Rx"] > 0 and pixel_r["Ry"] > 0 and pixel_r["Rz"] > 0
        assert metrics["nx"] <= 5 and metrics["ny"] <= 5
        assert metrics["cell_size"] == chip_pitch
        assert metrics["ir_source"] == "ngspice"
        assert "e_worst" in metrics
        assert math.isfinite(metrics["e_ir_reduced_vs_pixel_r"])
        assert math.isfinite(metrics["e_ir_original_vs_pixel_r"])
        assert metrics["relative_spectral_error"] >= 0.0

        spice = comp / "spice"
        for name in ("original.volt", "reduced_gprime.volt", "pixel_r.volt"):
            assert (spice / name).is_file()


@pytest.mark.skipif(not TC1.is_file(), reason="IBM TC1 spice not present")
def test_ibm_tc1_fit_ir_smoke(tmp_path):
    """Stages 01–05 with IR fitter (no ngspice)."""
    xmin, ymin, xmax, ymax = parse_spice(TC1).bbox
    chip_pitch = max(xmax - xmin, ymax - ymin) / 4.0
    out = tmp_path / "tc1_ir_smoke"

    stages = [
        [sys.executable, str(SPEC / "stage01_parse.py"), str(TC1), str(out)],
        [sys.executable, str(SPEC / "stage02_ports.py"), str(out), str(chip_pitch)],
    ]
    for cmd in stages:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr + proc.stdout

    comp = out / "comp1"
    for stage, extra in (
        ("stage03_assemble_kron.py", []),
        ("stage04_pixel_model.py", []),
        ("stage05_fit_ir.py", ["0"]),
    ):
        proc = subprocess.run(
            [sys.executable, str(SPEC / stage), str(comp), *extra],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr + proc.stdout

    pixel_r = json.loads((comp / "pixel_r.json").read_text())
    assert pixel_r["fit_method"] == "ir"
    assert pixel_r["Rx"] > 0 and pixel_r["Ry"] > 0 and pixel_r["Rz"] > 0
    assert float(pixel_r["relative_ir_error"]) >= 0.0
    assert (comp / "spectra.npz").is_file()
