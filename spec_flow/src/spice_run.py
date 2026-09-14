"""Run ngspice or Cadence Spectre decks and parse port voltages."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from spice_emit import (
    ORIGINAL_SP,
    PIXEL_SP,
    PORT_MAP_JSON,
    REDUCED_SP,
    SPICE_DIR,
    load_simulator,
    resolve_simulator,
    spice_dir,
)

VOLT_FILES = {
    "original": "original.volt",
    "reduced_gprime": "reduced_gprime.volt",
    "pixel_r": "pixel_r.volt",
}

DECKS = {
    "original": ORIGINAL_SP,
    "reduced_gprime": REDUCED_SP,
    "pixel_r": PIXEL_SP,
}

_PRINT_RE = re.compile(
    r"v\((?P<node>[^)]+)\)\s*=\s*(?P<val>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)

# Spectre / spice print variants, e.g. "v(pad_0) = 1.8" or "V(pad_0): 1.8"
_SPECTRE_PRINT_RE = re.compile(
    r"v\((?P<node>[^)]+)\)\s*[:=]\s*(?P<val>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)


def require_ngspice() -> str:
    exe = shutil.which("ngspice")
    if not exe:
        raise FileNotFoundError(
            "ngspice not found on PATH. Install ngspice (e.g. apt install ngspice)."
        )
    return exe


def require_spectre() -> str:
    exe = os.environ.get("SPECTRE_BIN") or shutil.which("spectre")
    if not exe:
        raise FileNotFoundError(
            "spectre not found. Set SPECTRE_BIN or put spectre on PATH "
            "(Cadence Spectre 23.1+)."
        )
    return exe


def require_simulator(simulator: str = "ngspice") -> str:
    sim = resolve_simulator(simulator)
    if sim == "spectre":
        return require_spectre()
    return require_ngspice()


def spectre_mt() -> int:
    raw = os.environ.get("SPECTRE_MT", "64")
    try:
        n = int(raw)
    except ValueError as exc:
        raise ValueError(f"SPECTRE_MT must be an int, got {raw!r}") from exc
    if n < 1:
        raise ValueError(f"SPECTRE_MT must be >= 1, got {n}")
    return n


def parse_volt_file(path: str | Path, order: List[str]) -> Dict[str, float]:
    """
    Parse ngspice `print v(node)` redirect output.
    Lines look like: v(pad_0) = 1.800000e+00
    Mapped onto canonical order by print sequence (order list).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"voltage file not found: {path}")
    vals: List[float] = []
    nodes_seen: List[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _PRINT_RE.search(line)
        if not m:
            continue
        nodes_seen.append(m.group("node").lower())
        vals.append(float(m.group("val")))

    if len(vals) != len(order):
        raise ValueError(
            f"{path}: expected {len(order)} voltages, got {len(vals)}"
        )
    # Prefer canonical labels from order (ngspice lowercases node names)
    return {label: float(v) for label, v in zip(order, vals)}


def parse_spectre_log(path: str | Path, order: List[str]) -> Dict[str, float]:
    """
    Extract printed node voltages from a Spectre +log file.
    Uses print order when counts match; otherwise maps by node name.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"spectre log not found: {path}")

    vals: List[float] = []
    nodes_seen: List[str] = []
    for raw in path.read_text(errors="replace").splitlines():
        m = _SPECTRE_PRINT_RE.search(raw)
        if not m:
            continue
        nodes_seen.append(m.group("node").lower())
        vals.append(float(m.group("val")))

    if len(vals) == len(order):
        return {label: float(v) for label, v in zip(order, vals)}

    # Fallback: last occurrence per node name
    by_node: Dict[str, float] = {}
    for node, val in zip(nodes_seen, vals):
        by_node[node] = val
    missing = [lab for lab in order if lab.lower() not in by_node]
    if missing:
        raise ValueError(
            f"{path}: expected {len(order)} voltages, got {len(vals)}; "
            f"missing nodes e.g. {missing[:5]}"
        )
    return {lab: float(by_node[lab.lower()]) for lab in order}


def write_compat_volt(path: Path, order: List[str], volts: Dict[str, float]) -> None:
    """Write ngspice-shaped .volt so stage08 / plots keep working."""
    lines = [f"v({lab}) = {float(volts[lab]):.12g}" for lab in order]
    path.write_text("\n".join(lines) + "\n")


def run_deck(
    deck_path: str | Path,
    *,
    simulator: str = "ngspice",
    exe: Optional[str] = None,
    ngspice: Optional[str] = None,
    timeout_s: Optional[float] = 21600.0,
    mt: Optional[int] = None,
    volt_path: Optional[str | Path] = None,
    order: Optional[List[str]] = None,
) -> Path:
    """
    Run one deck. Returns the log path.

    ``ngspice`` is kept as a legacy kwarg alias for the binary when simulator
    is ngspice.

    If ngspice exits nonzero but ``volt_path`` parses with exactly ``len(order)``
    voltages (OP finished, teardown aborted), treat as success.
    """
    sim = resolve_simulator(simulator)
    deck_path = Path(deck_path).resolve()
    log_path = deck_path.with_suffix(".log")
    cwd = str(deck_path.parent)

    if sim == "spectre":
        binary = exe or require_spectre()
        n_mt = int(mt) if mt is not None else spectre_mt()
        cmd = [
            binary,
            "-64",
            "-format",
            "fsdb",
            "+log",
            log_path.name,
            "+spice",
            f"+mt={n_mt}",
            "+timer",
            "+aps",
            deck_path.name,
        ]
    else:
        binary = exe or ngspice or require_ngspice()
        cmd = [binary, "-b", "-o", str(log_path), str(deck_path)]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=cwd,
    )
    if proc.returncode != 0:
        if (
            sim == "ngspice"
            and volt_path is not None
            and order is not None
        ):
            try:
                parse_volt_file(volt_path, order)
                return log_path
            except (FileNotFoundError, ValueError, OSError):
                pass
        raise RuntimeError(
            f"{sim} failed on {deck_path} (exit {proc.returncode}). "
            f"See {log_path}. stderr={proc.stderr[-2000:]}"
        )
    return log_path


def run_all(
    out_dir: str | Path,
    *,
    timeout_s: Optional[float] = 21600.0,
    simulator: Optional[str] = None,
    mt: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    sd = spice_dir(out_dir)
    if simulator is None:
        sim = load_simulator(out_dir)
    else:
        sim = resolve_simulator(simulator)

    port_map = json.loads((sd / PORT_MAP_JSON).read_text())
    order: List[str] = list(port_map["order"])
    exe = require_simulator(sim)

    results: Dict[str, Dict[str, float]] = {}
    for key, deck_name in DECKS.items():
        deck = sd / deck_name
        if not deck.is_file():
            raise FileNotFoundError(f"missing deck {deck}")
        volt_path = sd / VOLT_FILES[key]
        log_path = run_deck(
            deck,
            simulator=sim,
            exe=exe,
            timeout_s=timeout_s,
            mt=mt,
            volt_path=volt_path if sim == "ngspice" else None,
            order=order if sim == "ngspice" else None,
        )
        if sim == "spectre":
            results[key] = parse_spectre_log(log_path, order)
            write_compat_volt(volt_path, order, results[key])
        else:
            results[key] = parse_volt_file(volt_path, order)
        (sd / f"{key}.volt.json").write_text(json.dumps(results[key], indent=2))
    return results


def load_voltages(out_dir: str | Path) -> Dict[str, Dict[str, float]]:
    sd = Path(out_dir) / SPICE_DIR
    port_map = json.loads((sd / PORT_MAP_JSON).read_text())
    order = list(port_map["order"])
    out: Dict[str, Dict[str, float]] = {}
    for key in DECKS:
        jpath = sd / f"{key}.volt.json"
        if jpath.is_file():
            out[key] = {str(k): float(v) for k, v in json.loads(jpath.read_text()).items()}
        else:
            out[key] = parse_volt_file(sd / VOLT_FILES[key], order)
    return out
