#!/usr/bin/env python3
"""Stage 07: run ngspice or Spectre on the three IR decks and parse port voltages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spice_emit import SIMULATORS, load_simulator, resolve_simulator
from spice_run import run_all


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Solve three IR SPICE decks")
    p.add_argument("out", type=Path, help="Output / work directory")
    p.add_argument(
        "timeout_s",
        type=float,
        nargs="?",
        default=21600.0,
        help="Per-deck timeout seconds (default 21600 = 6h)",
    )
    p.add_argument(
        "--simulator",
        choices=SIMULATORS,
        default=None,
        help="ngspice or spectre (default: spice/simulator.json, else env / ngspice)",
    )
    args = p.parse_args(argv)

    if args.simulator is None:
        sim = load_simulator(args.out)
    else:
        sim = resolve_simulator(args.simulator)

    try:
        results = run_all(args.out, timeout_s=args.timeout_s, simulator=sim)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"error: spice solve failed: {exc}", file=sys.stderr)
        return 1

    summary = {
        "stage": "07_spice_solve",
        "out": str(args.out),
        "simulator": sim,
        "decks": list(results.keys()),
        "n_ports": len(next(iter(results.values()))),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
