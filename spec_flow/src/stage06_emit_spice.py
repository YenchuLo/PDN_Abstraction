#!/usr/bin/env python3
"""Stage 06: emit original / reduced-G' / Pixel-R SPICE decks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_artifacts import (
    load_fit,
    load_gprime,
    load_net,
    load_pixel_model_fields,
    load_ports,
)
from pixel_r import pixel_model_from_fields
from spice_emit import SIMULATORS, emit_all, resolve_simulator


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Emit three IR SPICE decks")
    p.add_argument("out", type=Path, help="Output / work directory")
    p.add_argument(
        "--simulator",
        choices=SIMULATORS,
        default=None,
        help="ngspice or spectre (default: env SPICE_SIMULATOR or ngspice)",
    )
    args = p.parse_args(argv)
    sim = resolve_simulator(args.simulator)

    net = load_net(args.out)
    ports, node_map = load_ports(args.out)
    Gprime, _meta = load_gprime(args.out)
    fields = load_pixel_model_fields(args.out)
    model = pixel_model_from_fields(ports, fields)
    fit_meta, _spectra = load_fit(args.out)

    paths = emit_all(
        args.out,
        net,
        ports,
        node_map,
        Gprime,
        pad_attach=model.pad_attach,
        arms=model.arms,
        n_edge_nodes=model.n_edge_nodes,
        Rx=float(fit_meta["Rx"]),
        Ry=float(fit_meta["Ry"]),
        Rz=float(fit_meta["Rz"]),
        simulator=sim,
    )
    print(
        json.dumps(
            {
                "stage": "06_emit_spice",
                "out": str(args.out),
                "simulator": sim,
                "decks": paths,
                "n_pads": ports.n_pads,
                "n_sinks": ports.n_sinks,
                "n_edge_nodes": model.n_edge_nodes,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
