#!/usr/bin/env python3
"""Stage 03: assemble G_M and Kron-reduce to G'_M (dual-layer ports)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import ensure_paths

ensure_paths()

from graph import assemble_conductance, galvanic_component_of, partition_ports  # noqa: E402
from io_artifacts import load_net, load_ports, save_gprime  # noqa: E402
from kron import kron_reduce  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Assemble conductance and Kron-reduce")
    p.add_argument("out", type=Path, help="Component OUT directory")
    args = p.parse_args(argv)

    net = load_net(args.out)
    ports, node_map = load_ports(args.out)
    keep = galvanic_component_of(net, node_map, ports.port_nodes)
    system = assemble_conductance(net, node_map, keep_roots=keep)
    port_idx, internal_idx = partition_ports(system, ports.port_nodes)
    Gprime = kron_reduce(system.G, port_idx, internal_idx)

    meta = {
        "n_nodes": len(system.nodes),
        "n_ports": int(ports.n_ports),
        "n_pads": int(ports.n_pads),
        "n_sinks": int(ports.n_sinks),
        "n_internal": int(len(internal_idx)),
        "n_vdd_roots": int(len(keep)),
        "Gprime_shape": list(Gprime.shape),
        "pitch_top": float(ports.pitch_top),
        "pitch_bot": float(ports.pitch_bot),
        **system.r_clip,
    }
    save_gprime(args.out, Gprime, meta)
    print(json.dumps({"stage": "03_assemble_kron", "out": str(args.out), **meta}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
