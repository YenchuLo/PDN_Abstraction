#!/usr/bin/env python3
"""Stage 03: assemble G_M and Kron-reduce to G'_M."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph import assemble_conductance, galvanic_component_of, partition_ports
from grid_reff import write_grid_reff
from io_artifacts import load_net, load_ports, save_gprime
from kron import kron_reduce
from pg_reff import write_pg_reff


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Assemble conductance and Kron-reduce")
    p.add_argument("out", type=Path, help="Output / work directory (component dir)")
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
        **system.r_clip,
    }
    save_gprime(args.out, Gprime, meta)
    reff_paths: dict = {}
    try:
        reff_paths = write_grid_reff(args.out, Gprime, ports)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: grid R_eff skipped: {exc}", file=sys.stderr)
    pg_paths: dict = {}
    try:
        pg_paths = write_pg_reff(args.out, system, ports, net.voltages)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: real-PG R_eff skipped: {exc}", file=sys.stderr)
    summary = {"stage": "03_assemble_kron", "out": str(args.out), **meta}
    if reff_paths:
        summary["grid_reff"] = reff_paths
    if pg_paths:
        summary["pg_reff"] = pg_paths
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
