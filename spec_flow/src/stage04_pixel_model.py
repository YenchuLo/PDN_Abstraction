#!/usr/bin/env python3
"""Stage 04: build single-layer Pixel-R star topology on the same ports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_artifacts import load_ports, save_pixel_model
from pixel_r import build_pixel_model


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build Pixel-R topology")
    p.add_argument("out", type=Path, help="Output / work directory")
    args = p.parse_args(argv)

    ports, _node_map = load_ports(args.out)
    model = build_pixel_model(ports)
    fields = model.to_fields()
    save_pixel_model(args.out, fields)
    summary = {
        "stage": "04_pixel_model",
        "out": str(args.out),
        "topology": fields.get("topology"),
        "n_pads": model.n_pads,
        "n_sinks": model.n_sinks,
        "n_edge_nodes": model.n_edge_nodes,
        "n_ew_shared": len(model.ew_shared),
        "n_ns_shared": len(model.ns_shared),
        "n_arms": len(model.arms),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
