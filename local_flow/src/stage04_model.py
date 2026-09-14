#!/usr/bin/env python3
"""Stage 04: build localized Pixel-R topology + pad-lattice partitions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import ensure_paths

ensure_paths()

from io_artifacts import load_ports  # noqa: E402
from localized_pixel import build_localized_model  # noqa: E402

MODEL_JSON = "localized_model.json"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build localized Pixel-R topology")
    p.add_argument("out", type=Path, help="Component OUT directory")
    p.add_argument(
        "--block",
        type=int,
        default=2,
        help="Region block size K (0=global, 1=per-cell, K=K×K pads)",
    )
    args = p.parse_args(argv)

    ports, _node_map = load_ports(args.out)
    model = build_localized_model(ports, args.block)
    fields = model.to_fields()
    (args.out / MODEL_JSON).write_text(json.dumps(fields, indent=2))
    print(
        json.dumps(
            {
                "stage": "04_localized_model",
                "out": str(args.out),
                "topology": fields["topology"],
                "block_size": model.partition.block_size,
                "n_regions": model.n_regions,
                "n_pads": model.n_pads,
                "n_sinks": model.n_sinks,
                "n_edge_nodes": model.star.n_edge_nodes,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
