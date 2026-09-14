#!/usr/bin/env python3
"""Stage 01: parse IBM SPICE → multi-layer R-network artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_artifacts import save_net
from spice_parser import parse_spice


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Parse IBM flat or TSMC n-port SPICE into OUT/"
    )
    p.add_argument(
        "spice",
        type=Path,
        help="Path to ibmpgN.spice or TSMC .sp (with include/subckt)",
    )
    p.add_argument("out", type=Path, help="Output / work directory")
    args = p.parse_args(argv)

    if not args.spice.is_file():
        print(f"error: spice file not found: {args.spice}", file=sys.stderr)
        return 2

    net = parse_spice(args.spice)
    save_net(args.out, net, args.spice)
    meta_path = args.out / "net_meta.json"
    meta = json.loads(meta_path.read_text())
    print(json.dumps({"stage": "01_parse", "out": str(args.out), **meta}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
