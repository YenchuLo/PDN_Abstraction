#!/usr/bin/env python3
"""Stage 01: parse IBM SPICE → multi-layer R-network artifact."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SPEC = _HERE.parent.parent / "spec_flow" / "src"
if str(_SPEC) not in sys.path:
    sys.path.insert(0, str(_SPEC))


def _load_main():
    path = _SPEC / "stage01_parse.py"
    spec = importlib.util.spec_from_file_location("spec_flow_stage01_parse", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


if __name__ == "__main__":
    raise SystemExit(_load_main()())
