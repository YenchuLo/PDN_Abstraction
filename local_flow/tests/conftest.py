"""Ensure local_flow/src and spec_flow/src resolve for tests."""

from __future__ import annotations

import sys
from pathlib import Path

_LOCAL = Path(__file__).resolve().parents[1] / "src"
_SPEC = Path(__file__).resolve().parents[2] / "spec_flow" / "src"
# Insert spec first, then local, so local_flow shadows same-named modules.
for p in (_SPEC, _LOCAL):
    s = str(p)
    if s in sys.path:
        sys.path.remove(s)
sys.path.insert(0, str(_SPEC))
sys.path.insert(0, str(_LOCAL))
