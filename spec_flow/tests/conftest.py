"""Ensure flat imports resolve for tests (spec_flow/src/ on sys.path)."""

from __future__ import annotations

import sys
from pathlib import Path

_SPEC_FLOW = Path(__file__).resolve().parents[1] / "src"
if str(_SPEC_FLOW) not in sys.path:
    sys.path.insert(0, str(_SPEC_FLOW))
