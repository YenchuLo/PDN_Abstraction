"""Put lipohan_work/src first, then spec_flow/src, on sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SPEC = _HERE.parent.parent / "spec_flow" / "src"


def ensure_paths() -> None:
    here_s = str(_HERE)
    spec_s = str(_SPEC)
    sys.path = [p for p in sys.path if p not in (here_s, spec_s)]
    sys.path.insert(0, spec_s)
    sys.path.insert(0, here_s)
