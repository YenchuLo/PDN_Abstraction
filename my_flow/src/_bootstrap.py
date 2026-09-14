"""Put my_flow/src and sibling spec_flow/src on sys.path for local / shared imports.

``my_flow/src`` is always first so local modules (io_artifacts, spice_emit, …)
shadow same-named files in ``spec_flow/src``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SPEC = _HERE.parent.parent / "spec_flow" / "src"


def ensure_paths() -> None:
    here_s = str(_HERE)
    spec_s = str(_SPEC)
    # Remove if present so we can re-order
    sys.path = [p for p in sys.path if p not in (here_s, spec_s)]
    # Insert SPEC then HERE so HERE (my_flow) ends up first
    sys.path.insert(0, spec_s)
    sys.path.insert(0, here_s)
