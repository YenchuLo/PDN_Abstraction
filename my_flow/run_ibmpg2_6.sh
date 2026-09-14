#!/usr/bin/env bash
# Run my_flow (parse → Kron → stagger + tri-square) on IBM PG2-6.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Prefer MY_FLOW_PYTHON; else first python that imports scipy.
if [[ -z "${MY_FLOW_PYTHON:-}" ]]; then
  if python3 -c "import scipy" 2>/dev/null; then
    MY_FLOW_PYTHON=python3
  elif python -c "import scipy" 2>/dev/null; then
    MY_FLOW_PYTHON=python
  else
    echo "error: set MY_FLOW_PYTHON to a Python with numpy+scipy" >&2
    exit 1
  fi
fi
export MY_FLOW_PYTHON
PY="$MY_FLOW_PYTHON"
HERE="$ROOT/my_flow"
SRC="$HERE/src"
SEED=0
K=2

run_one() {
  local spice="$1"
  local stem
  stem="$(basename "$spice" .spice)"
  local OUT="$HERE/outputs/${stem}_auto_k1"
  echo "======== $stem ========"
  mkdir -p "$OUT"
  "$PY" "$SRC/stage01_parse.py" "$spice" "$OUT"
  "$PY" "$SRC/stage02_ports.py" "$OUT" 0 1 --grid-to-pad-ratio 1.0
  local comps
  comps=$(python3 -c "import json; d=json.load(open('$OUT/components.json')); print(' '.join(c['path'] for c in d['components']))")
  for COMP in $comps; do
    local COUT="$OUT/$COMP"
    echo "---- $COMP ----"
    "$PY" "$SRC/stage03_assemble_kron.py" "$COUT"
    "$PY" "$SRC/run_tri_stagger.py" "$COUT" "$SEED"
    "$PY" "$SRC/run_tri_square.py" "$COUT" "$SEED" --k "$K"
  done
}

for tc in 2 3 4 5 6; do
  run_one "$ROOT/Benchmarks/IBM/TC${tc}/ibmpg${tc}.spice"
done

echo ""
echo "======== SUMMARY ========"
"$PY" - <<'PY'
import json
from pathlib import Path
root = Path("my_flow/outputs")
rows = []
for case_dir in sorted(root.glob("ibmpg*_auto_k1")):
    for comp in sorted(case_dir.glob("comp*")):
        row = {"case": case_dir.name, "comp": comp.name}
        for key, fname in (
            ("tri_stagger", "tri_stagger_r.json"),
            ("tri_square", "tri_square_r.json"),
        ):
            p = comp / fname
            if not p.is_file():
                continue
            m = json.loads(p.read_text())
            row[f"{key}_rel"] = m.get("relative_ir_error")
            row[f"{key}_e"] = m.get("e_worst") or (
                (m.get("e_ir_by_stimulus") or {}).get("original")
            )
        if len(row) > 2:
            rows.append(row)
print(f"{'case':22s} {'comp':6s} {'ts_rel':>9s} {'ts_e':>8s} {'tq_rel':>9s} {'tq_e':>8s}")
for r in rows:
    print(
        f"{r['case']:22s} {r['comp']:6s} "
        f"{r.get('tri_stagger_rel', float('nan')):9.4f} "
        f"{r.get('tri_stagger_e', float('nan')):8.4f} "
        f"{r.get('tri_square_rel', float('nan')):9.4f} "
        f"{r.get('tri_square_e', float('nan')):8.4f}"
    )
Path("my_flow/outputs/validation_summary_ibmpg2_6.json").write_text(
    json.dumps(rows, indent=2)
)
print("wrote my_flow/outputs/validation_summary_ibmpg2_6.json")
PY
