#!/usr/bin/env bash
# Run stages 01-08 for one IBM spice + chip_pitch (all VDD components).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HERE="$ROOT/spec_flow"
SRC="$HERE/src"
PY="${PY:-/home/yenchulo/anaconda3/bin/python}"
SPICE="$1"
PITCH="$2"
SEED="${3:-0}"
FIT_METHOD="${4:-eigen}"

if [[ "$FIT_METHOD" != "eigen" && "$FIT_METHOD" != "ir" ]]; then
  echo "error: fit_method must be eigen or ir (got: $FIT_METHOD)" >&2
  exit 1
fi

if [[ "$SPICE" != /* ]]; then
  SPICE="$ROOT/$SPICE"
fi
STEM="$(basename "$SPICE")"
STEM="${STEM%.spice}"
STEM="${STEM%.sp}"
if [[ "$FIT_METHOD" == "ir" ]]; then
  OUT="$HERE/outputs/${STEM}_n${PITCH}_ir"
  LOG="$HERE/logs/${STEM}_n${PITCH}_ir.log"
else
  OUT="$HERE/outputs/${STEM}_n${PITCH}"
  LOG="$HERE/logs/${STEM}_n${PITCH}.log"
fi
mkdir -p "$OUT" "$HERE/logs"

exec > >(tee -a "$LOG") 2>&1
echo "spice=$SPICE"
echo "chip_pitch=$PITCH"
echo "seed=$SEED"
echo "fit_method=$FIT_METHOD"
echo "OUT=$OUT"
echo "PY=$PY"
echo "start=$(date -Is)"

run() {
  local label="$1"; shift
  echo ""
  echo "=== $label ==="
  "$@"
}

run 01_parse        "$PY" "$SRC/stage01_parse.py" "$SPICE" "$OUT"
run 02_ports        "$PY" "$SRC/stage02_ports.py" "$OUT" "$PITCH"

mapfile -t COMPS < <("$PY" - <<PY
import json
from pathlib import Path
idx = json.loads(Path("$OUT/components.json").read_text())
for c in idx["components"]:
    print(c["path"])
PY
)

echo ""
echo "Processing ${#COMPS[@]} VDD component(s): ${COMPS[*]}"

for COMP in "${COMPS[@]}"; do
  COMP_OUT="$OUT/$COMP"
  echo ""
  echo "######## $COMP ########"
  run "${COMP}_03_assemble_kron" "$PY" "$SRC/stage03_assemble_kron.py" "$COMP_OUT"
  run "${COMP}_04_pixel_model"  "$PY" "$SRC/stage04_pixel_model.py" "$COMP_OUT"
  if [[ "$FIT_METHOD" == "ir" ]]; then
    run "${COMP}_05_fit_ir"     "$PY" "$SRC/stage05_fit_ir.py" "$COMP_OUT" "$SEED"
  else
    run "${COMP}_05_fit_eigen"  "$PY" "$SRC/stage05_fit_eigen.py" "$COMP_OUT"
  fi
  run "${COMP}_06_emit_spice"   "$PY" "$SRC/stage06_emit_spice.py" "$COMP_OUT"
  run "${COMP}_07_spice_solve"  "$PY" "$SRC/stage07_spice_solve.py" "$COMP_OUT"
  run "${COMP}_08_correlate_ir" "$PY" "$SRC/stage08_correlate_ir.py" "$COMP_OUT" "$SEED"
done

echo ""
echo "Done. Results in: $OUT"
for COMP in "${COMPS[@]}"; do
  echo "  $COMP metrics: $OUT/$COMP/metrics.json"
done
echo "end=$(date -Is)"
