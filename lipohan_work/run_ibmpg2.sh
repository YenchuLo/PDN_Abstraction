#!/usr/bin/env bash
# Run the five priority Pixel-R costs on ibmpg2.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${MY_FLOW_PYTHON:-/home/lipopo/miniconda3/envs/cfirstnet/bin/python}"
exec "$PY" lipohan_work/src/run_cost_sweep.py \
  --spice Benchmarks/IBM/TC2/ibmpg2.spice \
  "$@"
