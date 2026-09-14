# my_flow — PDN Abstraction (Pixel-R-style globals + real ports)

Staged abstraction on IBM power-grid benchmarks. Leaves
[`spec_flow/`](../spec_flow/) (Pixel-R) unchanged. Python sources live in
[`src/`](src/); drivers (`run_flow.tcl`, `run_ibmpg2_6.sh`) stay here.

All models share the same **R / port / fit** policy:

| Aspect | Behavior |
|--------|----------|
| Lateral R | Global **`Rx`, `Ry`** (length-proportional: \(R=Rx\cdot\|L\|/\mathrm{pitch}\)); tri-square uses per-sheet `Rx_u/Ry_u` and `Rx_l/Ry_l` |
| Via R | Global **`Rz`** (stagger); tri-square uses `Rz_pad` + `Rz_ul` |
| Ports | **Real C4** pad XY (cell has Vsrc iff a C4 overlaps it; drive = IBM C4 V); **full** `nx×ny` sink grid (zero-`I` cells kept; `--uniform-current` only changes the current map) |
| Attach | **Stubs** to nearest mesh node (`--via-stub rxry`, default); `--via-stub zero` snaps the via onto that node |
| Pitch (`pitch_bot=0`) | **Same as spec_flow**: C4 lattice, clamped to \(>\) max metal pitch |
| Fit | Same as `spec_flow` IR: \(\min\sum_j\|\mathrm{drop}_S-\mathrm{drop}_M\|^2\) on \(\log_{10}(R)\) |

Canonical attach:

```text
pad_k -- Rz -- landing(pad XY) -- stub(∝ Rx/Ry) -- nearest mesh node
sink_j -- Rz -- landing(sink XY) -- stub(∝ Rx/Ry) -- nearest mesh node
```

## Models

1. **tri-stagger** — mid square + distinct pad/sink landings + stubs (`Rx,Ry,Rz`)
2. **tri-square** — L2 mid at Pixel-R pitch + denser L3 (`pitch/k`), **6 R params** `(Rx_u,Ry_u,Rx_l,Ry_l,Rz_pad,Rz_ul)`, stubs

VoltSpot (`run_voltspot.py`) stays a separate diagnostic path.

## Setup

```bash
pip install -r my_flow/requirements.txt
# ngspice + tclsh on PATH (default), or Cadence Spectre 23.1+ for --simulator spectre
export MY_FLOW_PYTHON=/path/to/conda/python   # if system python lacks scipy
```

## Run

```tcl
# my_flow/run_flow.tcl
set ibm_case    "ibmpg2"   ;# ibmpg1..6 or TC1..6
set pitch_bot   0          ;# 0 = auto (spec_flow: C4 lattice, > max metal)
set coarsen_k   1
set grid_to_pad_ratio 1.0  ;# 1.0 = match C4 spacing; 4.0 denser (÷2)
set seed        0
set tri_square_k 2
set perturb_layer ""       ;# "" | metal int | all  (grid-cell R perturb)
set perturb_amp   0.2
set via_stub      "rxry"   ;# rxry (L-bend) | zero (snap to nearest grid)
```

```bash
export MY_FLOW_PYTHON="${MY_FLOW_PYTHON:-python3}"   # must import numpy+scipy
tclsh my_flow/run_flow.tcl            # default ibm_case → tri-stagger + tri-square
tclsh my_flow/run_flow.tcl ibmpg3     # CLI override (or TC3 / 3)
tclsh my_flow/run_flow.tcl ibmpg2 --spice   # also emit tri-stagger SPICE decks
tclsh my_flow/run_flow.tcl ibmpg2 --via-stub zero   # no L-bend; via on nearest node
tclsh my_flow/run_flow.tcl ibmpg2 --perturb-layer 5
tclsh my_flow/run_flow.tcl ibmpg2 --perturb-layer all --perturb-amp 0.2 --seed 0
bash my_flow/run_ibmpg2_6.sh
```

Output: `my_flow/outputs/<spice_stem>_auto_k<k>/` (or `_pb<pitch>_k<k>/`).
`--via-stub zero` appends `_nostub`.

### Stage by stage

```bash
export MY_FLOW_PYTHON="${MY_FLOW_PYTHON:-python3}"
OUT=my_flow/outputs/ibmpg2_auto_k1
COMP="$OUT/comp1"
"$MY_FLOW_PYTHON" my_flow/src/stage01_parse.py Benchmarks/IBM/TC2/ibmpg2.spice "$OUT"
"$MY_FLOW_PYTHON" my_flow/src/stage02_ports.py "$OUT" 0 1 --grid-to-pad-ratio 1.0 \
  --perturb-layer all --perturb-amp 0.2 --seed 0
"$MY_FLOW_PYTHON" my_flow/src/stage03_assemble_kron.py "$COMP"
"$MY_FLOW_PYTHON" my_flow/src/run_tri_stagger.py "$COMP" 0 --spice
"$MY_FLOW_PYTHON" my_flow/src/run_tri_square.py "$COMP" 0 --k 2
# Snap vias to nearest grid (no L-bend stubs)
"$MY_FLOW_PYTHON" my_flow/src/run_tri_stagger.py "$COMP" 0 --via-stub zero
"$MY_FLOW_PYTHON" my_flow/src/run_tri_square.py "$COMP" 0 --k 2 --via-stub zero
```

| Artifact | Meaning |
|----------|---------|
| `compN/tri_stagger_model.json`, `tri_stagger_r.json` | Mid-square + fitted `Rx,Ry,Rz` |
| `compN/tri_square_model.json`, `tri_square_r.json` | L2/L3 + fitted 6R |
| `compN/ir_*_tri_stagger.png` / `…_tri_square.png` | Reduced-vs-model IR plots |

Pad-only / zero-load components are **skipped** (`"skipped": true`); use `--strict` to fail.

### Eigen spectrum of an IR fit

Back-project the IR-fitted \(G_S\) into the \(G'\) eigenbasis
(\(\lambda_S=\mathrm{diag}(Q^\top G_S Q)\)) and optionally re-run eigenvalue LS on the
same model for a side-by-side compare:

```bash
"$MY_FLOW_PYTHON" my_flow/src/compare_ir_spectra.py "$OUT/comp1" --model tri_stagger
"$MY_FLOW_PYTHON" my_flow/src/compare_ir_spectra.py "$OUT/comp1" --model tri_square --refit-eigen
```

Writes `eigen_ir_compare_<model>.{json,npz,_scatter.png,_sorted.png,_relerr.png}`.

### IR plots

| File | Meaning |
|------|---------|
| `ir_scatter_reduced_vs_tri_stagger.png` / `…_tri_square.png` | Reduced \(G'\) vs model (IR % of Vdd, ±20% bands) |
| `ir_spatial_reduced_vs_tri_stagger.png` / `…_tri_square.png` | Side-by-side spatial IR % |
| `ir_error_heatmap_tri_stagger.png` / `…_tri_square.png` | Sink \(\lvert\Delta IR\rvert\) vs reduced |

3D viewer: **Tri-stagger** and **Tri-square**.

```bash
export MY_FLOW_PYTHON="${MY_FLOW_PYTHON:-python3}"
"$MY_FLOW_PYTHON" my_flow/src/visualize_pdn_3d.py my_flow/outputs/ibmpg2_auto_k1 \
  --model tri_stagger --comp 1 \
  --html my_flow/outputs/ibmpg2_auto_k1/pdn_view_3d_tri_stagger.html
"$MY_FLOW_PYTHON" my_flow/src/visualize_pdn_3d.py my_flow/outputs/ibmpg2_auto_k1 \
  --model tri_square --comp 1 \
  --html my_flow/outputs/ibmpg2_auto_k1/pdn_view_3d_tri_square.html
```

Each command also writes a matching `.png` snapshot via matplotlib (no kaleido; use `--no-png` to skip).

## Notes

- **ibmpg1** is a known outlier; validate on **ibmpg2–6**.
- Compare methodology to Pixel-R (`spec_flow`), not identical attach topology
  (Pixel-R shorts pad→center; my_flow uses real-XY stubs). Port lattice is the
  same C4-overlap / full-grid rule.
- Shared helpers: `stub_attach.py` (`--via-stub rxry|zero`), `fit_rxryrz.py`.
- `ports_dual.py` / `DualPortSet` name the port lattice (not a dual-layer model).

## Reference papers

See repo root / `my_flow/paper/`.
