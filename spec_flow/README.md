# Spec-flow: TSMC PDN Abstraction on IBM PG

Staged Pixel-R extraction on IBM power-grid benchmarks (`Benchmarks/IBM/TC*`),
driven by TCL. Python sources live in [`src/`](src/); drivers (`run_flow.tcl`,
`run_bench.sh`) stay here. Python stages are separate scripts with **positional
arguments only** (no flags). After Pixel-R fit, IR is obtained with **ngspice**
on three netlists: R-only multi-layer mesh, Kron-reduced port SPICE, and Pixel-R
SPICE.

## Port model

Raw IBM `V` / `I` instances are not each a port. Pads are real overlapping C4s
(same net); currents are lumped per grid cell. The flow discovers **VDD connected
components** (seeds: `inj<0` sinks + nonzero `V` terminals; expand via `R` and
cross-layer `V=0` vias), ordered **topmost-layer first** as `comp1`, `comp2`, ….
Each component is abstracted independently. GND-stack metals are omitted from
ports and from the R-mesh.
Components are split by **layer set** (interleaved rails with different metals
stay separate). Galvanic islands that share the same layers (e.g. several
disconnected {1, 3} meshes on TC1) are merged into one component. Multi-rail
stacks (e.g. ibmpg3/ibmpg5) produce multiple `compN/` results.

For every chip-pitch cell `(ix, iy)` on a given component (`port_mode=c4_overlap_grid`):

- **Voltage pad (Top / vsrc)** — real C4 (nonzero `V→gnd`) that **overlaps** the
  cell (same VDD component); `pad_node` / `pad_xy` are that C4. Cells with no C4
  are sink-only unless `--full-pads` (then unique nearest unused top-metal
  node; **all** pads share one common drive = representative of real C4 Vs).
  Driven at the **IBM C4 voltage** (e.g. 1.8 on ibmpg2), not a
  fixed 0.9.
- **Current sink / IR probe (Center)** — unique nearest node on that
  component’s **bottommost** layer to the cell center (full `nx×ny` grid;
  empty-current cells still get a sink).
- **Currents** — IBM instance injections on **that component** only, lumped by
  `(x, y)` onto each cell’s sink

Cross-layer `V=0` vias stay as near-ideal conductances (nodes not unioned) so
top/bot ports remain distinct. Multi-layer and Pixel-R share the same ports
(`n_sinks = nx * ny`; `n_pads ≤ n_sinks` with `pad_attach`; `n_pads = n_sinks`
when `--full-pads`). Port count is
`n_pads + n_sinks`, so keep `chip_pitch` coarse enough for dense `eigh` on `G'`.
Pad→center offset is treated as a short. Real-XY L-bend attach is `--via-stub`
on `my_flow`.

### Pixel-R star unit cell

Pixel-R is a **single-layer grid of identical stars** (not a sink-to-sink mesh):

- Center = sink / probe + `Isink`; Top = pad / `vsrc`
- `Rup = Rdown = Rz` (Top↔Center↔internal Bottom)
- Half-arms `Rx` / `Ry` stretch from Center to shared edge midpoints (boundary
  cells keep dangling half-arms). Neighbor path through a shared edge is
  `2·Rx` or `2·Ry` in series.
- Port admittance `G_S` is the Schur complement of edge + bottom internals.

**Constraint:** `chip_pitch` must be **strictly greater** than the maximum VDD
metal stripe pitch (inferred per layer as the largest consecutive preferred-direction
track gap, including inter-band voids). On IBM TC1 that max is typically ~1969
layout units (M3).

## Pipeline

1. `stage01_parse.py` — parse IBM flat SPICE (`R` / `I` / `V`) → `OUT/`
2. `stage02_ports.py` — discover VDD CCs; print count; ports per `OUT/compN/`
3. For each `compN/` (drivers loop automatically):
   1. `stage03_assemble_kron.py` — sparse \(G_M\); Kron → \(G'_M\)
   2. `stage04_pixel_model.py` — Pixel-R topology on the same ports
   3. Fit Pixel-R \((R_x, R_y, R_z)\) (choose one):
      - `stage05_fit_eigen.py` — eigenvalue LS (TSMC spec, **default**)
      - `stage05_fit_ir.py` — multi-stimulus mixed-BC IR voltage match
   4. `stage06_emit_spice.py` — emit three SPICE decks under `compN/spice/`
   5. `stage07_spice_solve.py` — run ngspice or Spectre `.op` on each deck
   6. `stage08_correlate_ir.py` — sink IR correlation + plots

## Setup

```bash
pip install -r spec_flow/requirements.txt
# ngspice on PATH (default), or Cadence Spectre 23.1+ for --simulator spectre
# Optional: export SPICE_SIMULATOR=spectre SPECTRE_BIN=/path/to/spectre SPECTRE_MT=64
```

## Run (TCL)

Edit parameters in [`run_flow.tcl`](run_flow.tcl):

```tcl
set ibm_case    "ibmpg2"   ;# ibmpg1..6 or TC1..6
set chip_pitch  0          ;# 0 = auto C4 lattice, clamped > max metal
set seed        0
set fit_method  "eigen"    ;# or "ir"
```

Then:

```bash
tclsh spec_flow/run_flow.tcl            # ibm_case + auto pitch
tclsh spec_flow/run_flow.tcl ibmpg3     # CLI overrides ibm_case
tclsh spec_flow/run_flow.tcl ibmpg2 ir  # optional fit_method
```

`OUT` is derived automatically:

```text
spec_flow/outputs/<spice_stem>_nauto               # chip_pitch=0, eigen
spec_flow/outputs/<spice_stem>_n<chip_pitch>       # explicit pitch, eigen
spec_flow/outputs/<spice_stem>_n…_ir               # fit_method=ir
# e.g. spec_flow/outputs/ibmpg2_nauto
```

Changing `ibm_case`, `chip_pitch`, or `fit_method` changes `OUT`. Auto pitch uses
the C4 lattice period (median unique-row/col spacing) when legal, otherwise
`max_metal_pitch+1` (Pixel-R requires `chip_pitch > max_metal_pitch`).

Bash driver (optional 4th arg = fit method; pass a numeric pitch or use stage02 auto):

```bash
./spec_flow/run_bench.sh Benchmarks/IBM/TC2/ibmpg2.spice 927.0 0 eigen
./spec_flow/run_bench.sh Benchmarks/IBM/TC2/ibmpg2.spice 927.0 0 ir
```

**Note:** With C4-overlap pads on the full sink grid, `chip_pitch` must be coarse
enough that `nx*ny` does not exceed the number of unique bottom power nodes
(and that `n_pads + n_sinks` stays tractable for dense eigendecomposition), **and**
`chip_pitch > max_metal_pitch`.

## Run stages one by one

```bash
OUT=spec_flow/outputs/ibmpg2_nauto
python3 spec_flow/src/stage01_parse.py Benchmarks/IBM/TC2/ibmpg2.spice "$OUT"
python3 spec_flow/src/stage02_ports.py "$OUT"          # 0 / omitted = auto legal pitch
# or: python3 spec_flow/src/stage02_ports.py "$OUT" 927.0
# stages 03–08 take a component directory (comp1 = topmost VDD rail)
COMP="$OUT/comp1"
python3 spec_flow/src/stage03_assemble_kron.py "$COMP"
python3 spec_flow/src/stage04_pixel_model.py "$COMP"
python3 spec_flow/src/stage05_fit_eigen.py "$COMP"
# or: python3 spec_flow/src/stage05_fit_ir.py "$COMP" 0
python3 spec_flow/src/stage06_emit_spice.py "$COMP"
python3 spec_flow/src/stage07_spice_solve.py "$COMP"
python3 spec_flow/src/stage08_correlate_ir.py "$COMP" 0
# Spectre (23.1+; decks must be emitted for spectre):
# python3 spec_flow/src/stage06_emit_spice.py "$COMP" --simulator spectre
# python3 spec_flow/src/stage07_spice_solve.py "$COMP" --simulator spectre
# Invokes: spectre -64 -format fsdb +log <deck>.log +spice +mt=$SPECTRE_MT +timer +aps <deck>.sp
```

## Artifacts in `OUT/`

| File | Content |
|------|---------|
| `net.npz` / `net_meta.json` | Parsed netlist (shared) |
| `components.json` | VDD CC index (rank, layers, paths) |
| `compN/ports.json` / `node_map.json` | C4-overlap pads + full-grid sinks for that rail |
| `compN/Gprime.npy` / `system_meta.json` | Kron-reduced port matrix |
| `compN/grid_reff.json` / `grid_reff.npz` / `grid_reff_maps.png` / `grid_reff_dp.png` | Kron-branch \(R_z, R_{\mathrm{EW}}, R_{\mathrm{NS}}\) plus driving-point \(Z_{kk}\) |
| `compN/pg_reff.json` / `pg_reff.npz` / `pg_reff_map.png` | Unreduced mesh \(R_{\mathrm{eff}}\) (real C4 pads grounded) |
| `compN/pixel_model.json` | Pixel-R topology |
| `compN/pixel_r.json` / `spectra.npz` | Fitted \(R_x,R_y,R_z\) + spectra |
| `compN/spice/original.sp` | That rail’s R-only mesh + C4 pad V / lumped sink I |
| `compN/spice/reduced_gprime.sp` | Port-level resistor net from \(G'\) |
| `compN/spice/pixel_r.sp` | Pixel-R star (half-arm Rx/Ry, Rup/Rdown) |
| `compN/spice/*.volt` / `*.volt.json` | port voltages (ngspice or Spectre) |
| `compN/spice/simulator.json` | selected simulator (`ngspice` / `spectre`) |
| `compN/metrics.json` / `*.png` | IR correlation metrics and plots |
| `ir_scatter_reduced_vs_pixel_r.png` | Reduced vs Pixel-R (IR % of Vdd, ±20% bands) |
| `ir_spatial_reduced_vs_pixel_r.png` | Side-by-side spatial IR % maps (reduced vs Pixel-R) |

## Eigen vs IR spectral compare

After running both fit methods (or with `--refit-eigen` on an IR OUT), back-project
the IR-fit \(G_S\) into the \(G'\) eigenbasis and overlay against the eigen LS spectrum:

```bash
python3 spec_flow/src/compare_eigen_ir.py \
  spec_flow/outputs/ibmpg2_nauto/comp1 \
  spec_flow/outputs/ibmpg2_nauto_ir/comp1

# Or only an IR OUT: re-run Pixel-R eigen LS for the comparison
python3 spec_flow/src/compare_eigen_ir.py --refit-eigen \
  spec_flow/outputs/ibmpg2_nauto_ir/comp1
```

Writes `eigen_ir_compare.json`, `eigen_ir_compare_spectra.npz`, and scatter / sorted /
per-mode relative-error PNGs under the IR comp dir (or `--out`).

## IR definition

For each deck: \(\mathrm{IR}_k = \mathrm{mean}(V_{\mathrm{pads}}) - V_{\mathrm{sink},k}\).

Reported pairs: original vs Pixel-R, original vs reduced, reduced vs Pixel-R.
Stage 08 always writes the reduced-vs-Pixel-R scatter and spatial PNGs above.

## Performance notes

- Kron uses one SuperLU (`splu`) factorization of \(G_{II}\) and a multi-RHS solve.
- Eigen fitting needs the **full** spectrum of dense \(G'\), so `stage05_fit_eigen` uses `eigh`.
- IR fitting (`stage05_fit_ir`) skips `eigh`: one \(G_{ss}\) factorization of \(G'_M\) plus a few mixed-BC solves per residual (same cost class, typically cheaper).
- Prefer coarser `chip_pitch`: port count is \(n_{\mathrm{pads}}+n_{\mathrm{sinks}}\)
  with \(n_{\mathrm{sinks}}=n_x n_y\) and \(n_{\mathrm{pads}}\le n_{\mathrm{sinks}}\).
- Stage 07 solves the R-only multi-layer mesh in ngspice (can dominate runtime).

## TSMC n-port virtual Pixel-R

When a filled region-`VDD_PORT` lattice is detected (≥4 cells), stage 02 builds
**bump-mapped** pads/sinks on the index→micron map (TC1: origin (30,30), pitch 60)
instead of IBM C4-overlap lattice ports:

- Sink = `region_*_VDD_PORT` at lattice microns
- Pad = unique `vpad_x_*_y_*` near-shorted to the tile named in
  `* BUMP_VDD_{bx}_{by} <tile> …` (not to `VDD_in`)
- Empty bump cells get an isolated open-circuit pad (zero G′ row/col)
- Same tile in several bumps → several pads onto that tile
- Package pin shorts (`Xdie1` tile→`VDD_in`) are ignored so tiles stay real mesh nodes
- Plots: `comp1/tsmc_virtual_sinks_current.png`, `tsmc_virtual_pads.png`
- Driver uses the same default as IBM: `fit_method=eigen` (pass `ir` for IR fit)

```bash
# Requires Benchmarks/TSMC/TC1/myALL_nportModel_without_package.sp
tclsh spec_flow/run_flow.tcl tsmc1          # eigen-fit by default
tclsh spec_flow/run_flow.tcl tsmc1 ir       # mixed-BC IR fit
python3 spec_flow/src/plot_tsmc_virtual_grid.py spec_flow/outputs/.../comp1
```

See also root [`README.md`](../README.md) § TSMC n-port.

## PDN visualizer

Interactive Plotly XY / 3D viewers for metal geometry. Multi-rail designs expose
**VDD1..VDDN** (topmost-first, same order as `compN/`) plus **VSS**. Reads a raw
IBM `.spice` or a flow `OUT/` with `net.npz`.

```bash
pip install -r spec_flow/requirements.txt
python3 spec_flow/src/visualize_pdn.py Benchmarks/IBM/TC1/ibmpg1.spice
python3 spec_flow/src/visualize_pdn.py Benchmarks/IBM/TC5/ibmpg5.spice --net vdd2
python3 spec_flow/src/visualize_pdn.py spec_flow/outputs/ibmpg1_n500 --net vss --layer 0
python3 spec_flow/src/visualize_pdn_3d.py spec_flow/outputs/ibmpg1_n5195.75 --comp 1
# ibmpg2 IR run (recommended example)
python3 spec_flow/src/visualize_pdn_3d.py spec_flow/outputs/ibmpg2_nauto_ir \
  --comp 1 --mode both \
  --html spec_flow/outputs/ibmpg2_nauto_ir/pdn_view_3d_pixel_r.html
# optional: open in a local browser (off by default; use on a desktop session)
python3 spec_flow/src/visualize_pdn.py Benchmarks/IBM/TC1/ibmpg1.spice --html /tmp/pdn.html --browser
```

Writes HTML only by default (no auto-open). In-page **View** lists each VDD
component and VSS (All or M\<layer\>). 3D Pixel-R overlay loads from
`OUT/compK/` for the selected `vddK` (`--comp K` sets the initial pair).

## Tests

```bash
python3 -m pytest spec_flow/tests -q
```

## Notes

- Full `nx×ny` sink grid (empty current cells still get a sink; current may be 0).
- Ground `V n 0 0` → ground merge; same-layer `V=0` merged; cross-layer `V=0` → via `R`.
- Voltage pads = real C4 sites that overlap a cell (same net); pad drive = IBM C4 V.
  With `--full-pads`, padless cells also get the unique nearest unused top-metal node.
- SRAM-PG is out of scope (no layout coordinates in node names).
