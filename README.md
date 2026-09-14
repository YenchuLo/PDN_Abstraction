# TSMC PDN Abstraction

IBM power-grid SPICE → port lattice → Kron-reduced \(G'\) → abstract R-model →
IR correlation plots.

All run outputs are written under `my_flow/outputs/`, `spec_flow/outputs/`, or
`local_flow/outputs/` (created automatically). Run everything from the **repo
root**.

| Directory                               | Model                               | Role                           |
| --------------------------------------- | ----------------------------------- | ------------------------------ |
| [`my_flow/src/`](my_flow/src/)         | Tri-stagger, tri-square             | Main abstraction + IR plots    |
| [`spec_flow/src/`](spec_flow/src/)     | Pixel-R + shared Kron/graph helpers | Required sibling of `my_flow` |
| [`local_flow/src/`](local_flow/src/)   | Localized Pixel-R                   | Uniformity ablation of Pixel-R |
| [`Benchmarks/IBM/`](Benchmarks/IBM/)   | `TC1`…`TC6` SPICE decks        | Optional local inputs (not in source tarball) |
| [`Benchmarks/TSMC/TC1/`](Benchmarks/TSMC/TC1/) | TSMC n-port deck | Optional local input for `tsmc1` |

`local_flow` reuses `spec_flow` stages 01–03. `my_flow` is a different
abstraction family (stubs onto a dual mesh), not a flag overlay on Pixel-R.
`my_flow/src` imports shared code from `spec_flow/src` via `_bootstrap.py`.
Prefer **ibmpg2–6** for IBM validation (ibmpg1 is a known irregular outlier).

| Flow | Model | Ports | R parameters | Role |
| ---- | ----- | ----- | ------------ | ---- |
| [`spec_flow`](spec_flow/) | Pixel-R identical stars | C4-overlap pads + full-grid lumped sinks | One global \((R_x,R_y,R_z)\) | TSMC spec path (eigen LS default) |
| [`local_flow`](local_flow/) | Same stars, region-wise R | Same as spec | Per-block / per-cell \((R_x,R_y,R_z)\) | Uniformity ablation of Pixel-R |
| [`my_flow`](my_flow/) | Tri-stagger + tri-square | Same C4/full-grid ports; attach at real pad/sink XY | Global 3R (stagger) or 6R (square) | Competing mesh + L-bend stubs |

### What the source `.tar.gz` can run

Extract and work from the archive root (directory that contains this `README.md`).
IBM / TSMC decks under `Benchmarks/` are optional local inputs.

| README path | Needs | Runnable from source alone? |
| ----------- | ----- | ---------------------------- |
| **TSMC/TC1** via `spec_flow` (`tsmc1`, plot lattice, eigen compare) | `Benchmarks/TSMC/TC1/` | No — add TSMC decks locally |
| `pytest` TSMC tests (deck-backed) | same | No — skipped if missing |
| `my_flow` / IBM quick start / batch | `Benchmarks/IBM/` | No — add IBM decks locally |

**Model comparison (primary deliverable):** run **both** `my_flow` and
`spec_flow` and compare the reduced-vs-model IR plots:

| Arm                      | Driver                    | Key PNGs (under each `compN/`)                                                                                  |
| ------------------------ | ------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Tri-stagger / tri-square | `my_flow` (§2 below)   | `ir_scatter_reduced_vs_tri_stagger.png`, `ir_spatial_reduced_vs_tri_stagger.png` (and `…_tri_square.png`) |
| Pixel-R                  | `spec_flow` (§3 below) | `ir_scatter_reduced_vs_pixel_r.png`, `ir_spatial_reduced_vs_pixel_r.png`                                     |

`my_flow` alone does **not** write the Pixel-R PNGs; you must run `spec_flow`.

---

## Prerequisites

- Python 3.10+ with `numpy`, `scipy`, `matplotlib` (and `plotly` for the 3D viewer)
- One SPICE backend for optional decks (see below): **ngspice** (default) or **Cadence Spectre**
- `tclsh` on `PATH` (optional; used by the Tcl drivers)

```bash
# from repo root
pip install -r spec_flow/requirements.txt   # also covers local_flow
pip install -r my_flow/requirements.txt     # Plotly 3D + my_flow extras

# if system python lacks scipy, point the flow at a conda/env python:
export MY_FLOW_PYTHON=/path/to/python   # must import numpy+scipy
```

| Package | Role |
| ------- | ---- |
| `numpy`, `scipy`, `matplotlib` | Numerics, IR plots, 3D PNG snapshots |
| `plotly` | PDN HTML viewers |
| `pytest` | Tests |

**Optional TSMC/TC1** (not in the source tarball; used by `tsmc1` and deck-backed TSMC tests):

```text
Benchmarks/TSMC/TC1/myALL_nportModel_without_package.sp
Benchmarks/TSMC/TC1/myALL_nportModel.sp.subckt
Benchmarks/TSMC/TC1/myALL_nportModel.sp.isrc
```

**Optional IBM PG** (not in the source tarball; only for `ibmpg*` / `TC2`…`TC6`):

```text
Benchmarks/IBM/TC2/ibmpg2.spice
...
Benchmarks/IBM/TC6/ibmpg6.spice
```

The fixture is a **tiny synthetic 2×2** n-port deck so the hierarchical parser +
virtual Pixel-R path runs without any external benchmark.

### SPICE backend: ngspice or Spectre

Optional decks (`spec_flow` stages 06–07; `my_flow --spice`) use either
**ngspice** or **Cadence Spectre 23.1+**. Default is ngspice.

| Choice                | Requirements                           | How to select                                                  |
| --------------------- | -------------------------------------- | -------------------------------------------------------------- |
| `ngspice` (default) | `ngspice` on `PATH`                | nothing, or `export SPICE_SIMULATOR=ngspice`                  |
| `spectre`           | Spectre on `PATH`, or `SPECTRE_BIN` | `--simulator spectre`, or `export SPICE_SIMULATOR=spectre` |

CLI (requires `--spice` to emit/solve decks in `my_flow`):

```bash
tclsh my_flow/run_flow.tcl ibmpg2 --spice                      # ngspice (default)
tclsh my_flow/run_flow.tcl ibmpg2 --spice --simulator spectre  # Cadence Spectre
```

Optional Spectre env vars (same effect as `--simulator spectre`):

```bash
export SPICE_SIMULATOR=spectre
export SPECTRE_BIN=/path/to/spectre   # if not on PATH
export SPECTRE_MT=64                  # multithread for spectre -mt=
```

---

## Shared concepts

### Case names

All three `run_flow.tcl` drivers accept `ibmpgN` / `TCN` / `N` / `tsmc1` /
`tsmc/tc1`, or an explicit `.sp`/`.spice` path (absolute or repo-relative).

- `ibmpgN` → `Benchmarks/IBM/TCN/ibmpgN.spice`
- `tsmc1` → `Benchmarks/TSMC/TC1/myALL_nportModel_without_package.sp`

Python used by Tcl: `MY_FLOW_PYTHON` if set, else `python` (if scipy imports).

### VDD connected components

Raw IBM `V` / `I` instances are not each a port. Pads are real overlapping C4s;
currents are lumped per grid cell. The flow discovers **VDD connected components**:

- **Seeds:** nodes with injection \(< 0\) and nonzero `V` terminals
- **Expand:** resistors + cross-layer `V=0` vias
- **Split:** by **metal-layer set** (interleaved rails stay separate)
- **Merge:** galvanic islands that share the same layer set (e.g. several
  disconnected `{1,3}` meshes on TC1 → one component)
- **Order:** topmost-layer first → `comp1`, `comp2`, …
- **Omit:** GND / VSS metals from ports and from the VDD R-mesh

Multi-rail stacks (ibmpg3 → 2 comps, ibmpg5 → 4) produce multiple `compN/`
trees; each is abstracted independently.

### Auto pitch

When pitch is `0` (default):

1. C4-lattice target = max(median unique-row \(\Delta x\), median unique-col \(\Delta y\))
   (`--grid-to-pad-ratio`, default `1.0` = match bump spacing; divide pitch by \(\sqrt{r}\))
2. If target \(\le\) `max_metal_pitch` → use `max_metal_pitch + ε` (≈ `+1` layout unit)
3. Else use the target

**Constraint (Pixel-R / spec + local):** auto `chip_pitch` is still clamped
**strictly greater** than the maximum VDD metal stripe pitch (per-layer: largest
consecutive preferred-direction track gap). `local_flow --pitch` may go below
that. In all cases a cell gets a voltage pad **iff a real C4 overlaps it**;
otherwise it is sink-only. Pass `--full-pads` to also attach the unique nearest
unused top-metal VDD node to every formerly padless cell and drive **all** pads
at one common voltage (representative of the real C4 voltages).

### Uniform sink current (`--uniform-current`)

Optional stage-02 override of lumped IBM currents. All three Tcl drivers accept
the same flag:

| Value | Behavior |
| ----- | -------- |
| off (`""` / `0` / `off` / …) | Real lumped injections |
| `conserve` (bare flag / `on` / `1`) | \(I_{\mathrm{each}} = \sum I / N\) (preserves total and sign) |
| positive number | Fixed per-sink draw in amps → stored as \(-\lvert I\rvert\) |
| negative number | Used as-is as nodal injection |

Recorded in `components.json` (`uniform_current_mode`, `i_each`, totals, …).

### Pixel-R star unit cell

A **single-layer grid of identical stars** (not a sink-to-sink mesh):

- **Center** = sink / probe + `Isink`; **Top** = pad / `vsrc`
- `Rup = Rdown = Rz` (Top ↔ Center ↔ internal Bottom)
- Half-arms `Rx` / `Ry` from Center to shared edge midpoints (boundary cells keep
  dangling half-arms). Neighbor path through a shared edge is \(2 R_x\) or \(2 R_y\)
- Port admittance \(G_S\) = Schur complement of edge + bottom internals
- Pad→center offset is treated as a **short**. Real-XY L-bend attach is
  `--via-stub` on `my_flow`.

**IR definition:** \(\mathrm{IR}_k = \mathrm{mean}(V_{\mathrm{pads}}) - V_{\mathrm{sink},k}\).
Reported pairs: original vs model, original vs reduced, reduced vs model.

---

## Quick start (ibmpg2 from scratch)

Requires `Benchmarks/IBM/` (not in the source tarball). For a self-contained run,
use [TSMC/TC1](#run-tsmctc1) instead.

All commands below are run from the **repo root**.

### 1. my_flow (parse → Kron → tri-stagger + tri-square)

Easiest driver (default `ibm_case=ibmpg2`; override via CLI or edit the Tcl):

```tcl
# my_flow/run_flow.tcl
set ibm_case          "ibmpg2"
set pitch_bot         0          ;# 0 = auto (C4 lattice, > max metal)
set coarsen_k         1
set grid_to_pad_ratio 1.0        ;# 1.0 = match C4 spacing; 4.0 denser (÷2)
set seed              0
set tri_square_k      2
set perturb_layer     ""         ;# "" | metal int | all
set perturb_amp       0.2
set via_stub          "rxry"     ;# rxry (L-bend) | zero (snap to nearest grid)
```

```bash
export MY_FLOW_PYTHON="${MY_FLOW_PYTHON:-python3}"   # must import numpy+scipy
tclsh my_flow/run_flow.tcl            # ibmpg2
tclsh my_flow/run_flow.tcl ibmpg3     # or TC3 / 3
tclsh my_flow/run_flow.tcl ibmpg2 --spice
tclsh my_flow/run_flow.tcl ibmpg2 --spice --simulator spectre
tclsh my_flow/run_flow.tcl ibmpg2 --via-stub zero
tclsh my_flow/run_flow.tcl ibmpg2 --uniform-current
tclsh my_flow/run_flow.tcl ibmpg2 --perturb-layer all --perturb-amp 0.2 --seed 0
```

| Positional / flag | Meaning |
| ----------------- | ------- |
| `<case>` | `ibmpgN` / `TCN` / `N` / `tsmc1` / `.sp` path |
| `--spice` | Also emit/solve tri-stagger SPICE decks |
| `--simulator ngspice\|spectre` | Backend when `--spice` is on |
| `--seed <int>` | IR-fit + perturbation RNG |
| `--uniform-current [MODE]` | conserve (bare) or amps |
| `--perturb-layer <L\|all>` | Grid-cell same-layer R perturbation |
| `--perturb-amp <A>` | default `0.2` (±20%) |
| `--via-stub rxry\|zero` | L-bend stubs vs snap via onto nearest node |

That reads `Benchmarks/IBM/TCn/ibmpgn.spice` and writes:

```text
my_flow/outputs/ibmpg2_auto_k1/
  net.npz, ports.json, components.json
  comp1/   # (and other VDD comps if present)
    Gprime.npy
    tri_stagger_model.json, tri_stagger_r.json
    tri_square_model.json, tri_square_r.json
    ir_*_tri_stagger.png, ir_*_tri_square.png
```

`--via-stub zero` appends `_nostub`. Uniform-current / perturb flags add `_uI…` /
`_pL…` tags. `pitch_bot=0` means **same auto pitch as spec_flow** (C4 lattice,
clamped `>` max metal). `--grid-to-pad-ratio 4` divides that lattice pitch by 2
before the max-metal clamp. Tri-stagger fits **`Rx, Ry, Rz`**; tri-square fits
**6R** `(Rx_u,Ry_u,Rx_l,Ry_l,Rz_pad,Rz_ul)`. Pad-only / zero-current components
are skipped by model runners (`--strict` fails instead).

Canonical attach:

```text
pad_k  -- Rz -- landing(pad XY)  -- stub(∝ Rx/Ry) -- nearest mesh node
sink_j -- Rz -- landing(sink XY) -- stub(∝ Rx/Ry) -- nearest mesh node
```

Equivalent stage-by-stage (same result):

```bash
export MY_FLOW_PYTHON="${MY_FLOW_PYTHON:-python3}"
OUT=my_flow/outputs/ibmpg2_auto_k1
COMP="$OUT/comp1"

"$MY_FLOW_PYTHON" my_flow/src/stage01_parse.py Benchmarks/IBM/TC2/ibmpg2.spice "$OUT"
"$MY_FLOW_PYTHON" my_flow/src/stage02_ports.py "$OUT" 0 1 --grid-to-pad-ratio 1.0
"$MY_FLOW_PYTHON" my_flow/src/stage03_assemble_kron.py "$COMP"
"$MY_FLOW_PYTHON" my_flow/src/run_tri_stagger.py "$COMP" 0 --spice
"$MY_FLOW_PYTHON" my_flow/src/run_tri_square.py "$COMP" 0 --k 2
```

### 2. Tri-stagger / tri-square IR plots

Under `$COMP` you get model artifacts plus IR plots, including:

| File                                                              | Meaning                                           |
| ----------------------------------------------------------------- | ------------------------------------------------- |
| `ir_scatter_reduced_vs_tri_stagger.png` / `…_tri_square.png` | Reduced \(G'\) vs model (IR % of Vdd, ±20% bands) |
| `ir_spatial_reduced_vs_tri_stagger.png` / `…_tri_square.png` | Side-by-side spatial IR % maps                    |
| `ir_error_heatmap_tri_stagger.png` / `…_tri_square.png`      | Sink \(\lvert\Delta IR\rvert\) heatmap             |
| `ir_scatter_original_vs_tri_stagger.png` (+ spatial)            | Original vs model when SPICE volts exist          |
| `ir_scatter_original_vs_reduced.png`                            | Original vs reduced (volts; if SPICE ran)         |

### 3. Pixel-R (`spec_flow`) + IR plots

**Required for the Pixel-R comparison arm.**

Same IBM case selector as `my_flow` (`ibm_case=ibmpg2` by default). Pitch defaults
to **auto**: C4 lattice pitch, clamped `>` max metal
(`chip_pitch > max_metal_pitch`; on ibmpg2 that still yields ~927 / 9×9).
Default fit is eigenvalue LS (TSMC spec); pass `ir` for mixed-BC IR matching.
After the fit, stages 06–07 solve three SPICE decks (original mesh, Kron \(G'\),
Pixel-R).

```tcl
# spec_flow/run_flow.tcl  (`chip_pitch` is Tcl-only; no --pitch CLI)
set ibm_case         "ibmpg2"
set chip_pitch       0          ;# 0 = auto legal pitch
set seed             0
set fit_method       "eigen"    ;# eigen | ir
set simulator        "ngspice"
set uniform_current  ""
set full_pads        0
set perturb_layer    ""
set perturb_amp      0.2
```

```bash
export MY_FLOW_PYTHON="${MY_FLOW_PYTHON:-python3}"   # honoured by spec_flow/run_flow.tcl
tclsh spec_flow/run_flow.tcl            # ibmpg2, auto pitch (default fit)
tclsh spec_flow/run_flow.tcl ibmpg3     # other case
tclsh spec_flow/run_flow.tcl ibmpg2 ir  # IR fit (recommended for IR PNGs)
tclsh spec_flow/run_flow.tcl ibmpg2 --simulator spectre
tclsh spec_flow/run_flow.tcl ibmpg2 ir --simulator spectre
tclsh spec_flow/run_flow.tcl ibmpg2 --uniform-current
tclsh spec_flow/run_flow.tcl ibmpg2 --full-pads
tclsh spec_flow/run_flow.tcl ibmpg2 --perturb-layer all --perturb-amp 0.2 --seed 0
```

| Positional / flag | Meaning |
| ----------------- | ------- |
| `<case>` | `ibmpgN` / `TCN` / `N` / `tsmc1` / `tsmc/tc1` / `.sp` path |
| `<fit_method>` | `eigen` (default) or `ir` |
| `--simulator ngspice\|spectre` | SPICE backend for emit + solve |
| `--seed <int>` | IR stimuli + layer-R RNG |
| `--uniform-current [MODE]` | conserve (bare) or amps |
| `--full-pads` | voltage pad on every cell (C4 + nearest top-metal fill) |
| `--perturb-layer <L\|all>` | grid-cell same-layer R perturbation |
| `--perturb-amp <A>` | default `0.2` (±20%) |

**OUT naming** (`spec_flow/outputs/`):

```text
<spice_stem>_n<pitch_tag><u_tag><fp_tag><p_tag>[_ir]
```

| Fragment | When | Example |
| -------- | ---- | ------- |
| `n<pitch_tag>` | always | `nauto` if `chip_pitch≤0`, else `n927.0` |
| `<u_tag>` | uniform current on | `_uI` (conserve) or `_uI0.001` (fixed) |
| `<fp_tag>` | `--full-pads` | `_fullPads` |
| `<p_tag>` | grid-cell R perturb on | `_pL5a0.2s0` / `_pLalla0.2s0` |
| `_ir` | `fit_method=ir` | appended last |

That writes under `spec_flow/outputs/ibmpg2_nauto/` (default) or
`spec_flow/outputs/ibmpg2_nauto_ir/` when `fit_method=ir`:

| File                                                           | Meaning                                             |
| -------------------------------------------------------------- | --------------------------------------------------- |
| `ir_scatter_reduced_vs_pixel_r.png`                          | Reduced \(G'\) vs Pixel-R (IR % of Vdd, ±20% bands) |
| `ir_spatial_reduced_vs_pixel_r.png`                          | Side-by-side spatial IR % maps                      |
| `ir_scatter_original_vs_pixel_r.png` / `…_vs_reduced.png` | Volt scatters when SPICE ran                        |
| `ir_error_heatmap.png`                                       | Sink \(\lvert\Delta IR\rvert\) Pixel-R vs original   |

**IBM ports** (`port_mode=c4_overlap_grid`): for every chip-pitch cell `(ix, iy)`,
a **voltage pad** is the real C4 that overlaps the cell (driven at the IBM C4
voltage, e.g. 1.8 on ibmpg2; sink-only if no C4), and a **current sink** is the
unique nearest node on the bottommost layer to the cell center (full `nx×ny`
grid; empty-current cells still get a sink). Cross-layer `V=0` vias stay as
near-ideal conductances (nodes **not** unioned). Same-layer `V=0` merges nodes;
ground `V n 0 0` merges to ground.

**Grid-cell R perturbation:** after chip pitch is known, each port cell draws
\(U\sim\mathrm{Uniform}[-A,A]\). Every same-layer R whose midpoint falls in
that cell is multiplied by \(1+U\). `--perturb-layer L` limits to metal `L`;
`--perturb-layer all` uses every metal in the current VDD (or VSS) rail stack.
Cross-layer vias are **not** perturbed. Not supported on TSMC virtual lattices.

Bash driver (thinner than Tcl: numeric pitch required, ngspice only, no uniform-
current / perturb flags):

```bash
./spec_flow/run_bench.sh Benchmarks/IBM/TC2/ibmpg2.spice 927.0 0 eigen
./spec_flow/run_bench.sh Benchmarks/IBM/TC2/ibmpg2.spice 927.0 0 ir
```

Tee log → `spec_flow/logs/<stem>_n<PITCH>[_ir].log`. Override interpreter with
`PY=/path/to/python`.

Stage-by-stage (same result as the Tcl driver):

```bash
OUT=spec_flow/outputs/ibmpg2_nauto
COMP="$OUT/comp1"
"$MY_FLOW_PYTHON" spec_flow/src/stage01_parse.py Benchmarks/IBM/TC2/ibmpg2.spice "$OUT"
"$MY_FLOW_PYTHON" spec_flow/src/stage02_ports.py "$OUT"          # 0 / omitted = auto
"$MY_FLOW_PYTHON" spec_flow/src/stage03_assemble_kron.py "$COMP"
"$MY_FLOW_PYTHON" spec_flow/src/stage04_pixel_model.py "$COMP"
"$MY_FLOW_PYTHON" spec_flow/src/stage05_fit_eigen.py "$COMP"
# or: "$MY_FLOW_PYTHON" spec_flow/src/stage05_fit_ir.py "$COMP" 0
"$MY_FLOW_PYTHON" spec_flow/src/stage06_emit_spice.py "$COMP"
"$MY_FLOW_PYTHON" spec_flow/src/stage07_spice_solve.py "$COMP"
"$MY_FLOW_PYTHON" spec_flow/src/stage08_correlate_ir.py "$COMP" 0
```

Stage 03 also writes Kron-branch grid \(R\) maps and unreduced-mesh driving-point
maps. Re-run without Kron:

```bash
"$MY_FLOW_PYTHON" spec_flow/src/grid_reff.py "$COMP"   # stencil + G' driving-point Z_kk
"$MY_FLOW_PYTHON" spec_flow/src/pg_reff.py "$COMP"     # unreduced mesh, real C4s grounded
```

`grid_reff.py` / `pg_reff.py` also work on `local_flow` `compN/` directories.

### 4. Localized Pixel-R (`local_flow`)

Same C4-overlap ports and Kron \(G'\) as spec_flow, but \((R_x,R_y,R_z)\) may
vary by pad-lattice **region** (`K×K` blocks). Shared EW/NS edges use series
half-arms \(1/(R^{(a)}+R^{(b)})\). Pad–sink uses per-cell \(R_z\). No SPICE
emit/solve, no `--perturb-*`, no `--simulator`. Default fit is Feng IR
(`--fit ir`).

```text
spec 01–03  →  partition (K×K pads)  →  uniform IR init  →  Feng block τ
            →  G' vs localized G_S metrics / plots
```

| `--block K` | Meaning |
| ----------- | ------- |
| `0` | One global region (uniform sanity; should ≈ spec_flow IR fit) |
| `1` | Per-cell (finest in-class) |
| `≥2` | \(K\times K\) pad blocks (default `2`) |

```bash
tclsh local_flow/run_flow.tcl                 # ibmpg2, --block 2
tclsh local_flow/run_flow.tcl ibmpg4
tclsh local_flow/run_flow.tcl ibmpg2 --block 0
tclsh local_flow/run_flow.tcl ibmpg2 --block 1
tclsh local_flow/run_flow.tcl ibmpg2 --uniform-current
tclsh local_flow/run_flow.tcl ibmpg2 --block 1 --uniform-current --fit eigen
tclsh local_flow/run_flow.tcl ibmpg2 --block 0 --uniform-current --pitch 463
```

| Positional / flag | Meaning |
| ----------------- | ------- |
| `<case>` | `ibmpgN` / `TCN` / `N` / `tsmc1` / `.sp` path |
| `--block <K>` | Partition size |
| `--seed <int>` | IR-fit RNG (default `0`) |
| `--uniform-current [MODE]` | same as spec_flow stage 02 |
| `--full-pads` | voltage pad on every cell |
| `--fit eigen\|ir` | `ir` = Feng (default); `eigen` = spectral LS (`_eigen` OUT) |
| `--pitch` / `--chip-pitch <n>` | Grid cell side (`0` = auto) |

Output: `local_flow/outputs/<stem>_n<auto|pitch>[_uI|_uI<amps>][_fullPads]_b<K>[_eigen]/`

| Artifact | Meaning |
| -------- | ------- |
| `localized_model.json` | Star topology + partition |
| `localized_r.json` | Per-region / per-cell R + uniform baseline |
| `metrics.json` | `e_ir_reduced_vs_localized` vs `e_ir_reduced_vs_uniform` |
| `ir_scatter_reduced_vs_localized.png` | IR % scatter |
| `ir_spatial_reduced_vs_localized.png` | Side-by-side spatial IR |
| `localized_r_maps.png` | Spatial \(R_x,R_y,R_z\) |
| `pad_occupancy.png` | Pad vs sink-only cells |

Compare to `spec_flow/outputs/ibmpg*_nauto_ir/comp1/metrics.json`
(`e_ir_reduced_vs_pixel_r`). More detail: [`local_flow/README.md`](local_flow/README.md).

### 5. Voltage-source / current-sink spatial maps

After parse (`net.npz` exists), plot raw SPICE V/I XY to check unevenness:

```bash
export MY_FLOW_PYTHON="${MY_FLOW_PYTHON:-python3}"

# from a spec_flow OUT/ (recommended)
"$MY_FLOW_PYTHON" spec_flow/src/plot_source.py spec_flow/outputs/ibmpg2_nauto_ir

# or from a my_flow OUT/ (same net.npz)
"$MY_FLOW_PYTHON" spec_flow/src/plot_source.py my_flow/outputs/ibmpg2_auto_k1
```

Writes next to `net.npz`:

| File                            | Meaning                                      |
| ------------------------------- | -------------------------------------------- |
| `voltage_sources_spatial.png` | V-source scatter (by layer) + density        |
| `current_sinks_spatial.png`   | I-sink scatter (by \|I\|) + injection density |

### 6. Viewers

Interactive Plotly viewers. HTML is written without auto-open unless
`--browser`. 3D also writes a matplotlib PNG beside the HTML (no kaleido);
`--no-png` skips it. Multi-rail designs expose **VDD1…VDDN** (topmost-first,
same order as `compN/`) plus **VSS**.

```bash
export MY_FLOW_PYTHON="${MY_FLOW_PYTHON:-python3}"

# 2D metal geometry (raw IBM .spice or a flow OUT/)
"$MY_FLOW_PYTHON" spec_flow/src/visualize_pdn.py Benchmarks/IBM/TC1/ibmpg1.spice
"$MY_FLOW_PYTHON" spec_flow/src/visualize_pdn.py Benchmarks/IBM/TC5/ibmpg5.spice --net vdd2

# Pixel-R / localized 3D — writes .html and a matching .png snapshot
"$MY_FLOW_PYTHON" spec_flow/src/visualize_pdn_3d.py spec_flow/outputs/ibmpg2_nauto_ir \
  --comp 1 --mode both \
  --html spec_flow/outputs/ibmpg2_nauto_ir/pdn_view_3d_pixel_r.html
"$MY_FLOW_PYTHON" spec_flow/src/visualize_pdn_3d.py local_flow/outputs/ibmpg2_nauto_uI_b1 \
  --comp 1 --mode both \
  --html local_flow/outputs/ibmpg2_nauto_uI_b1/pdn_view_3d_localized.html

# Tri-stagger / tri-square 3D
"$MY_FLOW_PYTHON" my_flow/src/visualize_pdn_3d.py my_flow/outputs/ibmpg2_auto_k1 \
  --model tri_stagger --comp 1 \
  --html my_flow/outputs/ibmpg2_auto_k1/pdn_view_3d_tri_stagger.html
"$MY_FLOW_PYTHON" my_flow/src/visualize_pdn_3d.py my_flow/outputs/ibmpg2_auto_k1 \
  --model tri_square --comp 1 \
  --html my_flow/outputs/ibmpg2_auto_k1/pdn_view_3d_tri_square.html
```

| Common flag | Default | Notes |
| ----------- | ------- | ----- |
| `--net` / `--comp K` | `vdd1` | `--comp` sets `vddK` (`vdd` ≡ `vdd1`) |
| `--layer` | `all` | or integer metal index |
| `--mode` | `both` | `both` \| `original` \| `pixel` (my_flow also `dual`) |
| `--html` | `OUT/pdn_view*.html` | |
| `--browser` | off | open after write |
| `--no-vias` | vias shown | |
| `--png` / `--no-png` | PNG beside HTML | matplotlib snapshot |

### 7. Eigen spectrum of an IR fit

IR fitting does not store eigenvalues. After an IR run, back-project the fitted
\(G_S\) into the \(G'\) eigenbasis and optionally compare against eigenvalue LS:

\[
G' = Q\,\mathrm{diag}(\lambda_M)\,Q^\top,\quad
\lambda_S = \mathrm{diag}(Q^\top G_S Q).
\]

**Pixel-R (`spec_flow`)** — compare a paired eigen OUT vs IR OUT, or refit eigen
on the IR OUT alone:

```bash
export MY_FLOW_PYTHON="${MY_FLOW_PYTHON:-python3}"

# Paired dirs (eigen default OUT + IR OUT)
"$MY_FLOW_PYTHON" spec_flow/src/compare_eigen_ir.py \
  spec_flow/outputs/ibmpg2_nauto/comp1 \
  spec_flow/outputs/ibmpg2_nauto_ir/comp1

# IR OUT only: re-run Pixel-R eigen LS for the overlay
"$MY_FLOW_PYTHON" spec_flow/src/compare_eigen_ir.py --refit-eigen \
  spec_flow/outputs/ibmpg2_nauto_ir/comp1
```

**Tri-stagger / tri-square (`my_flow`)** — evaluate IR `*_spectra.npz`; add
`--refit-eigen` to run eigenvalue LS on the same model:

```bash
"$MY_FLOW_PYTHON" my_flow/src/compare_ir_spectra.py \
  my_flow/outputs/ibmpg2_auto_k1/comp1 --model tri_stagger
"$MY_FLOW_PYTHON" my_flow/src/compare_ir_spectra.py \
  my_flow/outputs/ibmpg2_auto_k1/comp1 --model tri_square --refit-eigen
```

Writes under the IR (or `--out`) comp dir:

| File                                                          | Meaning                                                           |
| ------------------------------------------------------------- | ----------------------------------------------------------------- |
| `eigen_ir_compare.json` / `eigen_ir_compare_<model>.json` | Relative spectral error, Pearson, R params                        |
| `*_spectra.npz`                                             | `lam_M`, `lam_S_ir`, optional `lam_S_eigen`                 |
| `*_scatter.png`                                             | \(\lambda^M\) vs projected \(\lambda^S\) (IR vs eigen)            |
| `*_sorted.png`                                              | Sorted eigenvalues vs mode index                                  |
| `*_relerr.png`                                              | Per-mode \(\lvert\lambda^S-\lambda^M\rvert/\lvert\lambda^M\rvert\) |

Expect eigen-fit \(G_S\) to win on spectral error and IR-fit \(G_S\) to win on
mixed-BC IR — the two objectives are not equivalent. See
[`IR_FIT_WEIGHTED_SPECTRUM.md`](IR_FIT_WEIGHTED_SPECTRUM.md).

---

## TSMC n-port (hierarchical)

`spec_flow/src/spice_parser.py` flattens Spectre/SPICE n-port decks:

- `include` / `.include` (relative to the including file)
- `.SUBCKT` … `.ENDS` and `X…` instances (e.g. `Xdie1 … myALL_nportModel`)
- `+` continuation lines, tile/region node names, `*.isrc` current sources
- `.PRINT DC()` and other analysis cards are ignored

### Virtual Pixel-R region lattice (real TC1)

Tile/region `x_*/y_*` tokens are **grid indices**, not microns. For a filled
region-`VDD_PORT` lattice (≥4 cells), `spec_flow` maps indices to layout units:

| Index corner  | Micron       |
| ------------- | ------------ |
| `x_1_y_1`   | (30, 30)     |
| `x_43_y_1`  | (2550, 30)   |
| `x_43_y_33` | (2550, 1950) |
| `x_1_y_33`  | (30, 1950)   |

→ origin (30, 30), **pitch = 60**, default **43×33**. Each `*_VDD_PORT` is a
**current sink**. Top pads are **real bump-mapped tiles** from
`* BUMP_VDD_{bx}_{by} <tile> VDD <x> <y>` in the subckt (0-based bump indices
align to 1-based region indices by `+1`). Occupied cells near-short a unique
`vpad_x_*_y_*` onto that tile; empty bump cells keep an isolated open-circuit
pad (zero conductance). Package pin shorts onto `VDD_in` are ignored so the
tile mesh stays electrically distinct. Stage 02 writes `tsmc_virtual_*.png`
on the region lattice (`port_mode=tsmc_virtual_pixel_r`).

Geometry / detection: [`spec_flow/src/tsmc_region_grid.py`](spec_flow/src/tsmc_region_grid.py).
A filled region-`VDD_PORT` lattice (≥4 cells) takes the **virtual Pixel-R** path.

### Install the Spectre deck

Place the three filenames under `Benchmarks/TSMC/TC1/` (alias `tsmc1`), or pass
an explicit `.sp` path to the driver / `stage01_parse.py`.

```text
Benchmarks/TSMC/TC1/myALL_nportModel_without_package.sp
Benchmarks/TSMC/TC1/myALL_nportModel.sp.subckt
Benchmarks/TSMC/TC1/myALL_nportModel.sp.isrc
```

Expected roles:

| File                                    | Contents                                                                                                  |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `myALL_nportModel_without_package.sp` | Top: `VVDD`/`VVSS`, `.INCLUDE` subckt, `Xdie1` (package shorts + region ports), `.PRINT`/`.DC` |
| `myALL_nportModel.sp.subckt`          | `.SUBCKT` pins, `* BUMP_` XY, `R_*` mesh, `.INCLUDE` `.isrc`, `R_VSS_CONN_*`                  |
| `myALL_nportModel.sp.isrc`            | `I_*` from each `*_VDD_PORT` to its `*_VSS_PORT`                                                    |

`Xdie1` pin count must match the `.SUBCKT myALL_nportModel` port list.

### Run TSMC/TC1

Requires `Benchmarks/TSMC/TC1/` (see above):

```bash
export MY_FLOW_PYTHON="${MY_FLOW_PYTHON:-python3}"   # must import numpy+scipy
pip install -r my_flow/requirements.txt              # once

# Pixel-R on Benchmarks/TSMC/TC1 (eigen-fit by default, same as IBM)
tclsh spec_flow/run_flow.tcl tsmc1
tclsh spec_flow/run_flow.tcl tsmc1 eigen  # explicit (same as default)
tclsh spec_flow/run_flow.tcl tsmc1 ir     # mixed-BC IR fit instead

# Or pass the path explicitly (parse + virtual ports + lattice PNGs)
"$MY_FLOW_PYTHON" spec_flow/src/stage01_parse.py \
  Benchmarks/TSMC/TC1/myALL_nportModel_without_package.sp \
  spec_flow/outputs/tsmc_tc1_parse
"$MY_FLOW_PYTHON" spec_flow/src/stage02_ports.py \
  spec_flow/outputs/tsmc_tc1_parse
```

Typical OUT (stem = SPICE basename):

```text
spec_flow/outputs/myALL_nportModel_without_package_nauto/      # tsmc1 eigen (default)
spec_flow/outputs/myALL_nportModel_without_package_nauto_ir/   # tsmc1 ir
```

VDD tiles sit on synthetic layer `2`; region sinks stay on layer `1`. When the
region-`VDD_PORT` lattice is filled (≥4 cells), `spec_flow` takes the bump-tile
Pixel-R path (sinks on `*_VDD_PORT`, pads → bump-mapped tiles). Stage 02 writes
`comp1/tsmc_virtual_sinks_current.png` and `tsmc_virtual_pads.png`.

### Plot TSMC virtual square lattice

Stage 02 already writes the region-lattice PNGs into `comp1/`. Re-plot (or plot
after a partial run) with [`spec_flow/src/plot_tsmc_virtual_grid.py`](spec_flow/src/plot_tsmc_virtual_grid.py)
(square index→micron lattice: pads / sinks / currents):

```bash
export MY_FLOW_PYTHON="${MY_FLOW_PYTHON:-python3}"
TSMC_COMP=spec_flow/outputs/myALL_nportModel_without_package_nauto/comp1

# After tclsh spec_flow/run_flow.tcl tsmc1   (ir → …_nauto_ir/comp1)
"$MY_FLOW_PYTHON" spec_flow/src/plot_tsmc_virtual_grid.py "$TSMC_COMP"

# Manual parse OUT
"$MY_FLOW_PYTHON" spec_flow/src/plot_tsmc_virtual_grid.py \
  spec_flow/outputs/tsmc_tc1_parse/comp1
```

| File                               | Meaning                                     |
| ---------------------------------- | ------------------------------------------- |
| `tsmc_virtual_sinks_current.png` | Region `*_VDD_PORT` sinks colored by \|I\| |
| `tsmc_virtual_pads.png`          | Bump-lattice pads + sink markers            |

### Eigen compare on TSMC/TC1

Same tools as §7; point at the TSMC OUT:

```bash
# Compare eigen OUT vs IR OUT (run both fits first), or refit on an IR OUT:
"$MY_FLOW_PYTHON" spec_flow/src/compare_eigen_ir.py \
  spec_flow/outputs/myALL_nportModel_without_package_nauto/comp1 \
  spec_flow/outputs/myALL_nportModel_without_package_nauto_ir/comp1
"$MY_FLOW_PYTHON" spec_flow/src/compare_eigen_ir.py --refit-eigen \
  spec_flow/outputs/myALL_nportModel_without_package_nauto_ir/comp1
```

## Batch: IBM PG2–6

Requires `Benchmarks/IBM/` (not in the source tarball).

```bash
bash my_flow/run_ibmpg2_6.sh
```

Or use the tri-stagger batch helper (PG3–6; skips zero-current comps):

```bash
"$MY_FLOW_PYTHON" my_flow/src/run_tri_stagger_ibmpg3_6.py
```

---

## Other runners

VoltSpot (`run_voltspot.py`) is a separate diagnostic, not one of the paper
models. Extract-only by default; `--polish` fits two global sheet scales.

```bash
"$MY_FLOW_PYTHON" my_flow/src/run_voltspot.py
"$MY_FLOW_PYTHON" my_flow/src/run_voltspot.py --spice Benchmarks/IBM/TC2/ibmpg2.spice \
  --out my_flow/outputs/ibmpg2_voltspot --grid-to-pad-ratio 4.0
"$MY_FLOW_PYTHON" my_flow/src/run_voltspot.py --comp "$COMP" --polish --strict
```

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--spice` | IBM TC2 deck | Input SPICE |
| `--out` | VoltSpot OUT root | |
| `--comp` | off | Reuse an existing dual-flow `compN/` (skips parse/ports) |
| `--grid-to-pad-ratio` | `4.0` | Ignored with `--comp` |
| `--polish` | off | Optional \(\alpha_x/\alpha_y\) polish |
| `--strict` | off | Fail on pad-only / zero-current comps |

---

## Tests

Deck-backed tests **skip** if the corresponding `Benchmarks/` files are missing.

```bash
"$MY_FLOW_PYTHON" -m pytest spec_flow/tests/test_tsmc_nport_parse.py \
  spec_flow/tests/test_tsmc_region_grid.py -q
"$MY_FLOW_PYTHON" -m pytest spec_flow/tests -q
"$MY_FLOW_PYTHON" -m pytest local_flow/tests -q
"$MY_FLOW_PYTHON" -m pytest my_flow/tests -q
```

| spec_flow module | Covers |
| ---------------- | ------ |
| `test_imaginary_ports.py` | Full-grid unique pads/sinks, metal pitch, star topology, original deck emit |
| `test_ibm_spec_ports.py` | IBM C4-overlap pads, IBM pad V, full sink grid, current lumping |
| `test_sparse_pads.py` | Sink-only cells; `n_pads ≤ n_sinks`; `--full-pads` fill |
| `test_tsmc_bump_ports.py` | Bump pads short to tile, not `VDD_in` |
| `test_vdd_components_ibm.py` | ibmpg3 / ibmpg5 multi-rail CC counts; ibmpg1 island merge |
| `test_uniform_current.py` | Conserve / fixed apply; OUT `_uI` tags |
| `test_perturb_layer.py` | Grid-cell same-layer R perturb; vias untouched |
| `test_tsmc_nport_parse.py` | Hierarchical include/subckt/X flatten |
| `test_tsmc_region_grid.py` | Lattice microns, virtual ports |
| `test_pdn_geometry.py` / `test_pdn_3d.py` | 2D / 3D Plotly figures |
| `test_tiny_mesh.py` | Synthetic Kron Schur + eigen/IR fit sanity |
| `test_grid_reff.py` / `test_pg_reff.py` | Kron-branch and unreduced \(R_{\mathrm{eff}}\) |
| `test_ibm_tc1_smoke.py` | End-to-end stages 01–08 (needs ngspice + deck) |

---

## Environment variables

| Variable | Default | Role |
| -------- | ------- | ---- |
| `SPICE_SIMULATOR` | `ngspice` | Emit/run default when CLI omits `--simulator` |
| `SPECTRE_BIN` | `which spectre` | Spectre executable |
| `SPECTRE_MT` | `64` | Spectre `+mt=` |
| `MY_FLOW_PYTHON` | unset | Python used by all three `run_flow.tcl` drivers |
| `PY` | (bench script default) | Python used by `spec_flow/run_bench.sh` |

---

## Performance notes

- Kron: one SuperLU (`splu`) of \(G_{II}\) + multi-RHS solve
- Eigen fit needs the **full** spectrum of dense \(G'\) (`eigh`) — prefer coarser
  `chip_pitch` (port count \(= n_{\mathrm{pads}}+n_{\mathrm{sinks}}\), with
  \(n_{\mathrm{sinks}}=n_x n_y\))
- IR fit skips `eigh` in the optimizer; typically cheaper for large grids
- `chip_pitch` must also leave enough unique bottom-layer nodes for the sink grid
- spec_flow stage 07 on the R-only multi-layer mesh often dominates wall time

---

## Scope / limitations

- spec_flow / local_flow / my_flow: full `nx×ny` sink grid always instantiates
  every cell (zero-current sinks allowed). Pads only on C4-overlapping cells.
- SRAM-PG out of scope (no layout coordinates in node names)
- `spec_flow/run_bench.sh` is thinner than `run_flow.tcl` (no uniform current,
  R-perturb, or Spectre flags)
- `local_flow` does not emit SPICE or apply `--perturb-*`
- Compare my_flow methodology to Pixel-R, not identical attach topology
  (Pixel-R shorts pad→center; my_flow uses real-XY stubs)

---

## Stage maps

### spec_flow

| Stage | Script | Writes |
| ----- | ------ | ------ |
| 01 | `stage01_parse.py` | `net.npz` |
| 02 | `stage02_ports.py` | ports + `components.json` |
| 03 | `stage03_assemble_kron.py` | Kron \(G'\) |
| 04 | `stage04_pixel_model.py` | Pixel-R topology |
| 05 | `stage05_fit_eigen.py` / `stage05_fit_ir.py` | fitted \(R_x,R_y,R_z\) |
| 06–07 | `stage06_emit_spice.py` / `stage07_spice_solve.py` | three decks + `.volt` |
| 08 | `stage08_correlate_ir.py` | IR metrics + plots |

### my_flow

| Stage | Script                       | Writes                             |
| ----- | ---------------------------- | ---------------------------------- |
| 01    | `stage01_parse.py`         | `net.npz`                        |
| 02    | `stage02_ports.py`         | ports + `components.json`         |
| 03    | `stage03_assemble_kron.py` | Kron \(G'\)                         |
| —    | `run_tri_stagger.py`       | tri-stagger model/fit + IR plots   |
| —    | `run_tri_square.py`        | tri-square 6R model/fit + IR plots |

Per-flow extras: [`my_flow/README.md`](my_flow/README.md),
[`spec_flow/README.md`](spec_flow/README.md),
[`local_flow/README.md`](local_flow/README.md).

IBM spec-clause audit: [`SPEC_COMPLIANCE_IBM.md`](SPEC_COMPLIANCE_IBM.md).
Research notes: [`RESEARCH_DIRECTIONS.md`](RESEARCH_DIRECTIONS.md),
[`IR_FIT_WEIGHTED_SPECTRUM.md`](IR_FIT_WEIGHTED_SPECTRUM.md).

Flow / topology comparison (Pixel-R · Tri-stagger · Tri-square):
[`my_flow/paper/PDN_Abstraction_Flow_Comparison.pdf`](my_flow/paper/PDN_Abstraction_Flow_Comparison.pdf).
