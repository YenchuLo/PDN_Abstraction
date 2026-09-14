# local_flow — localized Pixel-R (uniformity ablation)

Same C4-overlap ports and Kron \(G'\) as [`spec_flow`](../spec_flow/) (IBM pad
voltages from the deck; full `nx×ny` sink grid), but
Pixel-R \((R_x,R_y,R_z)\) may vary by pad-lattice **region**:

```text
spec 01–03  →  partition (K×K pads)  →  uniform IR init  →  Feng block τ
            →  G' vs localized G_S metrics / plots
```

Shared EW/NS edges use series half-arms \(1/(R^{(a)}+R^{(b)})\). Pad–sink uses
per-cell \(R_z\).

## `--block K`

| K | Meaning |
|---|--------|
| `0` | One global region (uniform sanity; should ≈ spec_flow IR fit) |
| `1` | Per-cell (finest in-class) |
| `≥2` | \(K\times K\) pad blocks (default `2`) |

## Run

```bash
export MY_FLOW_PYTHON=/path/to/python   # must import numpy+scipy
tclsh local_flow/run_flow.tcl                 # ibmpg2, --block 2
tclsh local_flow/run_flow.tcl ibmpg4
tclsh local_flow/run_flow.tcl ibmpg4 --block 1
tclsh local_flow/run_flow.tcl ibmpg2 --block 0

# Equal current on every grid sink (conserve total I)
tclsh local_flow/run_flow.tcl ibmpg2 --uniform-current
tclsh local_flow/run_flow.tcl ibmpg2 --uniform-current=1e-3

# Per-pixel R (--block 1) + uniform current
tclsh local_flow/run_flow.tcl ibmpg2 --block 1 --uniform-current
tclsh local_flow/run_flow.tcl ibmpg4 --block 1 --uniform-current
tclsh local_flow/run_flow.tcl ibmpg2 --block 1 --uniform-current --fit eigen
tclsh local_flow/run_flow.tcl ibmpg4 --block 1 --uniform-current --fit eigen

# Pads only on cells that overlap a real C4 (any pitch)
tclsh local_flow/run_flow.tcl ibmpg2 --block 0 --uniform-current --pitch 463
```

`--uniform-current` is the same stage-02 override as [`spec_flow`](../spec_flow/):
bare flag / `conserve` shares total lumped \(I\) equally; a number is a fixed
per-sink draw in amps. IR fit and plots then use that map (recorded in
`components.json`). `--fit eigen` uses spectral LS instead of Feng IR
(OUT suffix `_eigen`).

Output: `local_flow/outputs/<stem>_n<auto|pitch>[_uI|_uI<amps>]_b<K>[_eigen]/compN/`

| Artifact | Meaning |
|----------|---------|
| `localized_model.json` | Star topology + partition |
| `localized_r.json` | Per-region / per-cell R + uniform baseline |
| `localized_spectra.npz` | `Gprime`, `Gs`, `Gs_uniform`, cell R |
| `metrics.json` | `e_ir_reduced_vs_localized` vs `e_ir_reduced_vs_uniform` |
| `ir_scatter_reduced_vs_localized.png` | IR % scatter |
| `ir_spatial_reduced_vs_localized.png` | Side-by-side spatial IR |
| `ir_error_heatmap.png` | Sink \(\lvert\Delta\mathrm{IR}\rvert\) |
| `localized_r_maps.png` | Spatial \(R_x,R_y,R_z\) |
| `localized_reff.json` / `localized_reff_map.png` | Per-cell driving-point \(R_{\mathrm{eff}}=Z_{kk}\) (Rx, Ry, Rz together) |
| `reff_spatial_reduced_vs_localized.png` | \(G'\) vs localized \(Z_{kk}\) |
| `grid_reff_maps.png` | Kron-branch \(R_z, R_{\mathrm{EW}}, R_{\mathrm{NS}}\) (also `grid_reff.json` / `.npz`) |
| `grid_reff_dp.png` | Benchmark \(G'\) driving-point \(R_{\mathrm{eff}}\) |
| `pg_reff_map.png` | Unreduced mesh \(R_{\mathrm{eff}}\) (real C4 pads grounded; also `pg_reff.json` / `.npz`) |
| `pad_occupancy.png` | Cells with a real C4 pad vs sink-only |

Compare to `spec_flow/outputs/ibmpg*_nauto_ir/comp1/metrics.json`
(`e_ir_reduced_vs_pixel_r`).

## Diagnostic snapshot (seed=0, auto pitch)

| Case | K | regions | \(e_{\mathrm{IR}}\) localized | \(e_{\mathrm{IR}}\) uniform |
|------|---|--------|-------------------------------|----------------------------|
| ibmpg2 | 0 | 1 | 0.043 | 0.043 |
| ibmpg2 | 2 | 25 | **0.0063** | 0.043 |
| ibmpg2 | 1 | 81 | **~0** | 0.043 |
| ibmpg4 | 0 | 1 | 0.104 | 0.104 |
| ibmpg4 | 2 | 64 | 0.104 (Feng/polish no gain) | 0.104 |
| ibmpg4 | 1 | 225 | **0.028** | 0.104 |

Held-out stimuli (`random` / `localized` / `striped`) also drop with K=1 on both cases, so the gain is not only overfitting the original current map. Uniformity is a major remaining error source on these decks; per-cell Pixel-R largely closes it while staying inside the star model class.

## Tests

```bash
"$MY_FLOW_PYTHON" -m pytest local_flow/tests -q
```
