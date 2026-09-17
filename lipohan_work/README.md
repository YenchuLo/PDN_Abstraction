# Pixel-R cost-function experiments (lipohan)

Does **not** replace `spec_flow`. It reuses stages 01–04 (parse → ports → Kron →
star topology) and only swaps the **fitting objective** over global
`(Rx, Ry, Rz)`.

For the complete derivation, units, mathematical meaning, physical
interpretation, limitations, and relationships among all objectives, see
[`OBJECTIVE_FUNCTIONS.md`](OBJECTIVE_FUNCTIONS.md).

## Priority costs (default `--costs priority`)

| Name | Cost | Residual used by the optimiser |
| ---- | ---- | ------------------------------ |
| `eigen` | \(J_\lambda=\sum_i(\lambda_i^M-\lambda_i^S)^2\) | TSMC spec; \(\lambda^S=\mathrm{diag}(Q^\top G_S Q)\) |
| `frobenius` | \(J_F=\|G_S-G'_M\|_F^2\) | \(\mathrm{vec}(G_S-G'_M)\) |
| `rayleigh` | \(J_{RQ}\) | Rayleigh on mixed test directions (eigen subsample + random + current lifts). **Not** identical to `eigen`; with every reference eigenvector it recovers \(J_\lambda\) up to the tiny numerical grounding shift. |
| `voltage` | \(J_V=\sum_k\|v_S^{(k)}-v_M^{(k)}\|_2^2\) | Mixed-BC IR drops (pads V-fixed, sinks I-driven). Same physics as `spec_flow` IR fit. |
| `minimax` | \(J_\infty=\max_{k,j}\|(v_S-v_M)/(|v_M|+\epsilon)\|\) | Nelder–Mead on max relative IR; warm-started from `voltage`. |

Also implemented (pass by name or `--costs all`):

`eigen_rel`, `frobenius_rel`, `weighted` (diagonal / neighbour / via weights),
`rayleigh_rel`, `voltage_rel` (the 20% IR relative error), `pnorm` (default \(p=8\)),
`reff` (port-to-port \(R_\mathrm{eff}\)), `spectral` (\(\|G_S-G'_M\|_2\)),
`hybrid` (\(\alpha J_\lambda+\beta J_{RQ}+\gamma J_V\) on normalised residuals).

Every fit is **evaluated on all metrics**, so you can see the trade-off even
when you only trained on one \(J\).

IR uses \(\mathrm{IR}_k=\mathrm{mean}(V_\mathrm{pads})-V_{\mathrm{sink},k}\).
`frac_within_20pct_*` is the fraction of sinks with
\(|\mathrm{IR}_S-\mathrm{IR}_M|/(|\mathrm{IR}_M|+\epsilon)\le 0.2\), matching the
±20% bands on the scatter plots.

## Run

From the **repo root**. Use a Python with numpy/scipy/matplotlib
(e.g. conda env `cfirstnet` or `train`).

```bash
export MY_FLOW_PYTHON="${MY_FLOW_PYTHON:-/home/lipopo/miniconda3/envs/cfirstnet/bin/python}"

# Closed-form 3×3 star (no IBM deck) — sanity that every cost recovers Rx,Ry,Rz
"$MY_FLOW_PYTHON" lipohan_work/src/run_cost_sweep.py --synthetic

# ibmpg2: frontend + five priority costs
"$MY_FLOW_PYTHON" lipohan_work/src/run_cost_sweep.py \
  --spice Benchmarks/IBM/TC2/ibmpg2.spice

# extra costs / hybrid weights
"$MY_FLOW_PYTHON" lipohan_work/src/run_cost_sweep.py \
  --comp lipohan_work/outputs/ibmpg2_nauto/comp1 \
  --costs eigen,frobenius,rayleigh,voltage,minimax,voltage_rel,pnorm,hybrid
```

Outputs land under `lipohan_work/outputs/`:

```text
ibmpg2_nauto/                 # stages 01–04 (shared with spec_flow layout)
  comp1/Gprime.npy …
  cost_sweep/
    comparison.json
    comparison.csv
    ir_scatter_by_cost.png
    bar_e_ir_original.png
    fit_eigen/  fit_frobenius/  …
```

Tests (no pytest required):

```bash
PYTHONPATH=lipohan_work/src "$MY_FLOW_PYTHON" lipohan_work/tests/test_costs.py
```

## Comprehensive synthetic study

`run_synthetic_study.py` tests all costs against exact-global, smooth/random
localized, long-range, and combined synthetic truths. It separates fitting
currents from 120 held-out currents, varies pad voltages, and includes current,
Rayleigh-basis, p-norm, hybrid-weight, seed, minimax multi-start, grid-size,
and conductance-noise ablations.

```bash
PYTHONPATH=lipohan_work/src "$MY_FLOW_PYTHON" \
  lipohan_work/src/run_synthetic_study.py \
  --out lipohan_work/outputs/synthetic_study_full \
  --max-nfev 250 --heldout-repeats 5
```

See [`RESULTS_SYNTHETIC.md`](RESULTS_SYNTHETIC.md) for the method, results,
limitations, and recommended objectives.

## ibmpg2 snapshot (seed 0, auto pitch 927, 159 ports)

| Train | Rx / Ry / Rz (Ω) | e_IR orig. | e_worst | J∞ | ≤20% orig. |
| ----- | ---------------- | ---------- | ------- | -- | ---------- |
| eigen | 7.83m / 17.3m / 0.331 | 18.4% | 18.4% | 874% | 100% |
| frobenius | 7.86m / 17.5m / 0.283 | **1.39%** | 2.50% | 39% | 100% |
| rayleigh | 7.79m / 17.2m / 0.352 | 26.1% | 26.1% | 1313% | 0% |
| voltage | 7.64m / 16.4m / 0.279 | **0.39%** | 3.17% | 200% | 100% |
| minimax | 7.79m / 19.0m / 0.305 | 9.33% | 9.38% | **24%** | 100% |

Identity 3×3 star recovers `(Rx,Ry,Rz)=(2,4,5)` for every priority cost. On ibmpg2 the three globals cannot match dense \(G'_M\); **which three you pick depends on J**. Voltage wins mean IR; minimax wins the relative tail; Frobenius already beats TSMC eigen on IR without seeing currents. Large \(J_\infty\) on eigen/voltage is a near-zero IR sink on a held-out stimulus, not an 800% die-wide miss.
