# IBM Spec Compliance Audit (ibmpg2-first)

Independent check of `spec_flow`, `local_flow`, and `my_flow` against
[`PDN_Abstraction_for_EMIR_Feasibility-TSMC_Spec.pdf`](PDN_Abstraction_for_EMIR_Feasibility-TSMC_Spec.pdf),
using **real IBM `V` / `I` devices** as port ground truth.

**Date:** 2026-09-01 (updated after A5 / A3 / docs fixes)  
**Vehicle:** `Benchmarks/IBM/TC2/ibmpg2.spice` (comp1)  
**Oracle / checker:** [`spec_flow/src/spec_oracle.py`](spec_flow/src/spec_oracle.py),
[`spec_flow/src/check_spec_ibm.py`](spec_flow/src/check_spec_ibm.py)  
**Audit OUTs:**

| Flow | OUT |
|------|-----|
| spec_flow | [`spec_flow/outputs/ibmpg2_nauto_spec_audit/comp1`](spec_flow/outputs/ibmpg2_nauto_spec_audit/comp1) |
| local_flow | [`local_flow/outputs/ibmpg2_nauto_spec_audit/comp1`](local_flow/outputs/ibmpg2_nauto_spec_audit/comp1) |
| my_flow | [`my_flow/outputs/ibmpg2_auto_k1_spec_audit/comp1`](my_flow/outputs/ibmpg2_auto_k1_spec_audit/comp1) |

Score JSON: [`spec_flow/outputs/_spec_audit_scores/`](spec_flow/outputs/_spec_audit_scores/).

---

## ibmpg2 oracle numbers (pitch = 927, auto)

| Metric | Value |
|--------|------:|
| Chip grid | \(9 \times 9\) = 81 cells |
| Real C4s (nonzero `V→gnd` on VDD) | 120 |
| Pads after per-cell collapse (nearest C4) | **78** |
| Extra C4s collapsed into shared cells | 42 |
| Sink-only cells (no C4 overlap) | 3 |
| IBM injections (`inj < 0`) on component | 18 419 |
| Nonzero-current cells at this pitch | **81** (all cells) |
| \(\sum I\) | \(-143.478\) A |
| IBM C4 voltage | **1.8 V** |

---

## Spec clauses × flow (ibmpg2 @ 927) — after fixes

Status legend: **pass** / **fail** / **intentional_deviation** / skip.

| Clause | Spec intent | spec_flow | local_flow | my_flow |
|--------|-------------|-----------|------------|---------|
| **A1** Pad occupancy | Pixel connects to VS **iff** C4 overlaps cell + same net | **pass** (78) | **pass** | **pass** |
| **A2** Current lumping | Lump IBM `I` instances per grid cell | **pass** | **pass** | **pass** |
| **A3** Full sink grid | One current sink per chip-grid cell | **pass** | **pass** | **pass** (incl. fine pitch 231.5) |
| **A4** Same net | Pads/sinks only from this VDD CC | **pass** | **pass** | **pass** |
| **A5** Pad voltage | Drive = IBM C4 voltage | **pass** (1.8) | **pass** (1.8) | **pass** (1.8) |
| **B** Kron \(G'\) shape | \(G' \in \mathbb{R}^{n_p\times n_p}\) | **pass** (159²) | **pass** | **pass** |
| **C** Pad–sink \(R_z\) / Pixel-R | Coupling only on occupied cells; identical stars | **pass** | **pass** coupling; **intentional_deviation** (localized \(R\)) | **intentional_deviation** (tri-stagger / tri-square) |
| **D** Eigen LS fit | \(\min\sum(\lambda^M-\mathrm{diag}(Q^\top G_S Q))^2\) | **pass** (eigen) | **intentional_deviation** (IR Feng) | **intentional_deviation** (IR RxRyRz) |

### Fixes applied (2026-09-01)

1. **A5:** [`build_ports`](spec_flow/src/ports.py) stamps each pad with its IBM C4 voltage; `PortSet.vdd` = common/median pad voltage (`DEFAULT_VDD` remains fallback / TSMC).
2. **A3:** [`my_flow` stage02](my_flow/src/stage02_ports.py) always uses `full_grid_sinks=True`; [`tri_stagger_ports_from_dual`](my_flow/src/ports_dual.py) no longer drops zero-`I` sinks.
3. **Docs / label:** `port_mode=c4_overlap_grid` (legacy `imaginary_grid` still loads); READMEs describe C4 overlap + IBM pad V.

### Still intentional (not bugs)

| Flow | Deviation |
|------|-----------|
| local_flow | Localized Pixel-R + IR Feng fit |
| my_flow | Tri-stagger / tri-square topology + IR fit |
| spec_flow `*_ir` OUTs | IR fit instead of eigen (when requested) |

### Re-run

```bash
export MY_FLOW_PYTHON=/path/to/python   # numpy+scipy
"$MY_FLOW_PYTHON" spec_flow/src/check_spec_ibm.py \
  Benchmarks/IBM/TC2/ibmpg2.spice \
  spec_flow/outputs/ibmpg2_nauto_spec_audit/comp1 --flow spec_flow
"$MY_FLOW_PYTHON" -m pytest spec_flow/tests/test_ibm_spec_ports.py -q
"$MY_FLOW_PYTHON" -m pytest local_flow/tests/test_ibm_spec_ports.py -q
"$MY_FLOW_PYTHON" -m pytest my_flow/tests/test_ibm_spec_ports.py -q
```

---

## Summary verdict

| Flow | Port clauses (A–B) on ibmpg2 | Spec Pixel-R + eigen (C–D) |
|------|------------------------------|----------------------------|
| **spec_flow** | **All pass** | Pass on eigen audit OUT |
| **local_flow** | **All pass** (shared ports) | Intentional: localized + IR |
| **my_flow** | **All pass** (full sink grid) | Intentional: dual mesh + IR |
