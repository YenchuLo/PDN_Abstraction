# PDN Abstraction for EMIR Feasibility — Research Directions

**Context:** TSMC 3Dblox EMIR feasibility uses a single-layer Pixel-R model. The spec proposes automatic extraction of Pixel-R \((R_x, R_y, R_z)\) from a real multi-layer PG by Kron-reducing the multi-layer admittance and matching eigenvalues of the reduced port matrix.

**Status (internship):** The root cause of the spec’s inaccuracy is identified. A spec-faithful reduction framework exists and can run IBM decks and the CoWoS-S SoC. Uniform Pixel-R plus eigenvalue least squares is not a viable feasibility model on realistic grids. Localized Pixel-R plus IR-drop fitting is the best *in-class* result. A better *model class* — a structured reduction of the Kron-reduced port matrix — is still open. Method progress beyond diagnosing the failure is limited.

**Success criterion (“better”):** Under **arbitrary** defined port voltages and currents, the abstracted model produces IR-drop results similar to the real (port-reduced) PDN.

**Source spec:** `PDN_Abstraction_for_EMIR_Feasibility-TSMC_Spec.pdf`

**Central question (updated):** Kron reduction at the chosen ports is exact. Error comes from (i) the uniform Pixel-R model class and (ii) fitting eigenvalues of \(G\) instead of the IR map. The remaining research problem is to reduce \(G'_M\) under a **structural constraint** (Pixel-R-like regular topology) with a controllable IR error bound.

---

## 0. Phase status

| Phase | Original goal | Status |
|-------|---------------|--------|
| **1. Case study + theory** | Prove that eigen-fitting of uniform Pixel-R must err | **Done.** Physical root cause: grid-level equivalent \(R\) is not spatially uniform once metal pitch \(\neq\) grid pitch. Theory: eigen LS \(\neq\) IR; three global parameters cannot span \(G'_M\). |
| **2. Improve Pixel-R** | Better analytical fit and/or localized Pixel-R, implemented | **Partial.** IR-vector fitting and per-grid / per-block \((R_x,R_y,R_z)\) are implemented. Localized Pixel-R + IR fit is the best in-class result. Uniform Pixel-R still cannot represent cell-to-cell \(R\) variation. |
| **3. Challenge Pixel-R** | New model beyond single-layer Pixel-R | **Started, no ceiling.** Dual-layer (sparse + dense) meshes with voltage-source / current-sink stubs were tried. They do not overturn localized Pixel-R. The right successor problem is **graph spectral sparsification of \(G'_M\) with a topology constraint**. |

```text
Phase 1 ──► Phase 2 ──► Phase 3
  done        partial      open (retargeted)
  root cause  IR + local   structured sparsification of G'
```

---

## 1. Problem summary (from spec)

| Item | Content |
|------|---------|
| Today | Feasibility PDN = single-layer R-network with user-specified Pixel-R |
| Gap | Pixel-R is experience-based; need systematic extraction from real PG |
| Intended inputs | Chiplet PG (LEF/DEF/GDS or P&R), RC-tech, chip grid size, power net names |
| Variables | Pixel-R \(x/y/z\) per power net |
| Assumption | Multi-layer and single-layer models share the **same number of ports** |
| Output | Optimal Pixel-R \(x/y/z\) |

### Spec pipeline (error loci)

1. Extract multi-layer R-network from PG.
2. Align ports: bumps/TSVs (voltage) + per-grid lumped current sinks.
3. Build MNA admittance; eliminate internals (Kron / Schur):
   \[
   G'_M = G_P - G_C^\top G_I^{-1} G_C
   \]
4. Build single-layer Pixel-R admittance \(G_S(R_x, R_y, R_z)\).
5. Fit by eigenvalue matching:
   \[
   \min \sum_i \bigl(\lambda_i(G'_M) - \lambda_i(G_S)\bigr)^2
   \]
   using \(G'_M = Q \lambda^M_{\mathrm{diag}} Q^\top\) and the assumption \(Q^\top G_S Q = \lambda^S_{\mathrm{diag}}\).

Kron reduction (step 3) is **exact** at the chosen ports. Error is in **steps 4–5** (model class + fitting objective), with secondary error from **step 2** (port lumping / grid alignment). That split is confirmed by the internship case study.

---

## 2. What was built

### 2.1 Vendor extraction (black box)

Cadence Voltus and Synopsys RedHawk-SC Model Extraction reference flows collapse a multi-layer PDN into a **die model** stored as a SPICE netlist. How the tools form that netlist is not documented.

A PG-database input path was attempted and abandoned: SPICE could not be recovered from the database. The working framework therefore takes **SPICE netlists** only (IBM open-source decks; CoWoS-S SoC die model).

### 2.2 Reduction framework

Input SPICE → port lattice → Kron \(G'_M\) → abstract R-model → IR correlation. Three arms:

| Arm | Model | Role |
|-----|--------|------|
| `spec_flow` | Uniform Pixel-R, one global \((R_x,R_y,R_z)\) | Spec replica (eigen LS default; IR fit available) |
| `local_flow` | Same stars, per-block / per-cell \((R_x,R_y,R_z)\) | Uniformity ablation |
| `my_flow` | Dual-ish meshes (tri-stagger / tri-square) with stubs at real pad/sink XY | Competing structured model |

IBM cases run end-to-end. The CoWoS-S SoC runs; **IR-drop fitting is numerically usable, eigenvalue fitting is not** (numerical corruption of the eigen problem on that netlist).

Details of IR fitting as a current-weighted impedance-spectrum match are in [`IR_FIT_WEIGHTED_SPECTRUM.md`](IR_FIT_WEIGHTED_SPECTRUM.md).

---

## 3. Root cause

### 3.1 Physical: grid-level \(R\) is not uniform

On the CoWoS-S testcase every metal layer is itself a **uniform** stripe grid, but **layer pitches differ** and the **Pixel-R grid pitch differs from those metal pitches**. After the die is cut into Pixel-R cells, each cell covers a different combination of stripes and vias, so the equivalent \((R_x,R_y,R_z)\) seen by that cell is **not** the same as its neighbor’s.

TSMC’s spec experiment used a testcase where **every metal pitch and the grid pitch were identical**. Then every cell sees the same equivalent \(R\), uniform Pixel-R is in the right model class, and eigen fitting looks good.

That mismatch is why feasibility and signoff IR **trends disagree** on CoWoS-S: the feasibility Pixel-R map is essentially **flat across the die**, while signoff retains spatial structure.

Uniform Pixel-R cannot represent cell-to-cell equivalent-\(R\) variation. This is the dominant, and unsurprising, failure mode.

### 3.2 Current pattern modulates how visible the error is

| Stimulus | Effect |
|----------|--------|
| **Uniform current** (no hotspot) | Every cell is equally important. Spatial \(R\) error is fully visible; IR maps diverge from golden. |
| **Non-uniform / floorplan current** | A few eigenmodes of \(Z = G^{-1}\) dominate. IR fitting can spend its few parameters on those modes; the IR *map* can look closer to golden even when the model class is still wrong. |

This is the same statement as IR LS being a **\(C\)-weighted match of the impedance spectrum**, not a uniform match of \(\mathrm{spec}(G)\). See §4.1 and `IR_FIT_WEIGHTED_SPECTRUM.md`.

### 3.3 Mathematical (still valid; now supporting, not the primary story)

**Accuracy** means \(\mathcal{L}(G_S) \approx \mathcal{L}(G'_M)\) under mixed BC (pads voltage-fixed, sinks current-driven), **not** \(\mathrm{spec}(G_S) \approx \mathrm{spec}(G'_M)\).

**(i) Eigenvectors are not shared.** Real symmetric \(A,B\) share an eigenbasis iff they commute. In general \(G'_M G_S \neq G_S G'_M\), so forcing \(Q^\top G_S Q\) onto its diagonal wipes modal coupling.

**(ii) Eigenvalue LS \(\neq\) IR.** \(V = G^{-1}I\) (or the load Schur map). Matching \(\sum_i(\lambda_i^M-\lambda_i^S)^2\) does not minimize \(\|G_S^{-1}I-(G'_M)^{-1}I\|\). Matrices with the same spectrum and different eigenbases give different IR.

**(iii) Model-class capacity.** Uniform Pixel-R has \(\dim\theta\le 3\). After Kron, \(G'_M\) is a dense port Laplacian, \(\Theta(n^2)\) degrees of freedom. Nearest-neighbor stars cannot close
\[
\inf_\theta \|G_S(\theta)-G'_M\|_F
\]
for large \(n\). Localization raises the dimension (one triple per cell) and is why Phase 2 helps *inside* the star family — it still does not recover long-range Kron fill-in.

**(iv) Mixed BC.** The operator that should be fitted is \(Z_{\mathrm{load}}\) (sink Schur given fixed pads), not free \(\mathrm{spec}(G)\).

Secondary: per-grid current lumping; grid pitch \(\gg\) bump pitch (the spec already flags this).

---

## 4. What was tried

### 4.1 IR-drop fitting (implemented)

Replace eigenvalue LS with a direct IR-vector loss on mixed-BC sink voltages:

\[
\min_\theta \sum_j \bigl\| V_S(\theta; I_j) - V_M(I_j) \bigr\|_2^2
=
\min_\theta \sum_j \bigl\| \bigl(Z_S(\theta)-Z_M\bigr) I_j \bigr\|_2^2.
\]

With training Gram \(C=\sum_j I_j I_j^\top\) this is Hilbert–Schmidt approximation of \(Z\) on \(\mathrm{range}(C)\). Limited parameters are spent on Laplacian modes **the current pattern actually excites**, and (because \(V=Z I\)) on small \(\lambda\) (global IR) rather than uniformly on \(\mathrm{spec}(G)\).

On CoWoS-S this is also the only numerically stable objective; eigen LS corrupts.

### 4.2 Localized Pixel-R (implemented)

Keep the star topology; let \((R_x,R_y,R_z)\) vary by \(K\times K\) pad block (`local_flow`: \(K=0\) global, \(K=1\) per cell). Shared edges use series half-arms. This is the in-class fix for §3.1.

Diagnostic snapshot (IBM, seed 0, auto pitch, vs uniform Pixel-R IR fit):

| Case | \(K\) | regions | \(e_{\mathrm{IR}}\) localized | \(e_{\mathrm{IR}}\) uniform |
|------|-------|---------|-------------------------------|-----------------------------|
| ibmpg2 | 0 | 1 | 0.043 | 0.043 |
| ibmpg2 | 2 | 25 | **0.0063** | 0.043 |
| ibmpg2 | 1 | 81 | **~0** | 0.043 |
| ibmpg4 | 0 | 1 | 0.104 | 0.104 |
| ibmpg4 | 1 | 225 | **0.028** | 0.104 |

Held-out `random` / `localized` / `striped` stimuli improve with \(K=1\) as well; the gain is not only overfitting the training current. **Localized Pixel-R + IR fitting is the best result in this project.**

### 4.3 Dual-layer / stub models (tried, not a successor)

`my_flow` attaches pads and sinks at **physical** C4 / current-sink XY through short stubs, and uses two mesh sheets (one coarser, one finer). This restores some layer-dependent \(R\) that single-layer Pixel-R erases. It is a legitimate feasibility-style topology. It has **not** been shown to beat localized Pixel-R on IR under arbitrary \(V/I\), and it does not solve the structural-sparsification problem in §6.

### 4.4 Experiment matrix

| Experiment | Vehicle | Role |
|------------|---------|------|
| Non-uniform current | CoWoS-S | Matches floorplan; IR map closer to golden |
| Uniform current | IBM | Amplifies spatial \(R\) error (no hotspot) |
| Random perturbation of grid-cell \(R\) | IBM | Stresses uniformity assumption |
| Localized Pixel-R (independent per-grid \(R_x,R_y,R_z\)) | IBM | Best in-class |
| Finer pitch (cells need not contain a voltage source) | IBM | Grid vs bump occupancy |
| Dual-layer sparse+dense + VS/sink stubs | IBM | Phase-3 prototype |

### 4.5 Practical takeaway for PG prototyping

If die size and Pixel-R grid size are chosen **while the PG is being built**, so that each cell covers a **repeated / similar** stripe pattern (ideally metal pitch commensurate with grid pitch), uniform Pixel-R is much closer to the right model class. That is an implementation guideline, not a reduction theorem.

---

## 5. Assessment

| Claim | Verdict |
|-------|---------|
| Uniform Pixel-R cannot capture grid-level equivalent-\(R\) differences | Confirmed; obvious once metal pitch \(\neq\) grid pitch |
| The spec testcase hid this by using matched pitches | Confirmed on CoWoS-S vs spec experiment |
| Eigen LS is the wrong objective for IR | Confirmed; also numerically unusable on CoWoS-S |
| IR fitting + localized Pixel-R is the best in-class method here | Confirmed on IBM; IR arm is the usable CoWoS-S path |
| A new model that overturns Pixel-R with a better accuracy/structure tradeoff | **Not achieved** |

The internship **clarified the failure**. It did **not** produce a new abstraction with a theory of error that feasibility tools can rely on. Phase 2 is an engineering mitigation (more Pixel-R parameters + the right loss). Phase 3 remains the research problem.

---

## 6. Remaining research

After Kron, the exact port model is the (generally dense) matrix \(G'_M\). Building a Pixel-R-like mesh is asking for a **sparse, highly structured** \(G_S\) such that the mixed-BC IR map of \(G_S\) tracks that of \(G'_M\).

Unconstrained, that is **graph spectral sparsification** (and related effective-resistance / Schur sparsifiers). Existing work (e.g. Spielman–Srivastava; Liu–Yu pGRASS and ICCAD’23 multilevel aggregation) sparsifies for solve accuracy or IR on the reduced *graph*, but **does not constrain the reduced topology** to a regular Pixel-R (or dual-layer mesh) stencil.

A feasibility Pixel-R is exactly \(G_S\) in a **structured subset** of Laplacian matrices: nearest-neighbor stars on a 2-D lattice, optionally with a second sheet and vias; parameters are edge weights in that stencil only. Little published work treats spectral sparsification **under that structural constraint**.

### 6.1 Directions worth pursuing

1. **Structured sparsification.** Approximate \(G'_M\) (or \(Z_{\mathrm{load}}\)) by a Pixel-R / dual-mesh Laplacian. Questions: which matrix distance is IR-relevant; how to project a general Schur complement onto a stencil; whether localization is the *optimal* stencil-weight assignment or only a greedy one.
2. **Error bounds.** Analytic bounds on \(\|Z_S I - Z_M I\|\) (or \(e_{\mathrm{IR}}\)) in terms of pitch mismatch, stencil bandwidth, and the current Gram \(C\). The internship’s qualitative claims (uniform \(I\) is worst; commensurate pitches help) should become theorems or sharp examples.
3. **Prototyping-aware grids.** Treat grid pitch as a design variable so cells see a periodic PG unit. This may be the highest-leverage *engineering* fix; it does not replace (1)–(2).
4. **Krylov / structured MOR** (optional). Compact multiport models with moment guarantees, if the feasibility API may leave a mesh. Not the lead bet: 3Dblox still wants a Pixel-R-shaped R-network.

### 6.2 What not to spend more time on

- Re-deriving that eigen LS \(\neq\) IR.
- Further uniform-\((R_x,R_y,R_z)\) tuning on mismatched-pitch CoWoS-S.
- Treating vendor die-model generation as a research target (black box; SPICE-in is enough).

---

## 7. Suggested paper / thesis narrative

1. **Problem.** Multi-layer PDN \(\to\) single-layer Pixel-R for 3Dblox EMIR feasibility; need IR fidelity under arbitrary port \(V/I\).
2. **Baseline.** TSMC eigen-fitting of uniform Pixel-R to Kron-reduced \(G'_M\). Looks good when metal pitch = grid pitch; fails on CoWoS-S (flat feasibility IR vs structured signoff).
3. **Diagnosis.** Cell-level equivalent \(R\) varies after gridding; three global parameters cannot represent it. Eigen LS matches the wrong operator; on CoWoS-S it is also numerically unstable.
4. **In-class mitigation.** IR-vector fitting (current-weighted \(Z\)) + localized Pixel-R. Best empirical result; still a star mesh.
5. **Open problem.** Spectral sparsification of \(G'_M\) **with a Pixel-R topology constraint**, plus IR error bounds. Optional: dual-layer stencil, Krylov MOR if the API may change.

**Metric:** \(e_{\mathrm{IR}}\) / \(e_{\mathrm{worst}}\) on held-out stimuli — not eigenvalue residual.

---

## 8. Related work

Prioritize sparsification **with structure**, then Kron / \(R_{\mathrm{eff}}\), then MOR.

### 8.1 Closest to the remaining problem

| Year | Paper | Why |
|------|-------|-----|
| **2023** | Z. Liu, W. Yu, *pGRASS-Solver*, **IEEE TCAD** | Spectral sparsification + effective resistance for large grids. **Gap:** no Pixel-R stencil constraint. [IEEE](https://doi.org/10.1109/tcad.2023.3235754) · [PDF](https://numbda.cs.tsinghua.edu.cn/papers/tcad23.pdf) |
| **2023** | Z. Liu, W. Yu, *Accuracy-Preserving Reduction of Sparsified Reduced Power Grids…*, **ICCAD** | Schur / sparsified reduced grids + IR. Hierarchical, still not a regular feasibility mesh. [IEEE](https://doi.org/10.1109/ICCAD57390.2023.10323865) · [PDF](https://numbda.cs.tsinghua.edu.cn/papers/iccad23.pdf) |
| **2011** | D. Spielman, N. Srivastava, *Graph Sparsification by Effective Resistances*, **SIAM J. Comput.** | Classical spectral sparsifiers; unconstrained support. [SIAM](https://doi.org/10.1137/080734029) |
| **2024** | A. Carlucci et al., *Structured Model Order Reduction of System-Level Power Delivery Networks*, **IEEE Access** | Krylov / structured MOR — optional Phase 3 if the API may leave a mesh. [IEEE](https://doi.org/10.1109/access.2024.3359853) |

### 8.2 Foundations

| Year | Paper | Why |
|------|-------|-----|
| **2013** | F. Dörfler, F. Bullo, *Kron Reduction of Graphs…*, **IEEE TCAS-I** | Exact port Schur; \(R_{\mathrm{eff}}\) invariance. [IEEE](https://doi.org/10.1109/tcsi.2012.2215780) |
| **2023** | T. Sugiyama, K. Sato, *Kron Reduction and Effective Resistance of Directed Graphs*, **SIAM J. Matrix Anal. Appl.** | Modern Kron / \(R_{\mathrm{eff}}\). [SIAM](https://doi.org/10.1137/22m1480823) · [arXiv](https://arxiv.org/abs/2202.12560) |
| **2016** | Abed, Najm et al., *A Fast Layer Elimination Approach for Power Grid Reduction*, **ICCAD** | Layer thinning → dual-layer stencils. [PDF](https://www.eecg.utoronto.ca/~najm/papers/iccad16-abed.pdf) |
| **2019** | S. Köse, E. Friedman, *Effective Resistance of Two-Dimensional Truncated Infinite Mesh Structures*, **IEEE TCAS-I** | Mesh \(R_{\mathrm{eff}}\). [IEEE](https://doi.org/10.1109/tcsi.2019.2933749) |

### 8.3 Chiplet / 2.5D context

| Year | Paper | Why |
|------|-------|-----|
| **2025** | *PDN analysis of 3D chiplet integration…*, **IEICE ELEX** | Chiplet PDN + IR. [DOI](https://doi.org/10.1587/elex.21.20240438) |
| **2024** | X. Ma et al., *Electrical-Thermal Co-Simulation of Chiplet Heterogeneous Integration*, **IEEE TVLSI** | Chiplet PDN equivalents. [IEEE](https://doi.org/10.1109/tvlsi.2024.3430498) |
| **2023** | A. Carlucci et al., *Compressed Multivariate Macromodeling…*, **IEEE TCPMT** | Response-preserving macromodels vs eigen-only fit. [IEEE](https://doi.org/10.1109/tcpmt.2023.3292449) |

### 8.4 Benchmarks

IBM PG (this repo) and SRAM-PG (Shen, Liu, Yu, 2024, [arXiv:2404.05260](https://arxiv.org/abs/2404.05260)). CoWoS-S die-model netlist is local / vendor, not public.

---

## 9. Open questions

1. What IR error does 3Dblox feasibility actually tolerate (mV vs %)?
2. May the feasibility API emit **localized** (per-cell) Pixel-R, or must output stay one global \((R_x,R_y,R_z)\)?
3. Can a Pixel-R / dual-mesh stencil be posed as a convex or provably near-optimal sparsifier of \(G'_M\) or \(Z_{\mathrm{load}}\)?
4. Error bound in terms of pitch mismatch and current Gram \(C\)?
5. Ports = bumps + grid sinks only, or instance-level currents later?
6. Static IR only, or RC / dynamic EMIR (would favor Krylov MOR)?

---

## 10. Document history

| Date | Note |
|------|------|
| 2026-07-21 | Initial consolidation from TSMC PDN Abstraction spec analysis. |
| 2026-07-21 | Merged official research plan: 3 phases; commutativity / over-simplification / \(G^{-1}\) proofs; physics-response fitting; localized Pixel-R; dual-layer model; Krylov MOR. |
| 2026-09-01 | Rewritten from internship results: CoWoS-S pitch/grid root cause; SPICE-only framework; IR fit + localized Pixel-R as best in-class; Phase 3 retargeted to structured spectral sparsification of \(G'_M\). |
