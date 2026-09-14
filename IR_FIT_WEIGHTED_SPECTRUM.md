# IR Fitting as Input-Weighted Spectral Approximation

IR-drop fitting is **not** a uniform match of the port-admittance spectrum. With only a few free parameters it is equivalent to a **current-weighted** approximation of the **impedance** spectrum: the limited degrees of freedom are spent on Laplacian modes that the training currents actually excite, and (because voltage is \(G^{-1}I\)) on those modes most strongly.

This note makes that statement precise and contrasts it with eigenvalue least squares (the TSMC Pixel-R spec).

---

## 1. Operators

Let \(G'_M \in \mathbb{R}^{n\times n}\) be the Kron-reduced port admittance (symmetric, Laplacian-like). An abstract model \(G_S(\theta)\) (Pixel-R, dual-layer, …) lives in a low-dimensional parameter family, typically \(\theta \in \mathbb{R}^p\) with \(p \ll n\) (e.g. \(p=3\) for \((R_x,R_y,R_z)\)).

Static IR uses **mixed boundary conditions**: pad voltages \(V_P\) fixed, sink currents \(I_S\) applied. Partition

\[
G =
\begin{pmatrix}
G_{PP} & G_{PS} \\
G_{SP} & G_{SS}
\end{pmatrix},
\qquad
G_{SS} V_S = I_S - G_{SP} V_P.
\]

The Dirichlet block \(G_{SS}\) is SPD. The **load impedance** and **effective current** are

\[
Z := G_{SS}^{-1},
\qquad
I_{\mathrm{eff}} := I_S - G_{SP} V_P,
\qquad
V_S = Z\, I_{\mathrm{eff}}.
\]

Sink IR relative to a pad reference \(\bar v\) is \(\mathrm{IR} = \bar v\,\mathbf{1} - V_S\). For fixed \(V_P\), matching IR is equivalent to matching \(V_S\), hence to matching the action of \(Z\) on \(I_{\mathrm{eff}}\).

Write \(Z_M = (G'_{M,SS})^{-1}\) and \(Z_S(\theta) = (G_{S,SS}(\theta))^{-1}\). Drop the subscript “eff” below: \(I\) means the mixed-BC right-hand side that actually drives \(Z\).

---

## 2. Two fitting objectives

**Eigenvalue LS** (spec; uniform over the admittance spectrum). With \(G'_M = Q\,\Lambda_M\,Q^\top\) and projected model eigenvalues \(\lambda_k^S(\theta) := q_k^\top G_S(\theta)\, q_k\),

\[
\min_\theta
\sum_{k=1}^{n}
\bigl(\lambda_k^M - \lambda_k^S(\theta)\bigr)^2
=
\min_\theta
\bigl\| \Lambda_M - \mathrm{diag}(Q^\top G_S(\theta)\, Q) \bigr\|_F^2.
\]

Every mode of \(G'_M\) is weighted equally. High-\(\lambda\) (short-range, “stiff”) modes count as much as low-\(\lambda\) (global IR) modes. No current pattern appears.

**IR LS** (physics-response; as in `fit_ir.py`). Given training currents \(\{I_j\}_{j=1}^{m}\),

\[
\min_\theta
\sum_{j=1}^{m}
\bigl\| V_S(\theta; I_j) - V_M(I_j) \bigr\|_2^2
=
\min_\theta
\sum_{j=1}^{m}
\bigl\| \bigl(Z_S(\theta) - Z_M\bigr) I_j \bigr\|_2^2.
\]

This *does* depend on the current ensemble. The rest of the note shows that this is a weighted spectral problem on \(Z\), not a uniform spectral problem on \(G\).

---

## 3. Exact operator form: \(C\)-weighted Hilbert–Schmidt

Collect the training currents as \(\mathcal{I} = [I_1 \cdots I_m]\) and define the **input Gram** (current covariance)

\[
C := \mathcal{I}\,\mathcal{I}^\top = \sum_{j=1}^{m} I_j I_j^\top \succeq 0.
\]

Because \(Z_S-Z_M\) is symmetric,

\[
\sum_{j=1}^{m}
\bigl\| (Z_S - Z_M)\, I_j \bigr\|_2^2
=
\bigl\| (Z_S - Z_M)\, \mathcal{I} \bigr\|_F^2
=
\mathrm{tr}\bigl( (Z_S - Z_M)^2\, C \bigr).
\]

Thus IR fitting is **Hilbert–Schmidt approximation of the impedance on \(\mathrm{range}(C)\)**:

\[
\min_\theta
\bigl\| (Z_S(\theta) - Z_M)\, C^{1/2} \bigr\|_F^2.
\]

Consequences that do not need an eigenbasis:

1. **Kernel of \(C\) is invisible.** Any mode (or current direction) orthogonal to all \(I_j\) does not enter the loss. The fit is silent there.
2. **The relevant subspace has dimension at most \(m\).** If the training set is a single realistic floorplan current, one is matching \(Z\) along a *line*. Adding random / localized / striped stimuli enlarges \(\mathrm{range}(C)\) and therefore the set of constrained directions.
3. **Changing the current pattern changes \(C\)**, hence changes the minimizer \(\theta^\star\). That is the current-pattern dependence.

---

## 4. Spectral form: input-weighted eigenspectrum of \(Z = G^{-1}\)

Let \(G_{M,SS} = Q\,\Lambda_M\,Q^\top\) with \(\Lambda_M = \mathrm{diag}(\lambda_k^M)\) and \(Q^\top Q = I\). Then

\[
Z_M = Q\,\Lambda_M^{-1}\,Q^\top,
\qquad
V_M(I) = \sum_{k=1}^{n}
\frac{q_k^\top I}{\lambda_k^M}\, q_k.
\]

The coefficient \(q_k^\top I\) is the **modal excitation** of current \(I\). Voltage of mode \(k\) is that excitation **divided by \(\lambda_k\)**: IR is an impedance (\(G^{-1}\)) phenomenon.

Change of basis \(\Delta := Q^\top (Z_S - Z_M)\, Q\) and \(\tilde C := Q^\top C\, Q\) yields the exact identity

\[
\mathrm{tr}\bigl( (Z_S - Z_M)^2\, C \bigr)
=
\mathrm{tr}\bigl( \Delta^2\, \tilde C \bigr).
\]

The diagonal of \(\tilde C\) is the **modal energy** of the training set:

\[
w_k
:=
\tilde C_{kk}
=
q_k^\top C\, q_k
=
\sum_{j=1}^{m} (q_k^\top I_j)^2.
\]

If \(G_S\) shared the eigenbasis \(Q\) (equivalently \(G_S G'_M = G'_M G_S\)), then \(\Delta\) would be diagonal with

\[
\Delta_{kk} = \frac{1}{\lambda_k^S(\theta)} - \frac{1}{\lambda_k^M},
\]

and the IR loss would collapse to a **weighted match of the impedance spectrum**:

\[
\sum_{k=1}^{n}
w_k
\left(
\frac{1}{\lambda_k^S(\theta)} - \frac{1}{\lambda_k^M}
\right)^2.
\tag{\(\ast\)}
\]

In general \(G_S\) does **not** commute with \(G'_M\), so \(\Delta\) has off-diagonal (modal coupling) entries. The spec’s projected eigenvalues \(\lambda_k^S = q_k^\top G_S q_k\) are the Rayleigh quotients of \(G_S\) in the *true* eigenbasis. Using those in \((\ast)\) is the natural diagonal / first-order reading of IR fitting as a spectral problem. The exact loss still includes \(\mathrm{tr}(\Delta_{\mathrm{off}}^2 \tilde C)\); the qualitative weighting \(w_k\) is unchanged.

Near a good fit \(\lambda_k^S \approx \lambda_k^M\),

\[
\frac{1}{\lambda_k^S} - \frac{1}{\lambda_k^M}
=
\frac{\lambda_k^M - \lambda_k^S}{\lambda_k^S\,\lambda_k^M}
\approx
\frac{\Delta\lambda_k}{(\lambda_k^M)^2},
\]

so \((\ast)\) is approximately a **weighted eigenvalue match**

\[
\sum_{k=1}^{n}
\frac{w_k}{(\lambda_k^M)^4}
\bigl(\lambda_k^M - \lambda_k^S(\theta)\bigr)^2.
\tag{\(\ast\ast\)}
\]

Compare with eigen LS, whose weights are identically \(1\). IR LS therefore differs from eigen LS in **two** structural ways:

| | Eigen LS | IR LS |
|---|---|---|
| Operator | admittance \(G\) | impedance \(Z = G^{-1}\) |
| Mode weight | \(1\) (uniform in \(k\)) | \(w_k = \sum_j (q_k^\top I_j)^2\) |
| Extra \(\lambda\)-bias | none | \(\sim \lambda_k^{-4}\) on \(\Delta\lambda_k\), or equivalently uniform on \(\Delta(1/\lambda_k)\) |

Even white currents (\(C \propto I\), so \(w_k\) constant) do **not** recover eigen LS: IR still matches \(1/\lambda\), which over-weights small eigenvalues (global, long-range IR modes) and under-weights large eigenvalues (local, high-frequency modes). Realistic floorplan currents are spatially correlated, so \(w_k\) itself is already concentrated on smooth modes.

---

## 5. Allocation of limited degrees of freedom

The map \(\theta \mapsto (\lambda_1^S(\theta),\ldots,\lambda_n^S(\theta))\) (or \(\theta \mapsto Z_S(\theta)\)) has differential of rank at most \(p\). At most \(p\) independent spectral directions can be matched.

- **Eigen LS** spends those \(p\) directions on the \(\ell^2\)-closest uniform fit of \(\{\lambda_k\}\) — including stiff modes that never appear in \(V = ZI\).
- **IR LS** spends them on the \(\ell^2(w)\)-closest fit of \(\{1/\lambda_k\}\), with \(w_k\) as above. Modes with \(w_k = 0\) are unconstrained; modes with large \(w_k / \lambda_k^4\) are fitted preferentially.

That is the precise meaning of “allocating limited degrees of freedom to the dominant modes excited by the actual current patterns, rather than uniformly approximating the entire spectrum.”

A POD reading of the same fact: the leading eigenvectors of \(C\) (snapshot PCA of the training currents) are the only input directions the loss sees. IR fitting is model-order reduction of \(Z\) on that snapshot subspace, not reduction of \(G\) in the Frobenius metric.

---

## 6. Why the two fits disagree on plots and on IR

Post-hoc projection \(\lambda_k^{S,\mathrm{IR}} = q_k^\top G_S^{\mathrm{IR}} q_k\) (as in `compare_eigen_ir.py`) typically looks **worse** than the eigen-LS spectrum against \(\lambda_k^M\), especially at large \(\lambda_k\). That is expected, not a bug:

- those modes have small \(w_k / \lambda_k^4\);
- the IR optimizer correctly refuses to spend parameters on them.

Conversely, eigen LS can report a small relative spectral error

\[
\frac{\|\lambda^M - \lambda^S\|_2}{\|\lambda^M\|_2}
\]

while producing large sink IR error, because that metric is dominated by large \(\lambda_k\) whereas IR is dominated by small \(\lambda_k\) in the directions \(q_k^\top I \neq 0\).

Held-out current patterns that lie outside \(\mathrm{range}(C_{\mathrm{train}})\) (or that excite previously dark modes) can degrade IR-fit accuracy sharply. Eigen LS is current-agnostic and therefore neither specialized nor brittle in that particular way; it is simply aimed at the wrong operator for EMIR.

---

## 7. Mixed BC, affine pad term, and training design

With a common pad vector \(V_P\),

\[
I_{\mathrm{eff},j} = I_{S,j} - G_{SP} V_P.
\]

The Gram \(C\) is built from \(\{I_{\mathrm{eff},j}\}\), not from sink currents alone. If \(V_P\) is a uniform VDD and the two models have different \(G_{SP}\), part of the IR residual is pad-coupling error, still of the form \((Z_S-Z_M)I_{\mathrm{eff}}\).

The stimuli used in this repo (`build_stimuli`: original lumped current, random, localized, striped) are a particular choice of \(\mathcal{I}\), hence of \(C\). The “original” column is the physically relevant floorplan; the others enlarge \(\mathrm{range}(C)\) so the fit is less of a single-pattern interpolant. There is no unique “correct” \(C\): it is a design choice that *defines* which spectral components of \(Z\) are required to be accurate.

---

## 8. One-line summary

\[
\underbrace{\min_\theta \sum_k (\lambda_k^M - \lambda_k^S)^2}_{\text{eigen LS: uniform on }G}
\quad\text{vs}\quad
\underbrace{\min_\theta \sum_k w_k \bigl(\lambda_k^{-S} - \lambda_k^{-M}\bigr)^2}_{\text{IR LS: }C\text{-weighted on }Z=G^{-1}},
\quad
w_k = \sum_j (q_k^\top I_j)^2.
\]

IR fitting is input-weighted spectral approximation of the load impedance, with weights equal to the energy of the training currents in each eigenmode of the true Dirichlet admittance.
