# Pixel-R Objective Functions: Mathematics and Physical Meaning

本文件完整說明 `lipohan_work` 中用來擬合 global Pixel-R
`(Rx, Ry, Rz)` 的 objective functions。內容以
[`src/costs.py`](src/costs.py) 的實際實作為準，目的不是只列公式，而是回答：

1. 每個 objective 真正比較哪一個數學物件？
2. 它對應什麼電路或 PDN 物理量？
3. 哪些誤差對它可見，哪些不可見？
4. 它的單位、數值尺度與已知病態是什麼？
5. 不同 objective 之間有哪些等價或不等價關係？

實驗設計與結果另見 [`RESULTS_SYNTHETIC.md`](RESULTS_SYNTHETIC.md)。Eigen 與
IR fitting 的 weighted-impedance 推導另見
[`../IR_FIT_WEIGHTED_SPECTRUM.md`](../IR_FIT_WEIGHTED_SPECTRUM.md)。

---

## 1. Common setup

### 1.1 Reference and model matrices

令

\[
G_M := G'_M \in \mathbb{R}^{n\times n}
\]

為真實多層 PDN 在選定 ports 上經 Kron reduction 後的導納矩陣；令

\[
G_S(\theta),\qquad \theta=(R_x,R_y,R_z)
\]

為 global Pixel-R 的 port 導納矩陣。兩者的 port ordering 都是
`[pads | sinks]`。定義矩陣誤差

\[
\Delta G(\theta):=G_S(\theta)-G_M.
\]

`G_M` 與 `G_S` 都是對稱、Laplacian-like matrices，矩陣元素單位為 Siemens
(S)。Kron reduction 對既定 ports 是精確的；以下 objective 的差異只在於如何把
高維的 `G_M` 投影成三參數模型 `G_S(\theta)`。

### 1.2 What the three Pixel-R parameters control

在目前 port-level Pixel-R stamp 中：

- 東西相鄰 sinks 的有效 conductance 為
  \[
  g_x=\frac{1}{2R_x}.
  \]
- 南北相鄰 sinks 的有效 conductance 為
  \[
  g_y=\frac{1}{2R_y}.
  \]
- Pad 與其附著 sink 的 conductance 在 zero-stub 模式為
  \[
  g_z=\frac{1}{R_z}.
  \]

因此 `(Rx, Ry, Rz)` 只控制 nearest-neighbour lateral coupling 與 pad-to-sink
coupling。一般 Kron matrix 中的 dense long-range coupling 無法由三個參數獨立
重建，這就是 model-class mismatch。

### 1.3 Mixed boundary conditions and IR

將任一 port matrix 分割成

\[
G=
\begin{pmatrix}
G_{PP} & G_{PS}\\
G_{SP} & G_{SS}
\end{pmatrix},
\]

其中 pad voltages `V_P` 固定、sink currents `I_S` 已知。Sink voltage 為

\[
G_{SS}V_S=I_S-G_{SP}V_P,
\qquad
V_S=G_{SS}^{-1}(I_S-G_{SP}V_P).
\]

定義 pad reference 與 sink IR drop：

\[
\bar V_P=\operatorname{mean}(V_P),
\qquad
d=\bar V_P\mathbf 1-V_S.
\]

因此一個模型同時可能在三個層次上近似真值：

1. 近似導納矩陣 `G_M` 本身。
2. 近似特定方向上的 quadratic form `x^T G_M x`。
3. 近似 mixed-BC response `d(I_S,V_P)`。

這三件事並不等價。

### 1.4 Optimization variables

所有電阻都在 log space 中最佳化：

\[
x=(\log_{10}R_x,\log_{10}R_y,\log_{10}R_z),
\qquad R_i=10^{x_i}.
\]

這會自然保證 `R_i>0`，並讓跨多個數量級的搜尋較穩定。Vector residuals 使用
SciPy trust-region reflective least squares；scalar objectives 使用 bounded
Nelder-Mead。

---

## 2. Overview and taxonomy

| Family | Objectives | Compared object | Depends on training currents? |
| --- | --- | --- | --- |
| Admittance spectrum | `eigen`, `eigen_rel`, `eigen_rel_floor` | Projected modal conductance | No |
| Admittance matrix | `frobenius`, `frobenius_rel`, `weighted`, `spectral` | `G_S-G_M` | No |
| Directional energy | `rayleigh`, `rayleigh_rel`, `rayleigh_rel_floor` | `x^T G x` | Partly; mixed basis includes current lifts |
| IR response | `voltage`, `voltage_rel`, `voltage_rel_floor` | Mixed-BC sink drops | Yes |
| Tail response | `pnorm`, `pnorm_rel_floor`, `minimax`, `minimax_floor` | Large/worst sink IR errors | Yes |
| Transfer resistance | `reff` | Pairwise effective resistance | No |
| Multi-objective | `hybrid` | Normalized spectrum + energy + IR | Yes |

最初的五個 priority objectives 是：

```text
eigen, frobenius, rayleigh, voltage, minimax
```

---

## 3. Primary objective 1: Eigen

### 3.1 Exact implementation

先對 reference matrix 做 eigendecomposition：

\[
G_M=Q\Lambda_MQ^T,
\qquad
\Lambda_M=\operatorname{diag}(\lambda_1^M,\ldots,\lambda_n^M).
\]

概念公式以上式表示；實作中的 `grounded_eigh` 實際分解

\[
G_M+10^{-10}I,
\]

以抬升 Laplacian nullspace。這個 shift 只用於數值穩定，並不代表新增實體
conductance。

對每個 reference eigenvector `q_i`，模型的 projected modal conductance 是

\[
\widehat\lambda_i^S(\theta)
:=q_i^TG_S(\theta)q_i
=\left[\operatorname{diag}(Q^TG_S(\theta)Q)\right]_i.
\]

程式最小化

\[
J_\lambda(\theta)
=
\sum_{i=1}^{n}
\left(\widehat\lambda_i^S(\theta)-\lambda_i^M\right)^2.
\]

Residual 的單位為 S，objective 的單位為 S^2。

### 3.2 Important mathematical qualification

`\widehat\lambda_i^S=q_i^TG_Sq_i` 是 `G_S` 在 reference eigenvector 上的
Rayleigh quotient，**不一定是 `G_S` 的真正 eigenvalue**。只有當

\[
G_SG_M=G_MG_S
\]

使兩個對稱矩陣共享 eigenbasis 時，`Q^T G_S Q` 才是 diagonal，此時
`\widehat\lambda_i^S` 才能逐項視為模型 eigenvalue。

一般情況下，objective 只使用

\[
\operatorname{diag}(Q^TG_SQ)
\]

而忽略 off-diagonal modal coupling。

### 3.3 Physical meaning

若 `q_i` 被視為一個 normalized port-voltage mode，則

\[
q_i^TGq_i
\]

是該 mode 的 effective conductance，也可視為在單位振幅下的耗散功率尺度。

- 大 `\lambda` mode：conductance 大、較 stiff、通常偏局部或高空間頻率。
- 小 `\lambda` mode：conductance 小、impedance 大，常更容易放大 voltage/IR。

Eigen objective 使用每個 mode 的**絕對導納誤差**。它沒有顯式加入 current
pattern，也沒有直接匹配 inverse operator `G^{-1}`。

### 3.4 What it sees and misses

可見：

- 所有 reference eigenvectors 上的 modal conductance。
- 完整 reference spectrum 中每一個 mode。

不可見或間接可見：

- `Q^T G_S Q` 的 off-diagonal modal mixing。
- 任一具體 current pattern。
- Mixed-BC pad coupling 與實際 sink IR。

### 3.5 Strengths and risks

優點：

- Current-independent，容易重現。
- 與原始 TSMC spec 一致。
- 只需一次 eigendecomposition。

風險：

- Matching `G` spectrum 不等於 matching `G^{-1}I`。
- 絕對誤差常讓大 eigenvalue 對總 loss 的數值影響較大。
- 即使 eigen residual 小，低 eigenvalue 的小偏差仍可能造成大 IR error。

---

## 4. Primary objective 2: Frobenius

### 4.1 Exact implementation

\[
J_F(\theta)
=
\|G_S(\theta)-G_M\|_F^2
=
\sum_{a=1}^{n}\sum_{b=1}^{n}
\left(\Delta G_{ab}(\theta)\right)^2.
\]

Residual 是整個 `\Delta G` flatten 後的向量。Residual 單位為 S，objective
單位為 S^2。

### 4.2 Mathematical meaning

Frobenius objective 是以 Euclidean matrix metric 找出 global 3R family 中最接近
`G_M` 的矩陣：

\[
G_S^*=\arg\min_{G_S\in\mathcal M_{3R}}\|G_S-G_M\|_F.
\]

它保留所有 matrix entries，因此也保留 eigen objective 捨棄的 modal
cross-coupling information。等價地，Frobenius norm 對任何 orthogonal basis 都
不變：

\[
\|\Delta G\|_F=\|Q^T\Delta GQ\|_F.
\]

所以它同時懲罰 eigenbasis 中的 diagonal error 與 off-diagonal error。

### 4.3 Physical meaning

對 Laplacian-like conductance matrix：

- Diagonal `G_aa` 代表 port `a` 連出去的總 conductance。
- Off-diagonal `G_ab` 代表 ports `a,b` 間的 coupling。

Frobenius fitting 問的是：

> Pixel-R 的整體 port coupling matrix 是否接近真實 Kron network？

它不指定 workload，因此是 current-agnostic structural fit。

### 4.4 What it sees and misses

可見：

- 所有 diagonal、near coupling 與 dense long-range Kron entries。
- 所有 modal diagonal/off-diagonal matrix error。

限制：

- 所有 entries 等權，不知道哪些 coupling 對部署 currents 比較重要。
- Global Pixel-R 無法表示 dense far coupling，仍只能做折衷。
- `\|\Delta G\|_F` 小不保證 `\|G_S^{-1}-G_M^{-1}\|` 小，尤其接近小
  eigenvalue 時。

### 4.5 Why it is a useful baseline

它不使用 current、不使用隨機 basis，計算便宜且結果穩定。若 Frobenius 已經明顯
優於 eigen，代表問題不只是 spectrum，而是 eigen objective 丟掉 matrix structure
或配置了不適合 IR 的誤差權重。

---

## 5. Primary objective 3: Rayleigh

### 5.1 Exact implementation

對一組 normalized port-space directions

\[
X=[x_1,\ldots,x_m],\qquad \|x_k\|_2=1,
\]

定義 Rayleigh value

\[
\rho_G(x_k)=x_k^TGx_k.
\]

程式最小化

\[
J_{RQ}(\theta)
=
\sum_{k=1}^{m}
\left[x_k^TG_S(\theta)x_k-x_k^TG_Mx_k\right]^2.
\]

Residual 單位為 S，objective 單位為 S^2。

### 5.2 Basis construction

預設 `mixed` basis 由三種 direction 組成：

1. 從 `G_M` eigenbasis 均勻抽樣的 eigenvectors。
2. Gaussian matrix 經 QR 得到的 random orthonormal directions。
3. 將 sink-current vector 放入 sink slots、pad slots 設為零後正規化的 current
   lifts。

近乎重複的方向會被移除。

`rq_basis=eigen` 只表示使用 eigen directions。它只有在**納入全部 eigenvectors**
時才完整等價於 `eigen`；若只抽樣前述數量，它仍只是 eigen objective 的 sketch。

### 5.3 Mathematical meaning

Rayleigh objective 是對 matrix quadratic form 的 sketch：不比較全部 `n^2`
entries，只要求模型在選定 directions 上產生相同 scalar action。

如果 `x_k=q_k` 是 reference eigenvector：

\[
x_k^TG_Mx_k=\lambda_k^M,
\]

所以 full-eigen Rayleigh 退化為 eigen objective。

實作上 Rayleigh target 使用未加 shift 的 `G_M`，而 `eigen` reference 使用上述
`10^{-10}I` shift，因此兩者仍相差這個極小的 numerical regularization。

Random linear combinations 會混合多個 eigenmodes，因而能間接感受到
`Q^T G_S Q` 的 off-diagonal coupling，這是 pure eigen objective 看不到的部分。

### 5.4 Physical meaning

若 `x` 被解讀為 port voltage pattern，則

\[
P=x^TGx
\]

是電阻網路的耗散功率。Rayleigh fitting 因此可以解讀為：

> 對選定的 normalized voltage patterns，模型與真實網路是否具有相同的
> effective conductance／耗散能量？

但要注意：程式中的 current-lift direction 只是把 current 數字當成 algebraic
port direction。它**沒有先解 `GV=I`**，所以 current-lift Rayleigh 不是實際 IR
response，也不能直接解讀為該 current 下的功率。

### 5.5 Strengths and risks

優點：

- 比完整 Frobenius 更像低維 matrix sketch。
- Mixed basis 同時取樣自然 modes、一般 coupling 與 workload-shaped directions。
- 可透過 basis 設計控制重視的 subspace。

風險：

- 未被 `X` span 覆蓋的方向完全不可見。
- Random basis 與結果可能依賴 seed。
- 它仍匹配 `G` 的 quadratic form，不是 `G^{-1}I`。

---

## 6. Primary objective 4: Voltage / IR response

### 6.1 Exact implementation

對每個 training stimulus `I_S^(k)`，分別解 reference 與 model mixed-BC：

\[
V_{M,S}^{(k)}
=G_{M,SS}^{-1}
\left(I_S^{(k)}-G_{M,SP}V_P\right),
\]

\[
V_{S,S}^{(k)}(\theta)
=G_{S,SS}(\theta)^{-1}
\left(I_S^{(k)}-G_{S,SP}(\theta)V_P\right).
\]

以相同 pad reference 定義 drops：

\[
d_M^{(k)}=\bar V_P\mathbf1-V_{M,S}^{(k)},
\qquad
d_S^{(k)}=\bar V_P\mathbf1-V_{S,S}^{(k)}.
\]

程式最小化

\[
J_V(\theta)
=
\sum_{k=1}^{m}
\|d_S^{(k)}(\theta)-d_M^{(k)}\|_2^2.
\]

Residual 單位為 V，objective 單位為 V^2。

### 6.2 Mathematical meaning

令 load impedance

\[
Z=G_{SS}^{-1}.
\]

則 mixed-BC map 是

\[
V_S(I_S)=ZI_S-ZG_{SP}V_P.
\]

因此 voltage objective 在 training currents 上匹配的是一個**affine response
map**：

- Linear part：`Z I_S`，也就是 load impedance 對 currents 的作用。
- Offset part：`-Z G_SP V_P`，也就是 pad coupling 造成的固定邊界貢獻。

若 reference 與 model 的 `G_SP` 相同，問題可簡化為匹配
`(Z_S-Z_M)I_eff`；一般情況下，objective 同時懲罰 impedance 與 pad coupling
差異。

### 6.3 Input-weighted operator interpretation

將 training effective currents 收集成

\[
\mathcal I=[I_1,\ldots,I_m],
\qquad C=\mathcal I\mathcal I^T,
\]

在固定 affine term 或相同 pad coupling 的簡化下：

\[
J_V
=
\|(Z_S-Z_M)\mathcal I\|_F^2
=
\operatorname{tr}\left((Z_S-Z_M)^2C\right).
\]

所以 voltage fitting 是在 `range(C)` 上逼近 impedance，而不是均勻逼近
admittance spectrum。

### 6.4 Physical meaning

這個 objective 直接問：

> 在指定 pad voltages 與 training sink currents 下，Pixel-R 是否產生正確的
> sink IR map？

它是五個主要方法中最直接對應 static IR use case 的方法。

### 6.5 What it sees and misses

可見：

- Training currents 真正激發的 impedance modes。
- Mixed-BC 下的 pad-to-sink coupling error。
- 實際 sink voltage/drop response。

不可見：

- 與所有 training currents 正交、且不影響 affine offset 的 directions。
- 未出現在 training distribution 的 workload。

### 6.6 Training-distribution dependence

程式預設使用：

```text
original, Gaussian random, center-localized, x-striped
```

不同 stimuli 的振幅沒有在 raw voltage objective 中逐組正規化，所以具有較大
voltage residual energy 的 stimulus 會得到較高權重。增加 random currents 不必然
改善泛化；真正定義 objective 的是 training Gram `C`，不是 stimulus 數量本身。

---

## 7. Primary objective 5: Minimax relative IR

### 7.1 Exact implementation

原始 minimax 定義為

\[
J_\infty(\theta)
=
\max_{k,j}
\frac{
|d_{S,j}^{(k)}(\theta)-d_{M,j}^{(k)}|
}{
|d_{M,j}^{(k)}|+\epsilon
},
\qquad \epsilon=10^{-12}.
\]

`k` 是 training stimulus，`j` 是 sink。Objective 無單位。

### 7.2 Mathematical meaning

Voltage LS 使用 `L_2` aggregation，允許少數大 error 被大量小 error 稀釋；
minimax 使用 `L_infinity` aggregation，只看最差的一個 stimulus/sink：

\[
L_2:\text{ optimize average energy},
\qquad
L_\infty:\text{ optimize worst component}.
\]

它是一個非平滑、非凸 scalar objective，程式使用 Nelder-Mead，原始 `minimax`
預設從 voltage-fit solution warm-start。

### 7.3 Physical meaning

它回答的是：

> 在 training cases 中，任何一個 sink 的**模型相對預測誤差**最多有多大？

這不是「實際 IR 不得超過某個 mV」的 signoff constraint。它限制的是 model error
相對於 golden IR，而不是 IR drop 本身。

### 7.4 Near-zero pathology

若某個 golden drop 很接近零，

\[
|d_{M,j}^{(k)}|\approx0,
\]

即使絕對誤差極小，relative error 仍可能非常大。Optimizer 可能犧牲整張 IR map，
只為降低一個 near-zero denominator 產生的 outlier。

此外，`L_infinity` 常形成平坦平台：不同 `(Rx,Ry,Rz)` 可能得到近乎相同的 worst
training error，卻有非常不同的 held-out error。因此 minimax 需要 multi-start，
而且應同時報告 mean、p95 與 held-out worst error。

---

## 8. Additional objective: Eigen-relative

### 8.1 Raw version

\[
J_{\lambda,\mathrm{rel}}
=
\sum_i
\left(
\frac{\widehat\lambda_i^S-\lambda_i^M}
{|\lambda_i^M|+\epsilon}
\right)^2.
\]

它讓每個 mode 的 percentage error 更接近等權，因而相較 absolute eigen 更強調
小 eigenvalues。這在物理上可能更接近高-impedance/global modes，但 Laplacian 的
null/near-null mode 會讓 denominator 病態。

### 8.2 Stabilized version

\[
J_{\lambda,\mathrm{floor}}
=
\sum_i
\left(
\frac{\widehat\lambda_i^S-\lambda_i^M}
{|\lambda_i^M|+\lambda_{\mathrm{floor}}}
\right)^2,
\]

其中

\[
\lambda_{\mathrm{floor}}
=
\max\left(10^{-6}\max_i|\lambda_i^M|,\epsilon\right).
\]

`eigen_rel_floor` 保留小 modes 的較高權重，但限制 null mode 的最大 leverage。

---

## 9. Additional objective: Frobenius-relative

程式 residual 為

\[
r_F=\frac{\operatorname{vec}(\Delta G)}{\|G_M\|_F+\epsilon},
\]

所以

\[
J_{F,\mathrm{rel}}
=
\frac{\|\Delta G\|_F^2}{(\|G_M\|_F+\epsilon)^2}.
\]

因 denominator 與 `theta` 無關，`frobenius_rel` 與 `frobenius` 在精確數學上有
相同 minimizer。差別只在 loss scale、optimizer stopping tolerance 與跨案例報表的
可比較性。它不是新的物理目標。

---

## 10. Additional objective: Weighted Frobenius

### 10.1 Formula

\[
J_W(\theta)=\|W\odot\Delta G(\theta)\|_F^2,
\]

其中 `odot` 是 elementwise product。預設 residual weights：

| Entry type | `W_ab` | Effective squared weight `W_ab^2` |
| --- | ---: | ---: |
| Diagonal | 5 | 25 |
| Neighbour sink edge | 3 | 9 |
| Pad-to-sink via | 3 | 9 |
| Other/far entry | 0.05 | 0.0025 |

因矩陣完整 flatten，對稱 off-diagonal entries `(a,b)` 與 `(b,a)` 都會出現在
Frobenius sum 中。

### 10.2 Mathematical and physical meaning

它把 Pixel-R topology 當作 prior：

- 優先匹配模型確實能表示的 local stencil。
- 優先匹配 diagonal degree 與 pad-via coupling。
- 對 global 3R 無法逐條表示的 dense long-range fill-in 降低權重。

這不是由唯一物理定律推導出的 weight，而是一種 structured projection metric。
它表達工程判斷：有限參數應先花在可實現且局部重要的 entries。

---

## 11. Additional objective: Rayleigh-relative

### 11.1 Raw version

\[
J_{RQ,\mathrm{rel}}
=
\sum_k
\left(
\frac{x_k^T\Delta Gx_k}
{x_k^TG_Mx_k+\epsilon}
\right)^2.
\]

它比較每個 direction 的 relative conductance/energy error，而不是 absolute error。
弱 conductance directions 因此得到更高相對權重。

### 11.2 Stabilized version

\[
J_{RQ,\mathrm{floor}}
=
\sum_k
\left(
\frac{x_k^T\Delta Gx_k}
{|x_k^TG_Mx_k|+\rho_{\mathrm{floor}}}
\right)^2,
\]

其中

\[
\rho_{\mathrm{floor}}
=
\max\left(10^{-6}\max_k|x_k^TG_Mx_k|,\epsilon\right).
\]

Floor 避免 near-null direction 獲得無限權重。在 PSD reference 上 Rayleigh target
理論上非負；absolute value 仍可提高數值防禦性。

---

## 12. Additional objective: Voltage-relative

### 12.1 Raw version

\[
J_{V,\mathrm{rel}}
=
\sum_{k,j}
\left(
\frac{d_{S,j}^{(k)}-d_{M,j}^{(k)}}
{|d_{M,j}^{(k)}|+\epsilon}
\right)^2.
\]

它讓每個 sink 的 percentage IR prediction error 接近等權，概念上對應 IR scatter
圖中的 relative-error band。

問題是 near-zero IR sink 會主導 objective；這種 sink 可能沒有重要的絕對電壓
誤差。

### 12.2 Stabilized version

對每個 stimulus `k` 定義

\[
f_k=
\max\left(0.02\max_j|d_{M,j}^{(k)}|,\epsilon\right),
\]

再使用

\[
J_{V,\mathrm{floor}}
=
\sum_{k,j}
\left(
\frac{d_{S,j}^{(k)}-d_{M,j}^{(k)}}
{|d_{M,j}^{(k)}|+f_k}
\right)^2.
\]

這不是硬性的 2% error tolerance；2% 是 denominator regularization scale。它讓小
IR sinks 不再無限放大，同時保留 relative weighting。

---

## 13. Additional objective: p-norm

### 13.1 Raw implementation

令 absolute drop residual 為

\[
e_{k,j}=d_{S,j}^{(k)}-d_{M,j}^{(k)}.
\]

程式把 vector residual 寫成

\[
r_{k,j}=\operatorname{sign}(e_{k,j})
(|e_{k,j}|+\epsilon)^{p/2}.
\]

Least-squares 外層平方後，實際 objective 為

\[
J_p=\sum_{k,j}(|e_{k,j}|+\epsilon)^p.
\]

- `p=2` 近似 voltage LS。
- `p>2` 對大 absolute errors 給更高權重。
- `p\to\infty` 的方向與 absolute minimax 有關，但有限 `p` 仍聚合所有 entries。

Objective 單位為 `V^p`，不同 `p` 的 loss 數值不能直接比較。

### 13.2 Stabilized relative p-norm

`pnorm_rel_floor` 直接最小化

\[
J_{p,\mathrm{rel-floor}}
=
\left[
\frac{1}{N}
\sum_{k,j}
\left|
\frac{e_{k,j}}
{|d_{M,j}^{(k)}|+f_k}
\right|^p
\right]^{1/p}.
\]

它是無單位的 scalar objective，使用 bounded Nelder-Mead。相較 raw high-order
residual，直接計算 norm 可減少 least-squares 在 residual 已很小時過早停止的問題。

高 `p` 仍可能犧牲平均 IR 來壓少數 tail errors，所以必須與 held-out mean/p95 一起
報告。

---

## 14. Additional objective: Effective resistance

### 14.1 Formula

對任意 ports `a,b`，effective resistance 是

\[
R_{ab}
=(e_a-e_b)^TG^+(e_a-e_b),
\]

其中 `G^+` 是 Laplacian pseudoinverse。程式以最後一個 port 作 reference，反解
principal minor，再利用

\[
R_{ab}=P_{aa}+P_{bb}-P_{ab}-P_{ba}
\]

得到所有 pairwise values。Reference grounding 不改變 connected resistive network
的 pairwise effective resistance。

Objective 為

\[
J_R(\theta)
=
\sum_{a<b}
\left(R_{ab}^S(\theta)-R_{ab}^M\right)^2.
\]

Residual 單位為 Ohm，objective 單位為 Ohm^2。

### 14.2 Physical meaning

`R_ab` 是在 port `a` 注入 1 A、從 port `b` 抽出 1 A 時的電壓差。它是 inverse
network 的全域 transfer property，比直接匹配 `G` 更接近阻抗行為。

但這個 objective：

- 對所有 pad-pad、pad-sink、sink-sink pairs 等權。
- 不等同於實際 pad-fixed/sink-current mixed BC。
- 可能把參數花在部署 workload 不會使用的 port pairs。

---

## 15. Additional objective: Spectral norm

\[
J_2(\theta)=\|G_S(\theta)-G_M\|_2.
\]

因 `Delta G` 對稱：

\[
\|\Delta G\|_2
=
\max_i|\lambda_i(\Delta G)|
=
\max_{x\ne0}
\frac{\|\Delta Gx\|_2}{\|x\|_2}.
\]

它控制 conductance error matrix 的最壞 absolute action direction。Objective 單位
為 S。

物理解讀是「任意 port vector 經過 conductance-error operator 後，最大可能的
absolute response」。它**不是**最壞 IR error，因為

\[
G_S^{-1}-G_M^{-1}
\]

與 `G_S-G_M` 並非線性等價。Spectral norm 在 singular-value/eigenvalue crossing
處也可能非平滑，所以程式使用 Nelder-Mead，成本通常高於 vector LS objectives。

---

## 16. Additional objective: Hybrid

### 16.1 Exact implementation

程式先建立三個 globally normalized residual blocks：

\[
\bar r_\lambda
=
\frac{\lambda_M-\widehat\lambda_S}
{\|\lambda_M\|_2+\epsilon},
\]

\[
\bar r_{RQ}
=
\frac{\rho_S-\rho_M}
{\|\rho_M\|_2+\epsilon},
\]

\[
\bar r_V
=
\frac{d_S-d_M}
{\|d_M\|_2+\epsilon}.
\]

Residual 串接為

\[
r_H=
\begin{bmatrix}
\sqrt\alpha\,\bar r_\lambda\\
\sqrt\beta\,\bar r_{RQ}\\
\sqrt\gamma\,\bar r_V
\end{bmatrix},
\]

所以 least-squares objective 是

\[
J_H
=
\alpha\|\bar r_\lambda\|_2^2
+\beta\|\bar r_{RQ}\|_2^2
+\gamma\|\bar r_V\|_2^2.
\]

預設 `alpha=beta=gamma=1`。Objective 無單位。

### 16.2 Meaning

Hybrid 嘗試同時保留：

- Eigen 的 global modal coverage。
- Rayleigh 的 directional matrix information。
- Voltage 的 workload-specific IR fidelity。

它是一個 multi-objective scalarization。權重不是物理常數；改變權重就是改變
「網路結構」與「部署 response」的工程取捨。Normalization 只平衡三個 block 的
總尺度，並不保證每個 block 對最終 IR 同等重要。

---

## 17. Stabilized minimax

`minimax_floor` 使用和 `voltage_rel_floor` 相同的 denominator：

\[
J_{\infty,\mathrm{floor}}
=
\max_{k,j}
\frac{|e_{k,j}|}{|d_{M,j}^{(k)}|+f_k}.
\]

它修正 near-zero denominator，但不會消除：

- `L_infinity` 的非平滑性。
- Nonconvex parameterization。
- 多組參數落在相同 worst-error plateau 的問題。

因此 stabilized minimax 仍應使用 multi-start，並以 held-out currents 驗證。

---

## 18. Units and direct comparability

| Objective | Residual/loss unit |
| --- | --- |
| `eigen`, `frobenius`, `weighted`, `rayleigh` | Residual S；squared loss S^2 |
| `spectral` | S |
| `voltage` | Residual V；squared loss V^2 |
| Raw `pnorm` | V^p |
| `reff` | Residual Ohm；squared loss Ohm^2 |
| Relative/floor variants | Dimensionless |
| `minimax`, `pnorm_rel_floor`, `hybrid` | Dimensionless |

不同 objective 的 training loss 數值不可直接互相比大小。例如 `J_F=1` 與
`J_V=1` 具有不同單位與維度，不能用前者小於後者來判斷模型比較好。

公平比較方式是：每個 objective 各自 fit 後，再使用同一組 held-out metrics，例如

\[
e_{IR}
=
\frac{\|d_S-d_M\|_2}{\|d_M\|_2},
\]

以及 held-out mean、p95、worst、within-20% 和 varying-pad boundary tests。

---

## 19. Exact relationships among objectives

### 19.1 Frobenius and Frobenius-relative

因 denominator 與 `theta` 無關：

\[
\arg\min_\theta J_F
=
\arg\min_\theta J_{F,\mathrm{rel}}.
\]

只有數值尺度與 stopping behavior 可能略有不同。

### 19.2 Eigen and full-eigen Rayleigh

若 Rayleigh basis 正好包含全部 reference eigenvectors：

\[
X=Q,
\]

則

\[
J_{RQ}\approx J_\lambda.
\]

等號在忽略 `grounded_eigh` 的 `10^{-10}I` numerical shift 時成立。只指定
`rq_basis=eigen` 但仍抽樣少於 `n` 個 eigenvectors 時，兩者也不完全相等。

### 19.3 Voltage and p=2

忽略 `epsilon` 的極小影響時：

\[
J_{p=2}=J_V.
\]

### 19.4 p-norm and minimax

對固定 finite-dimensional residual vector：

\[
\lim_{p\to\infty}\|e\|_p=\|e\|_\infty.
\]

但 raw `pnorm` 使用 absolute voltage error，raw `minimax` 使用 relative error，
所以兩者只有在採用相同 normalization 時才是直接的 `p -> infinity` 關係。

### 19.5 Eigen versus voltage

即使 currents 是 white、每個 mode 的 excitation energy 相同，voltage fitting 仍
匹配 `1/lambda` 而非 `lambda`。在共享 eigenbasis 且接近真值時，IR loss 對
`Delta lambda_i` 的近似權重為

\[
\frac{w_i}{(\lambda_i^M)^4},
\qquad
w_i=\sum_k(q_i^TI_k)^2.
\]

因此 eigen 與 voltage 在一般情況下不會得到同一組 `(Rx,Ry,Rz)`。

---

## 20. Which objective should be used?

| Goal | Suggested starting point | Reason |
| --- | --- | --- |
| Current-independent structural baseline | `frobenius` | Complete matrix metric、fast、deterministic |
| Respect Pixel-R stencil first | `weighted` | Downweights unrepresentable far fill-in |
| Compact sketch with broad directional coverage | `rayleigh` or `rayleigh_rel_floor` | Mixed basis samples modal and coupling behavior |
| Known deployment-current distribution | `voltage` or `voltage_rel_floor` | Direct mixed-BC response fit |
| Balance structure and response | `hybrid` | Explicit multi-objective compromise |
| Pairwise transfer behavior | `reff` | Matches effective resistance between all ports |
| Worst relative training sink | `minimax_floor` with multi-start | Tail control with denominator regularization |
| High matrix/extraction noise | Start with `rayleigh_rel_floor` | Directional averaging can regularize noisy entries |

這些是 objective 的選擇原則，不是 universal ranking。若真實 `G_M` 不屬於 global
3R model class，換 objective 只能決定三個參數應優先匹配哪一部分，不能消除
model-class error。

---

## 21. Common interpretation mistakes

1. **Projected model eigenvalues 不一定是 model spectrum。**
   `diag(Q^T G_S Q)` 只有在 matrices commute 時才是 `G_S` 的 eigenvalues。
2. **Rayleigh current lifts 不是 current response。**
   它們只是 port-space test directions；沒有解 `GV=I`。
3. **Small Frobenius error 不保證 small IR error。**
   Matrix inversion 會放大小 eigenvalue 附近的誤差。
4. **Voltage fit 不代表 arbitrary-current guarantee。**
   它只直接約束 training Gram 所涵蓋的 subspace。
5. **Minimax error 不是 IR signoff limit。**
   它是模型相對預測誤差，不是實際 drop 的 mV 上限。
6. **Raw relative error 可能只是在追 near-zero denominator。**
   需要 floor、absolute-error guard 或 domain-specific threshold。
7. **Training losses 不能跨 objective 比較。**
   它們可能具有不同單位、維度與 normalization。

---

## 22. Implementation map

| Topic | Source |
| --- | --- |
| Cost names and residuals | [`src/costs.py`](src/costs.py) |
| Log-R optimization and scalar/vector solvers | [`src/fit_costs.py`](src/fit_costs.py) |
| Common post-fit metrics | [`src/evaluate.py`](src/evaluate.py) |
| Original/synthetic sweep | [`src/run_cost_sweep.py`](src/run_cost_sweep.py) |
| Comprehensive held-out/ablation study | [`src/run_synthetic_study.py`](src/run_synthetic_study.py) |
| Synthetic model mismatch and boundary ensembles | [`src/synthetic_cases.py`](src/synthetic_cases.py) |
| Pixel-R matrix stamp | [`../spec_flow/src/pixel_r.py`](../spec_flow/src/pixel_r.py) |
| Mixed-BC solver | [`../spec_flow/src/correlate.py`](../spec_flow/src/correlate.py) |
| Detailed experiment results | [`RESULTS_SYNTHETIC.md`](RESULTS_SYNTHETIC.md) |

---

## 23. One-line conceptual summary

\[
\boxed{
\begin{array}{ll}
\text{Eigen}      & \text{match modal conductance}\\
\text{Frobenius}  & \text{match the full conductance matrix}\\
\text{Rayleigh}   & \text{match conductance energy on selected directions}\\
\text{Voltage}    & \text{match mixed-BC impedance response}\\
\text{Minimax}    & \text{limit the worst relative training-response error}
\end{array}
}
\]

它們不是五種方式在最小化同一件事；它們是五種不同定義的「什麼叫做一個好的
三參數 Pixel-R」。
