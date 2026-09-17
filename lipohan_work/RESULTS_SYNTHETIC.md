# Pixel-R Cost Function Synthetic Study

日期：2026-09-16

## 目的

固定抽象模型為一組 global `(Rx, Ry, Rz)` 的 Pixel-R，只改變 fitting
objective，測試以下問題：

1. 真值可由 global 3R 表示時，各方法能否恢復正確參數？
2. 真值具有 spatially varying R 或 long-range coupling 時，哪種 cost 對未見
   currents 的 IR 泛化最好？
3. 訓練 current、Rayleigh basis、p-norm、hybrid 權重與 optimizer 起點如何影響
   結果？
4. 結論是否能跨 grid size、pad voltage 與 conductance noise 維持？

## 方法

### 模型與真值

主要實驗使用 `5 x 5` sink grid、9 個 sparse voltage pads、25 個 sinks。待擬合
模型永遠只有三個 global 參數。建立五種真值：

| Scenario | 真值 |
| --- | --- |
| `exact_global` | 正好是 `(Rx,Ry,Rz)=(0.8,1.4,0.35)` 的 global Pixel-R |
| `smooth_local` | 每個 cell 的 R 依位置平滑變化，另有 via hotspot |
| `random_local` | 每個 cell 的 R 有固定 seed 的 log-normal disorder |
| `long_range` | Global Pixel-R 加上四條跨越 grid 的 nonlocal sink edges |
| `combined` | `smooth_local` 與 long-range edges 同時存在 |

Localized truth 的相鄰 cell conductance 為：

\[
g_x(a,b)=\frac{1}{R_x^{(a)}+R_x^{(b)}},\qquad
g_y(a,b)=\frac{1}{R_y^{(a)}+R_y^{(b)}}.
\]

### Fitting objectives

總共比較 19 種 cost。主要類別為：

各 objective 的完整公式、單位、數學／物理意義、可見與不可見誤差，以及彼此的
等價條件，整理於 [`OBJECTIVE_FUNCTIONS.md`](OBJECTIVE_FUNCTIONS.md)。

- `eigen`：projected eigenvalue least squares。
- `frobenius`：\(\|G_S-G'_M\|_F^2\)。
- `weighted`：對 diagonal、neighbour、pad-via entries 增加權重。
- `rayleigh` / `rayleigh_rel_floor`：在 eigen、random 與 current directions 上
  比較 Rayleigh quotients。
- `voltage`：mixed-BC IR-vector least squares。
- `reff`：比較 port-to-port effective resistance。
- `spectral`：matrix spectral norm。
- `hybrid`：normalized eigen、Rayleigh、voltage residual 串接。
- `minimax`、`pnorm`：壓制 relative IR tail。

所有 R 都在 `log10(R)` 空間最佳化。Vector residual 使用 SciPy TRF
least-squares；scalar spectral/minimax/p-norm 使用 bounded Nelder-Mead。

Raw relative costs 會被接近零的 eigenvalue 或 IR drop 放大，因此另外實作：

- `eigen_rel_floor`
- `rayleigh_rel_floor`
- `voltage_rel_floor`
- `minimax_floor`
- `pnorm_rel_floor`

Voltage denominator floor 是每個 stimulus 最大 drop 的 2%；eigen/Rayleigh
floor 是該 target scale 的 `1e-6`。

### Train/Test 分離

預設 fitting currents 只有四組：

1. 原始 nonuniform load。
2. Gaussian random current。
3. 中央 localized current。
4. X-direction striped current。

測試則使用 seeds 9001--9005，共 120 組未參與 fitting 的 currents：

- 40 組 nonnegative physical log-normal loads。
- 40 組 spatial hotspots。
- 20 組 smooth sinusoidal loads。
- 20 組 zero-mean signed currents。

另有 12 組 boundary tests，讓每個 pad voltage 產生 gradient、checkerboard 或
random variation，幅度為 nominal VDD 的 2%，並搭配不同 sink currents。

主要指標為：

\[
e_{\mathrm{IR}}=
\frac{\|\mathrm{IR}_S-\mathrm{IR}_M\|_2}
     {\|\mathrm{IR}_M\|_2}.
\]

相對 tail 指標使用 2% reference-drop floor，避免 near-zero sink 形成無意義的
數百倍百分比。

## 結果

共執行 178 次 fits，178 次 optimizer 均回報成功。單元測試為 7 passed。

### 1. Exact recovery

當真值確實屬於 global 3R model class，多數方法都恢復
`(0.8, 1.4, 0.35)` 到數值精度。兩個 raw cost 例外：

| Cost | Held-out mean e_IR |
| --- | ---: |
| `eigen_rel` | 8.574% |
| `pnorm` (raw p=8) | 1.038% |
| `eigen_rel_floor` | 約 0% |
| `pnorm_rel_floor` | 約 0% |

這表示 near-zero denominator 與 high-order residual scaling 本身就能讓 optimizer
停在錯誤解；不能把這種結果解讀成 model-class error。

### 2. Main mismatch ranking

| Scenario | Current held-out 最佳 | Mean e_IR | Boundary 最佳 | Mean e_IR |
| --- | --- | ---: | --- | ---: |
| `smooth_local` | `rayleigh_rel_floor` | **10.657%** | `eigen_rel_floor` | **9.023%** |
| `random_local` | `frobenius` | **11.170%** | `frobenius` | **10.147%** |
| `long_range` | `voltage` | **0.791%** | `voltage_rel_floor` | **2.266%** |
| `combined` | `eigen_rel_floor` | **10.947%** | `rayleigh` | **9.809%** |

四個 mismatch scenarios 等權平均後：

| Cost | Current mean e_IR | Boundary mean e_IR |
| --- | ---: | ---: |
| `weighted` | **8.655%** | 8.043% |
| `frobenius` | 8.663% | 8.092% |
| `hybrid` | 8.741% | **7.952%** |
| `voltage_rel_floor` | 9.031% | 8.380% |
| `rayleigh_rel_floor` | 9.036% | 8.181% |
| `eigen_rel_floor` | 9.189% | 8.284% |
| `eigen` | 9.250% | 8.367% |
| `voltage` | 10.211% | 9.084% |

因此沒有 universal winner，但 `weighted` / `frobenius` 是最穩的低成本預設；
`hybrid` 對變動 pad voltage 最穩。

### 3. Current-training ablation (`combined`)

| Training set | Held-out mean e_IR | Boundary mean e_IR |
| --- | ---: | ---: |
| 15 組 diverse currents | **11.142%** | **9.556%** |
| Original only | 11.317% | 9.614% |
| Standard four | 13.574% | 11.332% |
| Original + one Gaussian random | 15.562% | 13.053% |

增加刺激不保證改善；刺激的 distribution 比數量重要。單一未正規化 Gaussian
current 會把解拉離 physical held-out distribution。多個 physical/hotspot/smooth
patterns 才真正改善泛化。

### 4. Rayleigh basis

| Basis | Held-out mean e_IR |
| --- | ---: |
| Mixed eigen + random + current | **10.941%** |
| Full eigen | 11.592% |
| Current only | 12.284% |
| Random only | 13.806% |

Mixed directions 比任何單一 basis 穩定。

### 5. p-norm 與 hybrid

Raw p-norm 沒有改善 tail：`p=2` 為 13.574%，`p=4/8/16` 分別為
20.162%、19.575%、17.384%。原因是高次方讓小電壓 residual 的數值尺度過小，
也容易犧牲平均行為。

Hybrid 中，pure Rayleigh 為 11.239%，equal weights 為 11.469%，pure voltage
為 13.574%。把 voltage 權重放大 10 倍反而退化到 12.143%。

### 6. Seed stability

| Cost | Mean | Standard deviation | Range |
| --- | ---: | ---: | ---: |
| `rayleigh` | **11.300%** | **0.138%** | 11.128--11.508% |
| `voltage` | 13.315% | 2.765% | 11.208--18.573% |
| `hybrid` | 12.598% | 2.727% | 11.025--18.043% |

Rayleigh 對 random training direction 的 seed 明顯更穩。Voltage/hybrid 在 seed 3
產生離群解，因此實際 benchmark 不應只報單一 seed。

### 7. Minimax multi-start

`minimax_floor` 是非平滑且非凸的。同一 training max-error 平台可產生非常不同
的 held-out 結果：

| Start | Held-out mean e_IR |
| --- | ---: |
| Frobenius solution | **11.137%** |
| Voltage solution | 11.905% |
| Weighted solution | 12.972% |
| Rayleigh solution | 17.864% |
| Nominal scale | 18.778% |
| Eigen solution | 19.565% |

若要使用 minimax，必須 multi-start；但其 plateau 代表參數可識別性很差，不適合
作為唯一 objective。

### 8. Grid size

在 combined mismatch 下，六個代表方法的最佳 held-out mean：

| Grid | Best cost | Mean e_IR |
| --- | --- | ---: |
| 3x3 | `frobenius` | 7.180% |
| 5x5 | `rayleigh_rel_floor` | 11.002% |
| 7x7 | `rayleigh_rel_floor` | 10.518% |

誤差沒有隨 n 單調增加，因為 spatial field、pad density 與固定 nonlocal edges
也同時改變。不過 matrix/Rayleigh 類 objective 在三個尺度都位於前段，單一
`voltage` objective 在 3x3 與 5x5 均較弱。

### 9. Conductance-noise robustness

對 exact global truth 的每條 conductance 加 log-normal noise，以 noisy matrix
fit，再回到 clean truth 評估：

| Noise sigma | Best cost | Mean e_IR | Eigen e_IR |
| --- | --- | ---: | ---: |
| 1% | `voltage` | **0.072%** | 0.320% |
| 5% | `voltage` | **0.393%** | 2.609% |
| 10% | `hybrid` | **0.910%** | 5.308% |
| 25% | `rayleigh_rel_floor` | **0.969%** | 19.665% |

低噪聲時 response fitting 最直接；高噪聲時 scale-stabilized Rayleigh 顯示出較強
regularization，而 raw eigen objective 會追隨 noisy spectrum。此項目前每個
noise level 只有一個 noise realization，應再做多 seed 才能形成統計結論。

## 結論

1. **先修 objective 的數值定義。** Raw relative eigen/IR 和 raw minimax 會被
   near-zero target 支配；必須使用尺度 floor。
2. **若只能選一個穩健 baseline，選 weighted/Frobenius。** 它不依賴 current
   seed、速度快，跨 local/nonlocal mismatch 的平均結果最好。
3. **若重視 response，訓練集必須代表部署 currents。** Diverse physical currents
   有效；單一 Gaussian random stimulus 反而傷害泛化。
4. **Rayleigh 是值得保留的折衷。** Mixed basis 或 stabilized relative Rayleigh
   對 seed、grid size 與高 noise 都較穩。
5. **改 cost 無法消除 model-class error。** Spatial heterogeneity 下最佳 global
   3R 仍約 10--11% mean e_IR。要再下降，應改用 localized Pixel-R 或更豐富的
   structured topology，而不是繼續微調 global 3R objective。

## 重現

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r spec_flow\requirements.txt
$env:PYTHONPATH = "lipohan_work/src"
.\.venv\Scripts\python.exe -m pytest lipohan_work\tests -q
.\.venv\Scripts\python.exe lipohan_work\src\run_synthetic_study.py `
  --out lipohan_work\outputs\synthetic_study_full `
  --max-nfev 250 --heldout-repeats 5
```

完整資料位於 ignored output directory：

- `outputs/synthetic_study_full/study_results.json`
- `outputs/synthetic_study_full/study_results.csv`
- `outputs/synthetic_study_full/REPORT.md`
- `outputs/synthetic_study_full/*.png`

主要圖表會在本機執行後產生於以下路徑。`outputs/` 由 `.gitignore` 排除，
因此這些 raw/generated artifacts 不會推送到 GitHub：

- `outputs/synthetic_study_full/heldout_cost_heatmap.png`
- `outputs/synthetic_study_full/combined_cost_ranking.png`
- `outputs/synthetic_study_full/ablation_current_training.png`
- `outputs/synthetic_study_full/ablation_rayleigh_basis.png`
- `outputs/synthetic_study_full/ablation_seed_stability.png`
- `outputs/synthetic_study_full/ablation_minimax_multistart.png`
- `outputs/synthetic_study_full/ablation_grid_size.png`
- `outputs/synthetic_study_full/ablation_conductance_noise.png`

## 限制

- 目前 checkout 沒有 IBM/TSMC benchmark decks，因此本研究使用 synthetic truth，
  不能取代 ibmpg2--6 或 CoWoS-S 驗證。
- Noise robustness 目前每個 level 只用一個 matrix-noise seed。
- Grid-size cases 不是同一連續物理 die 的嚴格 mesh refinement。
- 結果只涵蓋 static resistive mixed-BC IR，不涵蓋 RC/dynamic EMIR。