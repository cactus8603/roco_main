# Optical Flow 實驗證據與進度

最後更新：2026-10-06（Asia/Taipei）

這裡只保留能支持研究判斷的結果、解讀與下一步驗證。每日重要里程碑放在
[`daily/`](daily/)，研究待辦放在 [`TODO.md`](TODO.md)。

## 目前結論

U2-main 在目前內部 validation protocol 上，最佳 EPE 已由 raw SEA-RAFT Spring-M 的
`3.1829` 降至 `1.3556`，相對下降約 `57.4%`。最佳結果連續出現在 epoch index 9 與
10，兩者僅差 `0.00004`，因此不像單次偶然尖峰。

但這個數字尚不能直接宣稱超過官方 SEA-RAFT Sintel Clean `1.44`。目前 validation 是
Clean+Final 混合的指定場景 crop，輸入尺度也和官方推論口徑不同；正式結論必須等待已
排程的標準 Sintel Clean/Final 評測。

## SEA-RAFT / U2 核心證據

### 同一內部 protocol 的效果

| 模型／階段 | Validation EPE | 相對 raw base 變化 | 解讀 |
| --- | ---: | ---: | --- |
| Raw SEA-RAFT Spring-M | 3.1829 | — | 固定 checkpoint 的內部基準 |
| U1 best | 3.1723 | -0.3% | 單獨 refiner 初始化的改善很小 |
| U2 first validation | 3.1713 | -0.4% | 剛進入交替訓練時仍接近基準 |
| U2 best（epoch index 10） | **1.3556** | **-57.4%** | 改善主要在 U2 交替訓練中形成 |

這組比較使用同一 validation 管線，因此可以支持「U2 訓練確實學到有效改進」。它不能
支持「已超越官方 hidden test」，因為兩者的資料與評測口徑不同。

### 最佳點是否穩定

| Epoch index | EPE | AUSE | Calibration MAE | 備註 |
| ---: | ---: | ---: | ---: | --- |
| 9 | 1.355645 | — | — | 第一次到達最佳區間 |
| 10 | **1.355602** | 0.019579 | 0.074043 | 目前 EPE best checkpoint |
| 11 | 1.456744 | 0.020787 | 0.070072 | 開始退化 |
| 12 | 1.456456 | 0.020593 | **0.068332** | calibration 較好，但 EPE 較差 |
| 13 | 1.563723 | 0.020548 | 0.073722 | 後段最差區間 |
| 14 | 1.563631 | 0.020658 | 0.081693 | 與 epoch 13 接近 |
| 15 | 1.413621 | 0.018910 | 0.078342 | EPE 開始恢復 |
| 16 | 1.413623 | **0.018535** | 0.077611 | ranking quality 最好，但 EPE 非最佳 |

解讀：

- epoch 9/10 幾乎相同，顯示 `1.3556` 有局部穩定性。
- 後續訓練沒有單調改善；使用 `best.pt` 而不是最後一輪是必要的。
- EPE、AUSE 與 calibration 的最佳 epoch 不同，代表 flow accuracy、uncertainty ranking
  與機率校準不是同一個目標，論文中應分開報告。
- epoch 15/16 從約 `1.56` 恢復到 `1.41`，但目前沒有證據支持把訓練延長到 20 epochs
  以上。

### Validation 到底測了什麼

- 場景：`alley_2`、`ambush_2`、`bamboo_1`；
- Clean 118 pairs + Final 118 pairs，共 236 pairs；
- deterministic center crop：384×832；
- EPE 在所有有效像素上計算；
- SEA-RAFT recurrent iterations：4。

因此 `1.3556` 是「這三個場景、Clean+Final crop protocol」的結果，不是完整 Sintel
Clean test EPE。上游 Spring-M checkpoint 的 TSKH lineage 亦包含 Sintel，所以這些場景
不能視為完全未見過的獨立 test set。

## Uncertainty 指標怎麼解讀

- **AUSE**：不確定性排序錯誤的面積，越低越好。
- **Spearman**：預測不確定性和實際誤差的排序相關，越高越好。
- **Severe-error AUROC**：辨識大誤差像素的能力。最佳 EPE epoch 為 `0.9903`，代表
  大錯誤定位訊號很強。
- **Calibration MAE**：多個 nominal coverage 與實際 coverage 的平均差距；`0.074`
  約代表平均相差 7.4 個百分點，越低越好。

目前 uncertainty ranking 很強，但 calibration 仍有改善空間；最佳 calibration epoch
也不是最佳 EPE epoch。若 9A 用 uncertainty 選 action，必須同時檢查排序與校準。

## SAM 更新

目前 SAM 只具備「方法與資料準備已完成」的證據，尚無效能結論：

- 使用固定 SHA-256 的官方 SAM ViT-H，採 smallest-mask-priority full segmentation；
- mask 會逐檔驗證來源、digest 與 complete manifest，dataset 採 fail-closed；
- U2 flow phase 加入 uncertainty-guided regional homography loss；observer phase 不接收
  此項 gradient，避免 observer/flow 目標混淆；
- 最多使用 6 個 regions，並要求可靠點比例、RANSAC inlier fraction 與 per-region loss cap；
- segmentation 目前為 1,029 / 1,348 frames（76.3%），U2-SAM 尚未產生 validation 結果。

因此現在不能寫「SAM 有提升」。SAM 的有效性必須由 U2-main vs U2-SAM、以及 9A-main
vs 9A-SAM 的同口徑結果判定，並額外看 segment boundary EPE 與 fitted-region coverage。

## WAFT 更新

### 已完成且可引用的舊證據

WAFT 的 frozen KITTI confirmation 已支持一個重要、跨 backbone 的結論：horizontal
equivariance discrepancy 很適合定位錯誤，但不適合直接選 action。

| Frozen evaluation | Error-localization AUROC | AP | Candidate-direction AUROC |
| --- | ---: | ---: | ---: |
| WAFT, KITTI C1, 48 scenes / 240 cases | **0.8601** | **0.8112** | 0.4844 |
| SEA-RAFT, KITTI C1, 48 scenes / 240 cases | **0.8864** | **0.7796** | 0.5150 |
| SEA-RAFT, Sintel, 23 scenes / 600 cases | **0.9261** | **0.7568** | 0.5014 |

這三組結果一致地顯示：disagreement 能回答「哪裡可能錯」，卻無法回答「哪個候選比較
好」。因此 uncertainty-only replacement 或 naive averaging 不應作為主方法。

WAFT frozen hard selector 在 KITTI confirmation 的結果為：

- clean gain：`+0.2819 px`；
- corrupt gain：`+0.2610 px`，scene-bootstrap 95% CI `[0.1520, 0.4137]`；
- worst harm：`0.1489 px`；
- frozen gates：PASS。

這是候選驗證與 selective repair 確實能改善 WAFT flow 的證據，不只是 uncertainty
correlation。不過它屬於既有 CESR-CTR / WAFT 實驗線，不能冒充目前 U2-SAM 的結果。

### 新 U2 cross-backbone 狀態

WAFT adapter 已實作相同的 uncertainty/refinement trainer contract，但新 U2 cross-backbone
config 仍明確為 `enabled=false`。原因是尚未：

1. 固定 WAFT checkpoint 與 SHA-256；
2. 訓練 WAFT-specific U0 observer；
3. 訓練 WAFT-specific U1 bounded refiner；
4. 依相同 held-out split 與 SAM contract 執行。

所以目前可以主張「舊 CESR-CTR 機制有 WAFT 跨 backbone 證據」，不能主張「新的 U2 / SAM
training gain 已在 WAFT 複現」。

## 尚不能下的結論

- U2 已超越 SEA-RAFT 官方 Sintel Clean 1.44；
- SAM regularization 已有效；
- action bank 已提升模型；
- 新的 U2/SAM 訓練效果已跨到 WAFT；
- 最後一個 epoch 優於 best checkpoint。

目前 SEA-RAFT 內部 protocol 另有 `[0,1]` 輸入，而官方 backbone 預期 `[0,255]` 後再於
模型內正規化的尺度差異。標準 full-resolution Clean/Final evaluator 已排程，將直接檢驗
57.4% 的內部改善能否跨口徑成立。

## 當前執行狀態

- U2-main：4,725 / 5,420 steps（87.2%）；最佳 EPE 仍為 1.3556。
- SAM masks：1,029 / 1,348 frames（76.3%）；尚無 SAM training metric。
- Main 與 SAM 的 U2 → 9A → evaluation 交接已自動化，工程細節不在此展開。

> 本頁只在出現新 validation、標準評測、跨 backbone confirmation 或消融結論時更新。
