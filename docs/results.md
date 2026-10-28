# 結果與驗證範圍

目前主流程使用完整凍結的 source-only WAFT，訓練 3,000,576 參數修復器，
以完整五步原生 MoL loss 更新。這與較早的 target-feature MSE 模型不同。

## 目前模型的固定中途評估

以下是固定第 16 epoch checkpoint 的**逐影像平均最終 EPE**，單位像素，越低越好：

| 資料 | 對數 | 原始 WAFT | 修復模型，第 16 epoch |
|---|---:|---:|---:|
| KITTI TRAIN | 160 | 2.949504 | 1.257547 |
| KITTI dev | 40 | 4.632284 | 3.519644 |

可回讀 [全部 200 筆的 init／epoch16 標量](results/native-epoch16-cases.csv)與
[人口、checkpoint SHA 及聚合證據](results/native-epoch16-evidence.json)。
這份重算核對彙總算術，沒有再執行一次原始 GT／光流評估。

這是固定 KITTI 160 TRAIN／40 dev、單一 seed 訓練的中途監測；
固定第 32 epoch 的訓練及最終獨立核驗尚不能由此表宣稱完成。
中途評估使用完整 200 筆分母，不是 online minibatch 的更新前 loss 或擬合分數。

40 dev 與 160 TRAIN 在案例、原始 RGB／GT 及已核對來源群組沒有重疊；
但 dev 已被歷史研究使用，不是官方隱藏 KITTI test，也不是全專案從未曝光資料。
這張表不能支持跨資料集、官方 leaderboard 或多 seed 的結論。
公開程式未捆綁目前訓練後權重；程式移植驗證與上述原生 run 的效果分別記錄。

## Loss、EPE 與錯誤追蹤

原生 MoL loss、平均 EPE 與 Fl outlier 比例是不同量，沒有逐步同降保證。
目前模型雖然可拆解 global／spatial delta，但完整的空間修復收益、傷害與分支原因
仍需保存光流、固定來源像素及控制介入評估。不能把 delta 圖當成經校準的錯誤機率。

五步歷程可解釋版與 where／why／how 設計是另外的研究版本，不是這裡第 16 epoch
結果所使用的模型；它們的小樣本結果不能混入本表。

## 成本定義

完整影像推論成本為：

```text
原始 encoder／prefix + baseline 五步 + prior／對齊處理
+ 修復器 + repaired 五步
```

3.00M 參數只是修復器的參數量。修復器單獨 latency 或靜態 MAC census 不包含 baseline
五步、資料搬移、正規化、重採樣與其他處理，不能當成整體方法快於／小於 WAFT 的證明。
訓練快取 prefix／baseline 可省去重複工作；推論比較仍需按完整流程量測。

## 較早的 target-feature MSE 與教師結果

[歷史結果文件](legacy-results.md)與 [既有公開 CSV](results/summary.csv)記錄的是
便宜／昂貴 target features 本身的品質，以及當時的移植驗證範圍。
它們不是目前 3.00M 模型的結果，不可把教師直接輸入 WAFT 的收益歸給修復器。
[逐案例 CSV](results/cases.csv)與 [證據摘要](results/evidence.json)保留原始來源。

原始 RGB、GT arrays、WAFT 權重、feature tensors、訓練 checkpoint 與執行 logs
不放入 repository。資料位置清單是歷史盤點，見 [資料清單](data-inventory.md)。
