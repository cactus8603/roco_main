# 整體架構：原生光流監督

目前主流程將一個 3,000,576 參數修復器接在完整凍結的 WAFT 特徵介面上。
完整命令見 [訓練指南](native-training.md)；較早的 target-feature MSE 架構見
[歷史版本](legacy-architecture.md)。

## 前向流程

```text
雙幀 RGB
  → 官方正規化與 padding
  → 凍結 encoder／融合／投影：F0 + 同次四層 observations
  → 原始 F0 執行凍結 WAFT 五步
  → 第五步 native flow／info → 固定 motion prior
  → 修復器：Fhat = F0 + sigma × delta
  → 由 Fhat 重建 hidden、由零光流開始完整 WAFT 五步
  → 五步 flow／info、最終原尺寸 flow
```

原始五步與修復後五步使用同一份凍結 WAFT 權重。原始五步提供推論時也可取得的資訊；
它不是 GT oracle。訓練可快取 prefix 與原始五步，修復後五步則保留對輸入特徵的梯度。

## 輸入與網格

| 資訊 | 形狀／座標 | 來源 |
|---|---|---|
| F0 | `[B,2,64,Hp/2,Wp/2]` | 官方投影後的完整雙幀特徵 |
| CNN pre S2 | 每幀 64 channels、stride 2 | 同次 encoder |
| CNN pre S4 | 每幀 128 channels、stride 4 | 同次 encoder |
| DAv2 block 6、12 | 各每幀 384 channels、patch stride 14 | 原 WAFT 的 DAv2 encoder |
| baseline flow／info | 原圖尺寸，分別 2／4 channels | 原始 WAFT 第五步 |
| sigma | 每筆 64 個正尺度 | 原始雙幀、native centers 的固定 RMS |

`Hp,Wp` 是官方 padding 後尺寸。CNN／F0 網格中心以 padded RGB 的座標定義；
DAv2 patch 中心為 `14j+6.5`。跨層重採樣依實際中心與 stride 轉換，
不能只按 tensor 大小縮放。native support 會計入原圖 offset 與 padding。
模型處理並輸出完整 padded 雙幀，空間均值也在完整 padded 網格計算。

## 跨幀對齊

在 stride 4 的來源網格，使用原始第五步 flow 的對應端點：

- frame 2 → frame 1：在端點 gather 第二幀資訊。
- frame 1 → frame 2：將來源資訊以 normalized bilinear forward splat 傳送至第二幀，保留空洞、支援與碰撞質量。

第二幀得到的 `-f12` 是運送後的反向提案，並非重新求得的 backward flow；
運送後的 info 仍是 forward uncertainty，也不是第二幀的獨立可靠度。
不存在的匹配不會被當成有效對應。

## 凍結與反向傳播

WAFT 所有 508 個 tensors 嚴格載入並保持 eval；其參數不在 optimizer 中，梯度保持為空。
訓練時保留的是 `loss → repaired five-step consumer → Fhat → repair` 的梯度。
「凍結 WAFT」不等於對 repaired consumer 使用 `no_grad()`。

consumer 保留原 WAFT 的 flow／coordinate detach 行為。這裡沒有改用昂貴教師的 full-feedback
最佳化圖，也不訓練 encoder、DAv2、迭代模組或讀出層。

## 監督與推論

TRAIN GT 僅供原生五步 MoL 損失及評分；dev GT 僅供評估。
推論不讀 GT、target features、樣本 ID 或資料切分，且不逐筆最佳化特徵。
目前沒有額外 DINOv3 encoder；DAv2 的特徵已來自原本的 WAFT。

接著閱讀：[修復模型](feature-repair.md) · [完整指南](native-training.md)。
