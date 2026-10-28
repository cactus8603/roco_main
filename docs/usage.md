# 命令與資料格式

目前的主流程是 `waft-repair native-*`；完整操作順序見 [原生光流訓練指南](native-training.md)。
程式可獨立安裝，不依賴本機 operations、背景工作或私人資料路徑。

## 目前命令

| 命令 | 用途 |
|---|---|
| `doctor` | 核對固定官方 WAFT 來源、checkpoint 與完整 state |
| `native-split-kitti` | 用固定公開 160／40 ID 建立原始 KITTI 影像清單 |
| `native-prepare` | 從 RGB 快取 F0、四層特徵、原始 baseline 與對齊依據 |
| `native-train` | 原生五步 MoL 訓練與續訓，只更新修復器 |
| `native-evaluate` | 固定 checkpoint 的 train／dev／all 光流評估 |
| `native-infer` | 不讀 GT，由兩張影像輸出修復後原尺寸光流 |
| `native-import-checkpoint` | 將相同原生模型的既有 full checkpoint 匯為 portable repair model |
| `convert-gt` | KITTI PNG／Sintel FLO 轉成統一數值 NPZ |

各命令的完整參數可用 `waft-repair COMMAND --help` 查看；也可寫成
`python -m waft_repair COMMAND ...`。

## 原始影像清單

[`examples/native/images.json`](../examples/native/images.json)使用：

```json
{
  "schema": "waft-native-images/v1",
  "samples": [
    {
      "id": "kitti2015/000000",
      "split": "train",
      "image1": "first.png",
      "image2": "second.png",
      "ground_truth": "flow.npz",
      "ground_truth_format": "npz"
    }
  ]
}
```

相對路徑以 JSON 所在位置解析。`split` 是 `train` 或 `dev`；`id` 唯一。
GT format 支援 `kitti`、`sintel`、`npz`，圖像和 GT 幾何需一致。
NPZ 的 flow 是 `float32[H,W,2]`、valid 是 `bool[H,W]`，以 `allow_pickle=False` 讀取。
推論命令不讀這份清單或 GT。

`native-prepare` 產生的 manifest 由工具建立，包含資料、幾何、雜湊與 WAFT 綁定。
不要手動將舊 target-feature manifest 改名後交給 native-train。

## 模型介面

目前的 `MotionFeatureRepair` 與 geometry、native consumer、loss、training 置於
[`src/waft_repair/motion`](../src/waft_repair/motion)。
模型接收 detached FP32 F0 `[B,2,64,Hf,Wf]`、正值 sigma `[B,64]`、
四層 observations、固定 motion prior 與 native／padding 幾何；輸出完整同形 Fhat。

```text
Fhat = F0 + sigma × (global + spatial_raw - mean_fullpad_HW(spatial_raw))
```

GT 不在模型 forward 中；native consumer 用輸出特徵重建 hidden，再從零光流完成原生五步。
`return_aux` 可提供 global／spatial delta 與對齊支援資訊。
這些不是經校準的錯誤機率，模型沒有錯誤分類／根因預測頭。
實作細節與精確座標見 [架構](architecture.md)及 [模型](feature-repair.md)。

## Checkpoint

- `checkpoint.pt` 是同一份資料與設定的完整續訓狀態。
- `model.pt` 是 portable repair 模型，供 native-infer／native-evaluate 使用。
- `native-import-checkpoint` 只匯入相同模型的 repair state，不嫁接不同 run 的 optimizer。

PT 使用受限制的 tensor／基本 metadata 載入；NPZ 不讀 pickle。
官方 checkpoint 與原始資料不隨套件分發，CLI 不會自動下載模型或啟動背景訓練。

## 歷史介面與授權

原有 `FeatureRepair`、`prepare`、`train`、`infer-features`、`infer-images`、
`cheap_target`／`expensive_target` 仍對應較早的 3.92M target-feature MSE 流程。
其資料格式、模型、loss 與目前不同，詳見 [歷史 API](legacy-usage.md)及
[歷史訓練指南](legacy-running.md)。

完整授權與來源保留於 [`third_party/feature_repair`](../third_party/feature_repair)。
其中 Mamba／SLiM 的授權用於保留的歷史程式；目前 motion 修復器不執行這些 extension。
WAFT 原碼與權重沿用官方授權，使用前請閱讀對應條款。
