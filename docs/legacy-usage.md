> 歷史版本：本頁記錄較早的 target-feature MSE 修復流程，使用 3,923,200 參數模型。
> 目前的 3,000,576 參數原生光流訓練，請看 [完整指南](native-training.md)。
> 歷史頁面的狀態描述不代表目前訓練狀態。

# 程式介面與資料格式

從安裝到執行的完整命令見 [執行指南](legacy-running.md)。
可重用程式放在 [`src/waft_repair`](../src/waft_repair)，不依賴本機研究目錄或背景工作。

## 模組

| 程式 | 用途 |
|---|---|
| `model.py`、`context.py` | 完整雙幀、局部與跨幀的無界 residual 修復模型 |
| `data.py` | 特徵封包、manifest、原始 RMS、便宜／昂貴快取匯入 |
| `ground_truth.py` | KITTI PNG／Sintel FLO 轉換 |
| `training.py` | Feature MSE 訓練、續訓與特徵推論 |
| `checkpoint.py` | 權重、optimizer、RNG 與雜湊保存／載入 |
| `backend.py` | 嚴格載入官方凍結 WAFT、影像資料準備及光流推論 |
| `targets.py` | 非遮擋 RGB 重建與可選 32／256／512 次特徵最佳化 |
| `cli.py` | `waft-repair` 命令 |

## 修復模型

```python
from waft_repair import FeatureRepair

model = FeatureRepair(backend="reference", seed=7)
# original: detached FP32 [B,2,64,H,W]
# sigma: positive original-only FP32 [64] or [B,64]
# native_mask: bool [1|B,1,1,H,W], geometry metadata
repaired = model(original, sigma, native_mask)
```

Reference 適合小型 CPU tensors；完整影像需明確選擇 `backend="native"` 與 CUDA。
模型 forward 不接收 target、GT、RGB 或 WAFT，輸出包含全部雙幀 padding。

## 訓練清單

[範例 JSON](../examples/repair/features.json) 的結構是：

```json
{
  "schema": "waft-repair-dataset",
  "waft_checkpoint_sha256": "1b49d1fc4513c6fa34a8de8f791dd8735d366d4b25d8dea47626a79056edd402",
  "target_kind": "cheap",
  "samples": [
    {
      "pair_id": "street-pair",
      "domain": "kitti",
      "native_hw": [375, 1242],
      "padding": [51, 51, 36, 37],
      "input_file": "street-input.pt",
      "target_file": "street-target.pt"
    }
  ]
}
```

這是檔案格式示例，影像幾何必須填入自己的實際值。
`target_kind` 為 `cheap` 或 `expensive`；`domain` 為 `kitti`、`sintel-clean`、`sintel-final` 或 `synthetic`。
相對路徑由清單所在位置解析。`manifest` 命令核對內容後，將來源路徑、bytes、SHA-256 綁入正式清單。

Input 封包使用 `features`，target 封包使用 `teacher_features`，皆為 CPU FP32 `[1,2,64,Hf,Wf]`。
兩種封包均包含 `pair_id`、`domain`、`native_hw`、`padding`、`waft_checkpoint_sha256`；
target 另包含 `target_kind` 與對應製法的 `target_method`。
昂貴版的 `target_method` 為 `waft-dense-final-epe-{steps}-v1`，
`optimizer_updates` 記錄 32、256 或 512，`selected_step` 位於 0 到該預算之間。
`prepare` 產生的 manifest 另保存 `target_steps` 與每筆 `teacher_updates`；
載入時會核對步數與方法一致。既有 512 步封包仍相容。
若同一個昂貴封包保存兩份特徵，可在清單指定 `input_key="original"` 與 `target_key="teacher_features"`。

`.pt` 透過 `weights_only=True` 載入，只接受 tensors 與基本 metadata。
`.npz` 使用同名數值陣列，以及 JSON 字串 `metadata`，並以 `allow_pickle=False` 載入。
σ 與 native mask 會由原始特徵重新計算，不使用 target 估計尺度。

對 Final 的便宜 target，清單還需 `target_source_pair_id`；target 封包識別對應 Clean pair，
完整保留兩張 Clean target 特徵。第二張與 Final 不同是允許且必要的配對監督行為。
KITTI／Clean 的便宜 target 第二張則必須完整等於原始第二張特徵。

## Python 訓練與推論

```python
from waft_repair.data import prepare_manifest
from waft_repair.training import train, infer_features
from waft_repair.backend import infer_images

prepare_manifest("data/features.json", "data/train.json")
train("data/train.json", "outputs/repair", device="cuda:0", backend="native", epochs=10)
infer_features("outputs/repair/checkpoint.pt", "data/input.pt", "outputs/repaired.pt",
               device="cuda:0", backend="native")
infer_images("outputs/repair/checkpoint.pt", "first.png", "second.png", "outputs/flow.npz",
             "external/WAFT", "checkpoints/waft/zero-shot.pth", device="cuda:0", backend="native")
```

這些呼叫使用已準備好的本機檔案；不自動下載權重、啟動背景訓練或分配共用 GPU。
訓練只使用修復模型；影像推論不需要 GT 或 target。

## Target API

```python
from waft_repair.targets import cheap_target, expensive_target
from waft_repair.backend import load_backend

backend = load_backend("external/WAFT", "checkpoints/waft/zero-shot.pth", device="cuda:0")
original = backend.encode_pair(rgb_pair)
cheap = cheap_target(backend, rgb_pair, gt_cpu, valid_cpu, visible_cpu, original=original)
expensive = expensive_target(backend, original, gt_cpu.to("cuda:0"), valid_cpu.to("cuda:0"),
                             steps=32)  # 32, 256, 512；省略時為 512
```

昂貴 API 必須在 autograd 啟用時呼叫，WAFT 權重仍保持凍結。
CLI 對應 `prepare --target expensive --target-steps 32`；
`prepare_dataset(..., target_kind="expensive", target_steps=32)` 可用於影像批次準備。
選項只改外層 AdamW 步數，WAFT 每次仍完整執行 5 次更新。
Final 的便宜 target 請傳入完整配對 Clean RGB 與其原始特徵，原 Final input features 另行保存。
舊的固定 target API 路徑保留相容性；新流程統一使用上述可讀模組。

## 授權

本地修復實作包含 SLiM（MIT）與 Mamba（Apache-2.0）的改寫部分，
特徵接入遵循 WAFT（BSD-3-Clause）。完整授權與來源列在
[`third_party/feature_repair`](../third_party/feature_repair)，並隨 wheel 一起分發。
