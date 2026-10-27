# WAFT target features: fixed cheap and expensive recipes

便宜 target 已依使用者於 **2026-10-27** 的選擇固定為 **非遮擋區完整重建 RGB**
(`waft-visible-pull-rgb-v1`, historical arm `visible_pull_1`)。
昂貴 target 保留 **逐筆 dense feature delta、固定 512 次 AdamW 更新**
(`waft-dense-final-epe-512-v1`)。兩者都是使用 GT 製作的離線訓練 target。

固定機器可讀設定：[waft_target_methods_v1.json](../../configs/stablebridge/waft_target_methods_v1.json)。
可重用實作：[waft_targets_v1.py](../../src/stablebridge/waft_targets_v1.py)。

## 共同 WAFT 與特徵介面

兩法使用相同 Chairs+Things source-only WAFT-a2：DAv2 encoder、vits iterative module，
全部 **508 tensors / 56,932,619 parameters** 保持 FP32、eval、frozen、無參數梯度。
官方 checkpoint 不隨 repository 上傳，預期 SHA-256：

```text
1b49d1fc4513c6fa34a8de8f791dd8735d366d4b25d8dea47626a79056edd402
```

特徵使用官方 `normalize_image -> Padder(factor=112) -> encoder + fnet[0] -> fmap_conv`，
形狀為 `[1,2,64,Hpad/2,Wpad/2]`。完整雙幀 padded features 在 **hidden 初始化之前、
第1步之前**注入；每次由零光流開始，執行全部5次更新和5次原生高解析度讀出。
比較第5張 native flow 的 vector EPE。

GT 評分支援維持原始 finite-valid mask，包含遮擋與落在影像外的對應點；
RGB 重建使用的可見／界內 mask 不會縮小 EPE 分母。

## 便宜版：visible pull RGB，alpha=1

1. 取 forward GT `g(x)`、原始 GT validity，以及非遮擋標註。
2. 建立 `M = valid & finite(g) & inbounds(x+g(x)) & nonoccluded`。
   KITTI 使用 `flow_occ` 的 GT flow，`flow_noc` 的 validity 作非遮擋支援；
   Sintel 使用 `occlusions == 0`。不以未知遮擋代替可見。
3. 在 M 上以 CPU FP64 bilinear 計算 `I1*(x)=I2(x+g(x))`，最後轉一次 FP32。
   M 外保留原始 I1；I2 全部保留，零 flow 也是有效對應。
4. 以原 WAFT prefix 編碼 I1*。若原始 teacher features 已存，重用同一 teacher pair 的
   完整 I2 feature，因此增量只需 **一次 changed-image encoder**。
5. 保留完整雙幀 padded target；不再投影到 ±0.25σ。RGB 未直接更改的位置，其編碼
   context/padding 仍可能變動，不能用 RGB mask 再裁切 target features。

### 各資料集 RGB 來源

| 修復模型的原始輸入 | 便宜 target 使用的 RGB pair | 昂貴 target 的 F0 |
|---|---|---|
| KITTI | 同一 KITTI pair | 原 KITTI features |
| Sintel Clean | 同一 Clean pair | 原 Clean features |
| Sintel Final | **對應 Clean 的兩張 RGB** | **原 Final features** |

Final 的修復輸入仍是原 Final features；便宜 target 的兩張 features 和全部 padding
皆來自對應 Clean 重建 pair，必須核對 pair、GT、validity 與 geometry。
不要把 Clean 的第一張 target 與 Final 的第二張 feature 拼在一起。

## 昂貴版：dense final-EPE，固定512更新

每筆獨立建立 `delta=0`，以原始 F0 的兩幀 native feature centers 計算每 channel RMS σ；
FP64 reduction，floor 為 `0.01*lower_median(RMS)+1e-8`。
`F(delta)=F0+sigma*delta`，padding 保持原始 bytes。

每次重新初始化 hidden/zero flow，執行完整5步，**只以第5步 vector EPE 對 GT 求導**。
WAFT 本體保持 frozen，梯度經過完整 flow carry 和 sampling coordinates，只更新 delta。
AdamW：lr `.001`、betas `(.9,.999)`、eps `1e-8`、weight_decay `0`，
`foreach=False, fused=False`；全域梯度 norm 與縮放以 FP64 計算，clip1，分母加 `1e-6`，
再複製回 FP32。每次更新後硬 clamp delta 至 `[-.25,.25]`，padding delta 歸零。

固定512 optimizer updates，不 warmup、不因75%改善提早停止。比較 F0 與0..512全部候選，
以 FP64 分數選**最早 strict minimum**。最後再用 no-grad5步讀出核對選中 target。
品質旗標維持昂貴 cache 的既有規則：EPE下降至少20%、final component MSE嚴格下降、
native normalized feature MSE非零。未通過者仍保存 `qualified=false`、仍納入統計。
此品質旗標不改變便宜方法按整體平均 EPE 選擇的規則。

## 已完成的原生結果

原 checkpoint／資料／GT 固定，全部為 **TRAIN 上直接替換 target** 的結果。
它們不是 repair-model 預測結果，也不是 held-out、官方 leaderboard 或全域最優證據。

| 便宜 target 人口 | 原始 EPE | Target EPE | 降幅 | 改善／退步 |
|---|---:|---:|---:|---:|
| KITTI160 | 2.949504484 | 2.519179481 | 14.5897% | 152 / 8 |
| Sintel Clean100 | 1.561969927 | 1.145007085 | 26.6947% | 95 / 5 |
| Sintel Final100，使用 paired Clean target | 5.114360641 | 1.145007085 | 77.6119% | 99 / 1 |

Final 使用完全相同的 Clean target，已逐100pair核對 rawGT/validity/geometry，
feature-only consumer 無原 Final RGB 依賴，因此共用已驗 native5 flow 及分數。
這不是直接對 Final RGB 套重建的新測試。

昂貴版目前有兩筆完整匹配、獨立核驗的512步證據；人口不同，不能與上表做資料集均值排名：

| 昂貴 target 個案 | 原始 EPE | Target EPE | 選中步數 |
|---|---:|---:|---:|
| KITTI137 | 0.830411982 | 0.039074391 | 512 |
| Sintel Final market_5/frame_0029，原Final F0 | 18.940211480 | 4.023838412 | 483 |

Sintel100按固定 seed 分層涵蓋18個 TRAIN scenes；KITTI是固定160 TRAIN pairs。
便宜法並非逐筆都改善，完整 case 統計保留在 [cases.csv](cases.csv)，
不依 GT 對每筆挑選便宜或昂貴版本。
[results.csv](results.csv) / [evidence.json](evidence.json) 保存精簡結果及原始證據 SHA。

生成時間的歷史量測：KITTI便宜法中位約 `.3507s`，包含 CPU RGB 構造與 changed-image
encoder，**不包含 F0 抽取、存檔或G5驗證**。昂貴版須512次G5 backward/AdamW，
沒有同全人口端到端計時，不宣稱固定加速倍率。

## 可重用介面

先依 [官方 WAFT](https://github.com/princeton-vl/WAFT) 載入相同 checkpoint 並核對 SHA。
repository 不包含官方權重、資料集、完整 feature binaries 或機器上的 GPU scheduler。
`core` 必須是原始完整 WAFT，非只取 backbone。以下呼叫不自行載入或下載模型：

```python
import torch
from utils.utils import Padder, coords_grid, bilinear_sampler  # official WAFT
from stablebridge.waft_targets_v1 import (
    FrozenWAFTBackend, cheap_target, expensive_target,
)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
core.eval().requires_grad_(False)  # caller already loaded and placed official core
backend = FrozenWAFTBackend(core, Padder, coords_grid, bilinear_sampler)

# CPU FP32 RGB[1,2,3,H,W] in0..255, GT[1,2,H,W], bool valid/visible[1,H,W].
# teacher_rgb is pairedClean for Final; otherwise the input RGB pair itself.
teacher_f0 = backend.encode_pair(teacher_rgb)
cheap = cheap_target(backend, teacher_rgb, gt_cpu, valid_cpu, visible_cpu,
                     original=teacher_f0)

# The expensive target always starts from the actual input render, including Final.
input_f0 = backend.encode_pair(input_rgb)
device = input_f0.features.device
with torch.enable_grad():
    expensive = expensive_target(backend, input_f0, gt_cpu.to(device), valid_cpu.to(device))
```

顯存准入和 GPU ownership 由 caller 的既有排程負責；此介面不啟動背景 jobs。
保存時將 tensors `detach().cpu()`，並附 case/pair ID、input/teacher render、RGB/GT/visibility
來源 SHA、checkpoint SHA、方法 ID、native_hw、padding、baseline/target metrics。
輸入 F0、GT、valid、σ、native_mask 可共用；兩份 `teacher_features` 以不同 method ID 保存，
便宜 target 的 RGB generation mask 與昂貴版 native feature mask 各自保留。
對 Final 保留 input Final 與 teacher Clean 的獨立來源。

既有1752 dense-cache worker與 frozen sources 未因這次方法固定或 GitHub 發布而修改。
本次發布也不代表已生成全1752便宜 target，或已啟動修復模型訓練。

## 程式驗證範圍

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -q -p no:cacheprovider tests/stablebridge/test_waft_targets_v1.py
```

12項 portable CPU tests 包含 GT direction/visibility/OOB、完整5步與 checkpoint 梯度、
frozen core、native σ、signed-zero padding、固定512更新、最早最佳值與解析首末Adam行為，
以及公開設定／完整案例分母的一致性。
另以原凍結來源在 CPU synthetic core 核對 prefix、all5 flow/info、full-feedback feature
gradient、σ，以及512更新全部513組EPE/MSE與選中 delta bytes；四筆已保存原生影像的
重建 RGB bytes 亦一致。這些是 port 驗證，不是新 pretrained GPU EPE／速度測量。
