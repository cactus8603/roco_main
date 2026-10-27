# 程式使用方式

目前公開的可重用介面包含：凍結 WAFT 的特徵提取與回注、便宜 target 製作、昂貴 target 製作。
修復模型的完整訓練與推論工具尚未整理為公開套件。

## 安裝與外部依賴

專案使用 Python 3.10 以上。在獨立環境安裝：

```bash
python -m pip install -e ".[dev]"
```

實際製作 target 另需 PyTorch、NumPy，以及[官方 WAFT](https://github.com/princeton-vl/WAFT)
完整程式、相應依賴、權重與資料。需讓官方 WAFT modules 可被 Python 匯入。
僅載入 backbone 不足以執行完整特徵回注與光流估計。

使用的 checkpoint SHA-256：

```text
1b49d1fc4513c6fa34a8de8f791dd8735d366d4b25d8dea47626a79056edd402
```

呼叫者負責載入並核對 checkpoint、選擇裝置與管理資源。
下列介面不會自行下載模型、資料或啟動背景工作。

## 建立凍結 WAFT 介面

```python
import torch
from utils.utils import Padder, coords_grid, bilinear_sampler  # official WAFT
from stablebridge.waft_targets_v1 import FrozenWAFTBackend, cheap_target, expensive_target

# core: already loaded complete official WAFT with the verified checkpoint.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
core.eval().requires_grad_(False)
backend = FrozenWAFTBackend(core, Padder, coords_grid, bilinear_sampler)
```

檔名與設定中的版本後綴是固定 API 相容性識別；閱讀流程以「便宜 target」「昂貴 target」區分即可。

## 製作兩種 target

RGB 使用 CPU FP32、範圍 0～255、形狀 `[1,2,3,H,W]`；
GT 為 `[1,2,H,W]`，有效／非遮擋 mask 為 bool `[1,H,W]`。

```python
# teacher_rgb: input pair for KITTI/Clean; complete paired Clean for Final.
teacher_original = backend.encode_pair(teacher_rgb)
cheap = cheap_target(
    backend, teacher_rgb, gt_cpu, valid_cpu, visible_cpu,
    original=teacher_original,
)

# input_rgb always uses the actual input render, including Final.
input_original = backend.encode_pair(input_rgb)
device = input_original.features.device
with torch.enable_grad():
    expensive = expensive_target(
        backend, input_original,
        gt_cpu.to(device), valid_cpu.to(device),
    )
```

`input_original.features` 是修復模型的原始輸入；
`cheap["teacher_features"]` 或 `expensive["teacher_features"]` 是選定的監督目標。
保存時將特徵 `detach().cpu()`，並記錄影像配對、輸入／teacher render、
RGB／GT／非遮擋來源雜湊、checkpoint 雜湊、原圖尺寸與填補資訊。

## 修復後接回 WAFT

下列 `repaired_features` 代表修復模型對原始完整雙幀特徵的輸出。
此段只展示公開的 WAFT 回注介面，修復模型需另外建立與訓練。

```python
with torch.no_grad():
    result = backend.solve(
        repaired_features,
        input_original.native_hw,
        input_original.padding,
    )
flow = result["flow"][-1]  # full five updates; original image resolution
```

## 設定與測試

- [Target 配方設定](../configs/stablebridge/waft_target_methods_v1.json)
- [Target 與凍結 WAFT 介面](../src/stablebridge/waft_targets_v1.py)
- [CPU 測試](../tests/stablebridge/test_waft_targets_v1.py)

測試環境另需 `pytest`、PyTorch 與 NumPy：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
tests/stablebridge/test_waft_targets_v1.py
```

這些 tests 使用 CPU 合成模型檢查數值契約，不需要官方 checkpoint，
也不代表已重跑原生資料的 GPU 成效評估。
