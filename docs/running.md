# 從安裝到訓練與影像推論

這份指南使用同一個流程：凍結 WAFT 抽取原始雙幀特徵，修復模型學習離線 target，
再將修復特徵送回同一個凍結 WAFT，輸出完整光流。
只有修復模型的參數會在訓練時更新。

## 先用 CPU 確認安裝成功

已驗證的環境為 Python 3.10／PyTorch 2.5.1。建議使用 Python 3.10 或 3.11；
下列命令以 `python3.10` 為例，使用 3.11 時請相應替換。

```bash
git clone https://github.com/cactus8603/roco_main.git
cd roco_main
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
waft-repair doctor
waft-repair demo --output outputs/cpu-example --epochs 10
```

`demo` 會建立兩筆小型合成 features，實際訓練修復模型、保存權重，再載入權重做特徵推論。
不需要資料集、WAFT checkpoint 或 GPU。它驗證程式通路，不評估光流品質。

成功後會得到：

```text
outputs/cpu-example/
├── data/manifest.json
├── training/checkpoint.pt
├── training/history.json
├── training/result.json
└── repaired-features.pt
```

輸出目錄已存在時會拒絕覆寫；重新執行請指定新的目錄。
命令也可以寫成 `python -m waft_repair ...`。

## 完整尺寸的環境

完整影像的跨幀狀態空間運算使用 native backend，需要 Linux、NVIDIA GPU、
可用的驅動、CUDA toolkit／`nvcc`、C++ 編譯器與對應 Python 開發標頭。
CPU reference backend 適合小型驗證網格，無法直接處理完整 KITTI／Sintel features。

建議另建 GPU 環境。下列版本對應現行原生實作所使用的依賴：

```bash
python3.10 -m venv .venv-cuda
source .venv-cuda/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install packaging ninja wheel einops==0.8.0 transformers==4.44.2 tokenizers==0.19.1
python -m pip install -e ".[waft]"
python -m pip install --no-build-isolation -r requirements/repair-cuda.txt
waft-repair doctor
python -c "from waft_repair import FeatureRepair; FeatureRepair(backend='native'); print('Native modules ready')"
```

`mamba-ssm==2.2.2` 與 `causal-conv1d==1.2.1` 是此實作沿用的版本，需有可用的 CUDA extension；
程式不會在 native 載入失敗時暗中換成 reference。
不同 GPU／CUDA 環境可能需要在本機編譯 extension。
[Mamba 官方安裝說明](https://github.com/state-spaces/mamba/blob/v2.2.2/README.md)提供編譯需求；
其中建議的較新 causal-conv1d 版本不等於本專案已核對的原生環境。

## 準備官方凍結 WAFT

下載固定官方原碼，不需初始化其他模型的 submodules：

```bash
git clone https://github.com/princeton-vl/WAFT.git external/WAFT
git -C external/WAFT checkout b152ff1cad1af8c185ee7b141997c48ff3334c87
```

從[官方 checkpoint 下載資料夾](https://drive.google.com/drive/folders/1joBWKGoH2RUdCgcge8Tz2osOHcQUX5m_?usp=sharing)
取得本流程使用的 **WAFT a2／DAv2／vits、Chairs+Things source-only `zero-shot.pth`**，
放在 `checkpoints/waft/zero-shot.pth`。下載資料夾內有其他模型，程式會以完整檔案 SHA-256 核對：

```text
1b49d1fc4513c6fa34a8de8f791dd8735d366d4b25d8dea47626a79056edd402
```

```bash
waft-repair doctor \
  --waft-source external/WAFT \
  --waft-checkpoint checkpoints/waft/zero-shot.pth
```

這一步在 CPU 嚴格載入全部 508 個 WAFT tensors，核對來源與權重後保持凍結，不執行光流推論。
載入不會另外下載 DAv2 或 ViT 權重。
資料集與權重未包含在此 repository；本機現有資料的位置見 [資料清單](data-inventory.md)。

## 準備訓練資料

可以從原始影像製作新 features，也可以使用已有 features。兩條路最後都得到 `manifest.json`。

### 從 KITTI／Sintel 影像開始

先將 GT 轉為統一 NPZ。KITTI 範例：

```bash
waft-repair convert-gt --format kitti \
  --flow data/kitti/flow_occ/000000_10.png \
  --visibility data/kitti/flow_noc/000000_10.png \
  --output data/street-flow.npz
```

Sintel 範例：

```bash
waft-repair convert-gt --format sintel \
  --flow data/sintel/flow/alley_1/frame_0001.flo \
  --visibility data/sintel/occlusions/alley_1/frame_0001.png \
  --output data/street-flow.npz
```

兩個命令是不同資料集的替代範例，輸出請各自使用新的檔名。
NPZ 包含 `flow: float32[H,W,2]`、`valid: bool[H,W]`，以及便宜法需要的 `visible: bool[H,W]`。
KITTI 使用完整 GT validity；Sintel 使用 finite 且兩個 flow 分量絕對值皆小於 1000 的有效支援。
`visible` 只限制 RGB 重建，不縮小光流評分分母。

編輯 [影像清單範例](../examples/repair/images.json)，將每筆 `image1`、`image2`、`flow`
改成自己的檔案。相對路徑以 JSON 所在目錄為基準，`pair_id` 必須唯一。

便宜版：

```bash
waft-repair prepare --spec examples/repair/images.json \
  --target cheap --output outputs/cheap-features \
  --waft-source external/WAFT --waft-checkpoint checkpoints/waft/zero-shot.pth \
  --device cuda:0
```

昂貴版：

```bash
waft-repair prepare --spec examples/repair/images.json \
  --target expensive --output outputs/expensive-features \
  --waft-source external/WAFT --waft-checkpoint checkpoints/waft/zero-shot.pth \
  --device cuda:0
```

便宜法額外編碼重建影像；昂貴法每筆固定做 512 次完整五步 WAFT 特徵最佳化，成本高很多。
昂貴法不需要 `visible`，可以在 GT 轉換時省略 `--visibility`。
兩法都不更新 WAFT 權重，也不會在這個步驟訓練修復模型。

Sintel Final 便宜 target 使用 [Final 影像清單範例](../examples/repair/final-images.json)：
`image1/image2` 是原 Final，`teacher_image1/teacher_image2` 是完整配對 Clean，
`target_source_pair_id` 識別對應 Clean pair。必須使用相同配對、GT、有效像素與幾何。
昂貴 target 始終從原輸入 features 開始。

### 使用已有 input／target features

已有的 `.pt` 或 `.npz` features 可以依 [特徵清單範例](../examples/repair/features.json)
填寫路徑與 metadata，再核驗成正式清單：

```bash
waft-repair manifest --spec examples/repair/features.json --output data/train.json
```

清單格式與封包欄位見 [程式介面](usage.md)。核驗包含影像對、shape、padding、
WAFT checkpoint、target 版本、檔案雜湊與 Final 配對來源。

目前本機既有的可見區 target 描述檔，及已接受的昂貴 target 索引，可分別直接匯入：

```bash
waft-repair import-cache --directory "$CHEAP_PAIR_DIRECTORY" --output data/cheap-train.json
waft-repair import-expensive --entries "$EXPENSIVE_ENTRY_DIRECTORY" --output data/expensive-train.json
```

先依 [資料清單](data-inventory.md)設定上述兩個路徑；便宜目錄包含 `pair_*.json`，
昂貴目錄包含接受索引 `entry.json`。昂貴匯入只使用已接受且封包與審查雜湊吻合的條目，
不把未索引檔案算成訓練資料。
不同方法的 target 分別建立清單，這個匯入功能不替每筆資料挑選 EPE 最好的 target。

## 訓練修復模型

選擇一份已核驗清單，例如上面準備的便宜版本：

```bash
waft-repair train --manifest outputs/cheap-features/manifest.json \
  --output outputs/repair --epochs 10 --device cuda:0 --backend native
```

若改用昂貴 target，只需改成相應 manifest 與新的輸出目錄。
每筆資料一次更新，每個 epoch 使用全部資料一次，順序由 seed 與 epoch 決定。
固定 AdamW：learning rate 0.001、betas (0.9,0.999)、epsilon 1e-12、weight decay 0、
FP64 global clip 1，無 warmup。

訓練 loss 是完整雙幀及全部 padding 的 `mean(((repaired-target)/sigma)^2)`。
σ 只由原始 features 的 native centers 計算，模型輸出是無界 `original + raw_delta`。
這個步驟不執行 WAFT，也不直接使用 GT 光流 loss。

輸出：

| 檔案 | 用途 |
|---|---|
| `checkpoint.pt` | 修復模型、optimizer、訓練 cursor、RNG 與資料綁定 |
| `history.json` | 每次更新前的在線 feature loss |
| `progress.json` | 訓練進度 |
| `result.json` | 固定 checkpoint 的完整資料 feature MSE 與 identity 比較 |

續訓需使用原清單、設定與輸出目錄：

```bash
waft-repair train --manifest outputs/cheap-features/manifest.json \
  --output outputs/repair --epochs 10 --device cuda:0 --backend native \
  --resume outputs/repair/checkpoint.pt
```

程式在 epoch 邊界保存 checkpoint，也可用 `--checkpoint-every` 指定更新間隔。
CPU reference 以單執行緒核對續訓的權重與歷程一致；CUDA 恢復模型、optimizer 與 RNG，
未宣稱跨硬體或不同 CUDA kernel 的逐位元重播。

## 從兩張影像輸出光流

```bash
waft-repair infer-images \
  --checkpoint outputs/repair/checkpoint.pt \
  --image1 data/street-first.png --image2 data/street-second.png \
  --waft-source external/WAFT --waft-checkpoint checkpoints/waft/zero-shot.pth \
  --output outputs/prediction.npz --device cuda:0 --backend native
```

這個命令不讀取訓練 manifest、GT、target features 或配對 Clean。
它用輸入影像取得原始 features，經修復後，在 hidden 初始化之前放回同一個凍結 WAFT，
從零光流完成全部 5 次更新。

NPZ 的 `flow` 是 `float32[H,W,2]` 的原尺寸向量光流；
`flow_steps` 保存全部 5 次光流，`info_steps` 保存全部 5 次附加讀出。
可用 NumPy 查看形狀：

```python
import numpy as np
with np.load("outputs/prediction.npz", allow_pickle=False) as result:
    print(result["flow"].shape)
```

若已經有 input feature 封包，只做修復可使用：

```bash
waft-repair infer-features --checkpoint outputs/repair/checkpoint.pt \
  --input outputs/cheap-features/pair-000000-input.pt \
  --output outputs/repaired-features.pt --device cuda:0 --backend native
```

## 驗證範圍

CPU 測試涵蓋模型與現行實作的權重／forward／gradient 一致、GT 轉換、
完整 grid loss、兩種快取匯入、續訓、刪除 target 後推論、Final 配對與拒絕覆寫。

```bash
python -m pytest -q tests/waft_repair
```

另已實際以完整官方 checkpoint 執行小尺寸合成 RGB 的資料準備、一次修復更新與影像推論，
核對五步官方輸出與回注介面一致，未初始化 CUDA。
這是可執行性及移植驗證；未將新公開 CLI 當作完整 KITTI／Sintel 的 GPU 重跑或修復模型成效證明。
科學結果與限制見 [結果文件](results.md)。
