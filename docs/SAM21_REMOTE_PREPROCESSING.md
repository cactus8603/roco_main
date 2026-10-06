# 在另一台伺服器重跑 SAM 2.1 前處理

這份文件用來在另一台 Linux/NVIDIA 伺服器重現 StableBridge 的
SAM 2.1 Hiera-Large 前處理，先比較 automatic-mask 參數對 key-object
產率的影響，再決定是否跑完整 Sintel／KITTI inventory。

目前本機 baseline 的 full segmentation 正常，但 U²Flow key-object
產率異常低：Sintel 約 0.2% frame 非空，KITTI 目前為 0%。因此遠端第一階段
應先做 100-frame quality sweep，不要直接跑完整 20,228 張。

## 1. 固定版本與硬體

- Linux、Python 3.10 以上。
- PyTorch 2.5.1 以上及相容 torchvision/CUDA。
- SAM 2 source commit：`2b90b9f5ceec907a1c18123530e92e794ad901a4`
- 模型：SAM 2.1 Hiera-Large。
- checkpoint SHA-256：
  `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`
- baseline 在 RTX 3090、`points_per_batch=32` 時增加約 3.7 GiB 顯存；
  quality sweep 建議啟動前至少保留 12 GiB，crop/dense 版本可能更高。
- 每一組 100-frame sweep 建議準備 1–3 GiB 暫存；完整正式產物保守預留
  5–8 GiB。

官方 SAM2.1 checkpoint：

`https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt`

key-object 規則沿用 UnSAMFlow `sam_inference.py`：bbox 尺寸、bbox fill
至少 0.5，且至少與六個 SAM masks 相交：

`https://github.com/facebookresearch/UnSAMFlow/blob/main/sam_inference.py`

## 2. 取得已整理的 GitHub 分支

SAM2.1 前處理、quality report、joint action-bank trainer 與 Sintel no-HG
主線都在 `sam21-joint-nohg-mainline`：

```bash
git clone --branch sam21-joint-nohg-mainline --single-branch \
  https://github.com/cactus8603/roco_main.git /path/to/roco_main
cd /path/to/roco_main
git rev-parse --abbrev-ref HEAD
```

遠端至少要確認下列檔案存在：

```text
scripts/generate_sam21_training_masks.py
scripts/report_sam21_mask_quality.py
src/stablebridge/physical_repair/sam_semantic_smoothness.py
src/stablebridge/physical_repair/u2flow_training_data.py
configs/stablebridge/u0_sintel_u2flow_data_v1.json
configs/stablebridge/kitti_u2flow_data_v1.json
```

## 3. 建立環境與下載 SAM2.1

以下路徑都可更換，但後續命令必須一致：

```bash
REMOTE_REPO=/path/to/roco_main
REMOTE_VENV=/path/to/venvs/roco-sam21
SAM2_REPO=/path/to/vendor/sam2
SAM21_CKPT=/path/to/checkpoints/sam2.1_hiera_large.pt

python3.10 -m venv "${REMOTE_VENV}"
"${REMOTE_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel

# 先依該 server 的 CUDA 安裝相容的 torch/torchvision；本機參考版本是：
# torch 2.5.1+cu124、torchvision 0.20.1+cu124。
"${REMOTE_VENV}/bin/python" -m pip install -e "${REMOTE_REPO}[training]"

git clone https://github.com/facebookresearch/sam2.git "${SAM2_REPO}"
git -C "${SAM2_REPO}" checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
SAM2_BUILD_CUDA=0 "${REMOTE_VENV}/bin/python" -m pip install -e "${SAM2_REPO}"

mkdir -p "$(dirname "${SAM21_CKPT}")"
curl --fail --location \
  --output "${SAM21_CKPT}" \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
echo '2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318  '"${SAM21_CKPT}" \
  | sha256sum --check
```

環境自檢：

```bash
"${REMOTE_VENV}/bin/python" - <<'PY'
import torch, torchvision
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('cuda runtime:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## 4. 設定資料路徑

### Sintel

修改：

```text
configs/stablebridge/u0_sintel_u2flow_data_v1.json
```

將最上層 `root` 改成遠端 Sintel extracted root。該目錄應含
`training/clean`、`training/final`、`training/flow`、`training/invalid`、
`training/occlusions`。四個 frozen roles 合計應 inventory 到 2,128 張 RGB。

### KITTI

修改：

```text
configs/stablebridge/kitti_u2flow_data_v1.json
```

將 `storage_root` 改成遠端 KITTI root，並維持 JSON 中各 source 的
`extracted_relative_path` 目錄結構。正式版本只用 `training` RGB，預期
18,100 張；不要加入 benchmark testing 或 multiview testing。

先驗證檔案與 CLI：

```bash
REMOTE_REPO=/path/to/roco_main
REMOTE_VENV=/path/to/venvs/roco-sam21

cd "${REMOTE_REPO}"
PYTHONPATH="${REMOTE_REPO}/src" \
  "${REMOTE_VENV}/bin/python" scripts/generate_sam21_training_masks.py --help
```

## 5. 先跑 100-frame quality sweep

每組參數必須使用不同的 `--output-root`。產生器會跳過既有檔案，所以若
不同參數共用 output root，會得到混合且不可解釋的結果。

先設定共同變數：

```bash
REMOTE_REPO=/path/to/roco_main
REMOTE_VENV=/path/to/venvs/roco-sam21
SAM2_REPO=/path/to/vendor/sam2
SAM21_CKPT=/path/to/checkpoints/sam2.1_hiera_large.pt
SWEEP_ROOT=/path/to/sam21_quality_sweep
SINTEL_CONFIG="${REMOTE_REPO}/configs/stablebridge/u0_sintel_u2flow_data_v1.json"

cd "${REMOTE_REPO}"
```

### A. 本機 baseline

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="${REMOTE_REPO}/src:${SAM2_REPO}" \
"${REMOTE_VENV}/bin/python" scripts/generate_sam21_training_masks.py \
  --dataset sintel \
  --data-config "${SINTEL_CONFIG}" \
  --output-root "${SWEEP_ROOT}/sintel_baseline" \
  --sam2-source "${SAM2_REPO}" \
  --checkpoint "${SAM21_CKPT}" \
  --points-per-side 32 \
  --points-per-batch 32 \
  --pred-iou-thresh 0.80 \
  --stability-score-thresh 0.95 \
  --use-m2m \
  --maximum-frames 100
```

### B. 較密 point grid

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="${REMOTE_REPO}/src:${SAM2_REPO}" \
"${REMOTE_VENV}/bin/python" scripts/generate_sam21_training_masks.py \
  --dataset sintel \
  --data-config "${SINTEL_CONFIG}" \
  --output-root "${SWEEP_ROOT}/sintel_points64" \
  --sam2-source "${SAM2_REPO}" \
  --checkpoint "${SAM21_CKPT}" \
  --points-per-side 64 \
  --points-per-batch 32 \
  --pred-iou-thresh 0.80 \
  --stability-score-thresh 0.95 \
  --use-m2m \
  --maximum-frames 100
```

### C. 較密 grid 加較寬鬆品質門檻

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="${REMOTE_REPO}/src:${SAM2_REPO}" \
"${REMOTE_VENV}/bin/python" scripts/generate_sam21_training_masks.py \
  --dataset sintel \
  --data-config "${SINTEL_CONFIG}" \
  --output-root "${SWEEP_ROOT}/sintel_points64_relaxed" \
  --sam2-source "${SAM2_REPO}" \
  --checkpoint "${SAM21_CKPT}" \
  --points-per-side 64 \
  --points-per-batch 32 \
  --pred-iou-thresh 0.75 \
  --stability-score-thresh 0.90 \
  --use-m2m \
  --maximum-frames 100
```

### D. 一層 crop 的多尺度候選

這組通常最慢、顯存需求也可能最高，但最可能增加重疊候選：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="${REMOTE_REPO}/src:${SAM2_REPO}" \
"${REMOTE_VENV}/bin/python" scripts/generate_sam21_training_masks.py \
  --dataset sintel \
  --data-config "${SINTEL_CONFIG}" \
  --output-root "${SWEEP_ROOT}/sintel_crop1" \
  --sam2-source "${SAM2_REPO}" \
  --checkpoint "${SAM21_CKPT}" \
  --points-per-side 32 \
  --points-per-batch 32 \
  --pred-iou-thresh 0.80 \
  --stability-score-thresh 0.95 \
  --crop-n-layers 1 \
  --crop-n-points-downscale-factor 2 \
  --use-m2m \
  --maximum-frames 100
```

如果有四張卡，A–D 可分別指定 GPU 0–3 同時執行。不要在同一張卡上疊多個
SAM2.1 Large process。

## 6. 產生 quality report

對每一組輸出執行：

```bash
for NAME in sintel_baseline sintel_points64 sintel_points64_relaxed sintel_crop1; do
  PYTHONPATH="${REMOTE_REPO}/src" \
  "${REMOTE_VENV}/bin/python" scripts/report_sam21_mask_quality.py \
    --output-root "${SWEEP_ROOT}/${NAME}" \
    --expected-frames 100 \
    --json-output "${SWEEP_ROOT}/${NAME}_quality.json"
done
```

重點比較：

- `invalid_*_files` 必須為空。
- `paired_files` 必須等於 100。
- `region_count.median`、`region_count.p95` 不應因放寬門檻而失控。
- `key_objects.nonempty_frame_fraction` 是主要指標。
- `key_objects.total` 是 exact key-object 品質指標；愈多愈好，但 v2 訓練會在
  exact masks 不足時從 full segmentation 依 crop 選物件。最終是否可訓練，
  以 training launcher 的 100-object semantic preflight 為準。仍需抽看物件
  是否完整、不是碎片。
- baseline 若仍接近 0%，但 crop1／points64 顯著提高，正式版本應採提高後的
  最保守有效設定。

建議把四份 `*_quality.json`、每組執行時間、GPU 型號、峰值顯存，以及每組
10 張 representative full-seg PNG／key-object NPZ 傳回來。報告與少量樣本可打包：

```bash
tar -czf "${SWEEP_ROOT}/sam21_quality_reports.tgz" \
  "${SWEEP_ROOT}"/*_quality.json
```

## 7. 選定參數後跑完整資料

同一個 dataset 的所有 shards 必須使用完全相同的參數、相同 output root，且
能看到同一個共享檔案系統。以下以兩張 GPU 跑 Sintel 為例；將 `SELECTED_ARGS`
換成 sweep 選出的參數：

```bash
PROD_ROOT=/path/to/sam21_hiera_l_sintel
SELECTED_ARGS='--points-per-side 64 --points-per-batch 32 --pred-iou-thresh 0.75 --stability-score-thresh 0.90 --use-m2m'

for SHARD in 0 1; do
  CUDA_VISIBLE_DEVICES="${SHARD}" \
  PYTHONPATH="${REMOTE_REPO}/src:${SAM2_REPO}" \
  "${REMOTE_VENV}/bin/python" scripts/generate_sam21_training_masks.py \
    --dataset sintel \
    --data-config "${SINTEL_CONFIG}" \
    --output-root "${PROD_ROOT}" \
    --sam2-source "${SAM2_REPO}" \
    --checkpoint "${SAM21_CKPT}" \
    --shard-count 2 \
    --shard-index "${SHARD}" \
    ${SELECTED_ARGS} \
    >"${PROD_ROOT}.shard-${SHARD}.stdout.log" \
    2>"${PROD_ROOT}.shard-${SHARD}.stderr.log" &
done
wait
```

KITTI 做法相同，改成：

```text
--dataset kitti
--data-config /path/to/configs/stablebridge/kitti_u2flow_data_v1.json
--output-root /path/to/sam21_hiera_l_kitti_training
```

`--kitti-splits training` 是預設值；不要加入 `testing`。shard 數量可等於可用
GPU 數量，例如五張 GPU 就用 `--shard-count 5`、index 0–4。

## 8. 續跑、完整性與搬回本機

- 中斷後以完全相同命令重跑；既有且尺寸一致的 full/key pair 會被跳過。
- 改任何 SAM 參數時必須換新 output root。
- 所有 shard 完成後才會出現：

```text
OUTPUT_ROOT/manifest.json
OUTPUT_ROOT/full_seg/manifest.json
```

- Sintel 的 `frame_count` 應為 2,128；KITTI 應為 18,100。
- 最終再執行一次 `report_sam21_mask_quality.py`，確認 full/key 一一配對且無
  invalid files。

搬回本機：

```bash
rsync -a "${REMOTE_HOST}:${PROD_ROOT}/" \
  /ssd1/cactus8603/roco_main/.runtime_tmp/sam21_remote_candidate/
```

manifest 含遠端絕對路徑。若遠端與本機資料 root 不同，搬回後應在本機以相同
shard count 與相同 SAM 參數各重跑一次產生器；它會跳過已存在的 masks，只重算
hash 並發布符合本機路徑的 manifests。不要手動只改 JSON 路徑，否則 lineage
與 config SHA 會不一致。

## 9. 在遠端啟動 SAM2.1／no-HG 聯合主線

不要手改主線 JSON 中的本機絕對路徑。完整環境、SEA-RAFT、U0ft checkpoint、
preflight、正式啟動及續跑指令請接著依照
[`SAM21_REMOTE_TRAINING.md`](SAM21_REMOTE_TRAINING.md)。
