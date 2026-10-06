# 在另一台伺服器訓練 SAM2.1 聯合主線

這份文件的目標是讓另一台 Linux/NVIDIA server 能啟動目前的完整模型，而不只是
產生 SAM masks。固定訓練契約如下：

- SEA-RAFT Spring-M backbone。
- recurrent uncertainty observer + bounded flow refiner。
- `joint_decoupled` U2 schedule。
- 每個 epoch 以 homogeneous batches 平衡交錯 native + 9 個 actions。
- SAM2.1 Hiera-Large key-object semantic augmentation，epoch 10 開始。
- Sintel homography loss 關閉。
- 20 epochs、batch size 1、gradient accumulation 4。
- 以已完成的 U0ft integrated-v2 `best.pt` warm start；不匯入舊 optimizer/epoch。

目前正式訓練 domain 是 Sintel。KITTI SAM2.1 masks 可以同時準備供後續
domain-shift／calibration 實驗，但不會被這個 Sintel mainline launcher 讀取。

## 1. 取得分支與建立環境

```bash
REPO=/path/to/roco_main
VENV=/path/to/venvs/roco-sam21-train

git clone --branch sam21-joint-nohg-mainline --single-branch \
  https://github.com/cactus8603/roco_main.git "${REPO}"

python3.10 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel

# 先依該 server 的 CUDA 安裝相容的 torch/torchvision，再安裝本專案。
# 本機已驗證組合：torch 2.5.1+cu124、torchvision 0.20.1+cu124。
"${VENV}/bin/python" -m pip install -e "${REPO}[training]"
```

若先安裝 project extra 讓 pip 自動選擇 torch，可能拿到 CPU build；因此 GPU 版 torch
應依遠端 CUDA 環境先裝好。

## 2. 準備 pinned SEA-RAFT

```bash
SEA_ROOT=/path/to/vendor/SEA-RAFT
SEA_CKPT=/path/to/checkpoints/sea-raft-spring-M/model.safetensors
TORCH_HOME_DIR=/path/to/torch-cache

git clone https://github.com/princeton-vl/SEA-RAFT.git "${SEA_ROOT}"
git -C "${SEA_ROOT}" checkout 9137517ba24e628442aec097d3afe71d03503b75

mkdir -p "$(dirname "${SEA_CKPT}")"
"${VENV}/bin/python" - <<PY
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
    filename="model.safetensors",
    local_dir="$(dirname "${SEA_CKPT}")",
)
PY

mkdir -p "${TORCH_HOME_DIR}/hub/checkpoints"
curl --fail --location \
  --output "${TORCH_HOME_DIR}/hub/checkpoints/resnet34-b627a593.pth" \
  https://download.pytorch.org/models/resnet34-b627a593.pth

echo 'cb8cfbf14c5e0f6734b64add383708b7ff68cc6089a0007c67165d4761346102  '"${SEA_CKPT}" \
  | sha256sum --check
echo 'b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f  '"${TORCH_HOME_DIR}/hub/checkpoints/resnet34-b627a593.pth" \
  | sha256sum --check
```

launcher 也會重新檢查 SEA-RAFT Git commit、working tree cleanliness、7 個 critical
source hashes、Spring-M config、兩份權重的大小與 SHA-256。

## 3. 同步 U0ft 初始化 lineage

GitHub 不包含大型訓練產物。請從原 server 同步整個最小 lineage 目錄中的兩個檔案：

```text
experiments/U2_sintel_searaft_uncertainty_refinement_u0ft_v2/seed11/
├── best.pt        238,305,806 bytes
└── metrics.jsonl  必須包含 training_completed event
```

例如：

```bash
INIT_DIR=/path/to/checkpoints/U2_sintel_u0ft_v2_seed11
mkdir -p "${INIT_DIR}"
rsync -av user@original-server:/ssd1/cactus8603/roco_main/experiments/U2_sintel_searaft_uncertainty_refinement_u0ft_v2/seed11/best.pt "${INIT_DIR}/"
rsync -av user@original-server:/ssd1/cactus8603/roco_main/experiments/U2_sintel_searaft_uncertainty_refinement_u0ft_v2/seed11/metrics.jsonl "${INIT_DIR}/"

echo '452eb049594d9940882b15d8b15cd166f27f4db248cf742c1f9e71c86957ab6f  '"${INIT_DIR}/best.pt" \
  | sha256sum --check
echo '68a0d43d0e6b93603377025a3db98971b10a87a834f6f1e52408f162155ab206  '"${INIT_DIR}/metrics.jsonl" \
  | sha256sum --check
```

必須保留 `metrics.jsonl` 與 `best.pt` 在同一目錄；trainer 會用其中的
`training_completed` 證明來源 stage 已完成。

## 4. 準備 Sintel 與 SAM2.1 masks

```text
SINTEL_ROOT/
└── training/
    ├── clean/
    ├── final/
    ├── flow/
    ├── invalid/
    └── occlusions/

SAM_MASK_ROOT/
├── full_seg/
│   ├── manifest.json
│   └── training/...
├── key_objects/
│   └── training/...
└── manifest.json
```

正式 mask manifest 必須覆蓋完整 2,128 張 Sintel RGB、`inventory_truncated=false`。
目前 v2 會優先使用 exact key objects；不足時，會依每個訓練 crop 從 full
segmentation 選出可用物件，因此 exact key-object 數量只作為品質診斷，不再是啟動門檻。
launcher 會用真實訓練資料確認至少能填滿 100-object semantic cache。若 masks
是從別的檔案系統搬來，請依
[`SAM21_REMOTE_PREPROCESSING.md`](SAM21_REMOTE_PREPROCESSING.md) 的方式，以原參數
重跑 generator 發布新的絕對路徑 manifest；不要直接手改 JSON。

## 5. 先做 prepare-only preflight

不需要手改 repository 內含本機路徑的 JSON。launcher 會在 `RUN_DIR/portable_launch/`
產生遠端專用的 Sintel config、training config 與完整 preflight report。

```bash
SINTEL_ROOT=/path/to/Sintel/extracted
SAM_MASK_ROOT=/path/to/sam21_hiera_l_sintel
INIT_CKPT="${INIT_DIR}/best.pt"
RUN_DIR=/path/to/runs/U2_sintel_joint_sam21_nohg_seed11

cd "${REPO}"
PYTHONPATH="${REPO}/src" \
TORCH_HOME="${TORCH_HOME_DIR}" \
"${VENV}/bin/python" scripts/train_sam21_joint_nohg.py \
  --sintel-root "${SINTEL_ROOT}" \
  --sam-mask-root "${SAM_MASK_ROOT}" \
  --sea-raft-root "${SEA_ROOT}" \
  --sea-raft-checkpoint "${SEA_CKPT}" \
  --initialization-checkpoint "${INIT_CKPT}" \
  --torch-home "${TORCH_HOME_DIR}" \
  --run-dir "${RUN_DIR}" \
  --prepare-only
```

成功時會輸出 JSON，確認其中：

```text
training_contract.u2_schedule           = joint_decoupled
training_contract.action_schedule       = balanced_batches
training_contract.sam_semantic_enabled  = true
training_contract.fallback_to_full_segmentation = true
training_contract.homography_enabled    = false
sam21.frame_count                        = 2128
sam_semantic_preflight.state             = passed
sam_semantic_preflight.usable_objects    = 100
```

`sam21.key_object_count` 仍會列在報告中，但 v2 不要求它大於等於 100；
真正的啟動條件是上述 crop-aware semantic preflight 能填滿 cache。

## 6. 正式啟動與續跑

使用和 prepare-only 完全相同的參數，移除 `--prepare-only`：

```bash
cd "${REPO}"
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="${REPO}/src" \
TORCH_HOME="${TORCH_HOME_DIR}" \
"${VENV}/bin/python" scripts/train_sam21_joint_nohg.py \
  --sintel-root "${SINTEL_ROOT}" \
  --sam-mask-root "${SAM_MASK_ROOT}" \
  --sea-raft-root "${SEA_ROOT}" \
  --sea-raft-checkpoint "${SEA_CKPT}" \
  --initialization-checkpoint "${INIT_CKPT}" \
  --torch-home "${TORCH_HOME_DIR}" \
  --run-dir "${RUN_DIR}" \
  --device cuda \
  --resume auto
```

在共享 GPU 上不希望程式主動填高 allocator cache 時，加上 `--no-reserve-vram`。
中斷後以同一條命令重跑；`--resume auto` 只讀取同一 `RUN_DIR/latest.pt`，而
launcher 會拒絕悄悄改變已生成的 config。

主要輸出：

```text
RUN_DIR/
├── portable_launch/preflight.json
├── portable_launch/sintel_data.json
├── portable_launch/training_config.json
├── resolved_config.json
├── metrics.jsonl
├── action_shadow_cases.jsonl
├── latest.pt
└── best.pt
```

可用以下方式查看進度：

```bash
tail -f "${RUN_DIR}/metrics.jsonl"
nvidia-smi
```
