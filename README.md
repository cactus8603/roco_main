# RoCo — CVPR mainline

> **Project boundary.** This repository is the post-submission RoCo CVPR
> mainline. The frozen StableBridge submission is a separate predecessor; work
> added after that submission—including the action bank, selector,
> U0/U1/U2 uncertainty training, and SAM experiments—belongs to RoCo. The
> existing `stablebridge.*` Python import path is retained only for frozen
> artifact and experiment compatibility. It must not be used to infer project
> ownership or to move current RoCo work back into the StableBridge line.

RoCo 是一個針對 optical flow 與 stereo correspondence 的研究型框架。核心問題不是單純
增加更多 restoration tools，而是讓每個 action family 先提出適合的 strength／output
beta，再以共同的 downstream gain、harm、risk 與 execution cost 進行全域競爭。

目前 repository 提供的是可測試的研究架構與 fail-closed contracts，不代表任何 action、
uncertainty refiner 或 multi-action policy 已取得 production authority。

## 架構摘要

```text
native observation
    ├─ shared RGB / flow / stereo observables
    ├─ calibrated uncertainty observer (U0)
    └─ family-specific strength / beta proposals
                      │
                      ▼
        exact-control global competition
          (action, strength, beta, cost)
                      │
                      ▼
             Top-K probe / execution
                      │
                      ▼
       post-action benefit / harm verifier
              ├─ commit one candidate
              ├─ STOP at current checkpoint
              └─ retain / rollback by policy
```

主要設計原則：

- action 與 strength 共同競爭，不先不可逆地選定 action family。
- `native`、`STOP` 與 rollback 是不同語意。
- 不要求安全候選證明自己是唯一最佳；只要求通過 selection-aware risk gates 且保守效用
  為正。
- uncertainty 是 action-conditioned evidence，不是跨 action 共用的一個 hard threshold。
- U0 uncertainty 校準不會自動授權 U1 修改 flow；flow-changing provider 需要獨立 qualification。
- multi-action wiring 已接好但預設關閉；目前正式行為仍是最多 commit 一次。

## 主要模組

| 模組 | 用途 |
|---|---|
| `family_action_competition.py` | family-specific proposal 與 exact-control 全域競爭 |
| `selector_v7.py` | native abstention、selection-aware risk／cost decision |
| `uncertainty_aware_flow.py` | U0 observer、typed uncertainty roles、bounded U1 refiner |
| `uncertainty_training_contracts.py` | native-only U0 manifest、checkpoint 與 calibration receipts |
| `uncertainty_refinement_training.py` | U1 task／harm／anchor／smoothness objective |
| `integrated_uncertainty_flow.py` | recurrent hidden-state UE、detached flow feedback、affine flow transport、decoupled losses |
| `integrated_uncertainty_trainer.py` | HEAD／DUMMY／U2-MAIN trainer、checkpoint 與 EPE/AUSE/CC validation |
| `uncertainty_evaluation.py` | NLL、AURC/AUSE、AUROC、coverage 與 grouped bootstrap |
| `multi_action_orchestrator.py` | opt-in ROOT_RETRY／COMPOSED_CHAIN state machine |
| `root_retry_replay.py` | one-shot 與 ROOT_RETRY 的 outcome-only offline replay |
| `gpu_admission.py` | 空卡優先、共享卡容量門檻與雙 snapshot admission |

## 安裝

需要 Python 3.10 以上。建議使用獨立環境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

核心依賴列於 [`pyproject.toml`](pyproject.toml)。Optical-flow 主線使用 pinned
**SEA-RAFT Spring-M**；CroCo 只屬於 stereo 路徑，不是 U0 flow uncertainty 的 matcher。
真實執行仍需要外部官方程式、權重與資料集；它們不包含在 repository 中。

## 測試

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/stablebridge
```

目前 Python 3.12 完整 StableBridge regression 為：

```text
1263 passed, 19 subtests passed
```

本輪 recurrent uncertainty、spatial flow transport、U0/U1 相依測試在 SEA-RAFT Python 3.10
環境另為 `55 passed`；實際 Spring-M 權重亦已完成小尺寸及正式 384×832 train-batch
forward/backward smoke test。

測試通過表示 contracts、state transitions、receipt binding 與數學工具符合目前規格，
不等同於下游 correspondence 效益已通過 fresh scientific validation。

## U0 native-only 資料準備

U0 uncertainty observer 的資料規劃不依賴 action bank。以下命令只盤點並雜湊 native
Spring flow RGB endpoints、凍結 scene-level split／中央 crop／augmentation probes；它不會
解碼 GT、不會執行 matcher，也不會開始訓練：

```bash
PYTHONPATH=src python scripts/prepare_u0_native_data.py \
  --config configs/stablebridge/u0_native_flow_data_v1.json \
  --output experiments/U0_native_flow_data_v1/INPUT_PLAN.json
```

目前 development plan 包含 37 個 scene group、296 個 native pairs。GPU materializer 之後
必須以 plan／split／probe hashes 產生 native flow、risk 與 probe receipts，才能建立正式的
`U0NativeManifestV1`；這份 input plan 本身不具訓練、校準或部署 authority。

## U²Flow-style Sintel 開發資料

第一個 uncertainty 訓練／開發 domain 選用 **MPI-Sintel**，而不是先用 KITTI。現有機器已有
完整 Sintel training `clean/final`，且其非剛體場景較接近後續 Spring 評估；KITTI 的完整
U²Flow pipeline 另需 Raw、2012/2015 multi-view 與 SAM masks。

資料契約位於 `u2flow_training_data.py`，augmentation 位於
`u2flow_augmentations.py`：

- 基本取樣單位固定為同 scene 的相鄰二幀 `(t, t+1)`；Clean／Final 共 2,082 筆
  render-specific rows，依 scene 分成 fit／validation／calibration／evaluation。
- 三幀只保留給未來 bidirectional fusion，`fusion_context_enabled=false`，不改變目前二幀
  observer 的訓練語意；啟用時採 `[previous,current,next]`，scene 起點重複第一幀。現在已有
  disabled-by-default 的 U²Flow-style inference／validation post-process：以 `current→next`
  與 `current→previous` 兩次二幀推論，在雙向皆可信的區域逐 sample 擬合 Tiny CNN，只替換
  forward 不可信而 backward 可信的像素。它不進 training loss，未加入任何正式 config，
  因此現行 U0→U1→U2→9A lineage 仍是原本的 two-frame 語意。
- `sample_u2flow_role_recipe_v1` 強制 fit 使用由 `(master seed, epoch, row id)` 派生的 random
  native crop；validation 使用 epoch-invariant appearance probe＋center crop，calibration／evaluation
  使用 identity appearance＋center crop。crop 採 U²Flow 公開程式的 384×832；論文文字中的
  448×1024 仍保留為 provenance，不混成同一設定。
- U0 v1 baseline 仍只開啟 appearance transforms；integrated v2 會啟用 affine、flip 與
  endpoint swap，並輸出 native-crop→augmented-lattice 的 affine 及 endpoint order。teacher
  flow 以 inverse sampling、endpoint support 與 affine linear map 做 exact vector transport。
  所有 seed、matrix、valid support、profile 與 recipe 都可重播並雜湊。
- inventory 只讀 RGB，不搜尋或解碼 flow GT。未來 action-bank outcomes 以每列保留的獨立
  namespace 另外 join，不會成為 U0 runtime feature。

建立實際 manifest：

```bash
PYTHONPATH=src python scripts/prepare_u0_sintel_u2flow_data.py \
  --config configs/stablebridge/u0_sintel_u2flow_data_v1.json \
  --output experiments/U0_sintel_u2flow_data_v1/INPUT_MANIFEST.json
```

目前已有兩條明確分開的實驗路徑：post-hoc U0 baseline，以及 integrated recurrent v2。
後者從每次 SEA-RAFT hidden state 預測 log-variance，以 `sigmoid(-alpha).detach()` 縮放 flow
feature，再由 zero-initialized residual head 修正同一次 recurrent update。augmentation residual
同時產生有 flow gradient 的 `L_ar` 與 residual-detached 的 Laplace `L_unc`，不再把單獨外掛
uncertainty head 冒充完整 U²Flow。

## 訓練 U0 uncertainty

固定設定在：

```text
configs/stablebridge/u0_sintel_searaft_train_v1.json
```

直接訓練（需要可見 CUDA）：

```bash
PYTHONPATH=src /ssd7/cactus8603/roco_spring/optical-flow-track/.venv/bin/python \
  scripts/train_u0_uncertainty.py \
  --config configs/stablebridge/u0_sintel_searaft_train_v1.json \
  --device cuda --resume auto
```

正式 worker 由 `run_u0_uncertainty_when_gpu_free.py` 負責二次 GPU snapshot、free-memory／utilization
門檻、per-GPU lock、`CUDA_VISIBLE_DEVICES` 與斷點續訓。seed-11 baseline 已完成 20 epochs／
2720 steps；best checkpoint 是 epoch 3，而不是最後一輪。輸出位於：

```text
operations/U0_sintel_searaft_v1_seed11/status.json
experiments/U0_sintel_searaft_v1/seed11/{metrics.jsonl,latest.pt,best.pt}
```

查詢狀態：

```bash
PYTHONPATH=src /ssd7/cactus8603/roco_spring/optical-flow-track/.venv/bin/python \
  scripts/run_u0_uncertainty_when_gpu_free.py status \
  --state-dir operations/U0_sintel_searaft_v1_seed11
```

## 訓練 recurrent uncertainty-aware flow

正式消融固定為三個 v2 config，資料 split、optimizer、loss weights 與 iteration count 相同：

```text
u2_sintel_searaft_head_only_v2.json                 # matcher frozen；只訓練 hidden-state UE
u2_sintel_searaft_refinement_no_uncertainty_v2.json # 相同 refiner，以 ones/zero dummy maps 控制參數量
u2_sintel_searaft_uncertainty_refinement_v2.json    # detached uncertainty feedback（主模型）
```

正式流程先訓練 recurrent native-only U0；fit split 不讀 flow GT，只以 detached
augmentation discrepancy 訓練 uncertainty likelihood，best checkpoint 依 validation
uncertainty NLL 選擇：

```bash
PYTHONPATH=src /ssd7/cactus8603/roco_spring/optical-flow-track/.venv/bin/python \
  scripts/train_integrated_uncertainty_flow.py \
  --config configs/stablebridge/u0_sintel_searaft_recurrent_v2.json \
  --device cuda --resume auto
```

完成後，U1 會 fail-closed 地載入這份 native-only head checkpoint，核對 config digest、
matcher lineage、iteration count、完成 receipt 及 checkpoint SHA-256，再開始 refiner 訓練。

正式 U1 先凍結 SEA-RAFT 與 recurrent U0，只訓練每步最多 1 px 的 bounded refiner。
objective 同時包含 task、相對 frozen base 的 per-pixel harm、anchor 與 smoothness；validation
並列保存 `base_epe` 與 refined `epe`：

```bash
PYTHONPATH=src /ssd7/cactus8603/roco_spring/optical-flow-track/.venv/bin/python \
  scripts/train_integrated_uncertainty_flow.py \
  --config configs/stablebridge/u1_sintel_searaft_recurrent_refiner_v2.json \
  --device cuda --resume auto
```

主模型命令：

```bash
PYTHONPATH=src /ssd7/cactus8603/roco_spring/optical-flow-track/.venv/bin/python \
  scripts/train_integrated_uncertainty_flow.py \
  --config configs/stablebridge/u2_sintel_searaft_uncertainty_refinement_v2.json \
  --device cuda --resume auto
```

U2 必須同時載入同一條 lineage 的 completed U0 與 U1。每一 round 先進行 observer phase
（只更新 uncertainty head；flow objectives 關閉），再進行 flow phase（凍結 uncertainty，
只開啟 task／augmentation flow objectives）。checkpoint 保存兩份 initialization lineage、
phase cursor 由 epoch 決定，resume 不允許更換任一來源。`refinement_without_uncertainty`
消融需先使用獨立的
`u1_sintel_searaft_recurrent_refiner_no_uncertainty_v2.json` 訓練 parameter-matched U1；不得
沿用 uncertainty-guided U1 refiner。

9-anchor E292 capacity bank 已接到同一個 dataset／trainer。正式介面使用 epoch-homogeneous
action scheduling，避免同一 batch 混入不同 SEA-RAFT recurrent iteration counts；預設 20
epochs 組成 10 個 U2 rounds，每個 `native + 9 anchors` action 依序各跑一個 observer phase
與一個 flow phase，而 validation 固定 native，讓 checkpoint 選擇可比：

```bash
PYTHONPATH=src /ssd7/cactus8603/roco_spring/optical-flow-track/.venv/bin/python \
  scripts/train_integrated_uncertainty_flow.py \
  --config configs/stablebridge/u2_sintel_searaft_action_bank_training_v1.json \
  --device cuda --resume auto
```

bank 定義在 `configs/stablebridge/optical_flow_capacity_9anchor_v1.json`。訓練 checkpoint
會保存 bank hash 與 E292 source-manifest hash；每個 train-step receipt 也記錄 action 與實際
matcher iterations。影像 operators 同時套用在 native／augmented pair，iterations 8／12
則直接覆寫該 epoch 的 matcher 迭代數。這仍是 capacity-aware training，不等同 action
selector 已通過 fresh admission。這個 capacity run 也不能取代上述 U0→U1→U2 qualification；
應在 U2 provider 確認後另行 rebase，而不是把尚未配對的 U0/U1 checkpoint 混入正式 U2。

每次 validation 都保存 task／augmentation／uncertainty loss，以及 held-out GT 的 EPE、AUSE、
Spearman、severe-error AUROC 與 coverage calibration MAE。現有資料只有 Sintel Clean/Final，
因此 v2 以 Sintel GT 作 flow task carrier，而 uncertainty supervision 仍只來自 augmentation
consistency；這是 U²Flow-style SEA-RAFT adaptation，不宣稱是論文的 Raw→Clean/Final 完整
unsupervised reproduction。配置使用 SEA-RAFT Spring-M 的 4 recurrent iterations；所有消融都
固定相同 K。EPE 使用全部有效像素 streaming 累計；需要排序的 uncertainty metrics 使用
config 綁定的 deterministic `metric_spatial_stride=8` grid，避免 full-resolution validation
一次展開數千萬像素造成記憶體爆量。

先前的 `train_u1_flow_refiner.py` 與 `train_u2_uncertainty_flow.py` 保留為 one-shot post-hoc／
alternating ablation，但不是主模型，也不可標成 U²Flow reproduction。

integrated trainer 可沿用 durable GPU scheduler：

```bash
PYTHONPATH=src /ssd7/cactus8603/roco_spring/optical-flow-track/.venv/bin/python \
  scripts/run_gpu_training_when_free.py start \
  --state-dir operations/U2_sintel_searaft_uncertainty_refinement_v2_seed11 \
  --minimum-free-mib 22000 --maximum-utilization-percent 10 \
  --poll-seconds 180 --max-resource-retries 3 \
  --allowed-indices 0,1,2,3,4,5,6 -- \
  /usr/bin/env PYTHONPATH=/ssd1/cactus8603/roco_main/src:/ssd1/cactus8603/roco_main \
  /ssd7/cactus8603/roco_spring/optical-flow-track/.venv/bin/python \
  /ssd1/cactus8603/roco_main/scripts/train_integrated_uncertainty_flow.py \
  --config /ssd1/cactus8603/roco_main/configs/stablebridge/u2_sintel_searaft_uncertainty_refinement_v2.json \
  --device cuda --resume auto --reserve-vram \
  --target-vram-fraction 0.88 --vram-headroom-mib 3072
```

checkpoint resume 一律先在 CPU 還原 RNG state，再把 model／optimizer state 搬到目標裝置，
避免 CUDA `map_location` 把 `torch.set_rng_state` 所需的 CPU ByteTensor 搬錯裝置。scheduler
每 180 秒重查一次，只有 free VRAM ≥ 22,000 MiB 且 utilization ≤ 10% 才取得該卡鎖；
resource retry count 會傳入 child。可選的 CUDA memory guard 先初始化 cuBLAS，再把 PyTorch
allocator cache 補到 88% 高水位並保留至少 3 GiB library headroom；每次 resource retry 會把
目標降低 5%，最低 65%，避免保留策略本身形成 OOM retry loop。

SAM 版 U2 使用獨立 config／run directory，不會把新 objective 插入已開始的 U2 checkpoint：

SAM 2.1 Hiera-Large 的跨伺服器環境設定、100-frame key-object quality sweep、
多 GPU sharding、續跑與搬回本機流程，整理在
[`docs/SAM21_REMOTE_PREPROCESSING.md`](docs/SAM21_REMOTE_PREPROCESSING.md)。

```bash
PYTHONPATH=src /ssd7/cactus8603/roco_spring/optical-flow-track/.venv/bin/python \
  scripts/run_sam_u2_training.py \
  --trainer-config configs/stablebridge/u2_sintel_searaft_uncertainty_refinement_sam_u0ft_v1.json \
  --data-config configs/stablebridge/u0_sintel_u2flow_data_v1.json \
  --sam-output-root .runtime_tmp/sam_fullseg_sintel_vit_h \
  --sam-source .runtime_tmp/segment-anything-src \
  --sam-checkpoint .runtime_tmp/sam_vit_h_4b8939.pth \
  --sam-model-type vit_h --device cuda --resume auto
```

mask generator 固定官方 SAM ViT-H checkpoint SHA-256
`a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`，並依
UnSAMFlow 做 smallest-mask-priority full segmentation。完成 manifest 逐檔保存 source／mask
digest；dataset fail-closed 核對 root、checkpoint digest、完整狀態與每張 mask。flow phase
才加入 uncertainty-guided regional homography loss（最多 6 regions、variance threshold 2、
20% reliable support、RANSAC inlier ≥ 50%）；observer phase 不吃這條 gradient。

WAFT cross-backbone adapter 已實作相同 trainer contract，但
`configs/stablebridge/waft_cross_backbone_disabled_v1.json` 明確保持 `enabled=false`：尚未固定
WAFT checkpoint，也尚未各自訓練 WAFT U0／U1，因此不會進入本輪 scheduler 或冒充可比結果。

Action-bank selector 的 nested grouped trainer 已存在於
`research/action_bank_29_finalization_20261005/train_dynamic_group_router.py`。它要等 frozen
各 flow provider 重新 materialize action outcomes 後分開訓練；不同 flow provider 的
outcomes、normalizer、calibration 或 selector checkpoint 不得互用。正式消融至少保留：

```text
BASE:    frozen SEA-RAFT
HEAD:    recurrent uncertainty head, matcher frozen
DUMMY:   recurrent refiner with no uncertainty signal
U2-MAIN: recurrent refiner with detached uncertainty feedback
POSTHOC: frozen U0 + bounded one-shot refiner (optional legacy ablation)
```

## KITTI 2012／2015 本機資料

U²Flow stage-2 與之後 action-bank 驗證會用到的 KITTI 公開資料放在：

```text
/ssd6/cactus8603/kitti_u2flow_20261005/
├── downloads/                         # 四個原始 ZIP，保留供重驗
└── extracted/
    ├── kitti2012/{benchmark,multiview}/
    └── kitti2015/{benchmark,multiview}/
```

四包資料分別是 2012／2015 benchmark 與各自的 20-frame multiview extension；檔案來源、
byte size、SHA-256、必要目錄及解壓 inventory 固定在
`configs/stablebridge/kitti_u2flow_data_v1.json`。完整重驗命令為：

```bash
python scripts/verify_kitti_u2flow_data.py --verify-zip
```

四包 archive 合計 34,890,446,056 bytes，archive＋解壓資料目前約佔 66 GiB。官方 multiview
並非每個 scene 都有完整 21 幀；依實際檔名、相鄰幀與排除 09–12 的 U²Flow 規則，若照參考
實作同時使用 train＋test 左右 camera，實際共有 23,604 pairs，而不是公式粗估的 23,670。

這批資料不改變「先以 Sintel 開發 uncertainty observer」的決定；KITTI 會先作 domain-shift
檢查、calibration 與未來 action-bank outcome。benchmark 的 testing split 不得用於本專案
的訓練或調參；官方 U²Flow 雖把 multiview testing 影像納入 stage-2，若要重現該 transductive
protocol 必須另開明示設定。所有 KITTI 下載皆受官方註冊、用途與資料政策約束，不只 Raw。
完整 stage-1 所需的 Raw drives 本次未納入；SAM key-object／full-seg masks 必須由影像產生，
不是這四包官方資料的一部分。本版已加入官方 ViT-H full-seg 產生與 lineage 驗證，但大型
checkpoint／masks 保留在 `.runtime_tmp`、由 GPU 排程產生，不進 Git。

## 目前啟用邊界

```text
uncertainty runtime mode = observer_only
multi_action.enabled      = false
maximum committed actions = 1
continuous strength       = false
final validated bank      = empty
capacity training bank    = E292 9 anchors + native (TEST_ONLY)
```

29-arm catalog 是可追溯 inventory；E292 的 9-anchor bank 已作為 capacity-aware training
介面，但尚未產生通過 fresh selector admission 的 active production bank。既有 Work-B uncertainty artifacts 也混有 native／action
states，不能冒充 native-only U0 checkpoint。

## 文件與重算入口

- [Action bank／selector／uncertainty 完整設計](docs/research/action_bank_selector_architecture_20261005/README.md)
- [Action-bank reduction 與 fresh-validation work package](research/action_bank_29_finalization_20261005/README.md)
- [StableBridge package 說明](src/stablebridge/README.md)

大型 datasets、model weights、experiment outputs、runtime caches、外部 vendor repositories
與機器專屬 artifacts 不進 Git。所有科學結論都必須由獨立資料、固定 split、完整
choice-family calibration 與可重播 receipts 支持。

部分 frozen execution／source receipts 保留建立證據時的絕對路徑與 executable identity；
這些欄位是 provenance，不是可攜式預設值。新環境應建立自己的 dataset/model registry，
不能直接把舊機器的 authority receipt 當作有效授權。
