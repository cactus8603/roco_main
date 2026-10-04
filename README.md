# RoCo / StableBridge

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

核心依賴列於 [`pyproject.toml`](pyproject.toml)。真實 CroCo／SEA-RAFT 執行仍需要各自的
官方程式、權重與資料集；它們不包含在 repository 中。

## 測試

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/stablebridge
```

目前整理版本的完整 StableBridge regression 為：

```text
1187 passed, 19 subtests passed
```

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
  observer 的訓練語意；啟用時採 `[previous,current,next]`，scene 起點重複第一幀。
- `sample_u2flow_role_recipe_v1` 強制 fit 使用由 `(master seed, epoch, row id)` 派生的 random
  native crop；held-out roles 忽略 epoch，固定使用 identity appearance＋center crop。U²Flow 公開
  程式是 384×832 random crop、論文文字是 448×1024；目前 CroCo flow checkpoint 固定為
  320×384，因此採 320×384 且把差異寫入 hash-bound recipe，沒有假裝完全重現原論文。
- fit augmentation 包含共同 crop、horizontal／vertical flip、endpoint swap、可逆 affine、
  brightness／contrast／saturation、Gaussian blur 與 returned-last-frame erasing。所有 seed、
  affine inverse、valid support、profile 與 recipe 都可重播並雜湊。
- inventory 只讀 RGB，不搜尋或解碼 flow GT。未來 action-bank outcomes 以每列保留的獨立
  namespace 另外 join，不會成為 U0 runtime feature。

建立實際 manifest：

```bash
PYTHONPATH=src python scripts/prepare_u0_sintel_u2flow_data.py \
  --config configs/stablebridge/u0_sintel_u2flow_data_v1.json \
  --output experiments/U0_sintel_u2flow_data_v1/INPUT_MANIFEST.json
```

這是 U0 observer 的 development manifest 與 RGB view primitive，不是可直接開訓的 U²Flow
官方 reproduction：本機尚未加入 Sintel Raw、SAM masks 或官方 U²Flow checkpoint；teacher
flow／occlusion／uncertainty target 的同步幾何轉換及 training Dataset 尚未接上。公開程式中的
per-frame relative affine 也尚未逐式移植；目前 profile 明確標記為 shared-pair affine，之後
可在不改資料 split 的情況下升級。

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
不是這四包官方資料的一部分，也明確標為尚未具備。

## 目前啟用邊界

```text
uncertainty runtime mode = observer_only
multi_action.enabled      = false
maximum committed actions = 1
continuous strength       = false
final validated bank      = empty
```

29-arm catalog 是可追溯 inventory；opened-development reduction 目前只產生下一輪驗證候選，
尚未產生 final active action bank。既有 Work-B uncertainty artifacts 也混有 native／action
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
