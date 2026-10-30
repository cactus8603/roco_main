# Encoder-only LiteR 精簡版

本入口使用 14,960 參數的 `LiteFeatureRepair`，每次從兩張原始 RGB 重建凍結 encoder
features 與同次 CNN S2／S4、DAv2 block6／12 taps，先修復 feature，再初始化 hidden
並執行唯一一組五次 WAFT refinement。R 不讀 GT、flow、decoder info、motion prior 或 case ID。
GT 只用於 native5 MoL 訓練損失與末次評估，完整 WAFT 保持 eval／frozen。

程式位於 `waft_repair.lean`，與原 `waft_repair.motion` 分開。兩者的模型形狀、特徵介面及
checkpoint 不可互換。公開版使用新的來源綁定，不會將本機 operation checkpoint 靜默轉成公開版 exact resume。

## 計算與輸出

- 訓練仍完整計算五階段 flow／info 和 gamma 0.85 的 native MoL。
- AdamW peak LR 3e-4、epsilon 1e-8、weight decay 0、原 FP64 norm 1 clipping 保持不變。
- **只在全部訓練完成後評估一次 EPE，使用第五次 flow**；沒有初始化或中途 EPE forward。
- EPE 評估保留五次 refinement，只做最後一次完整解析度 readout，不計算評估 MoL。
- raw CPU cache 上限 128 MiB；整批放不下時全部改用 streaming。cache 不保存 encoder features 或 teacher。
- 固定幾何有界快取；不進模型 state。每步仍保留 GC、CUDA 同步、閒置 pinned-host cache 清理與 libc trim。
- 每 epoch 一行 `metrics.csv`；覆寫完整 `latest.pt`、`progress.json`；末次保存 `final.pt`、`result.json`。
  CSV 的 EPE 僅最後一行有值，評估 MoL 欄保留空白。`initial_EPE` 若沒有既有參考值則為空，不另跑 baseline。
- `train.log` 與 `runtime/` 只有生命週期／資源保護紀錄；不輸出逐筆 feature／flow／gradient dump。

## 安裝與 CPU 測試

沿用 README 的 Python／PyTorch／官方 WAFT 安裝方式。

```bash
python -m pip install -e ".[dev,waft]"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest tests/waft_repair/test_lean_*.py -q
```

測試使用小型 synthetic CPU core，涵蓋原 all5 數值、兩步 R／AdamW／RNG bytes、末次 EPE、
raw cache、必要記憶體回收順序、資料／模型／implementation 綁定、完整 optimizer／RNG、
累計 clock、續訓 pressure ledger 移交、雙重逾時及錯誤收尾。
這些測試不建立訓練服務，不執行官方 NN 或初始化 CUDA；CPU 等價不代表已驗證 CUDA 速度或新 EPE 結果。

## 必要的主機資源保護 adapter

公開套件不包含某台主機的 GPU UUID、UID、合作鎖目錄或既有作業身分。
`resource_control` 必須明確指向使用者自己的 Python adapter；缺少或介面不完整時，launcher
會在 admission／建立服務之前拒絕執行。沒有未受保護的 fallback，也不會修改 OOMd 或其他工作。

adapter 必須實作以下介面，語義與受保護的本機入口相同：

| 函式 | 必要行為 |
|---|---|
| `admit(protocol)` | CPU-only admission；回傳 `(admission, handle)`，包含精確 `owner` 與 `selected_gpu_uuid`、自有 lease 證據 |
| `wait_before_context(protocol, admission, existing_model_loads=1)` | 第一個 CUDA context 前再驗容量、RAM／PSI 與 lease |
| `setup_cuda(protocol, admission)` | 核對指定實體 UUID，設定裝置及 0.19 顯存比例；不得重建／重試 context |
| `check(admission, protocol, _sleep=...)` | 保留週期 GPU／主機檢查，壓力等待寫入 `running_resource_wait.json`，只透過所傳 sleep 計時 |
| `release_own_slot(handle, admission, protocol, child_waited=True, cleanup=...)` | 真正 wait 完成且 CUDA allocated／reserved 為零後，僅釋放本次自有 lease |

保留的限制為 memory fraction **0.19**、minimum free **5700 MiB**、headroom／context reserve
各 **512 MiB**、maximum total users **3**，主機 available RAM 至少 **64 GiB**、全機 PSI
some ≤ **5%**／full ≤ **1%**；壓力恢復需連續健康 **10 秒**。
service 設定為 MemoryHigh **14 GiB**、MemoryMax **16 GiB**、Swap **0**、TasksMax **512**，
不自動 restart，不用 force kill。週期檢查維持原 30 秒及 epoch 邊界強制檢查。

每個 dataset 的 clock 使用原有限預算：before-context 1800 秒、active 7200 秒、overhead 7200 秒、
running pressure wait 21600 秒、shutdown margin 300 秒；child timeout 43200 秒。
adapter 必須依照同一 profile 執行硬門檻，不能把這些數字只當作遙測。

## 產生私有設定，不啟動訓練

下列命令只核對原檔／checkpoint SHA、讀取影像尺寸並建立設定，不載入模型、不建立服務。
`--gpu-uuid` 使用主機實際選定的實體 UUID，`--resource-control` 指向已審核的 site adapter。

```bash
python -m waft_repair.lean.configure \
  --dataset kitti --data-root data/kitti \
  --waft-source external/WAFT --checkpoint checkpoints/waft/kitti.pth \
  --resource-control site/resource_control.py --gpu-uuid GPU-CHOSEN-UUID \
  --output outputs/lean-kitti --config outputs/lean-kitti-config.json
```

KITTI 固定 `000000`／`000001`／`000002`；32 epochs、96 updates、warmup 3。
Sintel 改用 `--dataset sintel --data-root data/sintel --checkpoint checkpoints/waft/sintel.pth`：
固定 `alley_1`／`ambush_7`／`market_5` 的第一對 Clean＋Final，共六筆；32 epochs、192 updates、warmup 6。
兩者各為單 GPU child，沒有自動全量 promotion。這些是預先固定的少樣本 fitting 設定，不能當成 heldout 泛化證據。

設定包含使用者本機的絕對路徑與 SHA，應留在自己的 outputs。輸出設定已存在時會拒絕覆寫。

## 明確啟動與續訓

只有使用者之後明確執行此命令，才會建立受限服務：

```bash
CUDA_VISIBLE_DEVICES='' python -m waft_repair.lean.launch \
  --dataset kitti --config outputs/lean-kitti-config.json
```

續訓沿用設定與輸出目錄，加上 `--resume outputs/lean-kitti/latest.pt`。
只接受完整 epoch checkpoint；新 checkpoint 核對原資料 descriptors、模型、排程、實作來源、
完整 AdamW 設定／moments、RNG 與 clock。主機 adapter 仍是必要條件。

若同目錄前輪留下資源等待 ledger，先核實前輪 actual wait、零 CUDA cleanup、own lease release
與 invocation token，再移交給新的 owner；累計 21600 秒不歸零。clock 同時承接前輪 terminal
與本次 setup 耗時。逾時且第二次 wait 仍未完成時保存 `WAIT_UNRESOLVED` 並保留 lease；
不得將舊 `result.json`、未完成回收或中斷當作成功。
