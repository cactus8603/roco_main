# 原生光流訓練完整指南

這份指南對應目前使用的 **3,000,576 參數 motion-aligned 修復器**。
WAFT 完整凍結；修復器輸出完整雙幀特徵，經原 WAFT 五步後以原生 MoL loss 訓練。
不需離線 target features、昂貴特徵最佳化、額外 DINOv3 或 Mamba extension。

## 1. 安裝環境

已核對環境為 Python 3.10、PyTorch 2.5.1、torchvision 0.20.1。
完整影像建議 Linux＋NVIDIA CUDA；小尺寸驗證可使用 CPU。

```bash
git clone https://github.com/cactus8603/roco_main.git
cd roco_main
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[dev,waft]"
python -m pip install -r requirements/native-flow.txt
waft-repair --help
```

若使用 CPU，在 PyTorch 安裝步驟改用 `https://download.pytorch.org/whl/cpu`。
`requirements/repair-cuda.txt` 是較早 target-feature 模型的 extension 依賴，
目前流程使用 `requirements/native-flow.txt`。

## 2. 準備官方 WAFT

```bash
git clone https://github.com/princeton-vl/WAFT.git external/WAFT
git -C external/WAFT checkout b152ff1cad1af8c185ee7b141997c48ff3334c87
```

從 [官方 checkpoint 資料夾](https://drive.google.com/drive/folders/1joBWKGoH2RUdCgcge8Tz2osOHcQUX5m_?usp=sharing)
取得 **WAFT a2／DAv2／vits** 的 source-only `zero-shot.pth`、官方 `kitti.pth` 或官方 `sintel.pth`，
放在 `checkpoints/waft/`。下方32epoch範例保留source-only設定；目前官方50／80epoch命令見新增段落。

固定 checkpoint SHA-256：

```text
1b49d1fc4513c6fa34a8de8f791dd8735d366d4b25d8dea47626a79056edd402
```

```bash
waft-repair doctor --waft-source external/WAFT \
  --waft-checkpoint checkpoints/waft/zero-shot.pth
```

`doctor` 以 CPU 核對官方來源與全部 508 個 tensors，不執行光流推論。
載入過程不另下載 DAv2 權重。loader也接受下列兩個明確登錄的官方checkpoint；其他variant、任意fine-tuned權重或未核對來源會拒絕載入。

| 檔案 | 檔案SHA256 | 全508 tensor digest |
|---|---|---|
| `kitti.pth` | `656bcbeb7c6b8627eae577d5564f2849f32e1d53f85dcac7324fbeddc96bc9da` | `2c5e8002cb19e0f8d05ee2bfbb1cec6b0507e7690defeebc2b938c7670bf377b` |
| `sintel.pth` | `aad14eea3d68a70a4009332d941a863a1e5f513af252f6d9136725e654808322` | `040e5864cde10395466f9102b3ea72f629263cb1ce1068d6afcc2400ba2b97c6` |

官方domain權重已有該資料域的訓練曝光；不能把本專案dev評估稱為底模從未見過的資料。

## 3. 建立影像清單

### 固定 KITTI 160／40

準備 KITTI 2015 training 資料，保留原始目錄：

```text
data/kitti/
└── training/
    ├── image_2/000000_10.png
    ├── image_2/000000_11.png
    └── flow_occ/000000_10.png
```

```bash
waft-repair native-split-kitti --kitti-root data/kitti \
  --output outputs/kitti-spec
```

輸出 `outputs/kitti-spec/images.json`，採用 [公開固定 ID](../examples/native/kitti-split.json)：
160 TRAIN、40 dev，兩者共同涵蓋官方 200 筆 training pairs。
沒有按 EPE、教師品質或是否容易擬合篩選案例。
helper 只建立清單，不複製資料、不開始 WAFT 推論。
產出的路徑指向使用者自己的資料位置；若搬移資料，重新產生清單。

這 40 筆是已被歷史研究使用的 dev，不是官方隱藏 test 或全專案未曝光資料。
訓練只讀 TRAIN 的 GT 計算更新，dev 只用於監測。

### 自訂資料

以 [影像清單範例](../examples/native/images.json)為起點：

```json
{
  "schema": "waft-native-images/v1",
  "samples": [
    {
      "id": "my-training-pair",
      "split": "train",
      "image1": "first.png",
      "image2": "second.png",
      "ground_truth": "flow.npz",
      "ground_truth_format": "npz"
    }
  ]
}
```

相對路徑以 spec 所在資料夾為基準，`id` 必須唯一。
`split` 為 `train` 或 `dev`；`ground_truth_format` 為 `kitti`、`sintel` 或 `npz`。
KITTI 可直接使用 `flow_occ` PNG，Sintel 使用 FLO。
NPZ 使用數值 `flow: float32[H,W,2]` 與 `valid: bool[H,W]`，不含 pickle。
影像、GT 與 valid 必須具有相同原始幾何；不會將稀疏 KITTI GT 插值成稠密標籤。

若需轉換既有 GT：

```bash
waft-repair convert-gt --format kitti \
  --flow data/kitti/training/flow_occ/000000_10.png \
  --output data/flow-000000.npz
```

原生光流訓練不需要 visibility、配對 Clean 或 teacher image。
自訂人口與固定 160／40 不同時，資料量、學習率排程與結果範圍應分別記錄。

## 4. 快取原始輸入

```bash
waft-repair native-prepare --spec outputs/kitti-spec/images.json \
  --output outputs/kitti-inputs --device cuda:0 \
  --waft-source external/WAFT --waft-checkpoint checkpoints/waft/zero-shot.pth
```

每筆只從原始 RGB 建立 F0、sigma、同次 encoder 四層 observations、
原始五步 baseline 與第五步 motion prior。TRAIN／dev 使用相同輸入製程，
不以 GT 修正輸入、不逐筆最佳化特徵，也不製作 cheap／expensive target。

輸出 manifest 綁定輸入、標籤、幾何、checkpoint 與檔案雜湊。
標籤與模型輸入分開保存；GT 不進修復器 forward。
原始 200 筆全部快取後才進訓練，後續每次更新仍需執行 repaired 五步。

輸出目錄請使用新路徑；不要以重複執行覆寫既有結果。
完整 cache 的容量依原始幾何與保存內容而定，不等於模型 checkpoint 大小。

## 5. 完整訓練

```bash
waft-repair native-train --manifest outputs/kitti-inputs/manifest.json \
  --output outputs/native-repair --epochs 32 --device cuda:0 \
  --waft-source external/WAFT --waft-checkpoint checkpoints/waft/zero-shot.pth
```

固定設定：

| 項目 | 值 |
|---|---|
| TRAIN／dev | 160／40，dev 不更新 |
| 更新數 | 32 epochs × 160，batch 1，共 5,120 |
| seed／順序 | 20261028；每 epoch 獨立 seed shuffle，全 TRAIN 各使用一次 |
| 資料增強 | 無 |
| Optimizer | AdamW，betas (0.9,0.999)，epsilon 1e-8，weight decay 0 |
| Learning rate | peak 3e-4；warmup 160 updates，cosine 至 peak 的 0.1 |
| Gradient clip | FP64 global norm，最大 1 |
| Loss | 原生五步 MoL，gamma 0.85；teacher auxiliary 為 0 |
| WAFT | 全部 508 tensors 固定、eval、parameter gradients 為空 |
| 最終模型 | 固定第 32 epoch，沒有 dev-best 選擇 |

每次由 Fhat 重建 hidden，從零光流完整跑五步。consumer 保留原 WAFT 的 flow／coordinate
detach 行為；不要改成 teacher full-feedback，也不要對 repaired consumer 使用 `no_grad()`。

主要輸出：

| 路徑 | 內容 |
|---|---|
| `checkpoint.pt` | 最新可續訓模型、optimizer、cursor、RNG、設定與資料綁定 |
| `model.pt` | 可供影像推論的 portable repair checkpoint |
| `history.jsonl` | 更新歷程；online 數值是該次更新前的模型表現 |
| `progress.json` | 當前進度 |
| `checkpoints/E###.pt` | 固定評估 epochs 的完整續訓 checkpoint；canonical run 為 E001／E004／E008／E016／E032 |
| `evaluations/init`、`E001`、`E004`、`E008`、`E016`、`E032` | 固定 checkpoint 的評估輸出 |

固定評估的 `summary.json` 保存逐例五步標量與人口彙總。canonical run 的 init／中途評估
只保存標量，E032 才另存完整五步 flow／info；獨立 `native-evaluate` 則保存所選人口的完整五步。

完整人口的固定評估與 online batch 指標用途不同。EPE、Fl 與 MoL 分別記錄，
MoL 下降不保證每筆或平均 EPE 同步下降。

### 續訓與短程驗證

沿用原本 manifest、輸出目錄與所有訓練設定：

```bash
waft-repair native-train --manifest outputs/kitti-inputs/manifest.json \
  --output outputs/native-repair --epochs 32 --device cuda:0 \
  --resume outputs/native-repair/checkpoint.pt \
  --waft-source external/WAFT --waft-checkpoint checkpoints/waft/zero-shot.pth
```

需要先確認數個更新可執行時，在第一次訓練加 `--stop-after-updates 2`，
保留 `--epochs 32` 與原排程，之後用上述續訓命令接續。
這個選項指定 run 的**絕對更新停止位置**，不把 LR 排程縮成兩步。
例如已完成 100 次時，再指定 120 表示執行至第 120 次，而非另外更新 120 次。
checkpoint 預設每 32 updates 保存，正常短程停止時也保存可續訓狀態。

CPU 小尺寸測試可以用自己準備的 32×48 RGB、相同大小的有限 GT／valid，建立只含 TRAIN
的 spec，先 `native-prepare --device cpu`，再以新輸出執行
`native-train --device cpu --epochs 2 --warmup-updates 1 --stop-after-updates 2`。
這是通路驗證，與完整 KITTI 的預期成效、容量與速度不同。

checkpoint 恢復模型、optimizer、cursor 與 RNG；跨 CUDA 硬體或 kernel 不宣稱逐位元重播。
不要更換資料或把其他版本的 optimizer 狀態接到本模型。
CPU thread 設定也會綁定；以相同環境與 thread 數續訓。`--threads` 放在子命令前，例如
`waft-repair --threads 1 native-train ...`。使用同一 run 最新的 `checkpoint.pt`；
不支援直接以較舊 checkpoint 回退並覆寫既有 run。
若異常中斷留下超過 checkpoint 的日誌尾端，程式會另存 `history-uncommitted-*.jsonl`，
再從已保存的更新位置接續，避免將未提交紀錄重複算成已完成更新。

## 官方 KITTI50／Sintel80 與記憶體處理

README 提供官方 KITTI50epoch及 Sintel80epoch的完整命令。KITTI的其他設定沿用上方單卡流程，
只在新run開始時固定 `--epochs 50` 和 `kitti.pth`，共8000updates；換checkpoint後須重新 `native-prepare`。
已存在的run不會因修改命令列而自動改排程；source、底模或資料綁定不同會拒絕 exact resume。

### Sintel雙卡RGB入口

`native-split-sintel`使用18個TRAIN場景與5個dev場景。
dev固定為 `bandage_1`、`bandage_2`、`mountain_1`、`shaman_2`、`shaman_3`；
Clean和Final採用相同場景分割，共1592TRAIN／490dev，不按結果選樣本。
Sintel原始有效支援為兩個flow分量皆有限且絕對值小於1000。

```bash
waft-repair native-split-sintel --sintel-root data/sintel --output outputs/sintel-spec
CUDA_VISIBLE_DEVICES=0,1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=2 \
  torchrun --standalone --nnodes=1 --nproc-per-node=2 --module waft_repair \
  --threads 2 native-train-ddp --spec outputs/sintel-spec/images.json \
  --output outputs/sintel-repair --epochs 80 --warmup-updates 796 \
  --waft-source external/WAFT --waft-checkpoint checkpoints/waft/sintel.pth
```

兩個rank各1筆，global batch2；80×796＝63680同步updates。
AdamW、native5 MoL gamma.85、peak3e-4、epsilon1e-8、weight decay0與FP64 norm1 clipping保持相同；
warmup796後cosine至peak的0.1。DDP只包覆修復器，以兩個完整case的局部平均loss做一次梯度平均。
不另除以world-size；rank0計算FP64 norm並廣播相同clipping比例。
每次visit直接從原始RGB重建F0、sigma、四層taps、baseline及prior，不存feature cache。

初始與固定E032／E080評估完整2082case，E001／E004／E008／E016僅評估490dev；
共8206case，每筆保存baseline與repair的全部五步指標，init與最終另保留4164份flow檔。
雙卡逐例標量寫入 `evaluations/<label>/rank0.jsonl` 和 `rank1.jsonl`；
其中EPE是該case原始有效像素平均。兩rank的TRAIN online loss為兩個case均值的算術平均，
與按整個資料集有效像素數加權的EPE分開看。dev不參與optimizer更新。

接續同一run時，在原始torchrun命令加上：

```text
--resume outputs/sintel-repair/checkpoint.pt
```

`--stop-after-updates 2`可做保留原63680步排程的短程驗證；接續時移除此選項。
checkpoint預設每256updates及正常停止時完整保存；兩個rank的repair／AdamW digest必須完全相同。
保存完整161R、161AdamW的step與moments、scheduler、下一對輸入及各rank的Torch／Python／NumPy／CUDA RNG。
兩份history都會保存未commit尾端後接續，不將尾端logs當成已保存更新。
本入口使用新的portable run綁定；現有主機operation的checkpoint不會被靜默換綁定或當成新run的exact resume。

### 主機記憶體界線

使用Linux／glibc與PyTorch2.5.1的已核對 pinned-host cache API，並在啟動Python前設定
`MALLOC_ARENA_MAX=2`。每筆輸出與梯度不再使用後：

1. 刪除上一筆暫存引用，完成垃圾回收並同步CUDA。
2. 清除僅供重用的閒置pinned-host blocks，再執行`malloc_trim(0)`。
3. 對已使用的自有普通檔案，透過唯讀、no-follow descriptor給予`POSIX_FADV_DONTNEED`建議。

此流程不修改原始檔案內容、不改有效tensor／loss／optimizer，也不呼叫CUDA `empty_cache`。
逐筆遙測包括RSS、cgroup current／peak／events；對共享但非本UID的輸入會明確拒絕給予cache advice。
被讀取的資料應由執行者擁有，或先建立自己的合法資料副本。
這些釋放動作不保證永不遇到OOM；仍須依主機監控及限制執行。

公開CLI為前景程式，不安裝systemd服務、不變更OOMd設定。
現有正式主機作業的GPU admission、PSI門檻、每rank14GiB high／16GiB max與Swap0由其凍結wrapper管理。
公開命令可額外指定 `--memory-fraction 0.19 --reserve-gib 64`；這兩個旗標分別限制PyTorch顯存比例及保留磁碟空間，
不會建立cgroup RAM限制或GPU合作鎖。共享主機請由既有scheduler提供資源分配。

雙卡trainer的step、完整checkpoint建構／驗證及每rank RNG載入，沿用已核對的原始native trainer計算。
套件化改用相對import與portable資料／程式雜湊綁定；這不代表主機的systemd／pidfd控制程式也已移植。
新portable入口以CPU雙程序synthetic consumer驗證梯度平均、完整AdamW／RNG、停止／接續；
官方完整508權重另外以CPU載入核對。未新增正式CUDA訓練實驗或宣稱新的EPE改善。

## 6. 固定 checkpoint 評估

```bash
waft-repair native-evaluate --manifest outputs/kitti-inputs/manifest.json \
  --repair-checkpoint outputs/native-repair/model.pt --split dev \
  --output outputs/dev-evaluation --device cuda:0 \
  --waft-source external/WAFT --waft-checkpoint checkpoints/waft/zero-shot.pth
```

`--split` 可選 `train`、`dev` 或 `all`，預設 `dev`。
評估保持修復器與 WAFT 固定、不做 optimizer 更新；使用完整原始 valid 分母，
所有退步案例保留。檢查中間 checkpoint 時使用獨立輸出目錄，不挑選或改寫最終 E032 規則。

## 7. 無 GT 影像推論

```bash
waft-repair native-infer --repair-checkpoint outputs/native-repair/model.pt \
  --image1 data/first.png --image2 data/second.png \
  --output outputs/prediction.npz --device cuda:0 \
  --waft-source external/WAFT --waft-checkpoint checkpoints/waft/zero-shot.pth
```

這條路只讀兩張 RGB、WAFT 與修復 checkpoint。它重新計算 prefix、baseline 五步與 motion prior，
修復特徵後再完成 repaired 五步。無 GT、teacher、dev manifest 或逐筆最佳化。

NPZ 使用 `allow_pickle=False` 可讀，包含原尺寸最終 flow 與完整五步 flow／info：

```python
import numpy as np
with np.load("outputs/prediction.npz", allow_pickle=False) as result:
    print(result["flow"].shape)        # H, W, 2
    print(result["flow_steps"].shape)  # 5, H, W, 2
    print(result["info_steps"].shape)  # 5, H, W, 4
```

完整推論時間必須包含 baseline 五步與 prior，不能只計算修復器本身。

## 8. 匯入既有原生修復 checkpoint

若已有相同 3.00M 原生 run 的 full checkpoint，可以只匯出 repair 權重：

```bash
waft-repair native-import-checkpoint --checkpoint saved-native-checkpoint.pt \
  --output outputs/imported-model.pt
```

可額外使用 `--expected-sha256` 核對來源。
匯入會嚴格核對模型結構，產生 portable model；它不移植舊 run 的 optimizer／cursor，
也不把舊 target-feature MSE 或 traceable 實驗模型轉成目前模型。

## 9. 驗證與範圍

```bash
python -m pytest -q tests/waft_repair
```

模型／consumer／loss 的移植需與原生來源核對；CPU 小尺寸驗證與完整資料集 GPU 訓練效果
分別記錄。模型拆解訊號不是錯誤根因證明，公開程式也不附訓練後權重。
現有中途結果與成本範圍見 [結果文件](results.md)。

所有命令以前景方式使用指定 device，不會自動建立 systemd job、重啟其他訓練，
或沿用本機研究工作的排程器。兩個流程的歷史比較請看 [舊 target-MSE 指南](legacy-running.md)。
