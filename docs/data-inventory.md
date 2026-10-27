# 資料存放清單

這份清單回答：原始影像、GT、WAFT 權重、input features、兩種 target，以及修復模型輸出各存在哪裡。
範圍是目前「凍結 WAFT → 特徵修復 → 凍結 WAFT」主線直接使用的資料及核驗來源。

**盤點時間：2026-10-27 10:24:50（Asia/Taipei）。**
這是一份儲存快照，正在產生的檔案與數量可能繼續變動。
名稱按用途整理；實際磁碟路徑保留既有名稱，方便直接查找。

下載完整的 [47 項資料清單 CSV](data-inventory.csv)，可依名稱、路徑、數量、bytes 與狀態搜尋。
GitHub 上傳的是清單與文件；影像、權重、特徵封包及訓練輸出仍在原磁碟。

## 先看資料分布

| 存放位置 | 主要內容 |
|---|---|
| **SSD4** | KITTI／Sintel 原始影像、GT 光流與遮擋標註 |
| **SSD7** | 官方預訓練 WAFT 權重 |
| **SSD1** | 官方 WAFT 程式、資料索引、原始 features、便宜／昂貴 targets、修復訓練與結果 |
| **GitHub** | 主線說明、可重用 target API、精簡結果與本清單 |

實體資料目前分散保存，沒有統一搬到一個新資料夾。
本次只讀取既有索引與檔案狀態，沒有移動、刪除或重新生成資料。

## 原始資料集

| 資料 | 實際資料夾 | 現有檔案 |
|---|---|---:|
| KITTI 雙幀 RGB | `/ssd4/cactus8603/flow_datasets/KITTI/extracted/kitti2015/benchmark/training/image_2` | 400 PNG，200 對 |
| KITTI 含遮擋光流 GT | `/ssd4/cactus8603/flow_datasets/KITTI/extracted/kitti2015/benchmark/training/flow_occ` | 200 PNG |
| KITTI 非遮擋標註 | `/ssd4/cactus8603/flow_datasets/KITTI/extracted/kitti2015/benchmark/training/flow_noc` | 200 PNG |
| Sintel Clean RGB | `/ssd4/cactus8603/flow_datasets/Sintel/training_release/extracted/training/clean` | 1,064 PNG，23 場景 |
| Sintel Final RGB | `/ssd4/cactus8603/flow_datasets/Sintel/training_release/extracted/training/final` | 1,064 PNG，23 場景 |
| Sintel 光流 GT | `/ssd4/cactus8603/flow_datasets/Sintel/training_release/extracted/training/flow` | 1,041 FLO |
| Sintel 遮擋標註 | `/ssd4/cactus8603/flow_datasets/Sintel/training_release/extracted/training/occlusions` | 1,041 PNG |

Sintel 每個 render 有 1,041 對具 GT 的相鄰影像。來源資料夾包含的影像多於目前選定的訓練資料。
遮擋 PNG 中有共享 inode 的 hardlinks：1,041 個路徑對應 1,035 個檔案實體；CSV 分開記錄。

| 資料範圍 | KITTI | Clean | Final | 合計 |
|---|---:|---:|---:|---:|
| 目前指定訓練資料 | 160 | 796 | 796 | 1,752 |
| 分開的開發資料 | 40 | 245 | 245 | 530 |
| 已完成便宜 target 比較 | 160 | 100 | 100 | 360 |

訓練與開發案例沒有重疊；開發集不代表歷史上完全未曝光或預訓練未使用的 held-out。
目前的 360 筆 target 結果也不代表全 1,752 筆皆已生成新便宜 target。

完整訓練案例列表位於：

```text
/ssd1/cactus8603/roco_main/operations/waft_dense_gt_feature_cache_1752_v2_20261026/protocol.json
```

其中 `train_cases` 記錄訓練影像與 GT；下列來源索引的 `excluded_dev_case_ids`
與 `canonical_data_protocol` 記錄排除的開發案例及其原始切分來源：

```text
/ssd1/cactus8603/roco_main/operations/waft_gt_feature_cache_v1_20261026/protocol.json
```

## 凍結 WAFT

| 內容 | 實際位置 |
|---|---|
| 官方完整程式 | `/ssd1/cactus8603/roco_main/.runtime_tmp/waft-src` |
| Source-only 預訓練權重 | `/ssd7/cactus8603/roco_main_how_artifacts/models/waft_source_only/zero-shot.pth` |
| 模型設定 | `/ssd1/cactus8603/roco_main/configs/stablebridge/waft_a2_dav2_source_only_v1.json` |

權重檔目前為 **227,903,354 bytes**。現行協定綁定的 SHA-256 為：

```text
1b49d1fc4513c6fa34a8de8f791dd8735d366d4b25d8dea47626a79056edd402
```

本次確認存在與大小，沒有重新雜湊整個權重檔。
`/ssd1/cactus8603/roco_main/artifacts/waft_how` 是指向
`/ssd7/cactus8603/roco_main_how_artifacts` 的 symbolic link，並非另一份模型副本。

## Input features 與便宜 target

目前選定方法是非遮擋區完整 RGB 重建。以下兩個資料夾保存已核驗的資料：

| 資料 | 實際資料夾 |
|---|---|
| KITTI，160 對 | `/ssd1/cactus8603/roco_main/operations/waft_gt_rgb_kitti_two_methods_v2_20261027/artifacts` |
| Sintel Clean，100 對 | `/ssd1/cactus8603/roco_main/operations/waft_gt_rgb_clean_method_search_v2_20261026/artifacts` |

每個資料夾內的實際檔案位於 `left/run/case_*/` 與 `right/run/case_*/`：

| 檔名 | 內容 | KITTI 數量 | Clean 數量 |
|---|---|---:|---:|
| `original.pt` | 原始完整雙幀 input features 與來源資料 | 160 | 100 |
| `visible_pull_1_target.pt` | 目前選定的便宜 target features | 160 | 100 |
| `baseline_final_flow.pt` | 原始 features 的 WAFT 最終光流 | 160 | 100 |
| `visible_pull_1_final_flow.pt` | 便宜 target 直接回注 WAFT 的最終光流 | 160 | 100 |

兩組原始特徵封包合計 **260 檔／22,805,135,344 bytes（約 21.24 GiB）**；
便宜 target 封包合計 **260 檔／21,613,772,576 bytes（約 20.13 GiB）**。
`.pt` 封包還含 RGB、GT、mask 或幾何等資料，這些大小不是純 feature tensor 的大小。
這裡使用選定的可見區重建檔，舊的未排除遮擋 target 不納入目前選定資料數量。

### Final 如何保存？

Final 的 100 筆便宜監督共用上列完整配對 Clean target，**新增實體檔案為 0**。
因此 260 份唯一便宜 target 封包可供 360 筆邏輯案例使用。

原 Final baseline 光流與評分另存在：

```text
/ssd1/cactus8603/roco_main/operations/waft_k160_s100_parallel_target_EPE_v2_20261026/artifacts/Final/run
```

其中有 100 份 `case_*/baseline_final_flow.pt`。
這個評估目錄沒有保存 100 份完整原始 Final features；保存的特徵 SHA 與 shape 也不能當作特徵檔本身。
這項缺口不代表沒有任何 Final 特徵存在：昂貴 cache 另有個別 Final 封包。

## 昂貴 target

昂貴版的精簡特徵封包與逐案例接受索引分開保存：

| 用途 | 實際資料夾 |
|---|---|
| 特徵封包 | `/ssd1/cactus8603/roco_main/operations/waft_dense_gt_feature_cache_1752_v2_20261026/artifacts/cache` |
| 已接受案例索引 | `/ssd1/cactus8603/roco_main/operations/waft_dense_gt_feature_cache_1752_v2_20261026/artifacts/entries` |

以 `entries/**/entry.json` 中 `qualified=true`、對應 `features.pt` 存在且大小吻合的案例為準：

| 快照範圍 | 份數 | 封包 bytes |
|---|---:|---:|
| 已接受 KITTI | 139 | 22,029,709,272 |
| 已接受 Clean | 9 | 1,193,390,280 |
| 已接受 Final | 1 | 132,598,856 |
| **已接受合計** | **149** | **23,355,698,408，約 21.75 GiB** |
| 目錄內全部 `features.pt` | 151 | 23,672,441,976 |

151 個實體檔中有 2 個尚未進入接受索引；不能把實體檔總數當作核驗通過數。
這是對計畫 1,752 對資料的**部分快取**。
本清單沒有因某個進度檔的數字較大，就增加已接受案例數。

兩筆保留完整最佳化證據的封包另存於：

```text
/ssd1/cactus8603/roco_main/operations/waft_dense_target_epe_continuation_v2_20261026/artifacts/job/run/teacher
```

其中 `case_0/teacher_target.pt` 與 `case_1/teacher_target.pt` 共 **439,738,688 bytes**。
它們與精簡 cache 中的原始兩案例重複，不能再算成兩筆獨立資料。

## 修復模型目前讀取與寫入的位置

### 訓練資料索引

```text
/ssd1/cactus8603/roco_main/operations/waft_kitti_visible_local_slim_repair_v2_20261027/data_cache
```

該目錄有 321 個檔案、共 **26,561,909 bytes**：

- `manifest.json`：全部 160 筆的總索引。
- `pair_*.json`：160 份逐筆 input／target／光流路徑描述。
- `normalization_*.pt`：160 份 σ 與幾何資料。

目前修復訓練引用這份既有索引及上面的 KITTI 特徵封包，沒有再複製一套 input／target features。
完整資料綁定另存同層的 `data_manifest.json`。

### 模型與輸出

目前模型程式：

```text
/ssd1/cactus8603/roco_main/src/stablebridge/physical_repair/waft_feature_repair_local_slim_v1.py
```

目前訓練與輸出根目錄：

```text
/ssd1/cactus8603/roco_main/operations/waft_kitti_visible_local_slim_repair_v3_20261027/artifacts/student
```

| 相對上述根目錄的位置 | 用途 | 盤點時狀態 |
|---|---|---|
| `run/progress.json` | 目前訓練進度 | 保存快照為訓練中，1,230／1,600 updates |
| `run/updates.jsonl`、`run/epochs.jsonl` | 逐步與逐 epoch 紀錄 | 已存在，持續寫入 |
| `run/witness_0001.pt` | 首步 optimizer 算術核驗 | 已存在，94,330,886 bytes；不是完成模型 checkpoint |
| `run/final_student.pt` | 固定訓練完成後的模型權重 | 尚未產生 |
| `run/prediction_*.pt` | 最終修復特徵 | 0 檔 |
| `flow/repaired_flow_*.npz` | 修復特徵回注凍結 WAFT 的光流 | 0 檔 |

最終審查預定保存於：

```text
/ssd1/cactus8603/roco_main/operations/waft_kitti_visible_local_slim_repair_v3_20261027/terminal_review.json
```

盤點時尚未產生，不能把 target 的 EPE 改善或在線訓練 loss 當作修復模型已完成的光流改善。

## 結果與困難案例

| 內容 | 實際索引或報告 |
|---|---|
| KITTI 便宜 target 獨立審查 | `/ssd1/cactus8603/roco_main/operations/waft_gt_rgb_kitti_two_methods_v2_20261027/independent_EPE_review.json` |
| Clean 便宜 target 獨立審查 | `/ssd1/cactus8603/roco_main/operations/waft_gt_rgb_clean_method_search_v2_20261026/independent_EPE_review.json` |
| Final 共用配對 Clean 的核對 | `/ssd1/cactus8603/roco_main/operations/waft_cheap_target_cross_domain_summary_v1_20261027/cross_domain_review.json` |
| 昂貴 target 獨立審查 | `/ssd1/cactus8603/roco_main/operations/waft_dense_target_epe_continuation_v2_20261026/independent_CPU_audit.json` |
| 固定困難案例清單 | `/ssd1/cactus8603/roco_main/operations/waft_gt_rgb_hard_cases_v1_20261026/hard_cases.csv` |
| 困難案例完整來源索引 | `/ssd1/cactus8603/roco_main/operations/waft_gt_rgb_hard_cases_v1_20261026/hard_case_manifest.json` |
| 困難案例使用規則 | `/ssd1/cactus8603/roco_main/operations/waft_gt_rgb_hard_cases_v1_20261026/development_policy.json` |

固定困難案例每個 domain 20 筆，共 60 筆，是既有訓練資料的開發子集，沒有額外複製一套影像。
GitHub 上容易閱讀的結果另見 [結果文件](results.md)、[彙總 CSV](results/summary.csv)
與 [逐案例 CSV](results/cases.csv)。

## 如何解讀數量與大小

- CSV 的 `file_count` 是套用 selector 與索引條件後的檔案路徑數；`unique_files` 依 device／inode 去除 hardlink 重複。
- `logical_bytes` 是檔案大小加總；`unique_logical_bytes` 去除 hardlink 重複，兩者皆非磁碟實際配置區塊數。
- 多列可能引用同一檔案，例如 Clean／Final 共用 target、總索引與分類小計；**不可直接加總整份 CSV**。
- 0 檔、路徑不存在、尚未完成與共用既有檔案會分別註明；預定位置不等於檔案已生成。
- 本次核對來源索引、路徑存在、檔案大小、接受條目與已有審查，不重新解碼全部影像或載入 tensor／模型。
- 清單涵蓋目前主線，沒有掃描全機磁碟，也不逐一列出舊研究分支的所有中間檔。

返回：[主線總覽](../README.md) · [修復模型](feature-repair.md) · [Target 製法](targets.md)。
