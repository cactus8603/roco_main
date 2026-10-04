# StableBridge 共用核心

E01 已有可執行的共同 pipeline。Stereo／flow 共用演算法、資料與評分契約，使用**同一種 CroCo v2 架構、兩份分開的凍結任務權重**；新增候選接受器預設共用參數。

程式完成不等於研究收益已驗證。實際 S00／S01–S03 狀態及成績，以各 run 的 manifest、status、geometry／report 為準。

新[S04](../../experiments/E01_evidence_mechanism/studies/S04_support_conditioned_rematching/README.md)使用同一支援條件化重新匹配架構，**修復器權重先按task分開訓練**。強E0改為原3候選加9次原pair重估；Q僅離線評分。它獨立於既有接受器及已封存的S02 composer。

- [執行架構與能力邊界](../../docs/specs/stablebridge_runtime_v1_20260916.md)
- [架構契約](../../docs/specs/stablebridge_architecture_contract_v1_20260916.md)
- [可更新 Query／Action 契約](../../docs/specs/stablebridge_adaptive_query_contract_v2_20260916.md)
- [大實驗索引](../../experiments/README.md)

## 模組

| 模組 | 責任 |
|---|---|
| `artifacts` | 官方 main 權重取得、續傳、hash 與 provenance 檢查 |
| `backbones` | 凍結 CroCo forward、encoder features、原生位移與未校準 uncertainty |
| `data`／`geometry` | Spring provider、確定性退化、GT 分離、原圖／patch 座標及有效取樣 |
| `contracts`／`memory` | Query 身份、causal cutoff、來源版本、容量／讀取、去重、失效，以及風險／action／狀態轉移契約 |
| `matching`／`pipeline` | 共用選區、支援提案、current-target 重匹配、M0／M1 輸出 |
| `learning` | Signed-gain／harm 模型、scene-separated fit 與選擇後校準 |
| `replay` | 固定候選的 learned query 評估，與完整 dense／closed-loop 結果分開 |
| `evaluation`／`reporting` | Spring native min4、固定分母、scene macro、actual／oracle 分開 |
| `runner`／`cli`／`util` | 原子 case artifact、來源快照、時序重播、命令入口 |
| `workflow` | 八階段序列工作、來源凍結、hash 驗證、失敗即停與相同版本續跑 |
| `support_rematch` | S04可訓練支援接口、30候選、query-target特徵/RGB取樣與幾何限制 |
| `support_data` | S04強E0、固定支援點、獨立inference/GT caches與成本 |
| `support_experiment` | S04凍結backbone、四種模型訓練、密集評估與新scene隔離 |
| `support_reporting` | S04實際輸出對照、全有效區傷害、oracle診斷及固定confirmation gate |

演算法放在本目錄；`experiments/` 保存大實驗、子問題、設定與獨立 run。舊 `research/hard_region` 和其歷史結果保留原地。

## 關鍵語意

1. `SpringProvider.read_pair()` 不讀 GT；評分端在 `predict()` 後另外呼叫 `read_gt()`。Pipeline 拒絕混入標籤的 pair。
2. Reference bank 預設為 `identity / median3 / gaussian1`。H* 是離線 GT capacity 診斷；runtime 維護可更新的 query state，不保存永久「不可修復」標籤。
3. M0 不留歷史。M1 僅讀 cutoff 前的合法觀測；現在只保存 source feature，不把接受後的位移自動寫成可信支援。
4. 空間／歷史證據提出候選後，必須回原 target 取樣重評分。S01 v4 已加入只供診斷的雙向 pair cycle；正式 gate 否定它作為兩 task 的單一主 donor selector，歷史 descriptor bridge 仍未實作完整 cycle 幾何驗證。
5. Learned acceptor 先在 eligible candidates 中選 gain 最大者，再套 gain／harm 門檻；校準遵循同一選擇流程。校準失敗可產生 `reject_all`，無 OOD 安全保證。
6. GT-valid 位置的失敗輸出保留分母；report 明列 failure，不用刪除失敗 case 取得較佳平均。Stereo／flow、actual／oracle 分開。
7. `RiskEstimate` 與 `ActionAssessment` 綁定校準／evidence／revision 版本；`advance_query_state` 對 accept／retain／retract 扣除明示成本，且把輸出接受與支援資格分開。

## 使用入口

安裝 package 後可直接使用 `python`；未安裝時，在 repository 根目錄設定
`PYTHONPATH=src`。實際模型執行需由使用者提供資料集、官方 source 與 checkpoint registry。

```text
python -m stablebridge.cli geometry --run-dir <new_run> --model-profile main --device <device>
python -m stablebridge.cli run --config <config.json> --run-dir <new_run> --device <device>
python -m stablebridge.cli collect-supervision --runs <run_A> <run_B> --output <supervision.npz>
python -m stablebridge.learning --dataset <supervision.npz> --output <acceptor.pth>
python -m stablebridge.replay --run-dir <existing_run> --acceptor <acceptor.pth> --output-dir <new_replay>
python -m stablebridge.workflow --output-dir operations/<new_workflow> --device <device> --epochs 30
python -m stablebridge.cli report --run-dir <run>
python -m stablebridge.reporting --run-dir <run>
```

正式訓練與長批次使用 `scripts/run_stablebridge_detached.py`，將上述 worker command 放在 `--` 後方。Launcher 保存命令、日誌與獨立 systemd unit，`--dry-run` 只產生可審閱計畫。完整例子見執行架構文件。

已登錄的首輪設定：S01 [`e01_s01_spatial_v1.json`](../../configs/stablebridge/e01_s01_spatial_v1.json)、S02 [`e01_s02_temporal_pilot_v1.json`](../../configs/stablebridge/e01_s02_temporal_pilot_v1.json)／[`e01_s02_temporal_full_v1.json`](../../configs/stablebridge/e01_s02_temporal_full_v1.json)、S03 [`e01_s03_source_control_v1.json`](../../configs/stablebridge/e01_s03_source_control_v1.json)。

## 驗證與尚未完成範圍

CPU 測試位於 `tests/stablebridge/`，涵蓋來源／時間契約、座標與取樣、官方 scorer parity、候選更新、來源失效、學習 split／calibration、失敗報告與無 shell 的 launcher。真實模型與研究成效由 run artifact 另行驗證。

目前尚未完成完整輪廓／部分約束／反證、learned VOI／觸發、估計傳播後的自動撤回閉環、deadline-union oracle、M2／M3、共同 backbone 訓練，以及 E02／E03 正式矩陣。不要把既有 API 或合成測試當作上述研究已完成。
