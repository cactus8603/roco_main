# 舊架構與暫存清理紀錄

日期：2026-10-05

原則：只有能證明可重建且不屬於 frozen artifact manifest、套件 RECORD 或執行中工作
的 cache 才搬移。沒有刪除資料；所有搬移皆可由同一路徑結構移回 repository。

## 已移至同 SSD 的 tmp_trash

共搬移 228 個 `__pycache__` directories 與 2 個 `.pytest_cache` directories，約 29 MB：

- `/ssd1/cactus8603/tmp_trash/roco_action_architecture_cleanup_20261005`
  - 34 個 `__pycache__`、1 個 `.pytest_cache`，約 12 MB。
- `/ssd1/cactus8603/tmp_trash/roco_action_architecture_cleanup_20261005_audit`
  - 唯讀引用／manifest 稽核後確認的 190 個 `__pycache__`，約 15 MB。
- `/ssd1/cactus8603/tmp_trash/roco_action_architecture_cleanup_20261005_posttest`
  - 全套驗證期間重新產生的 3 個 `__pycache__`、1 個 `.pytest_cache`，約 1.9 MB。
- `/ssd1/cactus8603/tmp_trash/roco_action_architecture_cleanup_20261005_final_compile_cache`
  - 本輪新模組 `compileall` 產生的 1 個 `__pycache__`、3 個 `.pyc`；完整測試通過後搬移。

沒有寫入 `/ssd8`。其他 SSD 沒有找到能確定屬於本次 selector 重構、又可安全搬移的
項目，因此未建立空的 trash batch。

## 明確保留

- `selector_v2.py` 至 `selector_v7_v9_*`：仍被 package exports、runtime modules、直接
  tests 與 frozen source hashes 使用；不是可移除舊版。
- `research/selector_v7_redesign_20261003`：仍被 tests、review/authority records 與
  source-closure hashes 引用。只搬其中未被 artifact manifest 封存的 bytecode cache，
  research source／receipt 全部保留。
- `experiments/E246_legacy7_action_qualification_v1`、
  `experiments/E249_legacy7_endpoint_mechanism_freeze_v1`：雖含 legacy 名稱，仍是
  successor-chain 與 frozen manifests 的依賴。
- `research/selector_p2_tiny_overfit_20261004`：是 factorized action-head 的 negative
  control evidence，不能用「目前零文字引用」判定為垃圾。
- `experiments/E274_legacy_strength_grid_screen_v1`：稽核時 `RUN_STATE=RUNNING` 且輸出
  持續成長；它也是新架構所需的 strength-grid 工作，完全未碰觸。
- `/ssd7/cactus8603/roco_action_qualification_20261004/E263/*`：仍由 repository symlinks、
  runtime 與 allowlist 使用。
- `.runtime_tmp` 中 9 個無外部文字引用的 outcome-open scout scripts：它們是唯一研究
  程式而非 cache，為保留可追溯性未搬移。

## 特意未搬的 cache

repository 目前仍有 23 個 `__pycache__` directories：

- `research/region_aware_action_learning_20261001/work_package_A/__pycache__`：其 paths／
  hashes 被 `output_manifest.json` 逐檔封存。
- `experiments/E242_recommended_restorer_route_v1/tools/gdown/**/__pycache__`：仍列在
  wheel `.dist-info/RECORD`。
- `experiments/E274_legacy_strength_grid_screen_v1/__pycache__`：對應執行中的 E274。

這些目錄即使名稱是 cache，也不能在沒有更新其 authority／package contract 前移動。
