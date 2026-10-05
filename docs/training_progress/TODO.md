# Evidence TODO

最後更新：2026-10-06

## P0：判斷 SEA-RAFT / U2 結果是否成立

- [ ] 用同一 standard evaluator 比較 raw Spring-M 與 U2 best 的完整 Sintel Clean/Final
  EPE。
- [ ] 檢查 1/3/5 px outlier rate，判斷改善是全面降低誤差，或只改善少數大誤差像素。
- [ ] 報告 U2 best 與 last checkpoint 的差異，量化後段退化。
- [ ] 建立正確 `[0,255]` 輸入的受控訓練版本，和現行 `[0,1]` 版本做單因子比較。

## P0：驗證 SAM 與 action bank 的貢獻

- [ ] 比較 U2-main 與 no-uncertainty dummy，隔離 uncertainty observer 的貢獻。
- [ ] 比較 U2-main 與 9A-main，判斷 action-bank cycle 是否帶來可重現提升。
- [ ] 比較 U2-main 與 U2-SAM，判斷 SAM homography regularization 是否有效。
- [ ] 比較 U2-SAM 與 9A-SAM，確認 SAM 與 action bank 是互補還是互相干擾。
- [ ] 對 SAM 額外報告 segment-boundary EPE、fitted-region coverage 與 per-region gain。

## P1：WAFT cross-backbone replication

- [ ] 固定新 U2 實驗使用的 WAFT checkpoint、upstream commit 與 SHA-256。
- [ ] 以相同 split 訓練 WAFT-specific U0 observer 與 U1 bounded refiner。
- [ ] 執行 WAFT U2 main / no-uncertainty dummy。
- [ ] 執行 WAFT U2-SAM，保持和 SEA-RAFT 相同的 mask 與 held-out contract。
- [ ] 用 WAFT 原生標準評測比較 base、U2、U2-SAM 與 9A。
- [ ] 將既有 CESR-CTR WAFT 結果標為舊機制證據，不與新 U2 training gain 混報。

## P1：Uncertainty 與 action-selection 證據

- [ ] 同時整理 EPE、AUSE、Spearman、severe-error AUROC 與 calibration MAE。
- [ ] 檢查 calibration 是偏 over-confident 或 under-confident，而不只報絕對 MAE。
- [ ] 比較各 action arm 在 observer/flow phase 的效益。
- [ ] 驗證 uncertainty 較好的 checkpoint 是否真的帶來更好的 action choice。

## P1：泛化與統計可信度

- [ ] 加入至少 3 個 random seeds，報告平均值與變異。
- [ ] 比較 Spring-M 與 Sintel-M initialization。
- [ ] 將三場景 crop validation、公開 Sintel training split 與 hidden test 分開報告。
- [ ] 檢查 TSKH/Sintel upstream exposure，避免描述成完全零樣本泛化。

## 已有的可信證據

- [x] 同 protocol 下 raw Spring-M `3.1829` → U2 best `1.3556`，EPE 相對降低約 57.4%。
- [x] epoch 9/10 皆約 `1.3556`，最佳區間不是單輪孤立尖峰。
- [x] 後段 EPE 退化而 uncertainty ranking 改善，不同指標的最佳 checkpoint 不一致。
- [x] WAFT frozen KITTI confirmation 證明 equivariance risk localization 可跨 backbone。
- [x] WAFT frozen selective repair 得到正 clean/corrupt gain 並通過 frozen gates。
- [x] 標準評測與 Main/SAM 接力已備妥，可自動產生下一批實驗證據。
