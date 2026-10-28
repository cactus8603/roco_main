# Motion-aligned 特徵修復模型

目前模型是原生光流監督的 **3,000,576 參數、161 個 state tensors** 修復器。
訓練由完整凍結 WAFT 的五步輸出計算 loss，沒有 target-feature MSE 輔助項。
較早的 3,923,200 參數模型見 [歷史版本](legacy-feature-repair.md)。

## 輸入

修復器使用完整雙幀 F0、原始特徵的固定逐通道尺度 sigma，以及同次凍結 encoder 的
CNN pre S2／S4 與 DAv2 block 6／12。motion prior 由原始 WAFT 第五步 native flow／info
製作，包含對齊端點、有效支援、forward splat 權重與碰撞質量。

所有訊號都能在無 GT 推論取得。GT、昂貴教師、便宜教師、pair ID 與 dev split 均不進模型 forward。
目前模型只讀 baseline **第五步**；五步軌跡分支與錯誤原因／收益預測頭不屬於這個訓練版本。

## 處理方式

模型以雙幀共享的 directional convolutional U-shape 處理 stride 4／8／16 上下文，
再回到 stride 2 做細部修復。對齊後的跨幀資訊與自身特徵共同參與修正。

除非線性上下文路徑外，模型保留兩條原始幅度路徑：

- 原始 `F0/sigma` 的完整 64-channel linear head。
- stride 4 對齊後的 signed／absolute 跨幀差異，經 128→64 linear head 回到細網格。

這兩條路徑保留未被逐位置 LayerNorm 消除的幅度資訊。
對齊的 gather／splat、跨層位置與 padding 定義見 [架構文件](architecture.md)。

## 輸出

```text
spatial = spatial_raw - mean_full_padded_HW(spatial_raw)
delta = global + spatial
Fhat = F0 + original_sigma × delta
```

`global` 是每筆、每幀、每通道的空間常數；`spatial` 在完整 padded 網格上均值為零。
完整輸出與 F0 同形 `[B,2,64,Hp/2,Wp/2]`，兩幀及 padding 都可調整。
修正無 tanh、硬裁切或 ±0.25sigma 限制。

sigma 由原始雙幀 native centers 的 FP64 RMS 計算，並套用固定數值下限，
之後轉為 FP32、保持 detached。它不是由 GT、教師特徵或修復後特徵重新估計。
輸出 heads 零初始化，因此初始修復保持原始 F0；後續才學習非零 delta。

## 損失與訓練

Fhat 在 hidden 初始化之前放回 WAFT。每次從零光流執行完整五步，
沿用官方 Mixture-of-Laplace 目標與 `gamma=0.85` 的五步序列加權：

```text
loss = sum(gamma ** (4-k) × native_MoL(flow[k], info[k], GT, valid)
           for k in range(5))
```

損失使用原生實作的有效像素與 component reduction；無效 GT 在運算前排除，
非有限預測視為執行錯誤，而非悄悄從評分支援移除。
EPE 另行在完整 GT-valid 分母評分。MoL loss 降低不保證每筆或平均 EPE 單調下降。

固定 KITTI 設定為 160 TRAIN、32 epochs、batch 1、seed 20261028，
每個 epoch 使用獨立 seed 的 shuffle。AdamW peak LR 3e-4、betas (0.9,0.999)、
epsilon 1e-8、weight decay 0；warmup 160 updates，之後 cosine 到 peak LR 的 0.1，
以 FP64 計算 global norm 並 clip 至 1。無資料增強，dev 不參與 optimizer。
固定第 32 epoch 為最終模型，不按 dev 最佳分數選 checkpoint。

## 可以解釋到哪裡

模型可輸出 normalized residual、global／spatial 分量、對齊 raw difference、
matched support 與 forward collision mass。這些資訊可以描述修正行為與對齊支援，
並用於固定分支介入；它們沒有被校準成錯誤機率，也不直接辨識唯一錯誤根因。

查看熱圖、比較 delta 能量，不能替代完整五步光流的介入評估。
目前主流程沒有 hard accept gate、錯誤類別專家或依 GT 決定的推論策略。
