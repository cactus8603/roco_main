# 執行指南

目前主流程是 **3.00M motion-aligned 修復器＋凍結 WAFT 原生五步光流損失**。
安裝、資料、訓練、續訓、評估與推論的完整步驟已整理至：

**[原生光流訓練完整指南](native-training.md)**

入口命令為 `waft-repair native-split-kitti`、`native-prepare`、`native-train`、
`native-evaluate`、`native-infer`。此流程不需製作離線 target features，
也不需 Mamba／causal-conv1d extension。

較早的 `prepare`／`train`／`infer-images` 命令仍提供歷史 target-feature MSE 流程，
其 3.92M 模型與目前模型不同，請見 [歷史完整指南](legacy-running.md)。
不要將舊 `train` 命令當作目前的原生光流訓練，也不要互換兩者 checkpoint。
