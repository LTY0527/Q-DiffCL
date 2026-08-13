# 3W Primary Protocol（冻结版）

## 数据与范围

- Petrobras 3W Dataset 2.0.0，外部仓库 commit `45c7a9bc713656538e5179c5f2702cbf3beac5f0`。
- Primary 只使用真实 WELL；SIMULATED/DRAWN 不混入训练、验证或测试。
- 阶段 0 基线 commit：`a66bafe`。
- 提示词中的 `...bead5f0` 与实际仓库不符；WELL 原生缺失率实际审计为 54.17%，不是约 34%。本协议采用可复核的阶段 0 结果。

## Classes 与 split

Primary 原始类别为 `0,1,2,4,5,7,8,9`，内部连续 target 为 `0..7`。class 3、6 各只有 2 口带故障标签的真实 WELL，无法三路覆盖，排除到 Secondary。

固定 manifest：`configs/3w_primary_well_split.csv`；split seed 体系为 7，24/8/8 口 WELL。同一 WELL 的所有 instance 只属于一份。划分先于预处理和 windowing，并要求每个 Primary 类在每份中均存在 observation-level fault label，而不只检查目录名。

| split | WELL | instance | class 0/1/2/4/5/7/8/9 instance |
|---|---:|---:|---|
| train | 24 | 505 | 253/2/16/152/6/33/8/35 |
| validation | 8 | 422 | 247/1/4/155/4/2/1/8 |
| test | 8 | 154 | 94/1/2/36/1/1/5/14 |

各单元格为 `distinct WELL / instance / 有效 window`：

| split | class 0 | class 1 | class 2 | class 4 | class 5 | class 7 | class 8 | class 9 |
|---|---|---|---|---|---|---|---|---|
| train | 4/253/213479 | 1/2/1596 | 3/16/2379 | 4/152/33921 | 1/6/9604 | 4/33/206814 | 6/8/72456 | 9/35/5575 |
| validation | 3/247/140051 | 1/1/300 | 2/4/764 | 2/155/34634 | 1/4/521 | 1/2/1360 | 1/1/19776 | 3/8/260 |
| test | 2/94/100696 | 1/1/451 | 2/2/306 | 1/36/8170 | 1/1/308 | 1/1/1227 | 2/5/26405 | 3/14/830 |

## Feature、missing 与 normalization

先排除 `class/state/timestamp`。只用 train observation 计算 feature coverage；保留 coverage ≥5% 的 22 个 process feature：

`ABER-CKGL, ABER-CKP, ESTADO-DHSV, ESTADO-M1, ESTADO-M2, ESTADO-PXO, ESTADO-SDV-GL, ESTADO-SDV-P, ESTADO-W1, ESTADO-W2, ESTADO-XO, P-ANULAR, P-JUS-CKGL, P-JUS-CKP, P-MON-CKP, P-PDG, P-TPT, QGL, T-JUS-CKP, T-MON-CKP, T-PDG, T-TPT`。

排除 `P-JUS-BS/P-MON-CKGL/P-MON-SDV-P/PT-P/QBS`，原因均为 train coverage <5%。

每个保留 feature 的处理顺序固定为：train-only 0.1%/99.9% quantile clip → train median 填充 native missing → train mean/std 标准化。同时把 22 维 native observation mask 作为附加通道，最终输入 44 通道。统计量只 fit(train)，再 transform train/validation/test；未添加 MCAR，处理后检查无 NaN/Inf。完整参数保存在 output 的 `preprocessor.json`。

## Label 与 window

- raw 0 → normal，分类 target 0。
- raw `100+class` → early，分类 target 为对应 fault class，不创建新类。
- raw `class` → established，分类 target 为对应 fault class。
- unlabeled observation 不产生监督窗口。
- fault onset 是 instance 内第一条 early 或 established observation 的 timestamp。
- Early Recall 是 early 窗口中预测为任一非 normal 类的比例。
- Detection Delay 是 onset 后首次预测为任一非 normal 类的窗口末端时间减 onset；未检出的 instance 计入 missed/detection rate，不伪造 delay。
- 无完整 normal→early→established 的 instance 仍可参加普通分类；只有存在 onset 和 onset 后 fault 窗口时才参加 delay。没有 early 的 instance 不参加 Early Recall。

固定 window=64 秒、stride=32 秒（50% overlap）。64 是 2 的幂，便于下一阶段公平复用 FFT；不进行 window 后随机 split。

## 复用约束

后续 Clean、Uniform Diffusion、Frequency-Selective R1 和消融必须复用同一 manifest、22+22 通道、label/window/onset 口径和 train-only preprocessing。测试不得用于 feature、imputation、scaler、epoch、threshold 或方法选择。
