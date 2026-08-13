# 3W Clean Baseline 类别塌缩诊断与平衡修复

最终状态：`3W_DATA_PROTOCOL_REAUDIT_REQUIRED`

本地 base commit 为 `831d12c`。本地 D0 是 Seed 7、Hard SupCon 20 epochs、Macro-F1 0.2882；提示词所述 Seed 42/35 epochs/0.2382 checkpoint 在本地不存在，因此没有伪造或重训 D0，D1 直接复用本地 Seed 7 checkpoint，D2/D3 也固定 Seed 7。

## Audit A：冻结 window 分布

| split | class | WELL | instance | window | window % |
|---|---:|---:|---:|---:|---:|
| train | 0/1/2/4/5/7/8/9 | 4/1/3/4/1/4/6/9 | 253/2/16/152/6/33/8/35 | 213479/1596/2379/33921/9604/206814/72456/5575 | 39.111/0.292/0.436/6.215/1.760/37.890/13.275/1.021 |
| validation | 0/1/2/4/5/7/8/9 | 3/1/2/2/1/1/1/3 | 247/1/4/155/4/2/1/8 | 140051/300/764/34634/521/1360/19776/260 | 70.852/0.152/0.387/17.522/0.264/0.688/10.005/0.132 |
| test | 0/1/2/4/5/7/8/9 | 2/1/2/1/1/1/2/3 | 94/1/2/36/1/1/5/14 | 100696/451/306/8170/308/1227/26405/830 | 72.761/0.326/0.221/5.904/0.223/0.887/19.080/0.600 |

largest/smallest window ratio：train 133.76，validation 538.66，test 329.07。所有 validation/test 类在 train 均有窗口；class 1 虽有 1,596 个训练窗口，但只来自 1 口 WELL，class 5 同样只来自 1 口训练 WELL。窗口数足以构造 batch，不等于跨 WELL 多样性足够。

## Audit B：普通 SupCon positive pair

D0 实际使用每类最多 4,000 个固定分层训练窗口，而不是直接 shuffle 545,824 个原始窗口。batch=256、Seed 7 的 110 个普通 shuffled batch 中：overall valid-anchor rate=1.000，所有内部 target 0..7 的 positive-anchor rate 均为 1.000，zero-positive rate 均为 0。因此 class 1/4/5/7 没有 positive pair 的假设被否定；它不是 D0 collapse 主因。

## Audit C：representation / probe

D0 embedding 的 mean/std/norm：train 0.2040/0.3520/1.9473，validation 0.2124/0.4002/1.9278，test 诊断样本 0.2380/0.4296/2.1826。平均 centroid distance 从 train 2.1553、validation 2.3123 降至 test 1.8691，提示跨 WELL representation shift。

D0 预测直方图（原始 class 0/1/2/4/5/7/8/9）：`66833/0/319/29213/490/111/37486/3941`。虽然 4/5/7 偶尔作为输出，但从未正确命中；class 1 完全不输出。

## D0-D3 统一比较

| Condition | Macro-F1 | Macro Recall | Fault Recall | Binary AUROC | Multiclass AUPRC | FAR | Early Recall | Mean Delay | Detection rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 当前 SupCon + unweighted probe | 0.2882 | 0.3422 | 0.7034 | 0.7426 | 0.3530 | 0.4473 | 0.9099 | 994.42 s | 0.2449 |
| D1 当前 encoder + balanced probe | 0.3312 | 0.4314 | 0.7146 | 0.7258 | 0.3480 | 0.6335 | 0.9226 | 1713.28 s | 0.3673 |
| D2 balanced SupCon + balanced probe | 0.2227 | 0.4423 | 0.9661 | 0.7452 | 0.3431 | 0.9095 | 0.9597 | 24.73 s | 0.9796 |
| D3 balanced CE sanity | 0.2646 | 0.4366 | 0.9532 | 0.7832 | 0.3267 | 0.6935 | 0.9520 | 16.89 s | 0.9592 |

D1 用 sampled train windows 的 sqrt inverse-frequency weight，仅恢复 class 5（Recall 0.8929/F1 0.5126），class 1/4/7 仍为零；且 FAR 恶化 0.1862。因此 probe imbalance 是因素但不是 dominant，也不是可接受修复。

D2 使用 P×K=`8×32`、125 batches/epoch。每类计划 4,000 samples；oversampling factor 为 class 0/1/2/4/5/7/8/9=`1.000/2.506/1.681/1.000/1.000/1.000/1.000/1.000`，最大值低于 3×上限，训练稳定。但 D2 Macro-F1 下降、FAR 达 0.9095，class 1/4/5 仍为零，说明 balanced batch 没有修复 representation。

D3 直接 balanced CE 仍有 class 4/5/7 三类 zero recall，并以 FAR 0.6935 换取 fault recall；按停止规则说明问题更偏向冻结数据协议下的跨 WELL/domain/feature-availability/label-window 可迁移性，而非单纯 SupCon 或 linear probe。

## 结论与停止线

`3W_DATA_PROTOCOL_REAUDIT_REQUIRED`。当前未获得 `3W_BALANCED_CLEAN_BASELINE_1SEED_GO`，不具备进入 Uniform Diffusion 1-Seed 的条件。下一阶段只能做 train/validation 驱动的协议再审计，优先检查单训练 WELL 类别、不同 WELL 的可用传感器/mask shortcut、设备域差异、normal window 来源和 class 4/7 的阶段/工况混杂；不得根据 test 继续调权重、sampler 或模型。

机器可读的 D0-D3 汇总、per-class CSV、混淆矩阵、预测直方图、embedding audit、训练历史与 sampler 统计位于 `outputs/3w_clean_collapse_diagnosis_seed7/`，大型 checkpoint 不提交 Git。
