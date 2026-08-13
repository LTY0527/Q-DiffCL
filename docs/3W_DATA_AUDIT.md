# 3W Dataset 阶段 0：Data Audit 与协议设计

最终判定：`3W_DATA_PROTOCOL_HOLD`

## 版本与范围

- Petrobras 3W Dataset：`2.0.0`
- 外部数据仓库 commit：`45c7a9bc713656538e5179c5f2702cbf3beac5f0`
- 完整读取 `2228` 个 Parquet instance、`76587318` 条 observation。
- `TRANSIENT_OFFSET=100`；101~109 是相应事件的 transient label，不是新故障类。
- 原始数据只读，未复制、移动、修改；本审计未添加 TEP 的 MCAR 30%。

## Class × Source

| class | instance | observation | WELL instance | distinct WELL | SIMULATED | DRAWN |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 594 | 12158183 | 594 | 9 | 0 | 0 |
| 1 | 128 | 9107107 | 4 | 3 | 114 | 10 |
| 2 | 38 | 737785 | 22 | 7 | 16 | 0 |
| 3 | 106 | 4949279 | 32 | 2 | 74 | 0 |
| 4 | 343 | 3689683 | 343 | 7 | 0 | 0 |
| 5 | 450 | 13301677 | 11 | 3 | 439 | 0 |
| 6 | 221 | 5882267 | 6 | 2 | 215 | 0 |
| 7 | 46 | 10284155 | 36 | 6 | 0 | 10 |
| 8 | 95 | 6995955 | 14 | 9 | 81 | 0 |
| 9 | 207 | 9481227 | 57 | 15 | 150 | 0 |

来源 instance 总数：WELL=1119，SIMULATED=1089，DRAWN=20。共有 `40` 口真实 WELL；其中 `30` 口含多个 instance，`13` 口覆盖多个 event class。

## Feature 与原生缺失

- 自动检测 `27` 个 process variable；其中 `23` 个至少存在一个有限观测，4 个全量缺失字段为 `['P-JUS-BS', 'P-MON-SDV-P', 'PT-P', 'QBS']`。`class`、`state`、timestamp 均不作为模型 feature。
- 多个字段存在明显极端哨兵/异常范围；不能仅凭 finite 判定有效。下一阶段清洗阈值与最终 feature 集必须仅由 train/equipment metadata 冻结。
- 所有统计来自全量审计，但未来 scaler、imputer、feature selection、D/E/S 只能在 training WELL 上拟合。
- 整体 native missing rate：`0.675726`。各 feature/class/source 明细见 CSV。
- Adapter 返回独立 observation mask；零填充只用于构造有限 tensor，不代表已冻结 imputation 策略。

## 序列、采样与频率 HARD CHECK

- 全局序列长度统计见 `3w_sequence_stats.csv`，并按 class/source 分组。
- 采样 interval mode 分布：`{'1.0': 2228}`。
- 所有 instance 内 interval 恒定：`True`。
- Frequency hard check：`True`。只有该项为真才可把窗口内 FFT bin 视为相同物理频率；即便为真，3W 的 D/E/S 与关键频率 mask 仍必须仅由 3W training split 重新估计，不能直接复用 TEP 的频率索引/mask。

## Label Audit

- raw `class` 的 normal/transient/event/NA 统计见 `3w_label_audit.csv`，逐类逐来源的实际阶段顺序见 `3w_label_sequence_audit.csv`。
- 事件类 `1, 2, 5, 6, 7, 8, 9` 在配置中声明 transient；3、4 未声明 transient。
- 当前只确认可以把 raw 0、100+class、class 分别解释为 normal、transient、event 候选阶段；NA 与异常时序必须保留。**本阶段不冻结最终 normal/early/established mapping。**

## 推荐协议

Primary：WELL-only、按 `well_id` 分组的 train/validation/test，建议起点 60/20/20；同一 WELL 的所有 instance 必须进入同一 split。候选 manifest 已生成，class coverage 为 `{'train': [0, 1, 2, 3, 4, 5, 7, 8, 9], 'validation': [0, 1, 2, 4, 5, 6, 7, 8, 9], 'test': [0, 1, 2, 3, 4, 5, 7, 8, 9]}`。所有拟合仅使用 train。

三路 real-only 均可覆盖的类别是 `[0, 1, 2, 4, 5, 7, 8, 9]`；class 3 和 6 各仅有 2 口真实 WELL，数学上无法同时覆盖 train/validation/test，这是当前 HOLD 的核心阻塞。

若候选划分不能让三份都覆盖目标类别：

- 方案 A（首选）：保留 real-only，把主任务限定为三份均有真实 WELL 覆盖的 class；其余 class 明示为 out-of-scope，不把 synthetic 混入 primary。
- 方案 B：保留全部 class，将 SIMULATED/DRAWN 作为单独的 secondary protocol/域迁移消融，绝不与 primary 结论混写。

## Loader Smoke Test

读取少量 WELL → native mask → window → float32 tensor → TCN forward 成功：`{'instances': ['WELL-00001_20140124083303', 'WELL-00002_20140126180030', 'WELL-00006_20170731170930'], 'input_shape': [6, 27, 64], 'input_dtype': 'torch.float32', 'mask_shape': [6, 27, 64], 'mask_dtype': 'bool', 'raw_last_labels': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'finite_tensor': True, 'logits_shape': [6, 10], 'native_missing_preserved_in_mask': True}`。未训练 scaler、probe、对比模型或扩散模型。

## 阶段结论

`3W_DATA_PROTOCOL_HOLD`。若为 HOLD，阻塞项是：采样间隔 HARD CHECK 未通过，或当前简单候选 well split 的 test 未覆盖 0~9；在冻结最终 split/label mapping 前不得开始完整训练。下一阶段应先冻结 label/window/imputation 与 class coverage 协议，然后只跑单 Seed clean/传统增强/Uniform diffusion/Frequency-Selective R1 最小比较；通过后才允许 3-Seed。
