# Q-DiffCL 阶段 0—1 收尾报告

当前协议状态：`REAL_DATA_SMOKE_TEST_PASSED`；修复后扩散诊断状态：`DIFFUSION_RECOVERY_READY_FOR_IDEA_RETEST`

| 模块 | 状态 | 数据类型 | 是否通过测试 | 是否可用于科研结论 | 下一步 |
|---|---|---|---|---|---|
| 阶段 0—1 工程实现 | 已完成 | 合成 + 真实数据适配 | 是 | 否，仅说明工程可运行 | 执行真实数据 Smoke Test |
| 合成数据 Debug | 已完成 | `SYNTHETIC` | 是 | 否 | 保留为回归测试 |
| Rieth 真实数据读取 | 已完成 | 真实 RData | 是 | 仅可用于数据结构结论 | 执行小规模 Smoke Test |
| 数据边界与 run_uid 协议 | 已完成 | 真实协议 | 是 | 可作为实验协议 | 固定清单并复核产物 |
| 真实数据 Smoke Test | 已完成 | 真实数据小子集 | 是 | 否 | 保留为工程回归 |
| 正式 CE/Hard SupCon 基线 | 尚未开始 | 真实完整数据 | 尚未执行 | 否 | Smoke Test 后审定正式配置 |
| 阶段 2 扩散恢复 | 尚未开始 | 未定 | 否 | 否 | 正式基线稳定后再开始 |

## 环境结论

项目正式解释器为 `E:/anaconda/envs/qdiffcl/python.exe`：Python 3.10.20、PyTorch 2.6.0+cu124、CUDA 12.4，NVIDIA GeForce RTX 4060 Laptop GPU 可用。`pyreadr`、pandas、NumPy、scikit-learn、PyYAML 和 pytest 均可导入。没有重建环境或升级 PyTorch。

## 真实数据结论

四个 Rieth et al. 2017 RData 已由正式环境依次读取，共 15,330,000 行、每表 55 列，NaN=0、Inf=0。字段为 `faultNumber`、`simulationRun`、`sample`、`xmeas_1`–`xmeas_41`、`xmv_1`–`xmv_11`。

- FaultFree Training：250,000 行，500 Run × 500 sample，sample 1–500。
- FaultFree Testing：480,000 行，500 Run × 960 sample，sample 1–960。
- Faulty Training：5,000,000 行，20 fault × 500 Run × 500 sample。
- Faulty Testing：9,600,000 行，20 fault × 500 Run × 960 sample。

真实数据确认 sample 为 1-based。结合 Harvard Dataverse DOI `10.7910/DVN/6C3JR1` 的公开协议，边界无歧义地定义为：

- Training：20 为最后正常 sample，21 为第一个故障 sample；
- Testing：160 为最后正常 sample，161 为第一个故障 sample；
- 实现只使用 `sample >= first_faulty_sample` 判为故障。

因此旧状态 `BLOCKED: DATASET_PROTOCOL_UNCONFIRMED` 已解除。

## run_uid 与任务协议

第一轮任务为 `binary_fault_detection`，normal=0，fault 1–20=1。`simulationRun` 会在不同 fault、状态和官方来源集合中重复，最终唯一键为：

```text
run_uid = source_split:state_or_fault_number:simulation_run
```

例如 `training:normal:0001` 与 `testing:fault_01:0001`。窗口元数据保存 `run_uid/start_sample/end_sample/faultNumber/is_transition/final_label/excluded`；窗口 ID 与确定性退化 ID 均可追溯到 `run_uid`。

## 划分、标准化与 transition

官方 Testing 完整保留为最终 test，不进入 train/validation。官方 Training 按 `faultNumber` 分层，以固定 seed 7 在 Run 级做 80% train / 20% validation。完整数据下正常状态和每个 fault 均为 400 个 train Run、100 个 validation Run。所有模型共享同一 split manifest。

StandardScaler 只在 train Run 上拟合。每个集合独立窗口化，禁止先滑窗后划分。主协议使用 `exclude_transition`，同时保留并测试 `label_by_last_step`、`label_by_fault_ratio`、`transition_class`。transition 数量、排除数量、比例和排除前后分布均写入元数据。

## 代码更新

- `datasets/protocol.py`：显式 `first_faulty_sample`、二分类标签、`run_uid`、按 fault 分层的 Training Run 划分、可追溯窗口元数据。
- `datasets/tep.py`：Harvard Dataverse 配置校验、无歧义边界校验、全局唯一 Run 转换、Smoke Test 每层限量读取接口。
- `scripts/common.py`：官方 Training/Testing 路由、分层 manifest、train-only scaler、真实数据 Smoke Test 小子集支持。
- `configs/tep_template.yaml`：更新为安全可执行的真实数据 Smoke Test 配置，明确标记 `REAL_DATA_SMOKE_TEST` 与 `NOT FOR SCIENTIFIC COMPARISON`。
- 测试：新增真实 RData、21/161 边界、run_uid 碰撞、官方 Testing 隔离、分层 manifest、窗口追溯测试。

## 文档更新

项目自有 Markdown 已直接中文化，没有创建 `_zh.md` 副本。已更新阶段审计、阶段报告、实验命令、环境说明、数据协议以及 `data/`、`outputs/` 下说明文件。没有修改第三方许可证、论文或外部数据说明。

## 测试结果

使用唯一正式解释器执行：

```text
31 passed in 128.03s (0:02:08)
failed = 0
skipped = 0
```

真实数据集成测试依次读取四个 RData，避免同时保留四份完整数据；PyTorch 模型、batch size 1、SupCon 无正样本和联合损失反向传播测试均在相同环境通过。

## 限制与下一步

- 真实数据 Smoke Test 已运行并通过；没有运行正式训练或阶段 2。
- `configs/tep_template.yaml` 的 window length、stride、batch size、epochs 和每层 Run 数仅为 Smoke Test 工程默认值，不是论文最终超参数。
- 全量正式配置尚未审定，不能标记为 `READY_FOR_FORMAL_EXPERIMENT`。
- 初次三道闸筛选的第二闸为 `NO-GO`；随后按 `docs/diffusion_recovery_debug_report.md` 修复训练/采样错配，当前只允许在相同固定子集上重新测试 Diffusion+Hard/Quality SupCon，仍不进入正式实验。
