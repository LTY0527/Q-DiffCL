# Rieth 2017 TEP 数据协议

## 数据身份

- 名称：Tennessee Eastman Process 扩展数据集
- 版本：`rieth_2017`
- 来源：Harvard Dataverse
- DOI：`10.7910/DVN/6C3JR1`
- 根目录：`E:/Datasets/TEP_Rieth2017/raw`
- 字段：`faultNumber`、`simulationRun`、`sample`、41 个 `xmeas`、11 个 `xmv`

四个 RData 的对象名、shape、范围和完整性汇总见 `docs/stage0_audit.md`。所有序列均以 sample 1 开始；Training 长度为 500，Testing 长度为 960。

## 第一轮任务与标签

第一轮真实任务固定为 `binary_fault_detection`：

- normal：0；
- fault 1–20：1。

Fault-free Run 始终为 0。Faulty Run 按时间点标注，绝不使用整条 Run 的 `faultNumber` 把故障前阶段误标为故障。

```text
Training: sample <= 20 为 normal，sample >= 21 为 fault
Testing:  sample <= 160 为 normal，sample >= 161 为 fault
```

配置使用 `last_normal_sample` 和 `first_faulty_sample` 两个显式字段；代码判定为：

```text
sample < first_faulty_sample  -> normal
sample >= first_faulty_sample -> fault
```

## Run 唯一标识

```text
run_uid = source_split:state_or_fault_number:simulation_run
```

正常状态使用 `normal`，故障使用两位编号 `fault_01`–`fault_20`，simulationRun 使用四位编号。例如：

```text
training:normal:0001
training:fault_01:0001
testing:fault_01:0001
```

该定义避免同号 Run 在不同 fault、状态或来源集合之间碰撞。split manifest 存储 `run_uid`；窗口 ID 形如 `run_uid:samples_START_END`；退化哈希使用该可追溯窗口 ID。

## Train/Validation/Test 协议

1. 官方 Testing 文件完整保留为最终 test，不从中抽取 validation。
2. 合并两个官方 Training 来源的独立 Run。
3. 按 `faultNumber` 分层，以 seed 7 在每层做 80% train / 20% validation。
4. 完整数据中，正常状态和每个 fault 各有 500 个 Training Run，因此每层为 400 train / 100 validation。
5. 划分后才生成时间点标签、拟合 scaler 和窗口化。
6. StandardScaler 只使用 train Run 的 clean 数据拟合。
7. validation/testing 不参与均值、标准差、阈值、原型或 checkpoint 训练。

Smoke Test 配置只取每层少量 Run；由于每层必须至少保留一个 validation Run，小子集比例仅用于工程连通性验证，不作为完整 80/20 统计结果。

## 窗口与 transition

窗口输入统一为 `[batch, channels, length]`。每个窗口记录：

- `run_uid`；
- `start_sample`；
- `end_sample`；
- `faultNumber`；
- `is_transition`；
- `final_label`；
- `excluded`。

同时包含 normal 与 fault 时间点的窗口是 transition window。支持：

- `exclude_transition`：主实验默认，排除并保留审计记录；
- `label_by_last_step`：按末时间点标注；
- `label_by_fault_ratio`：按配置阈值标注；
- `transition_class`：使用独立 transition 类。

每次窗口化保存总数、transition 数、排除数、比例以及排除前后类别分布，不允许静默删除。

## 当前状态

数据读取、边界、run_uid、分层划分、scaler 隔离和 transition 测试均已通过，真实数据 Smoke Test 也已完成。修复后的扩散诊断达到状态 C，只允许固定子集 Idea retest；当前仍不进入正式实验，正式 window length、stride、batch size、epochs 和多 seed 方案均未冻结。
