# 阶段 0 审计记录

审计日期：2026-08-03（Asia/Shanghai）。

## 工作区与 Git

- 工作区：`E:/Code/Q-DiffCL`。
- 初始状态为空白、已初始化 Git，分支为 `main`，尚无 commit。
- 当前新增文件仍未提交；未设置远程仓库，未执行 commit、push 或任何 reset/restore/clean/stash。
- 四个外部 RData 文件仅被只读打开，没有复制、转换、移动、重命名或覆盖。

## 正式环境

正式解释器为 `E:/anaconda/envs/qdiffcl/python.exe`，Python 3.10.20，PyTorch 2.6.0+cu124，CUDA 12.4 可用，设备为 NVIDIA GeForce RTX 4060 Laptop GPU。`pyreadr` 0.5.6 与 `pytest` 9.1.1 均可导入。完整信息见 `docs/environment.md`。

## Rieth 2017 数据来源

- 数据版本：Rieth et al. 2017 扩展版 Tennessee Eastman Process。
- 来源：Harvard Dataverse。
- DOI：`10.7910/DVN/6C3JR1`。
- 只读目录：`E:/Datasets/TEP_Rieth2017/raw`。
- 四个文件合计 15,330,000 行；均无 NaN、无 Inf。

| 文件 | 字节数 | RData 对象 | shape | faultNumber | simulationRun | sample | 每个 fault 的 Run 数 | 每 Run 样本数 |
|---|---:|---|---:|---|---|---|---:|---:|
| `TEP_FaultFree_Training.RData` | 24,678,017 | `fault_free_training` | 250,000 × 55 | 0 | 1–500 | 1–500 | 500 | 500 |
| `TEP_FaultFree_Testing.RData` | 47,327,663 | `fault_free_testing` | 480,000 × 55 | 0 | 1–500 | 1–960 | 500 | 960 |
| `TEP_Faulty_Training.RData` | 494,063,194 | `faulty_training` | 5,000,000 × 55 | 1–20 | 1–500 | 1–500 | 每个 fault 500 | 500 |
| `TEP_Faulty_Testing.RData` | 836,882,037 | `faulty_testing` | 9,600,000 × 55 | 1–20 | 1–500 | 1–960 | 每个 fault 500 | 960 |

四个对象均含 `faultNumber`、`simulationRun`、`sample`、`xmeas_1`–`xmeas_41`、`xmv_1`–`xmv_11`。不同对象中整数列的存储类型略有差异，总体为 int32 与 float64。每个 `(faultNumber, simulationRun)` 内 sample 均唯一、连续，并从 1 开始。

## 故障边界确认

真实文件确认 Training sample 为 1–500、Testing sample 为 1–960，且均为 1-based。结合 DOI 对应公开协议，正式边界固化为：

- Training：sample 20 是最后正常点，sample 21 是第一个故障点。
- Testing：sample 160 是最后正常点，sample 161 是第一个故障点。
- 代码统一使用 `sample < first_faulty_sample` 为正常、`sample >= first_faulty_sample` 为故障，避免 `fault_onset: 20` 的 off-by-one 歧义。

状态由 `BLOCKED: DATASET_PROTOCOL_UNCONFIRMED` 更新为 `READY_FOR_REAL_DATA_SMOKE_TEST`，但不代表可直接进行正式科研实验。

## run_uid 与划分审计

`simulationRun` 在不同 fault、正常/故障数据以及 Training/Testing 中重复，不能作为全局唯一键。当前定义为：

```text
run_uid = source_split:state_or_fault_number:simulation_run
```

例如 `training:normal:0001`、`training:fault_01:0001`、`testing:fault_01:0001`。split manifest、窗口 ID 和退化 sample ID 均使用或包含 `run_uid`。

官方 Testing 全部进入最终 test。Training 先按 `faultNumber` 分层，再以固定 seed 7 做 Run 级 80%/20% train/validation 划分；完整数据下正常类及每个 fault 分别为 400/100 个 Run。scaler 只在 train Run 上拟合。

## 数据泄漏风险与保护

- 禁止先滑窗后划分；窗口不会跨 Run 或集合。
- 官方 Testing 不参与训练、验证、scaler、阈值或 checkpoint 选择。
- 二分类标签固定为 normal=0、fault 1–20=1；故障注入前仍为 normal。
- transition window 显式记录并统计，主协议默认 `exclude_transition`。
- 退化只在划分、训练集标准化及窗口化之后发生，并基于 `run_uid` 可追溯的窗口 ID 确定性生成。
- 所有模型必须复用同一 split manifest。

## 测试

加入扩散诊断模块后的最终完整测试结果为 `31 passed in 128.03s`，failed=0，skipped=0。真实数据集成测试逐个载入文件，没有同时复制四个完整 DataFrame。
