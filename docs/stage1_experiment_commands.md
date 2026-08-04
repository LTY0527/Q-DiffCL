# 阶段 1 实验命令

所有命令从 `E:/Code/Q-DiffCL` 执行，并显式使用：

```powershell
$python = "E:\anaconda\envs\qdiffcl\python.exe"
```

截至 2026-08-04，第 1–4 项已执行；第 5 项及之后没有运行。初次扩散第二闸 `NO-GO` 后已完成最小训练/采样修复，状态 C 只允许固定子集 Idea retest，正式基线扩展仍暂停。

## 1. 环境验证（已执行）

```powershell
& $python -c "import sys; print(sys.executable)"
& $python --version
& $python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('torch_cuda=', torch.version.cuda); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
& $python -c "import pyreadr, pandas, numpy, sklearn, yaml, pytest; print('核心依赖导入成功')"
```

- 配置：无。
- 数据：不读取数据。
- 科研结果：否。
- 设备：CPU；仅 CUDA 查询涉及 GPU 状态。
- 输出：控制台环境信息。
- 耗时：很短；无需估算正式训练时间。

## 2. 数据协议检查（已执行）

```powershell
& $python -m pytest -q tests/test_tep_adapter.py -m integration
```

- 配置：`configs/tep_template.yaml`。
- 数据：四个真实 RData，只读、逐文件加载。
- 科研结果：仅数据结构与协议证据，不是模型结果。
- 设备：CPU。
- 输出：pytest 结果；不生成数据副本。
- 耗时：取决于磁盘与内存；当前完整测试实测为 128.03 秒。

## 3. 完整测试（已执行）

```powershell
& $python -m pytest -q
```

- 配置：Debug 与真实协议测试配置。
- 数据：合成夹具 + 真实 RData 集成检查。
- 科研结果：否。
- 设备：CPU，模型测试可检测 CUDA 但不依赖长训练。
- 当前输出：`31 passed in 128.03s`。

## 4. 真实数据 Smoke Test（已执行）

```powershell
& $python -m scripts.run_clean_baseline --config configs/tep_template.yaml
```

- 配置：`configs/tep_template.yaml`，每个 normal/fault 层最多 2 个 Run、1 epoch。
- 数据：真实 Rieth 2017 小子集。
- 科研结果：否；标记为 `REAL_DATA_SMOKE_TEST`、`NOT FOR SCIENTIFIC COMPARISON`。
- 设备：`device: auto`，当前预计使用 RTX 4060 Laptop GPU。
- 输出：`outputs/real_data_smoke/ce-clean-seed-7/`。
- 实测：GPU 训练与评价 4.37 秒，包含 RData 读取/预处理的脚本总耗时 82.15 秒，峰值 GPU 已分配显存 18.09 MiB。

## 5. Clean TCN + CE 单 seed（正式配置待审定，未执行）

```powershell
& $python -m scripts.run_clean_baseline --config configs/tep_clean_tcn_ce.yaml
```

- 配置：`configs/tep_clean_tcn_ce.yaml` 尚未创建；必须在 Smoke Test 后确定 window length、stride、batch size、epochs 并去除 Smoke 子集限制。
- 数据：真实完整数据。
- 科研结果：完成固定协议和验证选择后才可作为正式结果。
- 设备：预计 GPU。
- 输出：正式配置中指定的独立目录。
- 耗时：必须根据第 4 项实测后估算。

## 6. Clean TCN + CE 多 seed（未执行）

使用审定后的正式配置复制为多个 seed 配置，确保除 `random_seed/output_dir` 外协议一致，再逐个运行：

```powershell
& $python -m scripts.run_clean_baseline --config configs/tep_clean_tcn_ce_seed7.yaml
& $python -m scripts.run_clean_baseline --config configs/tep_clean_tcn_ce_seed17.yaml
& $python -m scripts.run_clean_baseline --config configs/tep_clean_tcn_ce_seed27.yaml
```

- 数据：真实完整数据；科研结果：是，但前提是配置冻结。
- 设备：预计 GPU；输出目录按 seed 隔离。
- seed 集合当前只是命令结构示例，正式采用前需审定，不能视为已确定论文设置。

## 7. CE robustness baseline（未执行）

```powershell
& $python -m scripts.run_robustness_baseline --config configs/tep_robustness.yaml
```

- 配置尚未创建；必须复用同一 split manifest 和固定退化 realization。
- 数据：真实完整数据；科研结果：协议冻结后才是。
- 设备：预计 GPU；耗时待 Smoke Test 实测。

## 8. Hard SupCon baseline（未执行）

```powershell
& $python -m scripts.run_contrastive_baseline --config configs/tep_supcon.yaml --mode linear_probe
& $python -m scripts.run_contrastive_baseline --config configs/tep_supcon.yaml --mode fine_tune
& $python -m scripts.run_contrastive_baseline --config configs/tep_supcon.yaml --mode joint
```

- 配置尚未创建；必须与 CE 使用相同 manifest、窗口和退化输入。
- 数据：真实完整数据；科研结果：协议冻结后才是。
- 设备：预计 GPU；耗时待实测。

## 9. 结果汇总（未执行）

```powershell
& $python -m scripts.summarize_results outputs/tep --output outputs/tep/summary.json
```

- 数据：已有实验元数据，不直接读取 RData。
- 科研结果：只汇总已通过协议审查的正式运行。
- 设备：CPU。
- 输出：`outputs/tep/summary.json`。
- 只有 manifest、退化 ID 和配置一致的运行才能计算 performance retention、drop rate 与 SupCon gain。
