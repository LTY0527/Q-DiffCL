# 项目环境说明

## 正式环境

- Conda 环境名：`qdiffcl`
- Python 解释器：`E:/anaconda/envs/qdiffcl/python.exe`
- Python：3.10.20
- PyTorch：2.6.0+cu124
- PyTorch CUDA：12.4
- CUDA 可用：是
- 设备：NVIDIA GeForce RTX 4060 Laptop GPU
- `pyreadr`：0.5.6
- `pytest`：9.1.1
- 项目安装：`q-diffcl-baselines==0.1.0`，editable 模式

不得为本项目测试切换到其他解释器，也不得无必要重新创建环境或升级 PyTorch。

## 激活与验证

```powershell
conda activate qdiffcl
$python = "E:\anaconda\envs\qdiffcl\python.exe"
& $python -c "import sys; print(sys.executable)"
& $python --version
& $python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('torch_cuda=', torch.version.cuda); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
& $python -c "import pyreadr, pandas, numpy, sklearn, yaml, pytest; print('核心依赖导入成功')"
```

## 当前验证状态

2026-08-03 使用正式解释器执行完整测试：

```text
31 passed in 128.03s (0:02:08)
```

其中包含依次读取四个真实 RData 的集成测试、质量加权 SupCon、扩散数学回算、DDPM clamping/确定性采样、one-batch overfit 和 gate 决策测试。`failed = 0`，`skipped = 0`。CUDA 当前可用。
