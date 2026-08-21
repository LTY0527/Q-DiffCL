# Efficiency / Practicality

环境：NVIDIA GeForce RTX 4060 Laptop GPU；PyTorch 2.6.0+cu124；CUDA 12.4；Python 3.10.20。所有 benchmark 在同一进程、同一硬件上重复 `3` 次。

| Dataset | Method | Training s | Peak GPU MiB | Aug. ms / 1024 | Inference ms / 1024 | Total params | Aug. params | Inference add. params |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 3W | NO_AUG | 70.8 | 1787.0 | 1.28 ± 0.04 | 0.81 ± 0.15 | 10596 | 0 | 0 |
| 3W | UNIFORM_DIFFUSION | 68.3 | 1786.6 | 21.60 ± 1.79 | 0.81 ± 0.15 | 10596 | 0 | 0 |
| 3W | STRONGEST_TRADITIONAL | 66.0 | 1787.0 | 61.25 ± 1.49 | 0.81 ± 0.15 | 10596 | 0 | 0 |
| 3W | FRERA | 65.2 | 1789.9 | N/A | 0.81 ± 0.15 | 10596 | 66 | 0 |
| 3W | FINAL_QDIFFCL | 66.8 | 1786.6 | 28.71 ± 1.73 | 0.81 ± 0.15 | 10596 | 0 | 0 |
| 3W | DCBR | 66.8 | 1786.6 | 29.08 ± 2.85 | 0.81 ± 0.15 | 10596 | 0 | 0 |
| TEP | NO_AUG | 275.8 | 76.2 | 3.35 ± 0.14 | 983.82 ± 21.91 | 13410 | 0 | 0 |
| TEP | UNIFORM_DIFFUSION | 324.4 | 76.2 | 57.22 ± 0.96 | 983.82 ± 21.91 | 13410 | 0 | 0 |
| TEP | STRONGEST_TRADITIONAL | 261.1 | 76.2 | 43.26 ± 2.75 | 983.82 ± 21.91 | 13410 | 0 | 0 |
| TEP | FRERA | 206.3 | 147.5 | N/A | 983.82 ± 21.91 | 13410 | 66 | 0 |
| TEP | FINAL_QDIFFCL | 221.8 | 76.2 | 60.83 ± 5.00 | 983.82 ± 21.91 | 13410 | 0 | 0 |
| TEP | DCBR | 218.2 | 76.2 | 106.44 ± 7.24 | 983.82 ± 21.91 | 13410 | 0 | 0 |

- 3W DCBR `rho=1` 与 FINAL 逐元素等价，训练与 augmentation timing 直接复用 FINAL。
- FRERA 的训练时间/显存来自已有 shared-backbone canonical run；其 augmenter checkpoint 未单独保存，因此不事后伪造 augmentation-only timing。
- 推理阶段所有方法只保留同一 frozen TCN + Linear Probe；DCBR/FINAL/传统增强均新增 0 个推理参数。
- MACs/FLOPs 未报告：冻结环境没有稳定计数器，避免引入新依赖后得到不可比数字。
