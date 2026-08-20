# Efficiency

| Dataset | Method | Training s | Peak GPU MiB | Aug. params | Inference params |
|---|---|---:|---:|---:|---:|
| 3W | SCALING | 282.6 | 1815.5 | 0 | 0 |
| 3W | UNIFORM_DIFFUSION | N/A | N/A | 0 | 0 |
| 3W | FINAL_QDIFFCL | N/A | N/A | 0 | 0 |
| 3W | FRERA | 64.7 | 1827.1 | 66 | 0 |
| 3W | DCBR | N/A | N/A | 0 | 0 |
| TEP | SCALING | 249.3 | 1484.3 | 0 | 0 |
| TEP | UNIFORM_DIFFUSION | 286.7 | 76.2 | 0 | 0 |
| TEP | FINAL_QDIFFCL | 244.5 | 76.2 | 0 | 0 |
| TEP | FRERA | 228.3 | 1498.6 | 66 | 0 |
| TEP | DCBR | 244.5 | 76.2 | 0 | 0 |

N/A 表示原实验复用了 checkpoint 且未保存可靠 wall-clock/peak-memory；不做事后估算。DCBR 推理新增参数为 0。
