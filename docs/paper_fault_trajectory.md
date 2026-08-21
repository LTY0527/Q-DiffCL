# Early Fault Score Trajectory

使用既有 development checkpoint 对 TEP 全部 40 个 fault runs 只读重放。每个方法按其 onset 前窗口分数做 z-normalization；阴影为 run-level bootstrap 95% CI。该证据不是 Paper-final outer evaluation。

| Method | Last pre-onset z | First early z | Early windows 0–3 mean z | Raw probability 0–3 |
|---|---:|---:|---:|---:|
| SCALING | -0.097 | 19.650 | 31.062 | 0.565 |
| UNIFORM_DIFFUSION | 0.278 | 11.261 | 15.747 | 0.630 |
| FINAL_QDIFFCL | 0.307 | 13.366 | 20.364 | 0.596 |
| DCBR | -0.161 | 15.079 | 21.118 | 0.634 |

- 聚合覆盖所有可用 fault runs，没有按结果选择 representative cases。
- 对齐点 0 是第一个 fully post-fault window；transition windows 按冻结协议排除。
