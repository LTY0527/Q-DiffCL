# Paper-final Protocol Amendment

状态：`PAPER_FINAL_PROTOCOL_AMENDMENT_GO`。问题在首次 outer training/metric 之前发现并修复；`first_outer_metric_at = null`。

## 原因

旧 dry-run 以原始 event/class 文件存在性检查 coverage，而正式 runner 消费的是经过冻结 label mapping、transition exclusion 和 windowization 后的 `WindowRef.target`。因此部分包含 class-9 文件的 WELL 实际没有 target-3 窗口。

## Hash

- Old manifest: `8dbef73a56bfea0c6efc6af6e9e0fe131543206ca93801bd5d4335e899757e69`
- Revised manifest: `e4ef68e85b57f0ae90359f8ad7a83e2949a363fdff3945297c44bd102bd66207`

## Deterministic rule

每个 outer seed 使用独立 NumPy RNG；候选只按 WELL 分组、实际 WindowRef coverage、20/8/8 数量及 test-Jaccard 约束判定；保留仍满足新约束的旧 split，否则接受第一个 valid assignment。没有训练模型、读取 outer-test 或比较性能。

| Seed | Preserved | Candidates checked | Train target windows | Validation target windows | Test target windows |
|---:|---|---:|---|---|---|
| 31001 | False | 1783 | 127773/312/41824/3039 | 175518/764/51665/3360 | 111942/2373/25148/266 |
| 31002 | False | 526 | 249965/2541/64114/2999 | 72865/190/47866/266 | 92403/718/6657/3400 |
| 31003 | False | 1848 | 340282/2932/35833/1237 | 44470/223/12648/3253 | 30481/294/70156/2175 |

TEP split/hash 未改变；FINAL_QDIFFCL、DCBR、baseline、model seeds、metrics 与统计规则均未改变。
