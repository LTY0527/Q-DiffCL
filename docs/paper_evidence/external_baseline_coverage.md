# External Baseline Coverage Audit

主表已覆盖 NoAug、Jitter、Scaling、Jitter+Scaling、Uniform Diffusion、FreRA、FINAL_QDIFFCL/DCBR，满足传统、频域自动增强和内部扩散对照的最低覆盖。

- 值得补：一个 recent automated time-series augmentation baseline，但仅在能固定 shared TCN/Hard-SupCon/split/probe 时进入主表。
- 值得补：一个 diffusion-based industrial contrastive baseline；若官方实现绑定不同 encoder/objective，则只做 method-native supplementary，并明确不可直接公平排名。
- 不再为超过某个 baseline 数字扩大搜索或修改主方法。
