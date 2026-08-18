# DRFD Stage A 机制审计

最终状态：`DRFD_MECHANISM_GO`。

Stage A 仅使用训练域统计，未训练 encoder/probe，也未读取 test 数据参与可靠性拟合。

## 3W

- Gate：`GO`
- 可靠 critical / ambiguous / 可靠 non-critical：210 / 22 / 494
- changed bins：508
- mean |t_DRFD-t_R1|：0.201519
- budget error：0.000007%
- 安全不变量：{'protected_timestep_not_increased': True, 'protected_variance_not_increased': True, 'ambiguous_variance_not_increased': True, 'extra_only_reliable_noncritical': True, 'budget_adjustment_only_reliable_noncritical': True, 'maximum_variance_respected': True, 'finite': True}

## TEP

- Gate：`GO`
- 可靠 critical / ambiguous / 可靠 non-critical：452 / 112 / 1152
- changed bins：1201
- mean |t_DRFD-t_R1|：0.212490
- budget error：0.000000%
- 安全不变量：{'protected_timestep_not_increased': True, 'protected_variance_not_increased': True, 'ambiguous_variance_not_increased': True, 'extra_only_reliable_noncritical': True, 'budget_adjustment_only_reliable_noncritical': True, 'maximum_variance_respected': True, 'finite': True}
