# CDVS 机制审计

结论：`CDVS_MECHANISM_GO`。未训练。

## 3W

- safe_prob：{'min': 0.0, 'p25': 0.7272727489471436, 'median': 0.9090909361839294, 'p75': 1.0, 'max': 1.0}
- unsafe R1 non-critical bins：334
- interior safe_prob fraction：61.98%
- 与 DRFD mean |Δt|：0.106985
- budget error：0.000000%
- invariants：{'protected_timestep_not_increased': True, 'protected_variance_not_increased': True, 'unsafe_variance_not_increased': True, 'budget_adjustment_only_safe_noncritical': True, 'maximum_variance_respected': True, 'finite': True}

## TEP

- safe_prob：{'min': 0.3333333432674408, 'p25': 1.0, 'median': 1.0, 'p75': 1.0, 'max': 1.0}
- unsafe R1 non-critical bins：304
- interior safe_prob fraction：21.21%
- 与 DRFD mean |Δt|：0.199456
- budget error：0.000000%
- invariants：{'protected_timestep_not_increased': True, 'protected_variance_not_increased': True, 'unsafe_variance_not_increased': True, 'budget_adjustment_only_safe_noncritical': True, 'maximum_variance_respected': True, 'finite': True}
