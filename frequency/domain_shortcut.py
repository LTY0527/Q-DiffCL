from __future__ import annotations

from typing import Any

import numpy as np

from .domain_reliability import percentile_ranks


def build_domain_shortcut_score(features: np.ndarray, labels: np.ndarray, stages: np.ndarray,
                                domain_ids: np.ndarray, minimum_observations: int = 2) -> dict[str, Any]:
    """Fit an ICC-like identity score using train normal/prefault windows only."""
    features=np.asarray(features,dtype=np.float64);labels=np.asarray(labels);stages=np.asarray(stages);domain_ids=np.asarray(domain_ids,dtype=object)
    if not (len(features)==len(labels)==len(stages)==len(domain_ids)) or features.ndim!=3:
        raise ValueError("domain shortcut arrays must align as [N,C,F]")
    if not np.isfinite(features).all(): raise ValueError("domain shortcut features must be finite")
    normal=(labels==0)|(stages=="prefault")
    means=[];variances=[];valid=[];invalid=[];support={}
    for domain in np.unique(domain_ids):
        selected=normal&(domain_ids==domain);count=int(selected.sum());support[str(domain)]=count
        if count<int(minimum_observations):invalid.append({"domain_id":str(domain),"reason":"insufficient normal/prefault support","observations":count});continue
        means.append(features[selected].mean(0));variances.append(features[selected].var(0));valid.append(str(domain))
    if len(means)<2: raise ValueError("domain shortcut score requires at least two valid train domains")
    means=np.stack(means);variances=np.stack(variances);between=means.var(0);within=variances.mean(0)
    score=between/(between+within+1e-8);mask=percentile_ranks(score[None])[0]
    return {"fit_split":"train","normal_prefault_only":True,"domain_score":score.astype(np.float32),
            "domain_mask":mask.astype(np.float32),"between_domain_variance":between.astype(np.float32),
            "within_domain_variance":within.astype(np.float32),"valid_domain_ids":valid,"invalid_domains":invalid,
            "normal_prefault_support":support,"test_or_validation_used":False}
