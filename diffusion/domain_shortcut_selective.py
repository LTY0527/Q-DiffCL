from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .frequency_selective import continuous_alpha_bar


def domain_shortcut_timestep(fault_mask: np.ndarray, domain_mask: np.ndarray,
                             t_uniform:int=3,t_critical:int=1,t_noncritical:int=5)->np.ndarray:
    fault=np.asarray(fault_mask,dtype=np.float64);domain=np.asarray(domain_mask,dtype=np.float64)
    if fault.shape!=domain.shape or not np.isfinite(fault).all() or not np.isfinite(domain).all():raise ValueError("DSFD masks must be finite and aligned")
    if np.any((fault<0)|(fault>1)|(domain<0)|(domain>1)):raise ValueError("DSFD masks must lie in [0,1]")
    value=t_uniform-fault*(t_uniform-t_critical)+(1-fault)*domain*(t_noncritical-t_uniform)
    if np.any(value<t_critical-1e-8) or np.any(value>t_noncritical+1e-8):raise RuntimeError("DSFD timestep outside frozen bounds")
    return value.astype(np.float32)


def matched_domain_shortcut_variance(alpha_bars:torch.Tensor,fault_mask:np.ndarray,domain_mask:np.ndarray,
                                     preserve_dc:bool=True,t_uniform:int=3,t_critical:int=1,t_noncritical:int=5,
                                     tolerance:float=1e-7)->tuple[torch.Tensor,dict[str,Any]]:
    timestep_np=domain_shortcut_timestep(fault_mask,domain_mask,t_uniform,t_critical,t_noncritical);device=alpha_bars.device
    timestep=torch.as_tensor(timestep_np,device=device);fault=torch.as_tensor(fault_mask,dtype=torch.float32,device=device);domain=torch.as_tensor(domain_mask,dtype=torch.float32,device=device)
    initial=1-continuous_alpha_bar(alpha_bars,timestep);maximum=torch.full_like(initial,1-alpha_bars[t_noncritical])
    if preserve_dc:initial[:,0]=0;maximum[:,0]=0
    final=initial.clone();protected=fault>=.5;eligible=~protected
    if preserve_dc:eligible[:,0]=False
    uniform=torch.full_like(final,1-alpha_bars[t_uniform])
    if preserve_dc:uniform[:,0]=0
    target=uniform.sum();difference=target-final.sum();weights=(1-fault)*(1+domain)*eligible
    if abs(float(difference))>tolerance and eligible.any():
        capacity=((maximum-final).clamp_min(0) if float(difference)>0 else final)*eligible
        amount=torch.minimum(torch.abs(difference),capacity.sum());remaining=amount;active=eligible.clone()
        for _ in range(final.numel()+1):
            if float(remaining)<=tolerance or not active.any():break
            current=weights*active
            if float(current.sum())<=0:break
            proposal=remaining*current/current.sum();limit=(maximum-final).clamp_min(0) if float(difference)>0 else final
            change=torch.minimum(proposal,limit)*active;final=final+change if float(difference)>0 else final-change
            remaining=amount-torch.abs(final-initial).sum();active=active&(limit>tolerance)
    adjustment=final-initial;residual=torch.abs(target-final.sum());error=residual/target.clamp_min(torch.finfo(final.dtype).eps)
    audit={"dsfd_timestep":timestep_np,"initial_variance":initial.cpu().numpy(),"final_variance":final.cpu().numpy(),
           "budget_error_fraction":float(error),"residual_budget_mismatch":float(residual),
           "protected_variance_not_increased":bool(torch.all(adjustment[protected]<=tolerance)),
           "budget_adjustment_only_low_fault":bool(torch.all((torch.abs(adjustment)<=tolerance)|eligible)),
           "maximum_variance_respected":bool(torch.all(final<=(1-alpha_bars[t_noncritical])+tolerance)),
           "minimum_timestep_respected":bool(torch.all(timestep>=t_critical-tolerance)),
           "maximum_timestep_respected":bool(torch.all(timestep<=t_noncritical+tolerance)),
           "changed_bin_count":int((torch.abs(timestep-t_uniform)>tolerance).sum()),"finite":bool(torch.isfinite(final).all())}
    return final,audit
