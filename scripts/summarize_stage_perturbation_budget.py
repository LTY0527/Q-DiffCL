from __future__ import annotations
from pathlib import Path
import numpy as np

def flat(x):
    t=x["test"]; m=t["metrics"]; return {"macro_f1":m["macro_f1"],"auprc":m["auprc"],"fault_recall":m["fault_recall"],"far":m["far"],"early_recall":t["stages"]["early"]["recall"],"mean_delay":t["detection_delay"]["mean_delay_samples"]}
def mean_std(x): a=np.asarray(x,float); return {"mean":float(a.mean()),"std":float(a.std(ddof=1)) if len(a)>1 else 0.}
def seed7_gate(m,g,budget_valid):
    r,b=m["R1"],m["B3"]; keep=b["macro_f1"]>=r["macro_f1"]-g["maximum_macro_f1_drop"] and b["far"]<=r["far"]+g["maximum_far_increase"] and b["fault_recall"]>=r["fault_recall"]-g["maximum_recall_drop"] and b["auprc"]>=r["auprc"]-g["maximum_auprc_drop"]
    gain=b["early_recall"]>=r["early_recall"]+g["minimum_early_gain"] or b["mean_delay"]<=r["mean_delay"]-g["minimum_delay_reduction_samples"]
    status="STAGE_BUDGET_IMPLEMENTATION_INVALID" if not budget_valid else "STAGE_PERTURBATION_BUDGET_SEED7_GO" if keep and gain else "STAGE_PERTURBATION_BUDGET_SEED7_NO_GO"
    return status,{"core_preserved":bool(keep),"early_or_delay_gain":bool(gain),"budget_order_valid":bool(budget_valid)}
def three_gate(sm,g):
    means={m:{k:np.mean([sm[s][m][k] for s in sm]) for k in sm[next(iter(sm))][m]} for m in ("R1","B3")}; r,b=means["R1"],means["B3"]
    keep=b["macro_f1"]>=r["macro_f1"]-g["maximum_mean_macro_f1_drop"] and b["far"]<=r["far"]+g["maximum_mean_far_increase"] and b["fault_recall"]>=r["fault_recall"]-g["maximum_mean_recall_drop"] and b["auprc"]>=r["auprc"]-g["maximum_mean_auprc_drop"]
    ew=sum(sm[s]["B3"]["early_recall"]>sm[s]["R1"]["early_recall"] for s in sm); dw=sum(sm[s]["B3"]["mean_delay"]<sm[s]["R1"]["mean_delay"] for s in sm); gain=(b["early_recall"]>r["early_recall"] and ew>=2) or (b["mean_delay"]<r["mean_delay"] and dw>=2)
    cat={s:bool(sm[s]["B3"]["macro_f1"]<sm[s]["R1"]["macro_f1"]-g["catastrophic_macro_f1_drop"] or sm[s]["B3"]["far"]>sm[s]["R1"]["far"]+g["catastrophic_far_increase"] or sm[s]["B3"]["fault_recall"]<sm[s]["R1"]["fault_recall"]-g["catastrophic_recall_drop"] or sm[s]["B3"]["early_recall"]<sm[s]["R1"]["early_recall"]-g["catastrophic_early_drop"]) for s in sm}
    return ("STAGE_PERTURBATION_BUDGET_3SEED_GO" if keep and gain and not any(cat.values()) else "STAGE_PERTURBATION_BUDGET_3SEED_NO_GO"),{"core_preserved":bool(keep),"industrial_gain":bool(gain),"early_wins":ew,"delay_wins":dw,"catastrophic":cat}
def summarize(c,sr,fp,result=None,report_path=None):
    sm={s:{m:flat(v["methods"][m]) for m in ("R1","B3")} for s,v in sr.items()}; s7,a=seed7_gate(sm["7"],c["seed7_gate"],sr["7"]["budget_order_valid"])
    status,gate=(three_gate(sm,c["three_seed_gate"]) if len(sm)==3 else (s7,{"three_seed_skipped":True})); summary={m:{k:mean_std([sm[s][m][k] for s in sm]) for k in sm[next(iter(sm))][m]} for m in ("R1","B3")}
    v=result or {"markers":c["markers"],"status":status,"seed7_status":s7,"seed7_gate":a,"three_seed_gate":gate,"seed_results":sr,"seed_metrics":sm,"summary":summary,"fingerprints":fp,"three_seeds_completed":len(sm)==3,"c3_stopped":status!="STAGE_PERTURBATION_BUDGET_3SEED_GO"}
    if report_path: render(v,report_path)
    return v
def render(r,path):
    rows=[]
    for s,ms in r["seed_metrics"].items():
        for m,x in ms.items(): rows.append(f"| {s} | {m} | {x['macro_f1']:.4f} | {x['auprc']:.4f} | {x['fault_recall']:.4f} | {x['far']:.4f} | {x['early_recall']:.4f} | {x['mean_delay']:.2f} |")
    audits=[]
    for m in ("R1","B3"):
        a=r["seed_results"]["7"]["methods"][m]["effective_timestep_history"][0]; audits.append(f"| {m} | {a['normalized_l1']:.5f} | {a['stages']['normal']['normalized_l1']:.5f} | {a['stages']['early']['normalized_l1']:.5f} | {a['stages']['middle']['normalized_l1']:.5f} | {a['stages']['stable']['normalized_l1']:.5f} | {a['critical_frequency_l1']:.5f} | {a['noncritical_frequency_l1']:.5f} |")
    delta={k:r["seed_metrics"]["7"]["B3"][k]-r["seed_metrics"]["7"]["R1"][k] for k in r["seed_metrics"]["7"]["R1"]}
    mechanism=r["seed_results"]["7"].get("budget_mechanism",{})
    mechanism_rows=[]
    for m in ("R1","B3"):
        if m in mechanism:
            x=mechanism[m]; mechanism_rows.append(f"| {m} | {x['critical_fisher_retention']:.5f} | {x['representation_l2']:.5f} | {x['stage_representation_l2']['normal']:.5f} | {x['stage_representation_l2']['early']:.5f} | {x['stage_representation_l2']['middle']:.5f} | {x['stage_representation_l2']['stable']:.5f} |")
    text=f"""# 故障阶段显式扰动预算 MVP

> **STAGE_EFFECT_AUDIT / STAGE_PERTURBATION_BUDGET_MVP / FIXED_R1_BASELINE / EXPLORATORY_TEP_SUBSET / NOT_FOR_PAPER_FINAL_CLAIMS**

## 结论

Stage Effect Audit 为 `STAGE_TIMESTEP_EFFECT_WEAK`，因此执行唯一固定 beta：normal/early/middle/stable=`1.0/0.6/0.8/1.0`。Seed 7 状态：`{r['seed7_status']}`；最终：`{r['status']}`；3-Seed 完成：`{r['three_seeds_completed']}`。

| Seed | 方法 | Macro-F1 | AUPRC | Recall | FAR | Early | Delay |
|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

B3−R1：Macro-F1 `{delta['macro_f1']:+.5f}`，AUPRC `{delta['auprc']:+.5f}`，Recall `{delta['fault_recall']:+.5f}`，FAR `{delta['far']:+.5f}`，Early Recall `{delta['early_recall']:+.5f}`，Delay `{delta['mean_delay']:+.2f}`。`ΔFAR<0`、`ΔDelay<0` 才表示改善。

| 方法 | Overall L1 | Normal | Early | Middle | Stable | Critical L1 | Noncritical L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(audits)}

## Fisher 与表征距离

| 方法 | Critical Fisher retention | Overall repr L2 | Normal repr L2 | Early repr L2 | Middle repr L2 | Stable repr L2 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(mechanism_rows)}

预算顺序有效：`{r['seed7_gate']['budget_order_valid']}`。Seed 7 Gate：`{r['seed7_gate']}`；3-Seed Gate：`{r['three_seed_gate']}`。beta 只缩放训练 R1 residual，不进入 validation threshold、encoder、Probe 或 test 推理。

B3 虽提高 Macro-F1 并降低 FAR，但 Early Recall 明显下降且 Delay 恶化，未满足工业收益条件。当前 TEP test 已多轮查看，本结果仍是探索性而非论文最终无偏结论。NO-GO 后彻底停止 C3，不搜索新 beta/stage/horizon，不增加 C4/C5；下一步冻结 R1 并转向第二数据集、新未触碰协议、第二种退化与强基线。
"""; Path(path).write_text(text,encoding="utf8")
