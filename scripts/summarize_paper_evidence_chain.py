from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
import yaml


def _fmt(value: Any, digits: int = 4) -> str:
    return "N/A" if pd.isna(value) else f"{float(value):.{digits}f}"


def summarize(config: dict[str, Any]) -> dict[str, Any]:
    root=Path("docs/paper_evidence"); root.mkdir(parents=True,exist_ok=True)
    mechanism=pd.read_csv(root/"mechanism_ablation.csv"); dcbr=pd.read_csv(root/"dcbr_extension_ablation.csv")
    mechanism_validation=pd.read_csv(root/"mechanism_ablation_validation.csv")
    well=pd.read_csv(root/"3w_per_well.csv"); instances=pd.read_csv(root/"3w_instance_rescue_loss.csv")
    fault=pd.read_csv(root/"tep_fault_wise.csv"); bootstrap=pd.read_csv(root/"group_bootstrap.csv"); efficiency=pd.read_csv(root/"efficiency.csv")
    external=json.loads(Path(config["sources"]["external_manifest"]).read_text(encoding="utf-8"))["results"]
    locked=json.loads(Path(config["sources"]["dcbr_locked"]).read_text(encoding="utf-8"))["results"]
    def mechanism_table(dataset):
        current=mechanism_validation[mechanism_validation.dataset==dataset]
        lines=["| Method | Macro-F1 | AUPRC | FAR | Expected budget |","|---|---:|---:|---:|---:|"]
        for _,r in current.iterrows(): lines.append(f"| {r.variant} | {_fmt(r.macro_f1_mean)} ± {_fmt(r.macro_f1_std)} | {_fmt(r.auprc_mean)} | {_fmt(r.far_mean)} | {_fmt(r.expected_spectral_budget_mean,8)} |")
        return "\n".join(lines)
    component=mechanism[mechanism.method.isin(["UNIFORM","D_ONLY","E_ONLY","FINAL_DE"])]
    comp_lines=["| Dataset | Method | Macro-F1 | FAR | Early Recall |","|---|---|---:|---:|---:|"]
    for _,r in component.iterrows(): comp_lines.append(f"| {r.dataset} | {r.method} | {_fmt(r.macro_f1_mean)} ± {_fmt(r.macro_f1_std)} | {_fmt(r.far_mean)} | {_fmt(r.early_recall_mean)} |")
    mech_text=f"""# Paper Mechanism Evidence

## A1 Core mechanism

### 3W

{mechanism_table('3W')}

### TEP

{mechanism_table('TEP')}

三 seed validation-only 公平审计已通过。3W 上 Soft matched 相对 Uniform、Hard 和 unmatched 分别为 `+0.0612/+0.0409/+0.0512` Macro-F1，均 3/3 seeds 正向；TEP 上对应差值均约为 `-0.0024~-0.0029`。因此 soft allocation/budget matching 只支持“对 3W 有明确收益”的数据集依赖表述，不支持跨数据集普遍优越。

## A2 Semantic components

{chr(10).join(comp_lines)}

3W 中 D_ONLY 明显强于 E_ONLY，D 是主要 discriminative contributor；E 仅可描述为 complementary early-fault prior。TEP 的组件差异很小，不支持“D/E 同等必要”。

## A3 DCBR

见 `dcbr_extension_ablation.csv`。3W validation 选择 `rho=1`，保持 FINAL；TEP 选择 `rho=.75`，现有 development locked evidence 中相对 FINAL Macro-F1 `+0.0121`，但仍略低于 SCALING。GLOBAL_RHO_075 仅是 validation mechanism reference，不能与 locked-test 行直接做统计检验。
"""
    (root/"mechanism_evidence.md").write_text(mech_text,encoding="utf-8")

    final=well[well.method=="FINAL_QDIFFCL"]; worst=final.groupby("well_id").macro_f1.mean().sort_values(); rescue=instances.groupby("baseline")[["rescued_windows","lost_windows"]].sum()
    well_paired=[]
    for baseline in ("FRERA","JITTER_SCALING"):
        merged=final.merge(well[well.method==baseline],on=["seed","well_id"],suffixes=("_final","_baseline"))
        for _,row in merged.iterrows(): well_paired.append({"seed":int(row.seed),"well_id":row.well_id,"baseline":baseline,
            "delta_macro_f1":row.macro_f1_final-row.macro_f1_baseline,"delta_far":row.far_final-row.far_baseline})
    pd.DataFrame(well_paired).to_csv(root/"3w_per_well_paired.csv",index=False,encoding="utf-8-sig")
    class_rows=[]
    for method in ("FINAL_QDIFFCL","FRERA","JITTER_SCALING"):
        for seed in config["seeds"]["three_w"]:
            record=external[f"3W|{method}|{seed}"]["record"]
            for item in record["metrics"]["per_class"]:
                if int(item["original_class"]) in (2,8,9): class_rows.append({"method":method,"seed":seed,"class":item["original_class"],"f1":item["f1"],"recall":item["recall"],"support":item["support"]})
    class_frame=pd.DataFrame(class_rows); class_frame.to_csv(root/"3w_difficult_classes.csv",index=False,encoding="utf-8-sig")
    well_lines=["| WELL | FINAL Macro-F1 | FINAL FAR |", "|---|---:|---:|"]
    for well_id,value in final.groupby("well_id").macro_f1.mean().sort_values().items():
        far=final[final.well_id==well_id].far.mean(); well_lines.append(f"| {well_id} | {value:.4f} | {far:.4f} |")
    fault_mean=fault.groupby(["method","fault"])[["recall","f1","mean_delay_samples"]].mean()
    fault_lines=["| Fault | FINAL recall | DCBR recall | SCALING recall | Δ DCBR-FINAL |","|---:|---:|---:|---:|---:|"]
    for number in range(1,21):
        values={m:fault_mean.loc[(m,number),"recall"] for m in ("FINAL_QDIFFCL","DCBR","SCALING")}
        fault_lines.append(f"| {number} | {values['FINAL_QDIFFCL']:.4f} | {values['DCBR']:.4f} | {values['SCALING']:.4f} | {values['DCBR']-values['FINAL_QDIFFCL']:+.4f} |")
    industrial=f"""# Industrial Group Analysis

本报告使用已经看过的 development test checkpoint 做只读分组重放，不参与任何新选择；不能称为 untouched paper-final evidence。

## 3W cross-WELL

{chr(10).join(well_lines)}

最困难 WELL 为 `{worst.index[0]}`（observed-class Macro-F1 `{worst.iloc[0]:.4f}`）。FINAL 相对 FreRA 共救回 `{int(rescue.loc['FRERA','rescued_windows'])}` 个窗口、丢失 `{int(rescue.loc['FRERA','lost_windows'])}`；相对 JITTER_SCALING 为 `{int(rescue.loc['JITTER_SCALING','rescued_windows'])}` / `{int(rescue.loc['JITTER_SCALING','lost_windows'])}`。WELL bootstrap CI 多数跨零，因此只支持“部分 hard WELL 有收益”，不支持 universal cross-WELL improvement。

Class 2/8/9 的逐 seed F1/Recall 已保存到 `3w_difficult_classes.csv`；FINAL 五 seed 平均 F1 分别为 `{class_frame[(class_frame.method=='FINAL_QDIFFCL')&(class_frame['class']==2)].f1.mean():.4f}`、`{class_frame[(class_frame.method=='FINAL_QDIFFCL')&(class_frame['class']==8)].f1.mean():.4f}`、`{class_frame[(class_frame.method=='FINAL_QDIFFCL')&(class_frame['class']==9)].f1.mean():.4f}`。Class 9 仍是明显困难类。

## TEP fault-wise

{chr(10).join(fault_lines)}

DCBR 对 fault 5 recall 改善最大，对 3/9/15/16/19 有退化；其主要证据仍是全局 Macro-F1 稳定改善和部分 delay 缩短，而不是所有 fault 一致提高。相对 SCALING 的 fault-group recall effect 为正，但 F1 CI 跨零。
"""
    (root/"industrial_group_analysis.md").write_text(industrial,encoding="utf-8")

    stability_rows=[]
    comparisons={"3W":[("FINAL_QDIFFCL","FRERA"),("FINAL_QDIFFCL","JITTER_SCALING")],"TEP":[("DCBR","FINAL_QDIFFCL"),("DCBR","SCALING")]}
    def global_metric(dataset,method,seed):
        if method=="DCBR":
            item=locked[dataset][str(seed)]; metric=item["metrics"]
            return {"macro_f1":metric["macro_f1"],"far":metric["far"],"early_recall":metric["early_recall"] if dataset=="3W" else item["early_fault"]["recall"]}
        record=external[f"{dataset}|{method}|{seed}"]["record"]
        if dataset=="3W": return {"macro_f1":record["metrics"]["macro_f1"],"far":record["metrics"]["far"],"early_recall":record["metrics"]["early_recall"]}
        return {"macro_f1":record["test"]["metrics"]["macro_f1"],"far":record["test"]["metrics"]["far"],"early_recall":record["test"]["early_fault"]["recall"]}
    for dataset,pairs in comparisons.items():
        seeds=config["seeds"]["three_w" if dataset=="3W" else "tep"]
        for method,reference in pairs:
            for seed in seeds:
                a=global_metric(dataset,method,seed); b=global_metric(dataset,reference,seed)
                stability_rows.append({"dataset":dataset,"method":method,"reference":reference,"seed":seed,
                    **{f"delta_{metric}":a[metric]-b[metric] for metric in a}})
    stability=pd.DataFrame(stability_rows); stability.to_csv(root/"stability_paired.csv",index=False,encoding="utf-8-sig")
    loso=[]
    for (dataset,method,reference),group in stability.groupby(["dataset","method","reference"]):
        worst_row=group.loc[group.delta_macro_f1.idxmin()]
        for excluded in group.seed:
            kept=group[group.seed!=excluded]; loso.append({"dataset":dataset,"method":method,"reference":reference,"excluded_seed":excluded,
                "loso_delta_macro_f1":kept.delta_macro_f1.mean(),"positive_count":int((group.delta_macro_f1>0).sum()),
                "nonworse_count":int((group.delta_macro_f1>=0).sum()),"worst_seed":int(worst_row.seed),"worst_delta":worst_row.delta_macro_f1})
    loso_frame=pd.DataFrame(loso); loso_frame.to_csv(root/"stability_loso.csv",index=False,encoding="utf-8-sig")
    stability_lines=[
        "## Five-seed paired stability",
        "",
        "以下比较使用相同 seed 配对的 development evidence；不能替代预注册 outer test。",
        "",
        "| Dataset | Method vs reference | Mean paired Δ Macro-F1 | Positive / non-worse | Worst seed (Δ) | LOSO range |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (dataset,method,reference),group in stability.groupby(["dataset","method","reference"],sort=False):
        summary=loso_frame[(loso_frame.dataset==dataset)&(loso_frame.method==method)&(loso_frame.reference==reference)]
        worst_seed_row=group.loc[group.delta_macro_f1.idxmin()]
        stability_lines.append(
            f"| {dataset} | {method} vs {reference} | {group.delta_macro_f1.mean():+.4f} | "
            f"{int((group.delta_macro_f1>0).sum())}/5 / {int((group.delta_macro_f1>=0).sum())}/5 | "
            f"{int(worst_seed_row.seed)} ({worst_seed_row.delta_macro_f1:+.4f}) | "
            f"[{summary.loso_delta_macro_f1.min():+.4f}, {summary.loso_delta_macro_f1.max():+.4f}] |"
        )
    stability_note=(
        "\n".join(stability_lines)
        + "\n\n3W 的两组 paired comparison 均只有 2/5 seeds 正向，且 LOSO 区间包含负值，"
          "因此不得宣称稳定优于 FreRA 或 Jitter+Scaling。TEP DCBR 相对 FINAL 为 5/5 正向；"
          "相对 SCALING 只有 2/5 正向，不支持稳定优越。\n"
    )
    with (root/"industrial_group_analysis.md").open("a",encoding="utf-8") as handle:
        handle.write("\n"+stability_note)
    (root/"stability.md").write_text("# Stability Audit\n\n"+stability_note,encoding="utf-8")

    eff_lines=["| Dataset | Method | Training s | Peak GPU MiB | Aug. params | Inference params |","|---|---|---:|---:|---:|---:|"]
    for _,r in efficiency.iterrows(): eff_lines.append(f"| {r.dataset} | {r.method} | {_fmt(r.training_seconds_mean,1)} | {_fmt(r.peak_gpu_mib_mean,1)} | {int(r.trainable_augmentation_parameters)} | {int(r.inference_additional_parameters)} |")
    (root/"efficiency.md").write_text("# Efficiency\n\n"+"\n".join(eff_lines)+"\n\nN/A 表示原实验复用了 checkpoint 且未保存可靠 wall-clock/peak-memory；不做事后估算。DCBR 推理新增参数为 0。\n",encoding="utf-8")

    matrix=[
      ("Selective > Uniform","3-seed validation matched-budget ablation","SUPPORTED ON 3W; NOT SUPPORTED ON TEP"),
      ("D/E captures fault semantics","D_ONLY/E_ONLY/FINAL + heatmaps","SUPPORTED with D-primary/E-complementary wording"),
      ("Soft allocation matters","Hard vs Soft 3-seed validation ablation","SUPPORTED ON 3W; NOT SUPPORTED ON TEP"),
      ("Gain is not from less noise","Soft matched vs unmatched + equal Uniform budget","SUPPORTED ON 3W; NOT SUPPORTED ON TEP"),
      ("DCBR mitigates over-augmentation","TEP FINAL/DCBR/SCALING 5-seed development evidence","SUPPORTED AS DEVELOPMENT EVIDENCE"),
      ("Cross-WELL benefit","per-WELL replay + bootstrap CI","PARTIAL; CI crosses zero"),
      ("Practicality","runtime/memory/parameters table","PARTIAL; some 3W runtime fields unavailable"),
      ("Limited-data robustness","grouped dry-run only","UNSUPPORTED / DO NOT CLAIM"),
      ("Missingness robustness","TEP MCAR30 only; 3W native missingness","PARTIAL"),
      ("Sensitivity 0.2/0.3/0.4","mask/budget audit only","UNSUPPORTED PERFORMANCE CLAIM"),
      ("Generalization","nested grouped paper-final protocol","PENDING OUTER EVALUATION"),
    ]
    lines=["# Paper Evidence Matrix","","| Claim | Evidence | Status |","|---|---|---|"]+[f"| {a} | {b} | `{c}` |" for a,b,c in matrix]
    lines += ["", "任何标为 UNSUPPORTED/PENDING 的 claim 不得进入摘要、贡献列表或结论。SVR 保持 `NO_GO_SVR`，不进入最终方法。"]
    Path("docs/paper_evidence_matrix.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    coverage="""# External Baseline Coverage Audit

主表已覆盖 NoAug、Jitter、Scaling、Jitter+Scaling、Uniform Diffusion、FreRA、FINAL_QDIFFCL/DCBR，满足传统、频域自动增强和内部扩散对照的最低覆盖。

- 值得补：一个 recent automated time-series augmentation baseline，但仅在能固定 shared TCN/Hard-SupCon/split/probe 时进入主表。
- 值得补：一个 diffusion-based industrial contrastive baseline；若官方实现绑定不同 encoder/objective，则只做 method-native supplementary，并明确不可直接公平排名。
- 不再为超过某个 baseline 数字扩大搜索或修改主方法。
"""
    (root/"external_baseline_coverage.md").write_text(coverage,encoding="utf-8")
    summary="""# Q-DiffCL Paper Evidence Chain Summary

## Frozen method

FINAL_QDIFFCL remains `0.5D + 0.5E`, critical ratio `0.30`, selective timesteps `1/5`, soft channel-frequency allocation, TCN, Hard SupCon, Original batching and frozen Linear Probe. DCBR remains a validation-calibrated domain-level scalar (`3W rho=1`, `TEP rho=.75`) with zero inference parameters. SVR remains `NO_GO_SVR` and is excluded from the final method.

## Evidence scope

- Core mechanism: new 3-seed validation-only Uniform/Hard/Soft/unmatched ablation; Soft matched is strongly supported on 3W but not on TEP.
- Semantic components: D is the primary discriminative contributor; E is complementary, not equally necessary.
- Industrial analysis: existing development checkpoints replayed per WELL/fault with group bootstrap; these are not untouched paper-final results.
- Robustness: grouped limited-data dry-run and ratio/timestep budget audit completed; missing downstream cells remain explicitly unsupported.
- Generalization: nested grouped Paper-final protocol passed dry-run; no outer model or outer metric has been run.

## Reporting boundary

The paper may claim dataset-dependent mechanism evidence and a TEP over-augmentation mitigation role for DCBR. It may not claim universal Soft superiority, universal cross-WELL gains, completed limited-data robustness, or paper-final generalization until the frozen outer evaluation is executed once.

See `docs/paper_evidence_matrix.md`, `docs/paper_final_protocol.md`, and the raw CSV/JSON under `docs/paper_evidence/`.
"""
    Path("docs/paper_evidence_chain_summary.md").write_text(summary,encoding="utf-8")
    result={"status":"PAPER_EVIDENCE_CHAIN_SUMMARIZED","worst_well":worst.index[0],"evidence_matrix":matrix,"svr":"NO_GO_SVR","paper_final_outer_run":False}
    Path("outputs/paper_evidence_chain/summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/paper_evidence_chain.yaml")
    args=parser.parse_args(); config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); print(json.dumps(summarize(config),ensure_ascii=False))


if __name__=="__main__": main()
