from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from utils import write_json


def summarize(config:dict):
    stage=json.loads(Path(config["stage_a"]["output"]).read_text(encoding="utf-8"));w=stage["three_w"];t=stage["tep"]
    summary={"status":"DSFD_DUAL_DATASET_NO_GO","stopping_status":stage["status"],"actual_new_training_runs":0,
        "stage_b_executed":False,"stage_c_executed":False,
        "three_w":{"domain_score_distribution":w["audit"]["domain_score_distribution"],"fault_domain_spearman":w["audit"]["fault_domain_spearman"],
            "quadrants":w["audit"]["quadrants"],"well_id_r1":w["probes"]["identity"]["R1"],"well_id_dsfd":w["probes"]["identity"]["DSFD"],
            "fault_r1":w["probes"]["fault"]["R1"],"fault_dsfd":w["probes"]["fault"]["DSFD"],
            "well_id_relative_accuracy_drop":stage["shortcut_gate"]["well_id_relative_accuracy_drop"],"fault_macro_f1_delta":stage["shortcut_gate"]["fault_macro_f1_delta"]},
        "tep":{"domain_score_distribution":t["audit"]["domain_score_distribution"],"fault_domain_spearman":t["audit"]["fault_domain_spearman"],
            "identity_r1":t["probes"]["identity"]["R1"],"identity_dsfd":t["probes"]["identity"]["DSFD"],
            "fault_r1":t["probes"]["fault"]["R1"],"fault_dsfd":t["probes"]["fault"]["DSFD"],
            "mean_absolute_timestep_change_from_r1":t["audit"]["mean_absolute_timestep_change_from_r1"]},
        "second_innovation_frozen":True,"next_stage":"R1 formal D/E/S ablation, external baselines, paper-final protocol"}
    write_json(Path(config["docs"]["final_json"]),summary)
    path=Path(config["docs"]["paired_csv"]);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as h:csv.DictWriter(h,fieldnames=("dataset","seed","metric","r1","dsfd","delta")).writeheader()
    Path(config["docs"]["three_w_report"]).write_text("# DSFD 3W 报告\n\n"
        f"Shortcut Gate：`{stage['status']}`，未执行正式训练。\n\n"
        f"- WELL-ID Accuracy：R1 {w['probes']['identity']['R1']['accuracy']:.6f}，DSFD {w['probes']['identity']['DSFD']['accuracy']:.6f}\n"
        f"- 相对下降：{stage['shortcut_gate']['well_id_relative_accuracy_drop']:.2%}（要求至少 5%）\n"
        f"- Fault Macro-F1：R1 {w['probes']['fault']['R1']['macro_f1']:.6f}，DSFD {w['probes']['fault']['DSFD']['macro_f1']:.6f}\n",encoding="utf-8")
    Path(config["docs"]["tep_report"]).write_text("# DSFD TEP 报告\n\n正式训练未执行。辅助 probe 中 run-ID Accuracy "
        f"{t['probes']['identity']['R1']['accuracy']:.6f} → {t['probes']['identity']['DSFD']['accuracy']:.6f}；Fault Macro-F1 保持 {t['probes']['fault']['R1']['macro_f1']:.6f}。"
        f"原始 DomainScore 较弱，但 percentile 映射仍产生 mean |Δt|={t['audit']['mean_absolute_timestep_change_from_r1']:.6f}，未自然退化为 R1。\n",encoding="utf-8")
    Path(config["docs"]["summary"]).write_text("# DSFD 总结\n\n"
        f"最终结论：`{summary['status']}`；停止状态：`{summary['stopping_status']}`；新增训练 run=0。\n\n"
        f"3W 存在 channel-frequency 域特异性，Fault/Domain Spearman={w['audit']['fault_domain_spearman']:.6f}，"
        f"Fault-low + Domain-high bins={w['audit']['quadrants']['fault_low_domain_high']}。但 DSFD 未降低 WELL-ID predictability："
        f"Accuracy {w['probes']['identity']['R1']['accuracy']:.6f} → {w['probes']['identity']['DSFD']['accuracy']:.6f}；"
        f"Fault Macro-F1 基本保持，delta={stage['shortcut_gate']['fault_macro_f1_delta']:+.6f}。\n\n"
        "因为核心 shortcut suppression 假设未得到操作性验证，Stage B/C 未执行，无法宣称改善 3W cross-WELL robustness 或完成 TEP preservation。"
        "第二创新正式冻结；下一阶段直接进入 R1 正式消融、external baselines 与 paper-final protocol。\n",encoding="utf-8")
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/domain_shortcut_aware_frequency_diffusion.yaml");a=p.parse_args()
    r=summarize(yaml.safe_load(Path(a.config).read_text(encoding="utf-8")));print(json.dumps(r,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
