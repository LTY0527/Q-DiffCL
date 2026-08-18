from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from utils import write_json


def summarize(config: dict):
    phase_a=json.loads(Path(config["phase_a"]["output"]).read_text(encoding="utf-8"))
    stage0=json.loads(Path(config["stage_0"]["output"]).read_text(encoding="utf-8"))
    kill=json.loads(Path(config["stage_1"]["output"]).read_text(encoding="utf-8"))
    w_r1=json.loads(Path("outputs/3w_diffusion_1seed_seed42/result.json").read_text(encoding="utf-8"))["methods"]["FREQUENCY_SELECTIVE_R1"]["metrics"]
    w=kill["three_w"]["method"]["metrics"]
    tep_source=json.loads(Path(config["tep"]["r1_result"]).read_text(encoding="utf-8"))["seed_results"]["7"]["methods"]["R1"]["test"]
    t=kill["tep"]["method"]["test"]
    paired=[]
    for metric in ("macro_f1","auprc_fault_vs_normal","far","early_recall","mean_detection_delay_seconds"):
        paired.append({"dataset":"3W","seed":42,"metric":metric,"r1":w_r1[metric],"cdvs":w[metric],"delta":w[metric]-w_r1[metric]})
    tep_pairs={"macro_f1":(tep_source["metrics"]["macro_f1"],t["metrics"]["macro_f1"]),
        "auprc":(tep_source["metrics"]["auprc"],t["metrics"]["auprc"]),"far":(tep_source["metrics"]["far"],t["metrics"]["far"]),
        "fault_recall":(tep_source["metrics"]["fault_recall"],t["metrics"]["fault_recall"]),
        "early_recall":(tep_source["early_fault"]["recall"],t["early_fault"]["recall"]),
        "delay_samples":(tep_source["detection_delay"]["mean_delay_samples"],t["detection_delay"]["mean_delay_samples"])}
    for metric,(old,new) in tep_pairs.items(): paired.append({"dataset":"TEP","seed":7,"metric":metric,"r1":old,"cdvs":new,"delta":new-old})
    summary={"status":"CDVS_DUAL_DATASET_NO_GO","stopping_status":kill["status"],"phase_a":phase_a["status"],
        "stage_0":stage0["status"],"actual_new_training_runs":2,"stage_2_executed":False,
        "three_w":{"r1":{k:w_r1[k] for k in ("macro_f1","auprc_fault_vs_normal","far","early_recall","mean_detection_delay_seconds")},
                   "cdvs":{k:w[k] for k in ("macro_f1","auprc_fault_vs_normal","far","early_recall","mean_detection_delay_seconds")},
                   "macro_f1_delta":kill["comparisons"]["3W"]["macro_f1_delta"],"far_delta":kill["comparisons"]["3W"]["far_delta"],
                   "kill_reason":"FAR increase exceeds +0.05"},
        "tep":{"r1":{k:v[0] for k,v in tep_pairs.items()},"cdvs":{k:v[1] for k,v in tep_pairs.items()},
               "macro_f1_delta":kill["comparisons"]["TEP"]["macro_f1_delta"],"far_delta":kill["comparisons"]["TEP"]["far_delta"]},
        "second_innovation_frozen":True,"next_stage":"R1 formal ablation, external baselines, paper-final protocol"}
    path=Path(config["docs"]["paired_csv"]);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as h:
        writer=csv.DictWriter(h,fieldnames=("dataset","seed","metric","r1","cdvs","delta"));writer.writeheader();writer.writerows(paired)
    write_json(Path(config["docs"]["final_json"]),summary);_reports(config,summary,stage0);return summary


def _reports(config,summary,stage0):
    w=summary["three_w"];t=summary["tep"]
    Path(config["docs"]["three_w_report"]).write_text("# CDVS 3W Kill Test\n\n`CDVS_KILL_TEST_NO_GO`。\n\n"
        f"- Macro-F1：R1 {w['r1']['macro_f1']:.6f}，CDVS {w['cdvs']['macro_f1']:.6f}，delta {w['macro_f1_delta']:+.6f}\n"
        f"- FAR：R1 {w['r1']['far']:.6f}，CDVS {w['cdvs']['far']:.6f}，delta {w['far_delta']:+.6f}\n"
        f"- AUPRC / Early Recall / Delay：{w['cdvs']['auprc_fault_vs_normal']:.6f} / {w['cdvs']['early_recall']:.6f} / {w['cdvs']['mean_detection_delay_seconds']:.3f}\n"
        "FAR 增加超过 +0.05，触发停止线。\n",encoding="utf-8")
    Path(config["docs"]["tep_report"]).write_text("# CDVS TEP Kill Test\n\n"
        f"- Macro-F1：R1 {t['r1']['macro_f1']:.6f}，CDVS {t['cdvs']['macro_f1']:.6f}，delta {t['macro_f1_delta']:+.6f}\n"
        f"- FAR：R1 {t['r1']['far']:.6f}，CDVS {t['cdvs']['far']:.6f}，delta {t['far_delta']:+.6f}\n"
        f"- AUPRC / Early Recall / Delay：{t['cdvs']['auprc']:.6f} / {t['cdvs']['early_recall']:.6f} / {t['cdvs']['delay_samples']:.3f}\n",encoding="utf-8")
    Path(config["docs"]["summary"]).write_text("# CDVS 双数据集总结\n\n"
        f"最终结论：`{summary['status']}`；停止状态：`{summary['stopping_status']}`。\n\n"
        f"Phase A=`{summary['phase_a']}`，Stage 0=`{summary['stage_0']}`，实际新增训练 run={summary['actual_new_training_runs']}。"
        "pseudo-unseen safety 与 DRFD rank reliability 存在实质差异，且机制约束成立；但 3W seed 42 FAR 灾难性增加，Stage 2 未执行。\n\n"
        "正式停止第二创新算法搜索，不开发 CDVS-v2/v3，不根据 test 修改方法。冻结 R1，下一阶段进入正式 D/E/S ablation、external baselines 和 paper-final protocol。\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/cross_domain_validated_safety.yaml");a=p.parse_args()
    r=summarize(yaml.safe_load(Path(a.config).read_text(encoding="utf-8")));print(json.dumps(r,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
