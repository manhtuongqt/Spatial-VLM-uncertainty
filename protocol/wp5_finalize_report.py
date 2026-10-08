#!/usr/bin/env python3
"""Build authoritative WP5 reports and SHA-256 artifact manifest."""
from __future__ import annotations
import csv,hashlib,json,math
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import wp5_spatial_uncertainty_estimator as est
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"results/spatial_vlm_refspatial_v1/wp5_spatial_uncertainty";DATA=ROOT/"datasets/Gazebo_train_uq_v2_full_r3"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(clean(v),indent=2,sort_keys=True,allow_nan=False)+"\n")
def clean(v):
 if isinstance(v,(float,np.floating)) and not math.isfinite(float(v)):return None
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,list):return [clean(x) for x in v]
 return v
def paired_correct(data,risk,threshold):
 d=est.decisions(data,risk,threshold);return float(np.mean([(x["action"]=="POINT" and x["hit"]) if x["state"]=="FOUND" else x["action"]=="ABSTAIN" for x in d]))
def metric(data,risk,threshold):
 m=est.metrics(data,risk,threshold);m["safe_task_accuracy"]=paired_correct(data,risk,threshold);return clean(m)
def main():
 selection=json.loads((OUT/"wp5_val_selection_decision.json").read_text());dev=json.loads((OUT/"wp5_dev_confirmation_report.json").read_text());model=json.loads((OUT/"selected_estimator.json").read_text());scored=est.rows(OUT/"wp5_scored_predictions.jsonl");gt={x["sample_id"]:x for x in est.rows(DATA/"evaluator_ground_truth.jsonl")};data=est.joined(DATA,OUT/"b0_full_r3/predictions.jsonl")
 for x in data:x["target_category"]=gt[x["sample_id"]].get("target_category")
 risk=np.asarray([next(y["estimated_unsafe_probability"] for y in scored if y["sample_id"]==x["sample_id"]) for x in data]);b0=est.baseline_risk(data);threshold=model["threshold"]
 stratified={}
 for field in ["state","relation","target_category"]:
  stratified[field]={}
  for value in sorted({str(x[field]) for x in data}):
   ix=[i for i,x in enumerate(data) if str(x[field])==value];part=[data[i] for i in ix];stratified[field][value]={"b0":metric(part,b0[ix],1.01),"method":metric(part,risk[ix],threshold)}
 dump(OUT/"wp5_metrics_by_state_relation_target.json",stratified)
 false_accept=[x for x in scored if x["answerability_state"]!="FOUND" and x["proposed_action"]=="POINT"]
 abstentions=[x for x in scored if x["proposed_action"]=="ABSTAIN"]
 for name,items in [("wp5_false_accept_examples.jsonl",false_accept),("wp5_abstention_cases.jsonl",abstentions)]:
  with (OUT/name).open("w") as f:
   for x in items:f.write(json.dumps(x,sort_keys=True)+"\n")
 boot=json.loads((OUT/"wp5_val_family_bootstrap_10000.json").read_text())
 with (OUT/"wp5_val_bootstrap_table.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=["metric","method_minus_b0","ci95_low","ci95_high","draws","seed"]);w.writeheader()
  for name,v in boot["effects"].items():w.writerow({"metric":name,"method_minus_b0":v["delta"],"ci95_low":v["ci95"][0],"ci95_high":v["ci95"][1],"draws":boot["draws"],"seed":boot["seed"]})
 pred_manifest={"schema_version":1,"status":"COMPLETE","raw_b0_predictions":{"path":"b0_full_r3/predictions.jsonl","count":320,"sha256":sha(OUT/"b0_full_r3/predictions.jsonl")},"family_scored_predictions":{"path":"wp5_scored_predictions.jsonl","count":320,"sha256":sha(OUT/"wp5_scored_predictions.jsonl")},"dev_confirmation_predictions":{"path":"wp5_dev_confirmation_predictions.jsonl","count":64,"sha256":sha(OUT/"wp5_dev_confirmation_predictions.jsonl")},"prediction_lock_sha256":sha(OUT/"b0_prediction_lock.json")};dump(OUT/"wp5_prediction_manifest.json",pred_manifest)
 geometry={"pilot_r6":"PASS_32_OF_32","full_attempt_01":"REJECT_317_OF_320","full_r2_attempt_01":"REJECT_316_OF_320","full_r3_attempt_01":"PASS_320_OF_320","full_r3_exact_rgb_unique":320,"full_r3_perceptual_duplicates":0,"dataset_train_families":256,"dataset_val_families":64}
 conclusion="WP5_COMPLETE_METHOD_ELIGIBLE_READY_FOR_CALIBRATION";go={"schema_version":1,"wp5_status":conclusion,"decision":"GO_TO_CALIBRATION","basis":"all preregistered Val-UQ eligibility clauses passed","warning":"single Dev-v2 confirmation showed severe FOUND Hit@0.08 regression (1.0 to 0.4375); calibration must quantify shift and cannot change backbone/architecture","constraints":["freeze B0 and selected estimator","fit calibration only on sealed Calibration split","do not open Test-IID/OOD until calibrator and threshold freeze","no robot ACT/REOBSERVE/ABSTAIN deployment claim"]};dump(OUT/"CALIBRATION_GO_NO_GO.json",go)
 metrics={"schema_version":2,"status":conclusion,"geometry_and_data_gates":geometry,"selected":{"method":"StandardScaler + L2 logistic regression","C":model["C"],"threshold":model["threshold"],"feature_count":len(model["feature_names"]),"raw_b0_scalar_confidence":"N/A_no_valid_reproducible_scalar","self_consistency_role":"uncalibrated_baseline_only"},"train_uq":{"b0":selection["metrics"]["b0_train"],"method":selection["metrics"]["method_train"]},"val_uq":{"b0":selection["metrics"]["b0_val"],"method":selection["metrics"]["method_val"],"paired_safe_task":selection["corrected_paired_safe_task"],"paired_bootstrap_10000":boot,"mcnemar":json.loads((OUT/"wp5_val_mcnemar.json").read_text()),"eligibility":"PASS_ALL_CLAUSES"},"dev_v2_confirmation":{"b0":dev["b0"],"method":dev["method"],"paired_family_bootstrap_10000":dev["paired_family_bootstrap_10000"],"mcnemar":dev["mcnemar"],"role":"one_confirmation_only_no_refit_no_reselection"},"stratified":stratified,"calibration_decision":go};dump(OUT/"WP5_SPATIAL_UNCERTAINTY_METRICS.json",metrics)
 v0=metrics["val_uq"]["b0"];v1=metrics["val_uq"]["method"];d0=dev["b0"];d1=dev["method"]
 report=f'''# WP5 Spatial Uncertainty — final result

## Conclusion

**{conclusion}**

The single preregistered spatial-risk method passes every Val-UQ eligibility clause. It is eligible to enter a separate calibration phase; it is **not deployment-ready**. The one frozen Dev-v2 confirmation exposes a substantial FOUND-coverage shift and must remain visible as a limitation.

## Immutable evidence chain

- WP4 stays closed: frozen B0 (`RoboRefer-2B-SFT`) is the only grounding backbone. B1 remains a negative ablation; B2 remains closed.
- Pilot r6 passed 32/32. Pilot/characterization rows were never training data.
- Full attempts 01 and r2 were rejected all-or-nothing (317/320 and 316/320). No passing rows were filtered or combined.
- Full-r3 passed geometry 320/320: Train-UQ 256 and Val-UQ 64 family-disjoint; 320 unique RGB hashes and no locked-rule perceptual duplicates.
- B0 predictions were hash-locked before evaluator labels were joined. Raw actions: 271 POINT, 49 INVALID, 0 ABSTAIN. INVALID is never counted as ABSTAIN.

## Val-UQ selection evidence (N=64)

| Metric | Frozen B0 | Proposed risk estimator |
|---|---:|---:|
| AURC ↓ | {v0['aurc']:.4f} | {v1['aurc']:.4f} |
| AUROC-error ↑ | {v0['auroc_error']:.4f} | {v1['auroc_error']:.4f} |
| ECE-10 ↓ | {v0['ece_10bin']:.4f} | {v1['ece_10bin']:.4f} |
| Brier ↓ | {v0['brier']:.4f} | {v1['brier']:.4f} |
| Non-FOUND false accept ↓ | {v0['nonfound_false_accept']:.4f} | {v1['nonfound_false_accept']:.4f} |
| Non-FOUND abstain recall ↑ | {v0['nonfound_abstain_recall']:.4f} | {v1['nonfound_abstain_recall']:.4f} |
| FOUND Hit@0.08 ↑ | {v0['found_hit_at_008']:.4f} | {v1['found_hit_at_008']:.4f} |
| POINT coverage | {v0['coverage']:.4f} | {v1['coverage']:.4f} |
| Exact output contract ↑ | {v0['exact_contract']:.4f} | {v1['exact_contract']:.4f} |
| Safe-task accuracy ↑ | {v0['safe_task_accuracy']:.4f} | {v1['safe_task_accuracy']:.4f} |

Selected only on Val-UQ: `C={model['C']}`, risk threshold `{model['threshold']}`. FOUND delta is exactly −0.125, the preregistered non-inferiority boundary. Paired family bootstrap (10,000 draws, seed 13092026) gives AURC delta {boot['effects']['aurc']['delta']:.4f}, 95% CI [{boot['effects']['aurc']['ci95'][0]:.4f}, {boot['effects']['aurc']['ci95'][1]:.4f}]. Corrected safe-task delta is {selection['corrected_paired_safe_task']['delta']:.4f}, 95% CI [{selection['corrected_paired_safe_task']['family_bootstrap_ci95'][0]:.4f}, {selection['corrected_paired_safe_task']['family_bootstrap_ci95'][1]:.4f}]; exact McNemar p={selection['corrected_paired_safe_task']['mcnemar']['exact_two_sided_p']:.3g}.

## One frozen Dev-v2 confirmation (N=64)

No refit, threshold change, or reselection occurred. AURC changed {d0['aurc']:.4f}→{d1['aurc']:.4f}; false accept {d0['nonfound_false_accept']:.4f}→{d1['nonfound_false_accept']:.4f}; abstain recall {d0['nonfound_abstain_recall']:.4f}→{d1['nonfound_abstain_recall']:.4f}. However FOUND Hit@0.08 fell {d0['found_hit_at_008']:.4f}→{d1['found_hit_at_008']:.4f}. This is evidence that the Val-selected threshold does not transfer cleanly; Dev-v2 is observed development evidence, not a final test.

## Scope and next gate

`r_spatial` is a separate uncertainty estimator; it does not modify B0. Self-consistency is an uncalibrated proxy, not a calibrated probability; raw B0 scalar confidence is N/A. No final calibrator was fit, Calibration and Test-IID/OOD remain sealed, SAM2 was not used, and WP5 contains no ACT/REOBSERVE/ABSTAIN robot deployment.

Decision: **GO_TO_CALIBRATION**, with the explicit Dev FOUND-regression warning. Calibration may fit calibration parameters/threshold under the frozen method, but cannot repair this by changing B0, features, architecture, or returning to Val selection. Test remains sealed until that new lock is complete.
''';(OUT/"WP5_SPATIAL_UNCERTAINTY_RESULT.md").write_text(report)
 manifest={"schema_version":1,"status":conclusion,"generated_at_utc":datetime.now(timezone.utc).isoformat(),"artifacts":{}}
 for p in sorted(x for x in OUT.rglob("*") if x.is_file() and x.name!="wp5_artifact_manifest.json"):manifest["artifacts"][str(p.relative_to(OUT))]={"bytes":p.stat().st_size,"sha256":sha(p)}
 dump(OUT/"wp5_artifact_manifest.json",manifest);print(json.dumps({"status":conclusion,"artifact_count":len(manifest["artifacts"]),"manifest_sha256":sha(OUT/"wp5_artifact_manifest.json")},indent=2))
if __name__=="__main__":main()
