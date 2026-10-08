#!/usr/bin/env python3
"""Freeze WP5 method/inference artifacts before the first full-data prediction."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"results/spatial_vlm_refspatial_v1/wp5_spatial_uncertainty";LOCK=OUT/"wp5_method_lock.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if LOCK.exists():raise FileExistsError("refusing to overwrite WP5 method lock")
 artifacts=["results/spatial_vlm_refspatial_v1/wp4_grounding_lock/grounding_model_lock.json","results/spatial_vlm_refspatial_v1/wp4_grounding_lock/hypothesis_lock.json","protocol/gazebo_train_uq_v2_full_r3_contract_lock.json","protocol/gazebo_train_uq_v2_full_r3_capture_compatibility_lock.json","protocol/gazebo_train_uq_v2_full_r3_v2_materialization_lock.json","datasets/Gazebo_train_uq_v2_full_r3/manifest.json","datasets/Gazebo_train_uq_v2_full_r3/inference_manifest.jsonl","protocol/gazebo_train_uq_v1_infer.py","protocol/gazebo_train_uq_v2_full_r3_infer.py","protocol/wp5_spatial_uncertainty_estimator.py"]
 value={"schema_version":1,"status":"LOCKED_BEFORE_FULL_B0_INFERENCE_AND_ESTIMATOR_FIT","locked_at_utc":datetime.now(timezone.utc).isoformat(),"source_artifact_sha256":{x:sha(ROOT/x) for x in artifacts},"backbone":"B0_frozen_RoboRefer_2B_SFT","base_model_directory_sha256":"042b822932ef41c36e775ab6dba0a0d1864f2c1fa0233bf29c7631584a85ffe7","inference":{"greedy":True,"max_new_tokens":40,"stochastic_draws":3,"temperature":.7,"top_p":.9,"top_k":50,"seed_rule":"sha256(protocol_id:sample_id:model_id:draw)[0:4]"},"unsafe_definition":"non-FOUND OR (FOUND and Euclidean normalized point error > 0.08)","method":{"count":1,"type":"StandardScaler + L2 logistic regression","C_grid":[.01,.1,1.,10.],"features_fit":"train_uq_only","C_and_threshold_selection":"val_uq_only","threshold_grid":"0.00..1.00 step 0.01"},"metrics":["false_accept","abstain_recall","ECE-10","Brier","AUROC-error","AURC","coverage","FOUND-Hit@0.08","exact-contract"],"bootstrap":{"unit":"parent_family","draws":10000,"seed":13092026,"paired":True},"sealed":{"gazebo_dev_answerability_v2":"one confirmation only after Val freeze","gazebo_calibration":True,"gazebo_test_iid":True,"gazebo_test_ood":True}}
 OUT.mkdir(parents=True,exist_ok=True);LOCK.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");print(json.dumps({"status":"LOCKED","sha256":sha(LOCK)},indent=2))
if __name__=="__main__":main()
