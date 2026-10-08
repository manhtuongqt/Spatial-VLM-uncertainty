#!/usr/bin/env python3
"""Lock all blinded B0 bytes before estimator supervision/oracle is joined."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/"results/spatial_vlm_refspatial_v1/wp5_spatial_uncertainty";RUN=BASE/"b0_full_r3/run.json";PRED=BASE/"b0_full_r3/predictions.jsonl";LOCK=BASE/"b0_prediction_lock.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if LOCK.exists():raise FileExistsError("refusing overwrite prediction lock")
 run=json.loads(RUN.read_text());lines=[json.loads(x) for x in PRED.read_text().splitlines() if x]
 if run.get("status")!="COMPLETED" or run.get("completed_count")!=320 or len(lines)!=320 or len({x["sample_id"] for x in lines})!=320:raise ValueError("B0 inference not complete")
 value={"schema_version":1,"status":"LOCKED_BEFORE_ORACLE_JOIN_AND_ESTIMATOR_FIT","locked_at_utc":datetime.now(timezone.utc).isoformat(),"prediction_count":320,"predictions_sha256":sha(PRED),"run_sha256":sha(RUN),"method_lock_sha256":sha(BASE/"wp5_method_lock.json"),"dataset_inference_manifest_sha256":sha(ROOT/"datasets/Gazebo_train_uq_v2_full_r3/inference_manifest.jsonl"),"oracle_opened_by_runner":run.get("oracle_opened_by_runner"),"model_id":run.get("model_id"),"base_model_sha256":run.get("base_model_sha256"),"decoding":{"greedy":run.get("greedy_decoding"),"stochastic_draws":run.get("stochastic_draws"),"max_new_tokens":run.get("max_new_tokens")}}
 LOCK.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");print(json.dumps({"status":"LOCKED","sha256":sha(LOCK)},indent=2))
if __name__=="__main__":main()
