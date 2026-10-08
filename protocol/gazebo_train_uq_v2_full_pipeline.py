#!/usr/bin/env python3
"""Validate, lock and preflight the 320-family Gazebo Train-UQ v2 dataset."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import yaml

import generate_gazebo_train_uq_v2_full_contract as generator
import gazebo_train_uq_v2_pilot_r4_pipeline as capture_common
import generate_gazebo_train_uq_v2_pilot_r4_contract as signature_common

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = generator.PROTOCOL_ID
SCENES, ANNOTATIONS, GATE = generator.OUTPUT_PATHS
FAMILY_MANIFEST = generator.FAMILY_MANIFEST_PATH
SPLIT_MANIFEST = generator.SPLIT_MANIFEST_PATH
LOCK = ROOT / "protocol/gazebo_train_uq_v2_full_contract_lock.json"
CAPTURE_LOCK = ROOT / "protocol/gazebo_train_uq_v2_full_capture_compatibility_lock.json"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_full"
STATIC_PREFLIGHT = RESULT / "PREFLIGHT_STATIC.json"
LIVE_PREFLIGHT = RESULT / "PREFLIGHT_LIVE.json"
GROUNDING_LOCK = ROOT / "results/spatial_vlm_refspatial_v1/wp4_grounding_lock/grounding_model_lock.json"
HYPOTHESIS_LOCK = ROOT / "results/spatial_vlm_refspatial_v1/wp4_grounding_lock/hypothesis_lock.json"
R6_LOCK = ROOT / "protocol/gazebo_train_uq_v2_pilot_r6_contract_lock.json"
R6_CAPTURE_LOCK = ROOT / "protocol/gazebo_train_uq_v2_pilot_r6_capture_compatibility_lock.json"
R6_QC = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_pilot_r6/GAZEBO_TRAIN_UQ_V2_PILOT_R6_GEOMETRY_QC.json"
R6_DECISION = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_pilot_r6/CAPTURE_ATTEMPT_01_DECISION.md"
PRIOR_CONFIGS = tuple(sorted((ROOT / "ur3/ur3_perception/config").glob("gazebo_train_uq*_scenes.yaml")))
DEV_MANIFESTS = (ROOT / "datasets/Gazebo_dev/evaluator_ground_truth.jsonl",
                 ROOT / "datasets/Gazebo_dev_answerability_v2/evaluator_ground_truth.jsonl")


def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load_jsonl(path: Path): return [json.loads(line) for line in path.read_text().splitlines() if line]
def dump(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")


def validate_contract():
    generator.main if False else None
    expected = generator.build(); scenes, annotations, gate = (yaml.safe_load(path.read_text()) for path in (SCENES,ANNOTATIONS,GATE))
    for path,payload in zip((SCENES,ANNOTATIONS,GATE),expected[:3]):
        if path.read_text()!=generator.serialized(payload): raise ValueError(f"deterministic drift: {path}")
    family_expected="".join(json.dumps(row,sort_keys=True)+"\n" for row in expected[3])
    if FAMILY_MANIFEST.read_text()!=family_expected or json.loads(SPLIT_MANIFEST.read_text())!=expected[4]: raise ValueError("family/split manifest drift")
    rows,oracle=scenes["scenes"],annotations["scenes"]
    if len(rows)!=320 or len(oracle)!=320 or len({r["scene_id"] for r in rows})!=320 or len({r["scene_family_id"] for r in rows})!=320 or len({r["layout_signature_sha256"] for r in rows})!=320: raise ValueError("320-family identity gate failed")
    counts=Counter((oracle[r["scene_id"]]["split"],oracle[r["scene_id"]]["state"],oracle[r["scene_id"]]["relation_variant"]) for r in rows)
    for split,repetitions in generator.SPLITS.items():
        if any(counts[(split,state,relation)]!=repetitions for state in generator.STATES for relation in generator.RELATIONS): raise ValueError(f"quota mismatch: {split}")
    train={r["scene_family_id"] for r in rows if oracle[r["scene_id"]]["split"]=="train_uq"}; val={r["scene_family_id"] for r in rows if oracle[r["scene_id"]]["split"]=="val_uq"}
    if len(train)!=256 or len(val)!=64 or train&val: raise ValueError("split family gate failed")
    old_families=set(); old_signatures=set()
    for path in DEV_MANIFESTS:
        old_families|={str(row["family_id"]) for row in load_jsonl(path)}
    for path in PRIOR_CONFIGS:
        if path==SCENES: continue
        for row in yaml.safe_load(path.read_text()).get("scenes",[]): old_signatures.add(signature_common.layout_signature(row["poses"]))
    if {r["scene_family_id"] for r in rows}&old_families or {r["layout_signature_sha256"] for r in rows}&old_signatures: raise ValueError("overlap with Dev/prior UQ")
    r6=json.loads(R6_QC.read_text())
    if r6.get("status")!="PASS" or r6.get("records")!=32 or r6.get("failed_scene_count")!=0: raise ValueError("r6 32/32 PASS missing")
    if sha(GROUNDING_LOCK)!="6e1da7e9759eacfaadf7be07467f4996e95575fd2a6225af25d87e4f74418b09" or sha(HYPOTHESIS_LOCK)!="077f65f3db1b9562bc589ba54ac47d19c5a6ee1c1a00d3a178228b5822e8c78a": raise ValueError("WP4/hypothesis lock changed")
    if gate["selection_rule"]["selection_split"]!="val_uq_only" or gate["statistical_protocol"]["bootstrap_draws"]!=10000: raise ValueError("selection/statistical lock mismatch")
    return scenes,annotations,gate


def lock_contract():
    if LOCK.exists() or CAPTURE_LOCK.exists(): raise FileExistsError("refusing overwrite full locks")
    scenes,_,gate=validate_contract()
    sources=[Path("protocol/generate_gazebo_train_uq_v2_full_contract.py"),Path("protocol/gazebo_train_uq_v2_full_pipeline.py"),
             SCENES.relative_to(ROOT),ANNOTATIONS.relative_to(ROOT),GATE.relative_to(ROOT),FAMILY_MANIFEST.relative_to(ROOT),SPLIT_MANIFEST.relative_to(ROOT),
             GROUNDING_LOCK.relative_to(ROOT),HYPOTHESIS_LOCK.relative_to(ROOT),R6_LOCK.relative_to(ROOT),R6_CAPTURE_LOCK.relative_to(ROOT),R6_QC.relative_to(ROOT),R6_DECISION.relative_to(ROOT),
             Path("ur3/ur3_perception/launch/roborefer_uq_capture.launch.py"),Path("ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf")]
    payload={"schema_version":1,"protocol_id":PROTOCOL_ID,"status":"LOCKED_BEFORE_CAPTURE_INFERENCE_AND_TRAINING","locked_at_utc":datetime.now(timezone.utc).isoformat(),
             "parent_family_count":320,"split_parent_family_count":{"train_uq":256,"val_uq":64},"state_relation_cell_quota":{"train_uq":16,"val_uq":4},
             "family_manifest_sha256":sha(FAMILY_MANIFEST),"split_manifest_sha256":sha(SPLIT_MANIFEST),
             "scene_layout_signature_set_sha256":hashlib.sha256(json.dumps(sorted(r["layout_signature_sha256"] for r in scenes["scenes"])).encode()).hexdigest(),
             "source_artifact_sha256":{str(p):sha(ROOT/p) for p in sources},"b0_grounding_lock_sha256":sha(GROUNDING_LOCK),"hypothesis_lock_sha256":sha(HYPOTHESIS_LOCK),
             "pilot_r6_geometry_qc_sha256":sha(R6_QC),"world_file":"ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf",
             "camera":gate["camera"],"visibility_gates":{"tie_margin_normalized":gate["tie_margin_normalized"],"minimum_sufficient_pixels":gate["min_visible_evidence_px"],"insufficient":gate["insufficient_evidence_rule"]},
             "oracle_isolation":{"model_input_allowlist":gate["model_input_allowlist"],"evaluator_only":gate["evaluator_only"]},
             "inference_lock":gate["inference_lock"],"risk_feature_schema":gate["risk_feature_schema"],"proposed_method":gate["proposed_method"],"selection_rule":gate["selection_rule"],"statistical_protocol":gate["statistical_protocol"],"sealed":gate["sealed"]}
    dump(LOCK,payload)
    cap_sources=set(capture_common.LEGACY_CAPTURE_SOURCES)|{str(p) for p in sources}
    compatibility={"schema_version":1,"protocol_id":PROTOCOL_ID,"status":"LOCKED_BEFORE_CAPTURE_AND_INFERENCE","locked_at_utc":datetime.now(timezone.utc).isoformat(),
                   "parent_contract_lock":str(LOCK.relative_to(ROOT)),"parent_contract_lock_sha256":sha(LOCK),
                   "purpose":"320-family no-manipulation capture adapter","source_artifact_sha256":{name:sha(ROOT/name) for name in sorted(cap_sources)},
                   "model_inventory_sha256":"NO_MODEL_INFERENCE_FULL_UQ_V2_CAPTURE","policies":{"no_model_inference_during_capture":True,"no_robot_manipulation":True,"semantic_labels_evaluator_only":True,"calibration_and_test_sealed":True}}
    dump(CAPTURE_LOCK,compatibility); print(json.dumps({"status":"LOCKED","contract_sha256":sha(LOCK),"capture_lock_sha256":sha(CAPTURE_LOCK)},indent=2))


def verify_locks():
    validate_contract(); lock=json.loads(LOCK.read_text()); cap=json.loads(CAPTURE_LOCK.read_text())
    if cap.get("parent_contract_lock_sha256")!=sha(LOCK): raise ValueError("lock linkage failed")
    for item in (lock,cap):
        for name,digest in item["source_artifact_sha256"].items():
            if sha(ROOT/name)!=digest: raise ValueError(f"locked source changed: {name}")


def static_preflight():
    verify_locks(); free=shutil.disk_usage(ROOT).free/2**30; passed=free>3
    report={"schema_version":1,"protocol_id":PROTOCOL_ID,"status":"PASS" if passed else "BLOCKED","checked_at_utc":datetime.now(timezone.utc).isoformat(),"contract_lock_sha256":sha(LOCK),"capture_lock_sha256":sha(CAPTURE_LOCK),"deterministic_320_family_contract":True,"family_disjoint_and_quota_checks":True,"pilot_r6_32_of_32_pass":True,"free_gib":free,"capture_authorized":passed}
    dump(STATIC_PREFLIGHT,report);print(json.dumps(report,indent=2));
    if not passed: raise SystemExit(2)


def output(command): return subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False).stdout
def live_preflight():
    verify_locks(); topics=output(["ros2","topic","list","-t"]);services=output(["ros2","service","list","-t"])
    req={"/wrist_camera/color/image_raw":"sensor_msgs/msg/Image","/wrist_camera/depth/image_raw":"sensor_msgs/msg/Image","/wrist_camera/evaluation_labels/labels_map":"sensor_msgs/msg/Image","/wrist_camera/color/camera_info":"sensor_msgs/msg/CameraInfo","/tf":"tf2_msgs/msg/TFMessage","/joint_states":"sensor_msgs/msg/JointState"}
    checks={name:f"{name} [{kind}]" in topics for name,kind in req.items()};service="/world/ur3_pick_place/set_pose [ros_gz_interfaces/srv/SetEntityPose]" in services;free=shutil.disk_usage(ROOT).free/2**30;passed=all(checks.values()) and service and free>3
    report={"schema_version":1,"protocol_id":PROTOCOL_ID,"status":"PASS" if passed else "BLOCKED","checked_at_utc":datetime.now(timezone.utc).isoformat(),"topic_checks":checks,"set_pose_service":service,"free_gib":free,"capture_authorized":passed,"contract_lock_sha256":sha(LOCK),"capture_lock_sha256":sha(CAPTURE_LOCK)}
    dump(LIVE_PREFLIGHT,report);print(json.dumps(report,indent=2));
    if not passed: raise SystemExit(2)


def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=("validate","lock","preflight-static","preflight-live"));cmd=p.parse_args().command
    if cmd=="validate":validate_contract();print("PASS: full v2 contract")
    elif cmd=="lock":lock_contract()
    elif cmd=="preflight-static":static_preflight()
    else:live_preflight()


if __name__=="__main__":main()
