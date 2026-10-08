#!/usr/bin/env python3
"""Append-only r6 lock/preflight/QC adapter over the frozen r5 evaluator."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path

import gazebo_train_uq_v2_pilot_r5_pipeline as base
import generate_gazebo_train_uq_v2_pilot_r6_contract as generator

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "gazebo_train_uq_v2_pilot_r6"
CONFIG = ROOT / "ur3/ur3_perception/config"
SCENES = CONFIG / f"{PROTOCOL_ID}_scenes.yaml"
ANNOTATIONS = CONFIG / f"{PROTOCOL_ID}_annotations.yaml"
GATE = CONFIG / f"{PROTOCOL_ID}_gate.yaml"
AMENDMENT = ROOT / "protocol/gazebo_train_uq_v2_pilot_r6_amendment.json"
LOCK = ROOT / "protocol/gazebo_train_uq_v2_pilot_r6_contract_lock.json"
CAPTURE_LOCK = ROOT / "protocol/gazebo_train_uq_v2_pilot_r6_capture_compatibility_lock.json"
RESULT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_pilot_r6"
CAPTURE = RESULT / "capture_attempt_01"
STATIC_PREFLIGHT = RESULT / "PREFLIGHT_STATIC.json"
LIVE_PREFLIGHT = RESULT / "PREFLIGHT_LIVE.json"
QC_REPORT = RESULT / "GAZEBO_TRAIN_UQ_V2_PILOT_R6_GEOMETRY_QC.json"
DECISION = RESULT / "CAPTURE_ATTEMPT_01_DECISION.md"
R5_LOCK = ROOT / "protocol/gazebo_train_uq_v2_pilot_r5_contract_lock.json"
R5_CAPTURE_LOCK = ROOT / "protocol/gazebo_train_uq_v2_pilot_r5_capture_compatibility_lock.json"
R5_MANIFEST = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_pilot_r5/capture_attempt_01/capture_manifest.json"
R5_DECISION = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_train_uq_v2_pilot_r5/CAPTURE_ATTEMPT_01_DECISION.md"
GROUNDING_LOCK = base.GROUNDING_LOCK
HYPOTHESIS_LOCK = base.HYPOTHESIS_LOCK
PRIOR_SCENES = base.PRIOR_SCENE_CONFIGS + (CONFIG / "gazebo_train_uq_v2_pilot_r5_scenes.yaml",)
PRIOR_ANNOTATIONS = base.PRIOR_ANNOTATION_CONFIGS + (CONFIG / "gazebo_train_uq_v2_pilot_r5_annotations.yaml",)


def configure_base():
    for name, value in {
        "PROTOCOL_ID": PROTOCOL_ID, "SCENES": SCENES, "ANNOTATIONS": ANNOTATIONS, "GATE": GATE,
        "AMENDMENT": AMENDMENT, "LOCK": LOCK, "CAPTURE_LOCK": CAPTURE_LOCK, "RESULT": RESULT,
        "CAPTURE": CAPTURE, "STATIC_PREFLIGHT": STATIC_PREFLIGHT, "LIVE_PREFLIGHT": LIVE_PREFLIGHT,
        "QC_REPORT": QC_REPORT, "DECISION": DECISION, "generator": generator,
        "PRIOR_SCENE_CONFIGS": PRIOR_SCENES, "PRIOR_ANNOTATION_CONFIGS": PRIOR_ANNOTATIONS,
    }.items():
        setattr(base, name, value)


def validate_contract():
    generated = generator.build()
    for path, payload in zip((SCENES, ANNOTATIONS, GATE), generated):
        if not path.is_file() or path.read_text() != generator.serialized(payload):
            raise ValueError(f"non-deterministic r6 input: {path}")
    scenes, annotations, gate = map(base.load_yaml, (SCENES, ANNOTATIONS, GATE))
    rows, oracle = scenes["scenes"], annotations["scenes"]
    if len(rows) != 32 or len(oracle) != 32 or set(r["scene_id"] for r in rows) != set(oracle):
        raise ValueError("r6 requires 32 aligned rows")
    vectors = ([r["scene_id"] for r in rows], [r["scene_family_id"] for r in rows],
               [r["layout_id"] for r in rows], [r["layout_signature_sha256"] for r in rows])
    if any(len(set(values)) != 32 for values in vectors): raise ValueError("r6 identifiers/signatures not unique")
    for row in rows:
        sid = row["scene_id"]; signature = generator.parent.layout_signature(row["poses"])
        if signature != row["layout_signature_sha256"] or signature != oracle[sid]["layout_signature_sha256"]:
            raise ValueError(f"signature mismatch: {sid}")
    states = Counter(item["state"] for item in oracle.values()); relations = Counter(item["relation_variant"] for item in oracle.values())
    cells = Counter((item["state"], item["relation_variant"]) for item in oracle.values())
    if states != Counter({s:8 for s in base.STATES}) or relations != Counter({r:8 for r in base.RELATIONS}) or set(cells.values()) != {2}:
        raise ValueError("r6 quota mismatch")
    old_families, old_scenes, old_signatures = set(), set(), set()
    for path in base.DEV_MANIFESTS:
        for item in base.load_jsonl(path): old_families.add(str(item["family_id"])); old_scenes.add(str(item["scene_id"]))
    for path in PRIOR_ANNOTATIONS:
        old_families |= {str(item["family_id"]) for item in base.load_yaml(path).get("scenes",{}).values() if item.get("family_id")}
    for path in PRIOR_SCENES:
        for item in base.load_yaml(path).get("scenes",[]):
            old_scenes.add(str(item["scene_id"])); old_signatures.add(generator.parent.layout_signature(item["poses"]))
    if set(vectors[0]) & old_scenes or set(vectors[1]) & old_families or set(vectors[3]) & old_signatures:
        raise ValueError("r6 overlaps Dev or a prior attempt")
    amendment = json.loads(AMENDMENT.read_text())
    if amendment.get("status") != "LOCKED_BEFORE_R6_DESIGN_AND_CAPTURE": raise ValueError("r6 amendment invalid")
    evidence = amendment["parent_attempt"]
    for path,key in ((R5_LOCK,"contract_lock_sha256"),(R5_CAPTURE_LOCK,"capture_lock_sha256"),
                     (R5_MANIFEST,"capture_manifest_sha256"),(R5_DECISION,"decision_sha256")):
        if base.sha256(path) != evidence[key]: raise ValueError(f"r5 evidence changed: {path}")
    if "INVALID_PREFLIGHT_ORDER" not in R5_DECISION.read_text(): raise ValueError("r5 invalid decision missing")
    if base.sha256(GROUNDING_LOCK) != "6e1da7e9759eacfaadf7be07467f4996e95575fd2a6225af25d87e4f74418b09": raise ValueError("B0 lock changed")
    if base.sha256(HYPOTHESIS_LOCK) != "077f65f3db1b9562bc589ba54ac47d19c5a6ee1c1a00d3a178228b5822e8c78a": raise ValueError("hypothesis lock changed")
    return scenes, annotations, gate


def lock_contract():
    if LOCK.exists() or CAPTURE_LOCK.exists(): raise FileExistsError("refusing overwrite r6 locks")
    scenes, _, gate = validate_contract()
    own = [Path("protocol/generate_gazebo_train_uq_v2_pilot_r6_contract.py"),
           Path("protocol/gazebo_train_uq_v2_pilot_r6_pipeline.py"), AMENDMENT.relative_to(ROOT),
           SCENES.relative_to(ROOT), ANNOTATIONS.relative_to(ROOT), GATE.relative_to(ROOT),
           Path("protocol/gazebo_train_uq_v2_pilot_r5_pipeline.py"),
           R5_LOCK.relative_to(ROOT), R5_CAPTURE_LOCK.relative_to(ROOT), R5_MANIFEST.relative_to(ROOT), R5_DECISION.relative_to(ROOT),
           GROUNDING_LOCK.relative_to(ROOT), HYPOTHESIS_LOCK.relative_to(ROOT),
           base.V8_REPORT.relative_to(ROOT), base.V9_REPORT.relative_to(ROOT),
           Path("ur3/ur3_perception/launch/roborefer_uq_capture.launch.py"),
           Path("ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf")]
    payload = {"schema_version":1,"protocol_id":PROTOCOL_ID,"status":"LOCKED_BEFORE_PILOT_CAPTURE",
               "locked_at_utc":datetime.now(timezone.utc).isoformat(),
               "purpose":"geometry validation only; live preflight must precede capture",
               "parent_families":32,"state_relation_cell_quota":2,
               "layout_signature_sha256":{r["scene_id"]:r["layout_signature_sha256"] for r in scenes["scenes"]},
               "source_artifact_sha256":{str(path):base.sha256(ROOT/path) for path in own},
               "grounding_model_lock_sha256":base.sha256(GROUNDING_LOCK),"hypothesis_lock_sha256":base.sha256(HYPOTHESIS_LOCK),
               "r6_amendment_sha256":base.sha256(AMENDMENT),
               "geometry_gate":{k:gate[k] for k in ("tie_margin_normalized","min_visible_evidence_px","insufficient_evidence_rule","layout_uniqueness")},
               "world_file":"ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf","policies":gate["policies"]}
    base.write_json(LOCK,payload)
    capture_sources=set(base.prior.LEGACY_CAPTURE_SOURCES)|{str(path) for path in own}
    compatibility={"schema_version":1,"protocol_id":PROTOCOL_ID,"status":"LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
                   "locked_at_utc":datetime.now(timezone.utc).isoformat(),"parent_contract_lock":str(LOCK.relative_to(ROOT)),
                   "parent_contract_lock_sha256":base.sha256(LOCK),"purpose":"generic ROS capture adapter; r6 geometry only",
                   "source_artifact_sha256":{name:base.sha256(ROOT/name) for name in sorted(capture_sources)},
                   "model_inventory_sha256":"NO_MODEL_INFERENCE_GEOMETRY_PILOT_R6","policies_inherited":gate["policies"]}
    base.write_json(CAPTURE_LOCK,compatibility)
    print(json.dumps({"status":"LOCKED","contract_sha256":base.sha256(LOCK),"capture_lock_sha256":base.sha256(CAPTURE_LOCK)},indent=2))


def geometry_qc():
    # Reuse the frozen r5 evaluator logic with only revision-scoped literals
    # replaced.  This adapter itself and the inherited evaluator are hashed.
    source = inspect.getsource(base.geometry_qc)
    source = source.replace("NO_MODEL_INFERENCE_GEOMETRY_PILOT_R5", "NO_MODEL_INFERENCE_GEOMETRY_PILOT_R6")
    source = source.replace("pilot_r5", "pilot_r6")
    scope = dict(base.__dict__); scope.update(globals()); exec(source, scope)
    scope["geometry_qc"]()


def main():
    configure_base(); base.validate_contract = validate_contract
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=("validate-contract","lock","preflight-static","preflight-live","geometry-qc")); command=parser.parse_args().command
    if command=="validate-contract": validate_contract(); print("PASS: Gazebo Train-UQ v2 pilot r6 contract")
    elif command=="lock": lock_contract()
    elif command=="preflight-static": base.preflight_static()
    elif command=="preflight-live": base.preflight_live()
    else: geometry_qc()


if __name__=="__main__": main()
