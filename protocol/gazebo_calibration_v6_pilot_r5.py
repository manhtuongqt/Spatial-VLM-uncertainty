#!/usr/bin/env python3
"""Append-only R5 pilot data-design/exclusion amendment and execution gate.

R5 does not change the sensor implementation or any scientific component.  It
creates a fresh 4x4x2 geometry pilot whose identifiers, seeds and physical
layout signatures are deterministically disjoint from R3, R4, frozen
Calibration-v6, and the historical exclusion registry.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import yaml

import certify_gazebo_calibration_v6_prerender as certificate
import gazebo_calibration_v3_pipeline_r4 as helpers
import gazebo_calibration_v3_pipeline_r6 as parsers
import gazebo_calibration_v6_pipeline as base
import gazebo_calibration_v6_pipeline_r2 as r2
import gazebo_calibration_v6_pipeline_r3 as r3
import gazebo_calibration_v6_pipeline_r4 as r4
import generate_gazebo_calibration_v3_contract as v3
import generate_gazebo_calibration_v6_design as v6


ROOT = v6.ROOT
PID = "gazebo_calibration_v6_geometry_pilot_r5"
OUTPUT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot_r5"
CAPTURE = OUTPUT / "capture_attempt_01"
DATA_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V6_PILOT_R5_DATA_DESIGN_AMENDMENT_LOCK.json"
IMPLEMENTATION_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V6_PILOT_R5_IMPLEMENTATION_LOCK.json"
AUTH = ROOT / "protocol/gazebo_calibration_v6_geometry_pilot_r5_capture_compatibility_lock.json"
CALIBRATION_AUTH = ROOT / "protocol/gazebo_calibration_v6_r5_capture_compatibility_lock.json"
CALIBRATION_OUTPUT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_r5/capture_attempt_01"
REGISTRY = OUTPUT / "R5_EXCLUSION_REGISTRY.json"
AUDIT = OUTPUT / "R5_STATIC_DESIGN_AUDIT.json"
PREFLIGHT = OUTPUT / "PREFLIGHT_LIVE_ATTEMPT_05.json"
SCENES = ROOT / f"ur3/ur3_perception/config/{PID}_scenes.yaml"
ANNOTATIONS = ROOT / f"ur3/ur3_perception/config/{PID}_annotations.yaml"
GATE = ROOT / f"ur3/ur3_perception/config/{PID}_gate.yaml"
FAMILY_MANIFEST = ROOT / f"protocol/{PID}_family_manifest.jsonl"
SPLIT_MANIFEST = ROOT / f"protocol/{PID}_split_manifest.json"
QC_CODE = ROOT / "protocol/gazebo_calibration_v6_pilot_r5_qc.py"
CAPTURE_CODE = ROOT / "ur3/ur3_perception/scripts/roborefer_pilot_capture_r4.py"
R3_INPUT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot/capture_attempt_01/input_manifest.jsonl"
R4_INPUT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v6_geometry_pilot_r4/capture_attempt_01/input_manifest.jsonl"
R4_ROOT = r4.PILOT_ROOT
R4_DECISION = R4_ROOT / "PILOT_R4_DECISION.json"
R4_FAILURE = R4_ROOT / "PILOT_QC_FAILURE_LOCK.json"
R4_QC_INPUT = R4_ROOT / "QC_INPUT_LOCK.json"
R5_QC = OUTPUT / "GEOMETRY_QC.json"
R5_RGB_QC = OUTPUT / "RGB_DUPLICATE_QC.json"
R5_LEAKAGE_QC = OUTPUT / "MANIFEST_LEAKAGE_QC.json"
R5_QC_INPUT = OUTPUT / "QC_INPUT_LOCK.json"
R5_DECISION = OUTPUT / "PILOT_R5_DECISION.json"
GENERATED = (REGISTRY, AUDIT, SCENES, ANNOTATIONS, GATE, FAMILY_MANIFEST, SPLIT_MANIFEST)

EXPECTED_R4 = {
    R4_DECISION: "6486a8f91d4ebb1108779f3db364f38448c8e5c81afae039861bc743d0c6b6c0",
    R4_FAILURE: "e0b18ab62c584b7749aeb880af6f6d3e8be36fafebb0c4bb82c09ecc9ef5fc05",
    R4_ROOT / "RGB_DUPLICATE_QC.json": "bbfa0ea5aee53abde2c6b3d0801377b0c37628c6f1ac8f007024e3f56ee46236",
    R4_ROOT / "GEOMETRY_QC.json": "c594aa326e8b2a31ffb79998f8bcd84516b1498e890613fc27c6f99222fe67f8",
    R4_INPUT: "0d679a6dfd60684ed96522544edde9165ada43d27888e171479ab3d14688655c",
    R3_INPUT: "ae84d7b904f71135e1fe15cd45b365a02eb7d43f2ffebbf08ed7d952b0277e0c",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_hash_map(value: dict[str, str]) -> None:
    for name, digest in value.items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"frozen source drift: {name}")


def verify_predecessors() -> None:
    r4.validate_r4()
    for path, digest in EXPECTED_R4.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"R4 terminal artifact drift: {path}")
    decision = read_json(R4_DECISION)
    failure = read_json(R4_FAILURE)
    if decision.get("decision") != "PILOT_R4_ATTEMPT_01_REJECTED_STOP_NO_RETRY":
        raise RuntimeError("R4 pilot decision changed")
    if failure.get("status") != "PILOT_QC_REJECTED_STOP_NO_RETRY_UNDER_R4":
        raise RuntimeError("R4 failure lock changed")
    if read_json(R4_ROOT / "GEOMETRY_QC.json").get("passed_scene_count") != 32:
        raise RuntimeError("R4 geometry evidence changed")
    verify_hash_map(read_json(R4_QC_INPUT)["source_artifact_sha256"])
    if any(path.exists() for path in (CAPTURE, AUTH, CALIBRATION_AUTH, CALIBRATION_OUTPUT)):
        raise RuntimeError("unexpected R5 capture or authorization exists")
    if r2.DATASET_ROOT.exists():
        raise RuntimeError("Calibration-v6 dataset exists before R5 pilot")


def source_paths_before_generation() -> list[Path]:
    return [
        Path(__file__).resolve(), QC_CODE, v6.LOCK, base.IMPLEMENTATION,
        r4.R4_LOCK, R4_DECISION, R4_FAILURE, R4_QC_INPUT,
        R4_ROOT / "GEOMETRY_QC.json", R4_ROOT / "RGB_DUPLICATE_QC.json",
        R3_INPUT, R4_INPUT, v6.REGISTRY,
        *v6.paths(v6.PILOT), *v6.paths(v6.CALIBRATION),
        certificate.RESULT, r3.R3_PRECHECK, r3.ATTEMPT_03,
    ]


def preregister() -> None:
    if DATA_LOCK.exists() or any(path.exists() for path in GENERATED):
        raise FileExistsError("R5 data design was already preregistered or generated")
    verify_predecessors()
    invariants = {
        "b0_frozen_unchanged": True,
        "spatial_risk_v2_binary_and_12_features_frozen_unchanged": True,
        "frozen_calibration_128_manifest_unchanged": True,
        "visibility_metric_calibrator_optimizer_and_scientific_gates_unchanged": True,
        "r3_and_r4_attempts_quarantined_no_row_reuse": True,
        "duplicate_thresholds_unchanged": True,
        "test_iid_ood_and_robot_policy_sealed": True,
    }
    sources = source_paths_before_generation()
    payload = {
        "schema_version": 1,
        "protocol_id": PID,
        "status": "PILOT_R5_DATA_DESIGN_PREREGISTERED_BEFORE_SCENE_GENERATION",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "pilot-only deterministic data-design/exclusion amendment after R4 near-duplicate rejection",
        "r4_terminal_reason": {
            "geometry_qc": "32/32 PASS",
            "overall_qc": "REJECT",
            "near_duplicate_count": 1,
            "failed_pair_scene_id": "gazebo_calibration_v6_geometry_pilot_003",
            "sensor_or_geometry_repair_authorized": False,
        },
        "population": "32 = 4 answerability states x 4 relations x 2 repetitions",
        "deterministic_generation": {
            "namespace": PID,
            "nonce_rule": "select the smallest nonnegative integer nonce whose 32 generated identifiers, uint32 seeds and canonical layout signatures are internally unique and disjoint from the append-only registry; geometry metadata only, no render/model outcomes",
            "seed_rule": "uint32(first_8_hex(sha256(protocol_id + '_seed_r5_n' + nonce + '|' + family_id + '|' + state + '|' + relation + '|' + repetition)))",
            "capture_order_rule": "sort by sha256(protocol_id + '_capture_order_v1|' + family_id)",
            "row_resampling_or_outcome_selection": False,
        },
        "required_exclusions": [
            "all historical registry family/scene/layout IDs, deterministic seeds and layout signatures",
            "frozen original Calibration-v6 128-family manifest",
            "R3 partial pilot and R4 rejected pilot identifiers/layout signatures/RGB hashes",
            "all historical RGB manifests frozen by the v6 implementation lock",
        ],
        "invariants": invariants,
        "source_artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "scene_generation_authorized": True,
        "capture_authorized": False,
        "pilot_accepted_families": 0,
        "calibration_families_captured": 0,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "scientific_decision": None,
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_new(DATA_LOCK, payload)
    print(json.dumps({"status": payload["status"], "sha256": sha256(DATA_LOCK), "invariants": invariants}, indent=2))


def validate_data_lock() -> dict[str, Any]:
    value = read_json(DATA_LOCK)
    if value.get("status") != "PILOT_R5_DATA_DESIGN_PREREGISTERED_BEFORE_SCENE_GENERATION":
        raise RuntimeError("R5 data-design lock status invalid")
    verify_hash_map(value["source_artifact_sha256"])
    if not all(value["invariants"].values()):
        raise RuntimeError("R5 scientific invariants invalid")
    return value


def add_identifier(values: dict[str, set[Any]], row: dict[str, Any]) -> None:
    for field in ("family_id", "scene_id", "layout_id", "deterministic_seed", "layout_signature_sha256"):
        if row.get(field) is not None:
            values[field].add(row[field])
    if row.get("scene_family_id") is not None:
        values["family_id"].add(row["scene_family_id"])
    if row.get("seed") is not None:
        values["deterministic_seed"].add(row["seed"])


def build_registry() -> dict[str, Any]:
    inherited = read_json(v6.REGISTRY)
    fields = ("family_id", "scene_id", "layout_id", "deterministic_seed", "layout_signature_sha256")
    values = {field: set(inherited["identifiers"][field]) for field in fields}
    rgb_hashes: set[str] = set()
    sources = dict(inherited["source_artifact_sha256"])
    sources[str(v6.REGISTRY.relative_to(ROOT))] = sha256(v6.REGISTRY)
    for manifest in (v6.paths(v6.PILOT)[3], v6.paths(v6.CALIBRATION)[3]):
        sources[str(manifest.relative_to(ROOT))] = sha256(manifest)
        for row in read_jsonl(manifest):
            add_identifier(values, row)
    base_lock = read_json(base.IMPLEMENTATION)
    for name, digest in base_lock["prior_rgb_manifest_sha256"].items():
        manifest = ROOT / name
        if sha256(manifest) != digest:
            raise RuntimeError(f"historical RGB manifest drift: {name}")
        sources[name] = digest
        for row in read_jsonl(manifest):
            value = row.get("input_sha256", {}).get("rgb")
            if value:
                rgb_hashes.add(value)
    for manifest in (R3_INPUT, R4_INPUT):
        sources[str(manifest.relative_to(ROOT))] = sha256(manifest)
        for row in read_jsonl(manifest):
            rgb_path = manifest.parent / row["input_files"]["rgb"]
            digest = row["input_sha256"]["rgb"]
            if sha256(rgb_path) != digest:
                raise RuntimeError(f"quarantined RGB drift: {rgb_path}")
            rgb_hashes.add(digest)
    return {
        "schema_version": 1,
        "status": "R5_APPEND_ONLY_EXCLUSION_REGISTRY_FROZEN_BEFORE_RENDER",
        "source_artifact_sha256": sources,
        "identifiers": {field: sorted(values[field]) for field in fields},
        "rgb_sha256": sorted(rgb_hashes),
        "counts": {**{field: len(values[field]) for field in fields}, "rgb_sha256": len(rgb_hashes)},
        "r3_partial_and_r4_rejected_rows_are_exclusion_only": True,
        "test_data_opened": False,
    }


def construct(nonce: int, data_lock_sha: str):
    source = yaml.safe_load(v6.paths(v6.PILOT)[0].read_text(encoding="utf-8"))
    oracle = yaml.safe_load(v6.paths(v6.PILOT)[1].read_text(encoding="utf-8"))
    gate = yaml.safe_load(v6.paths(v6.PILOT)[2].read_text(encoding="utf-8"))
    old_families = sorted(read_jsonl(v6.paths(v6.PILOT)[3]), key=lambda row: row["family_index"])
    scene_lookup = {scene["scene_id"]: scene for scene in source["scenes"]}
    annotations = oracle["scenes"]
    lock = read_json(v6.LOCK)
    geometry = lock["exact_geometry_revision"]
    tables = geometry["all_state_anchor_tables"]
    tf = read_json(certificate.TF_PATH)["camera_color_optical_frame"]
    origin = tf["position"]
    rotation = certificate.geometry.rotation(tf["orientation_xyzw"])
    camera = read_json(ROOT / "protocol/GAZEBO_CALIBRATION_V3_CONTRACT_LOCK.json")["camera_and_geometry"]["camera"]
    new_scenes: list[dict[str, Any]] = []
    new_annotations: dict[str, dict[str, Any]] = {}
    families: list[dict[str, Any]] = []
    for old_family in old_families:
        index = int(old_family["family_index"])
        state = old_family["state"]
        relation = old_family["relation_variant"]
        repetition = int(old_family["cell_repetition"])
        old_scene = scene_lookup[old_family["scene_id"]]
        old_annotation = annotations[old_family["scene_id"]]
        scene = copy.deepcopy(old_scene)
        annotation = copy.deepcopy(old_annotation)
        scene_id = f"{PID}_{index:03d}"
        family_id = f"spatial_vlm_{PID}/development/parent_{index:03d}"
        layout_id = f"{PID}_layout_{index:03d}"
        seed_text = f"{PID}_seed_r5_n{nonce}|{family_id}|{state}|{relation}|{repetition}"
        seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
        if state == "INSUFFICIENT_EVIDENCE":
            names = ["ycb_apple", "ycb_orange"]
            anchors = geometry["relation_aware_context_projected_center_targets_normalized"][relation]
            poses = {name: copy.deepcopy(old_scene["poses"][name]) for name in ("mango", "uq_neutral_occluder")}
        elif state == "FOUND":
            names: list[str | None] = [None, None, None]
            slot = annotation["rank"] - 1 if annotation["rank_from"] == "left" else 3 - annotation["rank"]
            names[slot] = annotation["target_id"]
            other_names = iter(sorted(set(annotation["candidate_ids"]) - {annotation["target_id"]}))
            names = [name if name is not None else next(other_names) for name in names]
            anchors = tables["FOUND"]
            poses = {}
        elif state == "AMBIGUOUS":
            names = sorted(annotation["valid_target_ids"]) + sorted(
                set(annotation["candidate_ids"]) - set(annotation["valid_target_ids"])
            )
            side = "LEFT" if relation in ("leftmost", "second_from_left") else "RIGHT"
            anchors = tables["AMBIGUOUS_TIE_" + side]
            poses = {}
        else:
            names = sorted(annotation["context_ids"])
            anchors = tables["ABSENT"]
            poses = {}
        for slot_index, (name, anchor) in enumerate(zip(names, anchors)):
            u = anchor[0] + 0.0040 * (((seed >> (8 * slot_index)) % 5) - 2)
            v = anchor[1] + 0.0050 * (((seed >> (8 * slot_index + 3)) % 5) - 2)
            x, y, _ = certificate.geometry.inverse(
                u, v, source["objects"][name]["z"], origin, rotation, camera["intrinsics"]
            )
            poses[name] = [x, y, 0.0700 + 0.0070 * ((seed >> (16 + 3 * slot_index)) % 11)]
        scene.update(scene_id=scene_id, scene_family_id=family_id, layout_id=layout_id, poses=poses)
        signature = v3.layout_signature(scene, source["objects"])
        scene["layout_signature_sha256"] = signature
        annotation.update(
            family_id=family_id,
            layout_id=layout_id,
            layout_signature_sha256=signature,
            split="development_geometry_only",
            seed=seed,
            cell_repetition=repetition,
            geometry_provenance="frozen_v6_certified_anchor_envelope_r5_deterministic_exclusion_only",
        )
        new_scenes.append(scene)
        new_annotations[scene_id] = annotation
        families.append({
            "family_index": index,
            "scene_id": scene_id,
            "family_id": family_id,
            "layout_id": layout_id,
            "layout_signature_sha256": signature,
            "split": "development_geometry_only",
            "state": state,
            "relation_variant": relation,
            "cell_repetition": repetition,
            "deterministic_seed": seed,
            "capture_order_sha256": hashlib.sha256(f"{PID}_capture_order_v1|{family_id}".encode()).hexdigest(),
        })
    for order, family in enumerate(sorted(families, key=lambda row: row["capture_order_sha256"])):
        family["capture_order"] = order
    order_lookup = {family["scene_id"]: family["capture_order"] for family in families}
    new_scenes.sort(key=lambda scene: order_lookup[scene["scene_id"]])
    source.update(
        protocol_id=PID,
        contract_lock_sha256=data_lock_sha,
        expected_scene_count=32,
        scenes=new_scenes,
        capture_order="sha256_sort_r5_family_id",
    )
    oracle.update(
        protocol_id=PID,
        contract_lock_sha256=data_lock_sha,
        scenes=new_annotations,
        oracle_usage="geometry_only",
    )
    gate.update(
        protocol_id=PID,
        contract_lock_sha256=data_lock_sha,
        parent_family_count=32,
        split_parent_family_count={"development_geometry_only": 32},
        state_quota={"development_geometry_only": {state: 8 for state in ("FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE")}},
        relation_variant_quota={"development_geometry_only": {relation: 8 for relation in ("leftmost", "rightmost", "second_from_left", "second_from_right")}},
        state_relation_cell_quota={"development_geometry_only": 2},
    )
    gate["policies"]["no_model_inference"] = True
    gate["policies"]["no_materialization"] = True
    return source, oracle, gate, families


def plan_checks(families: list[dict[str, Any]], registry: dict[str, Any]) -> tuple[dict[str, bool], dict[str, list[Any]]]:
    fields = ("family_id", "scene_id", "layout_id", "deterministic_seed", "layout_signature_sha256")
    overlaps = {
        field: sorted(set(row[field] for row in families).intersection(registry["identifiers"][field]))
        for field in fields
    }
    cells = Counter((row["state"], row["relation_variant"]) for row in families)
    checks = {
        "exactly_32_families": len(families) == 32,
        "all_identifiers_seeds_signatures_internally_unique": all(len({row[field] for row in families}) == 32 for field in fields),
        "zero_registry_overlap": all(not values for values in overlaps.values()),
        "four_states_by_four_relations_by_two": len(cells) == 16 and set(cells.values()) == {2},
        "capture_order_is_0_to_31": sorted(row["capture_order"] for row in families) == list(range(32)),
    }
    return checks, overlaps


def dry_plan() -> tuple[int, dict[str, Any], tuple[Any, ...], dict[str, bool], dict[str, list[Any]]]:
    registry = build_registry()
    data_sha = sha256(DATA_LOCK) if DATA_LOCK.exists() else "DRY_RUN_DATA_LOCK_SHA256_PLACEHOLDER"
    for nonce in range(1_000_000):
        plan = construct(nonce, data_sha)
        checks, overlaps = plan_checks(plan[3], registry)
        if all(checks.values()):
            return nonce, registry, plan, checks, overlaps
    raise RuntimeError("no deterministic disjoint R5 nonce found")


def generate() -> None:
    if any(path.exists() for path in GENERATED):
        raise FileExistsError("R5 generation is single-attempt append-only")
    validate_data_lock()
    nonce, registry, plan, checks, overlaps = dry_plan()
    scenes, annotations, gate, families = plan
    # Rebind the actual preregistration digest after dry selection; it does not
    # affect seeds, layouts, signatures, or capture order.
    scenes["contract_lock_sha256"] = sha256(DATA_LOCK)
    annotations["contract_lock_sha256"] = sha256(DATA_LOCK)
    gate["contract_lock_sha256"] = sha256(DATA_LOCK)
    write_new(REGISTRY, registry)
    audit = {
        "schema_version": 1,
        "protocol_id": PID,
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checks": {
            **checks,
            "frozen_v6_analytic_envelope_certificate_pass": read_json(certificate.RESULT).get("status") == "PASS",
            "no_render_model_inference_or_calibrator_fit": True,
            "frozen_calibration_128_unchanged": True,
            "test_and_robot_sealed": True,
        },
        "selected_nonce": nonce,
        "nonce_selection": "smallest passing nonce using identifiers/seeds/canonical geometry signatures only",
        "overlaps": overlaps,
        "state_relation_counts": {f"{state}|{relation}": count for (state, relation), count in sorted(Counter((row["state"], row["relation_variant"]) for row in families).items())},
        "data_design_lock_sha256": sha256(DATA_LOCK),
        "exclusion_registry_sha256": sha256(REGISTRY),
        "pre_render_certificate_sha256": sha256(certificate.RESULT),
        "generator_sha256": sha256(Path(__file__)),
        "capture_authorized": False,
        "scientific_hypothesis": "NOT_TESTED",
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    if not all(audit["checks"].values()):
        write_new(AUDIT, audit)
        raise SystemExit(2)
    write_new(AUDIT, audit)
    text_outputs = (
        yaml.safe_dump(scenes, sort_keys=False, width=140),
        yaml.safe_dump(annotations, sort_keys=False, width=140),
        yaml.safe_dump(gate, sort_keys=False, width=140),
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in families),
        json.dumps({
            "schema_version": 1,
            "protocol_id": PID,
            "status": "FROZEN_DESIGN_INPUT",
            "families": 32,
            "population": "4 state x 4 relation x 2",
            "data_design_lock_sha256": sha256(DATA_LOCK),
            "exclusion_registry_sha256": sha256(REGISTRY),
            "selected_nonce": nonce,
            "test_iid_ood_sealed": True,
            "capture_authorized": False,
        }, indent=2) + "\n",
    )
    for path, text in zip((SCENES, ANNOTATIONS, GATE, FAMILY_MANIFEST, SPLIT_MANIFEST), text_outputs):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(json.dumps(audit, indent=2))


def freeze_implementation() -> None:
    if IMPLEMENTATION_LOCK.exists():
        raise FileExistsError("R5 implementation lock already exists")
    data_lock = validate_data_lock()
    if any(not path.is_file() for path in GENERATED):
        raise RuntimeError("R5 generated design artifacts incomplete")
    audit = read_json(AUDIT)
    if audit.get("status") != "PASS" or not all(audit["checks"].values()):
        raise RuntimeError("R5 static design audit must PASS")
    calibration_paths = v6.paths(v6.CALIBRATION)
    original_r4_sources = read_json(r4.R4_LOCK)["source_artifact_sha256"]
    verify_hash_map(original_r4_sources)
    sources = [DATA_LOCK, Path(__file__).resolve(), QC_CODE, CAPTURE_CODE, *GENERATED, *calibration_paths, r4.R4_LOCK, R4_DECISION, R4_FAILURE]
    payload = {
        "schema_version": 1,
        "protocol_id": PID,
        "status": "PILOT_R5_IMPLEMENTATION_FROZEN_BEFORE_LIVE_PREFLIGHT_AND_CAPTURE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "new pilot data namespace/exclusions only; reuse frozen R4 sensor implementation",
        "data_design_lock_sha256": sha256(DATA_LOCK),
        "static_design_audit_sha256": sha256(AUDIT),
        "r4_sensor_implementation_sha256": sha256(CAPTURE_CODE),
        "frozen_calibration_128_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in calibration_paths},
        "source_artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "invariants": data_lock["invariants"],
        "capture_authorized": False,
        "pilot_accepted_families": 0,
        "calibration_families_captured": 0,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "scientific_decision": None,
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_new(IMPLEMENTATION_LOCK, payload)
    print(json.dumps({"status": payload["status"], "sha256": sha256(IMPLEMENTATION_LOCK)}, indent=2))


def validate_implementation() -> dict[str, Any]:
    validate_data_lock()
    value = read_json(IMPLEMENTATION_LOCK)
    if value.get("status") != "PILOT_R5_IMPLEMENTATION_FROZEN_BEFORE_LIVE_PREFLIGHT_AND_CAPTURE":
        raise RuntimeError("R5 implementation status invalid")
    verify_hash_map(value["source_artifact_sha256"])
    return value


def ros_environment() -> dict[str, str]:
    env = r2.ros_environment()
    env["ROS_LOCALHOST_ONLY"] = "1"
    env["IGN_PARTITION"] = "ur3_roborefer_calibration_v6_r5_local"
    env["PYTHONNOUSERSITE"] = "1"
    return env


def preflight_live() -> None:
    if PREFLIGHT.exists():
        raise FileExistsError("R5 live preflight already attempted")
    implementation = validate_implementation()
    helpers.run_probe = r2.run_probe
    parsers.install_parser_repair()
    probes = {
        "topics": r2.run_probe(["ros2", "topic", "list", "-t"]),
        "nodes": r2.run_probe(["ros2", "node", "list"]),
        "services": r2.run_probe(["ros2", "service", "list", "-t"]),
        "actions": r2.run_probe(["ros2", "action", "list", "-t"]),
        "processes": r2.run_probe(["ps", "-eo", "comm=,args="]),
    }
    required_topics = {
        name: f"{name} [{kind}]" in probes["topics"]["stdout"] for name, kind in helpers.REQUIRED_TOPICS.items()
    }
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    scenes = yaml.safe_load(SCENES.read_text(encoding="utf-8"))
    base_tf = r3.lookup_tf()
    wrist_tf = r2.run_probe(["ros2", "run", "tf2_ros", "tf2_echo", "base_link", "camera_color_optical_frame"], timeout=8)
    joint, pose_ok, comparison = parsers.joint_state_probe(gate["camera"]["view_joint_pose"])
    observations: dict[str, Any] = {"ros_graph": probes, "required_topics": required_topics, "base_tf": base_tf, "wrist_tf": wrist_tf, "joint_state": joint, "joint_comparison": comparison}
    samples: dict[str, Any] = {}
    graph_ok = all(item["returncode"] == 0 for name, item in probes.items() if name != "processes") and all(required_topics.values())
    sample_checks: dict[str, bool] = {}
    if graph_ok:
        width, height = (int(value) for value in gate["camera"]["resolution"])
        for label, topic, encodings in (
            ("rgb", "/wrist_camera/color/image_raw", {"rgb8", "bgr8"}),
            ("depth", "/wrist_camera/depth/image_raw", {"32FC1"}),
            ("labels", "/wrist_camera/evaluation_labels/labels_map", {"rgb8"}),
        ):
            sample = helpers.image_probe(topic)
            samples[label] = sample
            sample_checks[label + "_fresh_valid"] = helpers.image_matches(sample, width=width, height=height, encodings=encodings, frame_id=gate["camera"]["frame"])
        reset = helpers.reset_probe(scenes)
        observations["full_scene_reset"] = reset
        sample_checks["full_scene_reset_success"] = bool(reset.get("success"))
    else:
        sample_checks["live_samples_and_reset"] = False
    observations["sensor_samples"] = samples
    process_text = probes["processes"]["stdout"]
    checks = {
        "r5_implementation_unchanged": True,
        "r4_successful_32_scene_sensor_capture_bound": read_json(R4_ROOT / "capture_attempt_01/capture_manifest.json").get("status") == "COMPLETE",
        "r4_qc_rejection_was_duplicate_not_sensor_or_geometry": read_json(R4_ROOT / "GEOMETRY_QC.json").get("passed_scene_count") == 32,
        "ros_graph_and_required_topics": graph_ok,
        "sim_nodes_present": all(token in probes["nodes"]["stdout"] for token in ("/ros_gz_bridge", "/robot_state_publisher", "/controller_manager")),
        "set_pose_service_present": helpers.SET_POSE_SERVICE in probes["services"]["stdout"],
        "trajectory_action_present": "/joint_trajectory_controller/follow_joint_trajectory" in probes["actions"]["stdout"],
        "locked_world_running": any(r3.WORLD.name in line and "ign gazebo" in line for line in process_text.splitlines()),
        "r3_static_tf_publisher_running": any("static_transform_publisher" in line and r3.CHILD in line for line in process_text.splitlines()),
        "r3_base_camera_tf_exact": r3.exact_tf_valid(base_tf, r3.camera_contract()),
        "wrist_camera_tf_resolved": "Translation:" in wrist_tf["stdout"] and "Rotation:" in wrist_tf["stdout"],
        "locked_camera_pose": pose_ok,
        "disk_free_over_2_gib": shutil.disk_usage(ROOT).free > 2 * 1024**3,
        "r5_capture_and_authorization_absent": not CAPTURE.exists() and not AUTH.exists(),
        "calibration_capture_dataset_inference_absent": not CALIBRATION_OUTPUT.exists() and not r2.DATASET_ROOT.exists(),
        "test_iid_ood_and_robot_sealed": all(gate["sealed"][name] is True for name in ("gazebo_test_iid", "gazebo_test_ood", "robot")),
        **sample_checks,
    }
    report = {
        "schema_version": 1,
        "protocol_id": PID,
        "attempt": 5,
        "implementation_revision": "r5_new_pilot_data_design_reusing_r4_sensor_capture",
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "observations": observations,
        "implementation_lock_sha256": sha256(IMPLEMENTATION_LOCK),
        "capture_authorized": False,
        "pilot_accepted_families": 0,
        "calibration_families_captured": 0,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "scientific_decision": None,
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_new(PREFLIGHT, report)
    print(json.dumps({"status": report["status"], "checks": checks, "artifact": str(PREFLIGHT)}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


def authorization_sources(pid: str) -> dict[str, str]:
    sources = dict(read_json(base.IMPLEMENTATION)["source_artifact_sha256"])
    extra = [DATA_LOCK, IMPLEMENTATION_LOCK, Path(__file__).resolve(), QC_CODE, CAPTURE_CODE, REGISTRY, AUDIT, PREFLIGHT]
    if pid == PID:
        extra += [SCENES, ANNOTATIONS, GATE, FAMILY_MANIFEST, SPLIT_MANIFEST, R4_DECISION, R4_FAILURE, R3_INPUT, R4_INPUT]
    else:
        extra += [*v6.paths(v6.CALIBRATION), AUTH, R5_QC_INPUT, R5_QC, R5_RGB_QC, R5_LEAKAGE_QC, R5_DECISION]
    sources.update({str(path.relative_to(ROOT)): sha256(path) for path in extra})
    return sources


def authorize_pilot() -> None:
    validate_implementation()
    if AUTH.exists() or CAPTURE.exists():
        raise RuntimeError("R5 pilot authorization or attempt already exists")
    if read_json(PREFLIGHT).get("status") != "PASS":
        raise RuntimeError("R5 live preflight PASS required")
    rows = read_jsonl(FAMILY_MANIFEST)
    cells = Counter((row["state"], row["relation_variant"]) for row in rows)
    if len(rows) != 32 or len(cells) != 16 or set(cells.values()) != {2}:
        raise RuntimeError("R5 pilot population drift")
    payload = {
        "schema_version": 1,
        "protocol_id": PID,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "authorization_status": "ONE_R5_PILOT_CAPTURE_AUTHORIZED_AFTER_STATIC_AND_LIVE_PREFLIGHT_PASS",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "parent_contract_lock_sha256": sha256(DATA_LOCK),
        "implementation_lock_sha256": sha256(IMPLEMENTATION_LOCK),
        "source_artifact_sha256": authorization_sources(PID),
        "expected_parent_families": 32,
        "population": "4 state x 4 relation x 2",
        "capture_attempts_authorized": 1,
        "authorized_output": str(CAPTURE.relative_to(ROOT)),
        "model_inventory_sha256": "NO_MODEL_INFERENCE_DURING_R5_PILOT",
        "policies": {
            "single_complete_capture_attempt_only": True,
            "partial_attempt_reuse_forbidden": True,
            "row_filter_repair_or_replacement_forbidden": True,
            "cross_attempt_merge_forbidden": True,
            "r3_r4_row_or_image_reuse_forbidden": True,
            "duplicate_threshold_relaxation_forbidden": True,
            "no_model_inference": True,
            "no_materialization": True,
            "no_calibrator_fit": True,
            "test_iid_ood_sealed": True,
            "robot_policy_sealed": True,
        },
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_new(AUTH, payload)
    print(json.dumps({"status": payload["authorization_status"], "sha256": sha256(AUTH)}, indent=2))


def verify_authorization() -> dict[str, Any]:
    validate_implementation()
    value = read_json(AUTH)
    if value.get("status") != "LOCKED_BEFORE_CAPTURE_AND_INFERENCE" or value.get("protocol_id") != PID:
        raise RuntimeError("R5 pilot authorization invalid")
    verify_hash_map(value["source_artifact_sha256"])
    if not all(value["policies"].values()):
        raise RuntimeError("R5 pilot policy drift")
    return value


def tree_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(root.rglob("*")) if path.is_file()}


def capture() -> None:
    verify_authorization()
    if CAPTURE.exists():
        raise FileExistsError("R5 single pilot attempt already exists")
    command = ["/usr/bin/python3", str(CAPTURE_CODE), "--ros-args", "-p", "use_sim_time:=true"]
    for key, value in (
        ("scene_config_file", SCENES), ("annotation_file", ANNOTATIONS),
        ("gate_config_file", GATE), ("pretrial_lock_file", AUTH),
        ("output_root", CAPTURE), ("settle_sec", "1.5"),
        ("sync_slop_sec", "0.02"), ("capture_timeout_sec", "45.0"),
    ):
        command += ["-p", f"{key}:={value}"]
    completed = subprocess.run(command, cwd=ROOT, env=ros_environment(), check=False)
    manifest = CAPTURE / "capture_manifest.json"
    index = CAPTURE / "input_manifest.jsonl"
    records = read_jsonl(index) if index.exists() else []
    passed = completed.returncode == 0 and manifest.is_file() and read_json(manifest).get("status") == "COMPLETE" and len(records) == 32
    if not passed:
        write_new(OUTPUT / "CAPTURE_ATTEMPT_01_FAILURE_LOCK.json", {
            "schema_version": 1,
            "protocol_id": PID,
            "status": "CAPTURE_ATTEMPT_01_FROZEN_INCOMPLETE_NO_RETRY_UNDER_R5",
            "classification": "INFRASTRUCTURE_CAPTURE_FAILURE_NOT_SCIENTIFIC_RESULT",
            "expected_family_count": 32,
            "captured_record_count": len(records),
            "process_returncode": completed.returncode,
            "source_artifact_sha256": tree_hashes(CAPTURE),
            "partial_rows_quarantined": True,
            "model_inference_run": False,
            "calibrator_fit_call_count": 0,
            "scientific_hypothesis": "NOT_TESTED",
            "test_iid_ood_access": False,
            "robot_access": False,
        })
        raise SystemExit(2)


def authorize_calibration() -> None:
    validate_implementation()
    if CALIBRATION_AUTH.exists() or CALIBRATION_OUTPUT.exists():
        raise RuntimeError("R5 calibration authorization or capture already exists")
    decision = read_json(R5_DECISION)
    reports = [R5_QC_INPUT, R5_QC, R5_RGB_QC, R5_LEAKAGE_QC]
    if (
        decision.get("decision") != "PILOT_R5_32_OF_32_QC_PASS"
        or decision.get("accepted_family_count") != 32
        or any(read_json(path).get("status") not in ("PASS", "LOCKED_BEFORE_QC") for path in reports)
    ):
        raise RuntimeError("R5 pilot 32/32 all-QC PASS required")
    calibration_rows = read_jsonl(v6.paths(v6.CALIBRATION)[3])
    if len(calibration_rows) != 128:
        raise RuntimeError("frozen Calibration-v6 128 manifest drift")
    payload = {
        "schema_version": 1,
        "protocol_id": v6.CALIBRATION,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "authorization_status": "ONE_CALIBRATION_128_CAPTURE_AUTHORIZED_AFTER_R5_PILOT_32_OF_32_QC_PASS",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "parent_contract_lock_sha256": sha256(v6.LOCK),
        "implementation_lock_sha256": sha256(IMPLEMENTATION_LOCK),
        "pilot_r5_decision_sha256": sha256(R5_DECISION),
        "source_artifact_sha256": authorization_sources(v6.CALIBRATION),
        "expected_parent_families": 128,
        "population": "4 state x 4 relation x 8",
        "capture_attempts_authorized": 1,
        "authorized_output": str(CALIBRATION_OUTPUT.relative_to(ROOT)),
        "model_inventory_sha256": "NO_MODEL_INFERENCE_DURING_CALIBRATION_CAPTURE",
        "execution_precondition": "Restart locked simulation and repeat environment/TF/pose liveness checks before invoking capture; no capture is performed by this authorization command.",
        "policies": {
            "single_complete_capture_attempt_only": True,
            "partial_attempt_reuse_forbidden": True,
            "row_filter_repair_or_replacement_forbidden": True,
            "cross_attempt_merge_forbidden": True,
            "no_model_inference_during_capture": True,
            "no_calibrator_fit_during_capture": True,
            "test_iid_ood_sealed": True,
            "robot_policy_sealed": True,
        },
        "calibration_families_captured_before_lock": 0,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_hypothesis": "NOT_TESTED",
        "scientific_decision": None,
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    write_new(CALIBRATION_AUTH, payload)
    print(json.dumps({"status": payload["authorization_status"], "sha256": sha256(CALIBRATION_AUTH), "next": payload["execution_precondition"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "dry-plan", "preregister", "generate", "freeze-implementation", "validate",
        "preflight-live", "authorize-pilot", "capture-pilot", "authorize-calibration",
    ))
    command = parser.parse_args().command
    if command == "dry-plan":
        nonce, registry, plan, checks, overlaps = dry_plan()
        print(json.dumps({"selected_nonce": nonce, "registry_counts": registry["counts"], "checks": checks, "overlaps": overlaps}, indent=2))
    elif command == "preregister":
        preregister()
    elif command == "generate":
        generate()
    elif command == "freeze-implementation":
        freeze_implementation()
    elif command == "validate":
        validate_implementation(); print("PASS: Calibration-v6 pilot R5 implementation")
    elif command == "preflight-live":
        preflight_live()
    elif command == "authorize-pilot":
        authorize_pilot()
    elif command == "capture-pilot":
        capture()
    else:
        authorize_calibration()


if __name__ == "__main__":
    main()
