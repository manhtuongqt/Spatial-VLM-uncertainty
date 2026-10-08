#!/usr/bin/env python3
"""Generate the preregistered v4 pilot and calibration layouts without rendering.

All construction constants and the sole permissible v3 template use come from
GAZEBO_CALIBRATION_V4_DATA_DESIGN_LOCK.json.  This module neither accesses
model predictions nor selects a replacement layout based on observed images.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import yaml

import generate_gazebo_calibration_v3_contract as v3


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocol"
CONFIG = ROOT / "ur3/ur3_perception/config"
LOCK = PROTOCOL / "GAZEBO_CALIBRATION_V4_DATA_DESIGN_LOCK.json"
EXPECTED_LOCK_SHA256 = "3f31389613f8aec15bcd19f2132bfe4f939fb3cfa51bdf1e571d71058e7ce3d9"
TF_SNAPSHOT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/capture_attempt_01/gazebo_calibration_v3_100/input/tf_snapshot.json"
WORLDFILE = ROOT / "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf"
PILOT_ID = "gazebo_calibration_v4_geometry_pilot"
CALIBRATION_ID = "gazebo_calibration_v4"
STATES = ("FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE")
RELATIONS = ("leftmost", "rightmost", "second_from_left", "second_from_right")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_and_sources() -> dict:
    if sha256(LOCK) != EXPECTED_LOCK_SHA256:
        raise RuntimeError("v4 data-design lock hash drift")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    pinned = {lock["parent_contract"]["path"]: lock["parent_contract"]["sha256"]}
    pinned.update(lock["frozen_upstream_sha256"])
    pinned.update(lock["frozen_failure_evidence_sha256"])
    for name, digest in pinned.items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"frozen predecessor drift: {name}")
    failure = json.loads((ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v3/CALIBRATION_V3_CAPTURE_ATTEMPT_01_QC_FAILURE_LOCK.json").read_text(encoding="utf-8"))
    item = failure["capture_file_inventory"].get(str(TF_SNAPSHOT.relative_to(ROOT)))
    if item is None or sha256(TF_SNAPSHOT) != item["sha256"]:
        raise RuntimeError("camera transform is not in the frozen failed-attempt inventory")
    return lock


def camera_rotation(quaternion: list[float]) -> list[list[float]]:
    x, y, z, w = quaternion
    norm = math.sqrt(sum(value * value for value in quaternion))
    x, y, z, w = (value / norm for value in (x, y, z, w))
    return [
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ]


def rotate(rotation: list[list[float]], vector: tuple[float, float, float]) -> list[float]:
    return [sum(row[col] * vector[col] for col in range(3)) for row in rotation]


def camera_transform() -> tuple[list[float], list[list[float]]]:
    tf = json.loads(TF_SNAPSHOT.read_text(encoding="utf-8"))["camera_color_optical_frame"]
    if tf.get("parent_frame") != "base_link" or tf.get("child_frame") != "camera_color_optical_frame":
        raise RuntimeError("fixed view TF contract mismatch")
    return tf["position"], camera_rotation(tf["orientation_xyzw"])


def inverse_project(u: float, v: float, support_z: float, camera: dict, origin: list[float], rotation: list[list[float]]) -> tuple[float, float]:
    k = camera["intrinsics"]
    ray = rotate(rotation, ((u*640-k["cx"])/k["fx"], (v*480-k["cy"])/k["fy"], 1.0))
    if ray[2] >= -1e-10:
        raise ValueError("camera ray cannot intersect the support plane in front of camera")
    t = (support_z - origin[2]) / ray[2]
    if not (math.isfinite(t) and 0.1 <= t <= 2.0):
        raise ValueError("inverse projected support-plane depth invalid")
    return (origin[0] + t * ray[0], origin[1] + t * ray[1])


def project(point: tuple[float, float, float], camera: dict, origin: list[float], rotation: list[list[float]]) -> tuple[float, float, float]:
    d = [point[i] - origin[i] for i in range(3)]
    xyz = [sum(rotation[j][i]*d[j] for j in range(3)) for i in range(3)]
    if xyz[2] <= 0:
        raise ValueError("object projects behind the camera")
    k = camera["intrinsics"]
    return ((k["fx"]*xyz[0]/xyz[2]+k["cx"])/640, (k["fy"]*xyz[1]/xyz[2]+k["cy"])/480, xyz[2])


def seed_for(protocol_id: str, family: str, state: str, relation: str, repetition: int) -> int:
    raw = f"{protocol_id}_seed_v1|{family}|{state}|{relation}|{repetition}"
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16)


def capture_order_key(protocol_id: str, family: str) -> str:
    return hashlib.sha256(f"{protocol_id}_capture_order_v1|{family}".encode("utf-8")).hexdigest()


def audit_ie_geometry(scene: dict, origin: list[float], rotation: list[list[float]], camera: dict, objects: dict, relation: str) -> list[str]:
    poses = scene["poses"]
    target = poses["mango"]
    panel = poses["uq_neutral_occluder"]
    centers: dict[str, tuple[float, float, float]] = {}
    reasons: list[str] = []
    for name in ("mango", "uq_neutral_occluder", "ycb_apple", "ycb_orange"):
        pose = poses[name]
        center = project((pose[0], pose[1], objects[name]["z"]), camera, origin, rotation)
        centers[name] = center
        if name in ("ycb_apple", "ycb_orange"):
            # A fixed, conservative analytic screen; actual visibility is
            # evaluated only by the later immutable rendered pilot QC.
            if not (0.08 <= center[0] <= 0.92 and 0.08 <= center[1] <= 0.92):
                reasons.append(f"CANDIDATE_IMAGE_BORDER:{name}")
            if not (-0.8 + 0.038 <= pose[0] <= 0.8 - 0.038 and -0.65 + 0.038 <= pose[1] <= 1.55 - 0.038):
                reasons.append(f"CANDIDATE_OUTSIDE_TABLE:{name}")
            # Panel is a 0.1 x 0.1 box; using a circumscribed 0.071 m
            # radius is conservative for any locked panel yaw.
            for other, radius in (("mango", 0.030), ("uq_neutral_occluder", 0.071)):
                target_pose = poses[other]
                clearance = math.hypot(pose[0]-target_pose[0], pose[1]-target_pose[1]) - (objects[name]["z"] + radius)
                if clearance < 0.01:
                    reasons.append(f"CANDIDATE_FOOTPRINT_CLEARANCE:{name}:{other}:{clearance:.8f}")
                p = centers[other]
                if math.hypot(center[0]-p[0], center[1]-p[1]) < 0.08:
                    reasons.append(f"CANDIDATE_IMAGE_CENTER_CLEARANCE:{name}:{other}")
    p1, p2 = poses["ycb_apple"], poses["ycb_orange"]
    if math.hypot(p1[0]-p2[0], p1[1]-p2[1]) - objects["ycb_apple"]["z"] - objects["ycb_orange"]["z"] < 0.01:
        reasons.append("NON_TARGET_CANDIDATES_PHYSICALLY_OVERLAP")
    if math.hypot(centers["ycb_apple"][0]-centers["ycb_orange"][0], centers["ycb_apple"][1]-centers["ycb_orange"][1]) < 0.12:
        reasons.append("NON_TARGET_CANDIDATES_IMAGE_OVERLAP")
    u = centers["mango"][0]
    context_u = sorted([centers[name][0] for name in ("ycb_apple", "ycb_orange")])
    expected = {
        "leftmost": u < context_u[0], "rightmost": u > context_u[-1],
        "second_from_left": context_u[0] < u < context_u[1],
        "second_from_right": context_u[0] < u < context_u[1],
    }
    if not expected[relation]:
        reasons.append("RELATION_ORDER_MISMATCH")
    if not (-0.2 <= u < 1.2 and -0.2 <= centers["mango"][1] < 1.2 and 0.1 <= centers["mango"][2] <= 2):
        reasons.append("TARGET_PROJECTION_OUTSIDE_INHERITED_GATE")
    return reasons


def make_split(protocol_id: str, repetitions: int, lock: dict, camera: dict, origin: list[float], rotation: list[list[float]]) -> tuple[dict, dict, dict, list[dict], list[dict]]:
    v3_scenes = yaml.safe_load(v3.SCENES_PATH.read_text(encoding="utf-8"))
    v3_oracle = yaml.safe_load(v3.ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    gate = yaml.safe_load(v3.GATE_PATH.read_text(encoding="utf-8"))
    templates = {}
    for row in v3_scenes["scenes"]:
        annotation = v3_oracle["scenes"][row["scene_id"]]
        templates[(annotation["state"], annotation["relation_variant"], annotation["cell_repetition"])] = (row, annotation)
    rows, annotations, families, failures = [], {}, [], []
    pilot = protocol_id == PILOT_ID
    for state_index, state in enumerate(STATES):
        for relation_index, relation in enumerate(RELATIONS):
            for repetition in range(repetitions):
                index = ((state_index*4+relation_index)*repetitions)+repetition
                scene_id = f"{protocol_id}_{index:03d}"
                prefix = f"spatial_vlm_{protocol_id}"
                family_id = f"{prefix}/{'development' if pilot else 'calibration'}/parent_{index:03d}"
                layout_id = f"{protocol_id}_layout_{index:03d}"
                h = seed_for(protocol_id, family_id, state, relation, repetition)
                template, oracle = templates[(state, relation, repetition)]
                row = copy.deepcopy(template)
                row.update(scene_id=scene_id, scene_family_id=family_id, layout_id=layout_id)
                poses = row["poses"]
                if state == "INSUFFICIENT_EVIDENCE":
                    anchors = lock["geometry_design_revision"]["relation_aware_context_projected_center_targets_normalized"][relation]
                    for i, name in enumerate(sorted(("ycb_apple", "ycb_orange"))):
                        u = anchors[i][0] + 0.0040*((((h >> (8*i)) % 5))-2)
                        v = anchors[i][1] + 0.0050*((((h >> (8*i+3)) % 5))-2)
                        z = v3_scenes["objects"][name]["z"]
                        x, y = inverse_project(u, v, z, camera, origin, rotation)
                        poses[name] = [x, y, 0.0700+0.0070*((h >> (16+3*i)) % 11)]
                    failures.extend({"scene_id": scene_id, "reason": reason} for reason in audit_ie_geometry(row, origin, rotation, camera, v3_scenes["objects"], relation))
                else:
                    dx = 0.0020 + 0.0007*(h % 7)
                    dy = -0.0030 + 0.0006*((h >> 3) % 9)
                    yaw = 0.0340 + 0.0050*((h >> 7) % 11)
                    for pose in poses.values():
                        pose[0] += dx; pose[1] += dy; pose[2] += yaw
                signature = v3.layout_signature(row, v3_scenes["objects"])
                row["layout_signature_sha256"] = signature
                annotation = copy.deepcopy(oracle)
                annotation.update(family_id=family_id, layout_id=layout_id, layout_signature_sha256=signature,
                                  split="development" if pilot else "calibration", seed=h,
                                  cell_repetition=repetition, geometry_provenance="frozen_v4_data_design_geometry_only")
                annotations[scene_id] = annotation
                families.append({"family_index": index, "scene_id": scene_id, "family_id": family_id,
                                 "layout_id": layout_id, "layout_signature_sha256": signature,
                                 "split": annotation["split"], "state": state, "relation_variant": relation,
                                 "cell_repetition": repetition, "deterministic_seed": h,
                                 "capture_order_sha256": capture_order_key(protocol_id, family_id)})
                rows.append(row)
    ordered = sorted(families, key=lambda family: family["capture_order_sha256"])
    for order, family in enumerate(ordered): family["capture_order"] = order
    rows.sort(key=lambda row: next(f["capture_order"] for f in families if f["scene_id"] == row["scene_id"]))
    scenes = copy.deepcopy(v3_scenes)
    scenes.update(protocol_id=protocol_id, contract_lock_sha256=sha256(LOCK), expected_scene_count=len(rows),
                  capture_order="sha256_sort_v4_family_id", scenes=rows)
    annotation_doc = copy.deepcopy(v3_oracle)
    annotation_doc.update(protocol_id=protocol_id, contract_lock_sha256=sha256(LOCK),
                          oracle_usage="geometry_only_no_model" if pilot else "locked_affine_calibration_only_after_frozen_raw_predictions",
                          scenes=annotations)
    gate.update(protocol_id=protocol_id, contract_lock_sha256=sha256(LOCK), parent_family_count=len(rows),
                split_parent_family_count={"development" if pilot else "calibration": len(rows)},
                state_quota={"development" if pilot else "calibration": {state: len(rows)//4 for state in STATES}},
                relation_variant_quota={"development" if pilot else "calibration": {relation: len(rows)//4 for relation in RELATIONS}},
                state_relation_cell_quota={"development" if pilot else "calibration": repetitions})
    if pilot:
        gate["policies"].update(no_model_inference=True, no_materialization=True)
    return scenes, annotation_doc, gate, families, failures


def artifact_paths(protocol_id: str) -> tuple[Path, Path, Path, Path, Path]:
    return (CONFIG / f"{protocol_id}_scenes.yaml", CONFIG / f"{protocol_id}_annotations.yaml",
            CONFIG / f"{protocol_id}_gate.yaml", PROTOCOL / f"{protocol_id}_family_manifest.jsonl",
            PROTOCOL / f"{protocol_id}_split_manifest.json")


def generate() -> tuple[dict, dict]:
    lock = lock_and_sources()
    origin, rotation = camera_transform()
    camera = json.loads((ROOT / lock["parent_contract"]["path"]).read_text(encoding="utf-8"))["camera_and_geometry"]["camera"]
    pilot = make_split(PILOT_ID, 2, lock, camera, origin, rotation)
    calibration = make_split(CALIBRATION_ID, 8, lock, camera, origin, rotation)
    return {PILOT_ID: pilot, CALIBRATION_ID: calibration}, {
        "camera_tf_snapshot_path": str(TF_SNAPSHOT.relative_to(ROOT)),
        "camera_tf_snapshot_sha256": sha256(TF_SNAPSHOT),
        "world_sha256": sha256(WORLDFILE),
        "geometry_checks": {
            "pilot": pilot[-1], "calibration": calibration[-1],
        },
    }


def write_outputs(plans: dict) -> None:
    for protocol_id, (scenes, annotations, gate, families, _) in plans.items():
        paths = artifact_paths(protocol_id)
        if any(path.exists() for path in paths):
            raise FileExistsError(f"refusing to overwrite existing {protocol_id} implementation inputs")
        split = {"schema_version": 1, "protocol_id": protocol_id,
                 "status": "PREREGISTERED_IMPLEMENTATION_INPUT", "contract_lock_sha256": sha256(LOCK),
                 "families": len(families), "state_relation_cell_quota": len(families)//16,
                 "capture_authorized": False, "test_iid_ood_sealed": True,
                 "robot_access": False}
        texts = [yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=140)
                 for payload in (scenes, annotations, gate)]
        texts.append("".join(json.dumps(row, sort_keys=True) + "\n" for row in families))
        texts.append(json.dumps(split, indent=2, sort_keys=True) + "\n")
        for path, value in zip(paths, texts):
            path.write_text(value, encoding="utf-8")


if __name__ == "__main__":
    # Review the geometry audit before invoking write_outputs from an
    # implementation freezer; direct execution is deliberately read-only.
    plans, preview = generate()
    print(json.dumps({"preview": preview,
                      "counts": {name: len(plan[3]) for name, plan in plans.items()}},
                     indent=2, sort_keys=True))
