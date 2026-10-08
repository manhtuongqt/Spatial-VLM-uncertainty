#!/usr/bin/env python3
"""Prepare the preregistered 256-family Anti-Shortcut Val v1 capture.

This script creates capture inputs only.  It never opens a checkpoint and it
does not claim that a requested label is valid before semantic-geometry QC.
The design deliberately balances camera, object triplet, prompt template,
relation and answerability state so that those nuisance variables cannot be
used as answerability labels.
"""

from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ur3/ur3_perception/config"
OUT = ROOT / "ketqua1/09_danh_gia/anti_shortcut_val_v1_r7"
LOCK = ROOT / "protocol/MH_PCRAU_V3_ANTI_SHORTCUT_VAL_V1_LOCK.json"
PROTOCOL = "mh_pcrau_v3_anti_shortcut_val_v1_data_design_r7"
SOURCE_SCENES = CONFIG / "gazebo_train_uq_v2_full_r3_scenes.yaml"
SOURCE_ANNOTATIONS = CONFIG / "gazebo_train_uq_v2_full_r3_annotations.yaml"
SOURCE_GATE = CONFIG / "gazebo_train_uq_v2_full_r3_gate.yaml"
WORLD = ROOT / "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf"
CAPTURE_NODE = ROOT / "ur3/ur3_perception/scripts/roborefer_pilot_capture.py"
CAPTURE_LAUNCH = ROOT / "workspace/mh_pcrau_v3/g1/gazebo_pilot_512_view.launch.py"

STATES = ("FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE")
RELATIONS = ("leftmost", "rightmost", "second_from_left", "second_from_right")
PLACEHOLDERS = ("ycb_apple", "ycb_orange", "mango")
POSES = (
    (1.4315, -1.8273, 1.3428, -1.2863, -1.5708, -1.9601),
    (1.3315, -1.8273, 1.3428, -1.2863, -1.5708, -1.9601),
    (1.3815, -1.7773, 1.3428, -1.2863, -1.5708, -1.9601),
    (1.3815, -1.8773, 1.3428, -1.2863, -1.5708, -1.9601),
)
CUBE_LABELS = {
    "red_cube": 1,
    "blue_cube": 2,
    "green_cube": 3,
    "yellow_cube": 4,
    "orange_cube": 5,
    "purple_cube": 6,
    "pink_cube": 7,
}
CUBE_NAMES = {name: name.replace("_", " ") for name in CUBE_LABELS}
TRIPLETS = (
    ("yellow_cube", "orange_cube", "purple_cube"),
    ("blue_cube", "orange_cube", "pink_cube"),
    ("yellow_cube", "blue_cube", "pink_cube"),
    ("purple_cube", "yellow_cube", "blue_cube"),
    ("green_cube", "orange_cube", "purple_cube"),
    ("pink_cube", "green_cube", "yellow_cube"),
    ("orange_cube", "blue_cube", "purple_cube"),
    ("green_cube", "pink_cube", "orange_cube"),
)
TEMPLATES = {
    "leftmost": (
        "Among {items}, identify the leftmost object in the image.",
        "Considering {items}, point to the object furthest left in the image.",
        "From {items}, select the object at the far-left image position.",
        "Which object is leftmost among {items} in this image?",
    ),
    "rightmost": (
        "Among {items}, identify the rightmost object in the image.",
        "Considering {items}, point to the object furthest right in the image.",
        "From {items}, select the object at the far-right image position.",
        "Which object is rightmost among {items} in this image?",
    ),
    "second_from_left": (
        "Among {items}, identify the second object from left to right in the image.",
        "Considering {items}, point to the object second from the left in the image.",
        "From {items}, select the object in the second position when ordered left to right.",
        "Which object ranks second from the left among {items} in this image?",
    ),
    "second_from_right": (
        "Among {items}, identify the second object from right to left in the image.",
        "Considering {items}, point to the object second from the right in the image.",
        "From {items}, select the object in the second position when ordered right to left.",
        "Which object ranks second from the right among {items} in this image?",
    ),
}
SEED = 25092032
TF_SOURCES = tuple(
    ROOT
    / f"ketqua1/01_dau_vao_tien_xu_ly/ngay_03/gazebo_pilot_512_v1/batch_{batch}"
    / "capture_attempt_01"
    / f"v3_pilot_512_new_b{batch}_000/input/tf_snapshot.json"
    for batch in range(4)
)
CAMERA_SOURCES = tuple(path.with_name("camera_info.json") for path in TF_SOURCES)
REFERENCE_IE_INPUT = (
    ROOT
    / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03/gazebo_ie_repair_revision_v2"
    / "capture_attempt_01/g1_ie_repair_v2_000_1/input"
)
REFERENCE_IE_TF = REFERENCE_IE_INPUT / "tf_snapshot.json"
REFERENCE_IE_CAMERA = REFERENCE_IE_INPUT / "camera_info.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def full_layout_signature(objects: dict, poses: dict) -> str:
    full = {name: poses.get(name, spec["storage_pose"]) for name, spec in sorted(objects.items())}
    return digest(full)


def physical_signature(poses: dict) -> str:
    values = sorted(tuple(round(float(value), 6) for value in pose) for pose in poses.values())
    return digest(values)


def required_capture_sources() -> set[str]:
    tree = ast.parse(CAPTURE_NODE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "REQUIRED_SOURCE_ARTIFACTS"
            for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise RuntimeError("capture node REQUIRED_SOURCE_ARTIFACTS not found")


def historical_signatures() -> tuple[set[str], set[str]]:
    layouts: set[str] = set()
    physical: set[str] = set()

    def scene_rows(payload: dict) -> list[dict]:
        value = payload.get("scenes", [])
        if isinstance(value, dict):
            return [row for row in value.values() if isinstance(row, dict)]
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        return []

    for path in CONFIG.glob("*_scenes.yaml"):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for row in scene_rows(payload):
            if row.get("layout_signature_sha256"):
                layouts.add(str(row["layout_signature_sha256"]))
            if isinstance(row.get("poses"), dict):
                physical.add(physical_signature(row["poses"]))
    for path in ROOT.glob("ketqua1/**/*.yaml"):
        if OUT in path.parents or any(
            parent.name.startswith("anti_shortcut_val_v1_invalid_design")
            for parent in path.parents
        ):
            continue
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for row in scene_rows(payload):
            if row.get("layout_signature_sha256"):
                layouts.add(str(row["layout_signature_sha256"]))
            if isinstance(row.get("poses"), dict):
                physical.add(physical_signature(row["poses"]))
    return layouts, physical


def map_list(values: list, mapping: dict[str, str]) -> list:
    return [mapping.get(value, value) for value in values]


def map_annotation(source: dict, mapping: dict[str, str]) -> dict:
    result = deepcopy(source)
    for field in ("target_id", "target_category"):
        if result.get(field) in mapping:
            result[field] = mapping[result[field]]
    for field in ("candidate_ids", "context_ids", "valid_target_ids"):
        if field in result:
            result[field] = map_list(result[field], mapping)
    for ids_field, labels_field in (
        ("candidate_ids", "candidate_labels"),
        ("context_ids", "context_labels"),
        ("valid_target_ids", "valid_target_labels"),
    ):
        if ids_field in result:
            result[labels_field] = [CUBE_LABELS[name] for name in result[ids_field]]
    if result.get("target_id") is not None:
        result["target_label"] = CUBE_LABELS[result["target_id"]]
    return result


def rotation_matrix(quaternion: list[float]) -> np.ndarray:
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def projected_uv(
    x_value: float, y_value: float, z_value: float, transform: dict, camera: dict
) -> tuple[float, float]:
    point = np.asarray([x_value, y_value, z_value], dtype=np.float64)
    camera_point = rotation_matrix(transform["orientation_xyzw"]).T @ (
        point - np.asarray(transform["position"], dtype=np.float64)
    )
    if camera_point[2] <= 0:
        raise RuntimeError("candidate projects behind camera")
    intrinsics = camera["k"]
    return (
        float(intrinsics[0] * camera_point[0] / camera_point[2] + intrinsics[2]),
        float(intrinsics[4] * camera_point[1] / camera_point[2] + intrinsics[5]),
    )


def projected_u(x_value: float, y_value: float, transform: dict, camera: dict) -> float:
    return projected_uv(x_value, y_value, 0.025, transform, camera)[0]


def solve_projected_tie_y(
    first_pose: list[float], second_x: float, transform: dict, camera: dict,
    desired_offset_px: float,
) -> float:
    target_u = projected_u(first_pose[0], first_pose[1], transform, camera) + desired_offset_px
    low, high = first_pose[1] - 0.20, first_pose[1] + 0.20
    f_low = projected_u(second_x, low, transform, camera) - target_u
    f_high = projected_u(second_x, high, transform, camera) - target_u
    if f_low * f_high > 0:
        raise RuntimeError("cannot bracket projected tie solution")
    for _ in range(80):
        middle = 0.5 * (low + high)
        f_middle = projected_u(second_x, middle, transform, camera) - target_u
        if f_low * f_middle <= 0:
            high, f_high = middle, f_middle
        else:
            low, f_low = middle, f_middle
    solved = 0.5 * (low + high)
    if abs(projected_u(second_x, solved, transform, camera) - target_u) > 1e-7:
        raise RuntimeError("projected tie solver did not converge")
    return solved


def solve_occluder_offset(
    target_pose: list[float], transform: dict, camera: dict,
    reference_transform: dict, reference_camera: dict,
) -> tuple[float, float]:
    """Match the certified v2-panel image displacement at every camera pose.

    The reference pair (dx=0.0032 m, gap=0.0501 m) produced 71 visible target
    pixels in an earlier geometry-certified capture using the square v2 panel.
    R7 solves both world x offset and y gap from frozen camera calibration so
    the panel/target centre displacement in pixels is unchanged.  No R7 image,
    semantic map or model prediction is used by this solver.
    """
    x_value, y_value = float(target_pose[0]), float(target_pose[1])
    reference_target = projected_uv(x_value, y_value, 0.025, reference_transform, reference_camera)
    reference_panel = projected_uv(
        x_value + 0.0032, y_value - 0.0501, 0.060,
        reference_transform, reference_camera,
    )
    desired = np.asarray(reference_panel) - np.asarray(reference_target)
    actual_target = projected_uv(x_value, y_value, 0.025, transform, camera)

    def residual(offset: np.ndarray) -> np.ndarray:
        dx, gap = float(offset[0]), float(offset[1])
        panel = projected_uv(x_value + dx, y_value - gap, 0.060, transform, camera)
        return np.asarray(panel) - np.asarray(actual_target) - desired

    offset = np.asarray([0.0032, 0.0501], dtype=np.float64)
    epsilon = 1e-6
    for _ in range(12):
        value = residual(offset)
        if float(np.linalg.norm(value)) < 1e-7:
            break
        jacobian = np.column_stack([
            (residual(offset + np.asarray([epsilon, 0.0])) - value) / epsilon,
            (residual(offset + np.asarray([0.0, epsilon])) - value) / epsilon,
        ])
        offset -= np.linalg.solve(jacobian, value)
    if float(np.linalg.norm(residual(offset))) > 1e-4:
        raise RuntimeError("occluder-offset projection solver did not converge")
    dx, gap = (float(offset[0]), float(offset[1]))
    if not (-0.040 < dx < 0.040 and 0.020 < gap < 0.085):
        raise RuntimeError(f"occluder-offset solution outside safety bounds: {dx}, {gap}")
    return dx, gap


def prompt(relation: str, triplet: tuple[str, str, str], template_index: int) -> str:
    names = [CUBE_NAMES[name] for name in triplet]
    items = f"the {names[0]}, {names[1]}, and {names[2]}"
    return TEMPLATES[relation][template_index].format(items=items)


def jittered_poses(
    source: dict,
    mapping: dict[str, str],
    token: str,
    state: str,
    source_target: str | None,
    source_valid_targets: list[str] | None,
    relation: str,
    camera_stratum: int,
    transform: dict,
    camera: dict,
    reference_transform: dict,
    reference_camera: dict,
) -> dict:
    rng = random.Random(int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big"))
    dx = rng.uniform(-0.0090, 0.0090)
    dy = rng.uniform(-0.0060, 0.0060)
    dy = dy if abs(dy) >= 0.0007 else (0.0007 if dy >= 0 else -0.0007)
    yaw = rng.uniform(-0.035, 0.035)
    result = {}
    for old_name, pose in source.items():
        name = mapping.get(old_name, old_name)
        copied = [float(pose[0]) + dx, float(pose[1]) + dy, float(pose[2]) + yaw]
        result[name] = copied
    if state == "AMBIGUOUS":
        if not source_valid_targets or len(source_valid_targets) != 2:
            raise RuntimeError("ambiguous case lacks exactly two valid targets")
        first, second = (mapping[name] for name in source_valid_targets)
        # Keep physical separation while preregistering a 0.018-normalized
        # image-space gap.  Exact centre overlap in R4 sometimes hid one cube;
        # 0.018 remains below the 0.03 ambiguity threshold and leaves both
        # candidates visible.  Semantic-label QC remains authoritative.
        direction = 1.0 if result[second][0] >= result[first][0] else -1.0
        result[second][0] = result[first][0] + direction * 0.065
        image_direction = 1.0 if relation in ("leftmost", "second_from_left") else -1.0
        result[second][1] = solve_projected_tie_y(
            result[first], result[second][0], transform, camera,
            image_direction * 0.018 * float(camera["width"]),
        )
    if state == "INSUFFICIENT_EVIDENCE" and "uq_neutral_occluder" in result:
        if source_target not in mapping:
            raise RuntimeError(f"missing insufficient-evidence target mapping: {source_target}")
        target = mapping[source_target]
        dx, gap = solve_occluder_offset(
            result[target], transform, camera, reference_transform, reference_camera
        )
        result["uq_neutral_occluder"][0] = result[target][0] + dx
        result["uq_neutral_occluder"][1] = result[target][1] - gap
        result["uq_neutral_occluder"][2] = 1.5707963267948966
        others = sorted(name for name in mapping.values() if name != target)
        for index, name in enumerate(others):
            result[name][0] = -0.012 + 0.024 * index
            result[name][1] = 0.130 + 0.270 * index
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"append-only output already exists: {OUT}")
    if not LOCK.is_file():
        raise FileNotFoundError(LOCK)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("status") != "FROZEN_BEFORE_CAPTURE_AND_MODEL_INFERENCE":
        raise RuntimeError("anti-shortcut protocol lock is not frozen")
    for path in (SOURCE_SCENES, SOURCE_ANNOTATIONS, SOURCE_GATE, WORLD, CAPTURE_LAUNCH):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_scenes = yaml.safe_load(SOURCE_SCENES.read_text(encoding="utf-8"))
    source_annotations = yaml.safe_load(SOURCE_ANNOTATIONS.read_text(encoding="utf-8"))
    source_gate = yaml.safe_load(SOURCE_GATE.read_text(encoding="utf-8"))
    objects = deepcopy(source_scenes["objects"])
    source_rows = source_scenes["scenes"][:256]
    if len(source_rows) != 256:
        raise RuntimeError("expected 256 source geometry templates")
    historical_layouts, historical_physical = historical_signatures()

    OUT.mkdir(parents=True)
    batches: list[dict] = []
    design_rows: list[dict] = []
    new_layouts: set[str] = set()
    new_physical: set[str] = set()
    rng = random.Random(SEED)
    reference_transform = json.loads(REFERENCE_IE_TF.read_text(encoding="utf-8"))[
        "camera_color_optical_frame"
    ]
    reference_camera = json.loads(REFERENCE_IE_CAMERA.read_text(encoding="utf-8"))

    for batch in range(4):
        transform = json.loads(TF_SOURCES[batch].read_text(encoding="utf-8"))[
            "camera_color_optical_frame"
        ]
        camera = json.loads(CAMERA_SOURCES[batch].read_text(encoding="utf-8"))
        rows = []
        annotations = {}
        cell_rep = Counter()
        source_slice = source_rows[batch * 64:(batch + 1) * 64]
        for source in source_slice:
            source_ann = source_annotations["scenes"][source["scene_id"]]
            state = source_ann["state"]
            relation = source_ann["relation_variant"]
            rep = cell_rep[(state, relation)]
            cell_rep[(state, relation)] += 1
            triplet_index = (2 * batch + rep) % len(TRIPLETS)
            triplet = TRIPLETS[triplet_index]
            mapping = dict(zip(PLACEHOLDERS, triplet))
            opaque = hashlib.sha256(
                f"{PROTOCOL}|{batch}|{source['scene_id']}|{SEED}".encode()
            ).hexdigest()[:16]
            scene_id = f"asv1_{opaque}"
            family_id = f"mh_pcrau_v3/anti_shortcut_val_v1/{opaque}"
            poses = jittered_poses(
                source["poses"], mapping, opaque, state, source_ann.get("target_id"),
                source_ann.get("valid_target_ids"), relation, batch, transform, camera,
                reference_transform, reference_camera,
            )
            layout_signature = full_layout_signature(objects, poses)
            physical = physical_signature(poses)
            if layout_signature in historical_layouts or layout_signature in new_layouts:
                raise RuntimeError(f"layout signature overlap: {scene_id}")
            if physical in historical_physical or physical in new_physical:
                raise RuntimeError(f"physical signature overlap: {scene_id}")
            new_layouts.add(layout_signature)
            new_physical.add(physical)
            template_index = (rep + batch) % 4
            instruction = prompt(relation, triplet, template_index)
            if re.search(r"\b(FOUND|AMBIGUOUS|ABSENT|INSUFFICIENT)\b", instruction, re.I):
                raise RuntimeError("answerability state leaked into prompt")
            ann = map_annotation(source_ann, mapping)
            ann.update({
                "family_id": family_id,
                "layout_id": scene_id,
                "layout_signature_sha256": layout_signature,
                "physical_signature_sha256": physical,
                "split": "anti_shortcut_val_v1",
                "seed": int.from_bytes(hashlib.sha256(opaque.encode()).digest()[:4], "big"),
                "camera_stratum": batch,
                "object_triplet_stratum": triplet_index,
                "prompt_template_index": template_index,
                "template_provenance": "preregistered_geometry_class_only; no image or checkpoint access",
            })
            rows.append({
                "scene_id": scene_id,
                "scene_family_id": family_id,
                "layout_id": scene_id,
                "layout_signature_sha256": layout_signature,
                "physical_signature_sha256": physical,
                "task_type": "horizontal_ordinal_ranking_answerability",
                "instruction": instruction,
                "poses": poses,
            })
            annotations[scene_id] = ann
            design_rows.append({
                "scene_id": scene_id,
                "family_id": family_id,
                "batch": batch,
                "camera_stratum": batch,
                "object_triplet_stratum": triplet_index,
                "prompt_template_index": template_index,
                "state": state,
                "relation": relation,
                "layout_signature_sha256": layout_signature,
                "physical_signature_sha256": physical,
            })
        if len(rows) != 64 or set(cell_rep.values()) != {4} or len(cell_rep) != 16:
            raise RuntimeError(f"batch cell balance failed: {batch}: {cell_rep}")
        rng.shuffle(rows)
        folder = OUT / f"batch_{batch}"
        folder.mkdir()
        pid = f"{PROTOCOL}_b{batch}"
        scene_config = deepcopy(source_scenes)
        scene_config.update({
            "protocol_id": pid,
            "expected_scene_count": 64,
            "random_seed": SEED + batch,
            "view_joint_pose": list(POSES[batch]),
            "scenes": rows,
        })
        annotation_config = deepcopy(source_annotations)
        annotation_config.update({
            "protocol_id": pid,
            "oracle_usage": "evaluation_only_open_after_input_and_prediction_lock",
            "scenes": annotations,
        })
        gate = deepcopy(source_gate)
        gate.update({
            "protocol_id": pid,
            "parent_family_count": 64,
            "split_parent_family_count": {"anti_shortcut_val_v1": 64},
            "state_quota": {"anti_shortcut_val_v1": {state: 16 for state in STATES}},
            "relation_variant_quota": {"anti_shortcut_val_v1": {relation: 16 for relation in RELATIONS}},
            "state_relation_cell_quota": {"anti_shortcut_val_v1": 4},
            "camera": {**source_gate["camera"], "view_joint_pose": list(POSES[batch])},
            "policies": {
                **source_gate["policies"],
                "no_training": True,
                "no_model_inference_before_qc": True,
                "one_shot_evaluation": True,
                "no_materialization_before_qc": True,
            },
        })
        for name, payload in (
            ("scenes.yaml", scene_config),
            ("annotations.yaml", annotation_config),
            ("gate.yaml", gate),
        ):
            (folder / name).write_text(
                yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=140),
                encoding="utf-8",
            )
        batches.append({
            "batch": batch,
            "protocol_id": pid,
            "camera_stratum": batch,
            "view_joint_pose": list(POSES[batch]),
            "scenes": 64,
            "object_triplet_strata": sorted({annotations[row["scene_id"]]["object_triplet_stratum"] for row in rows}),
        })

    # Lock source/config bytes only after the complete design passes static checks.
    design_path = OUT / "DESIGN_INDEX.jsonl"
    design_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in design_rows),
        encoding="utf-8",
    )
    common = required_capture_sources() | {
        str(WORLD.relative_to(ROOT)),
        str(CAPTURE_LAUNCH.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        str(LOCK.relative_to(ROOT)),
    }
    common |= {str(path.relative_to(ROOT)) for path in TF_SOURCES + CAMERA_SOURCES}
    common |= {
        str(REFERENCE_IE_TF.relative_to(ROOT)),
        str(REFERENCE_IE_CAMERA.relative_to(ROOT)),
    }
    for batch in batches:
        folder = OUT / f"batch_{batch['batch']}"
        paths = common | {str((folder / name).relative_to(ROOT)) for name in ("scenes.yaml", "annotations.yaml", "gate.yaml")}
        source_lock = {
            "schema_version": 1,
            "protocol_id": batch["protocol_id"],
            "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
            "locked_at_utc": datetime.now(timezone.utc).isoformat(),
            "workspace_root": str(ROOT),
            "source_artifact_sha256": {name: sha256(ROOT / name) for name in sorted(paths)},
            "model_inventory_sha256": "NO_MODEL_INFERENCE_ANTI_SHORTCUT_CAPTURE",
            "capture_role": "INDEPENDENT_DEVELOPMENT_CHALLENGE",
            "planned_scene_count": 64,
            "view_joint_pose": batch["view_joint_pose"],
            "seals": {"calibration": True, "test_iid": True, "test_ood": True},
        }
        (folder / "CAPTURE_SOURCE_LOCK.json").write_text(
            json.dumps(source_lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    cell_counts = Counter((row["state"], row["relation"]) for row in design_rows)
    state_counts = Counter(row["state"] for row in design_rows)
    relation_counts = Counter(row["relation"] for row in design_rows)
    triplet_counts = Counter(row["object_triplet_stratum"] for row in design_rows)
    summary = {
        "schema_version": "1.0",
        "status": "DESIGN_STATIC_QC_PASS_CAPTURE_NOT_STARTED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL,
        "families": len(design_rows),
        "camera_strata": 4,
        "object_triplet_strata": 8,
        "state_counts": dict(state_counts),
        "relation_counts": dict(relation_counts),
        "triplet_counts": {str(key): value for key, value in sorted(triplet_counts.items())},
        "cell_counts": {f"{key[0]}/{key[1]}": value for key, value in sorted(cell_counts.items())},
        "historical_layout_overlap": 0,
        "historical_physical_overlap": 0,
        "ids_encode_labels": False,
        "sample_order": "seeded permutation inside each camera batch",
        "capture_started": False,
        "model_opened": False,
        "batches": batches,
        "artifacts": {
            "design_index": {"path": str(design_path.relative_to(ROOT)), "sha256": sha256(design_path)},
            "protocol_lock": {"path": str(LOCK.relative_to(ROOT)), "sha256": sha256(LOCK)},
        },
    }
    (OUT / "DESIGN_STATIC_QC.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": summary["status"],
        "families": len(design_rows),
        "cell_min": min(cell_counts.values()),
        "cell_max": max(cell_counts.values()),
        "triplet_counts": summary["triplet_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
