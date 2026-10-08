"""Generate four fresh 64-scene Gazebo pilot batches (256 new + 256 existing).

No image is claimed captured until a batch capture manifest is COMPLETE and
sensor/geometry QC passes. The old train split remains untouched.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import yaml

from .audit_development import GAZEBO, ROOT, sha256


PROTOCOL = "mh_pcrau_v3_gazebo_pilot_512"
OUT = ROOT / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03/gazebo_pilot_512_v1"
CONFIG = ROOT / "ur3/ur3_perception/config"
BASE_POSE = [1.3815, -1.8273, 1.3428, -1.2863, -1.5708, -1.9601]
POSES = [
    [1.4315, -1.8273, 1.3428, -1.2863, -1.5708, -1.9601],
    [1.3315, -1.8273, 1.3428, -1.2863, -1.5708, -1.9601],
    [1.3815, -1.7773, 1.3428, -1.2863, -1.5708, -1.9601],
    [1.3815, -1.8773, 1.3428, -1.2863, -1.5708, -1.9601],
]
GROUPS = [
    ({"ycb_apple": "ycb_apple", "ycb_orange": "ycb_orange", "mango": "mango"},
     {"ycb_apple": 26, "ycb_orange": 27, "mango": 29},
     {"ycb_apple": "red apple", "ycb_orange": "orange", "mango": "yellow mango"}),
    ({"ycb_apple": "red_cube", "ycb_orange": "blue_cube", "mango": "green_cube"},
     {"red_cube": 1, "blue_cube": 2, "green_cube": 3},
     {"ycb_apple": "red cube", "ycb_orange": "blue cube", "mango": "green cube"}),
    ({"ycb_apple": "ycb_cracker_box", "ycb_orange": "ycb_sugar_box", "mango": "ycb_tomato_soup_can"},
     {"ycb_cracker_box": 21, "ycb_sugar_box": 22, "ycb_tomato_soup_can": 23},
     {"ycb_apple": "cracker box", "ycb_orange": "sugar box", "mango": "tomato soup can"}),
    ({"ycb_apple": "ycb_mustard_bottle", "ycb_orange": "ycb_banana", "mango": "ycb_power_drill"},
     {"ycb_mustard_bottle": 24, "ycb_banana": 28, "ycb_power_drill": 30},
     {"ycb_apple": "mustard bottle", "ycb_orange": "banana", "mango": "power drill"}),
]


def map_prompt(text: str, names: dict[str, str]) -> str:
    # Replace longer noun phrases first to avoid transforming the word orange
    # inside a generated name. Existing coordinate suffix is unchanged.
    replacements = {
        "red apple": names["ycb_apple"],
        "yellow mango": names["mango"],
        "orange": names["ycb_orange"],
    }
    pattern = re.compile(r"red apple|yellow mango|\borange\b", re.IGNORECASE)
    return pattern.sub(lambda match: replacements[match.group(0).lower()], text)


def map_annotation(value: dict, mapping: dict[str, str], labels: dict[str, int]) -> dict:
    result = deepcopy(value)
    for field in ("target_id", "target_category"):
        if result.get(field) in mapping:
            result[field] = mapping[result[field]]
    for field in ("candidate_ids", "context_ids", "valid_target_ids"):
        if field in result:
            result[field] = [mapping.get(name, name) for name in result[field]]
    for ids_field, label_field in (("candidate_ids", "candidate_labels"),
                                   ("context_ids", "context_labels"),
                                   ("valid_target_ids", "valid_target_labels")):
        if ids_field in result:
            result[label_field] = [labels[name] for name in result[ids_field]]
    if result.get("target_id") is not None:
        result["target_label"] = labels[result["target_id"]]
    result["failure_tags"] = ["object_identity" if tag == "fruit_identity" else tag
                              for tag in result.get("failure_tags", [])]
    return result


def generate() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    sys.path.insert(0, str(ROOT / "protocol"))
    import generate_gazebo_train_uq_v2_pilot_r5_contract as geometry

    original_scenes = yaml.safe_load((CONFIG / "gazebo_train_uq_v2_full_r3_scenes.yaml").read_text())
    original_oracle = yaml.safe_load((CONFIG / "gazebo_train_uq_v2_full_r3_annotations.yaml").read_text())
    original_gate = yaml.safe_load((CONFIG / "gazebo_train_uq_v2_full_r3_gate.yaml").read_text())
    source_rows = original_scenes["scenes"][:256]
    assert len(source_rows) == 256
    OUT.mkdir(parents=True)
    inventory = []
    all_families = set()
    all_signatures = set()
    for batch in range(4):
        mapping, labels, names = GROUPS[batch]
        rows = []
        oracle = {}
        counts = Counter()
        for local, old in enumerate(source_rows[batch * 64:(batch + 1) * 64]):
            old_annotation = original_oracle["scenes"][old["scene_id"]]
            sid = f"v3_pilot_512_new_b{batch}_{local:03d}"
            family = f"mh_pcrau_v3/pilot_512/new/parent_{batch * 64 + local:03d}"
            poses = {mapping.get(name, name): deepcopy(pose) for name, pose in old["poses"].items()}
            # Small preregistered shift ensures no reuse of the old full pose.
            for index, (name, pose) in enumerate(sorted(poses.items())):
                pose[0] += 0.002 * (batch + 1)
                pose[2] += 0.003 * (index + 1)
            signature = geometry.layout_signature(poses)
            if signature in all_signatures or family in all_families:
                raise ValueError("Duplicate new layout/family")
            all_signatures.add(signature)
            all_families.add(family)
            ann = map_annotation(old_annotation, mapping, labels)
            ann.update({
                "family_id": family, "layout_id": sid,
                "layout_signature_sha256": signature, "split": "pilot_development_only",
                "seed": int.from_bytes(hashlib.sha256(sid.encode()).digest()[:4], "big"),
                "template_provenance": "new capture based on geometry class, not old image reuse",
                "view_id": f"view_{batch}", "object_group": batch,
            })
            prompt = map_prompt(old["instruction"], names)
            rows.append({
                "scene_id": sid, "scene_family_id": family, "layout_id": sid,
                "layout_signature_sha256": signature,
                "task_type": old["task_type"], "instruction": prompt, "poses": poses,
            })
            oracle[sid] = ann
            counts[(ann["state"], ann["relation_variant"])] += 1
        assert len(rows) == 64 and len(counts) == 16 and set(counts.values()) == {4}, counts
        pid = f"{PROTOCOL}_b{batch}"
        scene_config = deepcopy(original_scenes)
        scene_config.update({
            "protocol_id": pid, "expected_scene_count": 64,
            "random_seed": 24092026 + batch, "view_joint_pose": POSES[batch],
            "scenes": rows,
        })
        annotation = deepcopy(original_oracle)
        annotation.update({"protocol_id": pid, "oracle_usage": "pilot_geometry_qc_only", "scenes": oracle})
        gate = deepcopy(original_gate)
        gate.update({
            "protocol_id": pid, "parent_family_count": 64,
            "split_parent_family_count": {"pilot_development_only": 64},
            "state_quota": {"pilot_development_only": {name: 16 for name in geometry.STATES}},
            "relation_variant_quota": {"pilot_development_only": {name: 16 for name in geometry.RELATIONS}},
            "state_relation_cell_quota": {"pilot_development_only": 4},
            "camera": {**original_gate["camera"], "view_joint_pose": POSES[batch]},
            "policies": {**original_gate["policies"], "no_training": True,
                         "no_model_inference": True, "no_materialization": True},
        })
        folder = OUT / f"batch_{batch}"
        folder.mkdir()
        for name, payload in (("scenes.yaml", scene_config), ("annotations.yaml", annotation),
                              ("gate.yaml", gate)):
            (folder / name).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=140))
        inventory.append({
            "batch": batch, "protocol_id": pid, "planned_scenes": 64,
            "view_joint_pose": POSES[batch], "object_names": sorted(set(mapping.values())),
            "state_relation_cells": {f"{state}/{relation}": count
                                     for (state, relation), count in sorted(counts.items())},
            "configs": {name: {"path": str((folder / name).relative_to(ROOT)),
                               "sha256": sha256(folder / name)}
                        for name in ("scenes.yaml", "annotations.yaml", "gate.yaml")},
        })
    plan = {
        "status": "CAPTURE_DESIGN_ONLY", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL, "existing_gazebo_train_images": 256,
        "new_capture_target": 256, "pilot_image_target": 512,
        "new_parent_family_target": 256, "new_views": 4,
        "seals": {"calibration": "SEALED", "test_iid": "SEALED", "test_ood": "SEALED"},
        "old_dataset_manifest_sha256": sha256(GAZEBO / "manifest.json"),
        "batches": inventory,
        "note": "A planned state is not a verified label. All new images require capture and QC.",
    }
    (OUT / "CAPTURE_PLAN.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": plan["status"], "batches": 4, "new_target": 256}))


if __name__ == "__main__":
    generate()
