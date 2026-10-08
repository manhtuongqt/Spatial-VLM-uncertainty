"""Audit historical Gazebo identity/layout/RGB overlap without opening v3 Test.

Full-frame near-duplicate rule is the inherited gray MAD < .05 and changed
fraction < .002 (difference > 3). Area-mean thumbnails only prune pairs:
by Jensen's inequality a full-frame MAD < .05 must have thumbnail MAD < .05,
apart from bounded decoding/rounding noise; .06 is used conservatively.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from .audit_development import ROOT, sha256


OUT = ROOT / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03/HISTORICAL_EXCLUSION_AUDIT_V4.json"
REGISTRY = ROOT / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03/HISTORICAL_EXCLUSION_REGISTRY_V4.jsonl"


def scan() -> dict:
    captures = sorted((ROOT / "results/spatial_vlm_refspatial_v1").glob(
        "gazebo*/capture_attempt_*/capture_manifest.json"))
    captures += sorted((ROOT / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03").glob(
        "gazebo*/batch_*/capture_attempt_*/capture_manifest.json"))
    captures += sorted((ROOT / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03").glob(
        "gazebo_canary*/capture_attempt_*/capture_manifest.json"))
    captures += sorted((ROOT / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03").glob(
        "gazebo_ie_repair*/capture_attempt_*/capture_manifest.json"))
    configs = list((ROOT / "ur3/ur3_perception/config").glob("*scenes.yaml"))
    configs += list((ROOT / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03").glob(
        "gazebo*/**/scenes.yaml"))
    config_by_hash = {sha256(path): path for path in configs}
    annotation_paths = list((ROOT / "ur3/ur3_perception/config").glob("*annotations.yaml"))
    annotation_paths += list((ROOT / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03").glob(
        "gazebo*/**/annotations.yaml"))
    annotation_by_hash = {sha256(path): path for path in annotation_paths}
    rows = []
    unmatched = []
    missing_config = []
    for manifest_path in captures:
        manifest = json.loads(manifest_path.read_text())
        root = manifest_path.parent
        input_path = root / "input_manifest.jsonl"
        config_path = config_by_hash.get(manifest.get("scene_config_sha256"))
        if not input_path.is_file():
            unmatched.append(str(manifest_path.relative_to(ROOT)))
            continue
        if config_path is None:
            missing_config.append(str(manifest_path.relative_to(ROOT)))
        config = yaml.safe_load(config_path.read_text()) if config_path else {"scenes": []}
        by_scene = {row["scene_id"]: row for row in config["scenes"]}
        annotation_path = annotation_by_hash.get(manifest.get("annotation_file_sha256"))
        annotation = yaml.safe_load(annotation_path.read_text()) if annotation_path else {"scenes": {}}
        for line in input_path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            scene = by_scene.get(item["scene_id"])
            label = annotation.get("scenes", {}).get(item["scene_id"], {})
            rgb = root / item["input_files"]["rgb"]
            if not rgb.is_file():
                unmatched.append(str(rgb.relative_to(ROOT)))
                continue
            role = "HISTORICAL"
            if "gazebo_pilot_512_revision_v2" in str(manifest_path):
                role = "CURRENT_REVISION_V2"
            elif "gazebo_ie_repair_revision_v2" in str(manifest_path):
                role = "CURRENT_REPAIR_V2"
            elif "gazebo_pilot_512_v1" in str(manifest_path):
                role = "REJECTED_V1_ATTEMPT"
            elif "gazebo_canary" in str(manifest_path):
                role = "DEVELOPMENT_CANARY"
            image = cv2.imread(str(rgb), cv2.IMREAD_GRAYSCALE)
            if image is None:
                unmatched.append(str(rgb.relative_to(ROOT)))
                continue
            oracle_path = root / item["scene_id"] / "evaluator/capture_oracle.json"
            physical_signature = None
            if oracle_path.is_file():
                layout = json.loads(oracle_path.read_text()).get("requested_scene_layout_base_link")
                if layout is not None:
                    physical_signature = hashlib.sha256(json.dumps(
                        layout, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            effective_family = scene.get("scene_family_id") if scene else None
            if role == "DEVELOPMENT_CANARY":
                source_indices = (0, 1, 4, 5, 8, 9, 12, 13)
                local_index = int(item["scene_id"].rsplit("_", 1)[1])
                effective_family = ("spatial_vlm_uq_v2_full_r3/train_uq/"
                                    f"parent_{source_indices[local_index]:03d}")
            thumb = cv2.resize(image, (16, 12), interpolation=cv2.INTER_AREA).astype(np.float32)
            rows.append({"role": role, "protocol_id": item.get("protocol_id"),
                         "capture_manifest_path": str(manifest_path.relative_to(ROOT)),
                         "scene_id": item["scene_id"],
                         "family_id": scene.get("scene_family_id") if scene else None,
                         "effective_family_id": effective_family,
                         "layout_id": scene.get("layout_id") if scene else None,
                         "signature": scene.get("layout_signature_sha256") if scene else None,
                         "physical_signature": physical_signature,
                         "seed": label.get("seed"),
                         "rgb_path": str(rgb.relative_to(ROOT)),
                         "rgb_sha256": sha256(rgb), "thumb": thumb,
                         "image_shape": image.shape})
    return {"rows": rows, "unmatched": unmatched, "missing_config": missing_config,
            "capture_manifest_count": len(captures)}


def main() -> None:
    if OUT.exists() or REGISTRY.exists():
        raise FileExistsError("Audit and exclusion registry are append-only")
    data = scan()
    rows = data["rows"]
    fields = ("scene_id", "effective_family_id", "layout_id", "signature",
              "physical_signature", "rgb_sha256")
    collisions = {}
    for field in fields:
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            if row.get(field):
                groups[row[field]].append(index)
        cross = []
        for value, indices in groups.items():
            if len(indices) <= 1 or not any(rows[i]["role"].startswith("CURRENT_") for i in indices):
                continue
            cross.append({"value": value, "members": [
                {"role": rows[i]["role"], "scene_id": rows[i]["scene_id"],
                 "family_id": rows[i]["effective_family_id"]} for i in indices]})
        collisions[field] = cross
    # Conservative thumbnail prerequisite for the locked full-frame rule.
    current = [i for i, row in enumerate(rows) if row["role"].startswith("CURRENT_")]
    prior = [i for i, row in enumerate(rows) if not row["role"].startswith("CURRENT_")]
    near = []
    checked = 0
    for i in current:
        a = rows[i]
        for j in prior:
            b = rows[j]
            checked += 1
            if a["image_shape"] != b["image_shape"]:
                continue
            if float(np.mean(np.abs(a["thumb"] - b["thumb"]))) >= 0.06:
                continue
            image_a = cv2.imread(str(ROOT / a["rgb_path"]), cv2.IMREAD_GRAYSCALE)
            image_b = cv2.imread(str(ROOT / b["rgb_path"]), cv2.IMREAD_GRAYSCALE)
            diff = cv2.absdiff(image_a, image_b)
            mad = float(np.mean(diff))
            changed = float(np.mean(diff > 3))
            if mad < 0.05 and changed < 0.002:
                near.append({"current": a["scene_id"], "prior": b["scene_id"],
                             "prior_role": b["role"],
                             "same_family": a["effective_family_id"] == b["effective_family_id"],
                             "gray_mad": mad, "changed_pixel_fraction": changed})
    role_counts = Counter(row["role"] for row in rows)
    result = {
        "schema_version": "1.0", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Existing on-disk Gazebo capture_attempt manifests under results and Day 3; no v3 sealed Test opened",
        "capture_manifests_found": data["capture_manifest_count"],
        "rgb_images_read": len(rows), "role_counts": dict(role_counts),
        "unmatched_sources": data["unmatched"],
        "historical_capture_manifests_without_recoverable_scene_config": data["missing_config"],
        "current_vs_prior_pairs_coarse_screened": checked,
        "near_duplicate_rule": {"gray_mad_lt": 0.05, "changed_fraction_lt": 0.002,
                                "changed_pixel_absdiff_gt": 3, "thumbnail_prune_mad_ge": 0.06},
        "near_duplicate_pairs": near, "collisions_with_current": collisions,
        "interpretation": "Same-family old/new view is co-located, not an independent family. "
                          "Any cross-family exact/near/layout collision or unmatched historical source prevents G1 PASS. "
                          "Near-duplicate absence under this legacy pixel rule does not prove semantic independence.",
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    fields = ("role", "protocol_id", "capture_manifest_path", "scene_id", "family_id",
              "effective_family_id", "layout_id", "signature", "physical_signature",
              "seed", "rgb_path", "rgb_sha256")
    REGISTRY.write_text("".join(json.dumps({key: row.get(key) for key in fields}) + "\n"
                                for row in rows))
    print(json.dumps({"rgb_images_read": len(rows), "current": len(current),
                      "near_pairs": len(near), "unmatched": len(data["unmatched"]),
                      "missing_config": len(data["missing_config"]),
                      "collision_counts": {key: len(value) for key, value in collisions.items()}}))


if __name__ == "__main__":
    main()
