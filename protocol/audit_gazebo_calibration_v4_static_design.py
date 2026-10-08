#!/usr/bin/env python3
"""Freeze a pre-render feasibility audit for the immutable v4 data design."""
from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

import generate_gazebo_calibration_v4_design as generator


ROOT = generator.ROOT
AUDIT = ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v4/STATIC_DESIGN_AUDIT.json"
FAILURE_LOCK = ROOT / "protocol/GAZEBO_CALIBRATION_V4_STATIC_DESIGN_FAILURE_LOCK.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def prior_identifiers() -> tuple[dict[str, set], dict[str, str]]:
    values: dict[str, set] = {name: set() for name in ("scene_id", "family_id", "layout_id", "seed", "layout_signature")}
    source_hashes: dict[str, str] = {}
    for path in sorted((ROOT / "protocol").glob("gazebo_*family_manifest.jsonl")):
        if "test" in path.name.lower() or "calibration_v4" in path.name.lower():
            continue  # Never inspect sealed Test contents.
        source_hashes[str(path.relative_to(ROOT))] = sha256(path)
        for row in jsonl(path):
            for key, field in (("scene_id", "scene_id"), ("family_id", "family_id"),
                               ("layout_id", "layout_id"), ("deterministic_seed", "seed"),
                               ("layout_signature_sha256", "layout_signature")):
                if row.get(key) is not None:
                    values[field].add(row[key])
    for path in sorted((ROOT / "ur3/ur3_perception/config").glob("gazebo*_scenes.yaml")):
        if "test" in path.name.lower() or "calibration_v4" in path.name.lower():
            continue
        source_hashes[str(path.relative_to(ROOT))] = sha256(path)
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for scene in config.get("scenes", []):
            for key, field in (("scene_id", "scene_id"), ("scene_family_id", "family_id"),
                               ("layout_id", "layout_id")):
                if scene.get(key) is not None:
                    values[field].add(scene[key])
            if scene.get("poses"):
                values["layout_signature"].add(generator.v3.layout_signature(scene, config.get("objects", {})))
    return values, source_hashes


def collision_radii() -> dict[str, float]:
    world = ET.parse(generator.WORLDFILE)
    output = {}
    for name in ("ycb_apple", "ycb_orange"):
        node = world.find(f".//model[@name='{name}']/link/collision/geometry/sphere/radius")
        if node is None:
            raise ValueError(f"missing collision sphere in locked world: {name}")
        output[name] = float(node.text)
    return output


def audit() -> dict:
    lock = generator.lock_and_sources()
    plans, preview = generator.generate()
    radii = collision_radii()
    prior, prior_hashes = prior_identifiers()
    all_current = {key: set() for key in prior}
    splits = {}
    seen_scene_ids = set()
    for name, (scenes, annotations, gate, families, analytic_failures) in plans.items():
        identities = {
            "scene_id": {row["scene_id"] for row in families},
            "family_id": {row["family_id"] for row in families},
            "layout_id": {row["layout_id"] for row in families},
            "seed": {row["deterministic_seed"] for row in families},
            "layout_signature": {row["layout_signature_sha256"] for row in families},
        }
        for key, values in identities.items():
            all_current[key].update(values)
        cells = Counter((row["state"], row["relation_variant"]) for row in families)
        collision_witnesses = []
        for scene in scenes["scenes"]:
            sid = scene["scene_id"]
            seen_scene_ids.add(sid)
            if annotations["scenes"][sid]["state"] != "INSUFFICIENT_EVIDENCE":
                continue
            apple = scene["poses"]["ycb_apple"]
            orange = scene["poses"]["ycb_orange"]
            distance_xy = math.hypot(apple[0] - orange[0], apple[1] - orange[1])
            clearance_m = distance_xy - radii["ycb_apple"] - radii["ycb_orange"]
            if clearance_m <= 0:
                collision_witnesses.append({
                    "scene_id": sid,
                    "relation_variant": annotations["scenes"][sid]["relation_variant"],
                    "apple_center_xy_m": apple[:2],
                    "orange_center_xy_m": orange[:2],
                    "center_distance_m": distance_xy,
                    "sum_of_sdf_collision_radii_m": sum(radii.values()),
                    "physical_clearance_m": clearance_m,
                })
        splits[name] = {
            "family_count": len(families),
            "state_relation_cells": {f"{s}|{r}": n for (s, r), n in sorted(cells.items())},
            "unique_identifiers": {key: len(value) == len(families) for key, value in identities.items()},
            "overlaps_with_frozen_prior": {key: len(values & prior[key]) for key, values in identities.items()},
            "analytic_failure_count": len(analytic_failures),
            "analytic_failures": analytic_failures,
            "sdf_collision_sphere_overlaps": collision_witnesses,
        }
    all_paths = [path for name in plans for path in generator.artifact_paths(name)]
    no_generated_files = all(not path.exists() for path in all_paths)
    no_capture_or_dataset = all(not path.exists() for path in (
        ROOT / "datasets/Gazebo_calibration_v4",
        ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v4/capture_attempt_01",
        ROOT / "results/spatial_vlm_refspatial_v1/gazebo_calibration_v4_geometry_pilot/capture_attempt_01",
    ))
    checks = {
        "data_design_lock_sha256_verified": sha256(generator.LOCK) == generator.EXPECTED_LOCK_SHA256,
        "pilot_exact_4x4x2": len(plans[generator.PILOT_ID][3]) == 32 and
                             len(splits[generator.PILOT_ID]["state_relation_cells"]) == 16 and
                             set(splits[generator.PILOT_ID]["state_relation_cells"].values()) == {2},
        "calibration_exact_4x4x8": len(plans[generator.CALIBRATION_ID][3]) == 128 and
                                   len(splits[generator.CALIBRATION_ID]["state_relation_cells"]) == 16 and
                                   set(splits[generator.CALIBRATION_ID]["state_relation_cells"].values()) == {8},
        "all_ids_seeds_signatures_unique_across_pilot_and_calibration": all(
            len(values) == 160 for values in all_current.values()
        ),
        "all_ids_seeds_signatures_disjoint_from_prior": all(
            not (values & prior[key]) for key, values in all_current.items()
        ),
        "preregistered_candidate_footprints_collision_free": all(
            not split["sdf_collision_sphere_overlaps"] for split in splits.values()
        ),
        "other_prerender_checks_pass": all(
            not split["analytic_failures"] for split in splits.values()
        ),
        "no_v4_manifests_scenes_or_gate_configs_written": no_generated_files,
        "no_pilot_or_calibration_capture_or_dataset": no_capture_or_dataset,
        "test_iid_ood_sealed_without_opening_test": bool(lock["exclusion_and_privacy_boundary"]["test_iid_ood_sealed"]),
        "robot_unauthorized": bool(lock["exclusion_and_privacy_boundary"]["robot_unauthorized"]),
    }
    report = {
        "schema_version": 1,
        "protocol_id": generator.CALIBRATION_ID,
        "status": "PASS" if all(checks.values()) else "FAIL_PREREGISTERED_STATIC_GEOMETRY",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "BEFORE_RENDER_BEFORE_IMPLEMENTATION_LOCK_BEFORE_ANY_CAPTURE",
        "classification": "DATA_DESIGN_FAILURE_NOT_CALIBRATOR_SCIENTIFIC_NEGATIVE",
        "checks": checks,
        "collision_sphere_radii_from_locked_world_m": radii,
        "splits": splits,
        "prior_registry_counts": {key: len(values) for key, values in prior.items()},
        "prior_registry_source_sha256": prior_hashes,
        "fixed_camera_tf_snapshot_sha256": preview["camera_tf_snapshot_sha256"],
        "locked_world_sha256": preview["world_sha256"],
        "v4_data_design_lock_sha256": sha256(generator.LOCK),
        "generator_code_sha256": sha256(Path(generator.__file__)),
        "audit_code_sha256": sha256(Path(__file__)),
        "pilot_rendered_or_captured": False,
        "calibration_rendered_or_captured": False,
        "model_inference_run": False,
        "calibrator_fit_call_count": 0,
        "scientific_decision_available": False,
        "test_iid_ood_access": False,
        "robot_access": False,
    }
    return report


def main() -> None:
    if AUDIT.exists() or FAILURE_LOCK.exists():
        raise FileExistsError("refusing to rerun or overwrite frozen static-design audit")
    report = audit()
    write_new(AUDIT, report)
    if report["status"] != "PASS":
        sources = (generator.LOCK, Path(generator.__file__), Path(__file__),
                   generator.WORLDFILE, generator.TF_SNAPSHOT, AUDIT)
        payload = {
            "schema_version": 1,
            "protocol_id": generator.CALIBRATION_ID,
            "status": "STATIC_DESIGN_FAILURE_FROZEN_NO_PILOT_OR_CALIBRATION_CAPTURE",
            "classification": report["classification"],
            "locked_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_artifact_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
            "failed_checks": [name for name, passed in report["checks"].items() if not passed],
            "calibration_decision": None,
            "new_append_only_data_design_revision_required": True,
            "test_iid_ood_access": False,
            "robot_access": False,
        }
        write_new(FAILURE_LOCK, payload)
    print(json.dumps({"status": report["status"], "checks": report["checks"],
                      "pilot_sphere_collisions": len(report["splits"][generator.PILOT_ID]["sdf_collision_sphere_overlaps"]),
                      "calibration_sphere_collisions": len(report["splits"][generator.CALIBRATION_ID]["sdf_collision_sphere_overlaps"]),
                      "audit_path": str(AUDIT), "audit_sha256": sha256(AUDIT),
                      "failure_lock_sha256": sha256(FAILURE_LOCK) if FAILURE_LOCK.exists() else None},
                     indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
