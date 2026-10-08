#!/usr/bin/env python3
"""Preflight and seal the Test-IID capture design."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_test_iid import PROTOCOL_ID, TARGET_QUOTAS, build  # noqa: E402


PROTOCOL = ROOT / "new" / "test_iid" / "protocol"
MANIFEST_PATH = PROTOCOL / "test_iid_manifest.json"
PLAN_PATH = PROTOCOL / "test_iid_capture_plan.json"
FREEZE_PATH = ROOT / "new" / "test_iid" / "contracts" / "best_v2_freeze_lock.json"
REPORT_PATH = PROTOCOL / "preflight_report.json"
LOCK_PATH = PROTOCOL / "execution_lock.json"


def sha(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(value: Any, keys: set[str]) -> set[Any]:
    output: set[Any] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys and isinstance(child, (str, int)):
                output.add(child)
            output.update(collect(child, keys))
    elif isinstance(value, list):
        for child in value:
            output.update(collect(child, keys))
    return output


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    manifest, plan = load(MANIFEST_PATH), load(PLAN_PATH)
    freeze = load(FREEZE_PATH)
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, **details: Any) -> None:
        checks[name] = {"passed": bool(passed), **details}

    replay_manifest, replay_plan = build()
    replay_plan["manifest_sha256"] = sha(MANIFEST_PATH)
    check("deterministic_replay", replay_manifest == manifest and replay_plan == plan)

    families = manifest["families"]
    family_ids = [row["family_id"] for row in families]
    capture_ids = [row["capture_id"] for row in plan["captures"]]
    sample_ids = [f"{row['family_id']}__{variant}" for row in families for variant in row["variants"]]
    check(
        "population_and_identity",
        len(families) == len(set(family_ids)) == 200
        and len(capture_ids) == len(set(capture_ids)) == 400
        and len(sample_ids) == len(set(sample_ids)) == 1000
        and all(value.startswith("v211iid_family_") for value in family_ids),
    )

    target_counts = Counter(row["target_object_group"] for row in families)
    states = Counter(row["primary_answerability_stratum"] for row in families)
    categories = Counter(row["family_category"] for row in families)
    cube_anchors = sum(
        value.startswith("cube_")
        for row in families for value in row["variant_specs"][0]["anchor_ids"]
    )
    boxes_relation_safe = all(
        row["family_category"] == "direct_grounding"
        for row in families if row["target_object_group"] == "box"
    )
    check(
        "coverage_and_object_priority",
        dict(target_counts) == TARGET_QUOTAS
        and states == {"FOUND": 80, "INSUFFICIENT_EVIDENCE": 50, "AMBIGUOUS": 35, "ABSENT": 35}
        and categories == {
            "direct_grounding": 30, "front_behind_camera": 30,
            "multi_anchor_depth_order": 30, "nearer_farther": 35,
            "occlusion_depth_evidence": 35, "relation_2d": 40,
        }
        and cube_anchors == 40 and boxes_relation_safe,
        target_groups=dict(target_counts), states=dict(states), categories=dict(categories),
        primary_cube_anchor_count=cube_anchors,
    )

    prior_paths = [
        ROOT / "old" / "protocol" / "dataset_v2_1_development_manifest.json",
        ROOT / "old" / "protocol" / "dataset_v2_1_development_capture_plan.json",
        ROOT / "old" / "protocol" / "dataset_v2_1_calibration_manifest.json",
        ROOT / "old" / "protocol" / "dataset_v2_1_calibration_capture_plan.json",
    ]
    priors = [load(path) for path in prior_paths]
    overlaps = {}
    for label, keys in {
        "family_id": {"family_id"},
        "capture_id": {"capture_id"},
        "instruction": {"instruction"},
        "layout": {"layout_instance_fingerprint_sha256", "layout_fingerprint_sha256"},
    }.items():
        current = collect({"manifest": manifest, "plan": plan}, keys)
        historical = set().union(*(collect(value, keys) for value in priors))
        overlaps[label] = len(current & historical)
    check("historical_disjointness", not any(overlaps.values()), overlaps=overlaps)

    model_path = ROOT / freeze["selected_checkpoint"]["path"]
    metadata_path = model_path.parent / "metadata.json"
    calibrator_path = ROOT / freeze["frozen_calibrator"]["path"]
    hashes_ok = (
        sha(model_path) == freeze["selected_checkpoint"]["model_sha256"]
        and sha(metadata_path) == freeze["selected_checkpoint"]["metadata_sha256"]
        and sha(calibrator_path) == freeze["frozen_calibrator"]["sha256"]
    )
    check("best_v2_and_calibrator_frozen", hashes_ok)
    check(
        "iid_and_sealed_boundary",
        manifest["counts"]["ood_axis"] == {"none": 200}
        and manifest["eligible_for_test"] is True
        and manifest["test_opened"] is False
        and plan["training_authorized"] is False
        and plan["model_inference_authorized"] is False
        and plan["test_authorized"] is False,
    )

    passed = all(row["passed"] for row in checks.values())
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "passed": passed,
        "decision": "GO_TEST_IID_CAPTURE_CANARY" if passed else "FIX_PREFLIGHT",
        "checks": checks,
        "manifest_sha256": sha(MANIFEST_PATH),
        "capture_plan_sha256": sha(PLAN_PATH),
        "freeze_lock_sha256": sha(FREEZE_PATH),
        "test_inference_performed": False,
    }
    dump(REPORT_PATH, report)
    if passed:
        locked_paths = [
            "new/test_iid/contracts/best_v2_freeze_lock.json",
            "new/test_iid/scripts/generate_test_iid.py",
            "new/test_iid/scripts/capture.py",
            "new/test_iid/scripts/qc_batch.py",
            "new/test_iid/protocol/test_iid_manifest.json",
            "new/test_iid/protocol/test_iid_capture_plan.json",
        ]
        lock = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "CAPTURE_AUTHORIZED_MODEL_INFERENCE_STILL_SEALED",
            "decision": "GO_TEST_IID_CAPTURE_CANARY",
            "authorized_next_batch": "canary_000",
            "capture_authorized": True,
            "test_inference_authorized": False,
            "training_authorized": False,
            "capture_output_root": "new/test_iid/dataset",
            "manifest_sha256": sha(MANIFEST_PATH),
            "capture_plan_sha256": sha(PLAN_PATH),
            "preflight_report_sha256": sha(REPORT_PATH),
            "freeze_lock_sha256": sha(FREEZE_PATH),
            "checkpoint_model_sha256": freeze["selected_checkpoint"]["model_sha256"],
            "calibrator_sha256": freeze["frozen_calibrator"]["sha256"],
            "required_family_count": 200,
            "required_capture_count": 400,
            "required_sample_count": 1000,
            "locked_artifact_sha256": {relative: sha(ROOT / relative) for relative in locked_paths},
        }
        dump(LOCK_PATH, lock)
    print(json.dumps({"passed": passed, "decision": report["decision"], "checks": checks}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
