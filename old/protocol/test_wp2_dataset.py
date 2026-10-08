"""Software tests for WP2 schemas, family splits, replay and leakage isolation."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import jsonschema
import numpy as np

PROTOCOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROTOCOL_DIR))

from wp2_common import find_forbidden_inference_keys, sha256_file, write_json
from wp2_family_generator import OBJECTS, build_manifest, validate_internal
from wp2_leakage_validator import validate_leakage
from wp2_materialize import corrupt_depth, materialize, relative_depth
from wp2_replay_validator import replay_validate


def test_family_manifest_has_exact_locked_scale_and_family_split():
    manifest, capture_plan = build_manifest()
    validate_internal(manifest, capture_plan)
    schema = json.loads((PROTOCOL_DIR / "family_manifest_v1.schema.json").read_text())
    jsonschema.validate(manifest, schema)
    assert manifest["family_count"] == 50
    assert manifest["sample_count"] == 250
    assert manifest["depth_dependent_family_count"] == 30
    assert Counter(item["split"] for item in manifest["families"]) == {
        "train": 30, "dev": 8, "calibration": 6, "test": 6,
    }
    assert capture_plan["capture_count"] == 100


def test_inference_key_filter_catches_oracle_aliases():
    assert find_forbidden_inference_keys({"prompt": "safe", "rgb": "x"}) == []
    findings = find_forbidden_inference_keys({
        "prompt": "x", "Target-Mask": "bad", "nested": {"expected_intervention": "bad"}
    })
    assert "$.Target-Mask" in findings
    assert "$.nested.expected_intervention" in findings


def test_depth_corruptions_are_deterministic_and_non_identity():
    y, x = np.mgrid[:48, :64]
    depth = (0.4 + 0.003 * x + 0.002 * y).astype(np.float32)
    clean = relative_depth(depth)
    for kind, severity in (
        ("bias_noise", 1), ("localized_holes_edges", 2), ("inversion_shift", 3)
    ):
        first, _ = corrupt_depth(clean, 123, {"kind": kind, "severity": severity})
        second, _ = corrupt_depth(clean, 123, {"kind": kind, "severity": severity})
        assert np.array_equal(first, second)
        assert not np.array_equal(first, clean)


def make_fixture_dataset(root: Path) -> None:
    manifest, capture_plan = build_manifest()
    write_json(root / "family_manifest.json", manifest)
    write_json(root / "capture_plan.json", capture_plan)
    write_json(root / "object_registry.json", OBJECTS)
    smoke_ids = {"wp2_family_0001", "wp2_family_0031", "wp2_family_0041"}
    capture_ids = [
        item["capture_id"] for item in capture_plan["captures"]
        if item["family_id"] in smoke_ids
    ]
    write_json(root / "smoke_selection.json", {
        "schema_version": 1, "protocol_id": manifest["protocol_id"],
        "family_ids": sorted(smoke_ids), "capture_ids": capture_ids, "sample_count": 15,
    })
    selected = {item["capture_id"]: item for item in capture_plan["captures"] if item["capture_id"] in capture_ids}
    inventory = []
    for capture_id, capture in selected.items():
        directory = root / "raw" / "captures" / capture_id
        directory.mkdir(parents=True)
        height, width = 96, 128
        y, x = np.mgrid[:height, :width]
        rgb = np.dstack(((x * 2) % 255, (y * 3) % 255, ((x + y) * 2) % 255)).astype(np.uint8)
        depth = (0.35 + x * 0.002 + y * 0.001).astype(np.float32)
        labels = np.full((height, width), 40, dtype=np.uint8)
        visible_labels = [1, 2, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        for idx, label in enumerate(visible_labels):
            row, column = divmod(idx, 4)
            labels[5 + row*28:27 + row*28, 5 + column*30:27 + column*30] = label
        cv2.imwrite(str(directory / "rgb_original.png"), rgb)
        np.save(directory / "depth_metric.npy", depth, allow_pickle=False)
        cv2.imwrite(str(directory / "semantic_instance_labels.png"), labels)
        write_json(directory / "camera_info.json", {"width": width, "height": height})
        write_json(directory / "tf_snapshot.json", {"camera_color_optical_frame": {"position": [0, 0, 1]}})
        names = [
            "rgb_original.png", "depth_metric.npy", "semantic_instance_labels.png",
            "camera_info.json", "tf_snapshot.json",
        ]
        hashes = {name: sha256_file(directory / name) for name in names}
        meta = {
            "schema_version": 1, "protocol_id": manifest["protocol_id"],
            "capture_id": capture_id, "family_id": capture["family_id"],
            "condition": capture["condition"], "seed": capture["seed"],
            "capture_backend": "Gazebo Fortress RGB-D + semantic segmentation",
            "requested_layout_base_link": capture["layout"],
            "capture_timestamps": {"rgb_sec": 1.0, "depth_sec": 1.0, "labels_sec": 1.0, "max_spread_sec": 0.0},
            "sensor_qc": {"passed": True}, "artifact_sha256": hashes,
        }
        write_json(directory / "capture_meta.json", meta)
        inventory.append({"capture_id": capture_id, "family_id": capture["family_id"]})
    write_json(root / "raw" / "raw_capture_manifest.json", {
        "schema_version": 1, "protocol_id": manifest["protocol_id"],
        "capture_count": len(inventory), "captures": inventory,
    })


def test_smoke_materialize_schema_leakage_and_replay(tmp_path: Path):
    make_fixture_dataset(tmp_path)
    records = materialize(tmp_path, tmp_path, "smoke")
    assert len(records) == 15
    schema = json.loads((PROTOCOL_DIR / "dataset_v1.schema.json").read_text())
    for record in records:
        jsonschema.validate(record, schema)
        assert find_forbidden_inference_keys(record["inference_payload"]) == []
        assert record["provenance"]["source_dataset_ids"] == []
    leakage = validate_leakage(tmp_path, "smoke")
    assert leakage["passed"]
    replay = replay_validate(tmp_path, "smoke")
    assert replay["passed"], replay["mismatches"]


def test_family_variants_never_cross_split():
    manifest, _ = build_manifest()
    for family in manifest["families"]:
        assert len({family["split"]}) == 1
        assert {item["variant"] for item in family["variant_specs"]} == {
            "clean", "semantic_counterfactual", "relation_counterfactual",
            "depth_corruption", "occlusion_view_counterfactual",
        }
