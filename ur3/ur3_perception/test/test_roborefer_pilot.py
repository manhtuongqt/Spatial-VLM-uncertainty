"""CPU-only contract tests for the locked RoboRefer pilot protocol.

These tests deliberately use synthetic arrays and temporary manifests.  They
must never start Gazebo, load the VLM, contact the RoboRefer HTTP service, or
publish a robot target.
"""

import importlib.util
import base64
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
SCRIPTS = PACKAGE / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load_script(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


POINT_UTILS = _load_script("pilot_test_spatial_point_utils", "spatial_point_utils.py")
GROUNDER = _load_script("pilot_test_roborefer_grounder", "roborefer_grounder.py")
RUNNER = _load_script("pilot_test_roborefer_pilot_runner", "roborefer_pilot_runner.py")
VALIDATOR = _load_script(
    "pilot_test_roborefer_pilot_validate", "roborefer_pilot_validate.py"
)
EVALUATOR = _load_script(
    "pilot_test_roborefer_pilot_evaluator", "roborefer_pilot_evaluator.py"
)
SYNTHETIC_MODEL_SHA256 = "a" * 64


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_point_parsing_and_normalized_coordinate_conversion_are_explicit():
    answer = "[(0.0000, 1.0000), (0.5000, 0.5000), (1.2, -0.1)]"

    points = POINT_UTILS.parse_points(answer)
    pixels = POINT_UTILS.points_to_pixels(points, width=640, height=480)

    # Parsing does not silently clamp a model's out-of-range answer.  The
    # conversion step rejects it and maps only valid normalized coordinates.
    assert points == [(0.0, 1.0), (0.5, 0.5), (1.2, -0.1)]
    assert pixels == [(0, 479), (320, 240)]


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("coordinates: [(0.25, .75)]", [(0.25, 0.75)]),
        ("[(nan, 0.2)]", []),
        ("[(inf, 0.2)]", []),
        ("no point", []),
    ],
)
def test_point_parser_never_invents_missing_or_nonfinite_points(answer, expected):
    assert POINT_UTILS.parse_points(answer) == expected


def _compact_object_depth():
    height, width = 96, 144
    depth = np.full((height, width), 0.82, dtype=np.float32)
    yy, xx = np.ogrid[:height, :width]
    target = (xx - 48) ** 2 + (yy - 46) ** 2 <= 16 ** 2
    distractor = (xx - 111) ** 2 + (yy - 55) ** 2 <= 12 ** 2
    depth[target] = 0.61
    depth[distractor] = 0.615
    return depth, target, distractor


def test_frozen_depth_component_gate_selects_only_the_seeded_surface():
    depth, target, distractor = _compact_object_depth()

    result = GROUNDER.segment_seeded_depth_component(
        depth,
        [(48, 46)],
        seed_radius_px=4,
        roi_radius_px=80,
        near_tolerance_m=0.015,
        far_tolerance_m=0.055,
        min_area_px=100,
        max_area_fraction=0.20,
        bbox_padding_px=4,
    )

    assert result["mask"][46, 48] == 255
    assert np.count_nonzero(result["mask"][target]) > 700
    assert np.count_nonzero(result["mask"][distractor]) == 0
    assert result["supported_points_xy"] == [(48, 46)]
    assert result["grasp_pixel_xy"] == (48, 46)


def test_frozen_depth_component_gate_fails_closed_for_invalid_seed():
    depth, _, _ = _compact_object_depth()
    with pytest.raises(ValueError, match="no RoboRefer seed point"):
        GROUNDER.segment_seeded_depth_component(depth, [(-1, 46)])


@pytest.mark.parametrize(
    "answer, expected_status, expected_pixels",
    [
        ("[(0.5, 0.5)]", "EXACT_ONE_NORMALIZED_POINT", [(320, 240)]),
        ("point: [(0.5, 0.5)]", "FORMAT_VIOLATION", [(320, 240)]),
        ("[(0.2, 0.3), (0.4, 0.5)]", "MULTIPLE_POINTS", [(128, 144), (256, 240)]),
        ("[(1.1, 0.5)]", "OUT_OF_RANGE", []),
        ("I cannot identify it.", "NO_POINT", []),
    ],
)
def test_pilot_parser_distinguishes_format_and_semantic_failure_modes(
    answer, expected_status, expected_pixels
):
    status, _, pixels = RUNNER.parse_model_answer(answer, 640, 480)
    assert status == expected_status
    assert pixels == expected_pixels


@pytest.mark.parametrize(
    "leaked_payload, expected_suffix",
    [
        ({"target_model": "apple"}, ".target_model"),
        ({"capture": {"oracle": {"pose": [1, 2, 3]}}}, ".capture.oracle"),
        ({"items": [{"semantic_labels": [26]}]}, ".items[0].semantic_labels"),
        ({"Nested": {"Ground_Truth": "secret"}}, ".Nested.Ground_Truth"),
    ],
)
def test_recursive_leakage_scan_finds_nested_oracle_fields(
    leaked_payload, expected_suffix
):
    findings = RUNNER.find_forbidden_keys(leaked_payload)
    assert len(findings) == 1
    assert findings[0].endswith(expected_suffix)


def test_recursive_leakage_scan_allows_only_inference_side_fields():
    clean = {
        "scene_id": "pilot_scene_0001",
        "instruction": "point to the fruit",
        "input_files": {
            "rgb": "pilot_scene_0001/input/rgb.png",
            "depth_m": "pilot_scene_0001/input/depth_m.npy",
        },
        "capture": {"registered_metric_depth": True},
    }
    assert RUNNER.find_forbidden_keys(clean) == []


def _make_synthetic_capture(root: Path, gate_path: Path) -> Path:
    """Create a complete ten-scene capture containing no evaluator data."""
    depth, _, _ = _compact_object_depth()
    height, width = depth.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 1] = 30
    rgb[30:63, 32:65] = (20, 40, 210)
    records = []
    for index in range(1, 11):
        scene_id = f"pilot_scene_{index:04d}"
        input_dir = root / scene_id / "input"
        input_dir.mkdir(parents=True)
        rgb_path = input_dir / "rgb.png"
        depth_path = input_dir / "depth_m.npy"
        camera_info_path = input_dir / "camera_info.json"
        tf_path = input_dir / "tf_snapshot.json"
        scene_rgb = rgb.copy()
        marker_x = 3 * index
        scene_rgb[4:12, marker_x:marker_x + 8] = (
            10 + index,
            100 + index,
            220 - index,
        )
        assert cv2.imwrite(str(rgb_path), scene_rgb)
        np.save(depth_path, depth, allow_pickle=False)
        gray = cv2.cvtColor(scene_rgb, cv2.COLOR_BGR2GRAY)
        valid_depth = np.isfinite(depth) & (depth >= 0.05) & (depth <= 2.0)
        valid_values = depth[valid_depth]
        _write_json(camera_info_path, {
            "width": width,
            "height": height,
            "k": [100.0, 0.0, width / 2, 0.0, 100.0, height / 2, 0.0, 0.0, 1.0],
        })
        _write_json(tf_path, {"camera_color_optical_frame": {"position": [0, 0, 1]}})
        record = {
            "schema_version": 1,
            "protocol_id": RUNNER.PROTOCOL_ID,
            "scene_id": scene_id,
            "instruction": "Point to the compact red object.",
            "coordinate_suffix": "Return exactly [(x, y)].",
            "input_files": {
                "rgb": str(rgb_path.relative_to(root)),
                "depth_m": str(depth_path.relative_to(root)),
                "camera_info": str(camera_info_path.relative_to(root)),
                "tf_snapshot": str(tf_path.relative_to(root)),
            },
            "input_sha256": {
                "rgb": RUNNER.sha256_file(rgb_path),
                "depth_m": RUNNER.sha256_file(depth_path),
                "camera_info": RUNNER.sha256_file(camera_info_path),
                "tf_snapshot": RUNNER.sha256_file(tf_path),
            },
            "capture": {
                "rgb_depth_label_spread_sec": 0.01,
                "sensor_stamps_sec": [float(index), float(index) + 0.005,
                                      float(index) + 0.01],
                "registered_metric_depth": True,
                "robot_manipulation_performed": False,
                "input_only_sensor_qc": {
                    "passed": True,
                    "rgb_gray_std": float(np.std(gray)),
                    "rgb_gray_p99_minus_p01": float(
                        np.percentile(gray, 99.0) - np.percentile(gray, 1.0)
                    ),
                    "rgb_nonzero_fraction": (
                        float(np.count_nonzero(gray)) / float(gray.size)
                    ),
                    "valid_depth_fraction": (
                        float(np.count_nonzero(valid_depth)) / float(depth.size)
                    ),
                    "depth_p98_minus_p02_m": float(
                        np.percentile(valid_values, 98.0)
                        - np.percentile(valid_values, 2.0)
                    ),
                },
            },
        }
        _write_json(input_dir / "scene_input.json", record)
        records.append(record)

    manifest_path = root / "input_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    annotation_stub = root / "annotation_stub.yaml"
    annotation_stub.write_text("schema_version: 1\n", encoding="utf-8")
    workspace_root = root.parent.resolve()
    pretrial_path = workspace_root / "synthetic_pretrial_lock.json"
    source_hashes = {}
    for relative_name in sorted(VALIDATOR.REQUIRED_SOURCE_ARTIFACTS):
        source_path = workspace_root / relative_name
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(f"synthetic source: {relative_name}\n", encoding="utf-8")
        source_hashes[relative_name] = RUNNER.sha256_file(source_path)
    source_hashes[str(gate_path.resolve().relative_to(workspace_root))] = (
        RUNNER.sha256_file(gate_path)
    )
    _write_json(pretrial_path, {
        "schema_version": 1,
        "protocol_id": RUNNER.PROTOCOL_ID,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "workspace_root": str(workspace_root),
        "model_inventory_sha256": SYNTHETIC_MODEL_SHA256,
        "source_artifact_sha256": source_hashes,
    })
    evaluator_hashes = {}
    for index in range(1, 11):
        scene_id = f"pilot_scene_{index:04d}"
        evaluator_dir = root / scene_id / "evaluator"
        evaluator_dir.mkdir(parents=True, exist_ok=True)
        label_path = evaluator_dir / "semantic_labels.png"
        oracle_path = evaluator_dir / "capture_oracle.json"
        assert cv2.imwrite(str(label_path), np.zeros((height, width), np.uint8))
        _write_json(oracle_path, {"oracle_usage": "synthetic_test_only"})
        evaluator_hashes[f"{scene_id}/evaluator/semantic_labels.png"] = (
            RUNNER.sha256_file(label_path)
        )
        evaluator_hashes[f"{scene_id}/evaluator/capture_oracle.json"] = (
            RUNNER.sha256_file(oracle_path)
        )
    _write_json(root / "capture_manifest.json", {
        "schema_version": 1,
        "protocol_id": RUNNER.PROTOCOL_ID,
        "status": "COMPLETE",
        "scene_count": 10,
        "input_manifest_sha256": RUNNER.sha256_file(manifest_path),
        "annotation_semantics_consumed_by_capture": False,
        "annotation_bytes_hashed_before_inference": True,
        "annotation_file_sha256": RUNNER.sha256_file(annotation_stub),
        "gate_config_bytes_hashed_before_capture": True,
        "gate_config_file_sha256": RUNNER.sha256_file(gate_path),
        "pretrial_source_lock_file": str(pretrial_path),
        "pretrial_source_lock_sha256": RUNNER.sha256_file(pretrial_path),
        "source_artifacts_verified_before_capture": True,
        "model_inventory_sha256_preregistered": SYNTHETIC_MODEL_SHA256,
        "evaluator_artifact_sha256": evaluator_hashes,
        "input_only_sensor_qc": {
            "passed": True,
            "all_scene_rgb_sha256_unique": True,
            "rgb_scene_count": 10,
            "thresholds": {
                "min_rgb_std": 5.0,
                "min_rgb_dynamic_range": 20.0,
                "min_rgb_nonzero_fraction": 0.05,
                "min_valid_depth_fraction": 0.10,
                "min_depth_dynamic_range_m": 0.05,
                "max_rgb_depth_label_spread_sec": 0.15,
            },
        },
        "semantic_labels_are_evaluator_only": True,
        "target_handoff_published": False,
        "robot_manipulation_performed": False,
    })
    return root


def _write_synthetic_gate(path: Path) -> Path:
    path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "protocol_id": RUNNER.PROTOCOL_ID,
        "gate_profile_id": "synthetic_frozen_depth_gate",
        "frozen_before_pilot": True,
        "min_depth_m": 0.05,
        "max_depth_m": 2.0,
        "depth_seed_radius_px": 4,
        "depth_roi_radius_px": 80,
        "depth_near_tolerance_m": 0.015,
        "depth_far_tolerance_m": 0.055,
        "min_mask_area_px": 100,
        "max_mask_area_fraction": 0.20,
        "bbox_padding_px": 4,
    }), encoding="utf-8")
    return path


def _reseal_prediction_records(prediction_root: Path, records) -> None:
    """Rebuild all internal hashes after an intentional negative-test edit."""
    for record in records:
        record.pop("prediction_sha256", None)
        record["prediction_sha256"] = VALIDATOR.sha256_canonical_json(record)
        _write_json(prediction_root / record["record_file"], record)
    predictions_path = prediction_root / "predictions.jsonl"
    with predictions_path.open("wb") as stream:
        for record in records:
            stream.write(RUNNER.canonical_json_bytes(record))
    manifest_path = prediction_root / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["predictions_sha256"] = RUNNER.sha256_file(predictions_path)
    _write_json(manifest_path, manifest)
    lock_path = prediction_root / "prediction_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["predictions_sha256"] = RUNNER.sha256_file(predictions_path)
    lock["prediction_manifest_sha256"] = RUNNER.sha256_file(manifest_path)
    _write_json(lock_path, lock)


def _run_synthetic_pilot(tmp_path: Path, monkeypatch):
    gate = _write_synthetic_gate(tmp_path / "gate.yaml")
    dataset = _make_synthetic_capture(tmp_path / "capture", gate)
    output = tmp_path / "predictions"
    model_root = tmp_path / "model"
    model_root.mkdir()

    def fake_query(
        server_url,
        prompt,
        rgb_jpeg,
        depth_png,
        timeout_sec,
        expected_model_inventory_sha256=None,
    ):
        answer = "[(0.3357, 0.4842)]"
        generation = {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
        }
        model_fingerprint = {
            "model_name": "synthetic",
            "inventory_sha256": expected_model_inventory_sha256,
        }
        response_json = {
            "result": 1,
            "answer": answer,
            "generation_mode": "greedy",
            "generation_config": generation,
            "model_fingerprint": model_fingerprint,
        }
        response_body = json.dumps(response_json, separators=(",", ":")).encode()
        return answer, 1.0, {
            "http_status": 200,
            "response_sha256": hashlib.sha256(response_body).hexdigest(),
            "response_body_base64": base64.b64encode(response_body).decode(),
            "response_json": response_json,
            "generation_mode": "greedy",
            "generation_config": generation,
            "model_fingerprint": model_fingerprint,
        }

    monkeypatch.setattr(RUNNER, "query_model", fake_query)
    monkeypatch.setattr(
        RUNNER,
        "model_fingerprint",
        lambda _: {
            "model_name": "synthetic",
            "inventory_sha256": SYNTHETIC_MODEL_SHA256,
        },
    )
    RUNNER.run_pilot(
        dataset,
        output,
        gate,
        model_root,
        "http://offline.invalid",
        timeout_sec=0.01,
    )
    return dataset, output


def test_offline_runner_preserves_b0_b1_inputs_and_reuses_locked_b1_for_b2(
    tmp_path, monkeypatch
):
    gate = _write_synthetic_gate(tmp_path / "gate.yaml")
    dataset = _make_synthetic_capture(tmp_path / "capture", gate)
    output = tmp_path / "predictions"
    model_root = tmp_path / "model"
    model_root.mkdir()
    calls = []

    def fake_query(
        server_url,
        prompt,
        rgb_jpeg,
        depth_png,
        timeout_sec,
        expected_model_inventory_sha256=None,
    ):
        calls.append({
            "prompt": prompt,
            "rgb": bytes(rgb_jpeg),
            "depth": None if depth_png is None else bytes(depth_png),
        })
        answer = "[(0.3357, 0.4842)]"
        generation = {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
        }
        model_fingerprint = {
            "model_name": "synthetic",
            "inventory_sha256": expected_model_inventory_sha256,
        }
        response_json = {
            "result": 1,
            "answer": answer,
            "generation_mode": "greedy",
            "generation_config": generation,
            "model_fingerprint": model_fingerprint,
        }
        response_body = json.dumps(response_json, separators=(",", ":")).encode()
        return answer, 1.0, {
            "http_status": 200,
            "response_sha256": hashlib.sha256(response_body).hexdigest(),
            "response_body_base64": base64.b64encode(response_body).decode(),
            "response_json": response_json,
            "generation_mode": "greedy",
            "generation_config": generation,
            "model_fingerprint": model_fingerprint,
        }

    monkeypatch.setattr(RUNNER, "query_model", fake_query)
    monkeypatch.setattr(
        RUNNER,
        "model_fingerprint",
        lambda _: {
            "model_name": "synthetic",
            "inventory_sha256": SYNTHETIC_MODEL_SHA256,
        },
    )

    lock_path = RUNNER.run_pilot(
        dataset,
        output,
        gate,
        model_root,
        "http://offline.invalid",
        timeout_sec=0.01,
    )

    assert lock_path == output / "prediction_lock.json"
    assert len(calls) == 20  # exactly B0+B1; B2 must never query the model
    for b0_call, b1_call in zip(calls[0::2], calls[1::2]):
        assert b0_call["prompt"] == b1_call["prompt"]
        assert b0_call["rgb"] == b1_call["rgb"]
        assert b0_call["depth"] is None
        assert b1_call["depth"] is not None

    records = RUNNER.read_jsonl(output / "predictions.jsonl")
    assert len(records) == 30
    by_scene = {}
    for record in records:
        by_scene.setdefault(record["scene_id"], {})[record["mode"]] = record
        assert record["target_handoff_published"] is False
        assert record["robot_manipulation_performed"] is False
        assert record["shadow_evaluation_only"] is True
    assert len(by_scene) == 10
    for modes in by_scene.values():
        b0, b1, b2 = modes["B0"], modes["B1"], modes["B2"]
        assert b0["policy"] == "no_uncertainty_gate_shadow"
        assert b1["policy"] == "no_uncertainty_gate_shadow"
        assert b2["policy"] == "hard_gate_shadow"
        assert b0["prompt"] == b1["prompt"] == b2["prompt"]
        assert b0["request_rgb_sha256"] == b1["request_rgb_sha256"]
        assert b0["request_depth_sha256"] is None
        assert b1["request_depth_sha256"] == b2["request_depth_sha256"]
        assert b2["model_query_performed"] is False
        assert b2["source_b1_prediction_sha256"] == b1["prediction_sha256"]
        assert b2["normalized_points_xy"] == b1["normalized_points_xy"]
        assert b2["pixel_points_xy"] == b1["pixel_points_xy"]

    report = VALIDATOR.validate_all(dataset, output)
    assert report["valid"] is True
    assert report["scene_count"] == 10
    assert report["record_count"] == 30
    assert report["mode_counts"] == {"B0": 10, "B1": 10, "B2": 10}
    assert report["oracle_files_opened"] is False

    # Removing provenance is not rescued by recomputing every downstream
    # record/manifest/lock hash: source_input_sha256 is a required contract.
    records[0].pop("source_input_sha256")
    _reseal_prediction_records(output, records)
    with pytest.raises(
        VALIDATOR.ValidationError,
        match="source input hash mapping missing or mismatched",
    ):
        VALIDATOR.validate_all(dataset, output)


def test_runner_rejects_gate_bytes_changed_after_capture(tmp_path):
    locked_gate = _write_synthetic_gate(tmp_path / "gate_locked.yaml")
    dataset = _make_synthetic_capture(tmp_path / "capture", locked_gate)
    changed_gate = _write_synthetic_gate(tmp_path / "gate_changed.yaml")
    changed_payload = yaml.safe_load(changed_gate.read_text(encoding="utf-8"))
    changed_payload["depth_far_tolerance_m"] = 0.056
    changed_gate.write_text(yaml.safe_dump(changed_payload), encoding="utf-8")
    model_root = tmp_path / "model"
    model_root.mkdir()

    with pytest.raises(ValueError, match="differs from the copy locked before capture"):
        RUNNER.run_pilot(
            dataset,
            tmp_path / "must_not_exist",
            changed_gate,
            model_root,
            "http://offline.invalid",
        )

    assert not (tmp_path / "must_not_exist").exists()


@pytest.mark.parametrize("asset_kind", ["rgb", "depth"])
def test_validator_rejects_forged_request_asset_even_when_all_hashes_are_resealed(
    tmp_path, monkeypatch, asset_kind
):
    dataset, output = _run_synthetic_pilot(tmp_path, monkeypatch)
    records = RUNNER.read_jsonl(output / "predictions.jsonl")
    scene_records = [
        record for record in records
        if record["scene_id"] == "pilot_scene_0001"
    ]
    if asset_kind == "rgb":
        relative_asset = scene_records[0]["request_rgb_file"]
        forged = np.full((96, 144, 3), 127, dtype=np.uint8)
        forged_bytes = RUNNER.encode_image(
            forged, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        (output / relative_asset).write_bytes(forged_bytes)
        forged_hash = RUNNER.sha256_bytes(forged_bytes)
        for record in scene_records:
            record["request_rgb_sha256"] = forged_hash
            record["request_sha256"]["rgb"] = forged_hash
        error_pattern = "request RGB is not the locked q90 derivation"
    else:
        relative_asset = next(
            record["request_depth_file"]
            for record in scene_records if record["request_depth_file"] is not None
        )
        forged = np.full((96, 144, 3), 111, dtype=np.uint8)
        forged_bytes = RUNNER.encode_image(forged, ".png")
        (output / relative_asset).write_bytes(forged_bytes)
        forged_hash = RUNNER.sha256_bytes(forged_bytes)
        for record in scene_records:
            if record["request_depth_file"] is not None:
                record["request_depth_sha256"] = forged_hash
                record["request_sha256"]["depth"] = forged_hash
        error_pattern = "request depth is not the locked inverse-depth derivation"

    _reseal_prediction_records(output, records)
    with pytest.raises(VALIDATOR.ValidationError, match=error_pattern):
        VALIDATOR.validate_all(dataset, output)


@pytest.mark.parametrize(
    "tamper_kind, expected_error",
    [
        ("raw_answer", "raw answer differs from HTTP response"),
        (
            "parsed_point",
            "point fields differ from independently replayed raw-answer parser",
        ),
    ],
)
def test_validator_replays_raw_http_response_and_parser_after_resealed_tamper(
    tmp_path, monkeypatch, tamper_kind, expected_error
):
    dataset, output = _run_synthetic_pilot(tmp_path, monkeypatch)
    records = RUNNER.read_jsonl(output / "predictions.jsonl")
    b0 = next(
        record for record in records
        if record["scene_id"] == "pilot_scene_0001" and record["mode"] == "B0"
    )
    if tamper_kind == "raw_answer":
        # Leave the locked raw HTTP bytes untouched while forging the convenient
        # parsed field, then reseal every downstream artifact hash.
        b0["raw_answer"] = "[(0.9000, 0.9000)]"
    else:
        # Leave both HTTP bytes and raw_answer untouched, but forge the parser's
        # derived pixel.  Hash consistency alone must not make it acceptable.
        b0["pixel_points_xy"] = [[47, 46]]

    _reseal_prediction_records(output, records)
    with pytest.raises(VALIDATOR.ValidationError, match=expected_error):
        VALIDATOR.validate_all(dataset, output)


@pytest.mark.parametrize(
    "tamper_kind, expected_error",
    [
        ("decision", "gate acceptance differs from replay"),
        ("statistic", "gate mask area differs from replay"),
        ("mask_pixels", "stored mask pixels differ from replay"),
    ],
)
def test_validator_replays_b2_gate_after_resealed_tamper(
    tmp_path, monkeypatch, tamper_kind, expected_error
):
    dataset, output = _run_synthetic_pilot(tmp_path, monkeypatch)
    records = RUNNER.read_jsonl(output / "predictions.jsonl")
    b2 = next(
        record for record in records
        if record["scene_id"] == "pilot_scene_0001" and record["mode"] == "B2"
    )
    assert b2["gate"]["accepted"] is True
    assert b2["gate"]["mask_file"] is not None

    if tamper_kind == "decision":
        b2["gate"]["accepted"] = False
        b2["would_execute"] = False
    elif tamper_kind == "statistic":
        b2["gate"]["mask_area_px"] += 1
    else:
        mask_path = output / b2["gate"]["mask_file"]
        forged_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        assert forged_mask is not None
        forged_mask[0, 0] = 255 if forged_mask[0, 0] == 0 else 0
        assert cv2.imwrite(str(mask_path), forged_mask)
        b2["gate"]["mask_sha256"] = RUNNER.sha256_file(mask_path)

    _reseal_prediction_records(output, records)
    with pytest.raises(VALIDATOR.ValidationError, match=expected_error):
        VALIDATOR.validate_all(dataset, output)


def test_capture_validator_rejects_resealed_duplicate_frozen_rgb(tmp_path):
    gate = _write_synthetic_gate(tmp_path / "gate.yaml")
    dataset = _make_synthetic_capture(tmp_path / "capture", gate)
    records = RUNNER.read_jsonl(dataset / "input_manifest.jsonl")
    source = dataset / records[0]["input_files"]["rgb"]
    destination = dataset / records[1]["input_files"]["rgb"]
    destination.write_bytes(source.read_bytes())
    duplicate_hash = RUNNER.sha256_file(destination)
    records[1]["input_sha256"]["rgb"] = duplicate_hash
    _write_json(
        dataset / records[1]["scene_id"] / "input" / "scene_input.json",
        records[1],
    )
    manifest_path = dataset / "input_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    capture_manifest_path = dataset / "capture_manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    capture_manifest["input_manifest_sha256"] = RUNNER.sha256_file(manifest_path)
    _write_json(capture_manifest_path, capture_manifest)

    with pytest.raises(
        VALIDATOR.ValidationError,
        match="does not contain 10 byte-distinct RGB captures",
    ):
        VALIDATOR.validate_capture(dataset)


def test_validator_recursive_scan_catches_aliases_and_evaluator_paths():
    payload = {
        "nested": [{"target-model-v2": "apple"}],
        "innocent_key": "pilot_scene_0001/evaluator/capture_oracle.json",
        "camelCase": {"semanticLabels": [26]},
    }
    findings = VALIDATOR.find_forbidden_input_paths(payload)
    assert "$.nested[0].target-model-v2" in findings
    assert "$.innocent_key" in findings
    assert "$.camelCase.semanticLabels" in findings


def test_oracle_inconsistent_rows_remain_raw_but_are_excluded_from_primary_metrics():
    consistent = {
        "oracle_scene_consistent": True,
        "target_state": "single",
        "action_accepted": True,
        "strict_point_output": True,
        "non_actionable_output": False,
        "safe_non_execution": False,
        "active_gate_rejection": False,
        "point_hit": True,
        "safe_eroded_hit": True,
        "false_accept": False,
        "false_reject": False,
        "true_accept": True,
        "true_reject": False,
        "candidate_should_accept": True,
        "task_success": True,
        "safe_task_success": True,
    }
    inconsistent = {
        **consistent,
        "oracle_scene_consistent": False,
        "action_accepted": False,
        "strict_point_output": False,
        "point_hit": False,
        "safe_eroded_hit": False,
        "false_reject": True,
        "task_success": False,
        "safe_task_success": False,
    }

    summary = EVALUATOR._summarize([consistent, inconsistent])

    assert summary["raw_scene_count"] == 2
    assert summary["scene_count"] == 1
    assert summary["oracle_consistent_scene_count"] == 1
    assert summary["oracle_inconsistent_scene_count"] == 1
    assert summary["point_hit_count"] == 1
    assert summary["task_success_count"] == 1
    assert summary["false_reject_count"] == 0


@pytest.mark.parametrize("target_state", ["ambiguous", "absent"])
@pytest.mark.parametrize("parse_status", ["NO_POINT", "FORMAT_VIOLATION"])
def test_missing_or_malformed_output_is_safe_non_execution_not_selective_success(
    target_state, parse_status
):
    labels = np.zeros((12, 16), dtype=np.uint8)
    labels[2:5, 2:5] = 27
    labels[7:10, 10:13] = 29
    annotation = {
        "scene_family_id": "synthetic_selective",
        "task_type": "selective_grounding",
        "target_state": target_state,
    }
    if target_state == "ambiguous":
        annotation["plausible_semantic_labels"] = [27, 29]
    else:
        annotation["absent_semantic_label"] = 28
    points = [] if parse_status == "NO_POINT" else [[3, 3]]
    record = {
        "scene_id": "pilot_scene_selective",
        "mode": "B0",
        "policy": "no_uncertainty_gate_shadow",
        "image_width": 16,
        "image_height": 12,
        "parse_status": parse_status,
        "pixel_points_xy": points,
        "would_execute": False,
        "prediction_sha256": "b" * 64,
    }

    result = EVALUATOR.evaluate_semantic_record(record, annotation, labels, 1)

    assert result["non_actionable_output"] is True
    assert result["safe_non_execution"] is True
    assert result["decision_outcome"] == "NON_ACTIONABLE_OUTPUT"
    assert result["active_gate_rejection"] is False
    assert result["selective_task_credit"] is False
    assert result["task_success"] is False
    assert result["safe_task_success"] is False


def test_only_strict_b2_gate_rejection_gets_selective_task_credit():
    labels = np.zeros((12, 16), dtype=np.uint8)
    labels[2:5, 2:5] = 27
    labels[7:10, 10:13] = 29
    annotation = {
        "scene_family_id": "synthetic_ambiguous",
        "task_type": "selective_grounding",
        "target_state": "ambiguous",
        "plausible_semantic_labels": [27, 29],
    }
    record = {
        "scene_id": "pilot_scene_ambiguous",
        "mode": "B2",
        "policy": "hard_gate_shadow",
        "image_width": 16,
        "image_height": 12,
        "parse_status": "EXACT_ONE_NORMALIZED_POINT",
        "pixel_points_xy": [[3, 3]],
        "would_execute": False,
        "gate": {"accepted": False, "reason": "DEPTH_COMPONENT_REJECTED"},
        "prediction_sha256": "c" * 64,
    }

    result = EVALUATOR.evaluate_semantic_record(record, annotation, labels, 1)

    assert result["non_actionable_output"] is False
    assert result["safe_non_execution"] is True
    assert result["active_gate_rejection"] is True
    assert result["selective_task_credit"] is True
    assert result["decision_outcome"] == "TRUE_REJECT"
    assert result["task_success"] is True
