#!/usr/bin/env python3
"""Evaluate locked RoboRefer B0/B1/B2 pilot predictions against Gazebo labels.

The security boundary in this file is intentional: ``evaluate_pilot`` first
validates the inference inputs, COMPLETE prediction manifest, prediction bytes,
and prediction lock.  Only after those checks succeed does it open the
evaluator-only annotation YAML or any semantic-label image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from roborefer_pilot_validate import (  # noqa: E402
    EXPECTED_MODES,
    EXPECTED_SCENE_COUNT,
    PROTOCOL_ID,
    ValidationError,
    sha256_file,
    validate_all,
    verify_prediction_lock as _verify_prediction_lock,
)


SCHEMA_VERSION = 1
SINGLE_TARGET_STATE = "single"
SELECTIVE_TARGET_STATES = ("ambiguous", "absent")


class EvaluationError(RuntimeError):
    """Raised when locked artifacts cannot be evaluated faithfully."""


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def verify_prediction_lock(prediction_root: Path) -> dict:
    """Public evaluator wrapper around the pre-oracle lock verification."""

    try:
        return _verify_prediction_lock(prediction_root)
    except ValidationError as exc:
        raise EvaluationError(str(exc)) from exc


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def _load_yaml_after_lock(path: Path, description: str) -> dict:
    _expect(path.is_file(), f"missing {description}: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvaluationError(f"cannot read {description} {path}: {exc}") from exc
    _expect(isinstance(payload, dict), f"{description} root must be a mapping")
    return payload


def _validate_oracle_configuration(
    annotations: Mapping[str, Any],
    gate: Mapping[str, Any],
    expected_scene_ids: Sequence[str],
) -> Mapping[str, Mapping[str, Any]]:
    _expect(annotations.get("schema_version") == SCHEMA_VERSION,
            "annotation schema_version mismatch")
    _expect(annotations.get("protocol_id") == PROTOCOL_ID,
            "annotation protocol mismatch")
    _expect(
        annotations.get("oracle_usage") ==
        "evaluation_only_open_after_prediction_lock",
        "annotations are not explicitly marked post-lock evaluator-only",
    )
    _expect(
        annotations.get("semantic_label_source") ==
        "gazebo_evaluation_camera_not_vlm_input",
        "unexpected semantic-label source",
    )
    annotation_scenes = annotations.get("scenes")
    _expect(isinstance(annotation_scenes, Mapping),
            "annotations.scenes must be a mapping")
    _expect(len(annotation_scenes) == EXPECTED_SCENE_COUNT,
            "annotations must contain exactly 10 scenes")
    _expect(set(annotation_scenes) == set(expected_scene_ids),
            "annotation scene IDs differ from locked input scene IDs")

    for scene_id in expected_scene_ids:
        item = annotation_scenes[scene_id]
        _expect(isinstance(item, Mapping), f"{scene_id}: annotation must be a mapping")
        _expect(isinstance(item.get("scene_family_id"), str) and
                bool(item.get("scene_family_id")),
                f"{scene_id}: evaluator scene_family_id missing")
        _expect(isinstance(item.get("task_type"), str) and bool(item.get("task_type")),
                f"{scene_id}: evaluator task_type missing")
        state = item.get("target_state")
        _expect(state in (SINGLE_TARGET_STATE,) + SELECTIVE_TARGET_STATES,
                f"{scene_id}: invalid target_state {state}")
        if state == SINGLE_TARGET_STATE:
            labels = item.get("target_semantic_labels")
            _expect(isinstance(labels, list) and labels and
                    all(isinstance(value, int) for value in labels),
                    f"{scene_id}: target_semantic_labels must be integer list")
        elif state == "ambiguous":
            labels = item.get("plausible_semantic_labels")
            _expect(isinstance(labels, list) and len(labels) >= 2 and
                    all(isinstance(value, int) for value in labels),
                    f"{scene_id}: ambiguous scene needs at least two plausible labels")
            _expect(item.get("desired_decision") == "abstain_or_ask",
                    f"{scene_id}: ambiguous desired_decision mismatch")
        else:
            _expect(isinstance(item.get("absent_semantic_label"), int),
                    f"{scene_id}: absent_semantic_label missing")
            _expect(item.get("desired_decision") == "abstain",
                    f"{scene_id}: absent desired_decision mismatch")
        expected_qc_type = {
            "relative_2d": "centroid_x_greater",
            "depth_relation": "median_depth_less",
            "multi_reference": "bounded_between_corridor",
            "metric_comparison": "visible_bbox_height_greater",
            "shape_comparison": "visible_bbox_width_greater",
            "occlusion": "both_labels_visible",
        }.get(str(item.get("task_type")))
        if expected_qc_type is not None:
            relation_qc = item.get("relation_qc")
            _expect(
                isinstance(relation_qc, Mapping)
                and relation_qc.get("type") == expected_qc_type,
                f"{scene_id}: locked relation_qc must be {expected_qc_type}",
            )

    _expect(gate.get("schema_version") == SCHEMA_VERSION,
            "gate schema_version mismatch")
    _expect(gate.get("protocol_id") == PROTOCOL_ID, "gate protocol mismatch")
    _expect(gate.get("frozen_before_pilot") is True,
            "gate was not frozen before the pilot")
    _expect(isinstance(gate.get("evaluator_safe_margin_px"), int) and
            gate.get("evaluator_safe_margin_px") >= 0,
            "gate evaluator_safe_margin_px must be a non-negative integer")
    return annotation_scenes


def _load_label_image(path: Path) -> np.ndarray:
    _expect(path.is_file(), f"semantic-label image missing: {path}")
    labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    _expect(labels is not None, f"cannot decode semantic-label image: {path}")
    if labels.ndim == 3:
        _expect(
            labels.shape[2] >= 3
            and np.array_equal(labels[..., 0], labels[..., 1])
            and np.array_equal(labels[..., 1], labels[..., 2]),
            f"semantic-label RGB channels disagree: {path}",
        )
        labels = labels[..., 0]
    _expect(labels.ndim == 2, f"semantic labels must be HxW: {path}")
    return np.asarray(labels)


def _mask_for_labels(labels: np.ndarray, semantic_ids: Sequence[int]) -> np.ndarray:
    return np.isin(labels, np.asarray(list(semantic_ids), dtype=labels.dtype))


def _erode_mask(mask: np.ndarray, margin_px: int) -> np.ndarray:
    if int(margin_px) <= 0:
        return mask.astype(bool).copy()
    radius = int(margin_px)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    eroded = cv2.erode(
        mask.astype(np.uint8),
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return eroded > 0


def _one_in_bounds_point(record: Mapping[str, Any], width: int, height: int):
    points = record.get("pixel_points_xy")
    if not isinstance(points, list) or len(points) != 1:
        return None
    point = points[0]
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    try:
        x_pixel, y_pixel = int(point[0]), int(point[1])
    except (TypeError, ValueError):
        return None
    if not (0 <= x_pixel < width and 0 <= y_pixel < height):
        return None
    return x_pixel, y_pixel


def _safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _label_geometry(labels: np.ndarray, semantic_label: int) -> Optional[dict]:
    y_pixels, x_pixels = np.nonzero(labels == int(semantic_label))
    if not x_pixels.size:
        return None
    return {
        "visible_pixels": int(x_pixels.size),
        "centroid_xy": [float(np.mean(x_pixels)), float(np.mean(y_pixels))],
        "bbox_xyxy": [
            int(x_pixels.min()),
            int(y_pixels.min()),
            int(x_pixels.max()),
            int(y_pixels.max()),
        ],
        "bbox_width_px": int(x_pixels.max() - x_pixels.min() + 1),
        "bbox_height_px": int(y_pixels.max() - y_pixels.min() + 1),
    }


def evaluate_scene_condition(
    annotation: Mapping[str, Any],
    semantic_labels: np.ndarray,
    depth_m: Optional[np.ndarray] = None,
) -> dict:
    """Post-lock QC for the spatial condition claimed by a pilot scene.

    This checks scene construction, not model correctness.  It uses only the
    evaluator-only labels (and registered metric depth for the depth relation).
    The occlusion case is explicitly limited: visible masks can prove that both
    objects are visible, but cannot by themselves prove which surface occludes
    another without amodal masks or geometry/ray evidence.
    """

    task_type = str(annotation.get("task_type", ""))
    spec = annotation.get("relation_qc", {})
    result = {
        "required": False,
        "passed": None,
        "fully_verified": True,
        "note": "NOT_REQUIRED_FOR_THIS_TASK_TYPE",
        "measurements": {},
        "locked_spec": dict(spec) if isinstance(spec, Mapping) else {},
    }

    if task_type == "relative_2d":
        result["required"] = True
        target_label = int(spec.get("target_label", 26))
        reference_label = int(spec.get("reference_label", 24))
        min_delta = float(spec.get("min_delta_x_px", 0.0))
        target = _label_geometry(semantic_labels, target_label)
        reference = _label_geometry(semantic_labels, reference_label)
        result["measurements"] = {
            f"target_{target_label}": target,
            f"reference_{reference_label}": reference,
            "locked_min_delta_x_px": min_delta,
        }
        if target is None or reference is None:
            result.update(passed=False, note="RELATIVE_2D_REQUIRED_LABEL_NOT_VISIBLE")
        else:
            delta_x = target["centroid_xy"][0] - reference["centroid_xy"][0]
            result["measurements"]["target_minus_reference_centroid_x_px"] = delta_x
            result.update(
                passed=bool(delta_x > min_delta),
                note=(
                    "TARGET_CENTROID_RIGHT_OF_BOTTLE"
                    if delta_x > min_delta
                    else "TARGET_CENTROID_NOT_RIGHT_OF_BOTTLE"
                ),
            )
    elif task_type == "depth_relation":
        result["required"] = True
        if depth_m is None or np.asarray(depth_m).shape != semantic_labels.shape:
            result.update(passed=False, note="REGISTERED_DEPTH_MISSING_OR_SHAPE_MISMATCH")
        else:
            values = np.asarray(depth_m, dtype=np.float32)
            medians: Dict[str, Optional[float]] = {}
            target_label = int(spec.get("target_label", 26))
            reference_label = int(spec.get("reference_label", 27))
            min_margin = float(spec.get("min_depth_margin_m", 0.0))
            for name, label in (
                (f"target_{target_label}", target_label),
                (f"reference_{reference_label}", reference_label),
            ):
                samples = values[semantic_labels == label]
                samples = samples[np.isfinite(samples) & (samples > 0.0)]
                medians[name] = float(np.median(samples)) if samples.size else None
            result["measurements"] = {"median_depth_m": medians}
            target_depth = medians[f"target_{target_label}"]
            reference_depth = medians[f"reference_{reference_label}"]
            result["measurements"]["locked_min_depth_margin_m"] = min_margin
            if target_depth is None or reference_depth is None:
                result.update(passed=False, note="DEPTH_RELATION_HAS_NO_VALID_LABEL_DEPTH")
            else:
                delta = reference_depth - target_depth
                result["measurements"]["reference_minus_target_depth_m"] = delta
                result.update(
                    passed=bool(delta > min_margin),
                    note=(
                        "APPLE_MEDIAN_DEPTH_CLOSER_THAN_ORANGE"
                        if delta > min_margin
                        else "APPLE_NOT_CLOSER_THAN_ORANGE"
                    ),
                )
    elif task_type == "multi_reference":
        result["required"] = True
        reference_a_label = int(spec.get("reference_a_label", 27))
        reference_b_label = int(spec.get("reference_b_label", 25))
        candidate_label = int(spec.get("candidate_label", 23))
        projection_min = float(spec.get("projection_min", 0.0))
        projection_max = float(spec.get("projection_max", 1.0))
        max_perpendicular_ratio = float(
            spec.get("max_perpendicular_to_reference_distance_ratio", 0.35)
        )
        orange = _label_geometry(semantic_labels, reference_a_label)
        banana = _label_geometry(semantic_labels, reference_b_label)
        can = _label_geometry(semantic_labels, candidate_label)
        result["measurements"] = {
            "orange_27": orange,
            "banana_25": banana,
            "can_23": can,
            "locked_projection_bounds": [projection_min, projection_max],
            "locked_max_perpendicular_ratio": max_perpendicular_ratio,
        }
        if orange is None or banana is None or can is None:
            result.update(passed=False, note="BETWEEN_RELATION_REQUIRED_LABEL_NOT_VISIBLE")
        else:
            start = np.asarray(orange["centroid_xy"], dtype=np.float64)
            end = np.asarray(banana["centroid_xy"], dtype=np.float64)
            candidate = np.asarray(can["centroid_xy"], dtype=np.float64)
            axis = end - start
            norm_sq = float(np.dot(axis, axis))
            if norm_sq <= 1e-9:
                result.update(passed=False, note="REFERENCE_CENTROIDS_COINCIDE")
            else:
                projection = float(np.dot(candidate - start, axis) / norm_sq)
                projected = start + projection * axis
                perpendicular_px = float(np.linalg.norm(candidate - projected))
                reference_distance_px = float(np.sqrt(norm_sq))
                perpendicular_ratio = perpendicular_px / reference_distance_px
                result["measurements"].update({
                    "projection_parameter_orange_to_banana": projection,
                    "endpoint_margin": min(projection, 1.0 - projection),
                    "perpendicular_distance_to_reference_axis_px": perpendicular_px,
                    "reference_centroid_distance_px": reference_distance_px,
                    "perpendicular_to_reference_distance_ratio": perpendicular_ratio,
                })
                # "Between" is operationalized as an interior projection plus
                # a bounded lateral deviation from the reference segment.  The
                # 0.35 ratio is evaluator policy, not a learned/tuned threshold.
                passed = (
                    projection_min <= projection <= projection_max
                    and perpendicular_ratio <= max_perpendicular_ratio
                )
                result.update(
                    passed=bool(passed),
                    note=(
                        "CAN_LIES_WITHIN_BOUNDED_BETWEEN_CORRIDOR"
                        if passed
                        else "CAN_OUTSIDE_BOUNDED_BETWEEN_CORRIDOR"
                    ),
                )
    elif task_type == "metric_comparison":
        result["required"] = True
        target_label = int(spec.get("target_label", 24))
        comparison_labels = [int(value) for value in spec.get(
            "comparison_labels", [26, 27]
        )]
        geometries = {
            f"target_{target_label}": _label_geometry(semantic_labels, target_label),
            **{
                f"comparison_{label}": _label_geometry(semantic_labels, label)
                for label in comparison_labels
            },
        }
        result["measurements"] = geometries
        if any(value is None for value in geometries.values()):
            result.update(
                passed=False,
                fully_verified=False,
                note="HEIGHT_COMPARISON_REQUIRED_LABEL_NOT_VISIBLE",
            )
        else:
            target_height = geometries[f"target_{target_label}"]["bbox_height_px"]
            passed = all(
                target_height > geometries[f"comparison_{label}"]["bbox_height_px"]
                for label in comparison_labels
            )
            result.update(
                passed=bool(passed),
                fully_verified=False,
                note=(
                    "BOTTLE_VISIBLE_BBOX_TALLER; PHYSICAL_HEIGHT_AND_UPRIGHT_POSE_"
                    "NOT_PROVABLE_FROM_VISIBLE_MASKS"
                    if passed
                    else "BOTTLE_VISIBLE_BBOX_NOT_TALLER_THAN_BOTH_FRUITS"
                ),
            )
    elif task_type == "shape_comparison":
        result["required"] = True
        target_label = int(spec.get("target_label", 25))
        comparison_labels = [int(value) for value in spec.get(
            "comparison_labels", [26, 27]
        )]
        geometries = {
            f"target_{target_label}": _label_geometry(semantic_labels, target_label),
            **{
                f"comparison_{label}": _label_geometry(semantic_labels, label)
                for label in comparison_labels
            },
        }
        result["measurements"] = geometries
        if any(value is None for value in geometries.values()):
            result.update(passed=False, note="SHAPE_COMPARISON_REQUIRED_LABEL_NOT_VISIBLE")
        else:
            target_width = geometries[f"target_{target_label}"]["bbox_width_px"]
            passed = all(
                target_width > geometries[f"comparison_{label}"]["bbox_width_px"]
                for label in comparison_labels
            )
            result.update(
                passed=bool(passed),
                note=(
                    "BANANA_VISIBLE_BBOX_WIDER_THAN_BOTH_ROUND_FRUITS"
                    if passed
                    else "BANANA_VISIBLE_BBOX_NOT_WIDER_THAN_BOTH_ROUND_FRUITS"
                ),
            )
    elif task_type == "occlusion":
        result["required"] = True
        required_labels = [int(value) for value in spec.get(
            "required_labels", [27, 24]
        )]
        geometries = {
            f"required_{label}": _label_geometry(semantic_labels, label)
            for label in required_labels
        }
        result["measurements"] = geometries
        both_visible = all(value is not None for value in geometries.values())
        result.update(
            passed=bool(both_visible),
            fully_verified=False,
            note=(
                "BOTH_OBJECTS_VISIBLE; PARTIAL_OCCLUSION_NOT_PROVABLE_FROM_VISIBLE_MASKS"
                if both_visible
                else "OCCLUSION_SCENE_REQUIRED_LABEL_NOT_VISIBLE"
            ),
        )
    return result


def evaluate_semantic_record(
    record: Mapping[str, Any],
    annotation: Mapping[str, Any],
    semantic_labels: np.ndarray,
    safe_margin_px: int,
    depth_m: Optional[np.ndarray] = None,
) -> dict:
    """Evaluate one locked prediction record against one oracle label image."""

    height, width = semantic_labels.shape
    _expect(record.get("image_width") == width and record.get("image_height") == height,
            f"{record.get('scene_id')}/{record.get('mode')}: image dimensions differ "
            "from semantic labels")
    target_state = str(annotation["target_state"])
    point = _one_in_bounds_point(record, width, height)
    point_available = point is not None
    strict_point = (
        record.get("parse_status") == "EXACT_ONE_NORMALIZED_POINT"
        and point_available
    )

    target_mask = np.zeros_like(semantic_labels, dtype=bool)
    safe_mask = np.zeros_like(target_mask)
    plausible_mask = np.zeros_like(target_mask)
    point_hit: Optional[bool] = None
    safe_hit: Optional[bool] = None
    plausible_hit: Optional[bool] = None
    target_visible_pixels: Optional[int] = None
    safe_target_pixels: Optional[int] = None
    oracle_scene_consistent = True
    oracle_consistency_note = "OK"
    absent_label_visible_px: Optional[int] = None

    if target_state == SINGLE_TARGET_STATE:
        target_ids = annotation["target_semantic_labels"]
        target_mask = _mask_for_labels(semantic_labels, target_ids)
        safe_mask = _erode_mask(target_mask, safe_margin_px)
        target_visible_pixels = int(np.count_nonzero(target_mask))
        safe_target_pixels = int(np.count_nonzero(safe_mask))
        if target_visible_pixels == 0:
            oracle_scene_consistent = False
            oracle_consistency_note = "DECLARED_SINGLE_TARGET_NOT_VISIBLE"
        elif safe_target_pixels == 0:
            oracle_scene_consistent = False
            oracle_consistency_note = "TARGET_VISIBLE_BUT_SAFE_EROSION_EMPTY"
        if point is not None:
            x_pixel, y_pixel = point
            point_hit = bool(target_mask[y_pixel, x_pixel])
            safe_hit = bool(safe_mask[y_pixel, x_pixel])
        else:
            point_hit = False
            safe_hit = False
    elif target_state == "ambiguous":
        plausible_ids = annotation["plausible_semantic_labels"]
        plausible_mask = _mask_for_labels(semantic_labels, plausible_ids)
        visible_label_count = sum(
            int(np.count_nonzero(semantic_labels == label)) > 0
            for label in plausible_ids
        )
        if visible_label_count < 2:
            oracle_scene_consistent = False
            oracle_consistency_note = (
                "AMBIGUOUS_SCENE_HAS_FEWER_THAN_TWO_VISIBLE_PLAUSIBLE_LABELS"
            )
        if point is not None:
            x_pixel, y_pixel = point
            plausible_hit = bool(plausible_mask[y_pixel, x_pixel])
        else:
            plausible_hit = False
    else:
        absent_id = int(annotation["absent_semantic_label"])
        absent_mask = semantic_labels == absent_id
        absent_label_visible_px = int(np.count_nonzero(absent_mask))
        if absent_label_visible_px > 0:
            oracle_scene_consistent = False
            oracle_consistency_note = "DECLARED_ABSENT_LABEL_IS_VISIBLE"

    relation_qc = evaluate_scene_condition(
        annotation, semantic_labels, depth_m=depth_m
    )
    if relation_qc["required"] and relation_qc["passed"] is not True:
        oracle_scene_consistent = False
        if oracle_consistency_note == "OK":
            oracle_consistency_note = str(relation_qc["note"])
        else:
            oracle_consistency_note += "; " + str(relation_qc["note"])

    action_accepted = bool(record.get("would_execute"))
    non_actionable_output = not strict_point
    safe_non_execution = not action_accepted
    gate = record.get("gate") if record.get("mode") == "B2" else None
    active_gate_rejection = bool(
        record.get("mode") == "B2"
        and strict_point
        and isinstance(gate, Mapping)
        and gate.get("accepted") is False
    )
    # The selective decision oracle asks whether this *particular proposal* is
    # actionable.  Only a strict, contract-compliant point is a proposal that
    # can enter the accept/reject confusion matrix.  A parse/query/format
    # failure can prevent motion, but it is not evidence of intentional
    # abstention or uncertainty handling.
    candidate_should_accept = bool(
        strict_point
        and target_state == SINGLE_TARGET_STATE
        and point_hit is True
    )
    safe_candidate_should_accept = bool(
        strict_point
        and target_state == SINGLE_TARGET_STATE
        and safe_hit is True
    )
    if non_actionable_output:
        decision_outcome = "NON_ACTIONABLE_OUTPUT"
    elif action_accepted and candidate_should_accept:
        decision_outcome = "TRUE_ACCEPT"
    elif action_accepted and not candidate_should_accept:
        decision_outcome = "FALSE_ACCEPT"
    elif not action_accepted and candidate_should_accept:
        decision_outcome = "FALSE_REJECT"
    else:
        decision_outcome = "TRUE_REJECT"

    if target_state == SINGLE_TARGET_STATE:
        task_success = bool(strict_point and action_accepted and point_hit is True)
        safe_task_success = bool(
            strict_point and action_accepted and safe_hit is True
        )
        selective_task_credit = False
    else:
        # Preregistered selective success requires a valid B1 proposal followed
        # by an explicit B2 gate rejection.  B0/B1 have no uncertainty gate,
        # and a malformed/missing output receives no selective-task credit.
        selective_task_credit = active_gate_rejection
        task_success = selective_task_credit
        safe_task_success = selective_task_credit

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "scene_id": record["scene_id"],
        "scene_family_id": annotation["scene_family_id"],
        "task_type": annotation["task_type"],
        "mode": record["mode"],
        "policy": record.get("policy"),
        "target_state": target_state,
        "prediction_sha256": record["prediction_sha256"],
        "parse_status": record.get("parse_status"),
        "point_available": point_available,
        "strict_point_output": strict_point,
        "point_xy": list(point) if point is not None else None,
        "point_hit": point_hit,
        "safe_eroded_hit": safe_hit,
        "plausible_point_hit": plausible_hit,
        "target_visible_pixels": target_visible_pixels,
        "safe_target_pixels": safe_target_pixels,
        "absent_label_visible_pixels": absent_label_visible_px,
        "safe_margin_px": int(safe_margin_px),
        "oracle_scene_consistent": oracle_scene_consistent,
        "oracle_consistency_note": oracle_consistency_note,
        "scene_condition_qc": relation_qc,
        "action_accepted": action_accepted,
        "non_actionable_output": non_actionable_output,
        "safe_non_execution": safe_non_execution,
        "active_gate_rejection": active_gate_rejection,
        "selective_task_credit": selective_task_credit,
        "candidate_should_accept": candidate_should_accept,
        "safe_candidate_should_accept": safe_candidate_should_accept,
        "decision_outcome": decision_outcome,
        "true_accept": decision_outcome == "TRUE_ACCEPT",
        "true_reject": decision_outcome == "TRUE_REJECT",
        "false_accept": decision_outcome == "FALSE_ACCEPT",
        "false_reject": decision_outcome == "FALSE_REJECT",
        "task_success": task_success,
        "safe_task_success": safe_task_success,
        "b2_gate_accepted": (
            bool(gate.get("accepted")) if isinstance(gate, Mapping) else None
        ),
        "b2_gate_reason": gate.get("reason") if isinstance(gate, Mapping) else None,
        "target_handoff_published": False,
        "robot_manipulation_performed": False,
        "shadow_evaluation_only": True,
    }


def _summarize(records: Sequence[Mapping[str, Any]]) -> dict:
    raw_records = list(records)
    # Primary model metrics exclude oracle-QC failures; all raw rows remain in
    # evaluation.jsonl and their count is reported explicitly for audit.
    records = [item for item in raw_records if item["oracle_scene_consistent"]]
    count = len(records)
    single = [item for item in records if item["target_state"] == SINGLE_TARGET_STATE]
    selective = [item for item in records if item["target_state"] in SELECTIVE_TARGET_STATES]
    accepted = sum(bool(item["action_accepted"]) for item in records)
    strict = sum(bool(item["strict_point_output"]) for item in records)
    non_actionable = sum(bool(item["non_actionable_output"]) for item in records)
    safe_non_execution = sum(bool(item["safe_non_execution"]) for item in records)
    active_gate_rejections = sum(
        bool(item["active_gate_rejection"]) for item in records
    )
    point_hits = sum(item["point_hit"] is True for item in single)
    safe_hits = sum(item["safe_eroded_hit"] is True for item in single)
    false_accepts = sum(bool(item["false_accept"]) for item in records)
    false_rejects = sum(bool(item["false_reject"]) for item in records)
    correct_candidates = sum(bool(item["candidate_should_accept"]) for item in records)
    selective_accepts = sum(bool(item["action_accepted"]) for item in selective)
    task_successes = sum(bool(item["task_success"]) for item in records)
    safe_task_successes = sum(bool(item["safe_task_success"]) for item in records)
    return {
        "raw_scene_count": len(raw_records),
        "scene_count": count,
        "single_target_scene_count": len(single),
        "ambiguous_or_absent_scene_count": len(selective),
        "oracle_consistent_scene_count": count,
        "oracle_inconsistent_scene_count": len(raw_records) - count,
        "strict_point_output_count": strict,
        "strict_point_output_rate": _safe_ratio(strict, count),
        "non_actionable_output_count": non_actionable,
        "non_actionable_output_rate": _safe_ratio(non_actionable, count),
        "safe_non_execution_count": safe_non_execution,
        "safe_non_execution_rate": _safe_ratio(safe_non_execution, count),
        "active_gate_rejection_count": active_gate_rejections,
        "decision_evaluable_count": strict,
        "point_hit_count": point_hits,
        "point_hit_rate_single_targets": _safe_ratio(point_hits, len(single)),
        "safe_eroded_hit_count": safe_hits,
        "safe_eroded_hit_rate_single_targets": _safe_ratio(safe_hits, len(single)),
        "accepted_count": accepted,
        "non_execution_count": count - accepted,
        "true_accept_count": sum(bool(item["true_accept"]) for item in records),
        "true_reject_count": sum(bool(item["true_reject"]) for item in records),
        "false_accept_count": false_accepts,
        "false_reject_count": false_rejects,
        "false_accept_rate_among_accepts": _safe_ratio(false_accepts, accepted),
        "false_reject_rate_among_correct_candidates": _safe_ratio(
            false_rejects, correct_candidates
        ),
        "ambiguous_or_absent_accept_count": selective_accepts,
        "ambiguous_or_absent_active_gate_rejection_count": sum(
            bool(item["active_gate_rejection"]) for item in selective
        ),
        "task_success_count": task_successes,
        "task_success_rate": _safe_ratio(task_successes, count),
        "safe_task_success_count": safe_task_successes,
        "safe_task_success_rate": _safe_ratio(safe_task_successes, count),
    }


def _build_summary(evaluations: Sequence[Mapping[str, Any]]) -> dict:
    by_mode = {
        mode: _summarize([item for item in evaluations if item["mode"] == mode])
        for mode in EXPECTED_MODES
    }
    task_types = sorted({str(item["task_type"]) for item in evaluations})
    by_task_type = {
        task_type: {
            mode: _summarize([
                item for item in evaluations
                if item["task_type"] == task_type and item["mode"] == mode
            ])
            for mode in EXPECTED_MODES
        }
        for task_type in task_types
    }
    b0, b1, b2 = (by_mode[mode] for mode in EXPECTED_MODES)
    comparison = {
        "B1_minus_B0_point_hit_count":
            b1["point_hit_count"] - b0["point_hit_count"],
        "B1_minus_B0_safe_eroded_hit_count":
            b1["safe_eroded_hit_count"] - b0["safe_eroded_hit_count"],
        "B1_minus_B0_task_success_count":
            b1["task_success_count"] - b0["task_success_count"],
        "B2_minus_B1_false_accept_count":
            b2["false_accept_count"] - b1["false_accept_count"],
        "B2_minus_B1_false_reject_count":
            b2["false_reject_count"] - b1["false_reject_count"],
        "B2_minus_B1_task_success_count":
            b2["task_success_count"] - b1["task_success_count"],
    }
    inconsistent_scenes = sorted({
        str(item["scene_id"])
        for item in evaluations
        if not item["oracle_scene_consistent"]
    })
    condition_not_fully_verified = sorted({
        str(item["scene_id"])
        for item in evaluations
        if item.get("scene_condition_qc", {}).get("required") is True
        and item.get("scene_condition_qc", {}).get("fully_verified") is False
    })
    return {
        "mode_summary": by_mode,
        "task_type_summary": by_task_type,
        "comparisons": comparison,
        "oracle_inconsistent_scene_ids": inconsistent_scenes,
        "oracle_condition_not_fully_verified_scene_ids":
            condition_not_fully_verified,
    }


def _summary_rows(summary: Mapping[str, Any]) -> List[dict]:
    rows: List[dict] = []
    for mode, metrics in summary["mode_summary"].items():
        rows.append({"scope": "overall", "task_type": "ALL", "mode": mode, **metrics})
    for task_type, mode_items in summary["task_type_summary"].items():
        for mode, metrics in mode_items.items():
            rows.append({
                "scope": "task_type",
                "task_type": task_type,
                "mode": mode,
                **metrics,
            })
    return rows


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_ratio(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{100.0 * value:.1f}%"


def _result_markdown(summary: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    lines = [
        "# Kết quả pilot RoboRefer B0/B1/B2",
        "",
        "Đây là đánh giá **shadow/no-manipulation**: không tọa độ đích nào được "
        "gửi sang robot và robot không thực hiện gắp đặt.",
        "",
        "## Tóm tắt",
        "",
        "| Mode | Point hit | Safe hit | Chấp nhận | Non-actionable | Safe no-exec | "
        "False accept | False reject | Task success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in EXPECTED_MODES:
        item = summary["mode_summary"][mode]
        lines.append(
            f"| {mode} | {_format_ratio(item['point_hit_rate_single_targets'])} "
            f"({item['point_hit_count']}/{item['single_target_scene_count']}) | "
            f"{_format_ratio(item['safe_eroded_hit_rate_single_targets'])} "
            f"({item['safe_eroded_hit_count']}/{item['single_target_scene_count']}) | "
            f"{item['accepted_count']}/{item['scene_count']} | "
            f"{item['non_actionable_output_count']} | "
            f"{item['safe_non_execution_count']} | "
            f"{item['false_accept_count']} | {item['false_reject_count']} | "
            f"{_format_ratio(item['task_success_rate'])} |"
        )
    comparisons = summary["comparisons"]
    lines.extend([
        "",
        "## Đối chiếu trực tiếp",
        "",
        f"- B1 − B0: `{comparisons['B1_minus_B0_point_hit_count']:+d}` point hit, "
        f"`{comparisons['B1_minus_B0_safe_eroded_hit_count']:+d}` safe hit, "
        f"`{comparisons['B1_minus_B0_task_success_count']:+d}` task success.",
        f"- B2 − B1: `{comparisons['B2_minus_B1_false_accept_count']:+d}` false "
        f"accept, `{comparisons['B2_minus_B1_false_reject_count']:+d}` false reject, "
        f"`{comparisons['B2_minus_B1_task_success_count']:+d}` task success.",
        "",
        "## Cách đọc trung thực",
        "",
        "- `point hit`: điểm nằm trên semantic mask của đúng vật; `safe hit`: điểm "
        "còn nằm trong mask sau khi co biên theo cấu hình đã khóa.",
        "- `false accept`: hệ thống sẽ hành động với một điểm sai/không xác định; "
        "`false reject`: điểm đúng nhưng gate từ chối.",
        "- `non-actionable output` là lỗi query/parse/format hoặc thiếu đúng một "
        "điểm chuẩn hóa. `safe no-exec` chỉ nói rằng không có chuyển động; nó không "
        "được tính là abstention có chủ ý hay task success.",
        "- B0/B1 mang policy `no_uncertainty_gate_shadow`: nếu có đúng một điểm hợp "
        "lệ thì đề xuất sẽ được chấp nhận; chúng không có cơ chế uncertainty gate.",
        "- Với cảnh mơ hồ/vắng đích, chỉ B2 được credit task success khi B1 đã tạo "
        "một điểm strict hợp lệ và depth gate chủ động từ chối. Lỗi format/query "
        "không nhận credit.",
        "- B2 dùng nguyên điểm B1 đã khóa và không truy vấn RoboRefer lần nữa.",
    ])
    inconsistent = summary["oracle_inconsistent_scene_ids"]
    if inconsistent:
        lines.extend([
            "",
            "## Cảnh báo chất lượng oracle",
            "",
            "Các cảnh sau không nhất quán với nhãn preregistered: "
            f"`{', '.join(inconsistent)}`. Bản ghi thô vẫn được giữ để audit nhưng "
            "bị loại khỏi metric model chính; trạng thái đánh giá được đánh dấu "
            "`COMPLETE_WITH_ORACLE_QC_FAILURE`.",
        ])
    not_fully_verified = summary["oracle_condition_not_fully_verified_scene_ids"]
    if not_fully_verified:
        lines.extend([
            "",
            "## Giới hạn xác minh điều kiện cảnh",
            "",
            "Các cảnh sau chỉ có kiểm tra proxy bằng visible mask và chưa thể chứng "
            "minh đầy đủ điều kiện vật lý/amodal: "
            f"`{', '.join(not_fully_verified)}`. Cần montage/manual QC hoặc bằng "
            "chứng hình học bổ sung trước khi diễn giải kết quả theo điều kiện đó. "
            "Nếu không có lỗi QC khác, trạng thái là "
            "`COMPLETE_WITH_ORACLE_CONDITION_LIMITATION`.",
        ])
    lines.extend([
        "",
        "## Tính toàn vẹn",
        "",
        f"- Prediction manifest SHA256: `{metadata['prediction_manifest_sha256']}`",
        f"- Predictions SHA256: `{metadata['predictions_sha256']}`",
        f"- Prediction lock SHA256: `{metadata['prediction_lock_sha256']}`",
        f"- Annotation SHA256: `{metadata['annotations_sha256']}`",
        f"- Gate config SHA256: `{metadata['gate_config_sha256']}`",
        "- Oracle chỉ được mở sau khi COMPLETE manifest và prediction lock đã được "
        "xác minh: `true`.",
        "",
    ])
    return "\n".join(lines)


def evaluate_pilot(
    dataset_root: Path,
    prediction_root: Path,
    annotations_file: Path,
    gate_config_file: Path,
    output_root: Path,
) -> dict:
    """Validate, then open the oracle and evaluate all 30 locked records."""

    dataset = Path(dataset_root).expanduser().resolve()
    predictions = Path(prediction_root).expanduser().resolve()
    annotations_path = Path(annotations_file).expanduser().resolve()
    gate_path = Path(gate_config_file).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()

    # SECURITY BOUNDARY: do not move any annotation/label access above this.
    try:
        validation = validate_all(dataset, predictions)
    except ValidationError as exc:
        raise EvaluationError(
            f"pre-oracle validation failed; evaluator refused to open oracle: {exc}"
        ) from exc
    validation_completed_wall_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    _expect(not output.exists(), f"immutable evaluation output already exists: {output}")
    # Re-read the tiny lock/manifest pair immediately at the boundary as a
    # final guard against an accidental mutation after the full validation.
    locked_manifest = verify_prediction_lock(predictions)["manifest"]
    # From this point onward the predictions are demonstrably COMPLETE+LOCKED.
    oracle_opened_wall_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    annotations = _load_yaml_after_lock(annotations_path, "annotation file")
    gate = _load_yaml_after_lock(gate_path, "gate config")
    capture_manifest = json.loads(
        (dataset / "capture_manifest.json").read_text(encoding="utf-8")
    )
    _expect(
        capture_manifest.get("annotation_file_sha256")
        == sha256_file(annotations_path),
        "annotation file changed after capture preregistration",
    )
    _expect(locked_manifest.get("gate_config_sha256") == sha256_file(gate_path),
            "evaluator gate config differs from the frozen runner gate config")

    input_records = validation["capture"]["records"]
    scene_ids = [str(record["scene_id"]) for record in input_records]
    inputs_by_scene = {
        str(record["scene_id"]): record for record in input_records
    }
    annotation_scenes = _validate_oracle_configuration(
        annotations, gate, scene_ids
    )
    prediction_records = validation["records"]
    predictions_by_scene_mode = {
        (str(record["scene_id"]), str(record["mode"])): record
        for record in prediction_records
    }
    safe_margin = int(gate["evaluator_safe_margin_px"])

    evaluations: List[dict] = []
    semantic_hashes: Dict[str, str] = {}
    for scene_id in scene_ids:
        # Deliberately use only the evaluator-side file.  capture_oracle.json is
        # unnecessary and therefore never opened by this evaluator.
        label_path = dataset / scene_id / "evaluator" / "semantic_labels.png"
        relative_label = f"{scene_id}/evaluator/semantic_labels.png"
        _expect(
            capture_manifest.get("evaluator_artifact_sha256", {}).get(relative_label)
            == sha256_file(label_path),
            f"{scene_id}: semantic labels changed after capture lock",
        )
        semantic_labels = _load_label_image(label_path)
        depth_path = dataset / inputs_by_scene[scene_id]["input_files"]["depth_m"]
        try:
            depth_m = np.load(depth_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise EvaluationError(
                f"cannot load registered metric depth for {scene_id}: {exc}"
            ) from exc
        _expect(
            depth_m.ndim == 2 and depth_m.shape == semantic_labels.shape,
            f"registered depth/semantic-label shape mismatch for {scene_id}",
        )
        semantic_hashes[scene_id] = sha256_file(label_path)
        annotation = annotation_scenes[scene_id]
        for mode in EXPECTED_MODES:
            record = predictions_by_scene_mode[(scene_id, mode)]
            evaluations.append(evaluate_semantic_record(
                record, annotation, semantic_labels, safe_margin, depth_m=depth_m
            ))

    _expect(len(evaluations) == EXPECTED_SCENE_COUNT * len(EXPECTED_MODES),
            "internal evaluator record-count error")
    aggregate = _build_summary(evaluations)
    evaluation_status = (
        "COMPLETE_WITH_ORACLE_QC_FAILURE"
        if aggregate["oracle_inconsistent_scene_ids"]
        else (
            "COMPLETE_WITH_ORACLE_CONDITION_LIMITATION"
            if aggregate["oracle_condition_not_fully_verified_scene_ids"]
            else "COMPLETE"
        )
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": evaluation_status,
        "evaluation_mode": "shadow_no_manipulation",
        "scene_count": EXPECTED_SCENE_COUNT,
        "record_count": len(evaluations),
        "validation_completed_before_oracle_open": True,
        "validation_completed_wall_time": validation_completed_wall_time,
        "oracle_opened_wall_time": oracle_opened_wall_time,
        "oracle_opened_after_lock_verified": True,
        "capture_oracle_json_opened": False,
        "target_handoff_published": False,
        "robot_manipulation_performed": False,
        "prediction_manifest_sha256": validation["prediction_manifest_sha256"],
        "predictions_sha256": validation["predictions_sha256"],
        "prediction_lock_sha256": validation["prediction_lock_sha256"],
        "annotations_file": str(annotations_path),
        "annotations_sha256": sha256_file(annotations_path),
        "gate_config_file": str(gate_path),
        "gate_config_sha256": sha256_file(gate_path),
        "semantic_label_sha256": semantic_hashes,
        "safe_margin_px": safe_margin,
    }
    summary = {**metadata, **aggregate}

    output.mkdir(parents=True, exist_ok=False)
    evaluation_jsonl = output / "evaluation.jsonl"
    summary_json = output / "summary.json"
    summary_csv = output / "summary.csv"
    result_md = output / "RESULT.md"
    evaluation_manifest = output / "evaluation_manifest.json"
    _write_jsonl(evaluation_jsonl, evaluations)
    write_json(summary_json, summary)
    _write_summary_csv(summary_csv, _summary_rows(summary))
    result_md.write_text(_result_markdown(summary, metadata), encoding="utf-8")
    manifest = {
        **metadata,
        "completed_wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "files": {
            "evaluation_jsonl": evaluation_jsonl.name,
            "summary_json": summary_json.name,
            "summary_csv": summary_csv.name,
            "result_markdown": result_md.name,
        },
        "file_sha256": {
            "evaluation_jsonl": sha256_file(evaluation_jsonl),
            "summary_json": sha256_file(summary_json),
            "summary_csv": sha256_file(summary_csv),
            "result_markdown": sha256_file(result_md),
        },
    }
    write_json(evaluation_manifest, manifest)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        summary = evaluate_pilot(
            dataset_root=args.dataset_root,
            prediction_root=args.prediction_root,
            annotations_file=args.annotations,
            gate_config_file=args.gate_config,
            output_root=args.output_root,
        )
    except Exception as exc:
        print(f"PILOT_EVALUATION_FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    compact = {
        mode: {
            "point_hit": values["point_hit_count"],
            "safe_hit": values["safe_eroded_hit_count"],
            "false_accept": values["false_accept_count"],
            "false_reject": values["false_reject_count"],
            "task_success": values["task_success_count"],
        }
        for mode, values in summary["mode_summary"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
