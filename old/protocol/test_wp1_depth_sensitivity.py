"""CPU-only contract tests for the preregistered WP1 pipeline."""

from pathlib import Path

import cv2
import numpy as np

from wp1_common import (
    CONDITIONS,
    decision_from_metrics,
    flat_depth_view,
    inverted_depth_view,
    localized_holes_edge_view,
    normalize_metric_depth,
    parse_point,
    pilot_tree_digest,
    sha256_bytes,
    shuffled_source_scene,
)


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results/roborefer_pilot_v0_20260813_173305"


def test_condition_set_is_exactly_six_and_ordered():
    assert CONDITIONS == (
        "rgb_only", "correct_depth", "flat_depth", "shuffled_depth",
        "inverted_depth", "localized_holes_edge",
    )


def test_correct_depth_transform_matches_locked_pilot_request_bytes():
    depth = np.load(PILOT / "dataset_attempt_02/pilot_scene_0001/input/depth_m.npy")
    view = normalize_metric_depth(depth)
    success, encoded = cv2.imencode(".png", view)
    assert success
    locked = PILOT / "predictions/pilot_scene_0001/request_depth_registered_view.png"
    assert sha256_bytes(encoded.tobytes()) == sha256_bytes(locked.read_bytes())


def test_flat_and_inverted_transforms_have_locked_semantics():
    source = np.arange(3 * 12 * 16, dtype=np.uint8).reshape(12, 16, 3)
    assert np.all(flat_depth_view(source) == 127)
    assert np.array_equal(inverted_depth_view(source), 255 - source)


def test_localized_corruption_is_deterministic_and_bounded():
    source = np.tile(np.arange(120, dtype=np.uint8), (100, 1))
    source = np.repeat(source[..., None], 3, axis=-1)
    first, metadata = localized_holes_edge_view(source)
    second, other = localized_holes_edge_view(source)
    assert np.array_equal(first, second)
    assert metadata == other
    assert 0.01 < metadata["corrupted_fraction"] < 0.60
    assert np.count_nonzero(first == 0) > np.count_nonzero(source == 0)


def test_shuffled_mapping_is_a_cyclic_derangement():
    scenes = [f"scene_{index}" for index in range(10)]
    mapped = [shuffled_source_scene(scenes, item) for item in scenes]
    assert len(set(mapped)) == 10
    assert all(source != target for source, target in zip(scenes, mapped))
    assert mapped[-1] == scenes[0]


def test_strict_point_parser_contract():
    parsed = parse_point("[(0.25, 0.75)]", 640, 480)
    assert parsed["parse_status"] == "EXACT_ONE_NORMALIZED_POINT"
    assert parsed["pixel_points_xy"] == [[160, 359]]
    assert parse_point("point=[(0.25, 0.75)]", 640, 480)["parse_status"] == "NO_POINT"
    assert parse_point("[(1.2, 0.4)]", 640, 480)["parse_status"] == "OUT_OF_RANGE"


def metrics(correct_parse=1.0, correct_hits=8, rgb_hits=8, sensitive=0.5, switches=0.2):
    return {
        "condition_summary": {
            "correct_depth": {"strict_parse_rate": correct_parse, "point_hit_rate_single_targets": correct_hits / 8, "point_hit_count": correct_hits},
            "rgb_only": {"point_hit_count": rgb_hits},
        },
        "corruption_aggregate": {"displacement_sensitive_rate": sensitive, "instance_switch_rate": switches},
    }


THRESHOLDS = {
    "required_correct_depth_parse_rate": 1.0,
    "minimum_correct_depth_point_hit_rate": 0.75,
    "maximum_correct_depth_hit_deficit_vs_rgb": 1,
    "go_pcra_f_max_sensitive_pair_rate": 0.20,
    "go_pcra_f_max_instance_switch_rate": 0.10,
}


def test_decision_rule_is_exhaustive_and_preregistered():
    assert decision_from_metrics(metrics(correct_parse=0.9), THRESHOLDS)[0] == "FIX_DEPTH_PIPELINE_FIRST"
    assert decision_from_metrics(metrics(correct_hits=5), THRESHOLDS)[0] == "FIX_DEPTH_PIPELINE_FIRST"
    assert decision_from_metrics(metrics(sensitive=0.20, switches=0.10), THRESHOLDS)[0] == "GO_PCRA_F"
    assert decision_from_metrics(metrics(sensitive=0.21, switches=0.10), THRESHOLDS)[0] == "NO_GO_PCRA_F"


def test_pilot_digest_matches_wp0_lock():
    assert pilot_tree_digest(ROOT, "results/roborefer_pilot_v0_20260813_173305") == "df8ec334c82b14447463101cccb5e54836fb27e579a760322256f84d157a2b9c"

