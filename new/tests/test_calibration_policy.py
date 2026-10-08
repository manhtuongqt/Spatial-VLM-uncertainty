from __future__ import annotations

from pcrau.calibration import apply_calibrator, fit_calibrator
from pcrau.policy import decide
from pcrau.utils import load_config


def make_row(index: int) -> dict:
    error = index % 4 == 0
    found = 0.15 if error else 0.9
    return {
        "sample_id": f"sample_{index}",
        "family_id": f"family_{index // 5}",
        "variant": "clean",
        "spatial": {"entropy_normalized": 0.9 if error else 0.1, "peak_margin": 0.01 if error else 0.7, "mode_count": 3 if error else 1, "map_pixel_xy": [10, 20], "probability_grid": [[0.6, 0.3], [0.05, 0.05]]},
        "answerability_probabilities": {"FOUND": found, "AMBIGUOUS": 0.3 if error else 0.03, "ABSENT": 0.3 if error else 0.03, "INSUFFICIENT_EVIDENCE": 0.25 if error else 0.04},
        "source_probabilities": {"semantic": 0.6 if error else 0.1, "relation": 0.4 if error else 0.1, "spatial": 0.7 if error else 0.1, "depth": 0.3, "occlusion": 0.2},
        "relation_consistency": 0.2 if error else 0.9,
        "fusion_gate_mean": 0.5,
        "rgb_depth_cosine": 0.2 if error else 0.8,
        "rgb_depth_mae": 0.8 if error else 0.1,
        "roborefer_disagreement_normalized": None,
        "evaluation": {
            "error_event": error, "target_exists": True,
            "target_probability_mass": 0.2 if error else 0.8,
            "target_required_hd_mass": 0.9 if error else 0.6,
            "source_multihot_5": [int(error), int(error), int(error), 0, 0],
        },
    }


def test_calibrator_and_policy_are_serializable() -> None:
    config = load_config()
    config["calibration"]["minimum_accepted_families"] = 1
    config["calibration"]["risk_target"] = 1.0
    rows = [make_row(index) for index in range(50)]
    calibrator = fit_calibrator(rows, config)
    assert set(calibrator["source_thresholds"]) == {"semantic", "relation", "spatial", "depth", "occlusion"}
    risk = apply_calibrator(rows[1], calibrator)
    assert 0.0 <= risk <= 1.0
    decision = decide(rows[1], calibrator)
    assert decision["action"] in {"EXECUTE", "REOBSERVE", "ASK_USER", "ABSTAIN"}
