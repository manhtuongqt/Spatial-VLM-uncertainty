from __future__ import annotations

from typing import Any, Mapping

from .calibration import apply_calibrator
from .postprocess import highest_density_region


def decide(row: Mapping[str, Any], calibrator: Mapping[str, Any]) -> dict[str, Any]:
    risk = apply_calibrator(row, calibrator)
    answer = max(row["answerability_probabilities"], key=row["answerability_probabilities"].get)
    sources = row["source_probabilities"]
    threshold = calibrator["risk_policy"].get("threshold")
    if answer == "ABSENT":
        action, reason = "ABSTAIN", "target predicted absent"
    elif answer == "AMBIGUOUS":
        action, reason = "ASK_USER", "query predicted ambiguous"
    elif answer == "INSUFFICIENT_EVIDENCE":
        action, reason = "REOBSERVE", "insufficient visual evidence"
    elif threshold is not None and risk <= float(threshold):
        action, reason = "EXECUTE", "FOUND and calibrated risk passes"
    elif sources["semantic"] >= max(sources["depth"], sources["occlusion"]):
        action, reason = "ASK_USER", "semantic/query risk dominates"
    else:
        action, reason = "REOBSERVE", "visual/depth/occlusion risk dominates"
    region_mass = calibrator.get("conformal", {}).get("probability_mass")
    region = None
    if region_mass is not None and "probability_grid" in row["spatial"]:
        region = highest_density_region(row["spatial"]["probability_grid"], float(region_mass))
    threshold_rows = calibrator.get("source_thresholds", {})
    active_sources = [
        name for name, probability in sources.items()
        if probability >= float(threshold_rows.get(name, {}).get("threshold", 0.5))
    ]
    return {
        "sample_id": row["sample_id"],
        "action": action,
        "reason": reason,
        "calibrated_grounding_risk": risk,
        "risk_threshold": threshold,
        "selected_point_xy": row["spatial"]["map_pixel_xy"] if action == "EXECUTE" else None,
        "confidence_region": region,
        "active_uncertainty_sources": active_sources,
    }
