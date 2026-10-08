"""Development-only G1 power sensitivity; no Test/Calibration access.

This does not choose a scientific non-inferiority margin. It tabulates the
consequences of candidate margins before a numeric lock is issued.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from .audit_development import ROOT, read_jsonl, sha256


SOURCE = ROOT / "results/spatial_vlm_refspatial_v1/wp5_spatial_uncertainty/wp5_scored_predictions.jsonl"
OUT = ROOT / "ketqua1/00_quan_tri_khoa/ngay_03/POWER_SENSITIVITY_REVISION_V3.json"
STATES = ("FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE")


def wilson(numerator: int, denominator: int, z: float = 1.959963984540054) -> list[float]:
    p = numerator / denominator
    scale = 1 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / scale
    radius = z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator**2)) / scale
    return [center - radius, center + radius]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = [row for row in read_jsonl(SOURCE) if row.get("split") == "val_uq"]
    if len(rows) != 64 or Counter(row["answerability_state"] for row in rows) != {
        state: 16 for state in STATES
    } or len({row["family_id"] for row in rows}) != 64:
        raise ValueError("Development val inventory drift")
    found = [row for row in rows if row["answerability_state"] == "FOUND"]
    nonfound = [row for row in rows if row["answerability_state"] != "FOUND"]
    hit = sum(bool(row["grounding_hit_at_008"]) for row in found)
    false_accept = sum(row["b0_action"] == "POINT" for row in nonfound)
    paired = []
    pair_counts = Counter()
    for row in rows:
        unsafe = bool(row["unsafe"])
        hit008 = bool(row["grounding_hit_at_008"])
        b0 = int((row["b0_action"] == "ABSTAIN") if unsafe else
                 (row["b0_action"] == "POINT" and hit008))
        v2 = int((row["proposed_action"] == "ABSTAIN") if unsafe else
                 (row["proposed_action"] == "POINT" and hit008))
        paired.append(v2 - b0)
        pair_counts[f"{b0}->{v2}"] += 1
    rng = np.random.default_rng(24092026)
    draws = 10000
    values = np.asarray(paired, dtype=np.float64)
    boot = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    zsum2 = (NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.8)) ** 2
    scenarios = []
    for margin in (0.05, 0.10, 0.15):
        # D in [-1,1]; Var(D) <= 1. Two-sided alpha .05 is conservative for NI.
        required_endpoint_families = math.ceil(zsum2 / margin**2)
        scenarios.append({
            "absolute_margin_or_effect_gap": margin,
            "paired_binary_endpoint_families_worst_variance": required_endpoint_families,
            "balanced_four_state_total_if_endpoint_found_only": 4 * required_endpoint_families,
            "pilot_found_independent_family_shortfall": required_endpoint_families - 64,
            "normal_approximation_only": True,
        })
    result = {
        "schema_version": "1.0", "status": "DEVELOPMENT_SENSITIVITY_NOT_A_MARGIN_LOCK",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(SOURCE.relative_to(ROOT)), "source_sha256": sha256(SOURCE),
        "unit": "parent_family", "val_families": 64,
        "val_families_per_state": {state: 16 for state in STATES},
        "b0_found_hit_at_008": {"numerator": hit, "denominator": 16,
                                 "wilson_95": wilson(hit, 16)},
        "b0_nonfound_false_accept": {"numerator": false_accept, "denominator": 48,
                                       "wilson_95": wilson(false_accept, 48)},
        "historical_v2_safe_task_proxy": {
            "paired_counts": dict(sorted(pair_counts.items())),
            "mean_v2_minus_b0": float(values.mean()),
            "sample_variance": float(values.var(ddof=1)),
            "family_bootstrap_draws": draws, "bootstrap_seed": 24092026,
            "percentile_95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "not_a_v3_effect_estimate": True,
        },
        "design_assumptions": {"alpha_two_sided": 0.05, "target_power": 0.8,
                               "worst_case_paired_difference_variance": 1.0,
                               "zsum_squared": zsum2},
        "sensitivity": scenarios,
        "precision_reference": {"independent_families_per_class_for_95pct_halfwidth_0_07":
                                math.ceil(1.96**2 * 0.25 / 0.07**2)},
        "limitations": [
            "A margin is a scientific tolerance, not inferred from the B0 result or chosen to fit 512 images",
            "This bounded-difference calculation is not exact power for macro-F1 or Gaussian NLL",
            "No v3 predictions exist before Day 4; historical v2 proxy cannot stand in for v3 effect",
            "Test IID/OOD remain sealed and no test family has been selected",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "val_families": 64,
                      "b0_hit": f"{hit}/16", "b0_false_accept": f"{false_accept}/48",
                      "proxy_ci95": result["historical_v2_safe_task_proxy"]["percentile_95"]}))


if __name__ == "__main__":
    main()
