#!/usr/bin/env python3
"""Paired Test-IID grounding comparison with family-cluster bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
TEST_ROOT = ROOT / "new/test_iid"
DATASET_ROOT = TEST_ROOT / "dataset"
DEFAULT_B1 = TEST_ROOT / "evaluation/roborefer_original/predictions.jsonl"
DEFAULT_V2 = TEST_ROOT / "evaluation/best_v2/predictions.jsonl"
DEFAULT_TRUTH = TEST_ROOT / "protocol/test_iid_eval_manifest.json"
DEFAULT_OUTPUT = TEST_ROOT / "evaluation/comparison_original_vs_best_v2"
CHECKPOINT = ROOT / "new/outputs/pcrau_target_v2_full_seed_24082026/checkpoints/best/model.safetensors"
EXPECTED_V2_SHA256 = "1505152fca7b725d7db701c1341ad1a57049d6b5ac450a9f5ce0c8d78705ba26"
EXPECTED_B1_SHA256 = "5508fb49888c1f62b875d928266f7fff3e1224c795f6fc8b971631143a45a7fa"
IMAGE_DIAGONAL = 800.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mask(relative: str) -> np.ndarray:
    path = (DATASET_ROOT / relative).resolve()
    if DATASET_ROOT.resolve() not in path.parents:
        raise RuntimeError(f"Evaluator path escapes Test-IID: {relative}")
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (480, 640):
        raise RuntimeError(f"Invalid evaluator mask: {path}")
    return mask > 0


def score_point(point: list[int] | None, mask: np.ndarray) -> tuple[int, float | None]:
    if point is None or len(point) != 2:
        return 0, None
    x, y = (int(v) for v in point)
    if not (0 <= x < 640 and 0 <= y < 480):
        return 0, None
    hit = int(mask[y, x])
    outside = (~mask).astype(np.uint8)
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
    return hit, float(distance[y, x])


def ratio(success: int, total: int) -> dict[str, Any]:
    return {"success": int(success), "total": int(total), "rate": success / total if total else None}


def ci_record(draws: np.ndarray, estimate: float) -> dict[str, float]:
    finite = draws[np.isfinite(draws)]
    if not len(finite):
        return {"estimate": estimate, "ci95_low": math.nan, "ci95_high": math.nan}
    return {
        "estimate": float(estimate),
        "ci95_low": float(np.percentile(finite, 2.5)),
        "ci95_high": float(np.percentile(finite, 97.5)),
    }


def family_bootstrap(
    rows: list[dict[str, Any]],
    families: list[str],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Resample 200 family clusters; retain all eligible variants in each draw."""

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family_id"]].append(row)
    statistics = ("b1_hit", "v2_hit", "b1_interior", "v2_interior")
    sums = {name: np.asarray([sum(r[name] for r in by_family[f]) for f in families], dtype=float) for name in statistics}
    counts = np.asarray([len(by_family[f]) for f in families], dtype=float)
    b1_distance_sums = np.asarray([
        sum(float(r["b1_distance_px"]) for r in by_family[f] if r["b1_distance_px"] is not None)
        for f in families
    ])
    b1_distance_counts = np.asarray([
        sum(r["b1_distance_px"] is not None for r in by_family[f]) for f in families
    ], dtype=float)
    v2_distance_sums = np.asarray([
        sum(float(r["v2_distance_px"]) for r in by_family[f]) for f in families
    ])
    paired_distance_delta_sums = np.asarray([
        sum(
            float(r["v2_distance_px"] - r["b1_distance_px"])
            for r in by_family[f] if r["b1_distance_px"] is not None
        )
        for f in families
    ])
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(replicates, dtype=float) for name in statistics}
    draws["target_delta"] = np.empty(replicates, dtype=float)
    draws["interior_delta"] = np.empty(replicates, dtype=float)
    draws["b1_mean_distance_px"] = np.empty(replicates, dtype=float)
    draws["v2_mean_distance_px"] = np.empty(replicates, dtype=float)
    draws["paired_mean_distance_delta_px"] = np.empty(replicates, dtype=float)
    for i in range(replicates):
        selected = rng.integers(0, len(families), size=len(families))
        denominator = counts[selected].sum()
        if denominator == 0:
            for name in draws:
                draws[name][i] = np.nan
            continue
        for name in statistics:
            draws[name][i] = sums[name][selected].sum() / denominator
        draws["target_delta"][i] = draws["v2_hit"][i] - draws["b1_hit"][i]
        draws["interior_delta"][i] = draws["v2_interior"][i] - draws["b1_interior"][i]
        parsed_denominator = b1_distance_counts[selected].sum()
        draws["b1_mean_distance_px"][i] = b1_distance_sums[selected].sum() / parsed_denominator if parsed_denominator else np.nan
        draws["v2_mean_distance_px"][i] = v2_distance_sums[selected].sum() / denominator
        draws["paired_mean_distance_delta_px"][i] = paired_distance_delta_sums[selected].sum() / parsed_denominator if parsed_denominator else np.nan
    n = len(rows)
    result = {
        "unit": "family_cluster",
        "families_resampled_per_replicate": len(families),
        "families_with_eligible_samples": len(by_family),
        "variants_kept_together": True,
        "replicates": replicates,
        "seed": seed,
        "eligible_samples": n,
    }
    for name in statistics:
        result[name] = ci_record(draws[name], sum(r[name] for r in rows) / n)
    result["target_delta_v2_minus_b1"] = ci_record(
        draws["target_delta"], np.mean([r["v2_hit"] - r["b1_hit"] for r in rows])
    )
    result["interior_delta_v2_minus_b1"] = ci_record(
        draws["interior_delta"], np.mean([r["v2_interior"] - r["b1_interior"] for r in rows])
    )
    parsed = [r for r in rows if r["b1_distance_px"] is not None]
    result["b1_mean_distance_px_parsed_only"] = ci_record(
        draws["b1_mean_distance_px"], np.mean([r["b1_distance_px"] for r in parsed]) if parsed else math.nan
    )
    result["v2_mean_distance_px"] = ci_record(
        draws["v2_mean_distance_px"], np.mean([r["v2_distance_px"] for r in rows])
    )
    result["paired_mean_distance_delta_px_v2_minus_b1"] = ci_record(
        draws["paired_mean_distance_delta_px"],
        np.mean([r["v2_distance_px"] - r["b1_distance_px"] for r in parsed]) if parsed else math.nan,
    )
    return result


def family_sign_flip_test(rows: list[dict[str, Any]], families: list[str], permutations: int, seed: int) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family_id"]].append(row)
    cluster_delta_sum = np.asarray(
        [sum(r["v2_hit"] - r["b1_hit"] for r in by_family[f]) for f in families], dtype=float
    )
    denominator = len(rows)
    observed = float(cluster_delta_sum.sum() / denominator)
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    while completed < permutations:
        size = min(5000, permutations - completed)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(size, len(families)))
        permuted = signs @ cluster_delta_sum / denominator
        extreme += int(np.count_nonzero(np.abs(permuted) >= abs(observed) - 1e-15))
        completed += size
    return {
        "test": "paired_family_cluster_sign_flip_two_sided",
        "unit": "family_cluster",
        "families": len(families),
        "permutations": permutations,
        "seed": seed,
        "observed_delta": observed,
        "p_value": (extreme + 1) / (permutations + 1),
    }


def summarize(rows: list[dict[str, Any]], bootstrap: dict[str, Any]) -> dict[str, Any]:
    total = len(rows)
    both = sum(r["b1_hit"] and r["v2_hit"] for r in rows)
    b1_only = sum(r["b1_hit"] and not r["v2_hit"] for r in rows)
    v2_only = sum(not r["b1_hit"] and r["v2_hit"] for r in rows)
    neither = total - both - b1_only - v2_only
    b1_distances = [r["b1_distance_px"] for r in rows if r["b1_distance_px"] is not None]
    v2_distances = [r["v2_distance_px"] for r in rows]
    return {
        "samples": total,
        "families_represented": len({r["family_id"] for r in rows}),
        "original_b1": {
            "point_in_target": ratio(sum(r["b1_hit"] for r in rows), total),
            "point_in_interior": ratio(sum(r["b1_interior"] for r in rows), total),
            "parsed_points": ratio(len(b1_distances), total),
            "point_to_target_distance_px_parsed_only": {
                "count": len(b1_distances),
                "mean": float(np.mean(b1_distances)) if b1_distances else None,
                "median": float(np.median(b1_distances)) if b1_distances else None,
                "p95": float(np.percentile(b1_distances, 95)) if b1_distances else None,
                "maximum": float(np.max(b1_distances)) if b1_distances else None,
                "mean_normalized_image_diagonal": float(np.mean(b1_distances) / IMAGE_DIAGONAL) if b1_distances else None,
            },
        },
        "best_v2": {
            "point_in_target": ratio(sum(r["v2_hit"] for r in rows), total),
            "point_in_interior": ratio(sum(r["v2_interior"] for r in rows), total),
            "parsed_points": ratio(total, total),
            "point_to_target_distance_px": {
                "count": len(v2_distances),
                "mean": float(np.mean(v2_distances)),
                "median": float(np.median(v2_distances)),
                "p95": float(np.percentile(v2_distances, 95)),
                "maximum": float(np.max(v2_distances)),
                "mean_normalized_image_diagonal": float(np.mean(v2_distances) / IMAGE_DIAGONAL),
            },
        },
        "delta_v2_minus_original": {
            "point_in_target": float(np.mean([r["v2_hit"] - r["b1_hit"] for r in rows])),
            "point_in_interior": float(np.mean([r["v2_interior"] - r["b1_interior"] for r in rows])),
        },
        "paired_target_outcomes": {
            "both_hit": both,
            "original_only": b1_only,
            "v2_only": v2_only,
            "neither": neither,
        },
        "family_cluster_bootstrap_95ci": bootstrap,
    }


def pct(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"


def pp(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{100 * value:+.{digits}f} pp"


def ci_text(record: dict[str, float], *, delta: bool = False) -> str:
    render: Callable[[float], str] = pp if delta else pct
    return f"{render(record['estimate'])} [{render(record['ci95_low'])}, {render(record['ci95_high'])}]"


def run(args: argparse.Namespace) -> None:
    b1_path, v2_path, truth_path = (Path(args.b1).resolve(), Path(args.v2).resolve(), Path(args.truth).resolve())
    output_dir = Path(args.output_dir).resolve()
    truth_manifest = json.loads(truth_path.read_text(encoding="utf-8"))
    b1_rows, v2_rows = read_jsonl(b1_path), read_jsonl(v2_path)
    truth_rows = truth_manifest["entries"]
    if len(b1_rows) != 1000 or len(v2_rows) != 1000 or len(truth_rows) != 1000:
        raise RuntimeError(f"Expected 1,000 paired rows: B1={len(b1_rows)}, V2={len(v2_rows)}, truth={len(truth_rows)}")
    if truth_manifest.get("family_count") != 200 or truth_manifest.get("split") != "test_iid":
        raise RuntimeError("Evaluator manifest is not the locked 200-family Test-IID set")
    if sha256_file(CHECKPOINT) != EXPECTED_V2_SHA256:
        raise RuntimeError("Best V2 checkpoint hash changed")
    b1 = {r["sample_id"]: r for r in b1_rows}
    v2 = {r["sample_id"]: r for r in v2_rows}
    truth = {r["sample_id"]: r for r in truth_rows}
    if not (len(b1) == len(v2) == len(truth) == 1000 and set(b1) == set(v2) == set(truth)):
        raise RuntimeError("B1, V2, and evaluator sample identities are not exactly paired")
    if {r.get("model_inventory_sha256") for r in b1_rows} != {EXPECTED_B1_SHA256}:
        raise RuntimeError("Original RoboRefer checkpoint identity mismatch")

    paired: list[dict[str, Any]] = []
    for sample_id in sorted(truth):
        entry = truth[sample_id]
        supervision = entry["supervision"]
        target_mask = load_mask(supervision["target_mask_path"])
        target_exists = bool(target_mask.any())
        interior_mask = load_mask(supervision["target_interior_mask_path"]) if target_exists else target_mask
        b1_points = b1[sample_id].get("pixel_points_xy", [])
        b1_point = b1_points[0] if len(b1_points) == 1 else None
        v2_point = v2[sample_id]["spatial"]["map_pixel_xy"]
        if target_exists:
            b1_hit, b1_distance = score_point(b1_point, target_mask)
            v2_hit, v2_distance = score_point(v2_point, target_mask)
            b1_interior, _ = score_point(b1_point, interior_mask)
            v2_interior, _ = score_point(v2_point, interior_mask)
            assert v2_distance is not None
        else:
            b1_hit = v2_hit = b1_interior = v2_interior = 0
            b1_distance = v2_distance = None
        audit = entry["audit_only"]
        paired.append({
            "sample_id": sample_id,
            "family_id": entry["family_id"],
            "variant": entry["variant"],
            "answerability_state": supervision["answerability_state"],
            "target_exists": target_exists,
            "relation": audit["relation"],
            "family_category": audit["family_category"],
            "target_object_group": audit["target_object_group"],
            "b1_parse_status": b1[sample_id]["parse_status"],
            "b1_x": b1_point[0] if b1_point else None,
            "b1_y": b1_point[1] if b1_point else None,
            "v2_x": int(v2_point[0]),
            "v2_y": int(v2_point[1]),
            "b1_hit": b1_hit,
            "v2_hit": v2_hit,
            "b1_interior": b1_interior,
            "v2_interior": v2_interior,
            "b1_distance_px": b1_distance,
            "v2_distance_px": v2_distance,
        })

    families = sorted({r["family_id"] for r in paired})
    if len(families) != 200 or any(sum(r["family_id"] == f for r in paired) != 5 for f in families):
        raise RuntimeError("Test-IID does not contain exactly five linked variants per family")
    target_present = [r for r in paired if r["target_exists"]]
    truth_found = [r for r in paired if r["answerability_state"] == "FOUND"]
    if not all(r["target_exists"] for r in truth_found):
        raise RuntimeError("A truth-FOUND row has no target mask")

    target_bootstrap = family_bootstrap(target_present, families, args.bootstrap_replicates, args.seed)
    found_bootstrap = family_bootstrap(truth_found, families, args.bootstrap_replicates, args.seed + 1)
    primary = summarize(target_present, target_bootstrap)
    found = summarize(truth_found, found_bootstrap)
    sign_flip = family_sign_flip_test(target_present, families, args.permutations, args.seed + 2)

    stratified: dict[str, dict[str, Any]] = {}
    for field in ("variant", "answerability_state", "relation", "family_category", "target_object_group"):
        stratified[field] = {}
        for value in sorted({str(r[field]) for r in target_present}):
            subset = [r for r in target_present if str(r[field]) == value]
            boot = family_bootstrap(subset, families, args.bootstrap_replicates, args.seed + 100 + len(stratified[field]))
            stratified[field][value] = summarize(subset, boot)

    report = {
        "schema_version": 1,
        "status": "OFFICIAL_FROZEN_TEST_IID_PAIRED_COMPARISON_COMPLETE",
        "protocol_id": truth_manifest["protocol_id"],
        "sample_count": 1000,
        "family_count": 200,
        "variants_per_family": 5,
        "independence_and_ci_policy": {
            "independent_unit": "family",
            "independent_units": 200,
            "variant_rows_are_not_treated_as_independent": True,
            "bootstrap_method": "nonparametric percentile cluster bootstrap",
            "bootstrap_replicates": args.bootstrap_replicates,
            "confidence_level": 0.95,
            "bootstrap_seed": args.seed,
        },
        "artifacts": {
            "roborefer_original_predictions": str(b1_path.relative_to(ROOT)),
            "roborefer_original_predictions_sha256": sha256_file(b1_path),
            "best_v2_predictions": str(v2_path.relative_to(ROOT)),
            "best_v2_predictions_sha256": sha256_file(v2_path),
            "test_iid_evaluator_manifest": str(truth_path.relative_to(ROOT)),
            "test_iid_evaluator_manifest_sha256": sha256_file(truth_path),
        },
        "models": {
            "roborefer_original": {
                "name": "RoboRefer-2B-SFT",
                "input": "registered RGB-D",
                "generation": "greedy",
                "seed": 8132026,
                "model_inventory_sha256": EXPECTED_B1_SHA256,
            },
            "best_v2": {
                "name": "P-CRA-U best V2 sidecar over frozen RoboRefer features",
                "epoch": 13,
                "global_step": 1120,
                "checkpoint_sha256": EXPECTED_V2_SHA256,
            },
        },
        "primary_target_present_grounding": primary,
        "truth_found_grounding": found,
        "paired_family_sign_flip_test_primary_delta": sign_flip,
        "roborefer_original_all_1000_parse_status": dict(Counter(r["parse_status"] for r in b1_rows)),
        "stratified_target_present_grounding": stratified,
        "scope_notes": [
            "Primary grounding is point-in-target over every sample with a non-empty target mask.",
            "Truth-FOUND grounding is also reported because these samples are unambiguously answerable.",
            "A RoboRefer response without exactly one valid normalized point is scored as a grounding miss.",
            "Distance statistics for RoboRefer are parsed-point-only; hit-rate denominators retain parse failures.",
            "All 95% intervals resample 200 families and keep each family's five variants together.",
            "This is image-space Test-IID grounding, not physical robot task success.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "full_comparison.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = output_dir / "paired_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    strata_path = output_dir / "stratified_grounding.csv"
    with strata_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["stratum", "value", "samples", "families", "b1_hits", "b1_rate", "b1_ci_low", "b1_ci_high", "v2_hits", "v2_rate", "v2_ci_low", "v2_ci_high", "delta", "delta_ci_low", "delta_ci_high"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for field, values in stratified.items():
            for value, data in values.items():
                boot = data["family_cluster_bootstrap_95ci"]
                writer.writerow({
                    "stratum": field, "value": value, "samples": data["samples"],
                    "families": data["families_represented"],
                    "b1_hits": data["original_b1"]["point_in_target"]["success"],
                    "b1_rate": data["original_b1"]["point_in_target"]["rate"],
                    "b1_ci_low": boot["b1_hit"]["ci95_low"], "b1_ci_high": boot["b1_hit"]["ci95_high"],
                    "v2_hits": data["best_v2"]["point_in_target"]["success"],
                    "v2_rate": data["best_v2"]["point_in_target"]["rate"],
                    "v2_ci_low": boot["v2_hit"]["ci95_low"], "v2_ci_high": boot["v2_hit"]["ci95_high"],
                    "delta": data["delta_v2_minus_original"]["point_in_target"],
                    "delta_ci_low": boot["target_delta_v2_minus_b1"]["ci95_low"],
                    "delta_ci_high": boot["target_delta_v2_minus_b1"]["ci95_high"],
                })

    def table_row(label: str, data: dict[str, Any]) -> str:
        boot = data["family_cluster_bootstrap_95ci"]
        b1_ratio, v2_ratio = data["original_b1"]["point_in_target"], data["best_v2"]["point_in_target"]
        return (
            f"| {label} | {data['samples']} | "
            f"{b1_ratio['success']}/{b1_ratio['total']} ({ci_text(boot['b1_hit'])}) | "
            f"{v2_ratio['success']}/{v2_ratio['total']} ({ci_text(boot['v2_hit'])}) | "
            f"{ci_text(boot['target_delta_v2_minus_b1'], delta=True)} |"
        )

    lines = [
        "# RoboRefer gốc vs best V2 — Test-IID",
        "",
        "So sánh ghép cặp trên cùng 1.000 sample thuộc **200 family độc lập**. CI 95% là percentile cluster bootstrap; mỗi lần lặp lấy lại 200 family và luôn giữ 5 variant của family đi cùng nhau.",
        "",
        "| Phạm vi grounding | n sample | RoboRefer gốc | Best V2 | V2 − gốc |",
        "|---|---:|---:|---:|---:|",
        table_row("Target hiện hữu (chính)", primary),
        table_row("Ground-truth FOUND", found),
        "",
        "## Kết quả ghép cặp chính",
        "",
        f"- Cùng đúng: {primary['paired_target_outcomes']['both_hit']}; chỉ RoboRefer đúng: {primary['paired_target_outcomes']['original_only']}; chỉ V2 đúng: {primary['paired_target_outcomes']['v2_only']}; cùng sai: {primary['paired_target_outcomes']['neither']}.",
        f"- Kiểm định hoán vị sign-flip theo family: p = {sign_flip['p_value']:.6g} ({sign_flip['permutations']:,} lần).",
        f"- RoboRefer parse hợp lệ trên toàn bộ Test-IID: {sum(r['parse_status'] in {'EXACT_ONE_NORMALIZED_POINT', 'FORMAT_VIOLATION'} for r in b1_rows)}/1000; phân bố {dict(Counter(r['parse_status'] for r in b1_rows))}.",
        "",
        "## Interior grounding",
        "",
        f"- Target hiện hữu — RoboRefer: {primary['original_b1']['point_in_interior']['success']}/{primary['samples']} ({ci_text(target_bootstrap['b1_interior'])}); V2: {primary['best_v2']['point_in_interior']['success']}/{primary['samples']} ({ci_text(target_bootstrap['v2_interior'])}); chênh lệch {ci_text(target_bootstrap['interior_delta_v2_minus_b1'], delta=True)}.",
        f"- Truth FOUND — RoboRefer: {found['original_b1']['point_in_interior']['success']}/{found['samples']} ({ci_text(found_bootstrap['b1_interior'])}); V2: {found['best_v2']['point_in_interior']['success']}/{found['samples']} ({ci_text(found_bootstrap['v2_interior'])}); chênh lệch {ci_text(found_bootstrap['interior_delta_v2_minus_b1'], delta=True)}.",
        "",
        "## Khoảng cách tới target",
        "",
        f"- Target hiện hữu, RoboRefer (chỉ point parse được): mean {primary['original_b1']['point_to_target_distance_px_parsed_only']['mean']:.3f}px (95% CI {target_bootstrap['b1_mean_distance_px_parsed_only']['ci95_low']:.3f}–{target_bootstrap['b1_mean_distance_px_parsed_only']['ci95_high']:.3f}), median {primary['original_b1']['point_to_target_distance_px_parsed_only']['median']:.3f}px, p95 {primary['original_b1']['point_to_target_distance_px_parsed_only']['p95']:.3f}px, max {primary['original_b1']['point_to_target_distance_px_parsed_only']['maximum']:.3f}px.",
        f"- Target hiện hữu, V2: mean {primary['best_v2']['point_to_target_distance_px']['mean']:.3f}px (95% CI {target_bootstrap['v2_mean_distance_px']['ci95_low']:.3f}–{target_bootstrap['v2_mean_distance_px']['ci95_high']:.3f}), median {primary['best_v2']['point_to_target_distance_px']['median']:.3f}px, p95 {primary['best_v2']['point_to_target_distance_px']['p95']:.3f}px, max {primary['best_v2']['point_to_target_distance_px']['maximum']:.3f}px.",
        f"- Chênh lệch mean distance ghép cặp V2 − RoboRefer (chỉ các cặp parse được): {target_bootstrap['paired_mean_distance_delta_px_v2_minus_b1']['estimate']:+.3f}px (95% CI {target_bootstrap['paired_mean_distance_delta_px_v2_minus_b1']['ci95_low']:+.3f} đến {target_bootstrap['paired_mean_distance_delta_px_v2_minus_b1']['ci95_high']:+.3f}px).",
        "",
        "## Đầy đủ point-in-target theo phân tầng target-present",
        "",
        "| Phân tầng | Giá trị | n | family | RoboRefer gốc (95% CI) | Best V2 (95% CI) | V2 − gốc (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for field, values in stratified.items():
        for value, data in values.items():
            boot = data["family_cluster_bootstrap_95ci"]
            b1_ratio = data["original_b1"]["point_in_target"]
            v2_ratio = data["best_v2"]["point_in_target"]
            lines.append(
                f"| {field} | {value} | {data['samples']} | {data['families_represented']} | "
                f"{b1_ratio['success']}/{b1_ratio['total']} ({ci_text(boot['b1_hit'])}) | "
                f"{v2_ratio['success']}/{v2_ratio['total']} ({ci_text(boot['v2_hit'])}) | "
                f"{ci_text(boot['target_delta_v2_minus_b1'], delta=True)} |"
            )
    lines.extend([
        "",
        "Từng cặp sample và số liệu máy đọc nằm trong `paired_samples.csv`, `stratified_grounding.csv` và `full_comparison.json`.",
    ])
    md_path = output_dir / "SUMMARY.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "primary": primary,
        "truth_found": found,
        "paired_test": sign_flip,
        "outputs": {"json": str(json_path), "summary": str(md_path), "paired_csv": str(csv_path), "stratified_csv": str(strata_path)},
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b1", default=str(DEFAULT_B1))
    parser.add_argument("--v2", default=str(DEFAULT_V2))
    parser.add_argument("--truth", default=str(DEFAULT_TRUTH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--permutations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=24082027)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
