"""Train/dev evaluator for the locked anchor pilot; no inference oracle inputs."""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np

ARMS = ("M0", "M1", "M2")


def spatial_summary(logits):
    flat = np.asarray(logits, dtype=np.float64).reshape(-1)
    if not np.isfinite(flat).all():
        raise ValueError("Nonfinite spatial prediction")
    h, w = logits.shape
    y, x = divmod(int(flat.argmax()), w)
    probability = np.exp(flat - flat.max()); probability /= probability.sum()
    sigmoid = 1 / (1 + np.exp(-np.clip(flat, -700, 700)))
    return {"map_grid_xy": [x, y], "map_pixel_xy": [int((x + .5) * 640 / w), int((y + .5) * 480 / h)],
        "sigmoid_max": float(sigmoid.max()), "sigmoid_mean": float(sigmoid.mean()),
        "softmax_peak": float(probability.max()),
        "entropy_normalized": float(-(probability * np.log(probability.clip(1e-300))).sum() / math.log(h*w))}


def score_predictions(predictions, labels):
    """Join stored observable traces to evaluator-only labels after inference."""
    rows = []
    for pred in predictions:
        label = labels[pred["sample_id"]]
        anchor, target = label["anchor_full"], label["target_full"]
        pixels = int(anchor.sum())
        models = {}
        tx, ty = pred["M0"]["target"]["map_pixel_xy"]
        target_hit = bool(target[ty, tx])
        for arm in ARMS:
            summary = pred[arm]["anchor"]
            x, y = summary["map_pixel_xy"]
            gx, gy = summary["map_grid_xy"]
            models[arm] = {**summary, "anchor_hit": bool(anchor[y, x]) if pixels else None,
                "anchor_cell_overlap": bool(label["anchor_small"][gy, gx] > 0) if pixels else None,
                "anchor_inside_target": bool(target[y, x]),
                "target_hit": target_hit,
                "found_both_hit": bool(target_hit and anchor[y, x]) if label["truth"] == "FOUND" else None}
        rows.append({"sample_id": pred["sample_id"], "family_id": pred["family_id"], "split": pred["split"],
            "variant": label["variant"], "truth": label["truth"], "parsed": pred["parsed"],
            "feature_key": pred["feature_key"], "anchor_pixels": pixels,
            "anchor_ids": label["anchor_ids"], "anchor_mask_path": label["anchor_mask_path"],
            "target_mask_path": label["target_mask_path"], "rgb_path": label["rgb_path"], "models": models})
    return rows


def pair_rows(rows):
    by_family = defaultdict(dict)
    for row in rows:
        by_family[row["family_id"]][row["variant"]] = row
    pairs = []
    for family, variants in sorted(by_family.items()):
        if not {"clean", "relation_counterfactual"}.issubset(variants):
            continue
        a, b = variants["clean"], variants["relation_counterfactual"]
        visible = a["anchor_pixels"] > 0 and b["anchor_pixels"] > 0
        same = a["feature_key"] == b["feature_key"]
        changed = a["parsed"]["anchors"][0]["text"] != b["parsed"]["anchors"][0]["text"]
        pairs.append({"family_id": family, "split": a["split"], "sample_ids": [a["sample_id"], b["sample_id"]],
            "both_visible": visible, "same_cached_features": same, "anchor_phrase_changed": changed,
            # Protocol/S0 use the visible, same-cache matched subset. Phrase
            # change is a diagnostic, not an additional eligibility filter.
            "eligible": bool(visible and same),
            "models": {arm: {"both_hit": bool(a["models"][arm]["anchor_hit"] and b["models"][arm]["anchor_hit"]),
                "same_peak": a["models"][arm]["map_grid_xy"] == b["models"][arm]["map_grid_xy"]} for arm in ARMS}})
    return pairs


def aggregate(rows):
    visible = [r for r in rows if r["anchor_pixels"] > 0]
    empty = [r for r in rows if r["anchor_pixels"] == 0]
    found = [r for r in rows if r["truth"] == "FOUND"]
    eligible = [p for p in pair_rows(rows) if p["eligible"]]
    models = {}
    for arm in ARMS:
        hits = sum(r["models"][arm]["anchor_hit"] for r in visible)
        models[arm] = {"anchor_hits": hits, "anchor_hit_rate": hits/len(visible) if visible else None,
            "anchor_cell_hits": sum(r["models"][arm]["anchor_cell_overlap"] for r in visible),
            "found_target_hits": sum(r["models"][arm]["target_hit"] for r in found),
            "found_anchor_hits": sum(r["models"][arm]["anchor_hit"] is True for r in found),
            "found_both_hits": sum(r["models"][arm]["found_both_hit"] for r in found),
            "swap_both_hits": sum(p["models"][arm]["both_hit"] for p in eligible),
            "swap_same_peak": sum(p["models"][arm]["same_peak"] for p in eligible),
            "confident_wrong_anchor_ge_0p9": sum(not r["models"][arm]["anchor_hit"] and r["models"][arm]["sigmoid_max"] >= .9 for r in visible),
            "empty_mean_sigmoid_max": float(np.mean([r["models"][arm]["sigmoid_max"] for r in empty])) if empty else None,
            "empty_max_sigmoid_max": max((r["models"][arm]["sigmoid_max"] for r in empty), default=None),
            "wrong_anchor_inside_target": sum(not r["models"][arm]["anchor_hit"] and r["models"][arm]["anchor_inside_target"] for r in visible)}
    return {"samples": len(rows), "families": len({r["family_id"] for r in rows}), "visible": len(visible),
            "empty": len(empty), "found": len(found), "matched_swap_pairs": len(eligible), "models": models}


def paired_changes(rows, candidate, reference="M0"):
    fixed, broken = [], []
    for row in rows:
        if not row["anchor_pixels"]:
            continue
        a, b = row["models"][candidate]["anchor_hit"], row["models"][reference]["anchor_hit"]
        if a != b:
            (fixed if a else broken).append(row["sample_id"])
    return {"fixed_count": len(fixed), "broken_count": len(broken), "net_hits": len(fixed)-len(broken),
            "fixed_sample_ids": fixed, "broken_sample_ids": broken}


def family_bootstrap(rows, candidate, reference="M0", seed=24082026, resamples=5000):
    """Paired, family-clustered ratio deltas; all variants stay together."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["family_id"]].append(r)
    if not groups:
        raise ValueError("No families for bootstrap")
    families = sorted(groups)
    eligible = {p["family_id"]: p for p in pair_rows(rows) if p["eligible"]}
    counts = []
    for family in families:
        rs = groups[family]; visible = [r for r in rs if r["anchor_pixels"]]
        found = [r for r in rs if r["truth"] == "FOUND"]
        p = eligible.get(family)
        counts.append([len(visible), sum(int(r["models"][candidate]["anchor_hit"])-int(r["models"][reference]["anchor_hit"]) for r in visible),
            len(found), sum(int(r["models"][candidate]["found_both_hit"])-int(r["models"][reference]["found_both_hit"]) for r in found),
            int(p is not None), int(p["models"][candidate]["both_hit"])-int(p["models"][reference]["both_hit"]) if p else 0])
    table = np.asarray(counts, dtype=np.float64)
    indices = np.random.default_rng(seed).integers(0, len(families), size=(resamples, len(families)))
    totals = table[indices].sum(1); point = table.sum(0)
    result = {"families": len(families), "resamples": resamples, "seed": seed, "conditional_on_dev_selected_checkpoint": True}
    for name, denominator, numerator in (("anchor_hit_delta", 0, 1), ("found_both_delta", 2, 3), ("swap_both_delta", 4, 5)):
        valid = totals[:, denominator] > 0
        values = totals[valid, numerator] / totals[valid, denominator]
        result[name] = {"point": float(point[numerator]/point[denominator]) if point[denominator] else None,
            "percentile_ci95": np.quantile(values, [.025, .975]).tolist() if len(values) else None,
            "valid_resamples": int(valid.sum())}
    return result


def locked_gates(dev_rows, technical_pass, latency_pass):
    metrics = aggregate(dev_rows); m1, m2 = metrics["models"]["M1"], metrics["models"]["M2"]
    missing = [{"sample_id": r["sample_id"], "family_id": r["family_id"],
                **{arm: r["models"][arm]["sigmoid_max"] for arm in ARMS},
                "M1_pass": r["models"]["M1"]["sigmoid_max"] <= r["models"]["M0"]["sigmoid_max"] + 1e-6}
               for r in dev_rows if not r["anchor_pixels"]]
    if (metrics["visible"], metrics["matched_swap_pairs"], metrics["found"], metrics["empty"]) != (61, 15, 16, 3):
        raise ValueError("Locked dev denominators changed")
    nondecreasing = m1["anchor_hits"] >= m2["anchor_hits"] and m1["swap_both_hits"] >= m2["swap_both_hits"]
    phrase = nondecreasing and (m1["anchor_hits"] >= m2["anchor_hits"] + 2 or m1["swap_both_hits"] >= m2["swap_both_hits"] + 2)
    gates = {"technical": bool(technical_pass), "anchor_hit_ge_49_of_61": m1["anchor_hits"] >= 49,
        "swap_both_ge_8_of_15": m1["swap_both_hits"] >= 8, "found_both_ge_12_of_16": m1["found_both_hits"] >= 12,
        "empty_each_no_increase_1e_6": all(r["M1_pass"] for r in missing),
        "phrase_specific_vs_M2": bool(phrase), "latency_le_20_percent": bool(latency_pass)}
    return {"pass": all(gates.values()), "gates": gates, "missing_cases": missing,
        "phrase_deltas": {"visible_hits": m1["anchor_hits"]-m2["anchor_hits"], "swap_both_hits": m1["swap_both_hits"]-m2["swap_both_hits"]}}
