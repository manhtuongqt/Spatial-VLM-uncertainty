"""Freeze the Day-5 development cache and mini-overfit design before forward."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "ketqua1/01_dau_vao_tien_xu_ly/ngay_03/gazebo_pilot_512_revision_v2/PILOT_512_SELECTED_INDEX.jsonl"
G0_LOCK = ROOT / "protocol/MH_PCRAU_V3_G0_RUN_LOCK_R3.json"
G1_DECISION = ROOT / "ketqua1/00_quan_tri_khoa/ngay_03/G1_DECISION_REVISION_V3.json"
DAY4_DECISION = ROOT / "ketqua1/04_multihead_pcrau_v3/ngay_04/DAY4_DECISION.json"
OUT = ROOT / "ketqua1/03_backbone_h_spatial/ngay_05"
LOCK = ROOT / "protocol/MH_PCRAU_V3_DAY5_RUN_LOCK.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> None:
    if LOCK.exists():
        raise FileExistsError(f"Append-only lock already exists: {LOCK}")
    if json.loads(G1_DECISION.read_text())["outcome"] != "G1_PASS":
        raise RuntimeError("G1 not passed")
    day4 = json.loads(DAY4_DECISION.read_text())
    if day4["outcome"] != "DAY4_IMPLEMENTATION_SMOKE_PASS":
        raise RuntimeError("Day 4 not passed")
    rows = [json.loads(line) for line in INDEX.read_text().splitlines() if line.strip()]
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["origin"] == "EXISTING_TRAIN_UQ":
            cells[(row["supervision"]["relation"], row["supervision"]["answerability"])].append(row)
    expected_relations = ("leftmost", "rightmost", "second_from_left", "second_from_right")
    expected_states = ("FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE")
    selected = []
    for state in expected_states:
        for relation in expected_relations:
            available = sorted(cells[(relation, state)], key=lambda item: item["sample_id"])
            if len(available) < 2:
                raise RuntimeError(f"Insufficient cell: {relation}/{state}")
            selected.extend(available[:2])
    if len(selected) != 32 or len({row["family_id"] for row in selected}) != 32:
        raise RuntimeError("Expected 32 independent families")
    OUT.mkdir(parents=True, exist_ok=False)
    inference_rows = []
    supervision_rows = []
    for row in selected:
        rgb = ROOT / row["model_input"]["rgb_path"]
        metric_depth = ROOT / row["model_input"]["depth_path"]
        depth_view = metric_depth.with_name("depth_view.png")
        for path in (rgb, metric_depth, depth_view):
            if not path.is_file():
                raise FileNotFoundError(path)
        inference_rows.append({
            "sample_id": row["sample_id"], "family_id": row["family_id"],
            "rgb_path": rel(rgb), "rgb_sha256": sha256(rgb),
            "depth_view_path": rel(depth_view), "depth_view_sha256": sha256(depth_view),
            "metric_depth_path": rel(metric_depth), "metric_depth_sha256": sha256(metric_depth),
            "instruction": row["model_input"]["instruction"],
            "instruction_sha256": digest(row["model_input"]["instruction"]),
        })
        supervision_rows.append({
            "sample_id": row["sample_id"], "family_id": row["family_id"],
            "relation": row["supervision"]["relation"],
            "answerability": row["supervision"]["answerability"],
            "target_uv": row["supervision"].get("target_uv"),
            "head_mask": row["head_mask"],
        })
    inference_path = OUT / "CACHE_INPUT_MANIFEST.jsonl"
    supervision_path = OUT / "SUPERVISION_STORE.jsonl"
    inference_path.write_text("".join(json.dumps(row) + "\n" for row in inference_rows))
    supervision_path.write_text("".join(json.dumps(row) + "\n" for row in supervision_rows))
    mini_ids = [row["sample_id"] for row in inference_rows[::2]]
    if len(mini_ids) != 16:
        raise RuntimeError("Mini-batch design drift")
    candidates = [
        {"id": "W0_EQUAL", "relation": 1.0, "reasoning": 1.0, "spatial": 1.0, "source": 1.0, "answerability": 1.0, "confidence": 0.0, "language_modeling": 0.0},
        {"id": "W1_SPA025", "relation": 1.0, "reasoning": 1.0, "spatial": 0.25, "source": 1.0, "answerability": 1.0, "confidence": 0.0, "language_modeling": 0.0},
        {"id": "W2_SPA050", "relation": 1.0, "reasoning": 1.0, "spatial": 0.50, "source": 1.0, "answerability": 1.0, "confidence": 0.0, "language_modeling": 0.0},
        {"id": "W3_SPA200", "relation": 1.0, "reasoning": 1.0, "spatial": 2.0, "source": 1.0, "answerability": 1.0, "confidence": 0.0, "language_modeling": 0.0},
        {"id": "W4_REL050", "relation": 0.5, "reasoning": 1.0, "spatial": 1.0, "source": 1.0, "answerability": 1.0, "confidence": 0.0, "language_modeling": 0.0},
        {"id": "W5_ANS050", "relation": 1.0, "reasoning": 1.0, "spatial": 1.0, "source": 1.0, "answerability": 0.5, "confidence": 0.0, "language_modeling": 0.0}
    ]
    lock = {
        "schema_version": "1.0", "method_id": "MH-PCRA-U-v3",
        "status": "FROZEN_BEFORE_DAY5_MODEL_FORWARD",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 25092026,
        "prerequisites": [
            {"path": rel(G0_LOCK), "sha256": sha256(G0_LOCK)},
            {"path": rel(G1_DECISION), "sha256": sha256(G1_DECISION)},
            {"path": rel(DAY4_DECISION), "sha256": sha256(DAY4_DECISION)},
            {"path": rel(INDEX), "sha256": sha256(INDEX)},
        ],
        "cache_input_manifest": {"path": rel(inference_path), "sha256": sha256(inference_path)},
        "supervision_store": {"path": rel(supervision_path), "sha256": sha256(supervision_path)},
        "selection": {
            "rule": "EXISTING_TRAIN_UQ only; core 4 relation x 4 answerability cells; ascending sample_id; first 2 independent families/cell",
            "samples": 32, "families": 32, "families_per_relation_state_cell": 2,
            "mini_batch_sample_ids": mini_ids,
            "mini_batch_rule": "first selected member of each relation x answerability cell",
        },
        "model_precision_policy": {
            "checkpoint": "RoboRefer/models/RoboRefer-2B-SFT",
            "checkpoint_inventory_sha256": json.loads(G0_LOCK.read_text())["model_inventory_sha256"],
            "rgb_depth_towers_and_projectors": "bfloat16",
            "frozen_llm_and_h_spatial": "float32", "attention": "eager", "tf32": False
        },
        "cache_acceptance": {
            "expected_samples": 32, "shape": [1536], "dtype": "torch.float32",
            "finite_fraction": 1.0, "roundtrip_max_abs": 0.0,
            "unique_sample_ids": 32, "unique_family_ids": 32,
            "model_input_allowlist": ["rgb_path", "depth_view_path", "instruction"]
        },
        "gradient_scale": {
            "candidates": candidates,
            "selection_rule": "minimize max/min ratio of nonzero weighted shared-trunk gradient norms over active relation/spatial/answerability tasks; tie by candidate order",
            "maximum_candidates": 6
        },
        "mini_overfit": {
            "optimizer": "AdamW", "learning_rate": 0.002, "weight_decay": 0.0,
            "maximum_steps": 600, "minimum_steps": 50, "evaluation_interval": 10,
            "full_batch": 16, "dropout": 0.10,
            "pass_thresholds": {
                "relation_loss_final_over_initial_max": 0.25,
                "answerability_loss_final_over_initial_max": 0.25,
                "relation_accuracy_min": 0.9375,
                "answerability_accuracy_min": 0.9375,
                "point_mae_uv_max": 0.03,
                "spatial_nll_must_decrease": True,
                "log_variance_min_strictly_above": -7.95,
                "log_variance_max_strictly_below": 1.95,
                "finite_rate": 1.0
            }
        },
        "label_scope": "reasoning/source/confidence have Day-3 certified support 0 and remain masked; their learning is N/A, not PASS",
        "access_boundary": "Development only. No Calibration/Test/robot access."
    }
    LOCK.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": lock["status"], "samples": 32, "families": 32,
                      "lock_sha256": sha256(LOCK)}))


if __name__ == "__main__":
    main()
