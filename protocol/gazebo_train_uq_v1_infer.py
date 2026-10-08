#!/usr/bin/env python3
"""Blinded inference runner for a materialized Gazebo_train_uq_v1 split.

The runner reads only RGB, metric-depth visualisation, and instruction from an
inference manifest.  Evaluation-oracle fields are rejected before a model is
loaded.  It is intentionally generic so the frozen B0, clean B1, and future
B1-UQ adapter use identical decoding and parsing code.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "gazebo_train_uq_v1"
POINT_RE = re.compile(r"^POINT \[\(([0-9]*\.?[0-9]+), ([0-9]*\.?[0-9]+)\)\]$")
TOLERANT_POINT_RE = re.compile(r"\[?\s*\(\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)\s*\]?")
FORBIDDEN_INPUT_FIELDS = {"target_xy", "target_id", "target_mask", "answerability_state", "candidate_set", "semantic_label", "semantic_labels", "failure_tags", "scene_layout", "camera_to_world"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_response(answer: str | None) -> dict:
    text = (answer or "").strip()
    matches = TOLERANT_POINT_RE.findall(text)
    abstain = bool(re.search(r"\bABSTAIN\b", text, re.IGNORECASE))
    if len(matches) == 1 and not abstain:
        point = [float(value) for value in matches[0]]
        if all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in point):
            return {"action": "POINT", "prediction_xy": point, "exact_contract": bool(POINT_RE.fullmatch(text))}
    if not matches and abstain:
        return {"action": "ABSTAIN", "prediction_xy": None, "exact_contract": text == "ABSTAIN"}
    return {"action": "INVALID", "prediction_xy": None, "exact_contract": False}


def stable_seed(sample_id: str, model_id: str, draw: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{PROTOCOL_ID}:{sample_id}:{model_id}:{draw}".encode()).digest()[:4], "big")


def validate_manifest(dataset: Path) -> tuple[dict, list[dict]]:
    manifest_path = dataset / "manifest.json"
    inference_path = dataset / "inference_manifest.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("status") != "PASS":
        raise ValueError("dataset is not a PASS Gazebo_train_uq_v1 materialization")
    rows = load_jsonl(inference_path)
    expected = int(manifest.get("records", 0))
    if not expected or len(rows) != expected or len({row.get("sample_id") for row in rows}) != expected:
        raise ValueError("inference manifest is incomplete or sample IDs are not unique")
    leaked = [(row.get("sample_id"), key) for row in rows for key in FORBIDDEN_INPUT_FIELDS & set(row)]
    if leaked:
        raise ValueError(f"evaluation oracle leaked to inference: {leaked[:5]}")
    required = {"sample_id", "scene_id", "family_id", "split", "relation", "image", "depth", "instruction"}
    missing = [(row.get("sample_id"), sorted(required - set(row))) for row in rows if required - set(row)]
    if missing:
        raise ValueError(f"inference record missing input fields: {missing[:5]}")
    return manifest, rows


def run(args: argparse.Namespace) -> None:
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    base = args.base.resolve()
    adapter = args.adapter.resolve() if args.adapter else None
    manifest, rows = validate_manifest(dataset)
    if args.model_id == "b0" and adapter is not None:
        raise ValueError("B0 must not receive an adapter")
    if args.model_id != "b0" and (adapter is None or not adapter.is_file()):
        raise FileNotFoundError("an existing adapter is required for clean_b1 or b1_uq")
    if not base.is_dir():
        raise FileNotFoundError(base)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    run_path = output / "run.json"
    contract = {
        "protocol_id": PROTOCOL_ID, "model_id": args.model_id,
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "inference_manifest_sha256": sha256(dataset / "inference_manifest.jsonl"),
        "base_model_sha256": directory_sha256(base), "adapter_sha256": sha256(adapter) if adapter else None,
        "greedy_decoding": True, "stochastic_draws": args.draws, "max_new_tokens": args.max_new_tokens,
        "model_input_allowlist": ["rgb", "depth_view", "instruction"], "oracle_opened_by_runner": False,
    }
    completed = {}
    if predictions_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
        if any(previous.get(key) != value for key, value in contract.items()):
            raise ValueError("refusing incompatible inference resume")
        completed = {row["sample_id"]: row for row in load_jsonl(predictions_path)}
    pending = [row for row in rows if row["sample_id"] not in completed]
    run_payload = {**contract, "status": "RUNNING", "samples_requested": len(rows), "already_completed": len(completed), "pending": len(pending), "started_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(run_path, run_payload)
    if not pending:
        run_payload.update(status="COMPLETED", completed_count=len(rows), finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(run_path, run_payload)
        return
    sys.path.insert(0, str(ROOT / "RoboRefer"))
    import torch
    import llava
    from llava import conversation as conversation_lib
    from llava.media import Depth, Image

    torch.manual_seed(12092026)
    torch.cuda.manual_seed_all(12092026)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = llava.load(str(base)) if adapter is None else llava.load(str(adapter.parent), model_base=str(base))
    conversation_lib.default_conversation = conversation_lib.conv_templates["auto"].copy()
    greedy = copy.deepcopy(model.default_generation_config)
    greedy.do_sample = False; greedy.temperature = greedy.top_p = greedy.top_k = None; greedy.max_new_tokens = args.max_new_tokens
    stochastic = copy.deepcopy(model.default_generation_config)
    stochastic.do_sample = True; stochastic.temperature = 0.7; stochastic.top_p = 0.9; stochastic.top_k = 50; stochastic.max_new_tokens = args.max_new_tokens
    run_payload.update(model_load_seconds=time.perf_counter() - started, gpu_name=torch.cuda.get_device_name())
    write_json(run_path, run_payload)
    mode = "a" if predictions_path.exists() else "w"
    try:
        with predictions_path.open(mode, encoding="utf-8") as stream:
            for index, row in enumerate(pending, 1):
                item_started = time.perf_counter()
                record = {key: row[key] for key in ("sample_id", "scene_id", "family_id", "split", "relation")}
                try:
                    prompt = [Image(str(dataset / row["image"])), Depth(str(dataset / row["depth"])), row["instruction"]]
                    with torch.inference_mode():
                        answer = model.generate_content(prompt, generation_config=copy.deepcopy(greedy))
                    parsed = parse_response(answer)
                    draws = []
                    for draw in range(args.draws):
                        seed = stable_seed(row["sample_id"], args.model_id, draw)
                        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                        with torch.inference_mode():
                            sampled = model.generate_content(prompt, generation_config=copy.deepcopy(stochastic))
                        draws.append({"draw": draw, "seed": seed, "answer": sampled, **parse_response(sampled)})
                    if parsed["action"] == "POINT":
                        consistent = sum(draw["action"] == "POINT" and math.dist(parsed["prediction_xy"], draw["prediction_xy"]) <= 0.08 for draw in draws)
                    elif parsed["action"] == "ABSTAIN":
                        consistent = sum(draw["action"] == "ABSTAIN" for draw in draws)
                    else:
                        consistent = 0
                    record.update(answer=answer, **parsed, stochastic_draws=draws, self_consistency_confidence=consistent / args.draws, predictive_uncertainty=1 - consistent / args.draws)
                except Exception:
                    record.update(answer=None, action="INVALID", prediction_xy=None, exact_contract=False, stochastic_draws=[], self_consistency_confidence=0.0, predictive_uncertainty=1.0, error=traceback.format_exc())
                record["latency_seconds"] = time.perf_counter() - item_started
                stream.write(json.dumps(record, ensure_ascii=False) + "\n"); stream.flush()
                print(f"[{args.model_id} {index}/{len(pending)}] {record['sample_id']} {record['action']}", flush=True)
        run_payload.update(status="COMPLETED", completed_count=len(rows), cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20, cuda_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    except Exception:
        run_payload.update(status="FAILED", error=traceback.format_exc())
        raise
    finally:
        run_payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(run_path, run_payload)


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", choices=["b0", "clean_b1", "b1_uq"], required=True)
    parser.add_argument("--base", type=Path, default=ROOT / "RoboRefer/models/RoboRefer-2B-SFT")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.draws < 1 or args.max_new_tokens < 1:
        parser.error("draws and max-new-tokens must be positive")
    run(args)


if __name__ == "__main__":
    main()
