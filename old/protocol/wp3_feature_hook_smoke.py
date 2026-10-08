#!/usr/bin/env python3
"""Locked WP3 RoboRefer feature-hook and determinism smoke protocol.

The inference commands only read the oracle-free smoke manifest. Dataset record
JSON files are verified as opaque bytes; evaluator artifacts are never opened.
This module performs no optimization, gradient update, dataset generation or
robot publication.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "wp3_feature_hook_smoke_v1"
EXPECTED_MODEL_INVENTORY_SHA256 = (
    "5508fb49888c1f62b875d928266f7fff3e1224c795f6fc8b971631143a45a7fa"
)
RUN_IDS = ("hook_a", "hook_b")
EXPECTED_SAMPLE_COUNT = 15
EXPECTED_PREPROJECTOR_TAIL = (1024, 1152)
EXPECTED_PROJECTED_TAIL = (121, 1536)
EXPECTED_PATCH_GRID = (32, 32)
EXPECTED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
FORBIDDEN_CACHE_KEYS = {
    "answerability",
    "answerable",
    "anchorid",
    "anchorids",
    "anchormask",
    "anchormasks",
    "correctness",
    "expectedintervention",
    "groundtruth",
    "mask",
    "masks",
    "objectid",
    "objectids",
    "oracle",
    "relationgraph",
    "robotpose",
    "semanticlabels",
    "sourcelabel",
    "sourcelabels",
    "targetid",
    "targetids",
    "targetmask",
    "targetmasks",
    "testmetric",
    "uncertaintylabel",
    "validtargetids",
}


class FeatureGateError(RuntimeError):
    """Raised when a locked feature-gate invariant is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureGateError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FeatureGateError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve()))


def safe_resolve(base: Path, relative_path: str) -> Path:
    if not relative_path or Path(relative_path).is_absolute():
        raise FeatureGateError(f"path must be non-empty and relative: {relative_path!r}")
    resolved = (base / relative_path).resolve()
    if resolved != base.resolve() and base.resolve() not in resolved.parents:
        raise FeatureGateError(f"path escapes base {base}: {relative_path}")
    return resolved


def tree_digest(relative_root: str) -> str:
    tree_root = safe_resolve(ROOT, relative_root)
    if not tree_root.is_dir():
        raise FeatureGateError(f"protected tree is missing: {relative_root}")
    paths = [
        str(path.relative_to(ROOT))
        for path in tree_root.rglob("*")
        if path.is_file()
    ]
    sorted_bytes = subprocess.run(
        ["sort", "-z"],
        input=b"\0".join(item.encode("utf-8") for item in paths) + b"\0",
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    digest = hashlib.sha256()
    for raw_item in sorted_bytes.rstrip(b"\0").split(b"\0"):
        item = raw_item.decode("utf-8")
        digest.update(f"{sha256_file(ROOT / item)}  {item}\n".encode("utf-8"))
    return digest.hexdigest()


def model_inventory_sha256(model_root: Path) -> str:
    entries = []
    for path in sorted(item for item in model_root.rglob("*") if item.is_file()):
        entries.append(
            {
                "path": str(path.relative_to(model_root)),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return sha256_bytes(canonical_json_bytes(entries))


def normalize_key(value: str) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def forbidden_cache_findings(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            child_path = f"{path}.{raw_key}"
            normalized = normalize_key(str(raw_key))
            safe_attestation = child_path == "$.safety.evaluator_artifacts_read"
            if normalized in FORBIDDEN_CACHE_KEYS and not safe_attestation:
                findings.append(child_path)
            findings.extend(forbidden_cache_findings(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(forbidden_cache_findings(child, f"{path}[{index}]"))
    return findings


def json_schema_errors(schema_path: Path, instance: Any | None = None) -> list[str]:
    """Validate with Draft 2020-12 without polluting the RoboRefer environment.

    The clean RoboRefer venv intentionally has no jsonschema package. The system
    Python used by WP2/WP3 protocol checks does, so the fallback validates over
    stdin in an isolated subprocess and returns plain error messages.
    """

    try:
        from jsonschema import Draft202012Validator
    except ModuleNotFoundError:
        validator_source = """
import json
import sys
from jsonschema import Draft202012Validator
schema = json.load(open(sys.argv[1], encoding='utf-8'))
Draft202012Validator.check_schema(schema)
if sys.argv[2] == 'instance':
    value = json.load(sys.stdin)
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda error: list(error.path))
    if errors:
        print('\\n'.join(error.message for error in errors), file=sys.stderr)
        raise SystemExit(2)
"""
        mode = "instance" if instance is not None else "schema"
        schema_environment = dict(os.environ)
        schema_environment.pop("PYTHONNOUSERSITE", None)
        completed = subprocess.run(
            ["/usr/bin/python3", "-c", validator_source, str(schema_path), mode],
            input=canonical_json_bytes(instance) if instance is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=schema_environment,
            check=False,
        )
        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", errors="replace").strip()
            return [message or f"system schema validator exited {completed.returncode}"]
        return []
    schema = read_json(schema_path)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        return [str(exc)]
    if instance is None:
        return []
    return [
        error.message
        for error in sorted(
            Draft202012Validator(schema).iter_errors(instance),
            key=lambda error: list(error.path),
        )
    ]


def verify_feature_gate_lock(lock_path: Path, include_evaluator_artifacts: bool = False) -> dict:
    lock = read_json(lock_path)
    if lock.get("protocol_id") != PROTOCOL_ID or lock.get("status") != "LOCKED":
        raise FeatureGateError("feature-gate lock is not LOCKED/wp3_feature_hook_smoke_v1")
    for relative_path, expected in lock.get("artifacts", {}).items():
        path = safe_resolve(ROOT, relative_path)
        if not path.is_file() or sha256_file(path) != expected:
            raise FeatureGateError(f"locked artifact hash mismatch: {relative_path}")
    if include_evaluator_artifacts:
        for relative_path, expected in lock.get("evaluator_artifacts", {}).items():
            path = safe_resolve(ROOT, relative_path)
            if not path.is_file() or sha256_file(path) != expected:
                raise FeatureGateError(
                    f"locked evaluator artifact hash mismatch: {relative_path}"
                )
    verify_protected_inputs(lock)
    return lock


def protected_input_snapshot(lock: Mapping[str, Any]) -> dict:
    file_hashes = {
        relative_path: sha256_file(safe_resolve(ROOT, relative_path))
        for relative_path in lock.get("protected_files", {})
    }
    tree_hashes = {
        relative_path: tree_digest(relative_path)
        for relative_path in lock.get("protected_tree_digests", {})
    }
    return {"files": file_hashes, "trees": tree_hashes}


def verify_protected_inputs(lock: Mapping[str, Any]) -> dict:
    snapshot = protected_input_snapshot(lock)
    mismatches = []
    for relative_path, expected in lock.get("protected_files", {}).items():
        if snapshot["files"].get(relative_path) != expected:
            mismatches.append(relative_path)
    for relative_path, expected in lock.get("protected_tree_digests", {}).items():
        if snapshot["trees"].get(relative_path) != expected:
            mismatches.append(relative_path)
    if mismatches:
        raise FeatureGateError(f"protected WP0/WP1/WP2 inputs changed: {mismatches}")
    return snapshot


def validate_static(
    manifest_path: Path,
    schema_path: Path,
    selection_audit_path: Path | None,
) -> dict:
    schema_errors = json_schema_errors(schema_path)
    if schema_errors:
        raise FeatureGateError(f"feature cache schema is invalid: {schema_errors[0]}")
    manifest = read_json(manifest_path)
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise FeatureGateError("smoke manifest protocol mismatch")
    if manifest.get("status") != "LOCKED_BEFORE_RUNTIME":
        raise FeatureGateError("smoke manifest is not locked before runtime")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != EXPECTED_SAMPLE_COUNT:
        raise FeatureGateError("smoke manifest must contain exactly 15 entries")
    if manifest.get("expected_sample_count") != EXPECTED_SAMPLE_COUNT:
        raise FeatureGateError("smoke manifest expected_sample_count mismatch")
    if manifest.get("model_inventory_sha256") != EXPECTED_MODEL_INVENTORY_SHA256:
        raise FeatureGateError("smoke manifest model inventory mismatch")
    dataset_root = safe_resolve(ROOT, str(manifest["dataset_root"]))
    if sha256_file(dataset_root / "dataset_index.json") != manifest["dataset_index_sha256"]:
        raise FeatureGateError("dataset index differs from smoke manifest")
    required_entry_keys = {
        "sample_id",
        "family_id",
        "split",
        "variant",
        "record_path",
        "record_sha256",
        "instruction",
        "prompt",
        "rgb_path",
        "rgb_sha256",
        "depth_path",
        "depth_sha256",
    }
    sample_ids = set()
    family_ids = set()
    for entry in entries:
        if set(entry) != required_entry_keys:
            raise FeatureGateError(f"manifest entry fields differ: {entry.get('sample_id')}")
        if entry["split"] != "train":
            raise FeatureGateError(f"non-train smoke entry: {entry['sample_id']}")
        if entry["sample_id"] in sample_ids or entry["family_id"] in family_ids:
            raise FeatureGateError("smoke requires unique sample and family IDs")
        sample_ids.add(entry["sample_id"])
        family_ids.add(entry["family_id"])
        if not str(entry["sample_id"]).startswith(str(entry["family_id"]) + "__"):
            raise FeatureGateError(f"sample/family mismatch: {entry['sample_id']}")
        record_path = safe_resolve(dataset_root, str(entry["record_path"]))
        rgb_path = safe_resolve(dataset_root, str(entry["rgb_path"]))
        depth_path = safe_resolve(dataset_root, str(entry["depth_path"]))
        for path, expected in (
            (record_path, entry["record_sha256"]),
            (rgb_path, entry["rgb_sha256"]),
            (depth_path, entry["depth_sha256"]),
        ):
            if not path.is_file() or sha256_file(path) != expected:
                raise FeatureGateError(f"locked smoke input mismatch: {path}")
        if not entry["prompt"].startswith(entry["instruction"]):
            raise FeatureGateError(f"prompt/instruction mismatch: {entry['sample_id']}")
        lowered = (entry["instruction"] + " " + entry["prompt"]).lower()
        if any(token in lowered for token in ("<image>", "<depth>", "mask path", "object_id")):
            raise FeatureGateError(f"forbidden token in prompt: {entry['sample_id']}")

    selection_summary = None
    if selection_audit_path is not None:
        selection = read_json(selection_audit_path)
        if selection.get("access_scope") != "evaluator_only_not_loaded_by_inference_runner":
            raise FeatureGateError("selection audit access scope mismatch")
        selection_ids = {item["sample_id"] for item in selection.get("entries", [])}
        if selection_ids != sample_ids:
            raise FeatureGateError("selection audit and inference manifest sample IDs differ")
        required_states = {"FOUND", "INSUFFICIENT_EVIDENCE", "AMBIGUOUS", "ABSENT"}
        states = set(selection.get("coverage", {}).get("answerability_states", {}))
        required_categories = {
            "direct_grounding",
            "relation_2d",
            "nearer_farther",
            "front_behind_camera",
            "multi_anchor_depth_order",
            "occlusion_depth_evidence",
        }
        categories = set(selection.get("coverage", {}).get("family_categories", {}))
        if not required_states.issubset(states) or not required_categories.issubset(categories):
            raise FeatureGateError("15-sample selection does not meet locked coverage")
        selection_summary = selection["coverage"]
    return {
        "schema_valid": True,
        "sample_count": len(entries),
        "distinct_family_count": len(family_ids),
        "all_train": True,
        "selection_coverage": selection_summary,
    }


def seed_runtime(seed: int) -> None:
    import numpy as np
    import torch

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != EXPECTED_CUBLAS_WORKSPACE_CONFIG:
        raise FeatureGateError(
            "CUBLAS_WORKSPACE_CONFIG must be locked to :4096:8 before Python starts"
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def load_model(model_root: Path):
    if not __package__:
        robo_root = str(ROOT / "RoboRefer")
        if robo_root not in sys.path:
            sys.path.insert(0, robo_root)
    import llava
    from llava import conversation as clib

    model = llava.load(str(model_root))
    model.eval()
    clib.default_conversation = clib.conv_templates["auto"].copy()
    return model


def parse_point(answer: str) -> dict:
    try:
        value = ast.literal_eval(str(answer).strip())
    except (SyntaxError, ValueError):
        return {"parse_status": "NO_POINT", "normalized_points_xy": []}
    if not isinstance(value, list) or len(value) != 1:
        return {"parse_status": "MULTIPLE_OR_INVALID_POINTS", "normalized_points_xy": []}
    point = value[0]
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return {"parse_status": "NO_POINT", "normalized_points_xy": []}
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in point):
        return {"parse_status": "NO_POINT", "normalized_points_xy": []}
    x_value, y_value = float(point[0]), float(point[1])
    if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in (x_value, y_value)):
        return {
            "parse_status": "OUT_OF_RANGE",
            "normalized_points_xy": [[x_value, y_value]],
        }
    exact = str(answer).strip().startswith("[(") and str(answer).strip().endswith(")]")
    return {
        "parse_status": "EXACT_ONE_NORMALIZED_POINT" if exact else "FORMAT_VIOLATION",
        "normalized_points_xy": [[x_value, y_value]],
    }


def synchronize() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def generate_answer(model, rgb_path: Path, depth_path: Path, prompt: str, generation: dict) -> tuple[str, float]:
    from llava.media import Depth, Image

    config = copy.deepcopy(model.default_generation_config)
    config.do_sample = False
    config.temperature = None
    config.top_p = None
    config.top_k = None
    config.max_new_tokens = int(generation["max_new_tokens"])
    synchronize()
    started = time.monotonic()
    answer = model.generate_content(
        [Image(str(rgb_path)), Depth(str(depth_path)), prompt],
        generation_config=config,
    )
    synchronize()
    return str(answer), (time.monotonic() - started) * 1000.0


def dynamic_tile_metadata(width: int, height: int, model) -> dict:
    from llava.mm_utils import find_closest_aspect_ratio

    vision_processor = model.get_vision_tower().image_processor
    crop_size = vision_processor.crop_size if hasattr(vision_processor, "crop_size") else vision_processor.size
    tile_size = int(crop_size["height"])
    if int(crop_size["width"]) != tile_size:
        raise FeatureGateError("feature contract requires square input tiles")
    min_tiles = int(model.config.min_tiles)
    max_tiles = int(model.config.max_tiles)
    ratios = {
        (columns, rows)
        for count in range(min_tiles, max_tiles + 1)
        for columns in range(1, count + 1)
        for rows in range(1, count + 1)
        if min_tiles <= columns * rows <= max_tiles
    }
    ratios = sorted(ratios, key=lambda item: item[0] * item[1])
    columns, rows = find_closest_aspect_ratio(
        width / height, ratios, width, height, tile_size
    )
    canvas_width = tile_size * columns
    canvas_height = tile_size * rows
    local_count = columns * rows
    thumbnail_present = local_count != 1
    tiles = []
    for index in range(local_count):
        column = index % columns
        row = index // columns
        canvas_box = [
            column * tile_size,
            row * tile_size,
            (column + 1) * tile_size,
            (row + 1) * tile_size,
        ]
        original_box = [
            canvas_box[0] * width / canvas_width,
            canvas_box[1] * height / canvas_height,
            canvas_box[2] * width / canvas_width,
            canvas_box[3] * height / canvas_height,
        ]
        tiles.append(
            {
                "tile_index": index,
                "tile_kind": "local",
                "resized_canvas_xyxy": canvas_box,
                "original_image_xyxy": original_box,
                "patch_grid_height": EXPECTED_PATCH_GRID[0],
                "patch_grid_width": EXPECTED_PATCH_GRID[1],
            }
        )
    if thumbnail_present:
        tiles.append(
            {
                "tile_index": local_count,
                "tile_kind": "thumbnail",
                "resized_canvas_xyxy": [0, 0, canvas_width, canvas_height],
                "original_image_xyxy": [0, 0, width, height],
                "patch_grid_height": EXPECTED_PATCH_GRID[0],
                "patch_grid_width": EXPECTED_PATCH_GRID[1],
            }
        )
    return {
        "original_width": width,
        "original_height": height,
        "tile_size": tile_size,
        "grid_columns": columns,
        "grid_rows": rows,
        "local_tile_count": local_count,
        "thumbnail_present": thumbnail_present,
        "resized_canvas_width": canvas_width,
        "resized_canvas_height": canvas_height,
        "rgb_depth_aligned": True,
        "tile_order": "row_major_local_then_thumbnail",
        "tiles": tiles,
    }


def extract_features(model, rgb_path: Path, depth_path: Path) -> dict:
    import torch
    from PIL import Image as PILImage
    from llava.mm_utils import process_depth, process_image

    with PILImage.open(rgb_path) as rgb_image, PILImage.open(depth_path) as depth_image:
        rgb_size = rgb_image.size
        depth_size = depth_image.size
    if rgb_size != depth_size:
        raise FeatureGateError(f"RGB/depth original size mismatch: {rgb_size} vs {depth_size}")
    tiling = dynamic_tile_metadata(rgb_size[0], rgb_size[1], model)

    synchronize()
    preprocess_started = time.monotonic()
    model.config.image_processor = model.get_vision_tower().image_processor
    rgb_tiles = process_image(
        str(rgb_path), model.config, None, enable_dynamic_res=True
    ).half()
    model.config.image_processor = model.get_depth_tower().image_processor
    depth_tiles = process_depth(
        str(depth_path), model.config, None, enable_dynamic_res=True
    ).half()
    synchronize()
    preprocess_ms = (time.monotonic() - preprocess_started) * 1000.0
    if tuple(rgb_tiles.shape) != tuple(depth_tiles.shape):
        raise FeatureGateError("RGB/depth preprocessed tensor shapes differ")
    if int(rgb_tiles.shape[0]) != len(tiling["tiles"]):
        raise FeatureGateError("tile metadata count differs from upstream preprocessing")

    synchronize()
    feature_started = time.monotonic()
    with torch.inference_mode():
        r0_gpu = model.get_vision_tower()(rgb_tiles)
        d0_gpu = model.get_depth_tower()(depth_tiles)
        projected_rgb = model.get_mm_projector()(r0_gpu)
        projected_depth = model.get_depth_projector()(d0_gpu)
    synchronize()
    feature_ms = (time.monotonic() - feature_started) * 1000.0
    output = {
        "r0": r0_gpu.detach().contiguous().cpu(),
        "d0": d0_gpu.detach().contiguous().cpu(),
        "r0_device": str(r0_gpu.device),
        "d0_device": str(d0_gpu.device),
        "projected_rgb_shape": list(projected_rgb.shape),
        "projected_depth_shape": list(projected_depth.shape),
        "tiling": tiling,
        "preprocess_latency_ms": preprocess_ms,
        "feature_latency_ms": feature_ms,
    }
    del rgb_tiles, depth_tiles, r0_gpu, d0_gpu, projected_rgb, projected_depth
    return output


def runtime_environment(torch_module) -> dict:
    return {
        "gpu_name": torch_module.cuda.get_device_name(0),
        "torch_version": str(torch_module.__version__),
        "cuda_version": str(torch_module.version.cuda) if torch_module.version.cuda else None,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def run_baseline(manifest_path: Path, lock_path: Path, output_root: Path) -> None:
    import torch

    lock = verify_feature_gate_lock(lock_path)
    validate_static(manifest_path, ROOT / "protocol/feature_cache_v1.schema.json", None)
    manifest = read_json(manifest_path)
    run_root = output_root / "baseline"
    if run_root.exists():
        raise FileExistsError(f"baseline output is immutable and already exists: {run_root}")
    run_root.mkdir(parents=True)
    records_root = run_root / "records"
    records_root.mkdir()
    before = verify_protected_inputs(lock)
    model_root = safe_resolve(ROOT, manifest["model_root"])
    inventory = model_inventory_sha256(model_root)
    if inventory != manifest["model_inventory_sha256"]:
        raise FeatureGateError("model inventory differs before baseline")
    seed_runtime(int(manifest["random_seed"]))
    model_load_started = time.monotonic()
    model = load_model(model_root)
    model_load_s = time.monotonic() - model_load_started
    records = []
    dataset_root = safe_resolve(ROOT, manifest["dataset_root"])
    for index, entry in enumerate(manifest["entries"], start=1):
        seed_runtime(int(manifest["random_seed"]))
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        answer, generation_ms = generate_answer(
            model,
            safe_resolve(dataset_root, entry["rgb_path"]),
            safe_resolve(dataset_root, entry["depth_path"]),
            entry["prompt"],
            manifest["generation"],
        )
        parsed = parse_point(answer)
        record = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "run_id": "baseline",
            "sample_id": entry["sample_id"],
            "family_id": entry["family_id"],
            "split": entry["split"],
            "variant": entry["variant"],
            "input": {
                "rgb_sha256": entry["rgb_sha256"],
                "depth_sha256": entry["depth_sha256"],
                "instruction_sha256": sha256_text(entry["instruction"]),
                "request_sha256": sha256_bytes(
                    canonical_json_bytes(
                        {
                            "rgb_sha256": entry["rgb_sha256"],
                            "depth_sha256": entry["depth_sha256"],
                            "prompt": entry["prompt"],
                        }
                    )
                ),
            },
            "raw_answer": answer,
            "raw_answer_sha256": sha256_text(answer),
            **parsed,
            "generation_latency_ms": round(generation_ms, 3),
            "total_latency_ms": round((time.monotonic() - started) * 1000.0, 3),
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "safety": {
                "inference_manifest_only": True,
                "evaluator_artifacts_read": False,
                "training_performed": False,
                "dataset_scaled": False,
                "robot_target_published": False,
            },
        }
        write_json(records_root / f"{entry['sample_id']}.json", record)
        records.append(record)
        print(
            f"WP3_BASELINE {index:02d}/{EXPECTED_SAMPLE_COUNT} "
            f"{entry['sample_id']}: {answer}",
            flush=True,
        )
    after = verify_protected_inputs(lock)
    if before != after:
        raise FeatureGateError("protected inputs changed during baseline")
    run_manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "run_id": "baseline",
        "status": "COMPLETE",
        "created_at_utc": utc_now(),
        "sample_count": len(records),
        "random_seed": manifest["random_seed"],
        "generation": manifest["generation"],
        "model_inventory_sha256": inventory,
        "feature_gate_lock_sha256": sha256_file(lock_path),
        "model_load_s": round(model_load_s, 3),
        "runtime": runtime_environment(torch),
        "protected_inputs_before": before,
        "protected_inputs_after": after,
        "record_hashes": {
            record["sample_id"]: sha256_file(records_root / f"{record['sample_id']}.json")
            for record in records
        },
        "training_performed": False,
        "dataset_scaled": False,
    }
    write_json(run_root / "run_manifest.json", run_manifest)


def run_hook(
    run_id: str,
    manifest_path: Path,
    lock_path: Path,
    schema_path: Path,
    output_root: Path,
) -> None:
    import torch
    from safetensors.torch import save_file

    if run_id not in RUN_IDS:
        raise FeatureGateError(f"invalid hook run ID: {run_id}")
    lock = verify_feature_gate_lock(lock_path)
    validate_static(manifest_path, schema_path, None)
    manifest = read_json(manifest_path)
    run_root = output_root / run_id
    if run_root.exists():
        raise FileExistsError(f"hook output is immutable and already exists: {run_root}")
    run_root.mkdir(parents=True)
    cache_root = run_root / "cache"
    cache_root.mkdir()
    before = verify_protected_inputs(lock)
    model_root = safe_resolve(ROOT, manifest["model_root"])
    inventory = model_inventory_sha256(model_root)
    if inventory != manifest["model_inventory_sha256"]:
        raise FeatureGateError(f"model inventory differs before {run_id}")
    seed_runtime(int(manifest["random_seed"]))
    model_load_started = time.monotonic()
    model = load_model(model_root)
    model_load_s = time.monotonic() - model_load_started
    dataset_root = safe_resolve(ROOT, manifest["dataset_root"])
    lock_sha256 = sha256_file(lock_path)
    contract_sha256 = sha256_file(ROOT / "protocol/WP3_GRAPH_FEATURE_CONTRACT.md")
    root_config_sha256 = sha256_file(model_root / "config.json")
    code_hashes = {
        path: digest
        for path, digest in lock["artifacts"].items()
        if path.endswith(".py")
    }
    cache_records = []
    for index, entry in enumerate(manifest["entries"], start=1):
        seed_runtime(int(manifest["random_seed"]))
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        rgb_path = safe_resolve(dataset_root, entry["rgb_path"])
        depth_path = safe_resolve(dataset_root, entry["depth_path"])
        extracted = extract_features(model, rgb_path, depth_path)
        tensor_path = cache_root / f"{entry['sample_id']}.safetensors"
        save_file(
            {"D0": extracted["d0"], "R0": extracted["r0"]},
            str(tensor_path),
        )
        tensor_hash = sha256_file(tensor_path)
        answer, generation_ms = generate_answer(
            model, rgb_path, depth_path, entry["prompt"], manifest["generation"]
        )
        parsed = parse_point(answer)
        shared_feature = {
            "file": relative(tensor_path, run_root),
            "file_sha256": tensor_hash,
            "shape": list(extracted["r0"].shape),
            "dtype": str(extracted["r0"].dtype),
            "device_at_extraction": extracted["r0_device"],
            "stage": "tower_output_before_projector",
            "patch_grid_height": EXPECTED_PATCH_GRID[0],
            "patch_grid_width": EXPECTED_PATCH_GRID[1],
        }
        request_hash = sha256_bytes(
            canonical_json_bytes(
                {
                    "rgb_sha256": entry["rgb_sha256"],
                    "depth_sha256": entry["depth_sha256"],
                    "prompt": entry["prompt"],
                }
            )
        )
        cache = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "cache_id": f"{run_id}_{entry['sample_id']}",
            "run_id": run_id,
            "sample_id": entry["sample_id"],
            "family_id": entry["family_id"],
            "split": entry["split"],
            "variant": entry["variant"],
            "input": {
                "rgb_sha256": entry["rgb_sha256"],
                "depth_sha256": entry["depth_sha256"],
                "instruction_sha256": sha256_text(entry["instruction"]),
                "request_sha256": request_hash,
            },
            "model": {
                "checkpoint_inventory_sha256": inventory,
                "root_config_sha256": root_config_sha256,
                "feature_contract_sha256": contract_sha256,
                "feature_gate_lock_sha256": lock_sha256,
                "loader_runtime_dtype": str(extracted["r0"].dtype),
                "checkpoint_declared_dtype": "bfloat16",
                "code_hashes": code_hashes,
            },
            "features": {
                "storage_format": "safetensors",
                "shared_file": True,
                "r0": {**shared_feature, "tensor_key": "R0"},
                "d0": {
                    **shared_feature,
                    "tensor_key": "D0",
                    "shape": list(extracted["d0"].shape),
                    "dtype": str(extracted["d0"].dtype),
                    "device_at_extraction": extracted["d0_device"],
                },
                "projected_rgb_shape": extracted["projected_rgb_shape"],
                "projected_depth_shape": extracted["projected_depth_shape"],
            },
            "tiling": extracted["tiling"],
            "baseline_prediction": {
                "raw_answer": answer,
                "raw_answer_sha256": sha256_text(answer),
                **parsed,
            },
            "runtime": {
                "preprocess_latency_ms": round(extracted["preprocess_latency_ms"], 3),
                "feature_latency_ms": round(extracted["feature_latency_ms"], 3),
                "generation_latency_ms": round(generation_ms, 3),
                "total_latency_ms": round((time.monotonic() - started) * 1000.0, 3),
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                **runtime_environment(torch),
            },
            "safety": {
                "inference_manifest_only": True,
                "evaluator_artifacts_read": False,
                "training_performed": False,
                "dataset_scaled": False,
                "robot_target_published": False,
            },
        }
        errors = json_schema_errors(schema_path, cache)
        if errors:
            raise FeatureGateError(
                f"feature cache schema failed for {entry['sample_id']}: {errors[0]}"
            )
        findings = forbidden_cache_findings(cache)
        if findings:
            raise FeatureGateError(f"forbidden cache fields: {findings}")
        cache_path = cache_root / f"{entry['sample_id']}.json"
        write_json(cache_path, cache)
        cache_records.append(cache)
        print(
            f"WP3_{run_id.upper()} {index:02d}/{EXPECTED_SAMPLE_COUNT} "
            f"{entry['sample_id']}: R0={tuple(extracted['r0'].shape)} "
            f"D0={tuple(extracted['d0'].shape)} {answer}",
            flush=True,
        )
        del extracted
    after = verify_protected_inputs(lock)
    if before != after:
        raise FeatureGateError(f"protected inputs changed during {run_id}")
    run_manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "status": "COMPLETE",
        "created_at_utc": utc_now(),
        "sample_count": len(cache_records),
        "random_seed": manifest["random_seed"],
        "generation": manifest["generation"],
        "model_inventory_sha256": inventory,
        "feature_gate_lock_sha256": lock_sha256,
        "model_load_s": round(model_load_s, 3),
        "runtime": runtime_environment(torch),
        "protected_inputs_before": before,
        "protected_inputs_after": after,
        "cache_hashes": {
            cache["sample_id"]: {
                "metadata_sha256": sha256_file(cache_root / f"{cache['sample_id']}.json"),
                "tensor_sha256": cache["features"]["r0"]["file_sha256"],
            }
            for cache in cache_records
        },
        "training_performed": False,
        "dataset_scaled": False,
    }
    write_json(run_root / "run_manifest.json", run_manifest)


def stable_cache_view(cache: dict) -> dict:
    value = copy.deepcopy(cache)
    value.pop("cache_id", None)
    value.pop("run_id", None)
    value.pop("runtime", None)
    return value


def compare_tensors(path_a: Path, path_b: Path) -> dict:
    import torch
    from safetensors.torch import load_file

    tensors_a = load_file(str(path_a), device="cpu")
    tensors_b = load_file(str(path_b), device="cpu")
    if set(tensors_a) != {"R0", "D0"} or set(tensors_b) != {"R0", "D0"}:
        raise FeatureGateError("safetensors payload must contain exactly R0 and D0")
    result = {}
    for key in ("R0", "D0"):
        tensor_a = tensors_a[key]
        tensor_b = tensors_b[key]
        same_shape = tuple(tensor_a.shape) == tuple(tensor_b.shape)
        same_dtype = tensor_a.dtype == tensor_b.dtype
        if same_shape:
            difference = (tensor_a.float() - tensor_b.float()).abs()
            max_absolute = float(difference.max().item()) if difference.numel() else 0.0
            denominator = torch.maximum(tensor_a.float().abs(), torch.full_like(difference, 1e-12))
            max_relative = float((difference / denominator).max().item()) if difference.numel() else 0.0
            exact = bool(torch.equal(tensor_a, tensor_b))
        else:
            max_absolute = math.inf
            max_relative = math.inf
            exact = False
        result[key] = {
            "shape_a": list(tensor_a.shape),
            "shape_b": list(tensor_b.shape),
            "dtype_a": str(tensor_a.dtype),
            "dtype_b": str(tensor_b.dtype),
            "same_shape": same_shape,
            "same_dtype": same_dtype,
            "exact_equal": exact,
            "max_absolute_difference": max_absolute,
            "max_relative_difference": max_relative,
        }
    return result


def build_report(summary: dict, selection: dict) -> str:
    gates = summary["gates"]
    lines = [
        "# WP3 — Feature-hook runtime và determinism smoke",
        "",
        f"> **Kết luận:** `{summary['decision']}`",
        "",
        f"> **Thời điểm UTC:** {summary['generated_at_utc']}",
        "",
        "Protocol chạy đúng 15 sample train thuộc 15 family. Không train, không scale dataset, không đọc evaluator artifact trong baseline/hook và không publish target sang robot.",
        "",
        "## 1. Gate",
        "",
        "| Gate | Kết quả |",
        "|---|---:|",
    ]
    for name, passed in gates.items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    counts = summary["counts"]
    lines.extend(
        [
            "",
            "## 2. Runtime feature contract",
            "",
            f"- Baseline/hook A/hook B: {counts['sample_count']} sample mỗi run;",
            f"- pre-projector `R0/D0`: `{summary['observed']['preprojector_shape_pattern']}`;",
            f"- projector output: `{summary['observed']['projected_shape_pattern']}`;",
            f"- dynamic tiling: {summary['observed']['local_tile_count']} local tiles + {int(summary['observed']['thumbnail_present'])} thumbnail;",
            f"- runtime dtype: `{summary['observed']['runtime_dtype']}`;",
            f"- CuBLAS workspace: `{summary['observed']['cublas_workspace_config']}`;",
            f"- peak VRAM lớn nhất: {summary['observed']['max_peak_vram_bytes'] / 1024**3:.3f} GiB;",
            f"- feature latency median: {summary['observed']['median_feature_latency_ms']:.3f} ms/sample;",
            f"- generation latency median: {summary['observed']['median_generation_latency_ms']:.3f} ms/sample.",
            "",
            "Checkpoint JSON khai báo `bfloat16`, nhưng loader inference đang dùng trong RoboRefer (`llava/model/builder.py`) chuyển model/tower/projector sang `torch.float16`. Smoke ghi runtime truth là FP16 và không tự ý sửa loader hoặc checkpoint sau khi contract đã khóa.",
            "",
            "## 3. Determinism và non-interference",
            "",
            f"- Tensor exact equality: {counts['exact_tensor_pairs']}/{counts['tensor_pairs']};",
            f"- max absolute difference toàn bộ R0/D0: {summary['observed']['global_max_absolute_difference']};",
            f"- max relative difference toàn bộ R0/D0: {summary['observed']['global_max_relative_difference']};",
            f"- safetensors hash giống nhau A/B: {counts['matching_tensor_file_hashes']}/{counts['sample_count']};",
            f"- raw answer baseline = hook A = hook B: {counts['matching_baseline_answers']}/{counts['sample_count']};",
            f"- parsed point giống nhau: {counts['matching_parsed_predictions']}/{counts['sample_count']}.",
            "",
            "## 4. Coverage của 15 mẫu",
            "",
            "Selection audit là evaluator-only và chỉ được đọc ở bước comparison/report, không được loader baseline/hook đọc.",
            "",
            f"- Answerability states: `{selection['coverage']['answerability_states']}`;",
            f"- Family categories: `{selection['coverage']['family_categories']}`;",
            f"- Variants: `{selection['coverage']['variants']}`;",
            f"- Multi-anchor samples: {selection['coverage']['multi_anchor_samples']}.",
            "",
            "## 5. Safety và bất biến",
            "",
            "- Cache chỉ chứa R0/D0, input/checkpoint/code hashes, tile metadata, baseline prediction và runtime measurements;",
            "- không chứa mask, object identity, oracle/relation graph, answerability/source label, expected intervention, correctness, test metric hoặc Gazebo pose;",
            "- WP0 tree, WP1 tree và toàn bộ WP2 dataset tree giữ nguyên digest trước/sau;",
            "- schema/checkpoint/source/manifest đều được khóa SHA-256;",
            "- không có optimizer, backward, dataset generation hay robot command trong protocol.",
            "",
            "## 6. Quyết định",
            "",
        ]
    )
    if summary["decision"] == "GO_PCRA_U_OVERFIT_SMOKE":
        lines.extend(
            [
                "Feature gate WP3 đã pass. Bước kế tiếp được phép là **P-CRA-U overfit smoke trên một train subset rất nhỏ** để kiểm tra loader/loss/gradient.",
                "",
                "Quyết định này **không** cho phép development training, calibration fitting, mở prototype test để chọn model hoặc scale dataset.",
            ]
        )
    else:
        lines.append("Dừng tại `FIX_FEATURE_PIPELINE_FIRST`; chưa được train hoặc scale dataset.")
    return "\n".join(lines) + "\n"


def artifact_manifest(output_root: Path, lock_path: Path) -> dict:
    artifacts = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        artifacts.append(
            {
                "path": relative(path, output_root),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": utc_now(),
        "feature_gate_lock_sha256": sha256_file(lock_path),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def compare_runs(
    manifest_path: Path,
    lock_path: Path,
    schema_path: Path,
    selection_audit_path: Path,
    output_root: Path,
    report_path: Path,
) -> dict:
    import statistics

    lock = verify_feature_gate_lock(lock_path, include_evaluator_artifacts=True)
    static = validate_static(manifest_path, schema_path, selection_audit_path)
    manifest = read_json(manifest_path)
    selection = read_json(selection_audit_path)
    baseline_root = output_root / "baseline"
    run_roots = {run_id: output_root / run_id for run_id in RUN_IDS}
    manifests = {
        "baseline": read_json(baseline_root / "run_manifest.json"),
        **{run_id: read_json(path / "run_manifest.json") for run_id, path in run_roots.items()},
    }
    if any(item.get("status") != "COMPLETE" for item in manifests.values()):
        raise FeatureGateError("baseline/hook run is not COMPLETE")
    if any(item.get("sample_count") != EXPECTED_SAMPLE_COUNT for item in manifests.values()):
        raise FeatureGateError("baseline/hook sample count mismatch")

    comparisons = []
    feature_latencies = []
    generation_latencies = []
    peak_vram = []
    schema_errors = []
    forbidden_findings = []
    for entry in manifest["entries"]:
        sample_id = entry["sample_id"]
        baseline = read_json(baseline_root / "records" / f"{sample_id}.json")
        caches = {
            run_id: read_json(run_roots[run_id] / "cache" / f"{sample_id}.json")
            for run_id in RUN_IDS
        }
        for run_id, cache in caches.items():
            errors = json_schema_errors(schema_path, cache)
            schema_errors.extend(f"{sample_id}/{run_id}: {error}" for error in errors)
            forbidden_findings.extend(
                f"{sample_id}/{run_id}:{item}" for item in forbidden_cache_findings(cache)
            )
            tensor_path = safe_resolve(run_roots[run_id], cache["features"]["r0"]["file"])
            if sha256_file(tensor_path) != cache["features"]["r0"]["file_sha256"]:
                raise FeatureGateError(f"tensor hash mismatch: {sample_id}/{run_id}")
            feature_latencies.append(float(cache["runtime"]["feature_latency_ms"]))
            generation_latencies.append(float(cache["runtime"]["generation_latency_ms"]))
            peak_vram.append(int(cache["runtime"]["peak_vram_bytes"]))
        path_a = safe_resolve(run_roots["hook_a"], caches["hook_a"]["features"]["r0"]["file"])
        path_b = safe_resolve(run_roots["hook_b"], caches["hook_b"]["features"]["r0"]["file"])
        tensor_comparison = compare_tensors(path_a, path_b)
        raw_answers = [
            baseline["raw_answer"],
            caches["hook_a"]["baseline_prediction"]["raw_answer"],
            caches["hook_b"]["baseline_prediction"]["raw_answer"],
        ]
        parsed = [
            (baseline["parse_status"], baseline["normalized_points_xy"]),
            (
                caches["hook_a"]["baseline_prediction"]["parse_status"],
                caches["hook_a"]["baseline_prediction"]["normalized_points_xy"],
            ),
            (
                caches["hook_b"]["baseline_prediction"]["parse_status"],
                caches["hook_b"]["baseline_prediction"]["normalized_points_xy"],
            ),
        ]
        comparison = {
            "sample_id": sample_id,
            "tensor_file_hash_equal": sha256_file(path_a) == sha256_file(path_b),
            "stable_cache_metadata_equal": stable_cache_view(caches["hook_a"])
            == stable_cache_view(caches["hook_b"]),
            "baseline_raw_answer_equal": len(set(raw_answers)) == 1,
            "baseline_parsed_prediction_equal": parsed[0] == parsed[1] == parsed[2],
            "tensor_comparison": tensor_comparison,
        }
        comparisons.append(comparison)

    exact_tensor_pairs = sum(
        item["tensor_comparison"][key]["exact_equal"]
        for item in comparisons
        for key in ("R0", "D0")
    )
    shape_contract = all(
        tuple(item["tensor_comparison"][key]["shape_a"][-2:])
        == EXPECTED_PREPROJECTOR_TAIL
        for item in comparisons
        for key in ("R0", "D0")
    )
    projected_contract = True
    tile_alignment = True
    dtype_alignment = True
    observed_dtype = set()
    observed_local_tiles = set()
    observed_thumbnail = set()
    for entry in manifest["entries"]:
        cache = read_json(run_roots["hook_a"] / "cache" / f"{entry['sample_id']}.json")
        projected_contract &= tuple(cache["features"]["projected_rgb_shape"][-2:]) == EXPECTED_PROJECTED_TAIL
        projected_contract &= tuple(cache["features"]["projected_depth_shape"][-2:]) == EXPECTED_PROJECTED_TAIL
        tile_alignment &= bool(cache["tiling"]["rgb_depth_aligned"])
        tile_alignment &= len(cache["tiling"]["tiles"]) == cache["features"]["r0"]["shape"][0]
        dtype_alignment &= cache["features"]["r0"]["dtype"] == cache["features"]["d0"]["dtype"]
        observed_dtype.add(cache["features"]["r0"]["dtype"])
        observed_local_tiles.add(cache["tiling"]["local_tile_count"])
        observed_thumbnail.add(cache["tiling"]["thumbnail_present"])
    current_protected = verify_protected_inputs(lock)
    protected_runs_match = all(
        item["protected_inputs_before"] == item["protected_inputs_after"] == current_protected
        for item in manifests.values()
    )
    gates = {
        "feature_gate_lock_verified": True,
        "exactly_15_train_samples_from_15_families": static["sample_count"] == 15
        and static["distinct_family_count"] == 15
        and static["all_train"],
        "all_three_runs_complete": True,
        "feature_cache_schema_valid": not schema_errors,
        "inference_cache_oracle_free": not forbidden_findings,
        "preprojector_shape_contract": shape_contract,
        "projected_shape_contract": projected_contract,
        "rgb_depth_tile_alignment": tile_alignment,
        "runtime_dtype_aligned_and_documented": dtype_alignment
        and observed_dtype.issubset({"torch.float16", "torch.bfloat16"}),
        "hook_a_b_tensor_exact_equality": exact_tensor_pairs == EXPECTED_SAMPLE_COUNT * 2,
        "hook_a_b_tensor_file_hash_equality": all(
            item["tensor_file_hash_equal"] for item in comparisons
        ),
        "hook_a_b_stable_metadata_equality": all(
            item["stable_cache_metadata_equal"] for item in comparisons
        ),
        "hook_does_not_change_baseline_raw_answer": all(
            item["baseline_raw_answer_equal"] for item in comparisons
        ),
        "hook_does_not_change_parsed_prediction": all(
            item["baseline_parsed_prediction_equal"] for item in comparisons
        ),
        "wp0_wp1_wp2_inputs_unchanged": protected_runs_match,
        "no_training_no_dataset_scaling": all(
            not item["training_performed"] and not item["dataset_scaled"]
            for item in manifests.values()
        ),
    }
    decision = "GO_PCRA_U_OVERFIT_SMOKE" if all(gates.values()) else "FIX_FEATURE_PIPELINE_FIRST"
    global_max_abs = max(
        item["tensor_comparison"][key]["max_absolute_difference"]
        for item in comparisons
        for key in ("R0", "D0")
    )
    global_max_rel = max(
        item["tensor_comparison"][key]["max_relative_difference"]
        for item in comparisons
        for key in ("R0", "D0")
    )
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": utc_now(),
        "decision": decision,
        "gates": gates,
        "counts": {
            "sample_count": EXPECTED_SAMPLE_COUNT,
            "family_count": EXPECTED_SAMPLE_COUNT,
            "tensor_pairs": EXPECTED_SAMPLE_COUNT * 2,
            "exact_tensor_pairs": exact_tensor_pairs,
            "matching_tensor_file_hashes": sum(
                item["tensor_file_hash_equal"] for item in comparisons
            ),
            "matching_baseline_answers": sum(
                item["baseline_raw_answer_equal"] for item in comparisons
            ),
            "matching_parsed_predictions": sum(
                item["baseline_parsed_prediction_equal"] for item in comparisons
            ),
            "schema_errors": len(schema_errors),
            "forbidden_cache_findings": len(forbidden_findings),
        },
        "observed": {
            "preprojector_shape_pattern": "[N_tiles, 1024, 1152]",
            "projected_shape_pattern": "[N_tiles, 121, 1536]",
            "runtime_dtype": sorted(observed_dtype),
            "checkpoint_declared_dtype": "bfloat16",
            "loader_behavior": "load_pretrained_model coerces tower/projector to torch.float16",
            "cublas_workspace_config": EXPECTED_CUBLAS_WORKSPACE_CONFIG,
            "local_tile_count": sorted(observed_local_tiles),
            "thumbnail_present": all(observed_thumbnail),
            "global_max_absolute_difference": global_max_abs,
            "global_max_relative_difference": global_max_rel,
            "median_feature_latency_ms": statistics.median(feature_latencies),
            "median_generation_latency_ms": statistics.median(generation_latencies),
            "max_peak_vram_bytes": max(peak_vram),
        },
        "schema_errors": schema_errors,
        "forbidden_cache_findings": forbidden_findings,
        "protected_inputs": current_protected,
        "feature_gate_lock_sha256": sha256_file(lock_path),
        "comparisons": comparisons,
        "safety": {
            "training_performed": False,
            "dataset_scaled": False,
            "calibration_or_test_read": False,
            "robot_target_published": False,
        },
    }
    write_json(output_root / "comparison_summary.json", summary)
    report = build_report(summary, selection)
    (output_root / "WP3_FEATURE_HOOK_REPORT.md").write_text(report, encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    write_json(output_root / "artifact_manifest.json", artifact_manifest(output_root, lock_path))
    print(
        json.dumps(
            {
                "decision": decision,
                "gates": gates,
                "counts": summary["counts"],
                "observed": summary["observed"],
            },
            indent=2,
        )
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    static_parser = subparsers.add_parser("validate-static")
    static_parser.add_argument("--manifest", type=Path, required=True)
    static_parser.add_argument("--schema", type=Path, required=True)
    static_parser.add_argument("--selection-audit", type=Path, required=True)

    for command in ("baseline", "hook"):
        child = subparsers.add_parser(command)
        child.add_argument("--manifest", type=Path, required=True)
        child.add_argument("--lock", type=Path, required=True)
        child.add_argument("--output-root", type=Path, required=True)
        if command == "hook":
            child.add_argument("--schema", type=Path, required=True)
            child.add_argument("--run-id", choices=RUN_IDS, required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--manifest", type=Path, required=True)
    compare_parser.add_argument("--lock", type=Path, required=True)
    compare_parser.add_argument("--schema", type=Path, required=True)
    compare_parser.add_argument("--selection-audit", type=Path, required=True)
    compare_parser.add_argument("--output-root", type=Path, required=True)
    compare_parser.add_argument("--report-md", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-static":
        result = validate_static(
            args.manifest.resolve(), args.schema.resolve(), args.selection_audit.resolve()
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "baseline":
        run_baseline(args.manifest.resolve(), args.lock.resolve(), args.output_root.resolve())
        return 0
    if args.command == "hook":
        run_hook(
            args.run_id,
            args.manifest.resolve(),
            args.lock.resolve(),
            args.schema.resolve(),
            args.output_root.resolve(),
        )
        return 0
    summary = compare_runs(
        args.manifest.resolve(),
        args.lock.resolve(),
        args.schema.resolve(),
        args.selection_audit.resolve(),
        args.output_root.resolve(),
        args.report_md.resolve(),
    )
    return 0 if summary["decision"] == "GO_PCRA_U_OVERFIT_SMOKE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
