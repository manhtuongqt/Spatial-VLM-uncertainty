#!/usr/bin/env python3
"""Shared contracts and deterministic helpers for WP2 Dataset v1."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_ID = "roborefer_dataset_v1_prototype"
SCHEMA_VERSION = 1
GENERATOR_VERSION = "wp2_family_generator_v1.0.0"
RANDOM_SEED = 18082026
PILOT_RELATIVE = "results/roborefer_pilot_v0_20260813_173305"
WP1_RELATIVE = "results/roborefer_depth_sensitivity_v1_20260818_152700"
EXPECTED_PILOT_TREE_SHA256 = (
    "df8ec334c82b14447463101cccb5e54836fb27e579a760322256f84d157a2b9c"
)
EXPECTED_WP1_TREE_SHA256 = (
    "857e6b5e169b933572a41fb7251c9cbc78a4146d99f0f76845b27e9187472a9c"
)
EXPECTED_MODEL_INVENTORY_SHA256 = (
    "5508fb49888c1f62b875d928266f7fff3e1224c795f6fc8b971631143a45a7fa"
)

VARIANTS = (
    "clean",
    "semantic_counterfactual",
    "relation_counterfactual",
    "depth_corruption",
    "occlusion_view_counterfactual",
)
SPLITS = ("train", "dev", "calibration", "test")

# This is intentionally stricter than a string search over a serialized payload.
# Keys are normalized before matching, so target_mask and Target-Mask are equal.
FORBIDDEN_INFERENCE_KEY_TOKENS = {
    "anchorid",
    "anchorids",
    "anchormask",
    "anchormasks",
    "answerable",
    "answerability",
    "expectedintervention",
    "groundtruth",
    "graspable",
    "graspablemask",
    "interiormask",
    "oracle",
    "reachable",
    "reachablemask",
    "relationgraph",
    "semanticlabels",
    "target",
    "targetid",
    "targetids",
    "targetmask",
    "uncertainty",
    "validtargetids",
}


class WP2Error(RuntimeError):
    """Raised when a WP2 invariant is violated."""


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WP2Error(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WP2Error(f"JSON root must be an object: {path}")
    return value


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for record in records:
            stream.write(canonical_json_bytes(record))


def read_jsonl(path: Path) -> list[dict]:
    values = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WP2Error(f"invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise WP2Error(f"record is not an object: {path}:{line_number}")
            values.append(item)
    return values


def normalize_key(value: str) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def find_forbidden_inference_keys(value: Any, prefix: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            child_path = f"{prefix}.{raw_key}"
            if normalize_key(str(raw_key)) in FORBIDDEN_INFERENCE_KEY_TOKENS:
                findings.append(child_path)
            findings.extend(find_forbidden_inference_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(find_forbidden_inference_keys(child, f"{prefix}[{index}]"))
    return findings


def tree_digest(workspace: Path, relative_root: str) -> str:
    """Match the historical `find|sort|sha256sum|sha256sum` digest exactly."""

    root = workspace / relative_root
    if not root.is_dir():
        raise WP2Error(f"tree does not exist: {root}")
    relative_paths = [
        str(path.relative_to(workspace)) for path in root.rglob("*") if path.is_file()
    ]
    sorted_bytes = subprocess.run(
        ["sort", "-z"],
        input=b"\0".join(value.encode("utf-8") for value in relative_paths) + b"\0",
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    digest = hashlib.sha256()
    for raw_relative in sorted_bytes.rstrip(b"\0").split(b"\0"):
        relative = raw_relative.decode("utf-8")
        digest.update(f"{sha256_file(workspace / relative)}  {relative}\n".encode())
    return digest.hexdigest()


def artifact_entry(path: Path, root: Path) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def file_inventory(root: Path, excluded_names: Iterable[str] = ()) -> list[dict]:
    excluded = set(excluded_names)
    values = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in excluded:
            continue
        values.append(artifact_entry(path, root))
    return values


def ensure_new_directory(path: Path) -> None:
    if path.exists():
        raise WP2Error(f"immutable output already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)


def utc_now() -> str:
    # Respect SOURCE_DATE_EPOCH only in tests/replay harnesses.
    import datetime

    fixed = os.environ.get("SOURCE_DATE_EPOCH")
    if fixed:
        return datetime.datetime.fromtimestamp(
            int(fixed), tz=datetime.timezone.utc
        ).isoformat()
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
