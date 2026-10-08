#!/usr/bin/env python3
"""Publish lightweight, human-readable experiment evidence into ketqua/."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = WORKSPACE / "protocol/ketqua_catalog_v1.json"
ALLOWED_EXTENSIONS = {
    ".csv", ".json", ".jsonl", ".md", ".html", ".htm", ".svg",
    ".png", ".jpg", ".jpeg", ".txt", ".log",
}
FORBIDDEN_EXTENSIONS = {".safetensors", ".pt", ".pth", ".ckpt", ".npy", ".npz", ".bin"}
MAX_FILE_BYTES = 50 * 1024 * 1024
VERIFY_TOKENS = {"manifest", "inventory", "lock", "check", "schema", "gate_report"}


class PublishError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PublishError(f"Expected JSON object: {path}")
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copy2(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def sanitize(value: str) -> str:
    output = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    if not output:
        raise PublishError(f"Cannot sanitize empty label: {value!r}")
    return output


def category(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".svg"}:
        return "hinh_anh"
    if suffix in {".md", ".html", ".htm", ".txt", ".log"}:
        return "bao_cao"
    if any(token in name for token in VERIFY_TOKENS):
        return "kiem_tra_manifest"
    return "bang_so_lieu"


def source_files(source: Path) -> list[tuple[Path, Path]]:
    if source.is_file():
        candidates = [(source, Path(source.name))]
    elif source.is_dir():
        candidates = [(path, path.relative_to(source)) for path in sorted(source.rglob("*")) if path.is_file()]
    else:
        raise PublishError(f"Source does not exist: {source}")
    selected = []
    for path, relative in candidates:
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_EXTENSIONS:
            continue
        if suffix not in ALLOWED_EXTENSIONS:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            raise PublishError(f"Readable artifact exceeds {MAX_FILE_BYTES} bytes: {path}")
        selected.append((path, relative))
    return selected


def folder_name(experiment: dict[str, Any]) -> str:
    return f"{int(experiment['order']):02d}_{experiment['date']}_{experiment['slug']}"


def validate_catalog(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    if catalog.get("schema_version") != 1:
        raise PublishError("Unsupported catalog schema")
    experiments = catalog.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise PublishError("Catalog must contain experiments")
    orders = [int(item["order"]) for item in experiments]
    if len(set(orders)) != len(orders) or sorted(orders) != orders:
        raise PublishError("Experiment orders must be unique and ascending")
    names = [folder_name(item) for item in experiments]
    if len(set(names)) != len(names):
        raise PublishError("Experiment folder names must be unique")
    return experiments


def publish_experiment(root: Path, experiment: dict[str, Any], update: bool) -> dict[str, Any]:
    directory = root / folder_name(experiment)
    if directory.exists() and not update:
        raise PublishError(f"Experiment output exists; pass --update to refresh derived copies: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    category_notes = {
        "bang_so_lieu": "CSV/JSON/JSONL và số liệu máy đọc được của thí nghiệm.",
        "hinh_anh": "PNG/JPG/SVG dùng để xem trực quan; giữ relative path của source.",
        "bao_cao": "Markdown/HTML/log/text dùng để đọc kết luận và diễn giải.",
        "kiem_tra_manifest": "Manifest/lock/schema/check dùng để kiểm tra provenance và hash.",
    }
    for category_name, note in category_notes.items():
        atomic_text(directory / category_name / "README.md", f"# {category_name}\n\n{note}\n")
    copied = []
    for source_spec in experiment["sources"]:
        label = sanitize(source_spec["label"])
        source_root = (WORKSPACE / source_spec["path"]).resolve()
        if not source_root.is_relative_to(WORKSPACE):
            raise PublishError(f"Source escapes workspace: {source_root}")
        for source, relative in source_files(source_root):
            destination_relative = Path(category(source)) / label / relative
            destination = directory / destination_relative
            source_digest = sha256(source)
            if destination.exists() and update and sha256(destination) == source_digest:
                pass
            else:
                atomic_copy(source, destination)
            destination_digest = sha256(destination)
            if destination_digest != source_digest:
                raise PublishError(f"Copy hash mismatch: {source} -> {destination}")
            copied.append({
                "source": str(source.relative_to(WORKSPACE)),
                "destination": str(destination_relative),
                "category": category(source),
                "bytes": source.stat().st_size,
                "sha256": source_digest,
            })
    counts = Counter(item["category"] for item in copied)
    total_bytes = sum(item["bytes"] for item in copied)
    info = {
        "schema_version": 1,
        "order": int(experiment["order"]),
        "date": experiment["date"],
        "time": experiment["time"],
        "slug": experiment["slug"],
        "title": experiment["title"],
        "status": experiment["status"],
        "role": experiment["role"],
        "folder": directory.name,
        "source_paths": [item["path"] for item in experiment["sources"]],
        "artifact_count": len(copied),
        "category_counts": dict(sorted(counts.items())),
        "total_bytes": total_bytes,
        "large_artifacts_copied": False,
    }
    atomic_json(directory / "THONG_TIN_THI_NGHIEM.json", info)
    atomic_json(directory / "NGUON_ARTIFACT_MANIFEST.json", {
        "schema_version": 1,
        "experiment_folder": directory.name,
        "artifact_count": len(copied),
        "artifacts": copied,
    })
    warning = ""
    if experiment["status"].startswith("FAILED") or experiment["status"].startswith("ABORTED"):
        warning = "\n> Đây là lần chạy lỗi/aborted được giữ để truy nguyên; không dùng như kết quả chính thức.\n"
    elif experiment["status"] == "PLANNED_NOT_RUN":
        warning = "\n> Đây chỉ là scaffold đã lên kế hoạch; các ô `NOT_RUN` chưa phải kết quả đo.\n"
    elif experiment["role"].startswith("official_"):
        warning = "\n> Đây là kết quả gate chính thức.\n"
    readme = [
        f"# {int(experiment['order']):02d} — {experiment['title']}", "",
        f"- Ngày/giờ định danh: `{experiment['date']} {experiment['time']}`;",
        f"- Trạng thái: `{experiment['status']}`;",
        f"- Vai trò: `{experiment['role']}`;",
        f"- Artifact đọc được: {len(copied)} file / {total_bytes / (1024 ** 2):.2f} MiB;",
        "- Tensor/checkpoint/cache lớn được sao chép: `NO`.",
        warning,
        "## Cấu trúc", "",
        "- `bang_so_lieu/`: CSV, JSON, JSONL;",
        "- `hinh_anh/`: PNG, JPG, SVG;",
        "- `bao_cao/`: Markdown, HTML, log/text;",
        "- `kiem_tra_manifest/`: manifest, lock, schema và check JSON;",
        "- `NGUON_ARTIFACT_MANIFEST.json`: source path và SHA-256 từng file.", "",
        "## Nguồn", "",
    ]
    readme.extend(f"- `{item['path']}`" for item in experiment["sources"])
    atomic_text(directory / "README.md", "\n".join(readme) + "\n")
    atomic_json(directory / "KIEM_TRA_SAO_CHEP.json", {
        "schema_version": 1,
        "experiment_folder": directory.name,
        "source_artifact_count": len(copied),
        "destination_artifact_count": sum(1 for item in copied if (directory / item["destination"]).is_file()),
        "source_destination_hashes_match": all(
            sha256(WORKSPACE / item["source"]) == sha256(directory / item["destination"]) for item in copied
        ),
        "forbidden_large_extensions_copied": [
            str(path.relative_to(directory)) for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in FORBIDDEN_EXTENSIONS
        ],
        "passed": bool(copied) and all(
            sha256(WORKSPACE / item["source"]) == sha256(directory / item["destination"]) for item in copied
        ),
    })
    manifest_entries = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "MANIFEST.json"):
        manifest_entries.append({"path": str(path.relative_to(directory)), "bytes": path.stat().st_size,
                                 "sha256": sha256(path)})
    atomic_json(directory / "MANIFEST.json", {
        "schema_version": 1,
        "experiment_folder": directory.name,
        "artifact_count": len(manifest_entries),
        "artifacts": manifest_entries,
    })
    return info


def verify_manifest(base: Path, manifest_path: Path) -> list[str]:
    manifest = read_json(manifest_path)
    mismatches = []
    for item in manifest.get("artifacts", []):
        path = base / item["path"]
        if (not path.is_file() or path.stat().st_size != item["bytes"]
                or sha256(path) != item["sha256"]):
            mismatches.append(item["path"])
    return mismatches


def publish(catalog_path: Path, update: bool, only_order: int | None = None) -> dict[str, Any]:
    catalog = read_json(catalog_path)
    experiments = validate_catalog(catalog)
    root = (WORKSPACE / catalog.get("output_root", "ketqua")).resolve()
    if not root.is_relative_to(WORKSPACE) or root == WORKSPACE:
        raise PublishError(f"Unsafe output root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()) and not update and only_order is None:
        raise PublishError(f"Output root is not empty; pass --update to refresh: {root}")
    if only_order is None:
        records = [publish_experiment(root, experiment, update) for experiment in experiments]
    else:
        selected = [item for item in experiments if int(item["order"]) == only_order]
        if len(selected) != 1:
            raise PublishError(f"Catalog must contain exactly one experiment with order={only_order}")
        publish_experiment(root, selected[0], update)
        records = []
        for experiment in experiments:
            info_path = root / folder_name(experiment) / "THONG_TIN_THI_NGHIEM.json"
            if not info_path.is_file():
                raise PublishError(
                    f"Cannot rebuild catalog because experiment metadata is missing: {info_path}"
                )
            records.append(read_json(info_path))
    fields = ["order", "date", "time", "folder", "title", "status", "role", "artifact_count", "total_bytes"]
    write_csv(root / "DANH_MUC_THI_NGHIEM.csv", fields, records)
    atomic_json(root / "DANH_MUC_THI_NGHIEM.json", {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_count": len(records),
        "experiments": records,
    })
    official_records = [item for item in records if item["role"].startswith("official_")]
    if not official_records:
        raise PublishError("Catalog must contain at least one role beginning with official_")
    official = official_records[-1]
    latest = records[-1]
    atomic_text(root / "LATEST.md", (
        f"# Mốc mới nhất theo thời gian\n\n- [`{latest['folder']}`]({latest['folder']}/README.md)\n"
        f"- Status: `{latest['status']}`\n\nMốc này là planned nếu status là `PLANNED_NOT_RUN`.\n"
    ))
    atomic_text(root / "LATEST_OFFICIAL.md", (
        f"# Kết quả chính thức mới nhất\n\n- [`{official['folder']}`]({official['folder']}/README.md)\n"
        f"- Status: `{official['status']}`\n"
    ))
    readme = [
        "# Kết quả thí nghiệm", "",
        "Đây là thư mục trình bày tập trung. Source run trong `results/`, `datasets/` và `protocol/` vẫn giữ nguyên để bảo toàn provenance/hash.", "",
        "## Cách tìm", "",
        "Tên folder có dạng `STT_YYYYMMDD_ten_thi_nghiem`. Xem `DANH_MUC_THI_NGHIEM.csv` để lọc theo ngày, status và role.", "",
        "- `LATEST.md`: mốc mới nhất theo thời gian, có thể chỉ là planned;",
        "- `LATEST_OFFICIAL.md`: kết quả official mới nhất;",
        "- trong mỗi thí nghiệm: bảng, hình, báo cáo và manifest được chia riêng;",
        "- không chứa `.safetensors`, `.pt`, `.npy`, checkpoint hoặc raw cache lớn.", "",
        "## Thứ tự", "",
    ]
    readme.extend(
        f"{item['order']}. [`{item['folder']}`]({item['folder']}/README.md) — `{item['status']}`"
        for item in records
    )
    readme += ["", "## Cập nhật", "", "```bash", "python protocol/publish_ketqua.py --update", "```", "",
               "Thêm thí nghiệm mới vào `protocol/ketqua_catalog_v1.json` với STT/ngày/giờ/slug duy nhất trước khi cập nhật."]
    atomic_text(root / "README.md", "\n".join(readme) + "\n")
    top_entries = []
    excluded = {"KETQUA_MANIFEST.json", "KETQUA_CHECK_REPORT.json"}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in excluded):
        top_entries.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                            "sha256": sha256(path)})
    atomic_json(root / "KETQUA_MANIFEST.json", {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(top_entries),
        "artifacts": top_entries,
    })
    experiment_mismatches = {}
    for item in records:
        directory = root / item["folder"]
        mismatch = verify_manifest(directory, directory / "MANIFEST.json")
        if mismatch:
            experiment_mismatches[item["folder"]] = mismatch
    top_mismatches = verify_manifest(root, root / "KETQUA_MANIFEST.json")
    forbidden = [str(path.relative_to(root)) for path in root.rglob("*")
                 if path.is_file() and path.suffix.lower() in FORBIDDEN_EXTENSIONS]
    check = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_count": len(records),
        "experiment_manifest_mismatches": experiment_mismatches,
        "top_manifest_mismatches": top_mismatches,
        "forbidden_large_artifacts": forbidden,
        "official_run_count": len(official_records),
        "planned_not_run_count": sum(item["status"] == "PLANNED_NOT_RUN" for item in records),
        "passed": not experiment_mismatches and not top_mismatches and not forbidden,
    }
    atomic_json(root / "KETQUA_CHECK_REPORT.json", check)
    return {"root": str(root.relative_to(WORKSPACE)), "experiments": len(records),
            "artifacts": len(top_entries), "bytes": sum(item["bytes"] for item in top_entries),
            "check_passed": check["passed"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG.relative_to(WORKSPACE)))
    parser.add_argument("--update", action="store_true", help="Refresh listed derived copies; never delete source runs")
    parser.add_argument(
        "--only-order", type=int,
        help="Publish one catalog order and rebuild top-level indexes without refreshing older folders",
    )
    args = parser.parse_args()
    result = publish((WORKSPACE / args.catalog).resolve(), args.update, args.only_order)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
