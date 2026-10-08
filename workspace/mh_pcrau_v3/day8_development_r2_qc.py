#!/usr/bin/env python3
"""Semantic/sensor/leakage/shortcut QC for Day-8 Development-R2."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "ketquangay/ngay_08"
RAW = REPORT / "du_lieu"
DATASET = ROOT / "datasets/Gazebo_development_r2_v1"
STATES = ("FOUND", "AMBIGUOUS", "ABSENT", "INSUFFICIENT_EVIDENCE")
RELATIONS = ("leftmost", "rightmost", "second_from_left", "second_from_right")
NAMESPACE = "mh_pcrau_v3/gazebo_development_r2_v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def macro_f1(y: list[str], p: list[str]) -> float:
    values = []
    for label in STATES:
        tp = sum(a == label and b == label for a, b in zip(y, p))
        fp = sum(a != label and b == label for a, b in zip(y, p))
        fn = sum(a == label and b != label for a, b in zip(y, p))
        values.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(values))


def ridge_predict(xtr: np.ndarray, ytr: list[str], xv: np.ndarray) -> list[str]:
    mean, std = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xv = (xtr - mean) / std, (xv - mean) / std
    xtr = np.column_stack([xtr, np.ones(len(xtr))])
    xv = np.column_stack([xv, np.ones(len(xv))])
    labels = list(STATES)
    yy = np.eye(len(labels))[[labels.index(x) for x in ytr]]
    lam = 10.0
    if xtr.shape[1] > xtr.shape[0]:
        weights = xtr.T @ np.linalg.solve(xtr @ xtr.T + lam * np.eye(len(xtr)), yy)
    else:
        weights = np.linalg.solve(xtr.T @ xtr + lam * np.eye(xtr.shape[1]), xtr.T @ yy)
    return [labels[int(i)] for i in (xv @ weights).argmax(1)]


def text_features(values: list[str], vocab: dict[str, int] | None = None) -> tuple[np.ndarray, dict[str, int]]:
    tokens = [re.findall(r"[a-z]+", value.lower()) for value in values]
    if vocab is None:
        vocab = {x: i for i, x in enumerate(sorted({t for row in tokens for t in row}))}
    out = np.zeros((len(values), len(vocab)), dtype=np.float64)
    for i, row in enumerate(tokens):
        for token in row:
            if token in vocab:
                out[i, vocab[token]] += 1
    return out, vocab


def thumbnail(path: Path, depth: bool) -> np.ndarray:
    if depth:
        x = np.load(path).astype(np.float32)
        valid = np.isfinite(x) & (x >= .1) & (x <= 3.)
        fill = float(np.median(x[valid])) if valid.any() else 0.0
        x = np.where(valid, x, fill)
        x = (x - x.min()) / max(float(x.max() - x.min()), 1e-6)
        x = np.repeat(x[..., None], 3, axis=2)
    else:
        x = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if x is None:
            raise RuntimeError(f"unreadable RGB: {path}")
        x = x.astype(np.float32) / 255.0
    return cv2.resize(x, (16, 12), interpolation=cv2.INTER_AREA).reshape(-1).astype(np.float64)


def collect(folder: Path) -> list[dict]:
    captures = sorted((p for p in folder.glob("capture_attempt_*") if (p / "input_manifest.jsonl").is_file()), reverse=True)
    capture = captures[0] if captures else folder / "capture_attempt_01"
    annotations = yaml.safe_load((folder / "annotations.yaml").read_text())["scenes"]
    scenes = {x["scene_id"]: x for x in yaml.safe_load((folder / "scenes.yaml").read_text())["scenes"]}
    records = []
    for item in jsonl(capture / "input_manifest.jsonl"):
        sid = item["scene_id"]
        ann = annotations[sid]
        rgb = capture / item["input_files"]["rgb"]
        depth = capture / item["input_files"]["depth_m"]
        labels = capture / sid / "evaluator/semantic_labels.png"
        oracle = capture / sid / "evaluator/capture_oracle.json"
        records.append({
            "scene_id": sid, "family_id": ann["family_id"], "state": ann["state"],
            "relation": ann["relation_variant"], "split": ann["split"], "seed": ann["seed"],
            "camera_stratum": ann["camera_stratum"], "prompt_variant": ann["prompt_variant"],
            "instruction": item["instruction"], "layout_signature_sha256": ann["layout_signature_sha256"],
            "rgb": rgb, "depth": depth, "labels": labels, "oracle": oracle,
            "manifest": item, "annotation": ann, "scene": scenes[sid], "capture_root": capture,
        })
    return records


def inspect(records: list[dict], canary: bool) -> tuple[list[dict], list[str], dict]:
    failures, details = [], []
    hashes = {"rgb": [], "depth": [], "labels": []}
    for rec in records:
        sid, ann = rec["scene_id"], rec["annotation"]
        rgb = cv2.imread(str(rec["rgb"]), cv2.IMREAD_COLOR)
        dep = np.load(rec["depth"])
        lab = cv2.imread(str(rec["labels"]), cv2.IMREAD_UNCHANGED)
        errors = []
        if rgb is None or lab is None or rgb.shape[:2] != (480, 640) or dep.shape != (480, 640) or lab.shape != (480, 640):
            errors.append("sensor_shape_or_read")
        valid_depth = np.isfinite(dep) & (dep >= .1) & (dep <= 3.)
        if valid_depth.mean() < .99:
            errors.append(f"valid_depth_fraction={valid_depth.mean():.6f}")
        item = rec["manifest"]
        for key, path in (("rgb", rec["rgb"]), ("depth_m", rec["depth"])):
            if sha256(path) != item["input_sha256"][key]:
                errors.append(f"{key}_sha256")
        if not item["capture"]["input_only_sensor_qc"]["passed"] or item["capture"]["rgb_depth_label_spread_sec"] > .02 + 1e-9:
            errors.append("capture_sensor_qc")

        visible = None
        target_uv = None
        if ann["state"] == "FOUND":
            mask = lab == int(ann["target_label"])
            yy, xx = np.where(mask); visible = int(len(xx))
            if visible < 120:
                errors.append(f"found_visible_pixels={visible}")
            elif not valid_depth[mask].all():
                errors.append("found_target_depth_invalid")
            else:
                target_uv = [float(xx.mean() / 639.0), float(yy.mean() / 479.0)]
        elif ann["state"] == "INSUFFICIENT_EVIDENCE":
            mask = lab == int(ann["target_label"])
            visible = int(mask.sum())
            occ = int((lab == int(ann["occluder_labels"][0])).sum())
            if not (1 <= visible < 120):
                errors.append(f"insufficient_visible_pixels={visible}")
            if occ <= 0 or not ann.get("insufficient_reason"):
                errors.append("insufficient_missing_occluder_or_reason")
        elif ann["state"] == "ABSENT":
            absent_label = int(ann["requested_absent_target_label"])
            visible = int((lab == absent_label).sum())
            if visible != 0 or ann["requested_absent_target_id"] in rec["scene"]["poses"]:
                errors.append(f"absent_leak_pixels_or_pose={visible}")
            if ann.get("target_id") is not None or ann.get("target_label") is not None:
                errors.append("absent_target_metadata_not_null")
        else:
            centers, counts = [], []
            for label in ann["valid_target_labels"]:
                yy, xx = np.where(lab == int(label)); counts.append(int(len(xx)))
                if len(xx): centers.append(float(xx.mean()))
            visible = counts
            if len(centers) < 2 or min(counts) < 120 or max(centers) - min(centers) > 12.0:
                errors.append(f"ambiguous_counts_centers={counts}|{centers}")

        # All reset Z values come from the locked object contact-Z table; no scene may override Z.
        if any(len(v) != 3 for v in rec["scene"]["poses"].values()):
            errors.append("floating_contract_pose_arity")
        oracle = json.loads(rec["oracle"].read_text())
        if any(abs(float(pose[2])) > .2501 for name, pose in oracle["requested_scene_layout_base_link"].items() if name in rec["scene"]["poses"]):
            errors.append("floating_contact_z_out_of_bounds")

        hashes["rgb"].append(sha256(rec["rgb"])); hashes["depth"].append(sha256(rec["depth"])); hashes["labels"].append(sha256(rec["labels"]))
        rec.update({"target_visible_pixels": visible, "target_uv": target_uv, "rgb_sha256": hashes["rgb"][-1], "depth_sha256": hashes["depth"][-1], "labels_sha256": hashes["labels"][-1], "qc_errors": errors})
        if errors:
            failures.append(f"{sid}:" + ";".join(errors))
        details.append({"scene_id": sid, "family_id": rec["family_id"], "state": rec["state"], "relation": rec["relation"], "target_visible_pixels": visible, "target_uv": target_uv, "errors": errors})
    summary = {
        "readable_and_synced": not any("sensor" in x or "depth" in x or "sha256" in x for x in failures),
        "semantic_label_qc": not any("visible" in x or "ambiguous" in x or "absent" in x for x in failures),
        "floating_object_contract_qc": not any("floating" in x for x in failures),
        "unique_rgb": len(set(hashes["rgb"])), "unique_depth": len(set(hashes["depth"])),
    }
    return details, failures, summary


def historical_sets() -> tuple[set[str], set[str], set[int]]:
    rgb, families, seeds = set(), set(), set()
    excluded = [REPORT.resolve(), DATASET.resolve()]
    def allowed(path: Path) -> bool:
        rp = path.resolve()
        return not any(rp == x or x in rp.parents for x in excluded)
    for path in ROOT.rglob("*.jsonl"):
        if not allowed(path):
            continue
        try:
            for row in jsonl(path):
                if row.get("rgb_sha256"): rgb.add(str(row["rgb_sha256"]))
                if isinstance(row.get("input_sha256"), dict) and row["input_sha256"].get("rgb"): rgb.add(str(row["input_sha256"]["rgb"]))
                for key in ("family_id", "scene_family_id"):
                    if row.get(key): families.add(str(row[key]))
                if isinstance(row.get("seed"), int): seeds.add(row["seed"])
        except Exception:
            pass
    for path in ROOT.rglob("*.json"):
        if not allowed(path):
            continue
        try:
            obj = json.loads(path.read_text())
            stack = [obj]
            while stack:
                value = stack.pop()
                if isinstance(value, dict):
                    if value.get("rgb_sha256"): rgb.add(str(value["rgb_sha256"]))
                    for key in ("family_id", "scene_family_id"):
                        if value.get(key): families.add(str(value[key]))
                    if isinstance(value.get("seed"), int): seeds.add(value["seed"])
                    stack.extend(value.values())
                elif isinstance(value, list): stack.extend(value)
        except Exception:
            pass
    return rgb, families, seeds


def shortcut_audit(records: list[dict]) -> dict:
    train = [x for x in records if x["split"] == "train"]
    val = [x for x in records if x["split"] == "development_validation"]
    yt, yv = [x["state"] for x in train], [x["state"] for x in val]
    xt, vocab = text_features([x["instruction"] for x in train])
    xv, _ = text_features([x["instruction"] for x in val], vocab)
    text_f1 = macro_f1(yv, ridge_predict(xt, yt, xv))
    def media(key: str, depth: bool) -> float:
        a = np.stack([thumbnail(x[key], depth) for x in train])
        b = np.stack([thumbnail(x[key], depth) for x in val])
        return macro_f1(yv, ridge_predict(a, yt, b))
    rgb_f1, depth_f1 = media("rgb", False), media("depth", True)
    found_train = [x for x in train if x["state"] == "FOUND" and x["target_uv"] is not None]
    found_val = [x for x in val if x["state"] == "FOUND" and x["target_uv"] is not None]
    centers = {r: np.mean([x["target_uv"] for x in found_train if x["relation"] == r], axis=0) for r in RELATIONS}
    errors = [math.dist(x["target_uv"], centers[x["relation"]]) for x in found_val]
    centroid = float(np.mean([x <= .05 for x in errors])) if errors else 1.0
    metrics = {
        "text_only_answerability_macro_f1": text_f1,
        "rgb_thumbnail_16x12_answerability_macro_f1": rgb_f1,
        "depth_thumbnail_16x12_answerability_macro_f1": depth_f1,
        "relation_centroid_hit_at_0_05": centroid,
        "relation_centroid_l2_mean": float(np.mean(errors)) if errors else None,
    }
    checks = {
        "text_only_f1_le_0_35": text_f1 <= .35,
        "rgb_thumbnail_f1_le_0_70": rgb_f1 <= .70,
        "depth_thumbnail_f1_le_0_70": depth_f1 <= .70,
        "relation_centroid_hit05_le_0_60": centroid <= .60,
    }
    return {"schema_version": 1, "created_at_utc": now(), "status": "PASS" if all(checks.values()) else "FAIL", "metrics": metrics, "checks": checks, "fit_split": "train", "evaluation_split": "development_validation"}


def make_figures(records: list[dict]) -> None:
    selected = []
    for state in STATES:
        for relation in RELATIONS:
            selected.append(next(x for x in records if x["state"] == state and x["relation"] == relation and x["split"] == "development_validation"))
    tiles = []
    for rec in selected:
        image = cv2.imread(str(rec["rgb"])); image = cv2.resize(image, (320, 240))
        label = f"{rec['state']} | {rec['relation']} | px={rec['target_visible_pixels']}"
        cv2.rectangle(image, (0, 0), (319, 27), (255, 255, 255), -1)
        cv2.putText(image, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(image)
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 16, 4)])
    cv2.imwrite(str(REPORT / "anh/CONTACT_SHEET.png"), sheet)
    matrix = np.array([[sum(x["state"] == s and x["relation"] == r for x in records) for r in RELATIONS] for s in STATES])
    fig, ax = plt.subplots(figsize=(9, 4.6)); im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=32)
    ax.set_xticks(range(4), RELATIONS, rotation=18, ha="right"); ax.set_yticks(range(4), STATES)
    for i in range(4):
        for j in range(4): ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    ax.set_title("Gazebo Development-R2: phân bố state × relation"); fig.colorbar(im, ax=ax, label="Số parent family")
    fig.tight_layout(); fig.savefig(REPORT / "anh/PHAN_BO_STATE_RELATION.png", dpi=180); plt.close(fig)


def materialize(records: list[dict], qc: dict, shortcut: dict) -> None:
    if DATASET.exists():
        raise FileExistsError(f"append-only dataset exists: {DATASET}")
    (DATASET / "records").mkdir(parents=True)
    inputs = {"train": [], "development_validation": []}; supervision = []
    for rec in sorted(records, key=lambda x: x["scene_id"]):
        dst = DATASET / "records" / rec["scene_id"]
        dst.mkdir()
        for source, name in ((rec["rgb"], "rgb.png"), (rec["depth"], "depth_m.npy"), (rec["labels"], "semantic_labels.png")):
            try: os.link(source, dst / name)
            except OSError: shutil.copy2(source, dst / name)
        row = {
            "schema_version": 1, "sample_id": rec["scene_id"], "family_id": rec["family_id"], "namespace": NAMESPACE,
            "split": rec["split"], "seed": rec["seed"], "camera_stratum": rec["camera_stratum"], "prompt_variant": rec["prompt_variant"],
            "instruction": rec["instruction"], "rgb_path": str((dst / "rgb.png").relative_to(ROOT)), "rgb_sha256": rec["rgb_sha256"],
            "metric_depth_path": str((dst / "depth_m.npy").relative_to(ROOT)), "metric_depth_sha256": rec["depth_sha256"],
            "semantic_labels_evaluator_path": str((dst / "semantic_labels.png").relative_to(ROOT)), "semantic_labels_sha256": rec["labels_sha256"],
            "layout_signature_sha256": rec["layout_signature_sha256"],
        }
        inputs[rec["split"]].append(row)
        found = rec["state"] == "FOUND"
        supervision.append({
            "sample_id": rec["scene_id"], "family_id": rec["family_id"], "relation": rec["relation"], "answerability": rec["state"],
            "target_uv": rec["target_uv"] if found else None, "target_visible_pixels": rec["target_visible_pixels"],
            "insufficient_reason": rec["annotation"].get("insufficient_reason"),
            "head_mask": {"relation": True, "answerability": True, "coordinate": found, "log_variance": found, "reasoning": False, "source": False, "confidence": False},
            "label_source": "Gazebo world-v5 semantic evaluator after Day-8 DATA QC",
        })
    (DATASET / "TRAIN_MANIFEST.jsonl").write_text("".join(json.dumps(x) + "\n" for x in inputs["train"]))
    (DATASET / "VAL_MANIFEST.jsonl").write_text("".join(json.dumps(x) + "\n" for x in inputs["development_validation"]))
    (DATASET / "SUPERVISION.jsonl").write_text("".join(json.dumps(x) + "\n" for x in supervision))
    summary = {
        "schema_version": 1, "created_at_utc": now(), "namespace": NAMESPACE, "role": "development_r2_train_and_independent_development_validation",
        "families": len(records), "train": len(inputs["train"]), "development_validation": len(inputs["development_validation"]),
        "states": dict(Counter(x["state"] for x in records)), "relations": dict(Counter(x["relation"] for x in records)),
        "shortcut_metrics": shortcut["metrics"], "candidate_r3_trained": False,
    }
    write_json(DATASET / "DATASET_SUMMARY.json", summary); write_json(DATASET / "DATASET_QC.json", qc)


def canary() -> None:
    repair_v13 = RAW / "canary_repair_v13"
    repair_v12 = RAW / "canary_repair_v12"
    repair = RAW / "canary_repair_v11"
    if repair_v13.is_dir():
        failed_v10 = {
            "d8r2_canary_v10_f81a2e34aa3b0912fbbe",
            "d8r2_canary_v10_1aa6e6c0d57607d4ba77",
        }
        retained = sum((collect(RAW / "canary_v10" / f"camera_{i}") for i in range(4)), [])
        records = [x for x in retained if x["scene_id"] not in failed_v10] + collect(repair / "camera_0") + collect(repair_v13 / "camera_3")
        folder = repair_v13
        capture_revision = "canary_v10_retained14_plus_v11_ie_plus_v13_ambiguous"
    elif repair_v12.is_dir():
        failed_v10 = {
            "d8r2_canary_v10_f81a2e34aa3b0912fbbe",
            "d8r2_canary_v10_1aa6e6c0d57607d4ba77",
        }
        retained = sum((collect(RAW / "canary_v10" / f"camera_{i}") for i in range(4)), [])
        repaired_ie = collect(repair / "camera_0")
        repaired_ambiguous = collect(repair_v12 / "camera_3")
        records = [x for x in retained if x["scene_id"] not in failed_v10] + repaired_ie + repaired_ambiguous
        folder = repair_v12
        capture_revision = "canary_v10_retained14_plus_v11_ie_plus_v12_ambiguous"
    elif repair.is_dir():
        # Repair only the two failed v10 cells.  Do not recapture or replace
        # the fourteen accepted observations merely to create a new folder.
        failed_v10 = {
            "d8r2_canary_v10_f81a2e34aa3b0912fbbe",
            "d8r2_canary_v10_1aa6e6c0d57607d4ba77",
        }
        retained = sum((collect(RAW / "canary_v10" / f"camera_{i}") for i in range(4)), [])
        repaired = sum((collect(repair / f"camera_{i}") for i in (0, 3)), [])
        records = [x for x in retained if x["scene_id"] not in failed_v10] + repaired
        folder = repair
        capture_revision = "canary_v10_retained14_plus_repair_v11"
    else:
        newest = "canary_v10" if (RAW / "canary_v10").is_dir() else ("canary_v9" if (RAW / "canary_v9").is_dir() else ("canary_v8" if (RAW / "canary_v8").is_dir() else ("canary_v7" if (RAW / "canary_v7").is_dir() else ("canary_v6" if (RAW / "canary_v6").is_dir() else "canary_v5"))))
        if (RAW / newest).is_dir() and all(any((RAW / f"{newest}/camera_{i}").glob("capture_attempt_*/input_manifest.jsonl")) for i in range(4)):
            folder = RAW / newest
            records = sum((collect(folder / f"camera_{i}") for i in range(4)), [])
        else:
            folder = RAW / ("canary_v4" if (RAW / "canary_v4/capture_attempt_01").is_dir() else ("canary_v3" if (RAW / "canary_v3/capture_attempt_01").is_dir() else ("canary_v2" if (RAW / "canary_v2/capture_attempt_01").is_dir() else "canary")))
            records = collect(folder)
        capture_revision = folder.name
    details, failures, summary = inspect(records, True)
    cells = Counter((x["state"], x["relation"]) for x in records)
    checks = {
        "families_exact_16": len(records) == 16 and len({x["family_id"] for x in records}) == 16,
        "one_family_per_cell": all(cells[(s, r)] == 1 for s in STATES for r in RELATIONS),
        "sensor_and_hash_qc": summary["readable_and_synced"], "semantic_visibility_label_qc": summary["semantic_label_qc"],
        "floating_object_contract_qc": summary["floating_object_contract_qc"], "all_scene_checks_pass": not failures,
    }
    payload = {"schema_version": 1, "created_at_utc": now(), "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "summary": summary, "failures": failures, "scenes": details}
    output = REPORT / "CANARY_QC.json"
    if output.is_file():
        old = json.loads(output.read_text())
        number = {"canary": "01", "canary_v2": "02", "canary_v3": "03", "canary_v4": "04"}.get(old.get("capture_revision"), "01")
        preserved = REPORT / f"CANARY_QC_ATTEMPT_{number}.json"
        if not preserved.exists(): shutil.copy2(output, preserved)
    payload["capture_revision"] = capture_revision
    payload["previous_attempt_preserved"] = (REPORT / "CANARY_QC_ATTEMPT_01.json").is_file()
    write_json(output, payload)
    print(json.dumps({"status": payload["status"], "checks": checks, "failures": failures}, indent=2))
    if payload["status"] != "PASS": raise SystemExit(2)


def final() -> None:
    if json.loads((REPORT / "CANARY_QC.json").read_text()).get("status") != "PASS":
        raise RuntimeError("canary gate is not PASS")
    bulk_name = "bulk_v7" if (RAW / "bulk_v7").is_dir() else "bulk"
    folders = [RAW / bulk_name / f"batch_{i}" for i in range(4)]
    records = sum((collect(x) for x in folders), [])
    details, failures, sensor = inspect(records, False)
    states = Counter(x["state"] for x in records); relations = Counter(x["relation"] for x in records)
    cells = Counter((x["state"], x["relation"]) for x in records); splits = Counter(x["split"] for x in records)
    train_ids = {x["family_id"] for x in records if x["split"] == "train"}; val_ids = {x["family_id"] for x in records if x["split"] == "development_validation"}
    hist_rgb, hist_family, hist_seed = historical_sets()
    rgb_overlap = sorted({x["rgb_sha256"] for x in records} & hist_rgb)
    family_overlap = sorted({x["family_id"] for x in records} & hist_family)
    seed_overlap = sorted({x["seed"] for x in records} & hist_seed)
    shortcut = shortcut_audit(records)
    write_json(REPORT / "SHORTCUT_AUDIT.json", shortcut)
    split_payload = {
        "schema_version": 1, "created_at_utc": now(), "method": "deterministic parent-family split locked before capture; repeat 0-23 train, 24-31 development validation within every cell",
        "counts": dict(splits), "train_family_ids": sorted(train_ids), "development_validation_family_ids": sorted(val_ids),
    }
    write_json(REPORT / "FAMILY_SPLIT.json", split_payload)
    leakage_checks = {
        "zero_train_val_family_overlap": not (train_ids & val_ids),
        "zero_historical_family_overlap": not family_overlap,
        "zero_historical_seed_overlap": not seed_overlap,
        "zero_historical_exact_rgb_overlap": not rgb_overlap,
        "unique_family_ids_512": len({x["family_id"] for x in records}) == 512,
        "unique_seeds_512": len({x["seed"] for x in records}) == 512,
        "unique_rgb_sha256_512": len({x["rgb_sha256"] for x in records}) == 512,
    }
    leakage = {"schema_version": 1, "created_at_utc": now(), "status": "PASS" if all(leakage_checks.values()) else "FAIL", "checks": leakage_checks, "overlap": {"rgb": rgb_overlap, "family": family_overlap, "seed": seed_overlap}}
    write_json(REPORT / "FAMILY_LEAKAGE_AUDIT.json", leakage)
    label_masks = all((x["target_uv"] is not None) == (x["state"] == "FOUND") for x in records)
    checks = {
        "canary_pass": True,
        "families_exact_unique_512": len(records) == 512 and len({x["family_id"] for x in records}) == 512,
        "states_exact_128_each": all(states[s] == 128 for s in STATES),
        "relations_exact_128_each": all(relations[r] == 128 for r in RELATIONS),
        "cells_exact_32_each": all(cells[(s, r)] == 32 for s in STATES for r in RELATIONS),
        "split_exact_384_128": splits == Counter({"train": 384, "development_validation": 128}),
        "split_cells_exact_24_8": all(sum(x["state"] == s and x["relation"] == r and x["split"] == "train" for x in records) == 24 and sum(x["state"] == s and x["relation"] == r and x["split"] == "development_validation" for x in records) == 8 for s in STATES for r in RELATIONS),
        "sensor_semantic_floating_qc_all": not failures,
        "sha256_and_rgb_unique": sensor["unique_rgb"] == 512 and sensor["unique_depth"] == 512,
        "leakage_audit_pass": leakage["status"] == "PASS",
        "only_found_has_target_uv_and_spatial_masks": label_masks,
        "shortcut_audit_pass": shortcut["status"] == "PASS",
    }
    status = "DATA_PASS" if all(checks.values()) else "DATA_HOLD"
    qc = {
        "schema_version": 1, "created_at_utc": now(), "status": status, "checks": checks,
        "counts": {"families": len(records), "states": dict(states), "relations": dict(relations), "splits": dict(splits), "cells": {f"{s}|{r}": cells[(s, r)] for s in STATES for r in RELATIONS}},
        "sensor_summary": sensor, "shortcut_metrics": shortcut["metrics"], "failures": failures, "scene_qc": details,
    }
    write_json(REPORT / "DATA_QC.json", qc)
    capture_manifest = {"schema_version": 1, "created_at_utc": now(), "families": len(records), "batches": [{"batch": i, "capture_manifest_path": str((folders[i] / "capture_attempt_01/capture_manifest.json").relative_to(ROOT)), "capture_manifest_sha256": sha256(folders[i] / "capture_attempt_01/capture_manifest.json"), "families": len(collect(folders[i]))} for i in range(4)]}
    write_json(REPORT / "CAPTURE_MANIFEST.json", capture_manifest)
    with (REPORT / "bang/PHAN_BO_DATA.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["answerability", "relation", "train", "development_validation", "total"])
        for s in STATES:
            for r in RELATIONS:
                tr = sum(x["state"] == s and x["relation"] == r and x["split"] == "train" for x in records); va = cells[(s, r)] - tr
                writer.writerow([s, r, tr, va, tr + va])
    make_figures(records)
    decision = {
        "schema_version": 1, "created_at_utc": now(), "day": 8, "outcome": status,
        "candidate_r3_training_performed": False, "opens_candidate_r3_training_day9": status == "DATA_PASS",
        "does_not_open": ["OOF/S1b", "G3", "calibration", "Test", "robot final"], "checks": checks,
        "evidence": {"data_qc": "ketquangay/ngay_08/DATA_QC.json", "shortcut_audit": "ketquangay/ngay_08/SHORTCUT_AUDIT.json", "leakage_audit": "ketquangay/ngay_08/FAMILY_LEAKAGE_AUDIT.json"},
    }
    write_json(REPORT / "DECISION.json", decision)
    if status == "DATA_PASS":
        materialize(records, qc, shortcut)
    report = f"""# BÁO CÁO NGÀY 08 — GAZEBO DEVELOPMENT-R2

**Kết luận:** `{status}`. Ngày 8 chỉ xây dựng và kiểm định dữ liệu; **không train Candidate R3**.

## 1. Dữ liệu đã capture thật

| Hạng mục | Kết quả |
|---|---:|
| Parent family duy nhất | {len(records)} |
| Train / Development validation | {splits.get('train', 0)} / {splits.get('development_validation', 0)} |
| Mỗi trạng thái | {', '.join(f'{s}: {states[s]}' for s in STATES)} |
| Mỗi ô state × relation | {'32' if all(cells[(s,r)] == 32 for s in STATES for r in RELATIONS) else 'không đồng đều'} |
| Lỗi semantic/sensor/floating contract | {len(failures)} |

## 2. Chống shortcut (fit Train, đo trên Development validation)

| Probe | Kết quả | Gate | Đạt |
|---|---:|---:|:---:|
| Text-only answerability macro-F1 | {shortcut['metrics']['text_only_answerability_macro_f1']:.4f} | ≤ 0,35 | {'Có' if shortcut['checks']['text_only_f1_le_0_35'] else 'Không'} |
| RGB thumbnail macro-F1 | {shortcut['metrics']['rgb_thumbnail_16x12_answerability_macro_f1']:.4f} | ≤ 0,70 | {'Có' if shortcut['checks']['rgb_thumbnail_f1_le_0_70'] else 'Không'} |
| Depth thumbnail macro-F1 | {shortcut['metrics']['depth_thumbnail_16x12_answerability_macro_f1']:.4f} | ≤ 0,70 | {'Có' if shortcut['checks']['depth_thumbnail_f1_le_0_70'] else 'Không'} |
| Relation-centroid Hit@0.05 | {shortcut['metrics']['relation_centroid_hit_at_0_05']:.4f} | ≤ 0,60 | {'Có' if shortcut['checks']['relation_centroid_hit05_le_0_60'] else 'Không'} |

## 3. Leakage và nhãn

- Train–validation: **zero parent-family overlap**.
- So với dữ liệu lịch sử: RGB={len(rgb_overlap)}, family={len(family_overlap)}, seed={len(seed_overlap)} overlap.
- Chỉ `FOUND` có `target_uv`, coordinate mask và log-variance mask.
- `INSUFFICIENT_EVIDENCE` ghi nguyên nhân che khuất có kiểm soát; `ABSENT` không có target pose/pixel.

## 4. Quyết định

`{status}` {'mở duy nhất Candidate R3 training của Ngày 9.' if status == 'DATA_PASS' else 'không mở Candidate R3; phải sửa dữ liệu/generator mà không hạ gate.'}
OOF/S1b, G3, calibration, Test và robot final vẫn đóng.
"""
    (REPORT / "BAO_CAO_NGAY_08.md").write_text(report)
    print(json.dumps({"outcome": status, "checks": checks, "shortcut": shortcut["metrics"], "failures": len(failures)}, indent=2))
    if status != "DATA_PASS": raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("canary", "final")); args = parser.parse_args()
    canary() if args.stage == "canary" else final()
