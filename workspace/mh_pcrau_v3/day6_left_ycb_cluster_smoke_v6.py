#!/usr/bin/env python3
"""Preregister a compact, robot-side, collision-clear left-YCB layout smoke."""

from __future__ import annotations

from datetime import datetime, timezone
import ast
import hashlib
import json
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_contact_smoke_v5"
OUT = ROOT / "ketqua1/09_danh_gia/anti_shortcut_left_ycb_cluster_smoke_v6"
WORLD = ROOT / "workspace/mh_pcrau_v3/generated/day6_left_ycb_v5_contact_exact/left_ycb_top30_contact_exact.sdf"
CAPTURE = ROOT / "ur3/ur3_perception/scripts/roborefer_pilot_capture.py"
PROTOCOL = "mh_pcrau_v3_left_ycb_cluster_smoke_v6"

# Compact cluster shifted toward the robot (+X / lower-Y side of the prior
# layouts), while retaining a conservative no-object zone around its base.
POSES = {
    "ycb_power_drill": [-0.44, 0.30, 0.0],
    "ycb_banana": [-0.24, 0.30, 0.0],
    "ycb_mug": [-0.10, 0.30, 0.0],
    "ycb_bleach_cleanser": [-0.46, 0.48, 0.0],
    "ycb_cracker_box": [-0.31, 0.50, 0.0],
    "ycb_sugar_box": [-0.18, 0.49, 0.0],
    "ycb_tomato_soup_can": [-0.08, 0.49, 0.0],
    "ycb_mustard_bottle": [-0.46, 0.66, 0.0],
    "ycb_tuna_fish_can": [-0.34, 0.68, 0.0],
    "ycb_apple": [-0.22, 0.67, 0.0],
    "ycb_orange": [-0.10, 0.67, 0.0],
    "mango": [-0.42, 0.82, 0.0],
    "ycb_pear": [-0.28, 0.82, 0.0],
    "ycb_plum": [-0.16, 0.82, 0.0],
    "ycb_lemon": [5.34, 3.0, 0.0],
}

# Axis-aligned collision half extents in the table plane.  All smoke yaws are
# zero. Cylinders and spheres use their collision radius in both axes.
HALF_EXTENTS = {
    "ycb_power_drill": (0.091998, 0.093622),
    "ycb_banana": (0.054486, 0.089198),
    "ycb_mug": (0.045, 0.045),
    "ycb_bleach_cleanser": (0.040, 0.045),
    "ycb_cracker_box": (0.035860, 0.081993),
    "ycb_sugar_box": (0.024725, 0.046687),
    "ycb_tomato_soup_can": (0.03395, 0.03395),
    "ycb_mustard_bottle": (0.030, 0.030),
    "ycb_tuna_fish_can": (0.043, 0.043),
    "ycb_apple": (0.03771, 0.03771),
    "ycb_orange": (0.03701, 0.03701),
    "mango": (0.030, 0.030),
    "ycb_pear": (0.040, 0.040),
    "ycb_plum": (0.028, 0.028),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_sources() -> set[str]:
    tree = ast.parse(CAPTURE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "REQUIRED_SOURCE_ARTIFACTS"
            for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise RuntimeError("required capture inventory missing")


def clearance(first: str, second: str) -> float:
    ax, ay, _ = POSES[first]
    bx, by, _ = POSES[second]
    ahx, ahy = HALF_EXTENTS[first]
    bhx, bhy = HALF_EXTENTS[second]
    gap_x = max(abs(ax - bx) - ahx - bhx, 0.0)
    gap_y = max(abs(ay - by) - ahy - bhy, 0.0)
    return math.hypot(gap_x, gap_y)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"append-only output exists: {OUT}")
    names = list(HALF_EXTENTS)
    gaps = [
        (clearance(names[i], names[j]), names[i], names[j])
        for i in range(len(names)) for j in range(i + 1, len(names))
    ]
    minimum_gap, gap_a, gap_b = min(gaps)
    if minimum_gap < 0.015:
        raise RuntimeError(f"cluster clearance {minimum_gap:.6f} m: {gap_a}/{gap_b}")
    minimum_robot_radius = min(
        math.hypot(values[0], values[1]) for name, values in POSES.items()
        if name != "ycb_lemon"
    )
    if minimum_robot_radius < 0.30:
        raise RuntimeError("object entered 0.30 m robot-base safety radius")

    payloads = {}
    for filename in ("scenes.yaml", "annotations.yaml", "gate.yaml"):
        payload = yaml.safe_load((SOURCE / filename).read_text(encoding="utf-8"))
        payload["protocol_id"] = PROTOCOL
        payloads[filename] = payload
    scene = payloads["scenes.yaml"]["scenes"][0]
    scene.update({
        "scene_id": "cluster_smoke_v6_000",
        "scene_family_id": "mh_pcrau_v3/cluster_smoke_v6/000",
        "layout_id": "cluster_smoke_v6_000",
        "task_type": "compact_robot_side_layout_qc_only",
        "instruction": "Compact robot-side layout smoke; no model inference.",
        "poses": POSES,
    })
    signature = hashlib.sha256(
        json.dumps(POSES, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    scene["layout_signature_sha256"] = signature
    annotation = next(iter(payloads["annotations.yaml"]["scenes"].values()))
    annotation.update({
        "family_id": scene["scene_family_id"],
        "layout_id": scene["layout_id"],
        "layout_signature_sha256": signature,
        "failure_tags": [
            "compact_robot_side_layout_qc_only", "no_model_access", "lemon_excluded"
        ],
    })
    payloads["annotations.yaml"]["scenes"] = {scene["scene_id"]: annotation}
    payloads["gate.yaml"]["cluster_layout_gate"] = {
        "minimum_pairwise_footprint_clearance_m": 0.015,
        "observed_minimum_clearance_m": minimum_gap,
        "minimum_robot_base_radius_m": 0.30,
        "observed_minimum_robot_base_radius_m": minimum_robot_radius,
        "active_x_bounds_m": [min(v[0] for k, v in POSES.items() if k != "ycb_lemon"), max(v[0] for k, v in POSES.items() if k != "ycb_lemon")],
        "active_y_bounds_m": [min(v[1] for k, v in POSES.items() if k != "ycb_lemon"), max(v[1] for k, v in POSES.items() if k != "ycb_lemon")],
    }

    OUT.mkdir(parents=True)
    for filename, payload in payloads.items():
        (OUT / filename).write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=160),
            encoding="utf-8",
        )
    (OUT / "LAYOUT_QC.json").write_text(
        json.dumps({
            "schema_version": 1,
            "status": "STATIC_LAYOUT_QC_PASS_CAPTURE_NOT_STARTED",
            "active_object_count": len(HALF_EXTENTS),
            "lemon_excluded": True,
            "minimum_pairwise_footprint_clearance_m": minimum_gap,
            "minimum_clearance_pair": [gap_a, gap_b],
            "minimum_robot_base_radius_m": minimum_robot_radius,
            "active_x_bounds_m": [min(v[0] for k, v in POSES.items() if k != "ycb_lemon"), max(v[0] for k, v in POSES.items() if k != "ycb_lemon")],
            "active_y_bounds_m": [min(v[1] for k, v in POSES.items() if k != "ycb_lemon"), max(v[1] for k, v in POSES.items() if k != "ycb_lemon")],
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    relative = lambda path: str(path.relative_to(ROOT))
    artifacts = required_sources() | {
        relative(WORLD), relative(Path(__file__)),
        relative(OUT / "scenes.yaml"), relative(OUT / "annotations.yaml"),
        relative(OUT / "gate.yaml"), relative(OUT / "LAYOUT_QC.json"),
        "workspace/mh_pcrau_v3/g1/gazebo_pilot_512_view.launch.py",
    }
    lock = {
        "schema_version": 1, "protocol_id": PROTOCOL,
        "status": "LOCKED_BEFORE_CAPTURE_AND_INFERENCE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "source_artifact_sha256": {name: sha256(ROOT / name) for name in sorted(artifacts)},
        "model_inventory_sha256": "NO_MODEL_INFERENCE_CLUSTER_SMOKE",
        "capture_role": "COMPACT_ROBOT_SIDE_LAYOUT_QC_ONLY",
        "planned_scene_count": 1, "world_file": relative(WORLD),
        "lemon_policy": "EXCLUDED_FROM_NEXT_CANARY_AND_FINAL_FAMILY",
    }
    (OUT / "CAPTURE_SOURCE_LOCK.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "minimum_clearance_m": minimum_gap, "minimum_robot_radius_m": minimum_robot_radius}, indent=2))


if __name__ == "__main__":
    main()
