#!/usr/bin/env python3
"""Add YCB inventory V2 to a copy of the original UR3 pick-place world."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from build_ycb_inventory_v2_gallery import (  # noqa: E402
    DEFAULT_CONFIG,
    InventoryError,
    PACKAGE_ROOT,
    WORKSPACE_ROOT,
    fmt,
    load_and_validate,
    sha256,
)


DEFAULT_BASE_WORLD = PACKAGE_ROOT / "worlds/ur3_pick_place.sdf"
DEFAULT_OUTPUT_WORLD = PACKAGE_ROOT / "worlds/ur3_pick_place_inventory_v2.sdf"
DEFAULT_RESULTS = WORKSPACE_ROOT / "results/ycb_inventory_v2_color_fix_20260821_135403"


def inertia_box(mass: float, size: list[float]) -> tuple[float, float, float]:
    x, y, z = size
    return (
        mass * (y * y + z * z) / 12,
        mass * (x * x + z * z) / 12,
        mass * (x * x + y * y) / 12,
    )


def dynamic_model_sdf(instance: dict[str, Any], asset: dict[str, Any]) -> str:
    bounds = asset["bounds_m"]
    center = bounds["center"]
    size = bounds["size"]
    mass = float(instance["mass_kg"])
    ixx, iyy, izz = inertia_box(mass, size)
    pose = [
        float(instance["x"]),
        float(instance["y"]),
        size[2] / 2 + 0.0005,
        0.0,
        0.0,
        float(instance["yaw"]),
    ]
    visual_pose = [-center[0], -center[1], -center[2], 0.0, 0.0, 0.0]
    ycb_id = asset["ycb_id"]
    return f"""    <model name=\"{instance['name']}\">
      <pose>{fmt(pose)}</pose>
      <link name=\"object_link\">
        <inertial><mass>{mass:.6f}</mass><inertia><ixx>{ixx:.9f}</ixx><iyy>{iyy:.9f}</iyy><izz>{izz:.9f}</izz></inertia></inertial>
        <velocity_decay><linear>0.10</linear><angular>0.25</angular></velocity_decay>
        <collision name=\"collision\">
          <geometry><box><size>{fmt(size)}</size></box></geometry>
          <surface><friction><ode><mu>1.2</mu><mu2>1.2</mu2></ode></friction></surface>
        </collision>
        <visual name=\"ycb_visual\">
          <pose>{fmt(visual_pose)}</pose>
          <geometry><mesh><uri>../models/ycb/{ycb_id}/{ycb_id}.obj</uri></mesh></geometry>
        </visual>
      </link>
      <plugin filename=\"ignition-gazebo-pose-publisher-system\" name=\"gz::sim::systems::PosePublisher\"><publish_link_pose>false</publish_link_pose><publish_collision_pose>false</publish_collision_pose><publish_visual_pose>false</publish_visual_pose><publish_nested_model_pose>false</publish_nested_model_pose><use_pose_vector_msg>true</use_pose_vector_msg><static_publisher>false</static_publisher><update_frequency>15</update_frequency></plugin>
      <plugin filename=\"ignition-gazebo-label-system\" name=\"ignition::gazebo::systems::Label\"><label>{int(instance['label'])}</label></plugin>
    </model>"""


def validate_main_world_config(
    config: dict[str, Any], assets: dict[str, dict[str, Any]], base_text: str
) -> dict[str, Any]:
    policy = config.get("main_world_policy")
    additions = config.get("main_world_additions")
    if not isinstance(policy, dict) or not isinstance(additions, list):
        raise InventoryError("Missing main_world_policy/main_world_additions")
    if '<world name="ur3_pick_place">' not in base_text:
        raise InventoryError("Base world name is not ur3_pick_place")
    if base_text.count("</world>") != 1:
        raise InventoryError("Expected exactly one world closing tag")

    names = [item["name"] for item in additions]
    labels = [int(item["label"]) for item in additions]
    if len(names) != len(set(names)) or any(f'<model name="{name}">' in base_text for name in names):
        raise InventoryError("New entity names are duplicated or collide with base world")
    if len(labels) != len(set(labels)) or any(label < 1 or label > 255 for label in labels):
        raise InventoryError("New labels must be unique uint8 values")
    for label in labels:
        if f"<label>{label}</label>" in base_text:
            raise InventoryError(f"New label collides with base world: {label}")
    if any(item["semantic_class"] not in assets for item in additions):
        raise InventoryError("Main-world addition references unknown asset")

    existing = policy["existing_instances"]
    remove_base_models = policy.get("remove_base_models", [])
    if remove_base_models != ["cup", "kettle"]:
        raise InventoryError("V2 removal policy must explicitly remove cup and kettle")
    for name in remove_base_models:
        if base_text.count(f'<model name="{name}">') != 1:
            raise InventoryError(f"Base model to remove was not found exactly once: {name}")
    totals = Counter(item["semantic_class"] for item in additions)
    totals.update(existing.keys())
    for semantic_class in ("apple", "orange", "banana"):
        if totals[semantic_class] != 3:
            raise InventoryError(f"Main world must contain three {semantic_class} instances")
        name = existing[semantic_class]["name"]
        label = int(existing[semantic_class]["label"])
        if f'<model name="{name}">' not in base_text or f"<label>{label}</label>" not in base_text:
            raise InventoryError(f"Existing {semantic_class} instance not found in base world")
    for semantic_class in ("lemon", "pear", "plum", "tuna_fish_can", "mug"):
        if totals[semantic_class] != 1:
            raise InventoryError(f"Main world must contain one {semantic_class}")
    if totals["bleach_cleanser"] != 0:
        raise InventoryError("bleach_cleanser must be omitted from the main world")

    positions = [(float(item["x"]), float(item["y"])) for item in additions]
    if len(positions) != len(set(positions)):
        raise InventoryError("New main-world positions are duplicated")
    return {
        "additions": additions,
        "existing_instances": existing,
        "removed_base_models": remove_base_models,
        "omitted_inventory_models": policy.get("omit_inventory_models", []),
        "total_requested_inventory_counts": dict(sorted(totals.items())),
    }


def remove_base_model(base_text: str, name: str) -> str:
    start_marker = f'    <model name="{name}">'
    start = base_text.find(start_marker)
    if start < 0:
        raise InventoryError(f"Cannot remove absent base model: {name}")
    end_marker = "    </model>"
    end = base_text.find(end_marker, start)
    if end < 0:
        raise InventoryError(f"Cannot find closing tag for base model: {name}")
    end += len(end_marker)
    if end < len(base_text) and base_text[end] == "\n":
        end += 1
    return base_text[:start] + base_text[end:]


def build_world(
    base_text: str, additions_text: str, remove_base_models: list[str]
) -> str:
    derived_text = base_text
    for name in remove_base_models:
        derived_text = remove_base_model(derived_text, name)
    marker = "  </world>"
    if derived_text.count(marker) != 1:
        raise InventoryError("Base world closing indentation changed")
    provenance = (
        "\n    <!-- YCB inventory V2 additions. The source ur3_pick_place.sdf is "
        "left unchanged. The derived scene removes cup, kettle and omits "
        "bleach cleanser by explicit user request. -->\n"
    )
    return derived_text.replace(marker, provenance + additions_text + "\n" + marker)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-world", type=Path, default=DEFAULT_BASE_WORLD)
    parser.add_argument("--output-world", type=Path, default=DEFAULT_OUTPUT_WORLD)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    config_path = args.config.resolve()
    base_path = args.base_world.resolve()
    output_path = args.output_world.resolve()
    result_dir = args.result_dir.resolve()
    if base_path == output_path:
        raise InventoryError("Refusing to overwrite the original base world")

    config, resolved = load_and_validate(config_path)
    base_text = base_path.read_text(encoding="utf-8")
    base_hash_before = sha256(base_path)
    main = validate_main_world_config(config, resolved["assets"], base_text)
    additions_text = "\n\n".join(
        dynamic_model_sdf(item, resolved["assets"][item["semantic_class"]])
        for item in main["additions"]
    )
    output_path.write_text(
        build_world(base_text, additions_text, main["removed_base_models"]),
        encoding="utf-8",
    )
    base_hash_after = sha256(base_path)
    if base_hash_before != base_hash_after:
        raise InventoryError("Original world changed while generating V2")

    result_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "gate": "YCB_INVENTORY_V2_MAIN_WORLD_GENERATION",
        "base_world": str(base_path.relative_to(WORKSPACE_ROOT)),
        "base_world_sha256_before": base_hash_before,
        "base_world_sha256_after": base_hash_after,
        "base_world_unchanged": base_hash_before == base_hash_after,
        "output_world": str(output_path.relative_to(WORKSPACE_ROOT)),
        "output_world_sha256": sha256(output_path),
        "world_name_unchanged": "ur3_pick_place",
        "addition_count": len(main["additions"]),
        "removed_base_models": main["removed_base_models"],
        "omitted_inventory_models": main["omitted_inventory_models"],
        "existing_requested_instances": main["existing_instances"],
        "total_requested_inventory_counts": main["total_requested_inventory_counts"],
        "new_entity_names_unique": True,
        "new_labels_unique_and_noncolliding": True,
        "source_asset_audit": str(
            (WORKSPACE_ROOT / "results/ycb_inventory_v2_gallery_20260821_133253/YCB_INVENTORY_V2_AUDIT.json")
            .relative_to(WORKSPACE_ROOT)
        ),
        "scope": "visual_and_physics_integration_only_not_dataset_capture",
    }
    (result_dir / "YCB_INVENTORY_V2_MAIN_WORLD_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (result_dir / "README.md").write_text(
        "# UR3 pick-place + YCB inventory V2\n\n"
        "This generated world preserves the complete original `ur3_pick_place.sdf` "
        "and appends 11 dynamic YCB instances. The three original apple/orange/banana "
        "objects count toward the requested total of three per class.\n\n"
        "The derived scene removes `cup`, `kettle` and omits `ycb_bleach_cleanser` "
        "by explicit user request.\n\n"
        "The original world file and locked WP2 dataset are unchanged. This is an "
        "integration/visual qualification scene, not a scientific model result.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "base_world_unchanged": True,
        "additions": len(main["additions"]),
        "output_world": str(output_path),
        "results": str(result_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
