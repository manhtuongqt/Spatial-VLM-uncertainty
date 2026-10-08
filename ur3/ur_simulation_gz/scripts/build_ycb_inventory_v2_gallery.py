#!/usr/bin/env python3
"""Validate YCB inventory V2 and generate a deterministic Gazebo gallery.

This is an asset-qualification utility. It deliberately does not modify the
locked WP2 dataset or the original UR3 pick-place world.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PACKAGE_ROOT / "config/ycb_inventory_v2.json"
DEFAULT_WORLD = PACKAGE_ROOT / "worlds/ycb_inventory_v2_gallery.sdf"
DEFAULT_RESULTS = WORKSPACE_ROOT / "results/ycb_inventory_v2_gallery_20260821_133253"
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class InventoryError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def obj_bounds(path: Path) -> dict[str, list[float]]:
    vertices: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                fields = line.split()
                if len(fields) != 4:
                    raise InventoryError(f"Malformed OBJ vertex: {path}: {line!r}")
                vertices.append(tuple(float(value) for value in fields[1:4]))
    if not vertices:
        raise InventoryError(f"No vertices in OBJ: {path}")
    minimum = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
    maximum = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
    size = [maximum[axis] - minimum[axis] for axis in range(3)]
    center = [(maximum[axis] + minimum[axis]) / 2 for axis in range(3)]
    if min(size) <= 0 or max(size) > 0.5:
        raise InventoryError(f"Implausible metric bounds for {path}: {size}")
    return {"min": minimum, "max": maximum, "size": size, "center": center}


def load_and_validate(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 2:
        raise InventoryError("Expected ycb_inventory_v2 schema_version=2")
    if config.get("status") != "ASSET_QUALIFICATION_ONLY":
        raise InventoryError("Inventory must remain ASSET_QUALIFICATION_ONLY")

    assets: dict[str, dict[str, Any]] = {}
    asset_audit: list[dict[str, Any]] = []
    for spec in config["assets"]:
        semantic_class = spec["semantic_class"]
        ycb_id = spec["ycb_id"]
        if not NAME_PATTERN.fullmatch(semantic_class):
            raise InventoryError(f"Invalid semantic class: {semantic_class}")
        if semantic_class in assets:
            raise InventoryError(f"Duplicate semantic class: {semantic_class}")
        asset_dir = PACKAGE_ROOT / "models/ycb" / ycb_id
        paths = {
            suffix: asset_dir / f"{ycb_id}.{suffix}"
            for suffix in ("obj", "mtl", "png")
        }
        paths["glb"] = asset_dir / "textured.glb"
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise InventoryError(f"Missing asset files: {missing}")
        glb_digest = sha256(paths["glb"])
        if glb_digest != spec["glb_sha256"]:
            raise InventoryError(f"Source GLB hash mismatch: {ycb_id}")
        obj_text = paths["obj"].read_text(encoding="utf-8")
        mtl_text = paths["mtl"].read_text(encoding="utf-8")
        if f"mtllib {ycb_id}.mtl" not in obj_text:
            raise InventoryError(f"OBJ does not reference unique MTL: {ycb_id}")
        if f"map_Kd {ycb_id}.png" not in mtl_text:
            raise InventoryError(f"MTL does not reference unique PNG: {ycb_id}")
        bounds = obj_bounds(paths["obj"])
        record = {
            **spec,
            "paths": {key: str(path.relative_to(WORKSPACE_ROOT)) for key, path in paths.items()},
            "sha256": {key: sha256(path) for key, path in paths.items()},
            "bounds_m": bounds,
        }
        assets[semantic_class] = record
        asset_audit.append(record)

    instances = config["instances"]
    names = [instance["name"] for instance in instances]
    labels = [int(instance["label"]) for instance in instances]
    if len(names) != len(set(names)):
        raise InventoryError("Instance names are not unique")
    if len(labels) != len(set(labels)) or any(label < 1 or label > 255 for label in labels):
        raise InventoryError("Instance labels must be unique uint8 values 1..255")
    if any(not NAME_PATTERN.fullmatch(name) for name in names):
        raise InventoryError("Invalid instance entity name")
    if any(instance["semantic_class"] not in assets for instance in instances):
        raise InventoryError("Instance references an unknown semantic class")
    positions = [(float(item["x"]), float(item["y"])) for item in instances]
    if len(positions) != len(set(positions)):
        raise InventoryError("Gallery positions must be unique")
    actual_counts = Counter(item["semantic_class"] for item in instances)
    expected_counts = Counter(config["required_class_counts"])
    if actual_counts != expected_counts:
        raise InventoryError(
            f"Class multiplicity mismatch: actual={actual_counts}, expected={expected_counts}"
        )

    audit = {
        "schema_version": 1,
        "inventory_id": config["inventory_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "gate": "YCB_INVENTORY_V2_ASSET_QUALIFICATION",
        "source": config["source"],
        "integration_policy": config["integration_policy"],
        "asset_class_count": len(assets),
        "instance_count": len(instances),
        "class_counts": dict(sorted(actual_counts.items())),
        "unique_entity_names": True,
        "unique_uint8_labels": True,
        "source_glb_hashes_match": True,
        "obj_mtl_png_chains_valid": True,
        "metric_mesh_bounds_valid": True,
        "assets": asset_audit,
        "instances": instances,
        "scientific_scope": {
            "gallery_is_evidence_of_asset_loading_and_visual_diversity": True,
            "gallery_is_not_evidence_of_model_performance": True,
            "gallery_is_not_a_dataset_split": True,
            "locked_wp0_wp3_and_wp2_dataset_unchanged": True
        }
    }
    return config, {"assets": assets, "audit": audit}


def model_sdf(instance: dict[str, Any], asset: dict[str, Any]) -> str:
    bounds = asset["bounds_m"]
    center = bounds["center"]
    size = bounds["size"]
    pose = [
        float(instance["x"]),
        float(instance["y"]),
        size[2] / 2,
        0.0,
        0.0,
        float(instance["yaw"]),
    ]
    visual_pose = [-center[0], -center[1], -center[2], 0.0, 0.0, 0.0]
    ycb_id = asset["ycb_id"]
    return f"""    <model name=\"{instance['name']}\">
      <static>true</static>
      <pose>{fmt(pose)}</pose>
      <link name=\"object_link\">
        <visual name=\"ycb_visual\">
          <pose>{fmt(visual_pose)}</pose>
          <geometry><mesh><uri>../models/ycb/{ycb_id}/{ycb_id}.obj</uri></mesh></geometry>
        </visual>
      </link>
      <plugin filename=\"ignition-gazebo-label-system\" name=\"ignition::gazebo::systems::Label\"><label>{int(instance['label'])}</label></plugin>
    </model>"""


def gallery_sdf(config: dict[str, Any], assets: dict[str, Any]) -> str:
    models = "\n\n".join(
        model_sdf(instance, assets[instance["semantic_class"]])
        for instance in config["instances"]
    )
    return f"""<?xml version=\"1.0\"?>
<!-- Generated by scripts/build_ycb_inventory_v2_gallery.py.
     Asset-qualification gallery only; not a WP2 dataset scene. -->
<sdf version=\"1.7\">
  <world name=\"ycb_inventory_v2_gallery\">
    <physics name=\"gallery_physics\" type=\"ode\"><max_step_size>0.001</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename=\"libignition-gazebo-physics-system.so\" name=\"ignition::gazebo::systems::Physics\"/>
    <plugin filename=\"libignition-gazebo-user-commands-system.so\" name=\"ignition::gazebo::systems::UserCommands\"/>
    <plugin filename=\"libignition-gazebo-scene-broadcaster-system.so\" name=\"ignition::gazebo::systems::SceneBroadcaster\"/>

    <scene>
      <ambient>0.60 0.60 0.60 1</ambient>
      <background>0.82 0.86 0.91 1</background>
      <shadows>true</shadows>
      <grid>false</grid>
    </scene>
    <light name=\"key_light\" type=\"directional\">
      <pose>-1 -2 3 0 0 0</pose><cast_shadows>true</cast_shadows>
      <diffuse>0.95 0.95 0.95 1</diffuse><specular>0.25 0.25 0.25 1</specular>
      <direction>0.25 0.40 -1.0</direction>
    </light>
    <light name=\"fill_light\" type=\"point\">
      <pose>1.2 -1.2 1.8 0 0 0</pose><cast_shadows>false</cast_shadows>
      <diffuse>0.55 0.58 0.62 1</diffuse><specular>0.10 0.10 0.10 1</specular>
      <attenuation><range>6</range><constant>0.8</constant><linear>0.15</linear><quadratic>0.02</quadratic></attenuation>
    </light>

    <model name=\"gallery_floor\">
      <static>true</static><pose>0 0 -0.061 0 0 0</pose>
      <link name=\"link\"><visual name=\"visual\"><geometry><plane><normal>0 0 1</normal><size>8 8</size></plane></geometry><material><ambient>0.18 0.20 0.23 1</ambient><diffuse>0.24 0.27 0.31 1</diffuse></material></visual></link>
    </model>
    <model name=\"gallery_table\">
      <static>true</static><pose>0 0 -0.030 0 0 0</pose>
      <link name=\"table_link\"><visual name=\"table_visual\"><geometry><box><size>1.95 1.30 0.060</size></box></geometry><material><ambient>0.62 0.54 0.42 1</ambient><diffuse>0.76 0.67 0.54 1</diffuse><specular>0.08 0.08 0.08 1</specular></material></visual></link>
    </model>

{models}

    <gui fullscreen=\"0\">
      <plugin filename=\"MinimalScene\" name=\"3D View\">
        <ignition-gui><title>YCB inventory V2 — 15 instances</title><property type=\"bool\" key=\"showTitleBar\">true</property><property type=\"string\" key=\"state\">docked</property></ignition-gui>
        <engine>ogre2</engine><scene>scene</scene>
        <ambient_light>0.55 0.55 0.55</ambient_light><background_color>0.82 0.86 0.91</background_color>
        <camera_pose>0 -2.30 1.45 0 0.52 1.570796</camera_pose>
        <camera_clip><near>0.05</near><far>100</far></camera_clip>
      </plugin>
      <plugin filename=\"GzSceneManager\" name=\"Scene Manager\"><ignition-gui><property key=\"state\" type=\"string\">floating</property><property key=\"showTitleBar\" type=\"bool\">false</property></ignition-gui></plugin>
      <plugin filename=\"InteractiveViewControl\" name=\"Interactive view control\"><ignition-gui><property key=\"state\" type=\"string\">floating</property><property key=\"showTitleBar\" type=\"bool\">false</property></ignition-gui></plugin>
      <plugin filename=\"EntityContextMenuPlugin\" name=\"Entity context menu\"><ignition-gui><property key=\"state\" type=\"string\">floating</property><property key=\"showTitleBar\" type=\"bool\">false</property></ignition-gui></plugin>
      <plugin filename=\"EntityTree\" name=\"Entity tree\"><ignition-gui><property type=\"bool\" key=\"showTitleBar\">true</property><property type=\"string\" key=\"state\">docked</property></ignition-gui></plugin>
    </gui>
  </world>
</sdf>
"""


def write_outputs(
    config_path: Path,
    world_path: Path,
    result_dir: Path,
    config: dict[str, Any],
    resolved: dict[str, Any],
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    world_path.parent.mkdir(parents=True, exist_ok=True)
    world_path.write_text(gallery_sdf(config, resolved["assets"]), encoding="utf-8")

    audit = resolved["audit"]
    audit["config_path"] = str(config_path.relative_to(WORKSPACE_ROOT))
    audit["config_sha256"] = sha256(config_path)
    audit["world_path"] = str(world_path.relative_to(WORKSPACE_ROOT))
    audit["world_sha256"] = sha256(world_path)
    (result_dir / "YCB_INVENTORY_V2_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (result_dir / "object_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["row", "column", "entity_name", "semantic_class", "ycb_id", "label", "x_m", "y_m", "yaw_rad"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, instance in enumerate(config["instances"]):
            writer.writerow({
                "row": index // 5 + 1,
                "column": index % 5 + 1,
                "entity_name": instance["name"],
                "semantic_class": instance["semantic_class"],
                "ycb_id": resolved["assets"][instance["semantic_class"]]["ycb_id"],
                "label": instance["label"],
                "x_m": instance["x"],
                "y_m": instance["y"],
                "yaw_rad": instance["yaw"],
            })

    (result_dir / "README.md").write_text(
        "# YCB inventory V2 — asset qualification gallery\n\n"
        "Status: `PASS` if `YCB_INVENTORY_V2_AUDIT.json` remains PASS.\n\n"
        "This scene contains 15 uniquely named and labelled instances: three apples, "
        "three oranges, three bananas, and one each of lemon, pear, plum, tuna fish "
        "can, mug, and bleach cleanser. All nine source classes are from ai-habitat/YCB.\n\n"
        "The gallery proves source integrity, Gazebo asset loading and visible diversity. "
        "It is not a model result and is not yet part of the locked WP2 dataset.\n\n"
        f"World: `{world_path.relative_to(WORKSPACE_ROOT)}`\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    config, resolved = load_and_validate(args.config.resolve())
    write_outputs(
        args.config.resolve(),
        args.world.resolve(),
        args.result_dir.resolve(),
        config,
        resolved,
    )
    print(json.dumps({
        "status": "PASS",
        "assets": resolved["audit"]["asset_class_count"],
        "instances": resolved["audit"]["instance_count"],
        "world": str(args.world.resolve()),
        "results": str(args.result_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
