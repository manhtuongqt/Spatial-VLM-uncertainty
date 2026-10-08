"""Offline structural validator for a recorded UR3 spatial dataset run."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ALLOWED_RELATIONS = {
    "left_of", "right_of", "above", "below",
    "front_of", "behind", "inside", "near",
}


def _load_json(path: Path, errors: list):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{path}: cannot read JSON: {exc}")
        return None


def validate_run(run_root: Path) -> list:
    errors = []
    manifest_path = run_root / "manifest.json"
    index_path = run_root / "dataset_index.jsonl"
    if not manifest_path.is_file():
        errors.append(f"missing {manifest_path}")
    else:
        manifest = _load_json(manifest_path, errors)
        if manifest and manifest.get("robot") != "ur3":
            errors.append("manifest.robot must be ur3")
        if manifest and manifest.get("oracle_policy") != "dataset_and_evaluation_only_not_control":
            errors.append("manifest oracle policy is unsafe or unspecified")
    if not index_path.is_file():
        errors.append(f"missing {index_path}")

    episode_dirs = sorted(path for path in run_root.glob("episode_*") if path.is_dir())
    if not episode_dirs:
        errors.append("no episode directories found")
    seen_scene_ids = set()
    required = [
        "episode.json", "result.json", "rgb/view.png", "depth/view.npy",
        "camera_info.json", "tf/tf_snapshot.json", "joint_state.json",
        "ground_truth.json", "spatial_scene.json", "rosbag/metadata.yaml",
    ]
    for episode_dir in episode_dirs:
        for relative in required:
            if not (episode_dir / relative).is_file():
                errors.append(f"{episode_dir.name}: missing {relative}")
        record = _load_json(episode_dir / "episode.json", errors) if (
            episode_dir / "episode.json"
        ).is_file() else None
        scene = _load_json(episode_dir / "spatial_scene.json", errors) if (
            episode_dir / "spatial_scene.json"
        ).is_file() else None
        if record:
            scene_id = record.get("scene_id")
            if scene_id in seen_scene_ids:
                errors.append(f"duplicate scene_id: {scene_id}")
            seen_scene_ids.add(scene_id)
            if not record.get("snapshot_complete"):
                errors.append(f"{episode_dir.name}: snapshot_complete is false")
            if "result" not in record:
                errors.append(f"{episode_dir.name}: result missing")
        if scene:
            objects = scene.get("objects", [])
            object_ids = [item.get("id") for item in objects]
            if len(object_ids) != len(set(object_ids)):
                errors.append(f"{episode_dir.name}: duplicate canonical object ID")
            known = set(object_ids)
            for relation in scene.get("relations", []):
                predicate = relation.get("predicate")
                if predicate not in ALLOWED_RELATIONS:
                    errors.append(f"{episode_dir.name}: invalid relation {predicate}")
                if relation.get("subject_id") not in known or relation.get("object_id") not in known:
                    errors.append(f"{episode_dir.name}: relation references unknown object")
            if scene.get("oracle_usage") != "dataset_and_evaluation_only_not_control":
                errors.append(f"{episode_dir.name}: unsafe oracle usage marker")
            if not scene.get("view_transform_frozen"):
                errors.append(f"{episode_dir.name}: VIEW_POSE transform was not frozen")
        rgb_path = episode_dir / "rgb" / "view.png"
        depth_path = episode_dir / "depth" / "view.npy"
        if rgb_path.is_file() and cv2.imread(str(rgb_path), cv2.IMREAD_COLOR) is None:
            errors.append(f"{episode_dir.name}: RGB image is unreadable")
        if depth_path.is_file():
            try:
                depth = np.load(depth_path, allow_pickle=False)
                if depth.ndim != 2 or not np.isfinite(depth).any():
                    errors.append(f"{episode_dir.name}: invalid depth array")
            except Exception as exc:
                errors.append(f"{episode_dir.name}: depth is unreadable: {exc}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, help="run directory containing manifest.json")
    arguments = parser.parse_args()
    run_root = arguments.run_root.expanduser().resolve()
    errors = validate_run(run_root)
    if errors:
        print(f"DATASET_INVALID: {len(errors)} error(s)")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    episodes = len(list(run_root.glob("episode_*/episode.json")))
    print(f"DATASET_VALID: {run_root} ({episodes} episode(s))")


if __name__ == "__main__":
    main()
