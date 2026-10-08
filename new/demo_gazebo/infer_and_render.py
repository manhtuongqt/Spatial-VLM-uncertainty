#!/usr/bin/env python3
"""Run frozen best V2 on one fresh Gazebo RGB-D pair and save an observation-only panel."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "new/src"))
sys.path.insert(0, str(ROOT / "old/protocol"))

from pcrau.checkpoint import load_model_checkpoint  # noqa: E402
from pcrau.dataset import ANSWER_CLASSES, SOURCE_CLASSES  # noqa: E402
from pcrau.engine import autocast_context  # noqa: E402
from pcrau.model import PCRAUTargetV2  # noqa: E402
from pcrau.policy import decide  # noqa: E402
from pcrau.postprocess import summarize_heatmap  # noqa: E402
from pcrau.text import prompt_anchor_mask, relation_ids, tokenize  # noqa: E402
from pcrau.utils import read_json, seed_everything, sha256_file  # noqa: E402
from pcra_u_development_common import pool_raw_feature, seed_runtime  # noqa: E402
from wp3_feature_hook_smoke import extract_features, load_model, model_inventory_sha256  # noqa: E402


DEFAULT_PROMPT = (
    "For the presently visible object arrangement, locate the banana. "
    "Use the current RGB-D observation. Return the target pixel as (x, y) in image coordinates."
)


def frozen_paths() -> tuple[dict, Path, Path, Path]:
    freeze = read_json(ROOT / "new/test_iid/contracts/best_v2_freeze_lock.json")
    checkpoint = ROOT / freeze["selected_checkpoint"]["path"]
    calibrator_path = ROOT / freeze["frozen_calibrator"]["path"]
    config_path = ROOT / "new/configs/pcrau_target_v2.json"
    checks = [
        (checkpoint, freeze["selected_checkpoint"]["model_sha256"]),
        (checkpoint.parent / "metadata.json", freeze["selected_checkpoint"]["metadata_sha256"]),
        (calibrator_path, freeze["frozen_calibrator"]["sha256"]),
        (config_path, freeze["selected_checkpoint"]["config_sha256"]),
    ]
    for path, expected in checks:
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Frozen artifact missing or hash differs: {path}")
    return freeze, checkpoint, calibrator_path, config_path


def extract_frozen_features(rgb_path: Path, depth_path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    model_root = ROOT / "RoboRefer/models/RoboRefer-2B-SFT"
    manifest = read_json(ROOT / "old/protocol/pcra_u_calibration_feature_manifest.json")
    inventory = model_inventory_sha256(model_root)
    if inventory != manifest["model_inventory_sha256"]:
        raise ValueError("RoboRefer model inventory differs from frozen feature contract")
    seed_runtime(24082026, strict=False)
    started = time.perf_counter()
    backbone = load_model(model_root)
    try:
        extracted = extract_features(backbone, rgb_path, depth_path)
        r_grid, r_thumb = pool_raw_feature(extracted["r0"])
        d_grid, d_thumb = pool_raw_feature(extracted["d0"])
    finally:
        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    tensors = {"r0": r_grid, "d0": d_grid, "r_thumb": r_thumb, "d_thumb": d_thumb}
    return tensors, {
        "model_inventory_sha256": inventory,
        "preprocess_latency_ms": float(extracted["preprocess_latency_ms"]),
        "backbone_feature_latency_ms": float(extracted["feature_latency_ms"]),
        "backbone_load_and_extract_elapsed_s": time.perf_counter() - started,
        "feature_shapes": {name: list(value.shape) for name, value in tensors.items()},
    }


def predict_v2(features: dict[str, torch.Tensor], prompt: str, config: dict,
               checkpoint: Path, config_path: Path, variant: str) -> tuple[dict, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(int(config["seed"]))
    model = PCRAUTargetV2(config).to(device)
    metadata = load_model_checkpoint(model, checkpoint, sha256_file(config_path))
    model.eval()
    cfg = config["model"]
    token_ids, token_mask = tokenize(prompt, int(cfg["max_tokens"]), int(cfg["vocab_size"]))
    rel_ids, rel_mask = relation_ids(prompt, int(cfg["max_relations"]))
    anchor_mask = prompt_anchor_mask(prompt, int(cfg["max_anchors"]))
    batch = {
        **{name: value[None].float().to(device) for name, value in features.items()},
        "token_ids": torch.tensor([token_ids], dtype=torch.long, device=device),
        "token_mask": torch.tensor([token_mask], dtype=torch.bool, device=device),
        "relation_ids": torch.tensor([rel_ids], dtype=torch.long, device=device),
        "relation_mask": torch.tensor([rel_mask], dtype=torch.bool, device=device),
        "anchor_mask": torch.tensor([anchor_mask], dtype=torch.bool, device=device),
    }
    started = time.perf_counter()
    with torch.inference_mode(), autocast_context(device, config["optimization"]):
        output = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logits = output["target_logits"][0].float().cpu()
    heatmap = summarize_heatmap(logits)
    heatmap["probability_grid"] = torch.softmax(logits.flatten(), 0).reshape(logits.shape).tolist()
    answer_prob = output["answerability_logits"][0].float().softmax(-1).cpu()
    source_prob = output["source_logits"][0].float().sigmoid().cpu()
    edge_prob = output["relation_edge_logits"][0].float().sigmoid().cpu()
    edge_mask = torch.tensor(rel_mask, dtype=torch.bool) & torch.tensor(anchor_mask[:len(rel_mask)])
    r_thumb, d_thumb = features["r_thumb"].float(), features["d_thumb"].float()
    sample_id = f"gazebo_observation_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    prediction = {
        "schema_version": 1,
        "sample_id": sample_id,
        "family_id": None,
        "variant": variant,
        "prompt": prompt,
        "spatial": heatmap,
        "answerability_probabilities": {name: float(answer_prob[i]) for i, name in enumerate(ANSWER_CLASSES)},
        "source_probabilities": {name: float(source_prob[i]) for i, name in enumerate(SOURCE_CLASSES)},
        "relation_edge_probabilities": edge_prob.tolist(),
        "relation_consistency": float(edge_prob[edge_mask].mean()) if bool(edge_mask.any()) else 1.0,
        "fusion_gate_mean": float(output["fusion_gate_mean"][0].float().cpu()),
        "rgb_depth_cosine": float(torch.nn.functional.cosine_similarity(r_thumb[None], d_thumb[None]).item()),
        "rgb_depth_mae": float((r_thumb - d_thumb).abs().mean().item()),
        "roborefer_point_xy": None,
        "roborefer_disagreement_normalized": None,
    }
    return prediction, {
        "v2_checkpoint_metadata": metadata,
        "v2_forward_latency_ms": elapsed_ms,
        "device": str(device),
        "relation_ids": rel_ids,
        "relation_mask": rel_mask,
        "anchor_mask": anchor_mask,
    }


def render_panel(output_dir: Path, prediction: dict, decision: dict,
                 camera_id: str) -> None:
    bgr = cv2.imread(str(output_dir / "capture/rgb_original.png"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(output_dir / "capture/depth_relative_model_input.png"), cv2.IMREAD_GRAYSCALE)
    if bgr is None or depth is None:
        raise ValueError("Missing captured RGB or relative depth")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    probability = np.asarray(prediction["spatial"]["probability_grid"], dtype=np.float32)
    if probability.shape != (24, 32):
        raise ValueError("Frozen V2 heatmap must be 24x32")
    upsampled = cv2.resize(probability, (640, 480), interpolation=cv2.INTER_LINEAR)
    candidate_x, candidate_y = prediction["spatial"]["map_pixel_xy"]
    answer = max(prediction["answerability_probabilities"],
                 key=prediction["answerability_probabilities"].get)
    sources = sorted(prediction["source_probabilities"].items(), key=lambda item: item[1], reverse=True)
    risk = float(decision["calibrated_grounding_risk"])
    threshold = float(decision["risk_threshold"])

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 6.0))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.76, bottom=0.23, wspace=0.035)
    fig.suptitle("Gazebo RGB-D → frozen best V2 → selective decision", x=0.02, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(0.02, 0.87, f"Prompt: {prediction['prompt']}", fontsize=8.9)
    axes[0].imshow(rgb)
    axes[0].set_title("(a) Fresh Gazebo RGB", loc="left", fontweight="bold", fontsize=10)
    axes[1].imshow(depth, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("(b) Relative depth input", loc="left", fontweight="bold", fontsize=10)
    axes[2].imshow(rgb)
    visible_heatmap = np.ma.masked_where(upsampled < float(probability.max()) * 0.01, upsampled)
    axes[2].imshow(visible_heatmap, cmap="magma", vmin=0, vmax=float(probability.max()), alpha=0.70)
    axes[2].scatter([candidate_x], [candidate_y], s=145, marker="x", color="#00BBD6",
                    linewidths=2.7, zorder=10)
    axes[2].set_title("(c) V2 target heatmap + candidate MAP", loc="left", fontweight="bold", fontsize=10)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)
    fig.text(0.02, 0.165,
             f"Action: {decision['action']}     predicted state: {answer}     "
             f"model-reported risk: {risk:.3f}     frozen τ: {threshold:.3f}     MAP: ({candidate_x}, {candidate_y})",
             fontsize=11.2, fontweight="bold", color="#20252B")
    fig.text(0.02, 0.115,
             f"Top uncertainty sources: {sources[0][0]} {sources[0][1]:.2f}, "
             f"{sources[1][0]} {sources[1][1]:.2f}   ·   "
             "Cyan × is a candidate image point; no actuation command is sent.",
             fontsize=9.1, color="#4E5A63")
    warning = ("Fixed overhead camera is outside the evaluated wrist-camera setup"
               if camera_id == "top_table_camera" else
               "Wrist-mounted preview pose and rearranged tabletop differ from evaluated Test-IID")
    fig.text(0.02, 0.055,
             f"{warning}: risk here is illustrative, not validated calibration or grasp success.",
             fontsize=8.5, color="#A33824")
    fig.savefig(output_dir / "demo_panel.png", dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "capture/capture.json").is_file():
        raise FileNotFoundError("Capture metadata missing; run Gazebo RGB-D capture first")
    prompt = args.prompt.strip()
    if not prompt:
        raise ValueError("Prompt must be nonempty")
    freeze, checkpoint, calibrator_path, config_path = frozen_paths()
    config = read_json(config_path)
    calibrator = read_json(calibrator_path)
    capture_meta = read_json(run_dir / "capture/capture.json")
    if capture_meta["rgb_topic"] not in ("/top_table_camera/image", "/wrist_camera/image"):
        raise ValueError("Inference accepts only the top or actual UR3 wrist RGB-D camera")
    if capture_meta["safety"]["robot_motion_commanded"]:
        raise ValueError("Observation-only demo refuses a robot-motion capture")
    camera_id = capture_meta["rgb_topic"].split("/")[1]
    if camera_id == "wrist_camera" and not capture_meta["safety"]["robot_spawned"]:
        raise ValueError("Wrist capture metadata must identify the static UR3 preview")
    rgb_path = run_dir / "capture/rgb_model_input.jpg"
    depth_path = run_dir / "capture/depth_relative_model_input.png"
    if sha256_file(rgb_path) != capture_meta["artifacts_sha256"]["rgb_model_input"]:
        raise ValueError("Captured RGB hash mismatch")
    if sha256_file(depth_path) != capture_meta["artifacts_sha256"]["depth_model_input"]:
        raise ValueError("Captured depth hash mismatch")
    start = time.perf_counter()
    features, feature_runtime = extract_frozen_features(rgb_path, depth_path)
    prediction, model_runtime = predict_v2(
        features, prompt, config, checkpoint, config_path,
        variant=f"live_gazebo_{camera_id}")
    decision = decide(prediction, calibrator)
    with (run_dir / "prediction.json").open("x", encoding="utf-8") as handle:
        json.dump(prediction, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with (run_dir / "decision.json").open("x", encoding="utf-8") as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    render_panel(run_dir, prediction, decision, camera_id)
    world_path = (run_dir / "world_with_static_ur3.sdf" if camera_id == "wrist_camera"
                  else ROOT / "ur3/ur_simulation_gz/worlds/ur3_pick_place_uq_occlusion_v2.sdf")
    if not world_path.is_file():
        raise FileNotFoundError(f"Gazebo world provenance missing: {world_path}")
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "OBSERVATION_ONLY_DEMO",
        "camera": camera_id,
        "view_domain": ("overhead_outside_evaluated_wrist_camera_distribution"
                        if camera_id == "top_table_camera" else
                        "wrist_preview_pose_and_layout_not_evaluated_test_iid"),
        "world_path": str(world_path.relative_to(ROOT)),
        "world_sha256": sha256_file(world_path),
        "prompt": prompt,
        "checkpoint_path": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": freeze["selected_checkpoint"]["model_sha256"],
        "calibrator_path": str(calibrator_path.relative_to(ROOT)),
        "calibrator_sha256": freeze["frozen_calibrator"]["sha256"],
        "source_rgb_sha256": capture_meta["artifacts_sha256"]["rgb_model_input"],
        "source_depth_sha256": capture_meta["artifacts_sha256"]["depth_model_input"],
        "outputs_sha256": {
            name: sha256_file(run_dir / name)
            for name in ("prediction.json", "decision.json", "demo_panel.png")
        },
        "runtime": {**feature_runtime, **model_runtime, "elapsed_total_s": time.perf_counter() - start},
        "safety": {"robot_spawned": capture_meta["safety"]["robot_spawned"],
                   "robot_motion_commanded": False,
                   "physical_task_success_measured": False,
                   "risk_calibration_validated_for_this_camera": False},
    }
    with (run_dir / "run_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"run_dir": str(run_dir), "action": decision["action"],
                      "risk": decision["calibrated_grounding_risk"],
                      "map_pixel_xy": prediction["spatial"]["map_pixel_xy"],
                      "elapsed_total_s": manifest["runtime"]["elapsed_total_s"]}, indent=2))


if __name__ == "__main__":
    main()
