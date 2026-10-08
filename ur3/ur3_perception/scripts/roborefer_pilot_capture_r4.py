#!/usr/bin/env python3
"""Append-only R4 sensor-synchronization capture implementation.

The frozen capture implementation keeps only the latest frame from each
sensor.  This revision keeps bounded timestamp-keyed queues, selects the
newest exact RGB/depth/semantic tuple received after the post-settle freshness
barrier, and records liveness diagnostics on failure.  It does not change
scene geometry, capture order, timeout, synchronization tolerance, labels,
or any scientific gate.
"""
from __future__ import annotations

from collections import OrderedDict
import json
import time
from typing import Any

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock

import roborefer_pilot_capture as frozen


QUEUE_LIMIT = 64
IMPLEMENTATION_REVISION = "r4_timestamp_queue_liveness"


def stamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class R4PilotCapture(frozen.PilotCapture):
    """Frozen capture behavior with bounded exact-timestamp synchronization."""

    def __init__(self) -> None:
        self._queues: dict[str, OrderedDict[int, tuple[float, np.ndarray]]] = {
            name: OrderedDict() for name in ("rgb", "depth", "labels")
        }
        self._message_counts = {name: 0 for name in ("rgb", "depth", "labels", "camera_info", "clock")}
        self._decode_errors: dict[str, list[str]] = {name: [] for name in ("rgb", "depth", "labels")}
        self._last_wall_time: dict[str, float | None] = {
            name: None for name in ("rgb", "depth", "labels", "camera_info", "clock")
        }
        self._last_stamp_ns: dict[str, int | None] = {name: None for name in ("rgb", "depth", "labels")}
        self._clock_ns: int | None = None
        self._last_match_key_ns: int | None = None
        super().__init__()
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(Clock, "/clock", self._clock_callback, sensor_qos)

    def _remember(self, name: str, key: int, stamp: float, value: np.ndarray) -> None:
        queue = self._queues[name]
        queue[key] = (stamp, value)
        queue.move_to_end(key)
        while len(queue) > QUEUE_LIMIT:
            queue.popitem(last=False)
        self._message_counts[name] += 1
        self._last_wall_time[name] = time.monotonic()
        self._last_stamp_ns[name] = key

    def _record_decode_error(self, name: str, exc: Exception) -> None:
        errors = self._decode_errors[name]
        if len(errors) < 20:
            errors.append(str(exc))
        self.get_logger().warning(f"R4 {name} decode failed: {exc}")

    def _rgb_callback(self, message) -> None:
        try:
            frame = np.asarray(
                self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            ).copy()
            self._remember("rgb", stamp_ns(message), frozen.stamp_seconds(message), frame)
        except Exception as exc:  # pragma: no cover - exercised only by ROS transport
            self._record_decode_error("rgb", exc)

    def _depth_callback(self, message) -> None:
        try:
            depth = np.asarray(
                self._bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            )
            if depth.ndim == 3:
                depth = depth[..., 0]
            depth_m = (
                depth.astype(np.float32) * 0.001
                if message.encoding.upper() in ("16UC1", "MONO16")
                else depth.astype(np.float32)
            )
            self._remember("depth", stamp_ns(message), frozen.stamp_seconds(message), depth_m.copy())
        except Exception as exc:  # pragma: no cover
            self._record_decode_error("depth", exc)

    def _labels_callback(self, message) -> None:
        try:
            labels = np.asarray(
                self._bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            )
            if labels.ndim == 3:
                if labels.shape[2] != 3 or not (
                    np.array_equal(labels[..., 0], labels[..., 1])
                    and np.array_equal(labels[..., 1], labels[..., 2])
                ):
                    raise ValueError("semantic labels_map RGB channels disagree")
                labels = labels[..., 0]
            self._remember(
                "labels", stamp_ns(message), frozen.stamp_seconds(message), labels.astype(np.uint8).copy()
            )
        except Exception as exc:  # pragma: no cover
            self._record_decode_error("labels", exc)

    def _camera_info_callback(self, message) -> None:
        self._camera_info = message
        self._message_counts["camera_info"] += 1
        self._last_wall_time["camera_info"] = time.monotonic()

    def _clock_callback(self, message: Clock) -> None:
        self._clock_ns = int(message.clock.sec) * 1_000_000_000 + int(message.clock.nanosec)
        self._message_counts["clock"] += 1
        self._last_wall_time["clock"] = time.monotonic()

    def _clear_sync_buffers(self) -> None:
        for queue in self._queues.values():
            queue.clear()
        self._last_match_key_ns = None

    def _sensor_tuple(self):
        common = set(self._queues["rgb"]).intersection(self._queues["depth"], self._queues["labels"])
        if not common:
            return None
        # The newest exact tuple is both synchronized and maximally separated
        # from the reset/settle boundary.
        key = max(common)
        rgb_stamp, rgb = self._queues["rgb"][key]
        depth_stamp, depth = self._queues["depth"][key]
        labels_stamp, labels = self._queues["labels"][key]
        self._last_match_key_ns = key
        for queue in self._queues.values():
            for old_key in [candidate for candidate in queue if candidate <= key]:
                del queue[old_key]
        if rgb.shape[:2] != depth.shape[:2] or rgb.shape[:2] != labels.shape[:2]:
            raise RuntimeError(
                f"sensor size mismatch RGB={rgb.shape[:2]}, depth={depth.shape[:2]}, labels={labels.shape[:2]}"
            )
        stamps = [rgb_stamp, depth_stamp, labels_stamp]
        spread = max(stamps) - min(stamps)
        allowed = float(self.get_parameter("sync_slop_sec").value)
        if spread > allowed:
            raise RuntimeError(f"exact timestamp matcher produced spread {spread} > {allowed}")
        return rgb.copy(), depth.copy(), labels.copy(), spread, stamps

    def _publisher_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        topics = {
            "rgb": str(self.get_parameter("rgb_topic").value),
            "depth": str(self.get_parameter("depth_topic").value),
            "labels": str(self.get_parameter("semantic_label_topic").value),
            "camera_info": str(self.get_parameter("camera_info_topic").value),
            "clock": "/clock",
        }
        output: dict[str, list[dict[str, Any]]] = {}
        for name, topic in topics.items():
            rows = []
            for info in self.get_publishers_info_by_topic(topic):
                qos = info.qos_profile
                rows.append({
                    "node_name": info.node_name,
                    "node_namespace": info.node_namespace,
                    "topic_type": info.topic_type,
                    "qos_reliability": str(qos.reliability),
                    "qos_durability": str(qos.durability),
                    "qos_history": str(qos.history),
                    "qos_depth": int(qos.depth),
                })
            output[name] = rows
        return output

    def _liveness_snapshot(self, scene_id: str, phase: str) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "implementation_revision": IMPLEMENTATION_REVISION,
            "scene_id": scene_id,
            "phase": phase,
            "captured_at_monotonic_sec": now,
            "message_counts": dict(self._message_counts),
            "decode_errors": dict(self._decode_errors),
            "last_message_age_sec": {
                name: None if value is None else now - value
                for name, value in self._last_wall_time.items()
            },
            "last_sensor_stamp_ns": dict(self._last_stamp_ns),
            "clock_ns": self._clock_ns,
            "queue_lengths": {name: len(queue) for name, queue in self._queues.items()},
            "queue_stamp_ranges_ns": {
                name: ([min(queue), max(queue)] if queue else None)
                for name, queue in self._queues.items()
            },
            "last_match_key_ns": self._last_match_key_ns,
            "publishers": self._publisher_snapshot(),
        }

    def _save_scene(self, scene: dict, layout: dict, sensor_tuple) -> dict:
        record = super()._save_scene(scene, layout, sensor_tuple)
        record["capture"]["r4_synchronization"] = {
            "implementation_revision": IMPLEMENTATION_REVISION,
            "matcher": "newest_exact_header_timestamp_from_bounded_queues",
            "queue_limit_per_topic": QUEUE_LIMIT,
            "post_settle_freshness_barrier": True,
            "matched_stamp_ns": self._last_match_key_ns,
            "message_counts_at_capture": dict(self._message_counts),
            "decode_error_counts": {name: len(values) for name, values in self._decode_errors.items()},
        }
        frozen.write_json(
            self._output_root / scene["scene_id"] / "input/scene_input.json", record
        )
        return record

    def run(self) -> bool:
        if not self._reset_client.wait_for_service(
            timeout_sec=float(self.get_parameter("service_timeout_sec").value)
        ):
            self.get_logger().error("Gazebo SetEntityPose service unavailable")
            return False
        timeout = float(self.get_parameter("capture_timeout_sec").value)
        if not self._spin_until(lambda: self._camera_info is not None, timeout):
            self.get_logger().error("camera info unavailable")
            return False
        if not self._spin_until(self._view_pose_ready, timeout):
            self.get_logger().error("locked wrist-camera view joint pose was not reached")
            return False

        manifest = {
            "schema_version": 1,
            "protocol_id": self._config["protocol_id"],
            "created_wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "capture_mode": "no_manipulation_shadow",
            "implementation_revision": IMPLEMENTATION_REVISION,
            "scene_count": len(self._config["scenes"]),
            "scene_config_sha256": frozen.sha256_file(self._config_path),
            "annotation_semantics_consumed_by_capture": False,
            "annotation_bytes_hashed_before_inference": True,
            "annotation_file_sha256": frozen.sha256_file(self._annotation_path),
            "gate_config_bytes_hashed_before_capture": True,
            "gate_config_file_sha256": frozen.sha256_file(self._gate_config_path),
            "pretrial_source_lock_file": str(self._pretrial_lock_path),
            "pretrial_source_lock_sha256": frozen.sha256_file(self._pretrial_lock_path),
            "source_artifacts_verified_before_capture": True,
            "model_inventory_sha256_preregistered": self._pretrial_lock.get("model_inventory_sha256"),
            "semantic_labels_are_evaluator_only": True,
            "target_handoff_published": False,
            "robot_manipulation_performed": False,
            "view_joint_pose_verified": True,
            "view_joint_pose": self._config["view_joint_pose"],
            "sync_contract": {
                "matcher": "newest_exact_header_timestamp_from_bounded_queues",
                "queue_limit_per_topic": QUEUE_LIMIT,
                "sync_slop_sec_unchanged": float(self.get_parameter("sync_slop_sec").value),
                "capture_timeout_sec_unchanged": timeout,
                "post_settle_freshness_barrier": True,
                "automatic_restart_or_scene_retry": False,
            },
        }
        frozen.write_json(self._output_root / "capture_manifest.json", manifest)
        index_path = self._output_root / "input_manifest.jsonl"
        captured_rgb_hashes: list[str] = []
        total_scenes = len(self._config["scenes"])

        for index, scene in enumerate(self._config["scenes"], start=1):
            layout = frozen.scene_layout(self._config, scene)
            self.get_logger().info(f"R4_CAPTURE_RESET {index}/{total_scenes}: {scene['scene_id']}")
            for model_name, pose_values in layout.items():
                self._reset_entity(model_name, pose_values)

            self._clear_sync_buffers()
            settle_deadline = time.monotonic() + float(self.get_parameter("settle_sec").value)
            while rclpy.ok() and time.monotonic() < settle_deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
            # Frames acquired during settling are never eligible.
            self._clear_sync_buffers()
            clock_before = self._clock_ns
            captured_tuple = None

            def freeze_tuple():
                nonlocal captured_tuple
                captured_tuple = self._sensor_tuple()
                return captured_tuple is not None

            if not self._spin_until(freeze_tuple, timeout):
                diagnostic = self._liveness_snapshot(scene["scene_id"], "tuple_timeout")
                diagnostic["clock_before_wait_ns"] = clock_before
                diagnostic["clock_advanced_during_wait"] = (
                    clock_before is not None and self._clock_ns is not None and self._clock_ns > clock_before
                )
                frozen.write_json(self._output_root / "R4_SENSOR_FAILURE_DIAGNOSTICS.json", diagnostic)
                self.get_logger().error(f"R4 synchronized RGB-D-label tuple unavailable: {scene['scene_id']}")
                return False
            record = self._save_scene(scene, layout, captured_tuple)
            captured_rgb_hashes.append(record["input_sha256"]["rgb"])
            with index_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.get_logger().info(f"R4_SCENE_CAPTURED {index}/{total_scenes}: {scene['scene_id']}")

        if len(set(captured_rgb_hashes)) != len(captured_rgb_hashes):
            raise RuntimeError("RGB_FROZEN_STREAM_QC_FAILED: duplicate captured RGB SHA-256")
        manifest["input_only_sensor_qc"] = {
            "passed": True,
            "all_scene_rgb_sha256_unique": True,
            "rgb_scene_count": len(captured_rgb_hashes),
            "r4_final_liveness": self._liveness_snapshot("ALL", "capture_complete"),
            "thresholds": {
                "min_rgb_std": frozen.MIN_RGB_STD,
                "min_rgb_dynamic_range": frozen.MIN_RGB_DYNAMIC_RANGE,
                "min_rgb_nonzero_fraction": frozen.MIN_RGB_NONZERO_FRACTION,
                "min_valid_depth_fraction": frozen.MIN_VALID_DEPTH_FRACTION,
                "min_depth_dynamic_range_m": frozen.MIN_DEPTH_DYNAMIC_RANGE_M,
                "max_rgb_depth_label_spread_sec": float(self.get_parameter("sync_slop_sec").value),
            },
        }
        manifest["completed_wall_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        manifest["input_manifest_sha256"] = frozen.sha256_file(index_path)
        evaluator_hashes = {}
        for scene in self._config["scenes"]:
            scene_id = scene["scene_id"]
            for relative_name in ("evaluator/semantic_labels.png", "evaluator/capture_oracle.json"):
                artifact = self._output_root / scene_id / relative_name
                evaluator_hashes[f"{scene_id}/{relative_name}"] = frozen.sha256_file(artifact)
        manifest["evaluator_artifact_sha256"] = evaluator_hashes
        manifest["status"] = "COMPLETE"
        frozen.write_json(self._output_root / "capture_manifest.json", manifest)
        self.get_logger().info(f"R4_CAPTURE_COMPLETE: {self._output_root} ({total_scenes} scenes)")
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = R4PilotCapture()
    try:
        success = node.run()
    except Exception as exc:
        node.get_logger().error(f"R4_CAPTURE_FATAL: {exc}")
        success = False
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
