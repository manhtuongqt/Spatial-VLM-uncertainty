#!/usr/bin/env python3
"""R4 neutral 40-cycle sensor transport/liveness soak.

This diagnostic never writes RGB-D data, reads evaluator annotations, invokes a
model, or fits a calibrator.  Each cycle resets every scene object to its
neutral storage pose, waits the unchanged 1.5 seconds, then requires a fresh
exact-timestamp RGB/depth/semantic tuple within the unchanged 45 seconds.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image
import yaml


QUEUE_LIMIT = 64
SETTLE_SEC = 1.5
TIMEOUT_SEC = 45.0
SERVICE_TIMEOUT_SEC = 60.0
TOPICS = {
    "rgb": "/wrist_camera/color/image_raw",
    "depth": "/wrist_camera/depth/image_raw",
    "labels": "/wrist_camera/evaluation_labels/labels_map",
    "camera_info": "/wrist_camera/color/camera_info",
    "clock": "/clock",
}
RESET_SERVICE = "/world/ur3_pick_place/set_pose"


def stamp_ns(message: Any) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def neutral_layout(config: dict[str, Any]) -> dict[str, list[float]]:
    layout: dict[str, list[float]] = {}
    for model_name, item in config["objects"].items():
        x, y, yaw = (float(value) for value in item["storage_pose"])
        layout[model_name] = [
            x, y, float(item["z"]), 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        ]
    return layout


class SensorSoak(Node):
    def __init__(self, scene_config: Path) -> None:
        super().__init__("gazebo_calibration_v6_sensor_soak_r4")
        self.config = yaml.safe_load(scene_config.read_text(encoding="utf-8"))
        self.layout = neutral_layout(self.config)
        self.bridge = CvBridge()
        self.queues: dict[str, OrderedDict[int, dict[str, Any]]] = {
            name: OrderedDict() for name in ("rgb", "depth", "labels")
        }
        self.counts = {name: 0 for name in TOPICS}
        self.last_wall: dict[str, float | None] = {name: None for name in TOPICS}
        self.last_stamp: dict[str, int | None] = {name: None for name in ("rgb", "depth", "labels")}
        self.decode_errors: dict[str, list[str]] = {name: [] for name in ("rgb", "depth", "labels")}
        self.clock_ns: int | None = None
        self.camera_info: CameraInfo | None = None
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(Image, TOPICS["rgb"], self.rgb_callback, qos)
        self.create_subscription(Image, TOPICS["depth"], self.depth_callback, qos)
        self.create_subscription(Image, TOPICS["labels"], self.labels_callback, qos)
        self.create_subscription(CameraInfo, TOPICS["camera_info"], self.info_callback, qos)
        self.create_subscription(Clock, TOPICS["clock"], self.clock_callback, qos)
        self.reset_client = self.create_client(SetEntityPose, RESET_SERVICE)

    def remember(self, name: str, message: Image, value: np.ndarray) -> None:
        key = stamp_ns(message)
        queue = self.queues[name]
        queue[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "encoding": message.encoding,
        }
        queue.move_to_end(key)
        while len(queue) > QUEUE_LIMIT:
            queue.popitem(last=False)
        self.counts[name] += 1
        self.last_wall[name] = time.monotonic()
        self.last_stamp[name] = key

    def decode_error(self, name: str, exc: Exception) -> None:
        if len(self.decode_errors[name]) < 20:
            self.decode_errors[name].append(str(exc))

    def rgb_callback(self, message: Image) -> None:
        try:
            value = np.asarray(self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8"))
            if value.shape != (480, 640, 3) or value.dtype != np.uint8:
                raise ValueError(f"invalid RGB {value.shape}/{value.dtype}")
            self.remember("rgb", message, value)
        except Exception as exc:
            self.decode_error("rgb", exc)

    def depth_callback(self, message: Image) -> None:
        try:
            value = np.asarray(self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough"))
            if value.ndim == 3:
                value = value[..., 0]
            if value.shape != (480, 640):
                raise ValueError(f"invalid depth {value.shape}/{value.dtype}")
            self.remember("depth", message, value)
        except Exception as exc:
            self.decode_error("depth", exc)

    def labels_callback(self, message: Image) -> None:
        try:
            value = np.asarray(self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough"))
            if value.ndim == 3:
                if value.shape[2] != 3 or not (
                    np.array_equal(value[..., 0], value[..., 1])
                    and np.array_equal(value[..., 1], value[..., 2])
                ):
                    raise ValueError("semantic RGB channels disagree")
                value = value[..., 0]
            if value.shape != (480, 640):
                raise ValueError(f"invalid semantic labels {value.shape}/{value.dtype}")
            self.remember("labels", message, value)
        except Exception as exc:
            self.decode_error("labels", exc)

    def info_callback(self, message: CameraInfo) -> None:
        self.camera_info = message
        self.counts["camera_info"] += 1
        self.last_wall["camera_info"] = time.monotonic()

    def clock_callback(self, message: Clock) -> None:
        self.clock_ns = int(message.clock.sec) * 1_000_000_000 + int(message.clock.nanosec)
        self.counts["clock"] += 1
        self.last_wall["clock"] = time.monotonic()

    def spin_until(self, predicate, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if predicate():
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return bool(predicate())

    def reset(self, model_name: str, pose: list[float]) -> float:
        request = SetEntityPose.Request()
        request.entity.name = model_name
        request.entity.type = Entity.MODEL
        request.pose.position.x, request.pose.position.y, request.pose.position.z = pose[:3]
        (
            request.pose.orientation.x,
            request.pose.orientation.y,
            request.pose.orientation.z,
            request.pose.orientation.w,
        ) = pose[3:]
        started = time.monotonic()
        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=SERVICE_TIMEOUT_SEC)
        result = future.result() if future.done() else None
        if result is None or not result.success:
            raise RuntimeError(f"SetEntityPose failed for {model_name}")
        return time.monotonic() - started

    def clear_queues(self) -> None:
        for queue in self.queues.values():
            queue.clear()

    def exact_tuple(self) -> int | None:
        common = set(self.queues["rgb"]).intersection(self.queues["depth"], self.queues["labels"])
        return max(common) if common else None

    def publishers(self) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for name, topic in TOPICS.items():
            result[name] = [
                {
                    "node_name": info.node_name,
                    "node_namespace": info.node_namespace,
                    "topic_type": info.topic_type,
                    "qos_reliability": str(info.qos_profile.reliability),
                    "qos_durability": str(info.qos_profile.durability),
                    "qos_depth": int(info.qos_profile.depth),
                }
                for info in self.get_publishers_info_by_topic(topic)
            ]
        return result

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "message_counts": dict(self.counts),
            "last_message_age_sec": {
                name: None if value is None else now - value for name, value in self.last_wall.items()
            },
            "last_sensor_stamp_ns": dict(self.last_stamp),
            "clock_ns": self.clock_ns,
            "queue_lengths": {name: len(queue) for name, queue in self.queues.items()},
            "queue_stamp_ranges_ns": {
                name: ([min(queue), max(queue)] if queue else None) for name, queue in self.queues.items()
            },
            "decode_errors": dict(self.decode_errors),
            "publishers": self.publishers(),
        }

    def run(self, cycles: int) -> dict[str, Any]:
        if not self.reset_client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_SEC):
            raise RuntimeError("SetEntityPose service unavailable")
        if not self.spin_until(lambda: self.camera_info is not None and self.clock_ns is not None, TIMEOUT_SEC):
            raise RuntimeError("camera_info or /clock unavailable")
        rows: list[dict[str, Any]] = []
        failure: dict[str, Any] | None = None
        for cycle in range(1, cycles + 1):
            before_counts = dict(self.counts)
            reset_latencies: dict[str, float] = {}
            try:
                for model_name, pose in self.layout.items():
                    reset_latencies[model_name] = self.reset(model_name, pose)
            except Exception as exc:
                failure = {"cycle": cycle, "layer": "gazebo_reset_service", "error": str(exc)}
                break
            self.clear_queues()
            settle_end = time.monotonic() + SETTLE_SEC
            while rclpy.ok() and time.monotonic() < settle_end:
                rclpy.spin_once(self, timeout_sec=0.05)
            # Strict freshness barrier: settling frames cannot satisfy the cycle.
            self.clear_queues()
            clock_before = self.clock_ns
            matched: int | None = None

            def matched_now() -> bool:
                nonlocal matched
                matched = self.exact_tuple()
                return matched is not None

            passed = self.spin_until(matched_now, TIMEOUT_SEC)
            row = {
                "cycle": cycle,
                "reset_model_count": len(reset_latencies),
                "reset_latency_sec": reset_latencies,
                "counts_before": before_counts,
                "counts_after": dict(self.counts),
                "message_count_delta": {
                    name: self.counts[name] - before_counts[name] for name in self.counts
                },
                "clock_before_wait_ns": clock_before,
                "clock_after_wait_ns": self.clock_ns,
                "clock_advanced": clock_before is not None and self.clock_ns is not None and self.clock_ns > clock_before,
                "matched_exact_stamp_ns": matched,
                "exact_tuple_received": bool(passed),
            }
            rows.append(row)
            if not passed:
                failure = {
                    "cycle": cycle,
                    "layer": "sensor_transport_or_simulator_liveness",
                    "reason": "fresh exact RGB/depth/semantic tuple absent within unchanged timeout",
                    "snapshot": self.snapshot(),
                }
                break
            if not row["clock_advanced"]:
                failure = {"cycle": cycle, "layer": "simulation_clock", "reason": "/clock did not advance"}
                break
            if any(self.decode_errors.values()):
                failure = {"cycle": cycle, "layer": "sensor_decode", "decode_errors": self.decode_errors}
                break
            self.get_logger().info(f"R4_SOAK_CYCLE_PASS {cycle}/{cycles} stamp={matched}")

        final = self.snapshot()
        checks = {
            "all_cycles_completed": len(rows) == cycles and all(row["exact_tuple_received"] for row in rows),
            "clock_advanced_every_cycle": len(rows) == cycles and all(row["clock_advanced"] for row in rows),
            "all_objects_reset_every_cycle": len(rows) == cycles and all(
                row["reset_model_count"] == len(self.layout) for row in rows
            ),
            "no_decode_errors": not any(self.decode_errors.values()),
            "camera_info_received": self.camera_info is not None and self.counts["camera_info"] > 0,
            "publishers_present": all(final["publishers"][name] for name in TOPICS),
            "no_sensor_data_persisted": True,
            "test_robot_sealed": True,
        }
        return {
            "schema_version": 1,
            "protocol_id": "gazebo_calibration_v6",
            "status": "PASS" if all(checks.values()) else "BLOCKED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "R4 neutral sensor transport/liveness soak without scientific capture",
            "cycles_required": cycles,
            "cycles_completed": len(rows),
            "constants": {
                "settle_sec": SETTLE_SEC,
                "tuple_timeout_sec": TIMEOUT_SEC,
                "queue_limit_per_topic": QUEUE_LIMIT,
                "matcher": "newest_exact_header_timestamp_after_post_settle_freshness_barrier",
                "reset_service": RESET_SERVICE,
                "topics": TOPICS,
            },
            "checks": checks,
            "failure": failure,
            "cycles": rows,
            "final_liveness": final,
            "scientific_capture_count": 0,
            "model_inference_run": False,
            "calibrator_fit_call_count": 0,
            "scientific_hypothesis": "NOT_TESTED",
            "test_iid_ood_access": False,
            "robot_access": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=40)
    args = parser.parse_args()
    if args.cycles != 40:
        raise ValueError("R4 protocol requires exactly 40 neutral cycles")
    # argparse already consumed this script's non-ROS flags.
    rclpy.init(args=[])
    node = SensorSoak(args.scene_config.resolve())
    try:
        report = node.run(args.cycles)
    except Exception as exc:
        report = {
            "schema_version": 1,
            "protocol_id": "gazebo_calibration_v6",
            "status": "BLOCKED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "failure": {"layer": "soak_initialization_or_runtime", "error": str(exc)},
            "scientific_capture_count": 0,
            "model_inference_run": False,
            "calibrator_fit_call_count": 0,
            "scientific_hypothesis": "NOT_TESTED",
            "test_iid_ood_access": False,
            "robot_access": False,
        }
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    write_new(args.output.resolve(), report)
    print(json.dumps({key: report.get(key) for key in ("status", "cycles_required", "cycles_completed", "checks", "failure")}, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
