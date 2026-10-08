#!/usr/bin/env python3
"""Responsive UR3 camera dashboard with RoboRefer dimension comparison."""

from typing import Dict

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class MultimodalImageViewer(Node):
    """Cache latest image messages and render a lightweight 2x2 dashboard."""

    def __init__(self) -> None:
        super().__init__("multimodal_image_viewer")
        self.declare_parameter("camera_topic", "/wrist_camera/color/image_raw")
        self.declare_parameter(
            "roborefer_topic", "/ur3_perception/roborefer/image"
        )
        self.declare_parameter(
            "roborefer_mask_topic", "/ur3_perception/roborefer/mask"
        )
        self.declare_parameter(
            "roborefer_comparison_topic",
            "/ur3_perception/roborefer/dimension_comparison",
        )
        self.declare_parameter("roborefer_comparison_mode", "dimensions")
        self.declare_parameter("panel_width", 560)
        self.declare_parameter("panel_height", 420)
        # Rendering four resized panels at camera rate can consume a full CPU core.
        # Ten FPS remains responsive while leaving headroom for perception/control.
        self.declare_parameter("render_rate_hz", 10.0)

        self._messages: Dict[str, Image] = {}
        self._frames: Dict[str, np.ndarray] = {}
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        retained_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        topics = {"CAMERA": str(self.get_parameter("camera_topic").value)}
        self._comparison_mode = str(
            self.get_parameter("roborefer_comparison_mode").value
        ).strip().lower()
        self._top_right_label = "ROBOREFER DEPTH MASK"
        self._bottom_left_label = "ROBOREFER POINT + BBOX"
        if self._comparison_mode == "near_far_ablation":
            self._bottom_right_label = "RGB ONLY vs RGB-D: NEAR/FAR"
        elif self._comparison_mode == "relative_position":
            self._bottom_right_label = "RELATIVE POSITION: RGB-D 3D"
        elif self._comparison_mode == "five_step_reasoning":
            self._bottom_right_label = "5-STEP REASONING -> PICK GATE"
        else:
            self._bottom_right_label = "ROBOREFER -> DEPTH -> 3D"
        topics[self._top_right_label] = str(
            self.get_parameter("roborefer_mask_topic").value
        )
        topics[self._bottom_left_label] = str(
            self.get_parameter("roborefer_topic").value
        )
        topics[self._bottom_right_label] = str(
            self.get_parameter("roborefer_comparison_topic").value
        )
        for label, topic in topics.items():
            self.create_subscription(
                Image,
                topic,
                lambda message, key=label: self._image_callback(key, message),
                (
                    retained_qos if label != "CAMERA" else sensor_qos
                ),
            )

        if self._comparison_mode == "near_far_ablation":
            self._window_name = (
                "UR3 RoboRefer ablation: same RGB/prompt | "
                "RGB-only vs RGB-D near/far"
            )
        elif self._comparison_mode == "relative_position":
            self._window_name = (
                "UR3 RoboRefer RGB-D: relative position in camera and base_link"
            )
        elif self._comparison_mode == "five_step_reasoning":
            self._window_name = (
                "UR3 RoboRefer: five-clause spatial reasoning -> verified pick"
            )
        else:
            self._window_name = (
                "UR3 RoboRefer RGB-D: Camera | Depth mask | Point + bbox | "
                "Two objects -> metric 3D comparison"
            )
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        self._panel_width = int(self.get_parameter("panel_width").value)
        self._panel_height = int(self.get_parameter("panel_height").value)
        cv2.resizeWindow(
            self._window_name,
            self._panel_width * 2,
            self._panel_height * 2 + 114,
        )
        cv2.moveWindow(self._window_name, 20, 60)
        try:
            cv2.setWindowProperty(self._window_name, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            self.get_logger().warning("Window manager does not support top-most mode")
        render_rate = max(5.0, float(self.get_parameter("render_rate_hz").value))
        self.create_timer(1.0 / render_rate, self._render)
        self.get_logger().info(
            f"Multimodal dashboard ready at {render_rate:.0f} FPS; "
            "RoboRefer RGB-D pipeline"
        )

    def _image_callback(self, label: str, message: Image) -> None:
        # Keep only the newest ROS message. Conversion/resizing happens at the
        # dashboard rate, not once for every incoming camera frame.
        self._messages[label] = message

    @staticmethod
    def _image_to_bgr(message: Image) -> np.ndarray:
        """Decode common ROS image encodings without the NumPy-1-only cv_bridge."""
        encoding = message.encoding.lower()
        channels_by_encoding = {
            "bgr8": 3,
            "rgb8": 3,
            "bgra8": 4,
            "rgba8": 4,
            "mono8": 1,
            "8uc1": 1,
            "8uc3": 3,
            "8uc4": 4,
        }
        channels = channels_by_encoding.get(encoding)
        if channels is None:
            raise ValueError(f"unsupported image encoding: {message.encoding}")

        packed_width = int(message.width) * channels
        row_step = int(message.step)
        raw = np.frombuffer(message.data, dtype=np.uint8)
        required_size = int(message.height) * row_step
        if row_step < packed_width or raw.size < required_size:
            raise ValueError(
                f"invalid image buffer: {raw.size} bytes for "
                f"{message.width}x{message.height} {message.encoding}"
            )
        frame = raw[:required_size].reshape(int(message.height), row_step)
        frame = frame[:, :packed_width].reshape(
            int(message.height), int(message.width), channels
        )
        if encoding == "rgb8":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        if encoding in ("bgra8", "8uc4"):
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if channels == 1:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame.copy()

    @staticmethod
    def _letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        source_height, source_width = frame.shape[:2]
        scale = min(width / max(1, source_width), height / max(1, source_height))
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        resized = cv2.resize(
            frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA
        )
        x_offset = (width - resized_width) // 2
        y_offset = (height - resized_height) // 2
        canvas[
            y_offset:y_offset + resized_height,
            x_offset:x_offset + resized_width,
        ] = resized
        return canvas

    def _panel(self, label: str, width: int, height: int) -> np.ndarray:
        panel = np.full((height + 42, width, 3), 24, dtype=np.uint8)
        frame = self._frames.get(label)
        if frame is None:
            cv2.putText(
                panel,
                "Waiting for image...",
                (max(12, width // 2 - 105), height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (160, 160, 160),
                2,
            )
        else:
            panel[42:] = self._letterbox(frame, width, height)
        title_colour = {
            "CAMERA": (230, 230, 230),
            "ROBOREFER DEPTH MASK": (0, 255, 255),
            "ROBOREFER POINT + BBOX": (255, 170, 0),
            "ROBOREFER -> DEPTH -> 3D": (80, 255, 160),
            "RGB ONLY vs RGB-D: NEAR/FAR": (80, 255, 160),
            "RELATIVE POSITION: RGB-D 3D": (80, 255, 160),
        }.get(label, (230, 230, 230))
        cv2.putText(
            panel,
            label,
            (14, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            title_colour,
            2,
        )
        return panel

    def _render(self) -> None:
        for label, message in tuple(self._messages.items()):
            try:
                self._frames[label] = self._image_to_bgr(message)
            except Exception as exc:
                self.get_logger().warning(f"Cannot decode {label} image: {exc}")
            finally:
                # Do not retain large ROS messages after conversion.
                self._messages.pop(label, None)

        width = self._panel_width
        height = self._panel_height
        upper = np.hstack([
            self._panel("CAMERA", width, height),
            self._panel(self._top_right_label, width, height),
        ])
        lower = np.hstack([
            self._panel(self._bottom_left_label, width, height),
            self._panel(self._bottom_right_label, width, height),
        ])
        footer = np.full((42, width * 2, 3), 18, dtype=np.uint8)
        if self._comparison_mode == "near_far_ablation":
            footer_text = (
                "Controlled ablation: identical RGB + prompt | orange=RGB-only "
                "| green=RGB-D | metric depth referee"
            )
        elif self._comparison_mode == "relative_position":
            footer_text = (
                "B relative to A | Camera: left/right, above/below, depth | "
                "base_link: +X forward, +Y left, +Z up"
            )
        elif self._comparison_mode == "five_step_reasoning":
            footer_text = (
                "One prompt -> locked point -> 5 post-inference RGB-D checks | "
                "UR3 target released only when all 5 pass"
            )
        else:
            footer_text = (
                "RoboRefer selects A/B | Depth builds regions | Camera + TF "
                "measure physical width/height in metres"
            )
        cv2.putText(
            footer,
            footer_text,
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (230, 230, 230),
            1,
        )
        cv2.imshow(self._window_name, np.vstack((upper, lower, footer)))
        cv2.waitKey(1)

    def close(self) -> None:
        cv2.destroyWindow(self._window_name)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MultimodalImageViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
