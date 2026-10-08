#!/usr/bin/python3
"""Show the live left-oblique tabletop and UR3 wrist RGB streams."""

from __future__ import annotations

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from capture_ros_rgbd import decode_rgb


class TwoCameraViewer(Node):
    def __init__(self) -> None:
        super().__init__("ur3_two_camera_viewer")
        self.frames = {}
        self.windows = (
            ("Ban tu ben trai UR3 (goc xien)", "/left_oblique_table_camera/image", 1200, 30),
            ("Co tay UR3 (D435i RGB)", "/wrist_camera/image", 1200, 550),
        )
        for label, topic, x, y in self.windows:
            cv2.namedWindow(label, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(label, 640, 480)
            cv2.moveWindow(label, x, y)
            self.create_subscription(Image, topic,
                                     lambda message, key=label: self.on_image(key, message),
                                     qos_profile_sensor_data)
            self.get_logger().info(f"Viewing {topic}")

    def on_image(self, label: str, message: Image) -> None:
        self.frames[label] = decode_rgb(message)

    def show(self) -> None:
        for label, *_ in self.windows:
            frame = self.frames.get(label)
            if frame is not None:
                cv2.imshow(label, frame)


def main() -> None:
    rclpy.init()
    viewer = TwoCameraViewer()
    try:
        while rclpy.ok():
            rclpy.spin_once(viewer, timeout_sec=0.02)
            viewer.show()
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        viewer.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
