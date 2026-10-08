#!/usr/bin/env python3
"""Capture one RGB frame from each Gazebo camera and write a labeled montage."""

import argparse
import json
import time
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class TwoCameraCapture(Node):
    def __init__(self):
        super().__init__('two_camera_demo_capture')
        self.bridge = CvBridge()
        self.frames = {}
        self.metadata = {}
        self.create_subscription(Image, '/wrist_camera/color/image_raw',
                                 lambda msg: self._receive('wrist', msg), 10)
        self.create_subscription(Image, '/top_table_camera/color/image_raw',
                                 lambda msg: self._receive('top_table', msg), 10)

    def _receive(self, name, message):
        if name in self.frames:
            return
        self.frames[name] = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        self.metadata[name] = {
            'frame_id': message.header.frame_id,
            'stamp_sec': message.header.stamp.sec,
            'stamp_nanosec': message.header.stamp.nanosec,
            'width': message.width,
            'height': message.height,
            'encoding': message.encoding,
        }


def label(image, text):
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 38), (25, 25, 25), -1)
    cv2.putText(result, text, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, .72,
                (255, 255, 255), 2, cv2.LINE_AA)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=20.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = TwoCameraCapture()
    deadline = time.monotonic() + args.timeout
    while len(node.frames) < 2 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=.5)
    if len(node.frames) != 2:
        node.destroy_node()
        rclpy.shutdown()
        raise RuntimeError(f'Timed out; received {sorted(node.frames)}')

    wrist = label(node.frames['wrist'], 'Wrist RGB-D camera')
    top = label(node.frames['top_table'], 'Top-table RGB-D camera')
    cv2.imwrite(str(args.output_dir / 'wrist_rgb.png'), wrist)
    cv2.imwrite(str(args.output_dir / 'top_table_rgb.png'), top)
    cv2.imwrite(str(args.output_dir / 'two_camera_rgb.png'), cv2.hconcat([wrist, top]))
    (args.output_dir / 'capture_metadata.json').write_text(json.dumps({
        'status': 'CAPTURED', 'topics': {
            'wrist': '/wrist_camera/color/image_raw',
            'top_table': '/top_table_camera/color/image_raw',
        }, 'frames': node.metadata,
    }, indent=2) + '\n')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
