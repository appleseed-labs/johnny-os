#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import torch
import cv2
import numpy as np


class YoloDetector(Node):
    def __init__(self):
        super().__init__("detect_node")
        self.bridge = CvBridge()
        self.model = YOLO(
            "/home/appleseed_labs/johnny-os/src/perception/yolov8_detecter/yolov8_detecter/best.pt"
        )
        self.subscription = self.create_subscription(
            Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 10
        )
        self.publisher = self.create_publisher(Bool, "/perception/person_nearby", 10)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self.model(frame)[0]

        class_ids = results.boxes.cls.tolist() if results.boxes is not None else []
        class_names = results.names

        person_detected = any(class_names[int(cls)] == "person" for cls in class_ids)

        self.publisher.publish(Bool(data=person_detected))
        self.get_logger().info(
            f"Detected classes: {[class_names[int(cls)] for cls in class_ids]}"
        )
        self.get_logger().info(f"Person nearby: {person_detected}")

        # Draw bounding boxes and labels
        if results.boxes is not None:
            for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label = class_names[int(cls_id)]
                color = (0, 255, 0) if label == "person" else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                )

        # Show the frame in a window
        cv2.imshow("YOLO Detections", frame)
        cv2.waitKey(1)  # Needed to update the OpenCV window


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()
