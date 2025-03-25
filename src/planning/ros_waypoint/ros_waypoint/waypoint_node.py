import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Bool
from collections import deque


class WayPointController(Node):
    """Class to send waypoint signals to the motion controller"""

    def __init__(self):
        super().__init__("WayPointController")

        # Publisher
        self.waypoint_publisher = self.create_publisher(PoseArray, "/waypoints", 10)

        # Subscriber
        self.signal_subscription = self.create_subscription(
            Bool, "/waypoint_signal", self.signal_callback, 10
        )
        self.mc_subscription = self.create_subscription(
            Bool, "/controller_signal", self.mc_callback, 10
        )

        # Boolean needed
        self.mc_bool = True

        # Waypoint queue (example)
        self.waypoint_queue = deque(
            [
                [(0, 0), (-1.1, 0), (-2.0, -0.25)],  # Planting Area 1
                [(0, 0), (-2.75, -2), (-3.5, -4.25), (-4, -6.5)],  # Planting Area 2
                [(0, 0), (1.0, 3.0), (4.0, 9.75), (10.0, 12.0)],  # planting Area 3
            ]
        )

    def mc_callback(self, msg):
        """Callback to check if the motion controller is ready for waypoints"""
        self.mc_bool = msg.data
        self.get_logger().info("We trying to send the waypoints")

    def signal_callback(self, msg):
        """Get signal to send or not send waypoints"""
        if msg.data and self.mc_bool:
            # Send the waypoints
            self.send_waypoint()
            self.mc_bool = False

    def send_waypoint(self):
        """Send waypoints to the motion controller"""
        if self.waypoint_queue:
            waypoints = self.waypoint_queue.popleft()  # FIFO order
            pose_array = PoseArray()
            pose_array.header.stamp = self.get_clock().now().to_msg()
            pose_array.header.frame_id = "map"

            # Convert waypoints to Pose objects
            for x, y in waypoints:
                pose = Pose()
                self.get_logger().info(f"X: {x}, Y: {y}\n")
                pose.position.x = float(x)
                pose.position.y = float(y)
                pose.position.z = 0.0  # Assume 2D navigation
                pose_array.poses.append(pose)

            # Publish the PoseArray
            self.waypoint_publisher.publish(pose_array)
            self.get_logger().info(f"Published {len(waypoints)} waypoints")
        else:
            self.get_logger().info("No more waypoint arrays to publish")


def main(args=None):
    rclpy.init(args=args)
    waypoints_controller_cls = WayPointController()
    rclpy.spin(waypoints_controller_cls)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
