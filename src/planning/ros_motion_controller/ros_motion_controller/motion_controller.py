import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Imu
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Bool
import math, time
import numpy as np
import pyproj


class MotionController(Node):
    """Class to help with autonomous motion control using Pure Pursuit"""

    def __init__(self):
        super().__init__("MotionController")

        # Publishers
        self.twist_publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self.controller_signal_publisher = self.create_publisher(
            Bool, "/controller_signal", 10
        )

        # Subscribers
        self.gnss_subscription = self.create_subscription(
            NavSatFix, "gnss/fix", self.gnss_callback, 10
        )
        self.imu_subscription = self.create_subscription(
            Imu, "/imu", self.imu_callback, 10
        )
        self.waypoints_subscription = self.create_subscription(
            PoseArray, "/waypoints", self.waypoints_callback, 10
        )

        # NOTE: May need to make this dynamic
        # Proj converter for UTM (zone = 17 for Eastern America)
        self.proj_utm = pyproj.Proj(proj="utm", zone=17, ellps="WGS84", south=False)

        # Origin (set dynamically on first GNSS message)
        self.ref_x = None
        self.ref_y = None
        self.imu_reading = None

        self.waypoints = []

        # Current pose (initialize to None until received)
        self.current_x = None
        self.current_y = None

        # Counter to help us track the amount of lookahead_points not found
        self.lookahead_not_found_counter = 0
        # Initilize timer for control loop
        self.timer = None
        # NOTE: This is for testing purposes
        # self.controller_signal_publisher.publish(self.create_bool_message(True))

    def waypoints_callback(self, msg):
        """Callback to get waypoints from path planning"""
        poses = msg.poses
        # Get the needed x, y tuples for waypoints
        for pose in poses:
            loc_tuple = (pose.position.x, pose.position.y)
            self.waypoints.append(loc_tuple)

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

    def gnss_callback(self, msg):
        """Stores the first GNSS fix as (0,0) and converts subsequent coordinates."""
        utm_x, utm_y = self.proj_utm(msg.longitude, msg.latitude)  # UTM (E, N)

        if self.ref_x is None and self.ref_y is None:
            self.ref_x, self.ref_y = utm_x, utm_y  # Set first GNSS reading as origin
            self.get_logger().info(f"Set origin at: ({self.ref_x}, {self.ref_y})")

        # If we are facing north/south then adjust the gnss positioning accordingly
        # if self.imu_reading == None:
        #     return
        # elif self.imu_reading >= 0 and self.imu_reading <= math.pi:
        #     sign = -1 # Facing North
        # else:
        #     sign = 1 # Facing South

        # Convert current position relative to the origin
        self.current_x = (utm_x - self.ref_x) * -1
        self.current_y = (utm_y - self.ref_y) * -1
        self.get_logger().info(f"Set current at: ({self.current_x}, {self.current_y})")

    def imu_callback(self, msg):
        """Stores the angular value from the IMU"""
        if msg.orientation.w <= 0:
            return

        # Extract quaternion from IMU message
        quaternion = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )

        self.get_logger().info(f"{quaternion}")

        # Convert quaternion to roll, pitch, yaw
        _, _, yaw = R.from_quat(quaternion).as_euler("xyz")
        # Convert yaw from left handed NED to right handed ENU
        self.imu_reading = math.pi / 2 + yaw
        # self.get_logger().info(f'Set current at: ({self.imu_reading})')

    def find_lookahead_point(self, lookahead_distance):
        """Find a point on the path at lookahead distance ahead of the robot.

        Returns:
            Lookahead_point: Point that the robot should move towards
        """
        if self.current_x is None or self.current_y is None:
            return None

        for i in range(len(self.waypoints) - 1):
            p1 = np.array(self.waypoints[i])
            p2 = np.array(self.waypoints[i + 1])

            # Vector between waypoints
            path_vector = p2 - p1
            path_length = np.linalg.norm(path_vector)  # Length of the path segment

            # Vector from robot to the first waypoint
            to_p1 = p1 - np.array([self.current_x, self.current_y])

            # Quadratic coefficients for solving for the lookahead distance
            a = np.dot(path_vector, path_vector)
            b = 2 * np.dot(path_vector, to_p1)
            c = np.dot(to_p1, to_p1) - np.square(lookahead_distance)

            # Solve the quadratic equation: at^2 + bt + c = 0
            t = np.roots([a, b, c])

            # Choose the valid solution (if real and within the path segment)
            t = np.max(t)  # We want the largest solution in case of two roots

            if 0 <= t <= 1:  # Ensure the solution is within the path segment [0, 1]
                lookahead_point = p1 + t * path_vector  # Calculate the lookahead point
                return lookahead_point  # Return the lookahead point

        return None  # No valid lookahead point found

    def find_curvature(
        self, lookahead_point, robot_heading, robot_loc, lookahead_distance
    ):
        """Function used to find curvature to guide robot towards path

        Args:
            lookahead_point: Array of lookahead points (in [x, y])

        Returns:
            Curvature
        """
        # Unpack from the args
        x_pos = robot_loc[0]
        y_pos = robot_loc[1]

        lookahead_x = lookahead_point[0]
        lookahead_y = lookahead_point[1]

        a = -np.tan(
            robot_heading
        )  # The slope of the line at the current robot orientation
        b = 1  # The coefficient for the y-axis in the line equation
        c = x_pos * np.tan(self.imu_reading) - y_pos  # The line offset

        # Distance (N) from the robot's current position to the closest point on the path
        N = abs(a * lookahead_x + b * lookahead_y + c) / np.sqrt(
            np.square(a) + np.square(b)
        )

        # Determine the sign of the curvature (left or right turn)
        sign_pos = (
            lookahead_point - robot_loc
        )  # Vector from robot position to the pursuit point
        side = sign_pos[0] * np.sin(robot_heading) - sign_pos[1] * np.cos(robot_heading)
        side = np.sign(
            side
        )  # Sign is +1 or -1 depending on the direction (left or right)

        # The curvature is inversely proportional to the distance N and also depends on the path orientation
        curvature = 2 * N / np.square(lookahead_distance) * side * -1

        return curvature

    def stop(self):
        """Stop the robot and reset values"""
        # Stop if close to goal
        self.get_logger().info("Goal reached!")
        self.get_logger().info(f"Currently at: ({self.current_x}, {self.current_y})")
        self.reset()
        # Send signal to let waypoint node that we can accept more waypoints
        self.controller_signal_publisher.publish(self.create_bool_message(True))

    def reset(self):
        """Resets key values and the robot"""
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.twist_publisher.publish(twist_msg)
        self.waypoints = []
        self.timer.cancel()
        # Resets the coordinate origins
        # self.ref_x = None
        # self.ref_y = None

    def create_bool_message(self, data):
        """Creates a bool message to send based on data"""
        msg_bool = Bool()
        msg_bool.data = data
        return msg_bool

    def find_ang_lin_speed(
        self, lookahead_point, remaining_distance, adaptive_lookahead
    ):
        """Find the angular and linear speed

        Args:
            lookahead_point: Point closest to the curve we want to get to
            remaining_distance: The remaining distance to the end goal
            adaptive_lookahead: Lookahead Distance

        Returns:
            Angular and Linear speeds
        """
        # Compute curvature
        loc_cur = np.array([self.current_x, self.current_y])
        curvature = self.find_curvature(
            lookahead_point, self.imu_reading, loc_cur, adaptive_lookahead
        )
        angular_speed = math.atan(curvature)

        # Compute linear speed using proportional controller
        # NOTE: Let's adjust lin. speed first
        linear_speed = min(0.5 * remaining_distance, 1.0)  # Cap max speed to be 1m/s
        if remaining_distance < 0.75:  # If within 0.75m of goal, slow down
            linear_speed = min(linear_speed, 0.4)

        return angular_speed, linear_speed

    def find_adaptive_lookahead(self):
        """Dynamically calculate lookahead distance based on the closest waypoint segment."""
        if self.current_x is None or self.current_y is None:
            return 1.5  # Default lookahead if no position data

        min_dist = float("inf")
        best_lookahead = 1.5  # Default lookahead

        for i in range(len(self.waypoints) - 1):
            # Get the current and next waypoint
            x1, y1 = self.waypoints[i]
            x2, y2 = self.waypoints[i + 1]

            # Project the robot's position onto the path segment
            segment_vec = np.array([x2 - x1, y2 - y1])
            robot_vec = np.array([self.current_x - x1, self.current_y - y1])
            segment_length = np.linalg.norm(segment_vec)

            if segment_length == 0:
                continue  # Skip if waypoints are identical

            # Projection formula to find t (parametric representation)
            t = np.dot(robot_vec, segment_vec) / (segment_length**2)
            t = max(0, min(1, t))  # Clamp t between 0 and 1

            # Closest point on the segment to the robot
            closest_x = x1 + t * (x2 - x1)
            closest_y = y1 + t * (y2 - y1)

            # Distance from robot to closest point
            distance = np.linalg.norm(
                [self.current_x - closest_x, self.current_y - closest_y]
            )

            # Adaptive lookahead logic
            if distance < min_dist:
                min_dist = distance
                # self.get_logger().info(f"Distance: {distance}\n")
                # self.get_logger().info(f"Formula: {max(0.1, min(2.0, distance))}\n")
                best_lookahead = max(0.4, min(1.5, distance))  # Adjust dynamically

        # self.get_logger().info(f"Adaptive lookahead distance: {best_lookahead}")
        return best_lookahead

    def control_loop(self):
        """Compute and publish velocity commands using Pure Pursuit."""
        if self.current_x is None or self.current_y is None or self.imu_reading is None:
            return

        # Calculate the remaining distance
        remaining_distance = math.sqrt(
            (self.waypoints[-1][0] - self.current_x) ** 2
            + (self.waypoints[-1][1] - self.current_y) ** 2
        )

        # Stop if close to goal
        if remaining_distance < 0.25:
            self.get_logger().info("We are close to the waypoint!")
            # Stop the robot
            self.stop()
            return

        # Get the lookhead distance adaptively
        adaptive_lookahead = self.find_adaptive_lookahead()
        if adaptive_lookahead is None:
            self.get_logger().warn("Couldn't get a adaptive lookahead distance!")
            return

        # Find lookahead point
        lookahead_point = self.find_lookahead_point(adaptive_lookahead)
        # No lookahead point so stop
        if lookahead_point is None:
            self.get_logger().warn("No valid lookahead point found!")
            self.lookahead_not_found_counter += 1

            # If we have seen the no valid lookahead point then lets stop
            if self.lookahead_not_found_counter == 20:
                # Stop the robot
                self.stop()
                self.lookahead_not_found_counter = 0
            return

        # Compute angular and linear speed
        angular_speed, linear_speed = self.find_ang_lin_speed(
            lookahead_point, remaining_distance, adaptive_lookahead
        )

        # Publish Twist command
        twist_msg = Twist()
        twist_msg.linear.x = linear_speed
        twist_msg.angular.z = max(min(angular_speed, 1.5), -1.5)

        # self.get_logger().info(f'Lin speed and ang speed: ({linear_speed}, {angular_speed})')

        self.twist_publisher.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)
    motion_controller_cls = MotionController()
    rclpy.spin(motion_controller_cls)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
