import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Imu
from tf_transformations import euler_from_quaternion
import math, time
import pyproj

# NOTE: Need to put the following class in another file, just wanted to test this first
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        """Computes the needed speeds for the Robot

        Args:
            error: The amount of error from the goal to the current position
            dt: The change of time
        """
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output


class MotionController(Node):
    """Class to help with autonomous motion control"""

    def __init__(self):
        super().__init__("MotionController")

        # Publishers
        self.twist_publisher = self.create_publisher(Twist, "cmd_vel", 10)
        # Subscribers
        self.gnss_subscription = self.create_subscription(
            NavSatFix, "gnss/fix", self.gnss_callback, 10
        )
        self.imu_subscription = self.create_subscription(
            Imu, "/imu", self.imu_callback, 10
        )

        # Proj converter for UTM (zone = 17 for Eastern America)
        # NOTE: May need to come back to this to make it more dynamic
        # NOTE: Using UTM for now since we don't have to deal with a global coordinate system
        self.proj_utm = pyproj.Proj(proj="utm", zone=17, ellps="WGS84", south=False)

        # Origin (set dynamically on first GNSS message)
        self.ref_x = None
        self.ref_y = None
        self.imu_reading = None

        # Init PID Controller Classes
        self.linear_pid = PIDController(0.1, 0.05, 0.1)
        self.angular_pid = PIDController(0.8, 0.15, 0.1)

        # Hardcoded goal waypoints (modify as needed)
        self.goal_x = 3.0
        self.goal_y = -2.0

        self.get_logger().info(f"Moving towards ({self.goal_x}, {self.goal_y})\n")

        self.counter = 40

        # Current pose (initialize to None until received)
        self.current_x = None
        self.current_y = None

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

    def gnss_callback(self, msg):
        """Stores the first GNSS fix as (0,0) and converts subsequent coordinates."""
        ## NOTE: Need to reset the origin whenever you go to the previous waypoint (can do this at the end of the control loop)
        utm_x, utm_y = self.proj_utm(msg.longitude, msg.latitude)  # UTM (E, N)

        if self.ref_x is None and self.ref_y is None:
            self.ref_x, self.ref_y = utm_x, utm_y  # Set first GNSS reading as origin
            self.get_logger().info(f"Set origin at: ({self.ref_x}, {self.ref_y})")

        # Convert current position relative to the origin
        # Flip coordinate system around x axis
        self.current_x = -(utm_x - self.ref_x)
        self.current_y = -(utm_y - self.ref_y)
        self.get_logger().info(f"Set current at: ({self.current_x}, {self.current_y})")

    def imu_callback(self, msg):
        """Stores the angular value from the IMU"""
        # Extract quaternion from IMU message
        quaternion = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )

        # Convert quaternion to roll, pitch, yaw
        _, _, yaw = euler_from_quaternion(quaternion)
        # Need to convert yaw from left handed NED to right handed ENU
        self.imu_reading = math.pi / 2 + yaw

    def control_loop(self):
        """Compute and publish velocity commands."""
        ## NOTE: Can break up the path  into  multiple waypoints so the robot can have time to adjust its angle correctly around the obstacle
        if (self.ref_x is None and self.ref_y is None) or self.imu_reading is None:
            return  # Wait for first gnss update

        # Compute errors
        dx = self.goal_x - self.current_x
        dy = self.goal_y - self.current_y
        distance_error = math.sqrt(dx**2 + dy**2)
        target_heading = math.atan2(dy, dx)
        heading_error = target_heading - self.imu_reading  # May need this to be 0
        heading_error = math.atan2(
            math.sin(heading_error), math.cos(heading_error)
        )  # Normalize

        # Compute control outputs
        linear_speed = self.linear_pid.compute(distance_error, 0.1)
        angular_speed = self.angular_pid.compute(heading_error, 0.1)

        # Publish Twist message
        twist_msg = Twist()
        twist_msg.linear.x = max(min(linear_speed, 0.5), -0.5)  # Limit speed
        twist_msg.angular.z = max(min(angular_speed, 1.5), -1.5)

        self.twist_publisher.publish(twist_msg)

        # self.get_logger().info(f'Moving towards goal: ({self.goal_x}, {self.goal_y})')
        # Stop if close to goal
        if distance_error < 0.15:
            self.get_logger().info("Goal reached!")
            # twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0  # Stop!
            self.twist_publisher.publish(twist_msg)
            self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    motion_controller_cls = MotionController()
    rclpy.spin(motion_controller_cls)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
