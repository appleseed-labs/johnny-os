#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from johnny_msgs.msg import SystemwideStatus
from supabase import create_client, Client
import atexit


class SupabaseInterface(Node):
    """
    ROS2 node that publishes robot status to Supabase.

    Subscribes to /status topic and updates the stewards table in Supabase
    at a configured rate.
    """

    # Mapping from SystemwideStatus level to database status text
    STATUS_MAPPING = {
        SystemwideStatus.HEALTHY: "HEALTHY",
        SystemwideStatus.WARN: "WARN",
        SystemwideStatus.TELEOP_ONLY: "TELEOP_ONLY",
        SystemwideStatus.OUT_OF_SERVICE: "OUT_OF_SERVICE",
    }

    def __init__(self):
        super().__init__("supabase_interface")

        # Declare parameters
        self.declare_parameter("robot_name", "Johnny")
        self.declare_parameter("update_rate", 1.0)  # Hz
        self.declare_parameter("supabase_url", "")
        self.declare_parameter("supabase_key", "")

        # Get parameters
        self.robot_name = self.get_parameter("robot_name").value
        update_rate = self.get_parameter("update_rate").value
        supabase_url = self.get_parameter("supabase_url").value
        supabase_key = self.get_parameter("supabase_key").value

        # Validate parameters
        if not supabase_url or not supabase_key:
            self.get_logger().error(
                "supabase_url and supabase_key parameters are required!"
            )
            raise ValueError("Missing required Supabase credentials")

        # Initialize Supabase client
        try:
            self.supabase: Client = create_client(supabase_url, supabase_key)
            self.get_logger().info(f"Connected to Supabase at {supabase_url}")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to Supabase: {e}")
            raise

        # Verify robot exists in database
        self._verify_robot_exists()

        # Initialize status
        self.current_status = None
        self.last_status_text = "OUT_OF_SERVICE"

        # Subscribe to status topic
        self.status_subscription = self.create_subscription(
            SystemwideStatus, "/status", self.status_callback, 10
        )

        # Create timer for periodic updates
        update_period = 1.0 / update_rate
        self.update_timer = self.create_timer(update_period, self.update_callback)

        # Register shutdown handler to set status to OFFLINE
        atexit.register(self._shutdown_handler)

        self.get_logger().info(
            f"Supabase interface initialized for robot '{self.robot_name}' "
            f"at {update_rate} Hz"
        )

    def _verify_robot_exists(self):
        """Verify that the robot exists in the stewards table."""
        try:
            response = (
                self.supabase.table("stewards")
                .select("id, name")
                .eq("name", self.robot_name)
                .execute()
            )

            if not response.data or len(response.data) == 0:
                error_msg = (
                    f"Robot '{self.robot_name}' not found in stewards table. "
                    f"Please add the robot to the database before running this node."
                )
                self.get_logger().error(error_msg)
                raise ValueError(error_msg)

            self.get_logger().info(f"Found robot '{self.robot_name}' in stewards table")

        except Exception as e:
            self.get_logger().error(f"Failed to verify robot existence: {e}")
            raise

    def status_callback(self, msg: SystemwideStatus):
        """Callback for /status topic."""
        self.current_status = msg
        self.last_status_text = self.STATUS_MAPPING.get(msg.level, "OUT_OF_SERVICE")
        self.get_logger().debug(
            f"Received status update: level={msg.level}, "
            f"text={self.last_status_text}"
        )

    def update_callback(self):
        """Periodic callback to update Supabase."""
        try:
            # Update the stewards table
            response = (
                self.supabase.table("stewards")
                .update({"status": self.last_status_text})
                .eq("name", self.robot_name)
                .execute()
            )

            if response.data:
                self.get_logger().debug(
                    f"Updated status to '{self.last_status_text}' for "
                    f"robot '{self.robot_name}'"
                )
            else:
                self.get_logger().warn(
                    f"Failed to update status for robot '{self.robot_name}'. "
                    f"This is likely due to Row Level Security policies. "
                    f"Ensure you're using the service_role key or disable RLS on the stewards table."
                )

        except Exception as e:
            self.get_logger().error(f"Error updating Supabase: {e}")

    def _shutdown_handler(self):
        """Set status to OFFLINE when node is shut down."""
        try:
            self.get_logger().info(
                f"Shutting down - setting status to OFFLINE for "
                f"robot '{self.robot_name}'"
            )
            self.supabase.table("stewards").update({"status": "OFFLINE"}).eq(
                "name", self.robot_name
            ).execute()
        except Exception as e:
            self.get_logger().error(
                f"Error setting OFFLINE status during shutdown: {e}"
            )


def main(args=None):
    rclpy.init(args=args)

    try:
        node = SupabaseInterface()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
