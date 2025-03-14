import numpy as np
import ros2_numpy as rnp
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from nav_msgs.msg import OccupancyGrid


# This node should subscribe to the occupancy grid data from the obstacle detector
# plan the path using A* algorithm
# and publish the resulting path as a list of positions to go to 


class pathPlanner(Node):
    # publish rate
    RATE = 100

    def __init__(self):
        # initialize variables
        self.occupancy_grid

        # subscribe steward for point cloud
        self.occupancy_grid_subscriber = self.create_subscription(
            OccupancyGrid,
            "traj/occupancy_grid",
            self.occupancy_grid,
            1
        )

        # publish trajectory to "traj/local_frame_traj"
        self.traj_publisher = self.create_publisher(OccupancyGrid, "traj/local_frame_traj", 1)

        # ROS2 timer for stepping
        self.timer = self.create_timer(1.0 / self.RATE, self.step)

        super().__init__('obstacle_detector')
        self.get_logger().info('INITIALIZED.')

    def generate_traj(self):
        traj_length = 100
        traj = np.zeros((2, traj_length))
        return traj.tolist()

    def step(self):
        traj = self.generate_traj()
        self.traj_publisher.publish(traj)

def main(args=None):
    rclpy.init(args=args)
    path_planner = pathPlanner()
    rclpy.spin(path_planner)
    rclpy.shutdown()

if __name__ == "__main__":
    main()