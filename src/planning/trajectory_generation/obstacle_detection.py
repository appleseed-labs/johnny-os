import numpy as np
import ros2_numpy as rnp
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64


# This node should subscribe to the point cloud data from lidar,
# detect the obstacles based on the data
# and publish the occupancy grid for trajectory generation

# TODO LIST:
# 1. define a msg type for 1d array and width and height 
# refer to https://robotics.stackexchange.com/questions/97420/send-a-2d-array-through-topics-in-ros2
# 2. remove noise and detect the ground and obstacles from the point cloud
# library that might work: https://pcl.readthedocs.io/projects/tutorials/en/latest/walkthrough.html
# 3. process the data to form an occupancy grid
# 4. test with data from EcoSim 

class obstacleDetector(Node):
    RATE = 100

    def __init__(self):
        # initialize variables
        self.point_cloud

        # subscribe steward for point cloud
        self.point_cloud_subscriber = self.create_subscription(
            PointCloud2,
            "steward",
            self.point_cloud,
            1
        )

        # publish occupancy_grid to "traj/occupancy_grid"
        self.occupancy_grid_publisher = self.create_publisher(Float64, "traj/occupancy_grid", 1)

        # ROS2 timer for stepping
        self.timer = self.create_timer(1.0 / self.RATE, self.step)

        # super().__init__('obstacle_detector')
        # self.get_logger().info('INITIALIZED.')

    def processPointCloud(self):
        self.np_point_cloud = rnp.numpify(self.point_cloud_subscriber)

        return np.array([[0]])
    
    def step(self):
        # update velocity
        occupancy_grid = self.processPointCloud()

        occupancy_grid_flattened = occupancy_grid.flat()

        # publish velocity
        float_64_velocity = Float64()
        float_64_velocity.data = float(self.buggy_vel)
        self.velocity_publisher.publish(occupancy_grid_flattened)

def main(args=None):
    rclpy.init(args=args)
    obstacle_detector = obstacleDetector()
    rclpy.spin(obstacle_detector)
    rclpy.shutdown()

if __name__ == "__main__":
    main()