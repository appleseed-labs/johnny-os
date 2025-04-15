import numpy as np
import ros2_numpy as rnp
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64
from nav_msgs.msg import OccupancyGrid

# This node should subscribe to the point cloud data from lidar,
# detect the obstacles based on the data
# and publish the occupancy grid for trajectory generation

# TODO LIST:
# 1. test individual node with fake data
# 2. test in EcoSim, adjust parameters and finding origin
# ignore for now: remove noise
# library that might work: https://pcl.readthedocs.io/projects/tutorials/en/latest/walkthrough.html
# ignore for now: better ground detection


class obstacleDetector(Node):
    # publish rate
    RATE = 100
    # grid resolution (m/cell)
    GRID_RES = 0.1
    
    processed_pc = None
    map_load_time = None

    def __init__(self):
        super().__init__('obstacle_detector')

        # subscribe steward for point cloud
        self.point_cloud_subscriber = self.create_subscription(
            PointCloud2,
            "steward",
            self.process_point_cloud,
            1
        )

        # publish occupancy_grid to "traj/occupancy_grid"
        self.occupancy_grid_publisher = self.create_publisher(OccupancyGrid, "traj/occupancy_grid", 1)

        # ROS2 timer for stepping
        self.timer = self.create_timer(1.0 / self.RATE, self.step)

        self.get_logger().info('INITIALIZED.')

    def process_point_cloud(self, msg: PointCloud2):
        # transformation parameters (need to the changed to real values)
        theta = 1
        x_offset = 1
        y_offset = 1

        np_point_cloud = rnp.numpify(msg)
        self.map_load_time = self.get_clock().now().to_msg()
        num_points = np_point_cloud.size

        # rotation around y
        R = np.array([
            [np.cos(theta),  0, np.sin(theta)],
            [0,              1, 0],
            [-np.sin(theta), 0, np.cos(theta)]])
        # Translation
        T = np.array([x_offset, y_offset, 0])

        # go through all point clouds
        for i in range(num_points):
            np_point_cloud[i] = R * np_point_cloud[i] + T
        
        # filter by z > 0.5
        np_point_cloud = np_point_cloud[np_point_cloud[:, 2] > 0.5]

        self.processed_pc = np_point_cloud

    def create_occupancy_grid(self, processed_pc):
        max_x = max(processed_pc[:,0])
        min_x = min(processed_pc[:,0])
        max_y = max(processed_pc[:,1])
        min_y = min(processed_pc[:,1])
        grid_min_x = int(min_x/self.GRID_RES) * self.GRID_RES
        grid_min_y = int(min_y/self.GRID_RES) * self.GRID_RES

        width = int((max_x - grid_min_x)/self.GRID_RES) + 1
        height = int((max_y - grid_min_y)/self.GRID_RES) + 1

        grid = np.zeros((height, width), dtype = np.int8)

        for (x, y, z) in processed_pc:
            grid_x = int((x - grid_min_x)/self.GRID_RES)
            grid_y = int((y - grid_min_y)/self.GRID_RES)
            grid[grid_y, grid_x] = 1
        
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "robot_center_grid_frame"
        msg.info.map_load_time = self.map_load_time
        msg.info.resolution = self.self.GRID_RES
        msg.info.width = max_x
        msg.info.height = max_y
        msg.info.origin.position.x = (0-grid_min_x)/self.GRID_RES
        msg.info.origin.position.y = (0-grid_min_y)/self.GRID_RES
        msg.data = np.flatten(grid).to_list()
        return msg
    
    def step(self):
        occupancy_grid = self.create_occupancy_grid(self.processed_pc)
        self.occupancy_grid_publisher.publish(occupancy_grid)

def main(args=None):
    rclpy.init(args=args)
    obstacle_detector = obstacleDetector()
    rclpy.spin(obstacle_detector)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
