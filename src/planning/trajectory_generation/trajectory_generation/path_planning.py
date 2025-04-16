import numpy as np
import ros2_numpy as rnp
import rclpy
import heapq
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
import cv2 # only for dialation


# This node should subscribe to the occupancy grid data from the obstacle detector
# plan the path using A* algorithm
# and publish the resulting path as a list of positions to go to 


class pathPlanner(Node):
    # publish rate
    RATE = 100
    
    occupancy_grid = ""
    grid = [[0]]
    resolution = 0
    start_x = 0
    start_y = 0



    goal_x = 0
    goal_y = 0

    def __init__(self):
        super().__init__('path_planner')

        # subscribe to occupancy grid
        self.occupancy_grid_subscriber = self.create_subscription(
            OccupancyGrid,
            "traj/occupancy_grid",
            self.process_occupancy_grid,
            1
        )

        # subscribe to find the next goal position
        self.goal_position_subscriber = self.create_subscription(
            Pose,
            "traj/goal_pose",
            self.process_goal_pose,
            1
        )

        # publish trajectory to "traj/local_frame_traj"
        self.traj_publisher = self.create_publisher(Path, "traj/local_frame_traj", 1)

        # ROS2 timer for stepping
        self.timer = self.create_timer(1.0 / self.RATE, self.step)

        self.get_logger().info('INITIALIZED.')

    def dilate_grid(self, grid, dilate_width):
        dilated_kernel = np.ones((3, 3)).astype(np.uint8)
        np_grid = np.array(grid).astype(np.uint8)
        dilated_grid = (cv2.dilate((np_grid), dilated_kernel, iterations = dilate_width))
        return dilated_grid.tolist()
        
    def process_occupancy_grid(self, msg:OccupancyGrid):
        self.occupancy_grid = msg

        # get grid and info from occupancy_grid message
        self.resolution = self.occupancy_grid.info.resolution
        self.start_x = int(self.occupancy_grid.info.origin.position.x)
        self.start_y = int(self.occupancy_grid.info.origin.position.y)

        self.grid = np.array(self.occupancy_grid.data)
        self.grid = self.grid.reshape((self.occupancy_grid.info.width, self.occupancy_grid.info.height))
        self.grid = self.dilate_grid(self.grid, 1)
        
    def process_goal_pose(self, msg:Pose):
        self.goal_x = int(msg.position.x)
        self.goal_y = int(msg.position.y)
    
    def heuristic(self, a, b):
        # Manhattan distance heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def a_star_algorithm(self, input_occupancy_grid, start, goal):
        occupancy_grid = np.array(input_occupancy_grid)
        # if obstacle at start position, ignore
        if(occupancy_grid[start]==1):
            occupancy_grid[goal] = 0
        # if obstacle at end position, return None
        if(occupancy_grid[goal]==1):
            # print("goal is occupied")
            self.get_logger().info('goal is occupied!')
            return None
        
        cols = len(occupancy_grid)
        rows = len(occupancy_grid[0])
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_list:
            _, current = heapq.heappop(open_list)

            # Return path in correct order
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and occupancy_grid[neighbor] == 0:
                    tentative_g = g_score[current] + 1 

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return None  # No path found

    def generate_traj(self):

        # generate path
        path = self.a_star_algorithm(self.grid, (self.start_x, self.start_y), (self.goal_x, self.goal_y))

        # if no path found
        if(path == None):
            return None

        # translate path into path message
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "robot_center_frame"
        
        for (x, y) in path:
            pose = PoseStamped()
            pose.header = path_msg.header
            # set path points at the center of the grids
            real_x = (x + 0.5) * self.resolution
            real_y = (y + 0.5) * self.resolution
            pose.pose.position.x = float(real_x)
            pose.pose.position.y = float(real_y)
            path_msg.poses.append(pose)

        return path_msg

    def step(self):
        path_msg = self.generate_traj()
        # if(path_msg != None):
        self.get_logger().info('PUBLISHING.')
        self.traj_publisher.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    path_planner = pathPlanner()
    rclpy.spin(path_planner)
    rclpy.shutdown()

if __name__ == "__main__":
    main()

