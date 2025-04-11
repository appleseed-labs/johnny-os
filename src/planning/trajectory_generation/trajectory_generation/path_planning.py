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

    def __init__(self):
        # initialize variables
        self.occupancy_grid
        self.goal_pose

        # subscribe to occupancy grid
        self.occupancy_grid_subscriber = self.create_subscription(
            OccupancyGrid,
            "traj/occupancy_grid",
            self.occupancy_grid,
            1
        )

        # subscribe to find the next goal position
        self.goal_position_subscriber = self.create_subscription(
            Pose,
            "traj/goal_pose",
            self.goal_pose,
            1
        )

        # publish trajectory to "traj/local_frame_traj"
        self.traj_publisher = self.create_publisher(Path, "traj/local_frame_traj", 1)

        # ROS2 timer for stepping
        self.timer = self.create_timer(1.0 / self.RATE, self.step)

        super().__init__('obstacle_detector')
        self.get_logger().info('INITIALIZED.')

    def heuristic(self, a, b):
        # Manhattan distance heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def a_star_algorithm(self, occupancy_grid, start, goal):
        goal_occupied = False
        # if obstacle at start position, ignore
        if(occupancy_grid[start]==1):
            occupancy_grid[start] = 0
        # if obstacle at end position, return None
        if(occupancy_grid[goal]==1):
            goal_occupied = True
            print("goal is occupied")
            return None
        
        rows, cols = occupancy_grid.shape
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
                if(goal_occupied):
                    # return path expect the goal position
                    return path[:0:-1]
                else:
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
    def dilate_grid(self, grid, dilate_width):
        dilated_kernel = np.ones((3, 3))
        np_grid = np.array(grid)
        dilated_grid = (cv2.dilate((np_grid), dilated_kernel, iteration = dilate_width))
        return dilated_grid.to_list()

    def generate_traj(self):
        # get grid and info from occupancy_grid message
        resolution = self.occupancy_grid.info.resolution
        start_x = self.occupancy_grid.info.width.info.origin.position.x 
        start_y = self.occupancy_grid.info.width.info.origin.position.y
        goal_x = self.goal_pose.x
        goal_y = self.goal_pose.y

        grid = np.array(self.occupancy_grid.data)
        grid = grid.reshape((self.occupancy_grid.info.width, self.occupancy_grid.info.height))
        grid = self.dilate_grid(grid, 1)

        # generate path
        path = self.a_star_algorithm(grid, (start_x, start_y), (goal_x, goal_y))

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
            # set pah points at the center of the grids
            real_x = (x + 0.5) * resolution
            real_y = (y + 0.5) * resolution
            pose.pose.position.x = float(real_x)
            pose.pose.position.y = float(real_y)
            path_msg.poses.append(pose)

        return path_msg

    def step(self):
        path_msg = self.generate_traj()
        if(path_msg != None):
            self.traj_publisher.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    path_planner = pathPlanner()
    rclpy.spin(path_planner)
    rclpy.shutdown()

if __name__ == "__main__":
    main()