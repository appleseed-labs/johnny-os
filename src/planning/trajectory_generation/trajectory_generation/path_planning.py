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
    
    occupancy_grid = None
    grid = None
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
            "/planning/occupancy",
            self.process_occupancy_grid,
            1
        )

        # subscribe to find the next goal position
        self.goal_position_subscriber = self.create_subscription(
            Pose,
            "/goal_pose",
            self.process_goal_pose,
            1
        )

        # publish trajectory to "planning/traj"
        self.traj_publisher = self.create_publisher(Path, "planning/traj", 1)

        # ROS2 timer for stepping
        self.timer = self.create_timer(1.0 / self.RATE, self.step)

        self.get_logger().info('INITIALIZED.')
        
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
        # return abs(a[0] - b[0]) + abs(a[1] - b[1])
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** (1/2)

    def a_star_algorithm(self, occupancy_grid, start, goal):
        # if obstacle at start position, ignore
        if(occupancy_grid[start]==1):
            occupancy_grid[goal] = 0
        # if obstacle at end position, return None
        if(occupancy_grid[goal]==1):
            print("goal is occupied")
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
                        f_score[neighbor] = tentative_g +  self.heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return None  # No path found

    def dilate_grid(self, grid, dilate_width):
        dilated_kernel = np.ones((3, 3)).astype(np.uint8)
        np_grid = grid.astype(np.uint8)
        dilated_grid = (cv2.dilate((np_grid), dilated_kernel, iterations = dilate_width))
        return dilated_grid

    def bresenham_line(self, p0, p1):
        """Yield integer coordinates on the line from p0 to p1"""
        x0, y0 = p0
        x1, y1 = p1
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                yield (x, y)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                yield (x, y)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        yield (x1, y1)

    def clean_up_path(self, occupancy_grid, path):
        if not path or len(path) < 3:
            return path

        new_path = [path[0]]  # Always start with the first node
        last_added = path[0]

        for i in range(1, len(path)):
            line = list(self.bresenham_line(last_added, path[i]))
            print(path[i],":")
            print("line",line)
            print()

            if any(occupancy_grid[p] == 1 for p in line):
                # Obstacle encountered — last safe node is path[i-1]
                new_path.append(path[i - 1])
                last_added = path[i - 1]
                print("last_added:", last_added)

        # Ensure goal is added
        if new_path[-1] != path[-1]:
            new_path.append(path[-1])

        return new_path

    def generate_traj(self):
        if self.grid is None:
            return

        self.dilated_occupancy_grid = self.dilate_grid(self.grid, 2)
        # Run A* algorithm
        raw_path = self.a_star_algorithm(self.dilated_occupancy_grid, (self.start_x, self.start_y), (self.goal_x, self.goal_y))
        self.smooth_path = None
        if(raw_path != None):
            self.smooth_path = self.clean_up_path(self.dilated_occupancy_grid, raw_path)

        # if no path found
        if(self.smooth_path == None):
            return None

        # translate path into path message
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "robot_center_frame"
        
        for (x, y) in self.smooth_path:
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
        if(path_msg != None):
            self.get_logger().info('PUBLISHING.')
            self.traj_publisher.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    path_planner = pathPlanner()
    rclpy.spin(path_planner)
    rclpy.shutdown()

if __name__ == "__main__":
    main()

