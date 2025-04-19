#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_msgs.msg import String
import xml.etree.ElementTree as ET
import time
import random
import math
from threading import Lock
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation as R
from ikpy.chain import Chain
from ikpy.link import URDFLink
import tf2_ros

# ROS messages
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import MarkerArray, Marker
import tf2_geometry_msgs


class RRTStarPlanner:
    def __init__(
        self,
        start,
        goal,
        obstacle_list,
        chain: Chain,
        bounds: list[tuple],
        step_size=0.1,
        neighbor_radius=0.5,
        max_iters=5000,
        goal_bias=0.05,
    ):
        self.start = self.Node(start)
        self.goal = self.Node(goal)
        self.obstacle_list = obstacle_list
        self.bounds = bounds  # list of bounds [(min_x, max_y)...] for each joint
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        self.max_iters = max_iters
        self.goal_bias = goal_bias
        self.node_list = [self.start]
        self.chain = chain

    class Node:
        def __init__(self, config):
            self.config = config
            self.parent = None
            self.cost = 0.0

    def plan(self):
        for i in range(self.max_iters):

            # 1. Sample a random joint configuration
            if random.random() < self.goal_bias:
                random_config = self.goal.config
            else:
                random_config = self._get_random_config()

            # 2. Find the nearest neighbor in the tree
            nearest_node = self._find_nearest_neighbor(self.node_list, random_config)

            # 3. Steer towards the random configuration
            new_config = self._steer(nearest_node.config, random_config, self.step_size)
            new_node = self.Node(new_config)

            # 4. Check for collisions
            if not self._in_collision(nearest_node.config, new_config):
                # 5. Add the new node to the tree
                new_node.parent = nearest_node
                new_node.cost = nearest_node.cost + self._distance(
                    nearest_node.config, new_config
                )

                # 6. Rewire the tree
                near_nodes = self._find_near_nodes(new_node)
                for near_node in near_nodes:
                    if (
                        new_node.cost
                        + self._distance(new_node.config, near_node.config)
                        < near_node.cost
                    ):
                        if not self._in_collision(new_node.config, near_node.config):
                            near_node.parent = new_node
                            near_node.cost = new_node.cost + self._distance(
                                new_node.config, near_node.config
                            )

                self.node_list.append(new_node)

                # 7. Check if the new node is close to the goal
                if self._distance(new_node.config, self.goal.config) < self.step_size:
                    if not self._in_collision(new_node.config, self.goal.config):
                        self.goal.parent = new_node
                        self.goal.cost = new_node.cost + self._distance(
                            new_node.config, self.goal.config
                        )
                        return self._reconstruct_path()

        return None  # No path found

    def _get_random_config(self):
        config = []

        for lower, upper in self.bounds:
            config.append(random.uniform(lower, upper))

        return tuple(config)

    def _find_nearest_neighbor(self, node_list, config):
        min_dist = float("inf")
        nearest_node = None

        for node in node_list:
            dist = self._distance(node.config, config)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        return nearest_node

    def _steer(self, from_config, to_config, step_size):
        direction = tuple(t - f for f, t in zip(from_config, to_config))

        # Euclidean distance
        norm = math.sqrt(sum(d**2 for d in direction))

        if norm < step_size:
            return to_config
        else:
            # Scale the direction vector to the step size.
            # Add the direction vector to the from_config
            # to get the new configuration.
            scale = step_size / norm
            new_config = tuple(f + d * scale for f, d in zip(from_config, direction))
            return new_config

    def _in_collision(self, config1, config2):
        # Discretize the path between config1 and config2
        num_steps = 10  # Number of steps to check along the path
        for i in range(num_steps + 1):
            # Linearly interpolate between config1 and config2
            alpha = i / num_steps
            config = tuple(
                c1 * (1 - alpha) + c2 * alpha for c1, c2 in zip(config1, config2)
            )

            # Convert joint angles to end-effector and link positions using forward kinematics
            # This would typically be done using the robot's forward kinematics model
            # For now, we'll assume a function robot_fk(config) that returns link positions

            # Use forward kinematics to get robot geometry
            robot_geometry = self._get_robot_geometry(config)

            # Check for collisions with obstacles
            for obstacle in self.obstacle_list:
                if self._check_robot_obstacle_collision(robot_geometry, obstacle):
                    print("Collision detected!")
                    return True  # Collision detected

        # No collision detected
        return False

    def _get_robot_geometry(self, config):
        """
        Calculate the position and geometry of each robot link using forward kinematics.
        Returns a list of geometries representing the robot's links.
        """
        full_config = [0.0] * 10
        full_config[1:7] = config
        T = self.chain.forward_kinematics(full_config, full_kinematics=True)

        # For now, treat the joints as spheres

        geom = []
        for i, link in enumerate(self.chain.links):

            pos = T[i][:3, 3]
            rot = T[i][:3, :3]
            # Create a geometry representation of the link
            # For now, we'll just use the position and rotation
            geom.append(
                {
                    "position": pos,
                    "rotation": rot,
                    "name": link.name,
                    "radius": 0.05,  # Placeholder for link radius
                }
            )

        return geom

    def _check_robot_obstacle_collision(self, robot_geometry, obstacle):
        """
        Check for collision between the robot geometry and an obstacle.
        Returns True if there is a collision, False otherwise.
        """
        # For simplicity, assume obstacles are spheres with a position and radius

        obstacle_position = obstacle["position"]
        obstacle_radius = obstacle["radius"]

        # Check collision between each robot link and the obstacle
        for link in robot_geometry:
            link_position = link["position"]
            link_radius = link["radius"]

            # Calculate distance between link and obstacle centers
            distance = math.sqrt(
                sum((l - o) ** 2 for l, o in zip(link_position, obstacle_position))
            )

            # Collision if the distance is less than the sum of radii
            if distance < (link_radius + obstacle_radius):
                return True

        return False

    def _distance(self, config1, config2):
        return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(config1, config2)))

    def _find_near_nodes(self, new_node):
        near_nodes = []

        n_nodes = len(self.node_list)

        # As a simple heuristic, we can adjust the radius based on the number of nodes
        SCALING = 5.0
        radius = min(
            self.neighbor_radius,
            math.sqrt(math.log(n_nodes + 1) / (n_nodes + 1)) * SCALING,
        )

        for node in self.node_list:
            if self._distance(node.config, new_node.config) < radius:
                near_nodes.append(node)
        return near_nodes

    def _reconstruct_path(self):
        """
        Traverse the tree from the goal node to the start node
        and return the path as a list of configurations.
        """
        path = []

        current = self.goal

        while current is not None:
            path.append(current.config)
            current = current.parent

        # Reverse the path to get it from start to goal
        return list(reversed(path))


class ArmTrajectoryPlanner(Node):
    def __init__(self):
        super().__init__("arm_trajectory_planner")

        # QoS profile for getting the latest robot_description (transient local)
        robot_description_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.robot_description_sub = self.create_subscription(
            String,
            "/robot_description",
            self.robot_description_cb,
            qos_profile=robot_description_qos,
        )

        self.joint_command_pub = self.create_publisher(
            JointState,
            "/joint_commands",
            10,
        )

        self.clicked_point_sub = self.create_subscription(
            Point, "/ecosim/clicked_point", self.clicked_point_cb, 1
        )

        self.obstacle_vis_pub = self.create_publisher(
            MarkerArray,
            "/obstacle_visualization",
            10,
        )

        self.target_position = [
            -0.07324862480163574,
            0.32391729950904846,
            0.3905857503414154,
        ]

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.plan_sent = False
        self.plan = None
        # self.create_timer(1.0, self.try_plan)
        self.create_timer(0.1, self.execute_plan)

        # This stores the current joint angles
        # (including for inactive joints like link_base and the gripper link)
        # Only joints 1-6 are active (zero-indexed)
        self.current_joint_angles = [0.0] * 10

    def execute_plan(self):
        if self.plan is None or len(self.plan) == 0:
            return

        msg = JointState()
        msg.name = self.names
        msg.position = self.plan[0]
        self.plan = self.plan[1:]
        self.joint_command_pub.publish(msg)
        self.current_joint_angles = msg.position
        print(f"Executing plan:  {msg.position}")

    def try_plan(self):
        if self.plan_sent == True:
            return
        self.clicked_point_cb(
            Point(
                x=self.target_position[0],
                y=self.target_position[1],
                z=self.target_position[2],
            )
        )
        self.plan_sent = True

    def visualize_obstacles(self, obstacles: list[dict]):
        """
        Visualize each obstacle as a sphere in RViz
        """
        msg = MarkerArray()
        for i, obstacle in enumerate(obstacles):
            marker = Marker()
            marker.header.frame_id = obstacle["frame_id"]
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "obstacle"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = obstacle["position"][0]
            marker.pose.position.y = obstacle["position"][1]
            marker.pose.position.z = obstacle["position"][2]
            marker.scale.x = obstacle["radius"] * 2
            marker.scale.y = obstacle["radius"] * 2
            marker.scale.z = obstacle["radius"] * 2
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.5

            msg.markers.append(marker)

        self.obstacle_vis_pub.publish(msg)

    def transform_obstacle(self, obstacle: dict):
        """
        Transform the obstacle to the arm's frame
        """

        try:
            transform = self.tf_buffer.lookup_transform(
                "link_base", obstacle["frame_id"], rclpy.time.Time()
            )
        except tf2_ros.LookupException as e:
            self.get_logger().error(f"Transform not found: {e}")
            return obstacle

        obstacle_point = Point()
        obstacle_point.x = obstacle["position"][0]
        obstacle_point.y = obstacle["position"][1]
        obstacle_point.z = obstacle["position"][2]
        obstacle_point_transformed = tf2_geometry_msgs.do_transform_point(
            PointStamped(point=obstacle_point), transform
        )
        obstacle["position"] = [
            obstacle_point_transformed.point.x,
            obstacle_point_transformed.point.y,
            obstacle_point_transformed.point.z,
        ]
        obstacle["frame_id"] = "link_base"
        return obstacle

    def clicked_point_cb(self, msg: Point):
        if not hasattr(self, "chain"):
            self.get_logger().warn(
                "Chain not initialized. Waiting for robot_description."
            )
            return

        print(f"Clicked point: {msg.x}, {msg.y}, {msg.z}")

        links: list[URDFLink] = self.chain.links

        q = [0.0] * len(links)
        names = []
        limits = []
        for i, link in enumerate(links):
            if not isinstance(link, URDFLink) or link.joint_type == "fixed":
                continue

            limits.append(link.bounds)
            q[i] = random.uniform(link.bounds[0], link.bounds[1])
            names.append(link.name)

        target_pos = [msg.x, msg.y, msg.z]
        target_rot = [1.0, 0.0, 0.0]

        ik = self.chain.inverse_kinematics(target_pos, target_rot, orientation_mode="X")

        target_positions = [float(x) for x in ik[1:7]]
        print(target_positions)
        print(names)
        self.names = names

        print(f"Planning path from {self.current_joint_angles} to {target_positions}")

        obstacle_list = []

        center = [
            0.7930782,
            0.4170361,
            0.5,
        ]
        for x in np.linspace(-0.25, 0.25, 5):
            for y in np.linspace(-0.15, 0.15, 5):
                obstacle_list.append(
                    {
                        "position": [x + center[0], y + center[1], center[2]],
                        "radius": 0.05,
                        "frame_id": "base_link",
                    }
                )

        # Transform obstacles to the arm's frame
        for obstacle in obstacle_list:
            obstacle = self.transform_obstacle(obstacle)

        self.visualize_obstacles(obstacle_list)

        planner = RRTStarPlanner(
            start=self.current_joint_angles,
            goal=target_positions,
            obstacle_list=obstacle_list,
            chain=self.chain,
            bounds=limits,
            max_iters=1000,
        )
        plan = planner.plan()
        if plan is None:
            self.get_logger().warn("No path found")
            return

        self.plan = plan
        print(f"Plan: {plan}")

    def robot_description_cb(self, msg: String):
        with open("/tmp/johnny.urdf", "w") as f:
            f.write(msg.data)

        self.chain = Chain.from_urdf_file(
            "/tmp/johnny.urdf",
            base_elements=["link_base"],
            active_links_mask=[
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
            ],
        )

    def send_random_joint_angle(self):
        if not hasattr(self, "chain"):
            self.get_logger().warn(
                "Chain not initialized. Waiting for robot_description."
            )
            return

        links: list[URDFLink] = self.chain.links

        q = [0.0] * len(links)
        names = []
        for i, link in enumerate(links):
            if not isinstance(link, URDFLink) or link.joint_type == "fixed":
                continue

            q[i] = random.uniform(link.bounds[0], link.bounds[1])
            names.append(link.name)

        T = self.chain.forward_kinematics(q)

        target_pos = T[:3, 3]
        target_rot = T[:3, :3]
        ik = self.chain.inverse_kinematics(
            target_pos, target_rot, orientation_mode="all"
        )

        positions = [float(x) for x in ik[1:7]]

        msg = JointState()
        msg.name = names
        msg.position = positions

        self.joint_command_pub.publish(msg)

    def draw_spiral(self, radius=0.5, num_points=100):
        if not hasattr(self, "chain"):
            self.get_logger().warn(
                "Chain not initialized. Waiting for robot_description."
            )
            return

        links: list[URDFLink] = self.chain.links

        q = [0.0] * len(links)
        names = []
        limits = []
        for i, link in enumerate(links):
            if not isinstance(link, URDFLink) or link.joint_type == "fixed":
                continue

            limits.append(link.bounds)
            q[i] = random.uniform(link.bounds[0], link.bounds[1])
            names.append(link.name)

        T = self.chain.forward_kinematics(q)

        if not hasattr(self, "dtheta"):
            self.dtheta = 2 * math.pi / num_points
            self.theta = 0.0

            self.dz = 0.01
            self.target_z = 0.2

        x = radius * math.cos(self.theta)
        y = radius * math.sin(self.theta)

        target_pos = [x, y, self.target_z]

        target_rot = R.from_euler(
            "xyz", [-np.pi / 2, -np.pi / 2, self.theta - np.pi / 2], degrees=False
        ).as_matrix()

        T = np.eye(4)
        T[:3, :3] = target_rot
        T[:3, 3] = target_pos

        # IK can break if the robot is at its limits
        if self.at_limits(self.current_joint_angles[1:7], limits):
            print("At limits, setting initial_ik to zero")
            ik = self.chain.inverse_kinematics_frame(
                T,
                initial_position=[0.0] * 10,
                orientation_mode="all",
            )

        else:
            ik = self.chain.inverse_kinematics_frame(
                T,
                initial_position=self.current_joint_angles,
                orientation_mode="all",
            )

        self.current_joint_angles = ik

        positions = [float(x) for x in ik[1:7]]

        msg = JointState()
        msg.name = names
        msg.position = positions

        self.joint_command_pub.publish(msg)

        self.theta += self.dtheta
        if self.theta >= 2 * math.pi:
            self.theta = 0.0

        self.target_z += self.dz
        if self.target_z >= 0.8:
            self.dz = -self.dz

        elif self.target_z <= 0.2:
            self.dz = -self.dz

        print(self.theta)

    def at_limits(self, q, limits, tolerance=0.1):
        for i in range(len(q)):
            if q[i] < limits[i][0] + tolerance or q[i] > limits[i][1] - tolerance:
                return True
        return False


def main(args=None):
    rclpy.init(args=args)

    planner = ArmTrajectoryPlanner()

    # Use a multithreaded executor to handle callbacks concurrently
    executor = MultiThreadedExecutor()
    executor.add_node(planner)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        planner.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
