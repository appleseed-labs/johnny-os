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


class ArmTrajectoryPlanner(Node):
    def __init__(self):
        super().__init__("arm_trajectory_planner")

        # Create callback group for concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()

        # Current joint state
        self.current_joint_state = None
        self.joint_state_lock = Lock()

        # Robot description (URDF)
        self.robot_description = None
        self.robot_model = None

        # Joint limits
        self.joint_limits = None

        # Hard-coded velocity and acceleration limits
        # These will be properly assigned to joint names from URDF later
        self.joint_velocity_limits = {}  # rad/s
        self.joint_acceleration_limits = {}  # rad/s^2

        # Obstacles as 3D bounding boxes [x_min, y_min, z_min, x_max, y_max, z_max]
        self.obstacles = [
            [0.5, 0.5, 0.0, 0.7, 0.7, 0.5],  # Obstacle 1
            [0.5, -0.5, 0.0, 0.7, -0.3, 0.5],  # Obstacle 2
            [-0.5, 0.5, 0.0, -0.3, 0.7, 0.5],  # Obstacle 3
        ]

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_state",
            self.joint_state_callback,
            10,
            callback_group=self.callback_group,
        )

        self.robot_description_sub = self.create_subscription(
            String,
            "/robot_description",
            self.robot_description_callback,
            10,
            callback_group=self.callback_group,
        )

        # Subscriber for target pose
        self.target_pose_sub = self.create_subscription(
            Pose,
            "/target_pose",
            self.target_pose_callback,
            10,
            callback_group=self.callback_group,
        )

        # Publisher for joint commands
        self.joint_command_pub = self.create_publisher(JointState, "/joint_command", 10)

        self.get_logger().info("Arm Trajectory Planner initialized")

    def joint_state_callback(self, msg):
        with self.joint_state_lock:
            self.current_joint_state = msg
            self.get_logger().debug(f"Received joint state: {msg.position}")

    def robot_description_callback(self, msg):
        self.robot_description = msg.data
        self.parse_robot_description()
        self.get_logger().info("Received robot description")

    def target_pose_callback(self, msg):
        self.get_logger().info(f"Received target pose: {msg.position}")
        # Check if we have the necessary data
        if self.current_joint_state is None:
            self.get_logger().error("No joint state received yet")
            return

        if self.robot_model is None:
            self.get_logger().error("No robot description received yet")
            return

        # Plan and execute trajectory
        self.plan_and_execute_trajectory(msg)

    def parse_robot_description(self):
        if self.robot_description is None:
            return

        try:
            root = ET.fromstring(self.robot_description)

            # Extract joint limits
            self.joint_limits = {}

            for joint in root.findall(".//joint"):
                joint_name = joint.get("name")
                joint_type = joint.get("type")

                if joint_type in ["revolute", "prismatic"]:
                    limit = joint.find("limit")
                    if limit is not None:
                        lower = float(limit.get("lower", -np.pi))
                        upper = float(limit.get("upper", np.pi))
                        self.joint_limits[joint_name] = (lower, upper)

                        # Set hard-coded velocity and acceleration limits for each joint
                        self.joint_velocity_limits[joint_name] = 1.0  # rad/s
                        self.joint_acceleration_limits[joint_name] = 2.0  # rad/s^2

            # Simplified robot model for collision checking
            self.robot_model = {"joints": list(self.joint_limits.keys()), "links": []}

            for link in root.findall(".//link"):
                link_name = link.get("name")
                collision = link.find("collision")
                if collision is not None:
                    geometry = collision.find("geometry")
                    if geometry is not None:
                        # Use a bounding box for each link
                        link_bbox = self.extract_bounding_box(geometry)
                        self.robot_model["links"].append(
                            {"name": link_name, "bbox": link_bbox}
                        )

            self.get_logger().info(
                f'Parsed {len(self.joint_limits)} joints and {len(self.robot_model["links"])} links'
            )

        except Exception as e:
            self.get_logger().error(f"Error parsing robot description: {str(e)}")

    def extract_bounding_box(self, geometry):
        # Extract bounding box from geometry element
        box = geometry.find("box")
        if box is not None:
            size = box.get("size", "0.1 0.1 0.1").split()
            return [
                -float(size[0]) / 2,
                -float(size[1]) / 2,
                -float(size[2]) / 2,
                float(size[0]) / 2,
                float(size[1]) / 2,
                float(size[2]) / 2,
            ]

        cylinder = geometry.find("cylinder")
        if cylinder is not None:
            radius = float(cylinder.get("radius", "0.1"))
            length = float(cylinder.get("length", "0.1"))
            return [-radius, -radius, -length / 2, radius, radius, length / 2]

        sphere = geometry.find("sphere")
        if sphere is not None:
            radius = float(sphere.get("radius", "0.1"))
            return [-radius, -radius, -radius, radius, radius, radius]

        # Default small box if no geometry is specified
        return [-0.05, -0.05, -0.05, 0.05, 0.05, 0.05]

    def plan_and_execute_trajectory(self, target_pose):
        self.get_logger().info("Planning trajectory...")

        # Get current joint state
        with self.joint_state_lock:
            if self.current_joint_state is None:
                self.get_logger().error("No joint state available")
                return
            start_state = list(self.current_joint_state.position)

        # Convert target pose to joint space (inverse kinematics)
        goal_state = self.inverse_kinematics(target_pose)

        if goal_state is None:
            self.get_logger().error("Failed to compute inverse kinematics")
            return

        # Plan trajectory using RRT*
        trajectory = self.plan_rrt_star(start_state, goal_state)

        if trajectory is None or len(trajectory) == 0:
            self.get_logger().error("Failed to plan trajectory")
            return

        # Apply velocity and acceleration limits
        smooth_trajectory = self.apply_limits(trajectory)

        # Execute trajectory
        self.execute_trajectory(smooth_trajectory)

    def inverse_kinematics(self, target_pose):
        """
        Placeholder for inverse kinematics calculation
        In a real implementation, this would compute the joint angles
        that achieve the target pose using an IK solver
        """
        # For demonstration, return a random valid configuration
        # that would be close to the target pose
        random_config = []
        for joint_name in self.joint_limits:
            lower, upper = self.joint_limits[joint_name]
            random_config.append(lower + (upper - lower) * random.random())

        return random_config

    def plan_rrt_star(
        self,
        start_state,
        goal_state,
        max_iterations=1000,
        step_size=0.1,
        goal_sample_rate=0.1,
        search_radius=0.5,
    ):
        """
        RRT* algorithm for path planning

        Args:
            start_state: Starting joint configuration
            goal_state: Goal joint configuration
            max_iterations: Maximum iterations to run
            step_size: Distance to extend tree
            goal_sample_rate: Probability of sampling the goal
            search_radius: Radius to search for neighbors for rewiring

        Returns:
            List of joint configurations forming a path from start to goal
        """
        # Tree structure: {node_id: {'state': joint_state, 'parent': parent_id, 'cost': cost}}
        tree = {0: {"state": start_state, "parent": None, "cost": 0.0}}

        # Track nodes close to the goal
        goal_candidates = []

        for i in range(max_iterations):
            # Sample a random state
            if random.random() < goal_sample_rate:
                random_state = goal_state
            else:
                random_state = self.sample_random_state()

            # Find nearest node in the tree
            nearest_node_id = self.find_nearest(tree, random_state)

            # Extend towards the random state
            new_state = self.extend(
                tree[nearest_node_id]["state"], random_state, step_size
            )

            # Check if the new state is valid
            if not self.is_state_valid(new_state) or not self.is_motion_valid(
                tree[nearest_node_id]["state"], new_state
            ):
                continue

            # Add the new node to the tree
            new_node_id = len(tree)
            cost = tree[nearest_node_id]["cost"] + self.distance(
                tree[nearest_node_id]["state"], new_state
            )
            tree[new_node_id] = {
                "state": new_state,
                "parent": nearest_node_id,
                "cost": cost,
            }

            # Find neighbors within search radius
            neighbors = self.find_neighbors(tree, new_state, search_radius)

            # Connect to the best parent
            self.connect_to_best_parent(tree, new_node_id, neighbors)

            # Rewire the tree
            self.rewire_tree(tree, new_node_id, neighbors)

            # Check if we can connect to the goal
            if self.distance(
                new_state, goal_state
            ) < step_size and self.is_motion_valid(new_state, goal_state):
                goal_node_id = len(tree)
                goal_cost = tree[new_node_id]["cost"] + self.distance(
                    new_state, goal_state
                )
                tree[goal_node_id] = {
                    "state": goal_state,
                    "parent": new_node_id,
                    "cost": goal_cost,
                }
                goal_candidates.append((goal_node_id, goal_cost))

        # Return the best path to the goal if found
        if goal_candidates:
            best_goal_id = min(goal_candidates, key=lambda x: x[1])[0]
            return self.extract_path(tree, best_goal_id)

        # If no path to the goal is found, return the path to the node closest to the goal
        closest_node_id = self.find_nearest(tree, goal_state)
        return self.extract_path(tree, closest_node_id)

    def sample_random_state(self):
        """Sample a random joint configuration within joint limits"""
        random_state = []
        for joint_name in self.joint_limits:
            lower, upper = self.joint_limits[joint_name]
            random_state.append(lower + (upper - lower) * random.random())
        return random_state

    def find_nearest(self, tree, state):
        """Find the nearest node in the tree to the given state"""
        return min(
            tree.keys(),
            key=lambda node_id: self.distance(tree[node_id]["state"], state),
        )

    def distance(self, state1, state2):
        """Compute the Euclidean distance between two states"""
        return np.linalg.norm(np.array(state1) - np.array(state2))

    def extend(self, from_state, to_state, step_size):
        """Extend from one state towards another by a limited step size"""
        direction = np.array(to_state) - np.array(from_state)
        norm = np.linalg.norm(direction)
        if norm < step_size:
            return to_state
        else:
            return list(np.array(from_state) + step_size * direction / norm)

    def find_neighbors(self, tree, state, radius):
        """Find all nodes in the tree within a certain radius of the state"""
        return [
            node_id
            for node_id, node in tree.items()
            if self.distance(node["state"], state) < radius
        ]

    def connect_to_best_parent(self, tree, node_id, neighbors):
        """Connect the node to the best parent in its neighborhood"""
        if not neighbors:
            return

        current_state = tree[node_id]["state"]
        current_parent = tree[node_id]["parent"]
        current_cost = tree[node_id]["cost"]

        for neighbor_id in neighbors:
            if neighbor_id == current_parent or neighbor_id == node_id:
                continue

            neighbor_state = tree[neighbor_id]["state"]
            potential_cost = tree[neighbor_id]["cost"] + self.distance(
                neighbor_state, current_state
            )

            if potential_cost < current_cost and self.is_motion_valid(
                neighbor_state, current_state
            ):
                tree[node_id]["parent"] = neighbor_id
                tree[node_id]["cost"] = potential_cost

    def rewire_tree(self, tree, node_id, neighbors):
        """Rewire the tree by potentially changing the parents of neighboring nodes"""
        if not neighbors:
            return

        current_state = tree[node_id]["state"]
        current_cost = tree[node_id]["cost"]

        for neighbor_id in neighbors:
            if neighbor_id == tree[node_id]["parent"] or neighbor_id == node_id:
                continue

            neighbor_state = tree[neighbor_id]["state"]
            potential_cost = current_cost + self.distance(current_state, neighbor_state)

            if potential_cost < tree[neighbor_id]["cost"] and self.is_motion_valid(
                current_state, neighbor_state
            ):
                tree[neighbor_id]["parent"] = node_id
                tree[neighbor_id]["cost"] = potential_cost

    def extract_path(self, tree, node_id):
        """Extract the path from the start node to the given node"""
        path = []
        current_id = node_id

        while current_id is not None:
            path.append(tree[current_id]["state"])
            current_id = tree[current_id]["parent"]

        return list(reversed(path))

    def is_state_valid(self, state):
        """Check if a state is valid (within joint limits and no collisions)"""
        # Check joint limits
        for i, joint_name in enumerate(self.joint_limits):
            lower, upper = self.joint_limits[joint_name]
            if state[i] < lower or state[i] > upper:
                return False

        # Check self-collisions and collisions with obstacles
        if self.check_collisions(state):
            return False

        return True

    def is_motion_valid(self, from_state, to_state, checks=10):
        """Check if the motion between two states is valid"""
        # Interpolate between states and check each interpolated state
        for i in range(checks + 1):
            t = i / checks
            interpolated_state = [
                from_state[j] * (1 - t) + to_state[j] * t
                for j in range(len(from_state))
            ]

            if not self.is_state_valid(interpolated_state):
                return False

        return True

    def check_collisions(self, joint_state):
        """
        Check for collisions with the given joint state

        Returns True if collision detected, False otherwise
        """
        # Forward kinematics to get the positions of all links
        link_positions = self.forward_kinematics(joint_state)

        # Check self-collisions
        for i in range(len(link_positions)):
            for j in range(i + 1, len(link_positions)):
                if self.check_bbox_collision(link_positions[i], link_positions[j]):
                    return True

        # Check collisions with obstacles
        for link_pos in link_positions:
            for obstacle in self.obstacles:
                if self.check_bbox_collision(link_pos, obstacle):
                    return True

        return False

    def forward_kinematics(self, joint_state):
        """
        Perform forward kinematics to get the positions of all links

        This is a simplified approach - in a real implementation, you would
        use a proper kinematics library

        Returns:
            List of link bounding boxes in world coordinates
        """
        # Placeholder for forward kinematics
        # In a real implementation, this would compute the actual link positions
        # based on the joint state and the robot kinematics

        link_positions = []

        # Simplified model: assume a chain of links along the z-axis
        current_pos = [0, 0, 0]
        for i, angle in enumerate(joint_state):
            # Update position based on joint angle
            current_pos[0] += 0.1 * math.cos(angle)
            current_pos[1] += 0.1 * math.sin(angle)
            current_pos[2] += 0.1

            # Create a bounding box for this link
            bbox = [
                current_pos[0] - 0.05,
                current_pos[1] - 0.05,
                current_pos[2] - 0.05,
                current_pos[0] + 0.05,
                current_pos[1] + 0.05,
                current_pos[2] + 0.05,
            ]
            link_positions.append(bbox)

        return link_positions

    def check_bbox_collision(self, bbox1, bbox2):
        """Check if two bounding boxes collide"""
        return (
            bbox1[0] <= bbox2[3]
            and bbox1[3] >= bbox2[0]
            and bbox1[1] <= bbox2[4]
            and bbox1[4] >= bbox2[1]
            and bbox1[2] <= bbox2[5]
            and bbox1[5] >= bbox2[2]
        )

    def apply_limits(self, trajectory, dt=0.1):
        """
        Apply velocity and acceleration limits to a trajectory

        Args:
            trajectory: List of joint states
            dt: Time step between states

        Returns:
            Smooth trajectory that respects velocity and acceleration limits
        """
        if len(trajectory) <= 2:
            return trajectory

        # Extract joint names
        joint_names = list(self.joint_limits.keys())

        # Initialize smooth trajectory with the first point
        smooth_trajectory = [trajectory[0]]

        # Current velocities for each joint
        current_velocities = [0.0] * len(joint_names)

        for i in range(1, len(trajectory)):
            current_state = smooth_trajectory[-1]
            target_state = trajectory[i]

            # Calculate desired joint velocities
            desired_velocities = [
                (target_state[j] - current_state[j]) / dt
                for j in range(len(joint_names))
            ]

            # Apply velocity limits
            limited_velocities = []
            for j, joint_name in enumerate(joint_names):
                vel_limit = self.joint_velocity_limits[joint_name]
                if abs(desired_velocities[j]) > vel_limit:
                    limited_velocities.append(
                        vel_limit * np.sign(desired_velocities[j])
                    )
                else:
                    limited_velocities.append(desired_velocities[j])

            # Apply acceleration limits
            for j, joint_name in enumerate(joint_names):
                acc_limit = self.joint_acceleration_limits[joint_name]
                max_vel_change = acc_limit * dt
                vel_change = limited_velocities[j] - current_velocities[j]

                if abs(vel_change) > max_vel_change:
                    limited_velocities[j] = current_velocities[
                        j
                    ] + max_vel_change * np.sign(vel_change)

            # Update current velocities
            current_velocities = limited_velocities

            # Calculate new state
            new_state = [
                current_state[j] + current_velocities[j] * dt
                for j in range(len(joint_names))
            ]

            smooth_trajectory.append(new_state)

            # Check if we're close enough to the target
            if self.distance(new_state, target_state) < 0.01:
                # Skip to the next target point
                continue

        # Ensure the final point is exactly the goal state
        smooth_trajectory[-1] = trajectory[-1]

        return smooth_trajectory

    def execute_trajectory(self, trajectory, dt=0.1):
        """
        Execute a trajectory by publishing joint commands

        Args:
            trajectory: List of joint states
            dt: Time step between states
        """
        self.get_logger().info(f"Executing trajectory with {len(trajectory)} points")

        joint_names = list(self.joint_limits.keys())

        for i, state in enumerate(trajectory):
            # Create joint state message
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = joint_names
            msg.position = state

            # Calculate velocities for all but the first and last point
            if i > 0 and i < len(trajectory) - 1:
                prev_state = trajectory[i - 1]
                velocities = [
                    (state[j] - prev_state[j]) / dt for j in range(len(joint_names))
                ]
                msg.velocity = velocities
            else:
                msg.velocity = [0.0] * len(joint_names)

            # Publish joint command
            self.joint_command_pub.publish(msg)

            # Sleep to maintain the specified rate
            time.sleep(dt)

        self.get_logger().info("Trajectory execution completed")


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
