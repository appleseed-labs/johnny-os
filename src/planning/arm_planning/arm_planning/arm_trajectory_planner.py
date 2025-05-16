#!/usr/bin/env python3

import heapq
from tqdm import trange
from ament_index_python import get_package_share_directory
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

pos_dictionary = {
    "home": [
        -0.32572651,
        -80.93894022,
        -14.3142046,
        -188.32544039,
        -7.37580029,
        -170.15998538,
        0.0,
    ],
    "ready": [
        130.09470834,
        33.48371084,
        -139.9335396,
        -68.80289198,
        129.62064306,
        -175.63041452,
        0.0,
    ],
    "seedling_1_pre": [
        117.616547,
        -12.89321197,
        -55.8271168,
        -53.73828456,
        99.10324932,
        -147.33598243,
        0.0,
    ],
    "seedling_1_grab": [
        116.8671755,
        -9.62150837,
        -51.20936345,
        -53.80446119,
        96.2770204,
        -138.36598437,
        0.0,
    ],
    "seedling_1_lift": [
        138.18154925,
        11.32691724,
        -77.62907763,
        -43.34248103,
        115.49729071,
        -129.13036308,
        0.0,
    ],
    "over_hole": [-29.1542, 16.5677, -28.8071, -176.041, 62.1542, -158.9843, 0.0],
    "to_hole_1": [73.9455, -4.1437, -91.7979, -64.7164, 76.854, -187.4988, 0.0],
    "to_hole_2": [7.449, -51.7186, -27.6111, -84.6549, 8.9232, -187.504, 0.0],
}


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

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
            callback_group=self.callback_group,
        )

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

        # Publisher for joint commands
        self.joint_command_pub = self.create_publisher(JointState, "/joint_command", 10)

        self.get_logger().info("Arm Trajectory Planner initialized")

    def joint_state_callback(self, msg):
        with self.joint_state_lock:
            self.current_joint_state = msg
            self.get_logger().debug(f"Received joint state: {msg.position}")

    def send_test_pose(self):
        test_pose = Pose()
        test_pose.position.x = 0.5
        test_pose.position.y = 0.0
        test_pose.position.z = 1.5
        test_pose.orientation.x = 0.0
        test_pose.orientation.y = 0.0
        test_pose.orientation.z = 0.0
        test_pose.orientation.w = 1.0
        self.get_logger().info(f"Publishing test pose: {test_pose.position}")

        self.test_pose_pub.publish(test_pose)

    def robot_description_callback(self, msg):
        pass

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

    def extract_joint_positions(self, msg: JointState):
        """
        Extract joint positions from a JointState message

        Only consider joints with "joint" in their name, e.g. "joint1", "joint2"
        """
        joint_positions = [0.0] * self.num_joints
        for i, name in enumerate(msg.name):
            if name[:5] == "joint":
                joint_number = int(name.replace("joint", ""))
                joint_positions[joint_number - 1] = msg.position[i]
        return joint_positions

    def plan_and_execute_trajectory(self, target_pose):
        self.get_logger().info("Planning trajectory...")

        # Get current joint state
        with self.joint_state_lock:
            if self.current_joint_state is None:
                self.get_logger().error("No joint state available")
                return

            start_state = self.extract_joint_positions(self.current_joint_state)
            print(f"Start state: {start_state}")
            # start_state = list(self.current_joint_state.position)

        # Convert target pose to joint space (inverse kinematics)
        self.get_logger().info(
            f"Calculating IK for target pose: {target_pose.position}"
        )
        goal_state = self.inverse_kinematics(target_pose)
        self.get_logger().info(f"Goal state is {goal_state}")
        self.get_logger().info(f"Start state is {start_state}")

        if goal_state is None:
            self.get_logger().error("Failed to compute inverse kinematics")
            return

        if not hasattr(self, "prm"):
            self.get_logger().warn(
                "PRM not initialized. Waiting for robot_description."
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

        ik = self.chain.inverse_kinematics(target_pos)

        goal_config = [float(x) for x in self.chain.active_from_full(ik)]
        print(goal_config)
        print(names)
        self.names = names

        start_config = self.chain.active_from_full(self.current_joint_angles)
        print(f"Planning path from ({target_pos}) {start_config} to {goal_config}")

        valid_configs = self.prm.tolist()
        valid_configs.append(start_config)
        valid_configs.append(goal_config)

        # Check that each configuration has six elements
        assert all(
            len(config) == 6 for config in valid_configs
        ), "All configurations must have six elements"

        path = self.a_star_robot_arm(
            start_config,
            goal_config,
            valid_configs,
            neighbor_dist_threshold=2.0,
        )

        if path:
            print(f"Path found with {len(path)} configurations:")
            configs = []
            for i, config in enumerate(path):
                print(f"Step {i}: {config}")
                configs.append(config)

            self.path = configs

            # Calculate total path cost
            total_cost = 0
            for i in range(len(path) - 1):
                total_cost += self.joint_distance(path[i], path[i + 1])
            print(f"Total path cost: {total_cost}")
        else:
            print(
                "No path found. Try increasing the neighbor_dist_threshold or adding more valid configurations."
            )

    def a_star_robot_arm(
        self,
        start_config,
        goal_config,
        valid_configs,
        neighbor_dist_threshold=0.1,
    ):
        """
        Implementation of A* algorithm for a robot arm path planning.

        Args:
            start_config: Initial joint configuration (numpy array)
            goal_config: Target joint configuration (numpy array)
            valid_configs: List of all valid configurations (list of numpy arrays)
            neighbor_dist_threshold: Maximum distance for configurations to be considered neighbors

        Returns:
            List of configurations forming a path from start to goal if found, otherwise None
        """

        # Convert configs to tuples for hashing
        start_tuple = tuple(map(float, start_config))
        goal_tuple = tuple(map(float, goal_config))

        # Convert valid_configs to a set of tuples for faster lookup
        valid_config_tuples = {tuple(map(float, config)) for config in valid_configs}

        # Ensure start and goal are in valid configs
        valid_config_tuples.add(start_tuple)
        valid_config_tuples.add(goal_tuple)

        # Counter for tie-breaking in the priority queue
        counter = 0

        # Priority queue for open set (f_score, counter, config_tuple)
        open_set = [
            (self.joint_distance(start_config, goal_config), counter, start_tuple)
        ]
        counter += 1

        # Set to track nodes in open set for quick lookup
        open_set_hash = {start_tuple}

        # Dictionary to store g_scores (cost from start to node)
        g_score = {start_tuple: 0}

        # Dictionary to store f_scores (g_score + heuristic)
        f_score = {start_tuple: self.joint_distance(start_config, goal_config)}

        # Dictionary to store parent nodes for path reconstruction
        came_from = {}

        # Set to keep track of visited nodes
        closed_set = set()

        # Find neighboring configurations that are within the distance threshold
        def find_neighbors(config_tuple):
            """Find valid configurations that are nearby in configuration space"""
            neighbors = []
            config = np.array(config_tuple)

            for valid_tuple in valid_config_tuples:
                # Skip the current configuration and already visited ones
                if valid_tuple == config_tuple or valid_tuple in closed_set:
                    continue

                valid_config = np.array(valid_tuple)

                # Check if distance is within threshold
                if self.joint_distance(config, valid_config) <= neighbor_dist_threshold:
                    neighbors.append(valid_tuple)

            return neighbors

        # Main A* loop
        while open_set:
            # Get node with lowest f_score
            _, _, current_tuple = heapq.heappop(open_set)
            open_set_hash.remove(current_tuple)

            # If already visited, skip
            if current_tuple in closed_set:
                continue

            # Add to closed set
            closed_set.add(current_tuple)

            # Check if goal is reached (using approximate equality for floating point)
            if np.allclose(np.array(current_tuple), goal_config, atol=1e-6):
                # Reconstruct path
                path = [np.array(current_tuple)]
                temp_tuple = current_tuple
                while temp_tuple in came_from:
                    temp_tuple = came_from[temp_tuple]
                    path.append(np.array(temp_tuple))
                path.reverse()
                return path

            # Get neighboring configurations
            neighbors = find_neighbors(current_tuple)

            for neighbor_tuple in neighbors:
                # Calculate tentative g_score (cost from start to neighbor through current)
                tentative_g_score = g_score[current_tuple] + self.joint_distance(
                    np.array(current_tuple), np.array(neighbor_tuple)
                )

                # If new path is better
                if (
                    neighbor_tuple not in g_score
                    or tentative_g_score < g_score[neighbor_tuple]
                ):
                    # Update path
                    came_from[neighbor_tuple] = current_tuple
                    g_score[neighbor_tuple] = tentative_g_score

                    # f_score = g_score + heuristic
                    f_score_val = tentative_g_score + self.joint_distance(
                        np.array(neighbor_tuple), goal_config
                    )
                    f_score[neighbor_tuple] = f_score_val

                    # Add to open set if not already there
                    if neighbor_tuple not in open_set_hash:
                        heapq.heappush(open_set, (f_score_val, counter, neighbor_tuple))
                        counter += 1
                        open_set_hash.add(neighbor_tuple)

        # No path found
        return None

    def joint_distance(self, config1, config2, weights=None):
        """
        Calculate weighted Euclidean distance between joint configurations.

        Args:
            config1, config2: Joint configurations to compare
            weights: Optional weights for each joint (Default: equal weights)

        Returns:
            Weighted Euclidean distance between configurations
        """

        diff = np.array(config1) - np.array(config2)

        if weights is None:
            weights = np.ones_like(diff)

        # Apply weights to the difference (can give more importance to proximal joints)
        weighted_diff = diff * weights

        return np.linalg.norm(weighted_diff)

    def _load_mesh_from_package_path(self, path):
        assert path.startswith("package://")
        path = path[len("package://") :]
        package_name = path.split("/")[0]
        relative_path = path[len(package_name) :]

        # Get the package share directory
        package_share_directory = get_package_share_directory(package_name)

        # Remove the "package://" prefix
        # Construct the full path to the mesh file
        full_path = package_share_directory + relative_path

        # Load the mesh using trimesh
        mesh = trimesh.load_mesh(full_path)
        return mesh

    def _get_mesh(self, link: URDFLink):
        """
        Get the mesh for a given link
        """
        if not hasattr(self, "meshes"):
            self.get_logger().warn("No meshes loaded")
            return None

        # Get the mesh for the link
        if link.name == "arm_link_joint":
            mesh = self.meshes.get("link_base")
        else:
            mesh = self.meshes.get(link.name)

        # if mesh is None:
        #     self.get_logger().warn(f"No mesh found for link {link.name}")
        return mesh

    def show(self):
        T = self.chain.forward_kinematics(
            self.current_joint_angles, full_kinematics=True
        )

        scene = trimesh.Scene()

        # Add axes to scene
        axes = trimesh.creation.axis()
        scene.add_geometry(axes)
        scene.add_geometry(self.meshes.get("base_link"))

        # print(self.meshes)

        for i, link in enumerate(self.chain.links):
            if not isinstance(link, URDFLink):
                print(f"Link {link.name} is not a URDFLink")
                continue

            # Get the mesh for the link
            mesh = self._get_mesh(link)
            if mesh is not None:
                # Transform the mesh to the link's frame

                # Reset the mesh transform

                transformed_mesh = mesh.copy()
                transformed_mesh.apply_transform(T[i])

                scene.add_geometry(transformed_mesh)

            else:
                print(f"No mesh found for link {link.name}")

        scene.set_camera(
            angles=[np.pi / 2, 0.0, np.pi / 2], distance=4.0, center=[0, 0, 0.8]
        )

        scene.show()

    def robot_description_cb(self, msg: String):

        # Parse the URDF as an xml tree
        root = ET.fromstring(msg.data)

        meshes = {}

        # Find all collision elements, print their mesh filenames
        for collision in root.findall(".//collision"):
            mesh = collision.find(".//mesh")
            if mesh is not None:
                filename = mesh.get("filename")
                if filename is not None:
                    mesh = self._load_mesh_from_package_path(filename)
                    # Visualize the mesh using trimesh
                    link_name = (
                        filename.split("/")[-1].split(".")[0].replace("link", "joint")
                    )
                    if link_name == "end_tool":
                        link_name = "joint6"

                    elif link_name == "joint_base":
                        link_name = "link_base"
                    elif link_name == "amiga_chassis_lowpoly":
                        link_name = "base_link"

                    elif link_name == "multi_eef_collision":
                        link_name = "multi_eef_joint"
                    meshes[link_name] = mesh
                else:
                    print("No filename attribute found in the mesh element.")

        self.meshes = meshes

        # Now create a Trimesh collision manager
        self.collision_manager = trimesh.collision.CollisionManager()
        for link_name, mesh in meshes.items():
            # Add the mesh to the collision manager
            print(
                f"Adding {link_name} to collision manager with {mesh.vertices.shape[0]} vertices"
            )
            self.collision_manager.add_object(link_name, mesh)

            if link_name == "link_base":
                # Set the base link to be static
                self.collision_manager.add_object("arm_link_joint", mesh)

        # Change "continuous" to "revolute" in the URDF
        urdf = msg.data.replace("continuous", "revolute")

        with open("/tmp/johnny.urdf", "w") as f:
            f.write(urdf)

        self.chain = Chain.from_urdf_file(
            "/tmp/johnny.urdf",
            base_elements=["base_link", "arm_link_joint"],
            active_links_mask=[
                False,
                False,
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

        self.chain.links[0].name = "base_link"

        try:
            self.prm = np.load(
                os.path.join(
                    get_package_share_directory("arm_planning"), "data", "prm.npy"
                )
            )
            self.get_logger().info(f"PRM loaded with {len(self.prm)} samples")
        except FileNotFoundError:
            self.get_logger().warn("PRM not found, generating new PRM")
            self.build_prm(10000)
            self.save_prm()

    def save_prm(self):
        """Save the PRM to a .npy file"""

        package_share_directory = get_package_share_directory("arm_planning")
        import os

        prm_path = os.path.join(package_share_directory, "data", "prm.npy")

        # Create the path if necessary

        os.makedirs(os.path.dirname(prm_path), exist_ok=True)
        np.save(prm_path, self.prm)
        self.get_logger().info(f"PRM saved to {prm_path}")

    def build_prm(self, n_samples=10000):

        valid_joint_configs = []

        for i in trange(n_samples):
            q = [0.0] * len(self.chain.links)
            names = []
            for i, link in enumerate(self.chain.links):
                if not isinstance(link, URDFLink) or link.joint_type == "fixed":
                    continue

                q[i] = random.uniform(link.bounds[0], link.bounds[1])
                names.append(link.name)

            self.current_joint_angles = q

            T = self.chain.forward_kinematics(q, full_kinematics=True)

            # Update the transforms in the collision manager
            for i, link in enumerate(self.chain.links):
                if not isinstance(link, URDFLink):
                    continue

                # Get the mesh for the link
                mesh = self._get_mesh(link)
                if mesh is not None:
                    self.collision_manager.set_transform(link.name, T[i])

            (
                in_collision,
                objects_in_collision,
            ) = self.collision_manager.in_collision_internal(return_names=True)

            # self_collision_okay = [
            #     ("joint6", "multi_eef_joint"),
            #     ("arm_link_joint", "base_link"),
            # ]

            # num_accounted_for_objects = 0
            # for obj1, obj2 in self_collision_okay:
            #     for obj3, obj4 in objects_in_collision:
            #         if obj1 == obj3 and obj2 == obj4:
            #             num_accounted_for_objects += 1
            #         elif obj1 == obj4 and obj2 == obj3:
            #             num_accounted_for_objects += 1

            # if num_accounted_for_objects == len(self_collision_okay):
            #     in_collision = False

            if in_collision and len(objects_in_collision) > 2:
                # self.show()
                continue

            else:
                valid_joint_configs.append(self.chain.active_from_full(q))

            # if in_collision:
            #     self.show()
            #     exit()

        self.prm = valid_joint_configs

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

        positions = [float(x) for x in self.chain.active_from_full(ik)]

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

    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
