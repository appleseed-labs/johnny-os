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
from ikpy.chain import Chain
from ikpy.link import URDFLink
import tf2_ros
import trimesh
import pyglet
import os

# ROS messages
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import MarkerArray, Marker
import tf2_geometry_msgs


class BoxObstacle:
    def __init__(self, box_min, box_max, frame_id):
        assert len(box_min) == 3
        assert len(box_max) == 3
        assert all(
            box_min[i] <= box_max[i] for i in range(3)
        ), "Box min must be less than or equal to box max"
        self.box_min = box_min
        self.box_max = box_max
        self.frame_id = frame_id


class SphereObstacle:
    def __init__(self, center, radius, frame_id):
        assert len(center) == 3
        assert radius > 0, "Radius must be positive"
        self.center = center
        self.radius = radius
        self.frame_id = frame_id


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

        self.create_subscription(JointState, "/joint_states", self.joint_state_cb, 1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # This stores the current joint angles
        # (including for inactive joints like link_base and the gripper link)
        # Only joints 1-6 are active (zero-indexed)
        self.current_joint_angles = None

        self.create_timer(1.0, self.publish_path)

    def publish_path(self):

        if not hasattr(self, "path") or len(self.path) == 0:
            return

        msg = JointState()
        print(self.path[0])
        msg.position = [float(q) for q in self.path[0]]

        active_names = []
        active_link_mask = self.chain.active_links_mask
        for i, link in enumerate(self.chain.links):
            if not isinstance(link, URDFLink) or link.joint_type == "fixed":
                continue

            if active_link_mask[i]:
                active_names.append(link.name)

        print(active_names)

        msg.name = active_names
        self.joint_command_pub.publish(msg)

        self.path.pop(0)

    def joint_state_cb(self, msg: JointState):
        """
        Callback for joint states
        """

        if not hasattr(self, "chain"):
            self.get_logger().warn(
                "Chain not initialized. Waiting for robot_description."
            )
            return

        names_in_chain = [link.name for link in self.chain.links]

        self.current_joint_angles = [0.0] * len(names_in_chain)

        # Update the current joint angles
        for i, name in enumerate(msg.name):
            if name in names_in_chain:
                index = names_in_chain.index(name)
                self.current_joint_angles[index] = msg.position[i]

    def visualize_obstacles(self, obstacles: list[dict]):
        """
        Visualize each obstacle as a sphere in RViz
        """
        msg = MarkerArray()
        for i, obstacle in enumerate(obstacles):
            marker = Marker()
            marker.header.frame_id = obstacle.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "obstacle"
            marker.id = i

            if isinstance(obstacle, BoxObstacle):
                marker.type = Marker.CUBE
                marker.scale.x = obstacle.box_max[0] - obstacle.box_min[0]
                marker.scale.y = obstacle.box_max[1] - obstacle.box_min[1]
                marker.scale.z = obstacle.box_max[2] - obstacle.box_min[2]
                marker.pose.position.x = (obstacle.box_min[0] + obstacle.box_max[0]) / 2
                marker.pose.position.y = (obstacle.box_min[1] + obstacle.box_max[1]) / 2
                marker.pose.position.z = (obstacle.box_min[2] + obstacle.box_max[2]) / 2
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 0.5
            elif isinstance(obstacle, SphereObstacle):
                marker.type = Marker.SPHERE
                marker.scale.x = obstacle.radius * 2
                marker.scale.y = obstacle.radius * 2
                marker.scale.z = obstacle.radius * 2
                marker.pose.position.x = obstacle.center[0]
                marker.pose.position.y = obstacle.center[1]
                marker.pose.position.z = obstacle.center[2]
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                marker.color.a = 0.5
            else:
                self.get_logger().error(
                    "Unsupported obstacle type. Only BoxObstacle and SphereObstacle are supported."
                )
                continue

            msg.markers.append(marker)

        self.obstacle_vis_pub.publish(msg)

    def transform_obstacle(self, obstacle, target_frame="link_base"):
        """
        Transform the obstacle to the arm's frame
        """

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, obstacle.frame_id, rclpy.time.Time()
            )
        except tf2_ros.LookupException as e:
            self.get_logger().error(f"Transform not found: {e}")
            return obstacle

        if isinstance(obstacle, BoxObstacle):
            box_min = np.array(obstacle.box_min)
            box_max = np.array(obstacle.box_max)

            # Transform the box min and max points
            box_min_transformed = tf2_geometry_msgs.do_transform_point(
                PointStamped(
                    header=transform.header,
                    point=Point(x=box_min[0], y=box_min[1], z=box_min[2]),
                ),
                transform,
            )
            box_max_transformed = tf2_geometry_msgs.do_transform_point(
                PointStamped(
                    header=transform.header,
                    point=Point(x=box_max[0], y=box_max[1], z=box_max[2]),
                ),
                transform,
            )
            box_min = [
                box_min_transformed.point.x,
                box_min_transformed.point.y,
                box_min_transformed.point.z,
            ]
            box_max = [
                box_max_transformed.point.x,
                box_max_transformed.point.y,
                box_max_transformed.point.z,
            ]

            transformed_box_min_x = min(box_min[0], box_max[0])
            transformed_box_min_y = min(box_min[1], box_max[1])
            transformed_box_min_z = min(box_min[2], box_max[2])
            transformed_box_max_x = max(box_min[0], box_max[0])
            transformed_box_max_y = max(box_min[1], box_max[1])
            transformed_box_max_z = max(box_min[2], box_max[2])
            return BoxObstacle(
                [transformed_box_min_x, transformed_box_min_y, transformed_box_min_z],
                [transformed_box_max_x, transformed_box_max_y, transformed_box_max_z],
                frame_id=target_frame,
            )

        elif isinstance(obstacle, SphereObstacle):
            center = np.array(obstacle.center)
            radius = obstacle.radius

            # Transform the sphere center
            center_transformed = tf2_geometry_msgs.do_transform_point(
                PointStamped(
                    header=transform.header,
                    point=Point(x=center[0], y=center[1], z=center[2]),
                ),
                transform,
            )
            center = [
                center_transformed.point.x,
                center_transformed.point.y,
                center_transformed.point.z,
            ]
            return SphereObstacle(center, radius, frame_id=target_frame)
        else:
            self.get_logger().error(
                "Unsupported obstacle type. Only BoxObstacle and SphereObstacle are supported."
            )
            return None

    def clicked_point_cb(self, msg: Point):
        if not hasattr(self, "chain"):
            self.get_logger().warn(
                "Chain not initialized. Waiting for robot_description."
            )
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
