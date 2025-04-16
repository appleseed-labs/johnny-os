import time
import torch as th
import os
from ament_index_python.packages import get_package_share_directory

from stable_baselines3 import PPO
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.node import Node
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

from johnny_msgs.msg import Seedling, ForestPlan


class ForestPlanner(Node):
    """Node to plan seedling layouts using a trained RL model."""

    def __init__(self):
        super().__init__("forest_planner")

        self.species_names = [
            "abal",
            "acca",
            "acpl",
            "acps",
            "algl",
            "alin",
            "alvi",
            "bepe",
            "cabe",
            "casa",
            "coav",
            "fasy",
            "frex",
            "lade",
            "piab",
            "pice",
            "pimu",
            "pini",
            "pisy",
            "poni",
            "potr",
            "psme",
            "qupe",
            "qupu",
            "quro",
            "rops",
            "saca",
            "soar",
            "soau",
            "tico",
            "tipl",
            "ulgl",
        ]

        # The QoS profile allows any late subscriber to receive the latest message
        self.plan_pub = self.create_publisher(
            ForestPlan,
            "/planning/forest_plan",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

        self.grid_size = 100
        self.seedlings_planted = []

        # TODO: Make keepout non-zero given upstream data
        self.keepout = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        self.load_model()

        self.generate_plan(num_seedlings=100)

    def load_model(self):
        # Get the package directory
        package_dir = get_package_share_directory("forest_planning")

        # Path to the model weights
        print("Getting model path")
        model_path = os.path.join(package_dir, "data", "best_model.zip")

        print("Loading model from path:", model_path)
        model = PPO.load(model_path, device="cpu")

        print(model.observation_space.shape)

        self.model = model

    def get_seedling_msg(self, x, y, species_id: str):
        """Create a Seedling message from coordinates and species name."""

        seedling_msg = Seedling()
        seedling_msg.position.point.x = float(x)
        seedling_msg.position.point.y = float(y)
        seedling_msg.position.header.frame_id = "map"
        seedling_msg.position.header.stamp = self.get_clock().now().to_msg()
        seedling_msg.species_id = species_id

        return seedling_msg

    def generate_plan(self, num_seedlings: int):
        """Generate a seedling layout plan. Calls the model once per seedling"""

        start = time.time()

        obs = np.zeros((self.grid_size, self.grid_size, 1), dtype=np.uint8)

        seedling_msgs: list[Seedling] = []

        for i in range(num_seedlings):
            action, hidden = self.model.predict(obs)

            print(action)

            # Convert action to coordinates and species
            x = (action[0] + 1) / 2 * self.grid_size
            y = (action[1] + 1) / 2 * self.grid_size
            x = min(max(x, 0), self.grid_size - 1)
            y = min(max(y, 0), self.grid_size - 1)
            species_index = int((action[2] + 1) / 2 * (len(self.species_names) - 1))

            if self.keepout[int(x), int(y)] == 1:
                print("Coordinates are in the keepout area.")
                continue
                # reward = -100.0
                # return (self.get_observation(), reward, False, False, {})

            # Plant a seedling at the specified coordinates
            seedling = [x, y, self.species_names[species_index]]

            seedling_msg = self.get_seedling_msg(
                x, y, self.species_names[species_index]
            )

            self.seedlings_planted.append(seedling)

            seedling_msgs.append(seedling_msg)

            x_index, y_index = int(seedling[0]), int(seedling[1])
            species_index = self.species_names.index(seedling[2])
            obs[x_index, y_index, 0] = (
                species_index + 2
            )  # +2 because 0 is empty and 1 is keep out

        plan_msg = ForestPlan()
        plan_msg.seedlings = seedling_msgs
        plan_msg.header.frame_id = "map"
        plan_msg.header.stamp = self.get_clock().now().to_msg()

        self.plan_pub.publish(plan_msg)

        print(f"Done in {time.time() - start:.2f} seconds")

    # def generate_plan_with_visuals(self, num_seedlings: int):
    #     """Generate a seedling layout plan with animation. Calls the model once per seedling"""

    #     obs = np.zeros((self.grid_size, self.grid_size, 1), dtype=np.uint8)

    #     # Create figure and axis for the animation
    #     fig, ax = plt.subplots(figsize=(10, 10))
    #     plt.title("Seedling Layout Plan")
    #     ax.set_xlim(0, self.grid_size)
    #     ax.set_ylim(0, self.grid_size)

    #     # Create a dictionary to store scatter plots by species
    #     scatter_plots = {}

    #     # Create a colormap with distinct colors
    #     colors = list(mcolors.TABLEAU_COLORS.values())
    #     # Extend colors if needed for more species
    #     while len(colors) < len(self.species_names):
    #         colors.extend(
    #             list(mcolors.CSS4_COLORS.values())[
    #                 : len(self.species_names) - len(colors)
    #             ]
    #         )

    #     # Animation update function
    #     def update(frame):
    #         if frame >= num_seedlings:
    #             return

    #         action, _ = self.model.predict(obs)

    #         # Convert action to coordinates and species
    #         x = (action[0] + 1) / 2 * self.grid_size
    #         y = (action[1] + 1) / 2 * self.grid_size
    #         x = min(max(x, 0), self.grid_size - 1)
    #         y = min(max(y, 0), self.grid_size - 1)
    #         species_index = int((action[2] + 1) / 2 * (len(self.species_names) - 1))
    #         species = self.species_names[species_index]

    #         if self.keepout[int(x), int(y)] == 1:
    #             print(f"Frame {frame}: Coordinates are in the keepout area.")
    #             return

    #         # Plant a seedling at the specified coordinates
    #         seedling = [x, y, species]
    #         self.seedlings_planted.append(seedling)

    #         # Add the new point to the appropriate scatter plot
    #         if species in scatter_plots:
    #             # Get existing data
    #             x_data, y_data = scatter_plots[species].get_offsets().T
    #             # Append new point
    #             new_offsets = np.vstack((scatter_plots[species].get_offsets(), [x, y]))
    #             scatter_plots[species].set_offsets(new_offsets)
    #         else:
    #             # Create new scatter plot for this species
    #             scatter_plots[species] = ax.scatter(
    #                 x, y, color=colors[species_index % len(colors)], label=species
    #             )

    #         # Update observation
    #         x_index, y_index = int(seedling[0]), int(seedling[1])
    #         species_index = self.species_names.index(seedling[2])
    #         obs[x_index, y_index, 0] = (
    #             species_index + 2
    #         )  # +2 because 0 is empty and 1 is keep out

    #         # Update legend with unique entries
    #         handles, labels = ax.get_legend_handles_labels()
    #         by_label = dict(zip(labels, handles))
    #         ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    #         print(f"Frame {frame}: Planted {species} at ({x:.1f}, {y:.1f})")

    #         return list(scatter_plots.values())

    #     # Create animation
    #     ani = FuncAnimation(
    #         fig,
    #         update,
    #         frames=num_seedlings,
    #         interval=200,  # milliseconds between frames
    #         blit=False,
    #         repeat=False,
    #     )

    #     # Save animation as a GIF
    #     os.makedirs("generation", exist_ok=True)
    #     ani.save("generation/seedling_animation.gif", writer="pillow", fps=5)

    #     # Also save the final state as a PNG
    #     plt.savefig("generation/final_layout.png")
    #     plt.close()


def main(args=None):
    rclpy.init(args=args)

    node = ForestPlanner()

    rclpy.spin(node)

    # Destroy the node explicitly
    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
