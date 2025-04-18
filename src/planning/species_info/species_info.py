import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import json
import os
import geopandas as gpd

class SpeciesInfoNode(Node):
    def __init__(self):
        super().__init__('species_info_node')

        self.subscription = self.create_subscription(
            String,
            '/planning/bounds_geojson',
            self.geojson_callback,
            10
        )

        self.publisher = self.create_publisher(String, '/planning/species_info', 10)

        self.r_script_path = "/src/planning/species_info/speciesplanner.r"
        self.get_logger().info("Species Info Node Initialized.")

    def geojson_callback(self, msg):
        self.get_logger().info(f"Received GeoJSON: {msg.data}")

        geojson_data = json.loads(msg.data)

        geojson_polygon = geojson_data['features'][0]['geometry']

        with open("/tmp/polygon.geojson", 'w') as f:
            json.dump(geojson_data, f)

        # rscript here
        try:
            result = subprocess.run(
                ['Rscript', self.r_script_path, '/tmp/polygon.geojson'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                species_info = result.stdout
                #DISTANCE IS IN METERS
                self.publish_species_info(species_info)
            else:
                self.get_logger().error(f"R script failed: {result.stderr}")
        except Exception as e:
            self.get_logger().error(f"Error running R script: {e}")

    def publish_species_info(self, info):
        msg = String()
        msg.data = info
        self.publisher.publish(msg)
        self.get_logger().info(f"Published species info: {info}")


def main(args=None):
    rclpy.init(args=args)

    species_info_node = SpeciesInfoNode()

    rclpy.spin(species_info_node)

    # Cleanup
    species_info_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
