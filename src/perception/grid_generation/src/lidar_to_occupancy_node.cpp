#include <rclcpp/rclcpp.hpp>
#include "grid_generation/lidar_to_occupancy.hpp"

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<grid_generation::LidarToOccupancyNode>();

    // Use a multithreaded executor for better responsiveness
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);

    executor.spin();

    rclcpp::shutdown();
    return 0;
}
