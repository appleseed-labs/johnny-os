#include <rclcpp/rclcpp.hpp>
#include "sensor_processing/lidar_filter.hpp"

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<sensor_processing::LidarFilterNode>();

    // Use a multithreaded executor for better responsiveness
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);

    executor.spin();

    rclcpp::shutdown();
    return 0;
}
