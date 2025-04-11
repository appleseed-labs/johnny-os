#include <rclcpp/rclcpp.hpp>
#include "health_manager/health_manager.hpp"

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<health_manager::HealthMonitor>();

    // Use a multithreaded executor for better responsiveness
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);

    executor.spin();

    rclcpp::shutdown();
    return 0;
}
