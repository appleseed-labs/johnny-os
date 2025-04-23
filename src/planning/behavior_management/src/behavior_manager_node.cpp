#include <rclcpp/rclcpp.hpp>
#include "behavior_management/behavior_manager.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<behavior_management::BehaviorManagerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
