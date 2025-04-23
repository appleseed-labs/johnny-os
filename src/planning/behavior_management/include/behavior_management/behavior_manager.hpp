#ifndef BEHAVIOR_MANAGEMENT__BEHAVIOR_MANAGER_HPP_
#define BEHAVIOR_MANAGEMENT__BEHAVIOR_MANAGER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>

namespace behavior_management
{

/**
 * @brief A Finite State Machine (FSM) that manages the behavior of the robot
 */
class BehaviorManagerNode : public rclcpp::Node
{
public:
  /**
   * @brief Constructor for BehaviorManagerNode
   */
  BehaviorManagerNode();

private:
  /**
   * @brief Set up ROS parameters
   */
  void setUpParameters();

  // Callback methods for state transitions
  void onArrivedCallback(const std_msgs::msg::Empty::SharedPtr msg);
  void onPlanCompleteCallback(const std_msgs::msg::Empty::SharedPtr msg);
  void onDrillCompleteCallback(const std_msgs::msg::Empty::SharedPtr msg);
  void onPlantCompleteCallback(const std_msgs::msg::Empty::SharedPtr msg);
  void onUserPauseCallback(const std_msgs::msg::Empty::SharedPtr msg);
  void onErrorCallback(const std_msgs::msg::Empty::SharedPtr msg);

  // Subscription handles
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr on_arrived_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr on_plan_complete_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr on_drill_complete_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr on_plant_complete_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr on_user_pause_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr on_error_sub_;
};

}  // namespace behavior_management

#endif  // BEHAVIOR_MANAGEMENT__BEHAVIOR_MANAGER_HPP_
