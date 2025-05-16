/**
 * -----------------------------------------------------------------------------
 * Description: A Finite State Machine (FSM) that manages the behavior of the robot
 * Author: Will Heitman
 * (c) 2025 Appleseed Labs. CMU Robotics Institute
 * -----------------------------------------------------------------------------
 */

#include "behavior_management/behavior_manager.hpp"

namespace behavior_management
{

  BehaviorManagerNode::BehaviorManagerNode()
      : Node("behavior_manager_node")
  {
    setUpParameters();

    // Set up subscriptions to receive state transition requests
    on_arrived_sub_ = create_subscription<std_msgs::msg::Empty>(
        "/behavior/on_arrived", 1,
        std::bind(&BehaviorManagerNode::onArrivedCallback, this, std::placeholders::_1));

    on_plan_complete_sub_ = create_subscription<std_msgs::msg::Empty>(
        "/behavior/on_plan_complete", 1,
        std::bind(&BehaviorManagerNode::onPlanCompleteCallback, this, std::placeholders::_1));

    on_drill_complete_sub_ = create_subscription<std_msgs::msg::Empty>(
        "/behavior/on_drill_complete", 1,
        std::bind(&BehaviorManagerNode::onDrillCompleteCallback, this, std::placeholders::_1));

    on_plant_complete_sub_ = create_subscription<std_msgs::msg::Empty>(
        "/behavior/on_plant_complete", 1,
        std::bind(&BehaviorManagerNode::onPlantCompleteCallback, this, std::placeholders::_1));

    on_user_pause_sub_ = create_subscription<std_msgs::msg::Empty>(
        "/behavior/on_user_pause", 1,
        std::bind(&BehaviorManagerNode::onUserPauseCallback, this, std::placeholders::_1));

    on_error_sub_ = create_subscription<std_msgs::msg::Empty>(
        "/behavior/on_error", 1,
        std::bind(&BehaviorManagerNode::onErrorCallback, this, std::placeholders::_1));

    // Set up publishers
    start_drilling_pub_ = create_publisher<std_msgs::msg::Empty>("/behavior/start_drilling", 1);
    start_planting_pub_ = create_publisher<std_msgs::msg::Empty>("/behavior/start_planting", 1);
    start_driving_pub_ = create_publisher<std_msgs::msg::Empty>("/behavior/start_driving", 1);
  }

  void BehaviorManagerNode::setUpParameters()
  {
    // Parameter setup will go here
  }

  void BehaviorManagerNode::onArrivedCallback(const std_msgs::msg::Empty::SharedPtr msg)
  {
    RCLCPP_INFO(get_logger(), "Received on_arrived signal");

    // Here we would check for errors before proceeding...
    RCLCPP_INFO(get_logger(), "Sending start_drilling signal");
    auto start_drilling_msg = std::make_shared<std_msgs::msg::Empty>();
    start_drilling_pub_->publish(*start_drilling_msg);
  }

  void BehaviorManagerNode::onPlanCompleteCallback(const std_msgs::msg::Empty::SharedPtr msg)
  {
    RCLCPP_INFO(get_logger(), "Received on_plan_complete signal");
  }

  void BehaviorManagerNode::onDrillCompleteCallback(const std_msgs::msg::Empty::SharedPtr msg)
  {
    RCLCPP_INFO(get_logger(), "Received on_drill_complete signal");

    // Send the signal to start planting
    RCLCPP_INFO(get_logger(), "Sending start_planting signal");
    auto start_planting_msg = std::make_shared<std_msgs::msg::Empty>();
    start_planting_pub_->publish(*start_planting_msg);
  }

  void BehaviorManagerNode::onPlantCompleteCallback(const std_msgs::msg::Empty::SharedPtr msg)
  {
    RCLCPP_INFO(get_logger(), "Received on_plant_complete signal");
  }

  void BehaviorManagerNode::onUserPauseCallback(const std_msgs::msg::Empty::SharedPtr msg)
  {
    RCLCPP_INFO(get_logger(), "Received on_user_pause signal");
  }

  void BehaviorManagerNode::onErrorCallback(const std_msgs::msg::Empty::SharedPtr msg)
  {
    RCLCPP_INFO(get_logger(), "Received on_error signal");
  }

} // namespace behavior_management
