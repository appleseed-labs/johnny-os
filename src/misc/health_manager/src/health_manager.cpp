#include "health_manager/health_manager.hpp"

namespace health_manager
{

    HealthManager::HealthManager()
        : Node("health_manager")
    {
        // Declare parameters with descriptors
        rcl_interfaces::msg::ParameterDescriptor monitored_nodes_desc;
        monitored_nodes_desc.description = "List of nodes to monitor for health status";
        declare_parameter("monitored_nodes", std::vector<std::string>{"controller", "perception", "navigation"}, monitored_nodes_desc);

        rcl_interfaces::msg::ParameterDescriptor critical_nodes_desc;
        critical_nodes_desc.description = "List of nodes that are critical for system operation";
        declare_parameter("critical_nodes", std::vector<std::string>{"controller"}, critical_nodes_desc);

        rcl_interfaces::msg::ParameterDescriptor check_interval_desc;
        check_interval_desc.description = "Interval in seconds between health checks";
        declare_parameter("check_interval", 0.1, check_interval_desc); // seconds

        rcl_interfaces::msg::ParameterDescriptor node_timeout_desc;
        node_timeout_desc.description = "Time in seconds after which a node is considered unresponsive";
        declare_parameter("node_timeout", 0.5, node_timeout_desc); // seconds

        // Get parameters
        monitored_nodes_ = get_parameter("monitored_nodes").as_string_array();
        critical_nodes_ = get_parameter("critical_nodes").as_string_array();
        check_interval_ = get_parameter("check_interval").as_double();
        node_timeout_ = get_parameter("node_timeout").as_double();

        // Initialize state tracking
        system_healthy_ = true;
        for (const auto &node : monitored_nodes_)
        {
            node_last_seen_[node] = 0.0;
            node_status_[node] = false;
        }

        // Create publishers
        global_status_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticStatus>("/diagnostics/global", 10);

        // Create subscription for diagnostics
        auto callback_group = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        auto sub_options = rclcpp::SubscriptionOptions();
        sub_options.callback_group = callback_group;

        diagnostics_sub_ = create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
            "/diagnostics",
            10,
            std::bind(&HealthManager::diagnostics_callback, this, std::placeholders::_1),
            sub_options);

        // Create timer for health checking
        health_timer_ = create_wall_timer(
            std::chrono::duration<double>(check_interval_),
            std::bind(&HealthManager::check_health, this),
            callback_group);

        RCLCPP_INFO(get_logger(), "Health monitor initialized");
    }

    void HealthManager::diagnostics_callback(const diagnostic_msgs::msg::DiagnosticArray::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(state_mutex_);

        for (const auto &status : msg->status)
        {
            // Extract node name from full name
            std::string full_name = status.name;
            size_t pos = full_name.find_last_of('/');
            std::string node_name = (pos != std::string::npos) ? full_name.substr(pos + 1) : full_name;

            // Update node status if it's being monitored
            auto it = node_last_seen_.find(node_name);
            if (it != node_last_seen_.end())
            {
                it->second = now().seconds();
                node_status_[node_name] = (status.level == diagnostic_msgs::msg::DiagnosticStatus::OK);
            }
        }
    }

    void HealthManager::check_health()
    {
        double current_time = now().seconds();
        bool system_healthy = true;

        std::lock_guard<std::mutex> lock(state_mutex_);

        // Check each node's last heartbeat time
        for (const auto &[node, last_seen] : node_last_seen_)
        {
            if (current_time - last_seen > node_timeout_)
            {
                node_status_[node] = false;

                // Check if this is a critical node
                auto it = std::find(critical_nodes_.begin(), critical_nodes_.end(), node);
                if (it != critical_nodes_.end())
                {
                    system_healthy = false;
                    RCLCPP_ERROR(get_logger(), "Critical node %s is not responding", node.c_str());
                }
                else
                {
                    RCLCPP_WARN(get_logger(), "Node %s is not responding", node.c_str());
                }
            }
        }

        // Update system health if it changed
        if (system_healthy != system_healthy_)
        {
            system_healthy_ = system_healthy;
            RCLCPP_INFO(get_logger(), "System health changed to: %s", system_healthy ? "healthy" : "unhealthy");

            // Publish health status
            auto msg = std::make_unique<std_msgs::msg::Bool>();
            msg->data = system_healthy;
            // system_health_pub_->publish(std::move(msg));

            // Take actions based on health status
            if (!system_healthy)
            {
                initiate_safety_protocol();
            }
            else
            {
                restore_normal_operation();
            }
        }
    }

    void HealthManager::initiate_safety_protocol()
    {
        // Implement safety actions here
        RCLCPP_WARN(get_logger(), "Initiating safety protocol");
        // You could publish emergency stop commands here
    }

    void HealthManager::restore_normal_operation()
    {
        // Restore normal operation when system becomes healthy again
        RCLCPP_INFO(get_logger(), "Restoring normal operation");
    }

} // namespace health_manager
