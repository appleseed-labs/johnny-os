// include/health_manager/health_manager.hpp
#ifndef health_manager_HPP_
#define health_manager_HPP_

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "std_msgs/msg/bool.hpp"

namespace health_manager
{

    class HealthMonitor : public rclcpp::Node
    {
    public:
        /**
         * @brief Constructor for the HealthMonitor node
         */
        explicit HealthMonitor();

        /**
         * @brief Destructor for the HealthMonitor node
         */
        virtual ~HealthMonitor() = default;

    private:
        /**
         * @brief Callback function for diagnostics messages
         * @param msg The diagnostic array message
         */
        void diagnostics_callback(const diagnostic_msgs::msg::DiagnosticArray::SharedPtr msg);

        /**
         * @brief Check health status of monitored nodes
         */
        void check_health();

        /**
         * @brief Actions to take when system becomes unhealthy
         */
        void initiate_safety_protocol();

        /**
         * @brief Actions to take when system returns to healthy state
         */
        void restore_normal_operation();

        // Publishers
        rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticStatus>::SharedPtr global_status_pub_;

        // Subscribers
        rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_sub_;

        // Timers
        rclcpp::TimerBase::SharedPtr health_timer_;

        // Parameters
        std::vector<std::string> monitored_nodes_;
        std::vector<std::string> critical_nodes_;
        double check_interval_;
        double node_timeout_;

        // State variables
        std::unordered_map<std::string, double> node_last_seen_;
        std::unordered_map<std::string, bool> node_status_;
        bool system_healthy_;

        // Mutex for thread safety
        std::mutex state_mutex_;
    };

} // namespace health_manager

#endif // health_manager_HPP_
