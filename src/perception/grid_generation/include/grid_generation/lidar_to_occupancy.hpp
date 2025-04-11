#ifndef lidar_to_occupancy_HPP_
#define lidar_to_occupancy_HPP_

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/bool.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace grid_generation
{

    class LidarToOccupancyNode : public rclcpp::Node
    {
    public:
        /**
         * @brief Constructor for the LidarToOccupancy node
         */
        explicit LidarToOccupancyNode();

        /**
         * @brief Destructor for the LidarToOccupancy node
         */
        virtual ~LidarToOccupancyNode() = default;

    private:
        /**
         * @brief Callback function for lidar messages
         * @param msg The PointCloud2 message
         */
        void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
        rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_pub_;
        tf2_ros::Buffer tf_buffer_;
        tf2_ros::TransformListener tf_listener_;

        // Parameters
        int grid_size_cells_;
        double grid_resolution_; // meters per cell
    };

} // namespace grid_generation

#endif // lidar_to_occupancy_HPP_
