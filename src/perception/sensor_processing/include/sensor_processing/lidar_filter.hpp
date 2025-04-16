#ifndef lidar_filter_HPP_
#define lidar_filter_HPP_

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
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_eigen/tf2_eigen.hpp"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/segmentation/sac_segmentation.h"
#include "pcl/filters/extract_indices.h"
#include "pcl/common/transforms.h"
#include <tf2/transform_datatypes.h>

namespace sensor_processing
{

    class LidarFilterNode : public rclcpp::Node
    {
    public:
        /**
         * @brief Constructor for the LidarToOccupancy node
         */
        explicit LidarFilterNode();

        /**
         * @brief Destructor for the LidarToOccupancy node
         */
        virtual ~LidarFilterNode() = default;

    private:
        /**
         * @brief Callback function for lidar messages
         * @param msg The PointCloud2 message
         */
        void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

        void morphologicalOpening(std::vector<std::vector<float>> &heights,
                                  const std::vector<std::vector<bool>> &occupied,
                                  int window_size);

        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_pointcloud_pub_;
        tf2_ros::Buffer tf_buffer_;
        tf2_ros::TransformListener tf_listener_;

        // Parameters
        double ransac_distance_thresh_; // meters
    };

} // namespace sensor_processing

#endif // lidar_filter_HPP_
