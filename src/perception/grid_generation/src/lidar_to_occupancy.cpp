#include "grid_generation/lidar_to_occupancy.hpp"

namespace grid_generation
{

    LidarToOccupancyNode::LidarToOccupancyNode()
        : Node("lidar_to_occupancy"),
          tf_buffer_(this->get_clock()),
          tf_listener_(tf_buffer_)
    {

        // Declare parameters with descriptors
        rcl_interfaces::msg::ParameterDescriptor grid_size_cells_desc;
        grid_size_cells_desc.description = "The size of the square occupancy grid in cells";
        declare_parameter("grid_size_cells", 100, grid_size_cells_desc);

        rcl_interfaces::msg::ParameterDescriptor grid_resolution;
        grid_resolution.description = "The resolution of the occupancy grid in meters per cell";
        declare_parameter("grid_resolution", 0.2, grid_resolution);

        // Get parameters
        grid_size_cells_ = get_parameter("grid_size_cells").as_int();
        grid_resolution_ = get_parameter("grid_resolution").as_double();

        // Subscriber to PointCloud2
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points", 10,
            std::bind(&LidarToOccupancyNode::pointCloudCallback, this, std::placeholders::_1));

        // Publisher for OccupancyGrid
        occupancy_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/planning/occupancy", 10);

        RCLCPP_INFO(this->get_logger(), "LidarToOccupancyNode started.");
    }

    void LidarToOccupancyNode::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Lookup transform from lidar_link to map frame
        geometry_msgs::msg::TransformStamped lidar_to_map_transform;
        try
        {
            lidar_to_map_transform = tf_buffer_.lookupTransform("map", "lidar_link", tf2::TimePointZero);
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_WARN(this->get_logger(), "Could not transform lidar_link to map: %s", ex.what());
            return;
        }

        // Lookup transform from map to base_link frame
        geometry_msgs::msg::TransformStamped base_link_to_map_transform;
        try
        {
            base_link_to_map_transform = tf_buffer_.lookupTransform("map", "base_link", tf2::TimePointZero);
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_WARN(this->get_logger(), "Could not transform base_link to map: %s", ex.what());
            return;
        }

        // Convert PointCloud2 to PCL point cloud
        pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
        pcl::fromROSMsg(*msg, pcl_cloud);

        // Transform points to map frame
        pcl::PointCloud<pcl::PointXYZ> transformed_cloud;
        for (const auto &point : pcl_cloud)
        {
            geometry_msgs::msg::PointStamped point_in, point_out;
            point_in.header.frame_id = "lidar_link";
            point_in.point.x = point.x;
            point_in.point.y = point.y;
            point_in.point.z = point.z;

            tf2::doTransform(point_in, point_out, lidar_to_map_transform);

            pcl::PointXYZ transformed_point;
            transformed_point.x = point_out.point.x;
            transformed_point.y = point_out.point.y;
            transformed_point.z = point_out.point.z;
            transformed_cloud.push_back(transformed_point);
        }

        // Create OccupancyGrid
        auto occupancy_grid = std::make_shared<nav_msgs::msg::OccupancyGrid>();
        occupancy_grid->header.stamp = this->now();
        occupancy_grid->header.frame_id = "map";

        // Set grid parameters
        occupancy_grid->info.resolution = 0.2; // 20 cm per cell
        occupancy_grid->info.width = 100;      // 10m x 10m grid
        occupancy_grid->info.height = 100;

        // Center grid at base_link origin in map frame
        occupancy_grid->info.origin.position.x = base_link_to_map_transform.transform.translation.x - occupancy_grid->info.width * occupancy_grid->info.resolution / 2.0;
        occupancy_grid->info.origin.position.y = base_link_to_map_transform.transform.translation.y - occupancy_grid->info.height * occupancy_grid->info.resolution / 2.0;
        occupancy_grid->info.origin.position.z = 0.0;

        // Initialize grid data
        occupancy_grid->data.resize(occupancy_grid->info.width * occupancy_grid->info.height, -1);

        // Populate grid with transformed points
        for (const auto &point : transformed_cloud)
        {
            int x_idx = static_cast<int>((point.x - occupancy_grid->info.origin.position.x) / occupancy_grid->info.resolution);
            int y_idx = static_cast<int>((point.y - occupancy_grid->info.origin.position.y) / occupancy_grid->info.resolution);

            if (x_idx >= 0 && x_idx < static_cast<int>(occupancy_grid->info.width) &&
                y_idx >= 0 && y_idx < static_cast<int>(occupancy_grid->info.height))
            {
                int index = y_idx * occupancy_grid->info.width + x_idx;
                occupancy_grid->data[index] = 100; // Mark cell as occupied
            }
        }

        // Publish OccupancyGrid
        occupancy_pub_->publish(*occupancy_grid);
    }

}
