#include "sensor_processing/lidar_filter.hpp"

namespace sensor_processing
{

    LidarFilterNode::LidarFilterNode()
        : Node("lidar_filter"),
          tf_buffer_(this->get_clock()),
          tf_listener_(tf_buffer_)
    {

        // Declare parameters with descriptors
        rcl_interfaces::msg::ParameterDescriptor ransac_distance_thresh_desc;
        ransac_distance_thresh_desc.description = "Distance threshold for RANSAC segmentation in meters";
        declare_parameter("ransac_distance_thresh", 0.5, ransac_distance_thresh_desc);
        declare_parameter("grid_size_cells", 100);
        declare_parameter("grid_resolution", 0.2);

        // Get parameters
        ransac_distance_thresh_ = get_parameter("ransac_distance_thresh").as_double();

        // Subscriber to PointCloud2
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points", 10,
            std::bind(&LidarFilterNode::pointCloudCallback, this, std::placeholders::_1));

        // Publisher for OccupancyGrid
        filtered_pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lidar/filtered", 10);

        RCLCPP_INFO(this->get_logger(), "LidarFilterNode started.");
    }

    // Remove ground points using a Progressive Morphological Filter (PMF)
    // This uses a 2.5D grid to project points and then applies a morphological opening
    // to remove ground points iteratively.
    // The filter is designed to work with a 2D grid of height values, where each cell
    // contains the minimum height of points within that cell.
    // The filter uses a window size to determine the neighborhood for morphological operations.
    // The window size is increased progressively to adapt to the terrain.
    // This works better than RANSAC for non-flat terrain and can handle varying ground heights.
    void LidarFilterNode::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Lookup transform from the message's frame to the base_link frame
        geometry_msgs::msg::TransformStamped transform_stamped;
        try
        {
            transform_stamped = tf_buffer_.lookupTransform(
                "base_link", msg->header.frame_id,
                tf2::TimePointZero);
        }
        catch (tf2::TransformException &ex)
        {
            RCLCPP_WARN(this->get_logger(), "Could not transform pointcloud: %s", ex.what());
            return;
        }

        // Convert the PointCloud2 to PCL
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*msg, pcl_pc2);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromPCLPointCloud2(pcl_pc2, *cloud);

        // Transform points to base_link
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        Eigen::Affine3d transform;
        // tf2::fromMsg(transform_stamped.transform, transform);
        transform = tf2::transformToEigen(transform_stamped);
        pcl::transformPointCloud(*cloud, *transformed_cloud, transform);

        // 1. Project points to a 2.5D grid
        float grid_resolution = this->get_parameter("grid_resolution").as_double();
        int grid_size = this->get_parameter("grid_size_cells").as_int();

        // Create a height grid
        std::vector<std::vector<float>> min_heights(grid_size, std::vector<float>(grid_size, std::numeric_limits<float>::max()));
        std::vector<std::vector<bool>> grid_occupied(grid_size, std::vector<bool>(grid_size, false));

        // Project points to 2D grid and store minimum height in each cell
        float grid_origin_x = -grid_size * grid_resolution / 2.0;
        float grid_origin_y = -grid_size * grid_resolution / 2.0;

        for (const auto &point : transformed_cloud->points)
        {
            // Skip points outside grid bounds
            if (std::abs(point.x) > grid_size * grid_resolution / 2.0 ||
                std::abs(point.y) > grid_size * grid_resolution / 2.0)
            {
                continue;
            }

            int grid_x = static_cast<int>((point.x - grid_origin_x) / grid_resolution);
            int grid_y = static_cast<int>((point.y - grid_origin_y) / grid_resolution);

            if (grid_x >= 0 && grid_x < grid_size && grid_y >= 0 && grid_y < grid_size)
            {
                if (point.z < min_heights[grid_x][grid_y])
                {
                    min_heights[grid_x][grid_y] = point.z;
                    grid_occupied[grid_x][grid_y] = true;
                }
            }
        }

        // 2. Perform morphological opening with increasing window size
        const int max_window_size = 9;       // Adjust based on your terrain complexity
        const float height_threshold = 0.3f; // Adjust based on your terrain slope

        std::vector<std::vector<float>> ground_heights = min_heights;

        // Progressive morphological filtering
        for (int window_size = 3; window_size <= max_window_size; window_size += 2)
        {
            // Apply morphological opening (erosion followed by dilation)
            morphologicalOpening(ground_heights, grid_occupied, window_size);

            // Update ground model
            for (int x = 0; x < grid_size; x++)
            {
                for (int y = 0; y < grid_size; y++)
                {
                    if (grid_occupied[x][y] && (min_heights[x][y] - ground_heights[x][y]) > height_threshold * window_size / max_window_size)
                    {
                        // Mark as non-ground if height difference exceeds threshold
                        grid_occupied[x][y] = false;
                    }
                }
            }
        }

        // 3. Filter points based on ground model
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);

        for (const auto &point : transformed_cloud->points)
        {
            int grid_x = static_cast<int>((point.x - grid_origin_x) / grid_resolution);
            int grid_y = static_cast<int>((point.y - grid_origin_y) / grid_resolution);

            bool is_ground = false;
            if (grid_x >= 0 && grid_x < grid_size && grid_y >= 0 && grid_y < grid_size)
            {
                if (grid_occupied[grid_x][grid_y] &&
                    std::abs(point.z - ground_heights[grid_x][grid_y]) < height_threshold)
                {
                    is_ground = true;
                }
            }

            // Keep non-ground points
            if (!is_ground)
            {
                filtered_cloud->points.push_back(point);
            }
        }

        filtered_cloud->width = filtered_cloud->points.size();
        filtered_cloud->height = 1;
        filtered_cloud->is_dense = false;

        // Publish filtered cloud
        sensor_msgs::msg::PointCloud2 filtered_msg;
        pcl::toROSMsg(*filtered_cloud, filtered_msg);
        filtered_msg.header = msg->header;
        filtered_msg.header.frame_id = "base_link";

        filtered_pointcloud_pub_->publish(filtered_msg);
    }

    void LidarFilterNode::morphologicalOpening(std::vector<std::vector<float>> &heights,
                                               const std::vector<std::vector<bool>> &occupied,
                                               int window_size)
    {
        int grid_size = heights.size();
        std::vector<std::vector<float>> temp = heights;

        // Erosion
        for (int x = 0; x < grid_size; x++)
        {
            for (int y = 0; y < grid_size; y++)
            {
                if (!occupied[x][y])
                    continue;

                float min_val = std::numeric_limits<float>::max();
                int half_window = window_size / 2;

                for (int wx = -half_window; wx <= half_window; wx++)
                {
                    for (int wy = -half_window; wy <= half_window; wy++)
                    {
                        int nx = x + wx;
                        int ny = y + wy;

                        if (nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size && occupied[nx][ny])
                        {
                            min_val = std::min(min_val, heights[nx][ny]);
                        }
                    }
                }

                if (min_val != std::numeric_limits<float>::max())
                {
                    temp[x][y] = min_val;
                }
            }
        }

        heights = temp;

        // Dilation
        for (int x = 0; x < grid_size; x++)
        {
            for (int y = 0; y < grid_size; y++)
            {
                if (!occupied[x][y])
                    continue;

                float max_val = -std::numeric_limits<float>::max();
                int half_window = window_size / 2;

                for (int wx = -half_window; wx <= half_window; wx++)
                {
                    for (int wy = -half_window; wy <= half_window; wy++)
                    {
                        int nx = x + wx;
                        int ny = y + wy;

                        if (nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size && occupied[nx][ny])
                        {
                            max_val = std::max(max_val, temp[nx][ny]);
                        }
                    }
                }

                if (max_val != -std::numeric_limits<float>::max())
                {
                    heights[x][y] = max_val;
                }
            }
        }
    }

}
