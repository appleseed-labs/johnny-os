cmake_minimum_required(VERSION 3.8)
project(grid_generation)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies automatically
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Include directories
include_directories(include)

# Create a library containing the implementation
ament_auto_add_library(${PROJECT_NAME}_lib
  src/lidar_to_occupancy.cpp
)

# Build the executable
ament_auto_add_executable(lidar_to_occupancy
  src/lidar_to_occupancy_node.cpp
)
target_link_libraries(lidar_to_occupancy ${PROJECT_NAME}_lib)

# Install directories
install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

# Generate the package
ament_auto_package()
