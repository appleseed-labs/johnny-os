cmake_minimum_required(VERSION 3.8)
project(behavior_management)

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
  src/behavior_manager.cpp
)

# Build the executable
ament_auto_add_executable(behavior_manager
  src/behavior_manager_node.cpp
)
target_link_libraries(behavior_manager ${PROJECT_NAME}_lib)

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
