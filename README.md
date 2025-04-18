# johnny-os
ROS2 code for our robot

## Installation

This project contains submodules. You *must* clone this repo recursively:

```
$ git clone https://github.com/appleseed-labs/johnny-os.git --recursive
$ cd johnny-os

# Now update the submodules
$ git submodule sync
$ git submodule update --init --remote --recursive

# Set up pre-commit (enforces formatting and more)
$ pip install pre-commit  # If not already installed
$ pre-commit install      # Now checks will run automatically at git commit

# Install any dependencies with rosdep (you may need to run "sudo rosdep init" first)
$ rosdep update && rosdep install --from-paths src --ignore-src -r -y

$ colcon build  # Build with colcon
$ . install/setup.bash  # Source your workspace
```

## Running an example with EcoSim
First, ensure that EcoSim is running. See the [docs](https://wheitman.github.io/ecosim/) for instructions.

Now run the example launch file:

```bash
$ cd johnny-os
$ . install/setup.bash
$ ros2 launch launch/ecosim_base.launch.py
```

An Rviz2 window should open, showing sensor data.

## Folder structure

```
description/    # URDF, Xacro, and meshes
launch/         # ROS launch files
param/          # ROS *.param.yaml files
src/            # Source code
    external/   # Third-party code. Ideally git submodules.
    interfaces/ # Custom, hardware-specific code and simulation utils.
    perception/ # Sensor filters, classifiers, cost map generators, etc.
    planning/   # High- and low-level motion planning
```
