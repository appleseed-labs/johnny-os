# johnny-os
ROS2 code for our robot

## Installation

This project contains submodules. You *must* clone this repo recursively:

```
$ git clone https://github.com/appleseed-labs/johnny-os.git --recursive

# Now update the submodules
$ git submodule sync
$ git submodule update --init --remote --recursive
```

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