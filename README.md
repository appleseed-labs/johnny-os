# Johnny OS
Complete ROS2 codebase for Johnny, your reforestation companion.

Our [project site](https://appleseed-labs.github.io/johnny-os/) includes more information about our group at CMU.

> [!NOTE]
> **Update from 2026: This code is archived.** Johnny OS was
> solo-developed on the [Farm Robotics Challenge](https://farmroboticschallenge.ai/)
> deadline: roughly three months of work while I was balancing grad-school classes at CMU.
> Expect no test coverage, some dead code, and scattered TODOs.
>
> That was a deliberate call. On a competition timeline I chose to spend my hours getting a
> robot to plant *real* seedlings in a *real* field (all of this code runs on hardware, not
> just in sim) rather than on hygiene for what was fundamentally exploratory code. In a
> production setting the priorities invert, and I can tell you exactly what I'd fix first:
>
> 1. **Vendor discipline & dead code**: prune the commented-out subscriptions/handlers and
>    stubbed nodes (e.g. `hole_perception`), and finish wiring the `behavior_management` FSM
>    (several transitions currently only log).
> 2. **Real test coverage**: the existing tests are ROS lint stubs; add behavioral tests for
>    the planners and the FSM state transitions.
> 3. **Observability**: replace `print()` calls with the ROS logger, and add type hints on the
>    node interfaces.
>
> *-Will Heitman (@wheitman)*

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
