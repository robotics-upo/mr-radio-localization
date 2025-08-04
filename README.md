# mr-radio-localization

## Overview
This repository contains the developed code for the Master's Thesis "Radio-Based Multi-robot Odometry and Localization", the final project of the Master's Program in Robotics, Graphics and Computer Vision of University of Zaragoza.

### Abstract
Deployment of multi-robot teams in cooperative missions, like search and rescue, or exploration of large and dangerous environments (i.e. tunnels, mines, or rubble resulting from an accident), bring many unique challenges that are pushing research in exciting new ways. Particularly, localization is a crucial component which typically relies on cameras or LiDARs and underperforms in presence of poor lighting or dust in interiors, and adverse weather conditions in exteriors. In this context, radio-based methods such as Ultra-Wideband (UWB) and RADAR, which have been traditionally undersubscribed in robotics, are experiencing a boost in popularity thanks to their robustness to harsh environmental conditions and cluttered environments. Hence, this work proposes a multi-robot localization system that leverages the two technologies with inexpensive and readily-available sensors (IMU and wheel encoders) to estimate the position of a ground robot and an aerial robot with respect to a global reference frame, by using a pose-graph optimization framework with inter-robot constraints. The system has been developed for ROS2 using the Ceres optimizer, and has shown promising results in simulated conditions and on a real-world dataset. Furthermore, the standard factor graph formulation also makes it easily extensible to a full SLAM problem. 

## Dependencies

* [ROS2 Humble](https://docs.ros.org/en/humble/index.html)
* [Ceres Solver](https://github.com/ceres-solver/ceres-solver)
* [Sophus](https://github.com/strasdat/Sophus)
* [pcl_ros](https://github.com/ros-perception/perception_pcl)
* [small_gicp](https://github.com/koide3/small_gicp)
* [eliko_ros](https://github.com/robotics-upo/eliko_ros)
* [ars548_ros](https://github.com/robotics-upo/ars548_ros)
* [4D-Radar-Odom](https://github.com/robotics-upo/4D-Radar-Odom/tree/arco-drone-integration) branch ```arco_drone_integration```.

Clone this repository along with the dependency packages to your ROS 2 workspace and compile with the standard ```colcon build``` command.

## Main components

This repository contains two ROS2 packages:

* ```uwb_localization```: includes the UWB-based relative transformation estimation node and the pose-graph optimization node with radar constraints. The ```config``` folder in this package contains the parameter file for these two nodes.

![](images/TFM_architecture.drawio.png)

* ```uwb_simulator```: includes the odometry simulation node and the measurement simulation node.  The ```config``` folder in this package contains the parameter file for these two nodes.

![](images/TFM_diagram_simulation.drawio.png)


## Launch files

```uwb_localization``` contains two launch files. To launch the real world dataset experiment (which includes radar odometry), type:
``` 
ros2 launch uwb_localization localization.launch.py
```
**Note**: the real-world dataset has not yet been made public, but it will be made available soon. In the meantime, you can try the simulated version. 

To launch the simulated scenario with just UWB and generic odometry, type:
``` 
ros2 launch uwb_localization localization_sim.launch.py

```

## PX4 SITL Simulator

This package includes an enhanced simulator for relative localization which is integrated with [PX4](https://docs.px4.io/main/en/simulation/) Software In The Loop, which supports multi-vehicle simulation with Gazebo and ROS 2. We provide the following simulation tools:

* ```uwb_gz_simulation``` includes a ```models``` folder with modified versions the differential rover ```r1_rover``` and the ```x500``` UAV with UWB anchors and tags mounted onboard each respective platform, which act as drop-in replacements for the existing ones. Reference for the original models can be found [here](https://docs.px4.io/main/en/sim_gazebo_gz/vehicles.html). The folder ```uwb_gazebo_plugin``` contains a custom plugin that reports distances between each anchor and tag, which is meant to be included under the plugins directory of PX4-Autopilot. 

* ```px4_sim_offboard``` includes a set of nodes that interact with the simulator, allowing to obtain sensor readings and input commands to each of the vehicles. It includes a simple trajectory tracker for each of the robots. It also parses messages from ```px4_msgs``` format to standard ROS formats, for better integration with the optimizer. 

### Setup instructions (Ubuntu 24.04)

1) Install [ROS2](https://docs.ros.org/en/jazzy/index.html) Jazzy 

2) Install [Gazebo](https://gazebosim.org/docs/harmonic/ros_installation/) Harmonic.

3) Download [QGC](https://docs.qgroundcontrol.com/master/en/qgc-user-guide/releases/daily_builds.html) Daily Build.

4) Install the PX4 [Toolchain](https://docs.px4.io/main/en/dev_setup/dev_env_linux_ubuntu.html) for Ubuntu. 

5) Set up Micro [XRCE-DDS](https://docs.px4.io/main/en/ros2/user_guide.html#setup-micro-xrce-dds-agent-client) Agent & Client for PX4-ROS2 communication.

6) Build and run ROS2 [Workspace](https://docs.px4.io/main/en/ros2/user_guide.html#build-ros-2-workspace). To check that everything is working, we strongly encourage to also test the [multi-vehicle](https://docs.px4.io/main/en/sim_gazebo_gz/multi_vehicle_simulation.html) simulation example with ROS2 and Gazebo.

7) Copy the contents of the ```models``` folder in ```uwb_gz_simulation``` into ```/path/to/PX4-Autopilot/Tools/simulation/gz/models```

8) Add the custom plugin (steps taken from [template](https://github.com/PX4/PX4-Autopilot/tree/main/src/modules/simulation/gz_plugins/template_plugin) plugin instructions) 
    
    8.1: Copy the folder ```uwb_gazebo_plugin``` into ```/path/to/PX4-Autopilot/modules/simulation/gz_plugins```, and include the plugin for compilation by adding the following lines to the top-level```CMakeLists.txt```. 

```cmake
    add_subdirectory(uwb_gazebo_plugin)
    add_custom_target(px4_gz_plugins ALL DEPENDS OpticalFlowSystem MovingPlatformController TemplatePlugin GenericMotorModelPlugin BuoyancySystemPlugin SpacecraftThrusterModelPlugin UWBGazeboPlugin)
```
    8.2: Then, load the plugin by including this line in ```/path/to/PX4-Autopilot/src/modules/simulation/gz_bridge/server.config```.

```xml
<plugin entity_name="*" entity_type="world" filename="libUWBGazeboPlugin.so" name="custom::UWBGazeboSystem"/>
```

9) Build the code after adding the plugin: 
```
cd /path/to/PX4-Autopilot
make px4_sitl
```

10) Install [tmux](https://github.com/tmux/tmux/wiki/Installing) 

11) Update ```simulator_launcher.sh``` with the paths to your ROS 2 workspace, your PX4-Autopilot installation folder and the location of the QGC executable. By default, the script assumes that PX4 and the ROS 2 ws are on the root folder, and QGC is in ```~/Desktop```. 

12) Give permissions to the simulator script and launch the simulator: 

```
cd <ros2_ws>/mr-radio-localization
sudo chmod +x simulator_launcher.sh
./simulator_launcher.sh
```

Note that the simulator takes a while to load. After about 30 seconds, you should see the two robots start to move. 
