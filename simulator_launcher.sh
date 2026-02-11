#!/bin/bash

PX4_DIR=~/PX4-Autopilot
ROS_WS=~/radio_ws
QGC_PATH=~/Desktop/QGroundControl-x86_64.AppImage

# # # Robot base poses: [x, y, z, roll, pitch, yaw]
# # agv_pose = [13.351, 27.385, 0.636, 0.001, 0.001, 3.079]
# # uav_pose = [15.938, 22.979, 0.903, 0.000, 0.000, 0.028]

# UAV initial pose (x, y, z, roll, pitch, yaw)
UAV_X=0.5
UAV_Y=-0.5
UAV_Z=0.0
UAV_ROLL=0

UAV_PITCH=0
UAV_YAW=0.524

# Rover initial pose (x, y, z, roll, pitch, yaw)
ROVER_X=0.0
ROVER_Y=0.0
ROVER_Z=0.0
ROVER_ROLL=0
ROVER_PITCH=0
ROVER_YAW=0

SESSION="sim_session"

graceful_shutdown() {
  echo "[CLEANUP] Graceful shutdown…"

  # 1) Stop ros2 bag cleanly (send Ctrl-C to pane 1.3)
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[CLEANUP] Stopping ros2 bag…"
    tmux send-keys -t $SESSION:1.3 C-c
    # 2) Stop your other ROS nodes cleanly (optional)
    tmux send-keys -t $SESSION:1.2 C-c  # uwb_localization
    tmux send-keys -t $SESSION:0.4 C-c  # offboard_launch.py
    sleep 10

    # 3) Now kill simulators last
    echo "[CLEANUP] Killing simulators…"
    killall -9 gzserver gzclient ruby 2>/dev/null || true

    # 4) Finally, kill the tmux session
    tmux kill-session -t "$SESSION" 2>/dev/null || true
  fi
}

trap graceful_shutdown EXIT INT TERM

# Kill any previous session
tmux kill-session -t $SESSION 2>/dev/null

# Start new tmux session
tmux new-session -d -s $SESSION

tmux split-window -h -t $SESSION:0.0         
tmux split-window -h -t $SESSION:0.1         

tmux split-window -v -t $SESSION:0.0         
tmux split-window -v -t $SESSION:0.1        
tmux split-window -v -t $SESSION:0.2         

tmux select-layout -t $SESSION:0 tiled

# Set up QGroundControl
tmux send-keys -t $SESSION:0.0 "chmod +x $QGC_PATH && $QGC_PATH" C-m
# Set up Micro XRCE-DDS Agent
tmux send-keys -t $SESSION:0.1 "MicroXRCEAgent udp4 -p 8888" C-m
# UAV (gz_x500)
tmux send-keys -t $SESSION:0.2 "sleep 2 && cd $PX4_DIR && PX4_GZ_MODEL_POSE='${UAV_X},${UAV_Y},${UAV_Z},${UAV_ROLL},${UAV_PITCH},${UAV_YAW}' PX4_SYS_AUTOSTART=4001 PX4_SIM_MODEL=gz_x500 ./build/px4_sitl_default/bin/px4 -i 1" C-m
# Rover (gz_r1_rover)
tmux send-keys -t $SESSION:0.3 "sleep 4 && cd $PX4_DIR && PX4_GZ_STANDALONE=1 PX4_GZ_MODEL_POSE='${ROVER_X},${ROVER_Y},${ROVER_Z},${ROVER_ROLL},${ROVER_PITCH},${ROVER_YAW}' PX4_SYS_AUTOSTART=4009 PX4_SIM_MODEL=gz_r1_rover PX4_GZ_WORLD=default ./build/px4_sitl_default/bin/px4 -i 2" C-m

# Wait for everything to launch and spawn and then launch the ROS 2 nodes
tmux send-keys -t $SESSION:0.4 "sleep 25 && cd $ROS_WS && source install/setup.zsh && ros2 launch px4_sim_offboard offboard_launch.py" C-m
tmux send-keys -t $SESSION:0.5 "sleep 27 && cd $ROS_WS && source install/setup.zsh && ros2 topic echo /eliko/Distances" C-m

#odometry window
tmux new-window -t $SESSION:1 -n optimization
tmux split-window -h -t $SESSION:1.0 

# Add two more panes:
tmux split-window -v -t $SESSION:1.0    # Create pane 1.2 from 1.0
tmux split-window -v -t $SESSION:1.1    # Create pane 1.3 from 1.1

tmux select-layout -t $SESSION:1 tiled


# Echo each vehicles odometry topic
tmux send-keys -t $SESSION:1.0 "sleep 27 && source $ROS_WS/install/setup.zsh && ros2 topic echo /agv/odom" C-m
tmux send-keys -t $SESSION:1.1 "sleep 27 && source $ROS_WS/install/setup.zsh && ros2 topic echo /uav/odom" C-m

# UWB localization launch (to simulate and optimize at the same time)
tmux send-keys -t $SESSION:1.2 "sleep 30 && cd $ROS_WS && source install/setup.zsh && ros2 launch uwb_localization localization.launch.py" C-m

# #Record all ROS 2 topics (to simulate and optimize at the same tim)
# tmux send-keys -t $SESSION:1.3 "sleep 30 && cd $ROS_WS && source install/setup.zsh && ros2 bag record \
#   /uav/gt \
#   /agv/gt \
#   /uav/odom \
#   /agv/odom \
#   /eliko_optimization_node/optimized_T \
#   /eliko_optimization_node/optimized_T_nopr \
#   /eliko_optimization_node/ransac_optimized_T \
#   /pose_graph_node/uav_anchor \
#   /pose_graph_node/agv_anchor" C-m

#Record all ROS 2 topics (Post-processing version)
tmux send-keys -t $SESSION:1.3 "sleep 30 && cd $ROS_WS && source install/setup.zsh && ros2 bag record \
  /uav/gt \
  /agv/gt \
  /uav/odom \
  /agv/odom \
  /eliko/Distances " C-m


# Attach
tmux attach-session -t $SESSION


