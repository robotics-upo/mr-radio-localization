#!/bin/bash

PX4_DIR=~/PX4-Autopilot
ROS_WS=~/radio_ws
QGC_PATH=~/Desktop/QGroundControl-x86_64.AppImage

SESSION="sim_session"

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

tmux send-keys -t $SESSION:0.0 "chmod +x $QGC_PATH && $QGC_PATH" C-m
tmux send-keys -t $SESSION:0.1 "MicroXRCEAgent udp4 -p 8888" C-m
tmux send-keys -t $SESSION:0.2 "sleep 2 && cd $PX4_DIR && PX4_SYS_AUTOSTART=4001 PX4_SIM_MODEL=gz_x500 ./build/px4_sitl_default/bin/px4 -i 1" C-m
tmux send-keys -t $SESSION:0.3 "sleep 4 && cd $PX4_DIR && PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4009 PX4_GZ_MODEL_POSE='0,1' PX4_SIM_MODEL=gz_r1_rover PX4_GZ_WORLD=default ./build/px4_sitl_default/bin/px4 -i 2" C-m
tmux send-keys -t $SESSION:0.4 "sleep 25 && cd $ROS_WS && source install/setup.zsh && ros2 launch px4_sim_offboard offboard_launch.py" C-m
tmux send-keys -t $SESSION:0.5 "sleep 27 && source $ROS_WS/install/setup.zsh && ros2 topic echo /eliko/Distances" C-m

#odometry window
tmux new-window -t $SESSION:1 -n odom_echo
tmux split-window -h -t $SESSION:1.0 

tmux send-keys -t $SESSION:1.0 "sleep 27 && source $ROS_WS/install/setup.zsh && ros2 topic echo /agv/odom" C-m

tmux send-keys -t $SESSION:1.1 "sleep 27 && source $ROS_WS/install/setup.zsh && ros2 topic echo /uav/odom" C-m

# Attach
tmux attach-session -t $SESSION
