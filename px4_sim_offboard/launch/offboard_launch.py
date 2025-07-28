from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='px4_sim_offboard',
            executable='uav_offboard_control',
            name='uav_offboard_control',
            output='screen',
            parameters=['config/uav_offboard_params.yaml']
        )
    ])