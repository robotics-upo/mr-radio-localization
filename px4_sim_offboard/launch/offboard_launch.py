from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    uav_config_file = os.path.join(
        get_package_share_directory('px4_sim_offboard'),
        'config',
        'uav_offboard_params.yaml'
    )

    agv_config_file = os.path.join(
        get_package_share_directory('px4_sim_offboard'),
        'config',
        'agv_offboard_params.yaml'
    )
    

    return LaunchDescription([
        Node(
            package='px4_sim_offboard',
            executable='uav_offboard_control',
            name='uav_offboard_control',
            output='screen',
            parameters=[uav_config_file]
        ),

        Node(
            package='px4_sim_offboard',
            executable='agv_offboard_control',
            name='agv_offboard_control',
            output='screen',
            parameters=[agv_config_file]
        )
    ])