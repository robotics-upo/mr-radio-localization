from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
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

    # Path to ros_gz_bridge's built-in launch file
    ros_gz_bridge_launch_file = os.path.join(
        get_package_share_directory('ros_gz_bridge'),
        'launch',
        'ros_gz_bridge.launch.py'
    )

    uwb_bridge_config = '/home/amarsil/radio_ws/src/mr-radio-localization/uwb_gz_simulation/uwb_bridge.yaml'


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
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='ros_gz_parameter_bridge',
            output='screen',
            arguments=['--ros-args', '-p', f'config_file:={uwb_bridge_config}']
        )
    ])