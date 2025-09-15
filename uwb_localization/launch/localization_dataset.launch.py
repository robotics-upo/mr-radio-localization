import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import FrontendLaunchDescriptionSource
from launch.actions import ExecuteProcess


def generate_launch_description():

    package_dir = get_package_share_directory('uwb_localization')

    config = os.path.join(package_dir, 'config', 'params.yaml')

    #Put here the path to the dataset bag you want to use
    path_to_bag = "/home/amarsil/radio_ws/dataset/experiment1_gt_v2"

    node1 = Node(
                package='uwb_localization',
                executable='uav_odometry_node',
                name='uav_odometry_node',
                parameters=[config]
    )

    node2 = Node(
                package='uwb_localization',
                executable='agv_odometry_node',
                name='agv_odometry_node',
                parameters=[config]
    )
           
    node3 = Node(
                package='uwb_localization',
                executable='global_opt_node_eliko',
                name='eliko_global_opt_node',
                parameters=[config]
                )
                
    node4 = Node(
        package='uwb_localization',
        executable='pose_optimization_node',
        name='pose_optimization_node',
        parameters=[config]
        )

    bag_full = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', path_to_bag, '--clock'],
        output='screen'
    )

    nodes_to_execute = [node1, node2, node3, node4, bag_full]
    
    return LaunchDescription(nodes_to_execute)