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

    # Put here the path to the simulation bag you want to use, inside the  /bags folder
    bag_name = "dataset_lemniscate"
    path_to_bag = os.path.join(package_dir, 'bags', bag_name)
    
    node1 = Node(
                package='uwb_localization',
                executable='global_opt_node_eliko',
                name='eliko_global_opt_node',
                parameters=[config]
                )
                
    node2 = Node(
        package='uwb_localization',
        executable='pose_optimization_node',
        name='pose_optimization_node',
        parameters=[config]
        )
    
    bag_full = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', path_to_bag, '--clock'],
        output='screen'
    )

    nodes_to_execute = [node1, node2, bag_full]
    
    return LaunchDescription(nodes_to_execute)