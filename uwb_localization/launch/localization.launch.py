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
           
    node1 = Node(
                package='uwb_localization',
                executable='global_opt_node_eliko',
                name='eliko_global_opt_node',
                parameters=[config]
                )
                
    node2 = Node(
        package='uwb_localization',
        executable='optimizer_node_fusion',
        name='fusion_optimization_node',
        parameters=[config]
        )
    
    clock_pub = Node(
        package='uwb_simulator',
        executable='clock_publisher',
        name='clock_publisher',
        parameters=[config]
        )


    nodes_to_execute = [node1, node2, clock_pub]
    
    return LaunchDescription(nodes_to_execute)