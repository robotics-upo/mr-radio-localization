import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    package_dir = get_package_share_directory('uwb_localization')
    config = os.path.join(package_dir, 'config', 'params.yaml')
    
    node1 = Node(
                package='uwb_localization',
                executable='uav_odometry_node',
                name='uav_odometry_node',
                parameters=[config]
    )
           
    node2 = Node(
                package='uwb_localization',
                executable='global_opt_node_eliko',
                name='eliko_global_opt_node',
                parameters=[config]
                )
                
    node3 = Node(
        package='uwb_localization',
        executable='optimizer_node_fusion',
        name='fusion_optimization_node',
        parameters=[config]
        )

    nodes_to_execute = [node1, node2, node3]

    
    return LaunchDescription(nodes_to_execute)