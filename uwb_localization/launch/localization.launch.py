import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import FrontendLaunchDescriptionSource

def generate_launch_description():

    package_dir = get_package_share_directory('uwb_localization')
    dll_package_dir = get_package_share_directory('dll')
    dll_launch_file = os.path.join(dll_package_dir, 'launch', 'drone_arco.xml')

    config = os.path.join(package_dir, 'config', 'params.yaml')
    
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
        executable='optimizer_node_fusion',
        name='fusion_optimization_node',
        parameters=[config]
        )
    
    # Include XML launch file - This also plays the bag file!!
    dll_launch = IncludeLaunchDescription(
        FrontendLaunchDescriptionSource(dll_launch_file)
    )

    nodes_to_execute = [node1, node2, node3, node4, dll_launch]
    
    return LaunchDescription(nodes_to_execute)