import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import FrontendLaunchDescriptionSource

def generate_launch_description():

    package_dir = get_package_share_directory('uwb_localization')
    package_sim_dir = get_package_share_directory('uwb_simulator')

    config = os.path.join(package_dir, 'config', 'params.yaml')
    config_sim = os.path.join(package_sim_dir, 'config', 'params.yaml')
           
    
    ## Optimization nodes
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
    

    ## Simulation nodes

    node3 = Node(
                package='uwb_simulator',
                executable='measurement_simulator_eliko',
                name='measurement_simulator',
                parameters=[config_sim]
                )
                
    node4 = Node(
        package='uwb_simulator',
        executable='odometry_simulator',
        name='odometry_simulator',
        parameters=[config_sim]
        )
    
    node5 = Node(
        package='uwb_simulator',
        executable='clock_publisher',
        name='clock_publisher',
        parameters=[config_sim]
        )
    
    node6 = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            parameters=[{'use_sim_time': True}],
        )

    nodes_localization = [node1, node2]
    nodes_simulation = [node3, node4, node5, node6]
    
    nodes_to_execute = nodes_localization + nodes_simulation
    
    return LaunchDescription(nodes_to_execute)