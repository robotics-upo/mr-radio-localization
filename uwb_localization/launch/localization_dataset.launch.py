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
    bag_name = "experiment1_gt_v2_0/experiment1_gt_v2_0.mcap"
    path_to_bag = os.path.join(package_dir, 'bags', bag_name)

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

    # Radar nodes
    
    package_dir = get_package_share_directory('radar_odom')
    config_uav = os.path.join(package_dir, 'config', 'config_uav.yaml')
    config_agv = os.path.join(package_dir, 'config', 'config_agv.yaml')

    uav_radar_pcl_processor = Node(
        package='radar_odom',
        executable='radar_pcl_processor',
        output='screen',
        name='uav_radar_pcl_processor',
        remappings = [
            ('/filtered_pointcloud', 'uav/filtered_pointcloud'),
            ('/Ego_Vel_Twist', 'uav/Ego_Vel_Twist'),
            ('/inlier_pointcloud', 'uav/inlier_pointcloud'),
            ('/outlier_pointcloud', 'uav/outlier_pointcloud'),
            ('/raw_pointcloud', 'uav/raw_pointcloud')
        ],
        parameters=[config_uav]
    )

    agv_radar_pcl_processor = Node(
        package='radar_odom',
        executable='radar_pcl_processor',
        output='screen',
        name='agv_radar_pcl_processor',
        remappings = [
            ('/filtered_pointcloud', 'agv/filtered_pointcloud'),
            ('/Ego_Vel_Twist', 'agv/Ego_Vel_Twist'),
            ('/inlier_pointcloud', 'agv/inlier_pointcloud'),
            ('/outlier_pointcloud', 'agv/outlier_pointcloud'),
            ('/raw_pointcloud', 'agv/raw_pointcloud')
        ],
        parameters=[config_agv]
    )

    nodes_to_execute = [node1, node2, node3, node4, bag_full, uav_radar_pcl_processor, agv_radar_pcl_processor]
    
    return LaunchDescription(nodes_to_execute)