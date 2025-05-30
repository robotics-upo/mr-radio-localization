#!/usr/bin/env python3

import sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import re
from matplotlib.ticker import MultipleLocator


# === for paper‐quality bigger fonts ===
mpl.rcParams.update({
    'font.size':         14,   # base font size
    'axes.labelsize':    14,   # x/y label
    'axes.titlesize':    16,   # subplot title
    'xtick.labelsize':   12,   # tick numbers
    'ytick.labelsize':   12,
    'legend.fontsize':   12,
    'figure.titlesize':  18    # overall figure title
})



def quaternion_to_euler_angles(q):
    # Quaternion to Euler angles conversion
    # Adapted from: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    sinr_cosp = 2.0 * (q[0] * q[1] + q[2] * q[3])
    cosr_cosp = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q[0] * q[2] - q[3] * q[1])
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw




def plot_experiment_data(path_experiment_data, path_folder, gt_available = "True", simulation = "True"):

    
    print("gt_available set to: " + gt_available)
    print("simulation set to: " + simulation)

    #Get names of the topics we want to plot
    time_data_setpoint = '__time'
    
    uav_gt_topic_name = '/dll_node/pose_estimation'
    agv_gt_topic_name = '/dll_node_arco/pose_estimation'

    #These are the odometry sources used for relative transform estimation
    odom_topic_uav = "/uav/odom"
    odom_topic_agv = "/agv/odom"   #"/arco/idmind_motors/odom" #"/agv/odom"

    radar_topic_agv = "/agv/radar_odometry"

    tf_odom_agv = "arco/odom"

    ######### Values for dataset ##########
    # source_odom_origin = pose_to_matrix(np.array([13.694, 25.197, 0.58, 0.001, 0.001, 3.09])) #[13.694, 25.197, 0.58, 0.001, 0.001, 3.09], [10.682, 23.107, 0.214, 0.001, 0.001, -3.09]
    # target_odom_origin = pose_to_matrix(([15.926,22.978,0.901,0.0, 0.0, 0.0028])) #[15.925, 22.976, 0.911, 0.0, 0.0, 0.028]
    
    max_timestamp = None
    
    ##DLL AGV GT POSES
    source_gt_x_data = f'{agv_gt_topic_name}/pose/position/x'
    source_gt_y_data = f'{agv_gt_topic_name}/pose/position/y'
    source_gt_z_data = f'{agv_gt_topic_name}/pose/position/z'
    source_gt_q0_data = f'{agv_gt_topic_name}/pose/orientation/x'
    source_gt_q1_data = f'{agv_gt_topic_name}/pose/orientation/y'
    source_gt_q2_data = f'{agv_gt_topic_name}/pose/orientation/z'
    source_gt_q3_data = f'{agv_gt_topic_name}/pose/orientation/w'

    columns_source_gt_data = [time_data_setpoint, source_gt_x_data, source_gt_y_data, source_gt_z_data, source_gt_q0_data, source_gt_q1_data,source_gt_q2_data, source_gt_q3_data   ]
    source_gt_data_df = read_pandas_df(path_experiment_data, columns_source_gt_data, 
                                           timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
        
    ##DLL UAV GT POSES
    target_gt_x_data = f'{uav_gt_topic_name}/pose/position/x'
    target_gt_y_data = f'{uav_gt_topic_name}/pose/position/y'
    target_gt_z_data = f'{uav_gt_topic_name}/pose/position/z'
    target_gt_q0_data = f'{uav_gt_topic_name}/pose/orientation/x'
    target_gt_q1_data = f'{uav_gt_topic_name}/pose/orientation/y'
    target_gt_q2_data = f'{uav_gt_topic_name}/pose/orientation/z'
    target_gt_q3_data = f'{uav_gt_topic_name}/pose/orientation/w'
        
    columns_target_gt_data = [time_data_setpoint, target_gt_x_data, target_gt_y_data, target_gt_z_data, target_gt_q0_data , target_gt_q1_data ,target_gt_q2_data ,target_gt_q3_data ]
    target_gt_data_df = read_pandas_df(path_experiment_data, columns_target_gt_data, 
                                           timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
    
    ##Odometry topic AGV
    source_odom_x_data = f'{odom_topic_agv}/pose/pose/position/x'
    source_odom_y_data = f'{odom_topic_agv}/pose/pose/position/y'
    source_odom_z_data = f'{odom_topic_agv}/pose/pose/position/z'
    source_odom_q0_data = f'{odom_topic_agv}/pose/pose/orientation/x'
    source_odom_q1_data = f'{odom_topic_agv}/pose/pose/orientation/y'
    source_odom_q2_data = f'{odom_topic_agv}/pose/pose/orientation/z'
    source_odom_q3_data = f'{odom_topic_agv}/pose/pose/orientation/w'

    columns_source_odom_data = [time_data_setpoint, source_odom_x_data, source_odom_y_data, source_odom_z_data, source_odom_q0_data, source_odom_q1_data,source_odom_q2_data, source_odom_q3_data   ]

    source_odom_data_df = read_pandas_df(path_experiment_data, columns_source_odom_data, 
                                           timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
    
    ##Odometry topic UAV
    target_odom_x_data = f'{odom_topic_uav}/pose/pose/position/x'
    target_odom_y_data = f'{odom_topic_uav}/pose/pose/position/y'
    target_odom_z_data = f'{odom_topic_uav}/pose/pose/position/z'
    target_odom_q0_data = f'{odom_topic_uav}/pose/pose/orientation/x'
    target_odom_q1_data = f'{odom_topic_uav}/pose/pose/orientation/y'
    target_odom_q2_data = f'{odom_topic_uav}/pose/pose/orientation/z'
    target_odom_q3_data = f'{odom_topic_uav}/pose/pose/orientation/w'
    
    columns_target_odom_data = [time_data_setpoint, target_odom_x_data, target_odom_y_data, target_odom_z_data, target_odom_q0_data , target_odom_q1_data ,target_odom_q2_data ,target_odom_q3_data ]

    target_odom_data_df = read_pandas_df(path_experiment_data, columns_target_odom_data, 
                                           timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
    

    ##Lucia's AGV radar odometry
    source_radar_x_data = f'{radar_topic_agv}/pose/pose/position/x'
    source_radar_y_data = f'{radar_topic_agv}/pose/pose/position/y'
    source_radar_z_data = f'{radar_topic_agv}/pose/pose/position/z'
    source_radar_q0_data = f'{radar_topic_agv}/pose/pose/orientation/x'
    source_radar_q1_data = f'{radar_topic_agv}/pose/pose/orientation/y'
    source_radar_q2_data = f'{radar_topic_agv}/pose/pose/orientation/z'
    source_radar_q3_data = f'{radar_topic_agv}/pose/pose/orientation/w'

    columns_source_radar_data = [time_data_setpoint, source_radar_x_data, source_radar_y_data, source_radar_z_data, source_radar_q0_data , source_radar_q1_data ,source_radar_q2_data ,source_radar_q3_data ]

    # source_radar_data_df = read_pandas_df(path_experiment_data, columns_source_radar_data, 
    #                                        timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)

    # Extract origin from the first AGV GT pose
    first_agv_row = source_gt_data_df.iloc[0]
    agv_pos = first_agv_row[[source_gt_x_data, source_gt_y_data, source_gt_z_data]].values
    agv_quat = first_agv_row[[source_gt_q0_data, source_gt_q1_data, source_gt_q2_data, source_gt_q3_data]].values
    agv_rpy = R.from_quat(agv_quat).as_euler('zyx', degrees=False)  # yaw, pitch, roll
    source_odom_origin = pose_to_matrix(np.concatenate((agv_pos, agv_rpy[::-1])))  # [x, y, z, roll, pitch, yaw]
    print(source_odom_origin)

    # Extract origin from the first UAV GT pose
    first_uav_row = target_gt_data_df.iloc[0]
    uav_pos = first_uav_row[[target_gt_x_data, target_gt_y_data, target_gt_z_data]].values
    uav_quat = first_uav_row[[target_gt_q0_data, target_gt_q1_data, target_gt_q2_data, target_gt_q3_data]].values
    uav_rpy = R.from_quat(uav_quat).as_euler('zyx', degrees=False)
    target_odom_origin = pose_to_matrix(np.concatenate((uav_pos, uav_rpy[::-1])))  # [x, y, z, roll, pitch, yaw]
    print(target_odom_origin)

    
    #odom-gt AGV
    plot_states_temporal(path_experiment_data, "comparison_odom", source_odom_data_df, columns_source_odom_data, None, None, source_gt_data_df, columns_source_gt_data)
    plot_states_temporal(path_experiment_data, "comparison_odom", source_odom_data_df, columns_source_odom_data, None, None, source_gt_data_df, columns_source_gt_data, source_odom_origin)
    plot_states_3d(path_experiment_data, "3d_odom_trajectory", source_odom_data_df, columns_source_odom_data, None, None, source_gt_data_df, columns_source_gt_data, source_odom_origin)

    #odom-gt UAV
    plot_states_temporal(path_experiment_data, "comparison", target_odom_data_df, columns_target_odom_data, None, None, target_gt_data_df, columns_target_gt_data)
    plot_states_temporal(path_experiment_data, "comparison", target_odom_data_df, columns_target_odom_data, None, None, target_gt_data_df, columns_target_gt_data, target_odom_origin) 
    plot_states_3d(path_experiment_data, "3d_odom_trajectory", target_odom_data_df, columns_target_odom_data, None, None, target_gt_data_df, columns_target_gt_data, target_odom_origin)
    

def plot_states_3d(path, filename, data, cols_data, radar = None, cols_radar = None, gt=None, cols_gt=None, gt_origin=None):
    """
    Plots 3D trajectories (x, y, z) of estimated data and ground truth, with optional frame alignment.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract data trajectory
    data_x = np.array(data[cols_data[1]])
    data_y = np.array(data[cols_data[2]])
    data_z = np.array(data[cols_data[3]])

    ax.plot(data_x, data_y, data_z, c='b', label='Odom topic')

    if radar is not None and cols_radar is not None:

        radar_x = np.array(radar[cols_radar[1]])
        radar_y = np.array(radar[cols_radar[2]])
        radar_z = np.array(radar[cols_radar[3]])
        # Compute ground truth yaw from quaternion data.
        radar_yaw = []
        for i in range(len(radar)):
            quat = [radar[cols_radar[4]].iloc[i], radar[cols_radar[5]].iloc[i],
                    radar[cols_radar[6]].iloc[i], radar[cols_radar[7]].iloc[i]]
            yaw_val = R.from_quat(quat).as_euler('zyx', degrees=False)[0]
            radar_yaw.append(yaw_val)
        radar_yaw = np.array(radar_yaw)

        ax.plot(radar_x, radar_y, radar_z, c='g', label='Radar odometry')

    if gt is not None and cols_gt is not None:
        if gt_origin is not None:
            T_local_world = np.linalg.inv(gt_origin)
            gt_local_positions = []

            for i in range(len(gt)):
                pos = gt.iloc[i][[cols_gt[1], cols_gt[2], cols_gt[3]]].values
                quat = gt.iloc[i][[cols_gt[4], cols_gt[5], cols_gt[6], cols_gt[7]]].values

                T_world = np.eye(4)
                T_world[:3, 3] = pos
                T_world[:3, :3] = R.from_quat(quat).as_matrix()

                T_local = T_local_world @ T_world
                gt_local_positions.append(T_local[:3, 3])

            gt_local_positions = np.array(gt_local_positions)
            gt_x, gt_y, gt_z = gt_local_positions[:, 0], gt_local_positions[:, 1], gt_local_positions[:, 2]

        else:
            gt_x = np.array(gt[cols_gt[1]])
            gt_y = np.array(gt[cols_gt[2]])
            gt_z = np.array(gt[cols_gt[3]])

        ax.plot(gt_x, gt_y, gt_z, c='r', label='Ground Truth')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory: Odom topic vs Ground Truth')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_states_temporal(path, filename, data, cols_data, radar = None, cols_radar = None, gt = None, cols_gt = None, gt_origin = None):

    # Plot each variable vs. time
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    data_x = np.array(data[cols_data[1]])
    data_y = np.array(data[cols_data[2]])
    data_z = np.array(data[cols_data[3]])
    # Compute ground truth yaw from quaternion data.
    data_yaw = []
    for i in range(len(data)):
        quat = [data[cols_data[4]].iloc[i], data[cols_data[5]].iloc[i],
                data[cols_data[6]].iloc[i], data[cols_data[7]].iloc[i]]
        yaw_val = R.from_quat(quat).as_euler('zyx', degrees=False)[0]
        data_yaw.append(yaw_val)
    data_yaw = np.array(data_yaw)

    if radar is not None and cols_radar is not None:

        radar_x = np.array(radar[cols_radar[1]])
        radar_y = np.array(radar[cols_radar[2]])
        radar_z = np.array(radar[cols_radar[3]])
        # Compute ground truth yaw from quaternion data.
        radar_yaw = []
        for i in range(len(radar)):
            quat = [radar[cols_radar[4]].iloc[i], radar[cols_radar[5]].iloc[i],
                    radar[cols_radar[6]].iloc[i], radar[cols_radar[7]].iloc[i]]
            yaw_val = R.from_quat(quat).as_euler('zyx', degrees=False)[0]
            radar_yaw.append(yaw_val)
        radar_yaw = np.array(radar_yaw)

        axs[0].plot(np.array(radar[cols_radar[0]]), radar_x, c='g', label = 'radar')
        axs[1].plot(np.array(radar[cols_radar[0]]), radar_y, c='g', label = 'radar')
        axs[2].plot(np.array(radar[cols_radar[0]]), radar_z, c='g', label = 'radar')


    if gt is not None and cols_gt is not None:

        if gt_origin is not None: 
            T_local_world = np.linalg.inv(gt_origin)
            gt_local_positions = []
            gt_local_yaws = []

            for i in range(len(gt)):
                pos = gt.iloc[i][[cols_gt[1], cols_gt[2], cols_gt[3]]].values
                quat = gt.iloc[i][[cols_gt[4], cols_gt[5], cols_gt[6], cols_gt[7]]].values

                T_world = np.eye(4)
                T_world[:3, 3] = pos
                T_world[:3, :3] = R.from_quat(quat).as_matrix()

                T_local = T_local_world @ T_world

                gt_local_positions.append(T_local[:3, 3])
                yaw_local = R.from_matrix(T_local[:3, :3]).as_euler('zyx')[0]
                gt_local_yaws.append(yaw_local)

            gt_local_positions = np.array(gt_local_positions)
            gt_x = gt_local_positions[:, 0]
            gt_y = gt_local_positions[:, 1]
            gt_z = gt_local_positions[:, 2]
            gt_yaw = np.array(gt_local_yaws)
        
        else: 
            
            gt_x = np.array(gt[cols_gt[1]])
            gt_y = np.array(gt[cols_gt[2]])
            gt_z = np.array(gt[cols_gt[3]])
            # Compute ground truth yaw from quaternion data.
            gt_yaw = []
            for i in range(len(gt)):
                quat = [gt[cols_gt[4]].iloc[i], gt[cols_gt[5]].iloc[i],
                        gt[cols_gt[6]].iloc[i], gt[cols_gt[7]].iloc[i]]
                yaw_val = R.from_quat(quat).as_euler('zyx', degrees=False)[0]
                gt_yaw.append(yaw_val)
            gt_yaw = np.array(gt_yaw)
        
        axs[0].plot(np.array(gt[cols_gt[0]]), gt_x, c='r', label = 'ref')
        axs[1].plot(np.array(gt[cols_gt[0]]), gt_y, c='r', label = 'ref')
        axs[2].plot(np.array(gt[cols_gt[0]]), gt_z, c='r', label = 'ref')
        axs[3].plot(np.array(gt[cols_gt[0]]), gt_yaw, c='r', label = 'ref')

        
        axs[0].plot(np.array(data[cols_data[0]]), data_x, c='b', label = 'data')

        axs[0].set_ylabel('X (m)')
        axs[0].grid()
        
        axs[1].plot(np.array(data[cols_data[0]]), data_y, c='b', label = 'data')

        axs[1].set_ylabel('Y (m)')
        axs[1].grid()
        
        axs[2].plot(np.array(data[cols_data[0]]), data_z, c='b', label = 'data')

        axs[2].set_ylabel('Z (m)')
        axs[2].grid()

        axs[3].plot(np.array(data[cols_data[0]]), data_yaw, c='b', label = 'data')

        axs[3].set_ylabel('Yaw (rad)')
        axs[3].grid()

        axs[3].set_xlabel('Time (s)')

    
    plt.suptitle("Temporal Evolution of Pose (x, y, z, yaw)")

    plt.show()


def graph_dict_to_df(pose_graph):
    """
    Converts a pose graph dictionary into a DataFrame with separate columns.
    The dictionary is assumed to have keys: 'timestamp', 'position' (a 3-element array),
    'orientation' (a 4-element array), and 'yaw'.
    """
    data = []
    for key, pose in pose_graph.items():
        data.append({
            'timestamp': pose['timestamp'],
            'position_x': pose['position'][0],
            'position_y': pose['position'][1],
            'position_z': pose['position'][2],
            'orientation_yaw': pose['yaw']
        })
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df



def read_pandas_df(path, columns, timestamp_col=None, max_timestamp=None, dropna = True):


    """
    Reads a CSV file and filters rows based on timestamp, if provided.

    Parameters:
        path (str): Path to the CSV file.
        columns (list): List of columns to read from the file.
        timestamp_col (str, optional): Name of the timestamp column.
        max_timestamp (float, optional): Maximum allowable timestamp.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """

    try:
            # Read CSV file into a Pandas DataFrame
            df = pd.read_csv(path, usecols = columns)
            if(dropna): df = df.dropna()
            
             # Filter rows based on timestamp, if applicable
            if timestamp_col and max_timestamp is not None:
                df = df[df[timestamp_col] < df[timestamp_col].iloc[0] + max_timestamp]

            # Check if the specified columns exist in the DataFrame
            for column in columns:
                if column not in df.columns:
                    raise ValueError("One or more specified columns not found in the CSV file.")

    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: File '{path}' is empty.")
    except pd.errors.ParserError:
        print(f"Error: Unable to parse file '{path}'. Make sure it's a valid CSV file.")
    except ValueError as ve:
        print(f"Error: {ve}")

    return df

def normalize_angle(self,theta):
        return np.arctan2(np.sin(theta), np.cos(theta))


def compute_base_link_to_map_df(tf_map_to_odom_df, tf_odom_to_base_df, timestamp_col="__time"):
    """
    Computes the transformation from base_link to map (T_map_to_base)
    by chaining map->odom and odom->base_link.

    Returns:
        pd.DataFrame: with columns:
            - timestamp
            - position_x, position_y, position_z
            - orientation_x, orientation_y, orientation_z, orientation_w
    """
    data = []

    for i in range(len(tf_map_to_odom_df)):
        time = tf_map_to_odom_df.iloc[i][timestamp_col]

        # Extract map -> odom
        pos_map_odom = tf_map_to_odom_df.iloc[i][1:4].values
        quat_map_odom = tf_map_to_odom_df.iloc[i][4:8].values
        T_map_to_odom = np.eye(4)
        T_map_to_odom[:3, :3] = R.from_quat(quat_map_odom).as_matrix()
        T_map_to_odom[:3, 3] = pos_map_odom

        # Extract odom -> base_link
        pos_odom_base = tf_odom_to_base_df.iloc[i][1:4].values
        quat_odom_base = tf_odom_to_base_df.iloc[i][4:8].values
        T_odom_to_base = np.eye(4)
        T_odom_to_base[:3, :3] = R.from_quat(quat_odom_base).as_matrix()
        T_odom_to_base[:3, 3] = pos_odom_base

        # Chain the transforms
        T_map_to_base = T_map_to_odom @ T_odom_to_base
        pos = T_map_to_base[:3, 3]
        rot = R.from_matrix(T_map_to_base[:3, :3])
        quat = rot.as_quat()  # [x, y, z, w]

        data.append({
            timestamp_col: time,
            "position_x": pos[0],
            "position_y": pos[1],
            "position_z": pos[2],
            "orientation_x": quat[0],
            "orientation_y": quat[1],
            "orientation_z": quat[2],
            "orientation_w": quat[3],
        })

    df = pd.DataFrame(data)
    return df

def pose_to_matrix(pose):
    x, y, z, roll, pitch, yaw = pose
    rot = R.from_euler('zyx', [yaw,pitch,roll])
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = [x, y, z]
    return T

def add_yaw_to_df(df, col_qx, col_qy, col_qz, col_qw, new_col_name="yaw"):
    quats = df[[col_qx, col_qy, col_qz, col_qw]].values
    yaws = R.from_quat(quats).as_euler('zyx', degrees=False)[:, 0]  # extract yaw only
    df[new_col_name] = yaws
    return df

def main():

    if len(sys.argv) != 4:
        print("Usage: python script.py <csv_file_path>")
        sys.exit(1)

    path_to_data = sys.argv[1]
    gt_available = sys.argv[2]
    simulation = sys.argv[3]

    bag_csv_path = path_to_data + "/data.csv"

    plot_experiment_data(bag_csv_path, path_to_data, gt_available, simulation)


if __name__ == "__main__":
    main()