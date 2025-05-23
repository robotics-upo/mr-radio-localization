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

def compute_That_w_t(timestamp, target_gt_data_df, target_odom_data_df, t_That_s_data_df, columns_target_gt_data, columns_target_odom_data, columns_t_That_s_data):
    """
    Computes poses of the target wrt source (also world) according to the estimated t_That_s estimation and the target odometry poses

    """

    That_w_t_list = []

    # Ensure data is sorted by timestamp
    target_gt_data_df = target_gt_data_df.sort_values(timestamp).reset_index(drop=True)
    target_odom_data_df = target_odom_data_df.sort_values(timestamp).reset_index(drop=True)
    t_That_s_data_df = t_That_s_data_df.sort_values(timestamp).reset_index(drop=True)

    for _, row in t_That_s_data_df.iterrows():
        # Find the latest source_gt_data_df row with a timestamp <= current t_That_s_data_df timestamp
        latest_target_idx = target_gt_data_df[target_gt_data_df[timestamp] <= row[timestamp]].index.max()
        latest_target_odom_idx = target_odom_data_df[target_odom_data_df[timestamp] <= row[timestamp]].index.max()

        if latest_target_idx is not None and latest_target_odom_idx is not None:  # Ensure there's a valid matching source data
            # Get the corresponding source and t_That_s data
            target_row = target_gt_data_df.iloc[latest_target_idx]
            target_odom_row = target_odom_data_df.iloc[latest_target_odom_idx]

            target_row = target_gt_data_df.iloc[latest_target_idx]
            T_w_t = np.eye(4)
            T_w_t_odom = np.eye(4)
            That_t_s = np.eye(4)

            # Populate T_w_t_odom (target positions in odom frame)
            T_w_t_odom[:3, 3] = target_odom_row[[columns_target_odom_data[1], columns_target_odom_data[2], columns_target_odom_data[3]]].values
            q_w_t_odom = target_odom_row[[columns_target_odom_data[4], columns_target_odom_data[5], columns_target_odom_data[6], columns_target_odom_data[7]]].values
            T_w_t_odom[:3, :3] = R.from_quat(q_w_t_odom).as_matrix()

            # Populate T_w_t (target positions in gt frame)
            T_w_t[:3, 3] = target_row[[columns_target_gt_data[1], columns_target_gt_data[2], columns_target_gt_data[3]]].values
            q_w_t = target_row[[columns_target_gt_data[4], columns_target_gt_data[5], columns_target_gt_data[6], columns_target_gt_data[7]]].values
            T_w_t[:3, :3] = R.from_quat(q_w_t).as_matrix()

            # Populate That_t_s
            That_t_s[:3, 3] = row[[columns_t_That_s_data[1], columns_t_That_s_data[2], columns_t_That_s_data[3]]].values
            q_hat = row[[columns_t_That_s_data[4], columns_t_That_s_data[5], columns_t_That_s_data[6], columns_t_That_s_data[7]]].values
            That_t_s[:3, :3] = R.from_quat(q_hat).as_matrix()
            
            # Compute That_w_t -> computed from odometry
            That_w_t = np.linalg.inv(That_t_s) @ T_w_t_odom

            translation = That_w_t[:3, 3]  # Extract the translation part
            rotation = R.from_matrix(That_w_t[:3, :3]).as_quat()  # Extract rotation as quaternion
            yaw = R.from_matrix(That_w_t[:3, :3]).as_euler('zyx', degrees=False)[0]  # Extract yaw (rotation around Z-axis)

            That_w_t_list.append([row[timestamp], *translation, *rotation, yaw])  # Include timestamp, translation, and rotation


    # Create a DataFrame for That_w_t and metrics
    That_w_t_df = pd.DataFrame(That_w_t_list, columns=[timestamp, "x", "y", "z", "qx", "qy", "qz", "qw", "yaw"])
    
    return That_w_t_df


def compute_poses_local_to_world(timestamp_col, poses, anchor_df, anchor_columns, use_last = True, use_gt = False, gt_anchor = None):
    
    """
    For each pose in poses, apply the anchor transform (either closest in time or last row)
    to convert it from local to world frame.

    Parameters:
        poses (dict): Dictionary of poses. Each entry should have:
                      - 'timestamp': float (in seconds)
                      - 'position': np.array of shape (3,)
                      - 'orientation': np.array of shape (4,)
        anchor_df (pd.DataFrame): DataFrame with anchor transform data.
        timestamp_col (str): Name of timestamp column in anchor_df.
        anchor_columns (list): Column names for [timestamp, x, y, z, qx, qy, qz, qw].
        use_last (bool): If True, uses the last anchor transform for all poses.
        use_gt (bool): If True, uses the ground truth anchor to transform all poses

    Returns:
        dict: Transformed poses with keys:
              - 'timestamp', 'position', 'orientation', 'yaw'
    """
    poses_world = {}
    
    # Sort the anchor DataFrame and reset its index
    sorted_anchor_df = anchor_df.sort_values(timestamp_col).reset_index(drop=True)
    sorted_anchor_df[timestamp_col] -= sorted_anchor_df[timestamp_col][0]

    # Precompute last anchor transform if requested
    if use_gt is False and use_last is True:
        anchor_row = sorted_anchor_df.iloc[-1]
        T_anchor_last = np.eye(4)
        T_anchor_last[:3, 3] = anchor_row[[anchor_columns[1], anchor_columns[2], anchor_columns[3]]].values
        quat_anchor = anchor_row[[anchor_columns[4], anchor_columns[5], anchor_columns[6], anchor_columns[7]]].values
        T_anchor_last[:3, :3] = R.from_quat(quat_anchor).as_matrix()
        print("Last anchor (Transformation from source to target):\n", T_anchor_last)
    
    for key, pose in poses.items():
        if (pose['timestamp'] is None or 
            pose['position'] is None or 
            pose['orientation'] is None):
            continue  # skip invalid entries
        
        # Find the candidate rows with anchor timestamps <= current pose timestamp
        candidate_df = sorted_anchor_df[sorted_anchor_df[timestamp_col] <= pose['timestamp']]
        if candidate_df.empty:
            continue  # no anchor available for this pose

        # Get the row with the largest timestamp that is <= pose['timestamp']
        latest_anchor_idx = candidate_df.index[-1]
        anchor_row = sorted_anchor_df.loc[latest_anchor_idx]
        
        # Construct the anchor transform T_anchor (4x4 matrix)
        T_anchor = np.eye(4)
        # Translation: assumed stored in columns 1-3 of anchor_columns.
        T_anchor[:3, 3] = anchor_row[[anchor_columns[1], anchor_columns[2], anchor_columns[3]]].values  
        # Quaternion: assumed stored in columns 4-7 of anchor_columns.
        quat_anchor = anchor_row[[anchor_columns[4], anchor_columns[5], anchor_columns[6], anchor_columns[7]]].values
        T_anchor[:3, :3] = R.from_quat(quat_anchor).as_matrix()
        
        # Convert the current UAV pose into a transform T_pose.
        T_pose = np.eye(4)
        T_pose[:3, 3] = pose['position']
        T_pose[:3, :3] = R.from_quat(pose['orientation']).as_matrix()
        
        if(use_gt and gt_anchor is not None): T_world = gt_anchor @ T_pose
        # Apply the anchor transform: T_world = T_anchor * T_pose using last optimized anchor value
        elif(use_last): T_world = T_anchor_last @ T_pose
        # Apply the anchor transform: T_world = T_anchor * T_pose using the most recent at the time of this pose
        else: T_world = T_anchor @ T_pose

        # print("T_w_s (Transformation from robot to world):\n", T_world)
        
        new_translation = T_world[:3, 3]
        new_quat = R.from_matrix(T_world[:3, :3]).as_quat()
        new_yaw = R.from_matrix(T_world[:3, :3]).as_euler('zyx', degrees=False)[0]
        
        poses_world[key] = {
            'timestamp': pose['timestamp'],
            'position': new_translation,
            'orientation': new_quat,
            'yaw': new_yaw
        }
    
    return poses_world

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


def plot_3d_scatter(path, data_frame_ref, data_frame_ref_odom, data_frame_target, data_frame_target_odom, data_frame_opt, cols_ref, cols_ref_odom, cols_target, cols_target_odom, cols_opt, source_odom_origin, target_odom_origin):
    """
    Plots the 3D scatter including reference, target, odom, and marker positions.
    
    Parameters:
        data_frame_ref (pd.DataFrame): DataFrame for reference trajectory. +odom: odometry source
        data_frame_target (pd.DataFrame): DataFrame for target trajectory. +odom: odometry target
        marker_positions (pd.DataFrame): DataFrame for marker positions.
        cols_ref (list): Columns for reference trajectory [x, y, z].
        cols_target (list): Columns for target trajectory [x, y, z].
    """
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    # Plot ground truth and odom source trajectory
    ax.plot(data_frame_ref[cols_ref[1]], data_frame_ref[cols_ref[2]], data_frame_ref[cols_ref[3]], c='r', label='GT source', linewidth=2)
    # For the source odometry (e.g., AGV), use the source odom origin:
    # Extract local odom data
    odom_source_local = data_frame_ref_odom[[cols_ref_odom[1], cols_ref_odom[2], cols_ref_odom[3]]].values.T  # shape (3, N)
    T_source = source_odom_origin
    odom_source_world = T_source[:3, :3] @ odom_source_local + T_source[:3, 3:4]

    odom_source_x = odom_source_world[0, :]
    odom_source_y = odom_source_world[1, :]
    odom_source_z = odom_source_world[2, :]

    ax.plot(odom_source_x, odom_source_y, odom_source_z, c='r', label='odom source', linestyle='--', linewidth=2)

    # Plot ground truth target trajectory
    ax.plot(data_frame_target[cols_target[1]], data_frame_target[cols_target[2]], data_frame_target[cols_target[3]], c='g', label='GT Target', linewidth=2)
    # For the target odometry, use the target odom origin:

    odom_target_local = data_frame_target_odom[[cols_target_odom[1], cols_target_odom[2], cols_target_odom[3]]].values.T  # shape (3, N)
    T_target = target_odom_origin
    odom_target_world = T_target[:3, :3] @ odom_target_local + T_target[:3, 3:4]
    
    odom_target_x = odom_target_world[0, :]
    odom_target_y = odom_target_world[1, :]
    odom_target_z = odom_target_world[2, :]
    
    ax.plot(odom_target_x, odom_target_y, odom_target_z, c='g', linestyle = '--', label='Odom Target', linewidth=2)

    # Plot markers
    ax.scatter(data_frame_opt[cols_opt[1]], data_frame_opt[cols_opt[2]], data_frame_opt[cols_opt[3]], c='b', marker='o', label='Optimized', alpha=0.4, s=10)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()

    plt.savefig(path + '/pose_3D.png', bbox_inches='tight')

    # Create a top-down view
    ax.view_init(elev=90, azim=-90)  # Elevation of 90° for top-down view, azimuth adjusted for alignment
    plt.savefig(f"{path}/pose_top_down.png", bbox_inches='tight')

    plt.show()

def plot_pose_temporal_evolution(path, df_ref, df_experiment, df_covariance, df_covariance_odom, df_odom, cols_ref, cols_experiment, cols_covariance, cols_covariance_odom, cols_odom, t0, target_odom_origin):
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    plt.suptitle("Temporal Evolution of 3D Pose")

    # Subtract the first element of the timestamp column to start from 0
    #df_experiment[cols_experiment[0]] -= df_experiment[cols_experiment[0]].iloc[0]
    df_experiment[cols_experiment[0]] -= t0
    #df_covariance[cols_experiment[0]] -= df_covariance[cols_experiment[0]].iloc[0]
    df_covariance[cols_experiment[0]] -= t0

    df_ref[cols_ref[0]] -= t0
    df_odom[cols_odom[0]] -= t0
    df_covariance_odom[cols_covariance_odom[0]] -= t0

    # # Subtract the first element of the timestamp column to start from 0
    # df_experiment[cols_experiment[0]] *= 1e-6
    # df_ref[cols_ref[0]] *= 1e-6

    # Align timestamps and compute RMSE
    merged_cov_df = pd.merge_asof(df_covariance, df_covariance_odom, left_on=cols_covariance[0], right_on=cols_covariance_odom[0], direction='nearest')

    # Compute standard deviations (square root of diagonal covariance elements)
    std_x = np.sqrt(np.array(merged_cov_df[cols_covariance[1]])) + np.sqrt(np.array(merged_cov_df[cols_covariance_odom[1]]))  # Covariance_x
    std_y = np.sqrt(np.array(merged_cov_df[cols_covariance[2]])) + np.sqrt(np.array(merged_cov_df[cols_covariance_odom[2]])) # Covariance_y
    std_z = np.sqrt(np.array(merged_cov_df[cols_covariance[3]])) + np.sqrt(np.array(merged_cov_df[cols_covariance_odom[3]])) # Covariance_z
    
    #Convert the odom target frame to world frame used gt odometry offset

    odom_target_local = df_odom[[cols_odom[1], cols_odom[2], cols_odom[3]]].values.T  # shape (3, N)
    T_target = target_odom_origin
    odom_target_world = T_target[:3, :3] @ odom_target_local + T_target[:3, 3:4]
    
    target_odom_x = odom_target_world[0, :]
    target_odom_y = odom_target_world[1, :]
    target_odom_z = odom_target_world[2, :]
  
    axes[0].scatter(np.array(df_experiment[cols_experiment[0]]), np.array(df_experiment[cols_experiment[1]]), c='b', label = 'pose_opt', alpha=0.4, s=10)
    axes[0].plot(np.array(df_ref[cols_ref[0]]), np.array(df_ref[cols_ref[1]]), c='r', label = 'target gt')
    
    axes[0].plot(np.array(df_odom[cols_odom[0]]), target_odom_x, c='r', linestyle= '--', label = 'target odom')

    axes[0].fill_between(np.array(df_covariance[cols_experiment[0]]),
                         np.array(df_experiment[cols_experiment[1]]) - 2.0*std_x,
                         np.array(df_experiment[cols_experiment[1]]) + 2.0*std_x,
                         color='blue', alpha=0.2, label='±2σ Uncertainty')
    axes[0].legend()
    #plt.xlabel("Timestamp")
    axes[0].set_ylabel("X(m)")
    axes[0].grid()
    axes[1].scatter(np.array(df_experiment[cols_experiment[0]]), np.array(df_experiment[cols_experiment[2]]), c='b',alpha=0.4, s=10)
    axes[1].plot(np.array(df_ref[cols_ref[0]]), np.array(df_ref[cols_ref[2]]), c='r')
    axes[1].plot(np.array(df_odom[cols_odom[0]]), target_odom_y, c='r',linestyle= '--')

    axes[1].fill_between(np.array(df_experiment[cols_experiment[0]]),
                         np.array(df_experiment[cols_experiment[2]]) - 2.0*std_y,
                         np.array(df_experiment[cols_experiment[2]]) + 2.0*std_y,
                         color='blue', alpha=0.2, label='±2σ Uncertainty')
    axes[1].set_ylabel("Y(m)")
    axes[1].grid()

    axes[2].scatter(np.array(df_experiment[cols_experiment[0]]), np.array(abs(df_experiment[cols_experiment[3]])), c='b', alpha=0.4, s=10)
    axes[2].plot(np.array(df_ref[cols_ref[0]]), np.array(df_ref[cols_ref[3]]), c='r')
    axes[2].plot(np.array(df_odom[cols_odom[0]]), target_odom_z, c='r', linestyle= '--')

    axes[2].fill_between(np.array(df_experiment[cols_experiment[0]]),
                         np.array(df_experiment[cols_experiment[3]]) - 2.0*std_z,
                         np.array(df_experiment[cols_experiment[3]]) + 2.0*std_z,
                         color='blue', alpha=0.2, label='±2σ Uncertainty')
    axes[2].set_ylabel("Z(m)")
    axes[2].grid()

    axes[2].set_xlabel("Time(s)")

    plt.savefig(path + '/pose_t.png', bbox_inches='tight')

    plt.show()

def plot_transform(path, df_experiment, df_covariance, cols_experiment, cols_covariance, t0, source_odom_origin = None, target_odom_origin = None):

    gt_available = False
    if(source_odom_origin is not None and target_odom_origin is not None):
        T_t_s = np.linalg.inv(target_odom_origin) @ source_odom_origin
        print("T_t_s (Transformation from source to target):\n", T_t_s)
        # Extract constant ground truth transform from source to target
        t_t_s_translation = T_t_s[:3, 3]
        t_t_s_yaw = R.from_matrix(T_t_s[:3, :3]).as_euler('zyx')[0]
        gt_available = True

    fig, axes = plt.subplots(4, 1, figsize=(15, 15))
    plt.suptitle("Temporal Evolution of T_t_s")

    # Subtract the first element of the timestamp column to start from 0
    #df_experiment[cols_experiment[0]] -= df_experiment[cols_experiment[0]].iloc[0]
    df_experiment[cols_experiment[0]] -= t0
    #df_covariance[cols_experiment[0]] -= df_covariance[cols_experiment[0]].iloc[0]
    df_covariance[cols_experiment[0]] -= t0
 
    # Compute standard deviations (square root of diagonal covariance elements)
    std_x = np.sqrt(np.array(df_covariance[cols_covariance[1]]))  # Covariance_x
    std_y = np.sqrt(np.array(df_covariance[cols_covariance[2]]))  # Covariance_y
    std_z = np.sqrt(np.array(df_covariance[cols_covariance[3]]))  # Covariance_z
    std_yaw = np.sqrt(np.array(df_covariance[cols_covariance[4]]))  # Covariance_z


    # # Subtract the first element of the timestamp column to start from 0
    # df_experiment[cols_experiment[0]] *= 1e-6

    # Create a constant reference line over the same time span
    timestamps = np.array(df_experiment[cols_experiment[0]])
    if(gt_available): axes[0].plot(timestamps, [t_t_s_translation[0]] * len(timestamps), 'r--', label='GT x')

    axes[0].plot(timestamps, np.array(df_experiment[cols_experiment[1]]), c='b', label = 'pose_opt')


    axes[0].fill_between(timestamps,
                         np.array(df_experiment[cols_experiment[1]]) - 2.0*std_x,
                         np.array(df_experiment[cols_experiment[1]]) + 2.0*std_x,
                         color='blue', alpha=0.2, label='±2σ Uncertainty')
    axes[0].legend()
    #plt.xlabel("Timestamp")
    axes[0].set_ylabel("X(m)")
    axes[0].grid()

    if(gt_available): axes[1].plot(timestamps, [t_t_s_translation[1]] * len(timestamps), 'r--', label='GT y')
    axes[1].plot(timestamps, np.array(df_experiment[cols_experiment[2]]), c='b')

    axes[1].fill_between(timestamps,
                         np.array(df_experiment[cols_experiment[2]]) - 2.0*std_y,
                         np.array(df_experiment[cols_experiment[2]]) + 2.0*std_y,
                         color='blue', alpha=0.2, label='±2σ Uncertainty')
    axes[1].set_ylabel("Y(m)")
    axes[1].grid()

    if(gt_available): axes[2].plot(timestamps, [t_t_s_translation[2]] * len(timestamps), 'r--', label='GT z')
    axes[2].plot(timestamps, np.array(df_experiment[cols_experiment[3]]), c='b')


    axes[2].fill_between(timestamps,
                         np.array(df_experiment[cols_experiment[3]]) - 2.0*std_z,
                         np.array(df_experiment[cols_experiment[3]]) + 2.0*std_z,
                         color='blue', alpha=0.2, label='±2σ Uncertainty')
    axes[2].set_ylabel("Z(m)")
    axes[2].grid()

    if(gt_available): axes[3].plot(timestamps, [t_t_s_yaw] * len(timestamps), 'r--', label='GT yaw')
    axes[3].plot(timestamps, np.array(df_experiment[cols_experiment[-1]]), c='b')

    axes[3].fill_between(timestamps,
                         np.array(df_experiment[cols_experiment[-1]]) - 2.0*std_yaw,
                         np.array(df_experiment[cols_experiment[-1]]) + 2.0*std_yaw,
                         color='blue', alpha=0.2, label='±2σ Uncertainty')
    axes[3].set_ylabel("Yaw(rad)")
    axes[3].grid()

    axes[3].set_xlabel("Time(s)")

    # Add legend to yaw plot for clarity
    axes[3].legend()

    plt.savefig(path + '/t_That_s.png', bbox_inches='tight')

    plt.show()


def plot_attitude_temporal_evolution(path, df_ref_rpy, df_opt_rpy, df_covariance, df_covariance_odom, df_odom_rpy, cols_experiment, cols_covariance, cols_covariance_odom, target_odom_origin):
    

    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    plt.suptitle("Temporal Evolution of attitude")

    #df_opt_rpy[cols_experiment[0]] -= df_opt_rpy[cols_experiment[0]].iloc[0]
    #df_opt_rpy[cols_experiment[0]] -= t0

    #df_covariance[cols_experiment[0]] -= df_covariance[cols_experiment[0]].iloc[0]
    #df_covariance[cols_experiment[0]] -= t0

    # Subtract the first element of the timestamp column to start from 0
    #df_ref_rpy[cols_experiment[0]] -= t0


    # Align timestamps and compute RMSE
    merged_cov_df = pd.merge_asof(df_covariance, df_covariance_odom, left_on=cols_covariance[0], right_on=cols_covariance_odom[0], direction='nearest')
    std_yaw = np.sqrt(np.array(merged_cov_df[cols_covariance[4]])) + np.sqrt(np.array(merged_cov_df[cols_covariance_odom[4]]))  # Covariance_yaw

    # # Subtract the first element of the timestamp column to start from 0
    # df_experiment[cols_experiment[0]] *= 1e-6
    # df_ref_rp[cols_ref_rp[0]] *= 1e-6
    # df_ref_yaw[cols_ref_yaw[0]] *= 1e-6

    axes[0].scatter(np.array(df_opt_rpy[cols_experiment[0]]), np.array(df_opt_rpy[cols_experiment[1]]), c='b', label = 'optmized ±2σ', alpha=0.4, s=10)
    axes[0].plot(np.array(df_ref_rpy[cols_experiment[0]]), np.array(df_ref_rpy[cols_experiment[1]]), c='r', label = 'gt')
    axes[0].plot(np.array(df_odom_rpy[cols_experiment[0]]), np.array(df_odom_rpy[cols_experiment[1]]), c='r', linestyle='--', label = 'odom')

    axes[0].legend()
    #plt.xlabel("Timestamp")
    axes[0].set_ylabel("Roll(rad)")
    axes[0].grid()
    axes[1].scatter(np.array(df_opt_rpy[cols_experiment[0]]), np.array(df_opt_rpy[cols_experiment[2]]), c='b', alpha=0.4, s=10)
    axes[1].plot(np.array(df_ref_rpy[cols_experiment[0]]), np.array(df_ref_rpy[cols_experiment[2]]), c='r')
    axes[1].plot(np.array(df_odom_rpy[cols_experiment[0]]), np.array(df_odom_rpy[cols_experiment[2]]), c='r', linestyle='--')

    axes[1].set_ylabel("Pitch(rad)")
    axes[1].grid()

    # yaw_odom = np.array(df_odom_rpy[cols_experiment[3]])
    # yaw_world = np.arctan2(np.sin(yaw_odom + target_odom_origin[3]),
    #                        np.cos(yaw_odom + target_odom_origin[3]))
    
    # Get rotation from target odom origin (world_T_odom)
    R_target_odom = target_odom_origin[:3, :3]
    # Get roll, pitch, yaw from df_odom_rpy
    roll = np.array(df_odom_rpy[cols_experiment[1]])
    pitch = np.array(df_odom_rpy[cols_experiment[2]])
    yaw = np.array(df_odom_rpy[cols_experiment[3]])

    # Build odom-local rotation matrices
    odom_rot_mats = R.from_euler('zyx', np.vstack((yaw, pitch, roll)).T).as_matrix()  # shape: (N, 3, 3)

    yaw_world = [R.from_matrix(R_target_odom @ R_odom).as_euler('zyx')[0] for R_odom in odom_rot_mats]

    axes[2].scatter(np.array(df_opt_rpy[cols_experiment[0]]),
                    np.array(df_opt_rpy[cols_experiment[3]]), c='b', alpha=0.4, s=10)
    axes[2].plot(np.array(df_ref_rpy[cols_experiment[0]]),
                 np.array(df_ref_rpy[cols_experiment[3]]), c='r')
    axes[2].plot(np.array(df_odom_rpy[cols_experiment[0]]),
                 yaw_world, c='r', linestyle='--', label='Odom (World Frame)')

    axes[2].fill_between(np.array(df_opt_rpy[cols_experiment[0]]),
                         np.array(df_opt_rpy[cols_experiment[-1]]) - 2.0*std_yaw,
                         np.array(df_opt_rpy[cols_experiment[-1]]) + 2.0*std_yaw,
                         color='blue', alpha=0.2, label='±2σ Uncertainty')
    axes[2].set_ylabel("Yaw(rad)")
    axes[2].grid()

    axes[2].set_xlabel("Time(s)")


    plt.savefig(path + '/attitude_t.png', bbox_inches='tight')

    plt.show() 

def plot_metrics(path, df_metrics, cols_metrics, t0, title, filename):

    fig, axes = plt.subplots(2, 1, figsize=(15, 15))

    # Subtract the first element of the timestamp column to start from 0
    df_metrics[cols_metrics[0]] -= t0

    # # Subtract the first element of the timestamp column to start from 0
    # df_metrics[cols_metrics[0]] *= 1e-6

    plt.title(title)

    axes[0].plot(np.array(df_metrics[cols_metrics[0]]), np.array(df_metrics[cols_metrics[3]]), c='r', label = 'rmse')

    axes[0].legend()
    #plt.xlabel("Timestamp")
    axes[0].set_ylabel("Rotational error (º)")
    axes[0].grid()
    axes[1].plot(np.array(df_metrics[cols_metrics[0]]), np.array(df_metrics[cols_metrics[-1]]), c='r')
    axes[1].set_ylabel("Translational error (m)")
    axes[1].grid()

    axes[1].set_xlabel("Time(s)")

    plt.savefig(path + f'/metrics_{filename}.png', bbox_inches='tight')

    plt.show()
    


def plot_experiment_data(path_experiment_data, path_folder, gt_available = "True", simulation = "True"):

    
    print("gt_available set to: " + gt_available)
    print("simulation set to: " + simulation)

    #Get names of the topics we want to plot
    time_data_setpoint = '__time'
    
    #This is ground truth data
    if simulation == "True":
        source_gt_frame_id = 'agv_gt'
        target_gt_frame_id = 'uav_gt'
    elif gt_available == "True":
        uav_gt_topic_name = '/dll_node/pose_estimation'
        agv_gt_topic_name = '/dll_node_arco/pose_estimation'

    #These are the odometry sources used for relative transform estimation
    odom_topic_uav = "/uav/odom"
    odom_topic_agv = "/agv/odom"   #"/arco/idmind_motors/odom" #"/agv/odom"

    tf_odom_agv = "arco/odom"

    #The origins of each local frame wrt to map 
    ####### Values for simulation ###########
    # (map coincides with AGV local frame in simulation)
    # target_odom_origin = pose_to_matrix(np.array([0.5,-0.5,2.0,0.0,0.0,0.524]))    #np.array([0.25,-0.25,2.0,0.0,0.0,0.17])
    # source_odom_origin = pose_to_matrix(np.array([0.0,0.0,0.0,0.0,0.0,0.0]))

    if(simulation == "True"):

        source_gt_x_data = f'/tf/world/{source_gt_frame_id}/translation/x'
        source_gt_y_data = f'/tf/world/{source_gt_frame_id}/translation/y'
        source_gt_z_data = f'/tf/world/{source_gt_frame_id}/translation/z'
        source_gt_q0_data = f'/tf/world/{source_gt_frame_id}/rotation/x'
        source_gt_q1_data = f'/tf/world/{source_gt_frame_id}/rotation/y'
        source_gt_q2_data = f'/tf/world/{source_gt_frame_id}/rotation/z'
        source_gt_q3_data = f'/tf/world/{source_gt_frame_id}/rotation/w'
        
        target_gt_x_data = f'/tf/world/{target_gt_frame_id}/translation/x'
        target_gt_y_data = f'/tf/world/{target_gt_frame_id}/translation/y'
        target_gt_z_data = f'/tf/world/{target_gt_frame_id}/translation/z'
        target_gt_q0_data = f'/tf/world/{target_gt_frame_id}/rotation/x'
        target_gt_q1_data = f'/tf/world/{target_gt_frame_id}/rotation/y'
        target_gt_q2_data = f'/tf/world/{target_gt_frame_id}/rotation/z'
        target_gt_q3_data = f'/tf/world/{target_gt_frame_id}/rotation/w'

        metrics_detR_data = '/optimization/metrics/data[0]'
        metrics_dett_data = '/optimization/metrics/data[1]'
        metrics_rmse_R_data = '/optimization/metrics/data[2]'
        metrics_rmse_t_data = '/optimization/metrics/data[3]'

        metrics_traj_detR_data = '/optimization/traj_metrics/data[0]'
        metrics_traj_dett_data = '/optimization/traj_metrics/data[1]'
        metrics_traj_rmse_R_data = '/optimization/traj_metrics/data[2]'
        metrics_traj_rmse_t_data = '/optimization/traj_metrics/data[3]'

    elif gt_available == "True":

        source_gt_x_data = f'{agv_gt_topic_name}/pose/position/x'
        source_gt_y_data = f'{agv_gt_topic_name}/pose/position/y'
        source_gt_z_data = f'{agv_gt_topic_name}/pose/position/z'
        source_gt_q0_data = f'{agv_gt_topic_name}/pose/orientation/x'
        source_gt_q1_data = f'{agv_gt_topic_name}/pose/orientation/y'
        source_gt_q2_data = f'{agv_gt_topic_name}/pose/orientation/z'
        source_gt_q3_data = f'{agv_gt_topic_name}/pose/orientation/w'

        target_gt_x_data = f'{uav_gt_topic_name}/pose/position/x'
        target_gt_y_data = f'{uav_gt_topic_name}/pose/position/y'
        target_gt_z_data = f'{uav_gt_topic_name}/pose/position/z'
        target_gt_q0_data = f'{uav_gt_topic_name}/pose/orientation/x'
        target_gt_q1_data = f'{uav_gt_topic_name}/pose/orientation/y'
        target_gt_q2_data = f'{uav_gt_topic_name}/pose/orientation/z'
        target_gt_q3_data = f'{uav_gt_topic_name}/pose/orientation/w'

        # tf map -> arco/odom
        source_odom_gt_x_data = f'/tf/map/{tf_odom_agv}/translation/x'
        source_odom_gt_y_data = f'/tf/map/{tf_odom_agv}/translation/y'
        source_odom_gt_z_data = f'/tf/map/{tf_odom_agv}/translation/z'
        source_odom_gt_q0_data = f'/tf/map/{tf_odom_agv}/rotation/x'
        source_odom_gt_q1_data = f'/tf/map/{tf_odom_agv}/rotation/y'
        source_odom_gt_q2_data = f'/tf/map/{tf_odom_agv}/rotation/z'
        source_odom_gt_q3_data = f'/tf/map/{tf_odom_agv}/rotation/w'
        

    max_timestamp = None

    target_odom_cov_x = f'{odom_topic_uav}/pose/covariance/[0;0]'
    target_odom_cov_y = f'{odom_topic_uav}/pose/covariance/[1;1]'
    target_odom_cov_z = f'{odom_topic_uav}/pose/covariance/[2;2]'
    target_odom_cov_yaw = f'{odom_topic_uav}/pose/covariance/[5;5]'

    source_odom_cov_x = f'{odom_topic_agv}/pose/covariance/[0;0]'
    source_odom_cov_y = f'{odom_topic_agv}/pose/covariance/[1;1]'
    source_odom_cov_z = f'{odom_topic_agv}/pose/covariance/[2;2]'
    source_odom_cov_yaw = f'{odom_topic_agv}/pose/covariance/[5;5]'

    #Odometry topic
    source_odom_x_data = f'{odom_topic_agv}/pose/pose/position/x'
    source_odom_y_data = f'{odom_topic_agv}/pose/pose/position/y'
    source_odom_z_data = f'{odom_topic_agv}/pose/pose/position/z'
    source_odom_q0_data = f'{odom_topic_agv}/pose/pose/orientation/x'
    source_odom_q1_data = f'{odom_topic_agv}/pose/pose/orientation/y'
    source_odom_q2_data = f'{odom_topic_agv}/pose/pose/orientation/z'
    source_odom_q3_data = f'{odom_topic_agv}/pose/pose/orientation/w'

    ##transformations arco/odom -> arco/base_link
    source_odom_x_data = f'/tf/{tf_odom_agv}/arco/base_link/translation/x'
    source_odom_y_data = f'/tf/{tf_odom_agv}/arco/base_link/translation/y'
    source_odom_z_data = f'/tf/{tf_odom_agv}/arco/base_link/translation/z'
    source_odom_q0_data = f'/tf/{tf_odom_agv}/arco/base_link/rotation/x'
    source_odom_q1_data = f'/tf/{tf_odom_agv}/arco/base_link/rotation/y'
    source_odom_q2_data = f'/tf/{tf_odom_agv}/arco/base_link/rotation/z'
    source_odom_q3_data = f'/tf/{tf_odom_agv}/arco/base_link/rotation/w'

    target_odom_x_data = f'{odom_topic_uav}/pose/pose/position/x'
    target_odom_y_data = f'{odom_topic_uav}/pose/pose/position/y'
    target_odom_z_data = f'{odom_topic_uav}/pose/pose/position/z'
    target_odom_q0_data = f'{odom_topic_uav}/pose/pose/orientation/x'
    target_odom_q1_data = f'{odom_topic_uav}/pose/pose/orientation/y'
    target_odom_q2_data = f'{odom_topic_uav}/pose/pose/orientation/z'
    target_odom_q3_data = f'{odom_topic_uav}/pose/pose/orientation/w'
    
    #Relative transform columns

    opt_T_target_source_x_data = '/eliko_optimization_node/optimized_T/pose/pose/position/x'
    opt_T_target_source_y_data = '/eliko_optimization_node/optimized_T/pose/pose/position/y'
    opt_T_target_source_z_data = '/eliko_optimization_node/optimized_T/pose/pose/position/z'
    opt_T_target_source_q0_data = '/eliko_optimization_node/optimized_T/pose/pose/orientation/x'
    opt_T_target_source_q1_data = '/eliko_optimization_node/optimized_T/pose/pose/orientation/y'
    opt_T_target_source_q2_data = '/eliko_optimization_node/optimized_T/pose/pose/orientation/z'
    opt_T_target_source_q3_data = '/eliko_optimization_node/optimized_T/pose/pose/orientation/w'

    covariance_x = '/eliko_optimization_node/optimized_T/pose/covariance[0]'
    covariance_y = '/eliko_optimization_node/optimized_T/pose/covariance[7]'
    covariance_z = '/eliko_optimization_node/optimized_T/pose/covariance[14]'
    covariance_yaw = '/eliko_optimization_node/optimized_T/pose/covariance[35]'

    ## Anchor columns

    anchor_target_x_data = "/pose_graph_node/uav_anchor/pose/pose/position/x"
    anchor_target_y_data = "/pose_graph_node/uav_anchor/pose/pose/position/y"
    anchor_target_z_data = "/pose_graph_node/uav_anchor/pose/pose/position/z"
    anchor_target_q0_data = "/pose_graph_node/uav_anchor/pose/pose/orientation/x"
    anchor_target_q1_data = "/pose_graph_node/uav_anchor/pose/pose/orientation/y"
    anchor_target_q2_data = "/pose_graph_node/uav_anchor/pose/pose/orientation/z"
    anchor_target_q3_data = "/pose_graph_node/uav_anchor/pose/pose/orientation/w"

    anchor_source_x_data = "/pose_graph_node/agv_anchor/pose/pose/position/x"
    anchor_source_y_data = "/pose_graph_node/agv_anchor/pose/pose/position/y"
    anchor_source_z_data = "/pose_graph_node/agv_anchor/pose/pose/position/z"
    anchor_source_q0_data = "/pose_graph_node/agv_anchor/pose/pose/orientation/x"
    anchor_source_q1_data = "/pose_graph_node/agv_anchor/pose/pose/orientation/y"
    anchor_source_q2_data = "/pose_graph_node/agv_anchor/pose/pose/orientation/z"
    anchor_source_q3_data = "/pose_graph_node/agv_anchor/pose/pose/orientation/w"

    ## Radar velocities
    radar_source_x_data = "/agv/Ego_Vel_Twist/twist/twist/linear/x"
    radar_source_y_data = "/agv/Ego_Vel_Twist/twist/twist/linear/y"
    radar_source_z_data = "/agv/Ego_Vel_Twist/twist/twist/linear/z"

    radar_target_x_data = "/uav/Ego_Vel_Twist/twist/twist/linear/x"
    radar_target_y_data = "/uav/Ego_Vel_Twist/twist/twist/linear/y"
    radar_target_z_data = "/uav/Ego_Vel_Twist/twist/twist/linear/z"

    radar_source_cols = [time_data_setpoint, radar_source_x_data, radar_source_y_data, radar_source_z_data]
    radar_target_cols = [time_data_setpoint, radar_target_x_data, radar_target_y_data, radar_target_z_data]

    pose_graph_agv_cols = ["id", "timestamp", "position_x", "position_y", "position_z", "orientation_x", "orientation_y", "orientation_z", "orientation_w"]
    pose_graph_uav_cols = ["id", "timestamp", "position_x", "position_y", "position_z", "orientation_x", "orientation_y", "orientation_z", "orientation_w"]

    anchor_target_uav_cols = [time_data_setpoint, anchor_target_x_data, anchor_target_y_data, anchor_target_z_data, anchor_target_q0_data, anchor_target_q1_data, anchor_target_q2_data, anchor_target_q3_data]
    anchor_source_agv_cols = [time_data_setpoint, anchor_source_x_data, anchor_source_y_data, anchor_source_z_data, anchor_source_q0_data, anchor_source_q1_data, anchor_source_q2_data, anchor_source_q3_data]

    #Get the pose graphs
    agv_csv_path = path_folder + "/agv_pose_graph.csv"
    uav_csv_path = path_folder + "/uav_pose_graph.csv"

    pose_graph_agv_df = read_pandas_df(agv_csv_path, pose_graph_agv_cols, timestamp_col="timestamp", max_timestamp=max_timestamp)
    pose_graph_uav_df = read_pandas_df(uav_csv_path, pose_graph_uav_cols, timestamp_col="timestamp", max_timestamp=max_timestamp)

    print("AGV pose graph shape:", pose_graph_agv_df.shape)
    print("UAV pose graph shape:", pose_graph_uav_df.shape)

    poses_agv = load_pose_graph(pose_graph_agv_df, pose_graph_agv_cols)
    poses_uav = load_pose_graph(pose_graph_uav_df, pose_graph_uav_cols)

    #Get the anchors
    anchor_target_df = read_pandas_df(path_experiment_data, anchor_target_uav_cols, timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
    anchor_source_df = read_pandas_df(path_experiment_data, anchor_source_agv_cols, timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)

    #Get odometry data
    columns_source_odom_data = [time_data_setpoint, source_odom_x_data, source_odom_y_data, source_odom_z_data, source_odom_q0_data, source_odom_q1_data,source_odom_q2_data, source_odom_q3_data   ]
    source_odom_data_df = read_pandas_df(path_experiment_data, columns_source_odom_data, 
                                           timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
    
    columns_target_odom_data = [time_data_setpoint, target_odom_x_data, target_odom_y_data, target_odom_z_data, target_odom_q0_data , target_odom_q1_data ,target_odom_q2_data ,target_odom_q3_data ]
    target_odom_data_df = read_pandas_df(path_experiment_data, columns_target_odom_data, 
                                           timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
    
    columns_odom_covariance = [time_data_setpoint, target_odom_cov_x, target_odom_cov_y, target_odom_cov_z, target_odom_cov_yaw]

    covariance_odom_data_df = read_pandas_df(path_experiment_data, columns_odom_covariance, 
                                        timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
    
    # Get the relative transform
    columns_t_That_s_data = [time_data_setpoint, opt_T_target_source_x_data, opt_T_target_source_y_data, opt_T_target_source_z_data, opt_T_target_source_q0_data, opt_T_target_source_q1_data, opt_T_target_source_q2_data, opt_T_target_source_q3_data]

    t_That_s_data_df = read_pandas_df(path_experiment_data, columns_t_That_s_data, 
                                      timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)

    #Insert the initial solution (identity transform) as the first row.
    if not t_That_s_data_df.empty:
        initial_timestamp = t_That_s_data_df[time_data_setpoint].iloc[0]
    else:
        initial_timestamp = 0  # or set a desired default timestamp

    initial_row = {
        time_data_setpoint: initial_timestamp - 0.5,
        opt_T_target_source_x_data: 0.0,
        opt_T_target_source_y_data: 0.0,
        opt_T_target_source_z_data: 0.0,
        opt_T_target_source_q0_data: 0.0,
        opt_T_target_source_q1_data: 0.0,
        opt_T_target_source_q2_data: 0.0,
        opt_T_target_source_q3_data: 1.0
    }
    initial_df = pd.DataFrame([initial_row])
    t_That_s_data_df = pd.concat([initial_df, t_That_s_data_df], ignore_index=True)

    #Append yaw to the last column of the estimated transform
    t_That_s_data_df = add_yaw_to_df(t_That_s_data_df,
                               opt_T_target_source_q0_data,
                               opt_T_target_source_q1_data,
                               opt_T_target_source_q2_data,
                               opt_T_target_source_q3_data)
    columns_t_That_s_data = columns_t_That_s_data + ["yaw"]

    #Get relative transform covariance
    columns_covariance = [time_data_setpoint, covariance_x, covariance_y, covariance_z, covariance_yaw]
    covariance_data_df = read_pandas_df(path_experiment_data, columns_covariance, 
                                        timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)

    # Insert the initial covariance row (identity covariance, i.e. all ones) so that it aligns with the
    # identity row you add to the optimized transform dataframe.
    if not covariance_data_df.empty:
        initial_timestamp_cov = covariance_data_df[time_data_setpoint].iloc[0]
    else:
        initial_timestamp_cov = 0  # or another default value
    initial_cov_row = {
        time_data_setpoint: initial_timestamp_cov - 0.5,
        covariance_x: 1.0,
        covariance_y: 1.0,
        covariance_z: 1.0,
        covariance_yaw: 1.0
    }

    initial_cov_df = pd.DataFrame([initial_cov_row])
    covariance_data_df = pd.concat([initial_cov_df, covariance_data_df], ignore_index=True)

    #Get the radar velocities
    radar_source_df = read_pandas_df(path_experiment_data, radar_source_cols, 
                                      timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
    radar_target_df = read_pandas_df(path_experiment_data, radar_target_cols, 
                                      timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
    
    
    if gt_available == "True":

        # Plot 3D representation

        columns_source_gt_data = [time_data_setpoint, source_gt_x_data, source_gt_y_data, source_gt_z_data, source_gt_q0_data, source_gt_q1_data,source_gt_q2_data, source_gt_q3_data   ]
        
        columns_target_gt_data = [time_data_setpoint, target_gt_x_data, target_gt_y_data, target_gt_z_data, target_gt_q0_data , target_gt_q1_data ,target_gt_q2_data ,target_gt_q3_data ]
        
        source_gt_data_df = read_pandas_df(path_experiment_data, columns_source_gt_data, 
                                           timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
        
        
        target_gt_data_df = read_pandas_df(path_experiment_data, columns_target_gt_data, 
                                           timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
        
        # Extract origin from the first AGV GT pose
        first_agv_row = source_gt_data_df.iloc[0]
        agv_pos = first_agv_row[[source_gt_x_data, source_gt_y_data, source_gt_z_data]].values
        agv_quat = first_agv_row[[source_gt_q0_data, source_gt_q1_data, source_gt_q2_data, source_gt_q3_data]].values
        agv_rpy = R.from_quat(agv_quat).as_euler('zyx', degrees=False)  # yaw, pitch, roll
        source_odom_origin = pose_to_matrix(np.concatenate((agv_pos, agv_rpy[::-1])))  # [x, y, z, roll, pitch, yaw]

        # Extract origin from the first UAV GT pose
        first_uav_row = target_gt_data_df.iloc[0]
        uav_pos = first_uav_row[[target_gt_x_data, target_gt_y_data, target_gt_z_data]].values
        uav_quat = first_uav_row[[target_gt_q0_data, target_gt_q1_data, target_gt_q2_data, target_gt_q3_data]].values
        uav_rpy = R.from_quat(uav_quat).as_euler('zyx', degrees=False)
        target_odom_origin = pose_to_matrix(np.concatenate((uav_pos, uav_rpy[::-1])))  # [x, y, z, roll, pitch, yaw]
        
        t0 = target_gt_data_df[columns_target_gt_data[0]].iloc[0]

        w_That_t_data_df = compute_That_w_t(time_data_setpoint, target_gt_data_df, target_odom_data_df, 
                                                           t_That_s_data_df, columns_target_gt_data, 
                                                           columns_target_odom_data, columns_t_That_s_data)

        
        columns_w_That_t_data = [time_data_setpoint, "x", "y", "z", "qx", "qy", "qz", "qw", "yaw"]


        # plot_3d_scatter(path_folder, source_gt_data_df, source_odom_data_df, 
        #                 target_gt_data_df, target_odom_data_df, w_That_t_data_df, 
        #                 columns_source_gt_data, columns_source_odom_data, columns_target_gt_data, 
        #                 columns_target_odom_data, columns_w_That_t_data, source_odom_origin, target_odom_origin)
                
        # #Plot position vs time
        # plot_pose_temporal_evolution(path_folder, target_gt_data_df, w_That_t_data_df, 
        #                              covariance_data_df, covariance_odom_data_df, target_odom_data_df, columns_target_gt_data, 
        #                              columns_w_That_t_data, columns_covariance, columns_odom_covariance, columns_target_odom_data, 
        #                              t0, target_odom_origin)

        # #Plot attitude vs time

        rpy_attitude_opt_data_df = w_That_t_data_df[["qw", "qx", "qy", "qz"]].apply(lambda row: pd.Series(quaternion_to_euler_angles(row), index=['roll', 'pitch', 'yaw']), axis=1)    
        rpy_attitude_gt_data_df = target_gt_data_df[[target_gt_q3_data, target_gt_q0_data, target_gt_q1_data, target_gt_q2_data]].apply(lambda row: pd.Series(quaternion_to_euler_angles(row), index=['roll', 'pitch', 'yaw']), axis=1)    
        rpy_attitude_odom_data_df = target_odom_data_df[[target_odom_q3_data, target_odom_q0_data, target_odom_q1_data, target_odom_q2_data]].apply(lambda row: pd.Series(quaternion_to_euler_angles(row), index=['roll', 'pitch', 'yaw']), axis=1)    


        # Add the timestamp column to the resulting DataFrame
        rpy_attitude_opt_data_df[time_data_setpoint] = w_That_t_data_df[time_data_setpoint]
        rpy_attitude_gt_data_df[time_data_setpoint] = target_gt_data_df[time_data_setpoint]
        rpy_attitude_odom_data_df[time_data_setpoint] = target_odom_data_df[time_data_setpoint]


        #Add the previously computed yaw column to optimized data
        rpy_attitude_opt_data_df["yaw"] = w_That_t_data_df["yaw"]

        # Reorder the columns to make timestamp the first column
        rpy_attitude_cols = [time_data_setpoint, "roll", "pitch", "yaw"]
        
        rpy_attitude_opt_data_df = rpy_attitude_opt_data_df[rpy_attitude_cols]
        rpy_attitude_gt_data_df = rpy_attitude_gt_data_df[rpy_attitude_cols]
        rpy_attitude_odom_data_df = rpy_attitude_odom_data_df[rpy_attitude_cols]

        # plot_attitude_temporal_evolution(path_folder, rpy_attitude_gt_data_df, rpy_attitude_opt_data_df, 
        #                                  covariance_data_df, covariance_odom_data_df, rpy_attitude_odom_data_df, 
        #                                  rpy_attitude_cols, columns_covariance, columns_odom_covariance, target_odom_origin)

        if simulation == "True":

            columns_metrics = [time_data_setpoint, metrics_detR_data, metrics_dett_data, metrics_rmse_R_data, metrics_rmse_t_data]
            columns_traj_metrics = [time_data_setpoint, metrics_traj_detR_data, metrics_traj_dett_data, metrics_traj_rmse_R_data, metrics_traj_rmse_t_data]

            metrics_df_data = read_pandas_df(path_experiment_data, columns_metrics,
                                                    timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
                    
            metrics_traj_df_data = read_pandas_df(path_experiment_data, columns_traj_metrics,
                                                    timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
            
            #plot metrics
            plot_metrics(path_folder, metrics_df_data, columns_metrics, t0, "Relative transform errors", "transform")
            plot_metrics(path_folder, metrics_traj_df_data, columns_traj_metrics, t0, "Overall trajectory errors", "trajectory")
        
        #Transform local poses of graph to a global frame of reference using the anchors
        poses_uav_world = compute_poses_local_to_world(time_data_setpoint, poses_uav, anchor_target_df, anchor_target_uav_cols, use_last=False, use_gt = False, gt_anchor = target_odom_origin)
        poses_agv_world = compute_poses_local_to_world(time_data_setpoint, poses_agv, anchor_source_df, anchor_source_agv_cols, use_last=False, use_gt = False, gt_anchor = source_odom_origin)
        
        #plot posegraphs
        plot_posegraph_temporal(path_folder, "posegraph_agv_world_poses", poses_agv_world, source_gt_data_df, columns_source_gt_data)
        plot_posegraph_temporal(path_folder, "posegraph_uav_world_poses", poses_uav_world, target_gt_data_df, columns_target_gt_data)
        
        plot_posegraph_temporal(path_folder, "posegraph_agv_local_poses", poses_agv, source_gt_data_df, columns_source_gt_data, source_odom_origin)
        plot_posegraph_temporal(path_folder, "posegraph_uav_local_poses", poses_uav, target_gt_data_df, columns_target_gt_data, target_odom_origin)
        
        plot_posegraphs_3d(path_folder, "posegraph_3d_poses", poses_agv_world, poses_uav_world, source_gt_data_df, target_gt_data_df, columns_source_gt_data, columns_target_gt_data)
        plot_posegraphs_3d_side_by_side(path_folder, "local_posegraph_3d_poses", poses_agv, poses_uav)

        rmse_agv_pos, rmse_agv_yaw = compute_rmse(poses_agv_world, source_gt_data_df, pose_graph_agv_cols, columns_source_gt_data)
        rmse_uav_pos, rmse_uav_yaw = compute_rmse(poses_uav_world, target_gt_data_df, pose_graph_uav_cols, columns_target_gt_data)
        print(f'RMSE AGV ----> Translation: {rmse_agv_pos} m, Rotation: {np.rad2deg(rmse_agv_yaw)} º')
        print(f'RMSE UAV ----> Translation: {rmse_uav_pos} m, Rotation: {np.rad2deg(rmse_uav_yaw)} º')

        plot_transform(path_folder, t_That_s_data_df, covariance_data_df, columns_t_That_s_data, columns_covariance, t0, source_odom_origin, target_odom_origin)
    
        #Plot radar
        plot_radar_velocities(path_folder, radar_source_df, radar_source_cols, label="AGV Radar", filename="radar_velocity_agv", gt_df = source_gt_data_df, gt_cols = columns_source_gt_data)
        plot_radar_velocities(path_folder, radar_target_df, radar_target_cols, label="UAV Radar", filename="radar_velocity_uav", gt_df = target_gt_data_df, gt_cols = columns_target_gt_data)    
    
    else:
        t0 = target_odom_data_df[columns_target_odom_data[0]].iloc[0]
        #Transform local poses of graph to a global frame of reference using the anchors
        poses_uav_world = compute_poses_local_to_world(time_data_setpoint, poses_uav, anchor_target_df, anchor_target_uav_cols, use_last=True)
        poses_agv_world = compute_poses_local_to_world(time_data_setpoint, poses_agv, anchor_source_df, anchor_source_agv_cols, use_last=True)
        plot_posegraph_temporal(path_folder, "posegraph_agv_world_poses", poses_agv_world)
        plot_posegraph_temporal(path_folder, "posegraph_uav_world_poses", poses_uav_world)
        plot_posegraph_temporal(path_folder, "posegraph_agv_local_poses", poses_agv, source_odom_data_df, columns_source_odom_data)
        plot_posegraph_temporal(path_folder, "posegraph_uav_local_poses", poses_uav, target_odom_data_df, columns_target_odom_data)
        plot_posegraphs_3d(path_folder, "posegraph_3d_poses", poses_agv_world, poses_uav_world)
        plot_transform(path_folder, t_That_s_data_df, covariance_data_df, columns_t_That_s_data, columns_covariance, t0)



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


def compute_rmse(pose_graph, gt_df, pose_graph_cols, gt_cols):

    pose_graph_df = graph_dict_to_df(pose_graph)

    #Substract initial timestamp so both start from 0
    pose_graph_df[pose_graph_cols[1]] -= pose_graph_df[pose_graph_cols[1]].iloc[0]
    gt_df[gt_cols[0]] = gt_df[gt_cols[0]] - gt_df[gt_cols[0]].iloc[0]

    # Merge the two DataFrames based on nearest timestamp.
    merged = pd.merge_asof(pose_graph_df, gt_df, left_on=pose_graph_cols[1], right_on=gt_cols[0], direction="nearest")

    gt_yaw = []
    for i in range(len(merged)):
        quat = [merged[gt_cols[4]].iloc[i], merged[gt_cols[5]].iloc[i],
                merged[gt_cols[6]].iloc[i], merged[gt_cols[7]].iloc[i]]
        yaw_val = R.from_quat(quat).as_euler('zyx', degrees=False)[0]
        gt_yaw.append(yaw_val)
    gt_yaw = np.array(gt_yaw)

    error_x = merged[gt_cols[1]] - merged[pose_graph_cols[2]]
    error_y = merged[gt_cols[2]] - merged[pose_graph_cols[3]]
    error_z = merged[gt_cols[3]] - merged[pose_graph_cols[4]]
 
    rmse_pos = np.sqrt(np.mean(np.square(error_x) + np.square(error_y) + np.square(error_z)))

    error_yaw = gt_yaw - merged['orientation_yaw']
    error_yaw = np.arctan2(np.sin(error_yaw), np.cos(error_yaw))
    rmse_yaw = np.sqrt(np.mean(np.square(error_yaw)))

    return rmse_pos, rmse_yaw

def plot_posegraphs_3d(path, filename, poses_agv, poses_uav, gt_agv = None, gt_uav = None, cols_gt_agv = None, cols_gt_uav = None):
    """
    Plots the 3D trajectories for both AGV and UAV.

    Parameters:
        poses_agv (dict): Dictionary of AGV poses. Each key is an index and each value is a dict with:
                          - 'timestamp': float (seconds)
                          - 'position': np.array of shape (3,)
                          - 'orientation': np.array of shape (4,) [optional]
        poses_uav (dict): Dictionary of UAV poses, with the same structure as poses_agv.
    """
    # Filter valid AGV poses (i.e. non-None and non-NaN timestamp and valid position)
    agv_data = []
    for idx, pose in poses_agv.items():
        if (pose.get('timestamp') is not None and 
            pose.get('position') is not None and 
            not np.isnan(pose['timestamp'])):
            agv_data.append((pose['timestamp'], pose['position']))
    if len(agv_data) == 0:
        print("No valid AGV poses found.")
        return
    agv_data.sort(key=lambda x: x[0])
    # Stack positions: each element is a (3,) vector.
    agv_positions = np.vstack([pos for _, pos in agv_data])
    
    # Filter valid UAV poses.
    uav_data = []
    for idx, pose in poses_uav.items():
        if (pose.get('timestamp') is not None and 
            pose.get('position') is not None and 
            not np.isnan(pose['timestamp'])):
            uav_data.append((pose['timestamp'], pose['position']))
    if len(uav_data) == 0:
        print("No valid UAV poses found.")
        return
    uav_data.sort(key=lambda x: x[0])
    uav_positions = np.vstack([pos for _, pos in uav_data])
    
    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot AGV trajectory (e.g., red)
    ax.plot(agv_positions[:, 0], agv_positions[:, 1], agv_positions[:, 2],
            '-o', color='red', label='AGV Trajectory')
    
    # Plot UAV trajectory (e.g., blue)
    ax.plot(uav_positions[:, 0], uav_positions[:, 1], uav_positions[:, 2],
            '-o', color='blue', label='UAV Trajectory')
    
    if gt_agv is not None and cols_gt_agv is not None:
        ax.plot(gt_agv[cols_gt_agv[1]], gt_agv[cols_gt_agv[2]], gt_agv[cols_gt_agv[3]], c='r', label='ref source', linewidth=2)

    if gt_uav is not None and cols_gt_uav is not None:
        ax.plot(gt_uav[cols_gt_uav[1]], gt_uav[cols_gt_uav[2]], gt_uav[cols_gt_uav[3]], c='b', label='ref target', linewidth=2)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.title("3D Trajectories of AGV and UAV")
    plt.tight_layout()

    plt.savefig(path + f'/{filename}.svg', format = 'svg', bbox_inches='tight')
    plt.savefig(path + f'/{filename}.png', format = 'png', bbox_inches='tight')

    plt.show()

def plot_posegraph_temporal(path, filename, poses, gt = None, cols_gt = None, gt_origin = None):
    """
    Plots the temporal evolution of x, y, z, and yaw given a dictionary of poses.
    
    Parameters:
        poses (dict): A dictionary where each key is an array index and each value is a dict with:
                      - 'timestamp': float (seconds)
                      - 'position': np.array of shape (3,) containing [x, y, z]
                      - 'orientation': np.array of shape (4,) representing the quaternion [x, y, z, w]
    """
        # Filter out incomplete poses.
    valid_poses = {idx: pose for idx, pose in poses.items()
                   if (pose['timestamp'] is not None and
                       pose['position'] is not None and
                       pose['orientation'] is not None and
                       not np.isnan(pose['timestamp']))}

    if not valid_poses:
        print("No valid poses to plot.")
        return

    # Gather data (filtering out entries with missing values)
    data = []
    for idx, pose in valid_poses.items():
        if (pose['timestamp'] is not None and
            pose['position'] is not None and
            pose['orientation'] is not None):
            t = pose['timestamp']
            x, y, z = pose['position']
            # Compute yaw from quaternion using 'zyx' convention, where the first element is yaw.
            r = R.from_quat(pose['orientation'])
            # as_euler('zyx') returns angles in order [yaw, pitch, roll]
            yaw = r.as_euler('zyx', degrees=False)[0]
            data.append((t, x, y, z, yaw))
    
    # Sort by timestamp
    data.sort(key=lambda x: x[0])
    data = np.array(data)
    timestamps = data[:, 0] - data[0,0]
    xs = data[:, 1]
    ys = data[:, 2]
    zs = data[:, 3]
    yaws = data[:, 4]


    # Plot each variable vs. time
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    axs[0].plot(timestamps, xs, marker='o', linestyle='-')
    axs[0].set_ylabel('X (m)')
    axs[0].grid()

    axs[1].plot(timestamps, ys, marker='o', linestyle='-')
    axs[1].set_ylabel('Y (m)')
    axs[1].grid()

    axs[2].plot(timestamps, zs, marker='o', linestyle='-')
    axs[2].set_ylabel('Z (m)')
    axs[2].grid()


    axs[3].plot(timestamps, yaws, marker='o', linestyle='-')
    axs[3].set_ylabel('Yaw (rad)')
    axs[3].grid()

    axs[3].set_xlabel('Time (s)')

    if gt is not None and cols_gt is not None:

        gt[cols_gt[0]] = gt[cols_gt[0]] - gt[cols_gt[0]].iloc[0]

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
            
            # axs[0].plot(np.array(gt[cols_gt[0]]), np.array(gt[cols_gt[1]]), c='r', label = 'target ref')
            # axs[1].plot(np.array(gt[cols_gt[0]]), np.array(gt[cols_gt[2]]), c='r', label = 'target ref')
            # axs[2].plot(np.array(gt[cols_gt[0]]), np.array(gt[cols_gt[3]]), c='r', label = 'target ref')
            
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
            #axs[3].plot(np.array(gt[cols_gt[0]]), gt_yaw, c='r', label = 'target ref')


        axs[0].plot(np.array(gt[cols_gt[0]]), gt_x, c='r', label = 'target ref')
        axs[1].plot(np.array(gt[cols_gt[0]]), gt_y, c='r', label = 'target ref')
        axs[2].plot(np.array(gt[cols_gt[0]]), gt_z, c='r', label = 'target ref')
        axs[3].plot(np.array(gt[cols_gt[0]]), gt_yaw, c='r', label = 'target ref')

    
    plt.suptitle("Temporal Evolution of Pose (x, y, z, yaw)")

    plt.savefig(path + f'/{filename}.svg', format = 'svg', bbox_inches='tight')
    plt.savefig(path + f'/{filename}.png', format = 'png', bbox_inches='tight')

    plt.show()

def plot_posegraphs_3d_side_by_side(path, filename, poses_agv, poses_uav, gt_agv=None, gt_uav=None, cols_gt_agv=None, cols_gt_uav=None):
    """
    Plots the 3D trajectories for AGV and UAV side-by-side in different subplots.

    Parameters:
        poses_agv (dict): Dictionary of AGV poses. Each key is an index and each value is a dict with:
                          - 'timestamp': float (seconds)
                          - 'position': np.array of shape (3,)
        poses_uav (dict): Dictionary of UAV poses, with the same structure as poses_agv.
        gt_agv (np.array, optional): Ground truth data for AGV.
        gt_uav (np.array, optional): Ground truth data for UAV.
        cols_gt_agv (list, optional): List of indices to extract columns from gt_agv.
        cols_gt_uav (list, optional): List of indices to extract columns from gt_uav.
    """
    # Process AGV poses.
    agv_data = []
    for idx, pose in poses_agv.items():
        if (pose.get('timestamp') is not None and 
            pose.get('position') is not None and 
            not np.isnan(pose['timestamp'])):
            agv_data.append((pose['timestamp'], pose['position']))
    if len(agv_data) == 0:
        print("No valid AGV poses found.")
        return
    agv_data.sort(key=lambda x: x[0])
    agv_positions = np.vstack([pos for _, pos in agv_data])
    
    # Process UAV poses.
    uav_data = []
    for idx, pose in poses_uav.items():
        if (pose.get('timestamp') is not None and 
            pose.get('position') is not None and 
            not np.isnan(pose['timestamp'])):
            uav_data.append((pose['timestamp'], pose['position']))
    if len(uav_data) == 0:
        print("No valid UAV poses found.")
        return
    uav_data.sort(key=lambda x: x[0])
    uav_positions = np.vstack([pos for _, pos in uav_data])
    
    # Create figure with 2 side-by-side 3D subplots.
    fig = plt.figure(figsize=(14, 6))
    ax_agv = fig.add_subplot(121, projection='3d')
    ax_uav = fig.add_subplot(122, projection='3d')
    
    # Plot AGV trajectory.
    ax_agv.plot(agv_positions[:, 0], agv_positions[:, 1], agv_positions[:, 2],
                '-o', color='red', label='AGV Trajectory')
    ax_agv.set_title("AGV Trajectory")
    ax_agv.set_xlabel("X (m)")
    ax_agv.set_ylabel("Y (m)")
    ax_agv.set_zlabel("Z (m)")
    ax_agv.legend()
    
    # Optionally, plot ground truth for AGV.
    if gt_agv is not None and cols_gt_agv is not None:
        ax_agv.plot(gt_agv[cols_gt_agv[1]], gt_agv[cols_gt_agv[2]], gt_agv[cols_gt_agv[3]], 
                    c='r', label='AGV Ground Truth', linewidth=2)
    
    # Plot UAV trajectory.
    ax_uav.plot(uav_positions[:, 0], uav_positions[:, 1], uav_positions[:, 2],
                '-o', color='blue', label='UAV Trajectory')
    ax_uav.set_title("UAV Trajectory")
    ax_uav.set_xlabel("X (m)")
    ax_uav.set_ylabel("Y (m)")
    ax_uav.set_zlabel("Z (m)")
    ax_uav.legend()
    
    # Optionally, plot ground truth for UAV.
    if gt_uav is not None and cols_gt_uav is not None:
        ax_uav.plot(gt_uav[cols_gt_uav[1]], gt_uav[cols_gt_uav[2]], gt_uav[cols_gt_uav[3]],
                    c='b', label='UAV Ground Truth', linewidth=2)
    
    plt.suptitle("3D Trajectories of AGV and UAV")
    plt.tight_layout()
    plt.savefig(path + f'/{filename}.png', format='png', bbox_inches='tight')
    plt.show()


def plot_radar_velocities(path, radar_df, cols_radar, label="Radar", filename="radar_velocity",
                          gt_df=None, gt_cols=None):
    """
    Plots smoothed radar linear velocities over time and optionally compares to estimated ground truth velocities.

    Parameters:
        path (str): Directory to save the plot.
        radar_df (pd.DataFrame): DataFrame with radar velocity data.
        cols_radar (list): [timestamp, vx, vy, vz] column names in radar_df.
        label (str): Label prefix for legends.
        filename (str): Output filename (without extension).
        gt_df (pd.DataFrame, optional): Ground truth pose data (position over time).
        gt_cols (list, optional): Column names [timestamp, x, y, z] in gt_df.
    """

    radar_df = radar_df.copy()
    radar_df[cols_radar[0]] -= radar_df[cols_radar[0]].iloc[0]
    time = radar_df[cols_radar[0]].values
    vx = radar_df[cols_radar[1]].rolling(window=10, center=True).mean()
    vy = radar_df[cols_radar[2]].rolling(window=10, center=True).mean()
    vz = radar_df[cols_radar[3]].rolling(window=10, center=True).mean()

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    plt.suptitle(f"{label} Linear Velocities (Smoothed)")

    # Estimate ground truth velocities by finite difference (if GT is provided)
    if gt_df is not None and gt_cols is not None:
        gt_df = gt_df.copy()
        gt_df[gt_cols[0]] -= gt_df[gt_cols[0]].iloc[0]
        gt_df = gt_df.sort_values(gt_cols[0]).reset_index(drop=True)

        gt_time = gt_df[gt_cols[0]].values
        dt = np.gradient(gt_time)  # time deltas

        vx_gt = np.gradient(gt_df[gt_cols[1]].values, dt)
        vy_gt = np.gradient(gt_df[gt_cols[2]].values, dt)
        vz_gt = np.gradient(gt_df[gt_cols[3]].values, dt)

        # Interpolate GT velocities to radar timestamps
        vx_gt_interp = np.interp(time, gt_time, vx_gt)
        vy_gt_interp = np.interp(time, gt_time, vy_gt)
        vz_gt_interp = np.interp(time, gt_time, vz_gt)

        gt_available = True
    else:
        gt_available = False

    # Plot X velocity
    axes[0].plot(time, vx, color='b', label=f"{label} Vx")
    if gt_available:
        axes[0].plot(time, vx_gt_interp, color='r', linestyle='--', label='GT Vx')
    axes[0].set_ylabel("Vx (m/s)")
    axes[0].legend()
    axes[0].grid()

    # Plot Y velocity
    axes[1].plot(time, vy, color='b', label=f"{label} Vy")
    if gt_available:
        axes[1].plot(time, vy_gt_interp, color='r', linestyle='--', label='GT Vy')
    axes[1].set_ylabel("Vy (m/s)")
    axes[1].legend()
    axes[1].grid()

    # Plot Z velocity
    axes[2].plot(time, vz, color='b', label=f"{label} Vz")
    if gt_available:
        axes[2].plot(time, vz_gt_interp, color='r', linestyle='--', label='GT Vz')
    axes[2].set_ylabel("Vz (m/s)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid()

    plt.savefig(path + f"/{filename}.png", bbox_inches='tight')
    plt.show()

def load_pose_graph(df, cols):
    """
    Reads a CSV file of pose graph data and returns a dictionary of poses.
    
    The CSV is assumed to have columns:
      - id, timestamp, frame_id,
      - position_x, position_y, position_z,
      - orientation_x, orientation_y, orientation_z, orientation_w,
      - (optionally covariance columns, which we ignore here)
      
    Returns:
        dict: Keys are pose indices (int), and each value is a dict with keys:
              'timestamp' (float),
              'position' (np.array of shape (3,)),
              'orientation' (np.array of shape (4,)),
              'yaw' (float, computed from the quaternion).
    """
    poses = {}
    for _, row in df.iterrows():
        idx = int(row[cols[0]])
        timestamp = row[cols[1]]
        position = np.array([row[cols[2]], row[cols[3]], row[cols[4]]])
        orientation = np.array([row[cols[5]], row[cols[6]], row[cols[7]], row[cols[8]]])
        # Compute yaw using a 'zyx' Euler conversion; the first element is yaw.
        yaw = R.from_quat(orientation).as_euler('zyx', degrees=False)[0]
        poses[idx] = {
            'timestamp': timestamp,
            'position': position,
            'orientation': orientation,
            'yaw': yaw
        }
    return poses


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