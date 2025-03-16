#!/usr/bin/env python3

import sys
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import re


def compute_That_w_t(timestamp, source_gt_data_df, source_odom_data_df, target_gt_data_df, t_That_s_data_df, columns_source_gt_data, columns_source_odom_data, columns_target_gt_data, columns_t_That_s_data):
    """
    Computes That_w_t for each row in t_That_s_data_df using the latest available source_gt_data_df.

    Parameters:
        source_gt_data_df (pd.DataFrame): DataFrame with ground truth source data. Must include timestamp column.
        t_That_s_data_df (pd.DataFrame): DataFrame with optimized That_t_s data. Must include timestamp column.

    Returns:
        pd.DataFrame: DataFrame containing That_w_t positions (x, y, z).
    """
    That_w_t_list = []
    T_t_s_list = []

    # Ensure data is sorted by timestamp
    source_gt_data_df = source_gt_data_df.sort_values(timestamp).reset_index(drop=True)
    source_odom_data_df = source_odom_data_df.sort_values(timestamp).reset_index(drop=True)
    t_That_s_data_df = t_That_s_data_df.sort_values(timestamp).reset_index(drop=True)

    for _, row in t_That_s_data_df.iterrows():
        # Find the latest source_gt_data_df row with a timestamp <= current t_That_s_data_df timestamp
        latest_source_idx = source_gt_data_df[source_gt_data_df[timestamp] <= row[timestamp]].index.max()
        latest_source_odom_idx = source_odom_data_df[source_odom_data_df[timestamp] <= row[timestamp]].index.max()

        if latest_source_idx is not None and latest_source_odom_idx is not None:  # Ensure there's a valid matching source data
            # Get the corresponding source and t_That_s data
            source_row = source_gt_data_df.iloc[latest_source_idx]
            source_odom_row = source_odom_data_df.iloc[latest_source_odom_idx]

            target_row = target_gt_data_df.iloc[latest_source_idx]
            T_w_t = np.eye(4)
            T_w_s = np.eye(4)
            T_w_s_odom = np.eye(4)
            T_t_s = np.eye(4)
            That_t_s = np.eye(4)

            # Populate T_w_s
            T_w_s[:3, 3] = source_row[[columns_source_gt_data[1], columns_source_gt_data[2], columns_source_gt_data[3]]].values
            q_w_s = source_row[[columns_source_gt_data[4], columns_source_gt_data[5], columns_source_gt_data[6], columns_source_gt_data[7]]].values
            T_w_s[:3, :3] = R.from_quat(q_w_s).as_matrix()

            # Populate T_w_s_odom
            T_w_s_odom[:3, 3] = source_odom_row[[columns_source_odom_data[1], columns_source_odom_data[2], columns_source_odom_data[3]]].values
            q_w_s_odom = source_odom_row[[columns_source_odom_data[4], columns_source_odom_data[5], columns_source_odom_data[6], columns_source_odom_data[7]]].values
            T_w_s_odom[:3, :3] = R.from_quat(q_w_s_odom).as_matrix()

            # Populate T_w_t
            T_w_t[:3, 3] = target_row[[columns_target_gt_data[1], columns_target_gt_data[2], columns_target_gt_data[3]]].values
            q_w_t = target_row[[columns_target_gt_data[4], columns_target_gt_data[5], columns_target_gt_data[6], columns_target_gt_data[7]]].values
            T_w_t[:3, :3] = R.from_quat(q_w_t).as_matrix()

            # Populate That_t_s
            That_t_s[:3, 3] = row[[columns_t_That_s_data[1], columns_t_That_s_data[2], columns_t_That_s_data[3]]].values
            q_hat = row[[columns_t_That_s_data[4], columns_t_That_s_data[5], columns_t_That_s_data[6], columns_t_That_s_data[7]]].values
            That_t_s[:3, :3] = R.from_quat(q_hat).as_matrix()

            #Compute T_t_s -> ground truth information
            T_t_s = np.linalg.inv(T_w_t) @ T_w_s
            
             # Compute That_w_t -> computed from odometry
            That_w_t = T_w_s_odom @ np.linalg.inv(That_t_s)

            translation = That_w_t[:3, 3]  # Extract the translation part
            rotation = R.from_matrix(That_w_t[:3, :3]).as_quat()  # Extract rotation as quaternion
            yaw = R.from_matrix(That_w_t[:3, :3]).as_euler('zyx', degrees=False)[0]  # Extract yaw (rotation around Z-axis)

            That_w_t_list.append([row[timestamp], *translation, *rotation, yaw])  # Include timestamp, translation, and rotation

            # Extract translation, rotation, and yaw for T_t_s
            translation = T_t_s[:3, 3]
            rotation = R.from_matrix(T_t_s[:3, :3]).as_quat()
            yaw = R.from_matrix(T_t_s[:3, :3]).as_euler('zyx', degrees=False)[0]  # Extract yaw (rotation around Z-axis)
            T_t_s_list.append([row[timestamp], *translation, *rotation, yaw])

    # Create a DataFrame for That_w_t and metrics
    That_w_t_df = pd.DataFrame(That_w_t_list, columns=[timestamp, "x", "y", "z", "qx", "qy", "qz", "qw", "yaw"])
    T_t_s_df = pd.DataFrame(T_t_s_list, columns=[timestamp, "x", "y", "z", "qx", "qy", "qz", "qw", "yaw"])
    
    return That_w_t_df, T_t_s_df


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
    odom_source_x = data_frame_ref_odom[cols_ref_odom[1]] + source_odom_origin[0]
    odom_source_y = data_frame_ref_odom[cols_ref_odom[2]] + source_odom_origin[1]
    odom_source_z = data_frame_ref_odom[cols_ref_odom[3]] + source_odom_origin[2]

    ax.plot(odom_source_x, odom_source_y, odom_source_z, c='r', label='odom source', linestyle='--', linewidth=2)


    # Plot ground truth target trajectory
    ax.plot(data_frame_target[cols_target[1]], data_frame_target[cols_target[2]], data_frame_target[cols_target[3]], c='g', label='GT Target', linewidth=2)
    odom_target_x = data_frame_target_odom[cols_target_odom[1]] + target_odom_origin[0]
    odom_target_y = data_frame_target_odom[cols_target_odom[2]] + target_odom_origin[1]
    odom_target_z = data_frame_target_odom[cols_target_odom[3]] + target_odom_origin[2]
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

def plot_pose_temporal_evolution(path, df_ref, df_experiment, df_covariance, df_covariance_odom, df_odom, cols_ref, cols_experiment, cols_covariance, cols_covariance_odom, cols_odom, t0, source_odom_origin, target_odom_origin):
    

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

    axes[0].scatter(np.array(df_experiment[cols_experiment[0]]), np.array(df_experiment[cols_experiment[1]]), c='b', label = 'pose_opt', alpha=0.4, s=10)
    axes[0].plot(np.array(df_ref[cols_ref[0]]), np.array(df_ref[cols_ref[1]]), c='r', label = 'target gt')
    
    target_odom_x = np.array(df_odom[cols_odom[1]]) + target_odom_origin[0]
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
    target_odom_y = np.array(df_odom[cols_odom[2]]) + target_odom_origin[1]
    axes[1].plot(np.array(df_odom[cols_odom[0]]), target_odom_y, c='r',linestyle= '--')

    axes[1].fill_between(np.array(df_experiment[cols_experiment[0]]),
                         np.array(df_experiment[cols_experiment[2]]) - 2.0*std_y,
                         np.array(df_experiment[cols_experiment[2]]) + 2.0*std_y,
                         color='blue', alpha=0.2, label='±2σ Uncertainty')
    axes[1].set_ylabel("Y(m)")
    axes[1].grid()

    axes[2].scatter(np.array(df_experiment[cols_experiment[0]]), np.array(abs(df_experiment[cols_experiment[3]])), c='b', alpha=0.4, s=10)
    axes[2].plot(np.array(df_ref[cols_ref[0]]), np.array(df_ref[cols_ref[3]]), c='r')
    target_odom_z = np.array(df_odom[cols_odom[3]]) + target_odom_origin[2]
    axes[2].plot(np.array(df_odom[cols_odom[0]]), target_odom_z, c='r', linestyle= '--')

    axes[2].fill_between(np.array(df_experiment[cols_experiment[0]]),
                         np.array(df_experiment[cols_experiment[3]]) - 2.0*std_z,
                         np.array(df_experiment[cols_experiment[3]]) + 2.0*std_z,
                         color='blue', alpha=0.2, label='±2σ Uncertainty')
    axes[2].set_ylabel("Z(m)")
    axes[2].grid()

    axes[2].set_xlabel("Time(s)")

    # Align timestamps and compute RMSE
    merged_df = pd.merge_asof(df_ref, df_experiment, left_on=cols_ref[0], right_on=cols_experiment[0], direction='nearest')

    error_x = merged_df[cols_ref[1]] - merged_df[cols_experiment[1]]
    error_y = merged_df[cols_ref[2]] - merged_df[cols_experiment[2]]
    error_z = merged_df[cols_ref[3]] - merged_df[cols_experiment[3]]

    rmse_pos = np.sqrt(np.mean(np.square(error_x) + np.square(error_y) + np.square(error_z)))

    print(f'RMSE position_optimized: {rmse_pos} m')


    # Align timestamps and compute RMSE
    merged_df = pd.merge_asof(df_ref, df_odom, left_on=cols_ref[0], right_on=cols_odom[0], direction='nearest')

    error_x = merged_df[cols_ref[1]] - (merged_df[cols_odom[1]] + target_odom_origin[0])
    error_y = merged_df[cols_ref[2]] - (merged_df[cols_odom[2]] + target_odom_origin[1])
    error_z = merged_df[cols_ref[3]] - (merged_df[cols_odom[3]] + target_odom_origin[2])

    rmse_pos_odom = np.sqrt(np.mean(np.square(error_x) + np.square(error_y) + np.square(error_z)))

    print(f'RMSE position_odom: {rmse_pos_odom} m')

    # Update title to include RMSE
    plt.suptitle(f"Temporal Evolution of 3D Pose (RMSE Position: {rmse_pos:.4f} m)")

    plt.savefig(path + '/pose_t.png', bbox_inches='tight')

    plt.show()

def plot_transform(path, df_experiment, df_gt, df_covariance, cols_experiment, cols_gt, cols_covariance, t0):

    fig, axes = plt.subplots(4, 1, figsize=(15, 15))
    plt.suptitle("Temporal Evolution of T_t_s")

    # Subtract the first element of the timestamp column to start from 0
    #df_experiment[cols_experiment[0]] -= df_experiment[cols_experiment[0]].iloc[0]
    df_experiment[cols_experiment[0]] -= t0
    #df_covariance[cols_experiment[0]] -= df_covariance[cols_experiment[0]].iloc[0]
    df_covariance[cols_experiment[0]] -= t0
    df_gt[cols_gt[0]] -= t0

 
    # Compute standard deviations (square root of diagonal covariance elements)
    std_x = np.sqrt(np.array(df_covariance[cols_covariance[1]]))  # Covariance_x
    std_y = np.sqrt(np.array(df_covariance[cols_covariance[2]]))  # Covariance_y
    std_z = np.sqrt(np.array(df_covariance[cols_covariance[3]]))  # Covariance_z
    std_yaw = np.sqrt(np.array(df_covariance[cols_covariance[4]]))  # Covariance_z


    # # Subtract the first element of the timestamp column to start from 0
    # df_experiment[cols_experiment[0]] *= 1e-6

    axes[0].plot(np.array(df_experiment[cols_experiment[0]]), np.array(df_experiment[cols_experiment[1]]), c='b', label = 'pose_opt')
    axes[0].plot(np.array(df_gt[cols_gt[0]]), np.array(df_gt[cols_gt[1]]), c='r', label = 'pose_gt')


    axes[0].fill_between(np.array(df_experiment[cols_experiment[0]]),
                         np.array(df_experiment[cols_experiment[1]]) - 2.0*std_x,
                         np.array(df_experiment[cols_experiment[1]]) + 2.0*std_x,
                         color='blue', alpha=0.2, label='±2σ Uncertainty')
    axes[0].legend()
    #plt.xlabel("Timestamp")
    axes[0].set_ylabel("X(m)")
    axes[0].grid()
    axes[1].plot(np.array(df_experiment[cols_experiment[0]]), np.array(df_experiment[cols_experiment[2]]), c='b')
    axes[1].plot(np.array(df_gt[cols_gt[0]]), np.array(df_gt[cols_gt[2]]), c='r')


    axes[1].fill_between(np.array(df_experiment[cols_experiment[0]]),
                         np.array(df_experiment[cols_experiment[2]]) - 2.0*std_y,
                         np.array(df_experiment[cols_experiment[2]]) + 2.0*std_y,
                         color='blue', alpha=0.2, label='±2σ Uncertainty')
    axes[1].set_ylabel("Y(m)")
    axes[1].grid()

    axes[2].plot(np.array(df_experiment[cols_experiment[0]]), np.array(df_experiment[cols_experiment[3]]), c='b')
    axes[2].plot(np.array(df_gt[cols_gt[0]]), np.array(df_gt[cols_gt[3]]), c='r')


    axes[2].fill_between(np.array(df_experiment[cols_experiment[0]]),
                         np.array(df_experiment[cols_experiment[3]]) - 2.0*std_z,
                         np.array(df_experiment[cols_experiment[3]]) + 2.0*std_z,
                         color='blue', alpha=0.2, label='±2σ Uncertainty')
    axes[2].set_ylabel("Z(m)")
    axes[2].grid()

    axes[3].plot(np.array(df_experiment[cols_experiment[0]]), np.array(df_experiment[cols_experiment[-1]]), c='b')
    axes[3].plot(np.array(df_gt[cols_gt[0]]), np.array(df_gt[cols_gt[-1]]), c='r')


    axes[3].fill_between(np.array(df_experiment[cols_experiment[0]]),
                         np.array(df_experiment[cols_experiment[-1]]) - 2.0*std_yaw,
                         np.array(df_experiment[cols_experiment[-1]]) + 2.0*std_yaw,
                         color='blue', alpha=0.2, label='±2σ Uncertainty')
    axes[3].set_ylabel("Yaw(rad)")
    axes[3].grid()

    axes[3].set_xlabel("Time(s)")

    plt.savefig(path + '/t_That_s.png', bbox_inches='tight')

    plt.show()


def plot_attitude_temporal_evolution(path, df_ref_rpy, df_opt_rpy, df_covariance, df_covariance_odom, df_odom_rpy, cols_experiment, cols_covariance, cols_covariance_odom, t0, source_odom_origin, target_odom_origin):
    

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

    axes[2].scatter(np.array(df_opt_rpy[cols_experiment[0]]), np.array(df_opt_rpy[cols_experiment[3]]), c='b',alpha=0.4, s=10)
    axes[2].plot(np.array(df_ref_rpy[cols_experiment[0]]), np.array(df_ref_rpy[cols_experiment[3]]), c='r')
    target_odom_yaw = np.array(df_odom_rpy[cols_experiment[3]]) + target_odom_origin[3]
    axes[2].plot(np.array(df_odom_rpy[cols_experiment[0]]), target_odom_yaw, c='r', linestyle='--')

    axes[2].fill_between(np.array(df_opt_rpy[cols_experiment[0]]),
                         np.array(df_opt_rpy[cols_experiment[-1]]) - 2.0*std_yaw,
                         np.array(df_opt_rpy[cols_experiment[-1]]) + 2.0*std_yaw,
                         color='blue', alpha=0.2, label='±2σ Uncertainty')
    axes[2].set_ylabel("Yaw(rad)")
    axes[2].grid()

    axes[2].set_xlabel("Time(s)")

    #Compute RMS directly odom-gt

    # Align timestamps and compute RMSE
    merged_df = pd.merge_asof(df_ref_rpy, df_opt_rpy, left_on=cols_experiment[0], right_on=cols_experiment[0], direction='nearest')

    yaw_error = np.array(merged_df[cols_experiment[3] + '_x'] - merged_df[cols_experiment[3] + '_y'])
    yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]

    rmse_angle = np.sqrt(np.mean(np.square(yaw_error)))
    
    print(f'RMSE angle: {np.rad2deg(rmse_angle)}º')


    # Align timestamps and compute RMSE
    merged_df = pd.merge_asof(df_ref_rpy, df_odom_rpy, left_on=cols_experiment[0], right_on=cols_experiment[0], direction='nearest')

    yaw_error = np.array(merged_df[cols_experiment[3] + '_x'] - (merged_df[cols_experiment[3] + '_y']) + target_odom_origin[3])
    yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]

    rmse_angle_odom = np.sqrt(np.mean(np.square(yaw_error)))
    
    print(f'RMSE angle odom: {np.rad2deg(rmse_angle_odom)}º')

     # Add RMSE text to the title
    plt.suptitle(f"Temporal Evolution of Attitude (RMSE Yaw: {np.rad2deg(rmse_angle):.4f}º)")

    plt.savefig(path + '/attitude_t.png', bbox_inches='tight')

    plt.show() 

def plot_metrics(path, df_metrics, cols_metrics, t0):

    fig, axes = plt.subplots(2, 1, figsize=(15, 15))

    # Subtract the first element of the timestamp column to start from 0
    df_metrics[cols_metrics[0]] -= t0

    # # Subtract the first element of the timestamp column to start from 0
    # df_metrics[cols_metrics[0]] *= 1e-6

    axes[0].plot(np.array(df_metrics[cols_metrics[0]]), np.array(df_metrics[cols_metrics[1]]), c='b', label = 'instant')
    axes[0].plot(np.array(df_metrics[cols_metrics[0]]), np.array(df_metrics[cols_metrics[3]]), c='r', label = 'rmse')

    axes[0].legend()
    #plt.xlabel("Timestamp")
    axes[0].set_ylabel("Rotational error (º)")
    axes[0].grid()
    axes[1].plot(np.array(df_metrics[cols_metrics[0]]), np.array(df_metrics[cols_metrics[2]]), c='b')
    axes[1].plot(np.array(df_metrics[cols_metrics[0]]), np.array(df_metrics[cols_metrics[-1]]), c='r')
    axes[1].set_ylabel("Translational error (m)")
    axes[1].grid()

    axes[1].set_xlabel("Time(s)")

    plt.savefig(path + '/metrics.png', bbox_inches='tight')

    plt.show()

def get_columns_with_prefix(path, prefix):
    
    try:
        # Read only the header (first row) to get the column names
        df_header = pd.read_csv(path, nrows=0)

    except FileNotFoundError:
            print(f"Error: File '{path}' not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: File '{path}' is empty.")
    except pd.errors.ParserError:
        print(f"Error: Unable to parse file '{path}'. Make sure it's a valid CSV file.")
    except ValueError as ve:
        print(f"Error: {ve}")

    return [col for col in df_header.columns if col.startswith(prefix)]


def plot_experiment_data(path_experiment_data, path_figs, sim = "True"):

    
    print("Simulation set to: " + sim)
    #Get names of the topics we want to plot
    time_data_setpoint = '__time'
    
    source_gt_frame_id = 'agv_gt'
    target_gt_frame_id = 'uav_gt'

    source_odom_frame_id = 'agv/odom'
    target_odom_frame_id = 'uav/odom'
    source_body_frame_id = 'agv/base_link'
    target_body_frame_id = 'uav/base_link'

    target_odom_origin = np.array([0.25,-0.25,2.0,0.0])
    source_odom_origin = np.array([0.0,0.0,0.0,0.0])

    if sim == "True":

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

    target_odom_cov_x = '/uav/odom/pose/covariance/[0;0]'
    target_odom_cov_y = '/uav/odom/pose/covariance/[1;1]'
    target_odom_cov_z = '/uav/odom/pose/covariance/[2;2]'
    target_odom_cov_yaw = '/uav/odom/pose/covariance/[5;5]'

    source_odom_cov_x = '/agv/odom/pose/covariance/[0;0]'
    source_odom_cov_y = '/agv/odom/pose/covariance/[1;1]'
    source_odom_cov_z = '/agv/odom/pose/covariance/[2;2]'
    source_odom_cov_yaw = '/agv/odom/pose/covariance/[5;5]'

    source_odom_x_data = '/agv/odom/pose/pose/position/x'
    source_odom_y_data = '/agv/odom/pose/pose/position/y'
    source_odom_z_data = '/agv/odom/pose/pose/position/z'
    source_odom_q0_data = '/agv/odom/pose/pose/orientation/x'
    source_odom_q1_data = '/agv/odom/pose/pose/orientation/y'
    source_odom_q2_data = '/agv/odom/pose/pose/orientation/z'
    source_odom_q3_data = '/agv/odom/pose/pose/orientation/w'

    target_odom_x_data = '/uav/odom/pose/pose/position/x'
    target_odom_y_data = '/uav/odom/pose/pose/position/y'
    target_odom_z_data = '/uav/odom/pose/pose/position/z'
    target_odom_q0_data = '/uav/odom/pose/pose/orientation/x'
    target_odom_q1_data = '/uav/odom/pose/pose/orientation/y'
    target_odom_q2_data = '/uav/odom/pose/pose/orientation/z'
    target_odom_q3_data = '/uav/odom/pose/pose/orientation/w'

    max_timestamp = None
    
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


    pose_graph_agv_columns = get_columns_with_prefix(path_experiment_data, "/pose_graph_node/global_agv_poses")
    pose_graph_uav_columns = get_columns_with_prefix(path_experiment_data, "/pose_graph_node/global_uav_poses")

    pose_graph_agv_cols = [time_data_setpoint] + pose_graph_agv_columns
    pose_graph_uav_cols = [time_data_setpoint] + pose_graph_uav_columns

    # # Assume you have read the CSV into a DataFrame:
    # pose_graph_agv_df = read_pandas_df(path_experiment_data, pose_graph_agv_cols, timestamp_col=time_data_setpoint, max_timestamp=max_timestamp, dropna = False)
    # print("AGV DataFrame shape without dropna:", pose_graph_agv_df.shape)
    # pose_graph_uav_df = read_pandas_df(path_experiment_data, pose_graph_uav_cols, timestamp_col=time_data_setpoint, max_timestamp=max_timestamp, dropna = False)
    # print("UAV DataFrame shape without dropna:", pose_graph_uav_df.shape)

    # last_non_nan_agv = pose_graph_agv_df.apply(lambda col: col.dropna().iloc[-1] if not col.dropna().empty else np.nan)
    # poses_agv = extract_poses_from_series(last_non_nan_agv, "/pose_graph_node/global_agv_poses")
    # print("Extracted AGV poses:")
    # for idx, pose in sorted(poses_agv.items()):
    #     print(f"Pose index {idx}:")
    #     print("  Position:", pose['position'])
    #     print("  Orientation:", pose['orientation'])
    #     print("  Timestamp:", pose['timestamp'])


    # last_non_nan_uav = pose_graph_uav_df.apply(lambda col: col.dropna().iloc[-1] if not col.dropna().empty else np.nan)
    # poses_uav = extract_poses_from_series(last_non_nan_uav, "/pose_graph_node/global_uav_poses")
    # print("Extracted AGV poses:")
    # for idx, pose in sorted(poses_uav.items()):
    #     print(f"Pose index {idx}:")
    #     print("  Position:", pose['position'])
    #     print("  Orientation:", pose['orientation'])
    #     print("  Timestamp:", pose['timestamp'])

    # plot_posegraph_temporal(poses_agv)
    # plot_posegraph_temporal(poses_uav)
    # plot_posegraphs_3d(poses_agv, poses_uav)

    columns_covariance = [time_data_setpoint, covariance_x, covariance_y, covariance_z, covariance_yaw]
    covariance_data_df = read_pandas_df(path_experiment_data, columns_covariance, 
                                        timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
    
    columns_odom_covariance = [time_data_setpoint, target_odom_cov_x, target_odom_cov_y, target_odom_cov_z, target_odom_cov_yaw]

    covariance_odom_data_df = read_pandas_df(path_experiment_data, columns_odom_covariance, 
                                        timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
        
    columns_t_That_s_data = [time_data_setpoint, opt_T_target_source_x_data, opt_T_target_source_y_data, opt_T_target_source_z_data, opt_T_target_source_q0_data, opt_T_target_source_q1_data, opt_T_target_source_q2_data, opt_T_target_source_q3_data]

    t_That_s_data_df = read_pandas_df(path_experiment_data, columns_t_That_s_data, 
                                      timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)

    
    if sim == "True":
        # Plot 3D representation

        columns_source_gt_data = [time_data_setpoint, source_gt_x_data, source_gt_y_data, source_gt_z_data, source_gt_q0_data, source_gt_q1_data,source_gt_q2_data, source_gt_q3_data   ]
        columns_source_odom_data = [time_data_setpoint, source_odom_x_data, source_odom_y_data, source_odom_z_data, source_odom_q0_data, source_odom_q1_data,source_odom_q2_data, source_odom_q3_data   ]
        
        columns_target_gt_data = [time_data_setpoint, target_gt_x_data, target_gt_y_data, target_gt_z_data, target_gt_q0_data , target_gt_q1_data ,target_gt_q2_data ,target_gt_q3_data ]
        columns_target_odom_data = [time_data_setpoint, target_odom_x_data, target_odom_y_data, target_odom_z_data, target_odom_q0_data , target_odom_q1_data ,target_odom_q2_data ,target_odom_q3_data ]
        
        columns_metrics = [time_data_setpoint, metrics_detR_data, metrics_dett_data, metrics_rmse_R_data, metrics_rmse_t_data]

        
        source_gt_data_df = read_pandas_df(path_experiment_data, columns_source_gt_data, 
                                           timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
        
        source_odom_data_df = read_pandas_df(path_experiment_data, columns_source_odom_data, 
                                           timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
        
        target_gt_data_df = read_pandas_df(path_experiment_data, columns_target_gt_data, 
                                           timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
        
        target_odom_data_df = read_pandas_df(path_experiment_data, columns_target_odom_data, 
                                           timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)


        w_That_t_data_df, t_T_s_data_df = compute_That_w_t(time_data_setpoint, source_gt_data_df, source_odom_data_df, 
                                                           target_gt_data_df, t_That_s_data_df, columns_source_gt_data, 
                                                           columns_source_odom_data, columns_target_gt_data, columns_t_That_s_data)

        metrics_df_data = read_pandas_df(path_experiment_data, columns_metrics,
                                         timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)

        
        columns_w_That_t_data = [time_data_setpoint, "x", "y", "z", "qx", "qy", "qz", "qw", "yaw"]

        plot_3d_scatter(path_figs, source_gt_data_df, source_odom_data_df, 
                        target_gt_data_df, target_odom_data_df, w_That_t_data_df, 
                        columns_source_gt_data, columns_source_odom_data, columns_target_gt_data, 
                        columns_target_odom_data, columns_w_That_t_data, source_odom_origin, target_odom_origin)
        
        t0 = target_gt_data_df[columns_target_gt_data[0]].iloc[0]
        
        # #Plot position vs time
        plot_pose_temporal_evolution(path_figs, target_gt_data_df, w_That_t_data_df, 
                                     covariance_data_df, covariance_odom_data_df, target_odom_data_df, columns_target_gt_data, 
                                     columns_w_That_t_data, columns_covariance, columns_odom_covariance, columns_target_odom_data, 
                                     t0, source_odom_origin, target_odom_origin)

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

        plot_attitude_temporal_evolution(path_figs, rpy_attitude_gt_data_df, rpy_attitude_opt_data_df, 
                                         covariance_data_df, covariance_odom_data_df, rpy_attitude_odom_data_df, 
                                         rpy_attitude_cols, columns_covariance, columns_odom_covariance, t0, source_odom_origin, target_odom_origin)

        #plot metrics
        plot_metrics(path_figs, metrics_df_data, columns_metrics, t0)

    # Plot transforms
    columns_t_T_s_data = [time_data_setpoint, "x", "y", "z", "qx", "qy", "qz", "qw", "yaw"]

    plot_transform(path_figs, t_That_s_data_df, t_T_s_data_df, covariance_data_df, columns_t_That_s_data, columns_t_T_s_data, columns_covariance, t0)

def plot_posegraphs_3d(poses_agv, poses_uav):
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
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.title("3D Trajectories of AGV and UAV")
    plt.tight_layout()
    plt.show()

def plot_posegraph_temporal(poses):
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

    data = []
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
    
    plt.suptitle("Temporal Evolution of Pose (x, y, z, yaw)")
    plt.show()

def extract_poses_from_series(s, prefix):
    """
    Extracts pose information (position, orientation, and timestamp)
    from a pandas Series `s` where column names follow a naming convention
    that includes an array index. The timestamp is assumed to be stored in two
    parts: sec and nanosec.

    Parameters:
        s (pd.Series): A pandas Series where keys are column names.
        prefix (str): The prefix used in the CSV for the pose graph, for example,
                      "/pose_graph_node/global_agv_poses"

    Returns:
        dict: A dictionary where each key is an array index (int) and each value is a
              dictionary with keys:
                - 'position': a numpy array of shape (3,) or None
                - 'orientation': a numpy array of shape (4,) or None
                - 'timestamp': a float (sec + nanosec * 1e-9) or None
    """
    poses = {}
    # Patterns for position and orientation.
    pos_pattern = re.compile(rf"{re.escape(prefix)}/array\[(\d+)\]/pose/pose/position/(x|y|z)")
    ori_pattern = re.compile(rf"{re.escape(prefix)}/array\[(\d+)\]/pose/pose/orientation/(x|y|z|w)")
    # Patterns for timestamp: sec and nanosec.
    stamp_sec_pattern = re.compile(rf"{re.escape(prefix)}/array\[(\d+)\]/header/stamp/sec")
    stamp_nsec_pattern = re.compile(rf"{re.escape(prefix)}/array\[(\d+)\]/header/stamp/nanosec")
    
    for col, value in s.items():
        # Check for position
        pos_match = pos_pattern.match(col)
        if pos_match:
            idx = int(pos_match.group(1))
            axis = pos_match.group(2)
            if idx not in poses:
                poses[idx] = {'position': {}, 'orientation': {}, 'timestamp': {}}
            poses[idx]['position'][axis] = value
            continue

        # Check for orientation
        ori_match = ori_pattern.match(col)
        if ori_match:
            idx = int(ori_match.group(1))
            axis = ori_match.group(2)
            if idx not in poses:
                poses[idx] = {'position': {}, 'orientation': {}, 'timestamp': {}}
            poses[idx]['orientation'][axis] = value
            continue

        # Check for timestamp seconds.
        stamp_sec_match = stamp_sec_pattern.match(col)
        if stamp_sec_match:
            idx = int(stamp_sec_match.group(1))
            if idx not in poses:
                poses[idx] = {'position': {}, 'orientation': {}, 'timestamp': {}}
            poses[idx]['timestamp']['sec'] = value
            continue

        # Check for timestamp nanoseconds.
        stamp_nsec_match = stamp_nsec_pattern.match(col)
        if stamp_nsec_match:
            idx = int(stamp_nsec_match.group(1))
            if idx not in poses:
                poses[idx] = {'position': {}, 'orientation': {}, 'timestamp': {}}
            poses[idx]['timestamp']['nanosec'] = value
            continue

    # Post-process each pose entry:
    for idx in poses:
        # Process position: require x, y, z.
        pos = poses[idx]['position']
        if all(k in pos for k in ['x', 'y', 'z']):
            poses[idx]['position'] = np.array([pos['x'], pos['y'], pos['z']])
        else:
            poses[idx]['position'] = None

        # Process orientation: require x, y, z, w.
        ori = poses[idx]['orientation']
        if all(k in ori for k in ['x', 'y', 'z', 'w']):
            poses[idx]['orientation'] = np.array([ori['x'], ori['y'], ori['z'], ori['w']])
        else:
            poses[idx]['orientation'] = None

        # Process timestamp: combine seconds and nanoseconds if available.
        ts = poses[idx]['timestamp']
        if isinstance(ts, dict) and 'sec' in ts and 'nanosec' in ts:
            try:
                poses[idx]['timestamp'] = ts['sec'] + ts['nanosec'] * 1e-9
            except Exception as e:
                poses[idx]['timestamp'] = None
        else:
            poses[idx]['timestamp'] = None

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

def main():

    if len(sys.argv) != 3:
        print("Usage: python script.py <csv_file_path>")
        sys.exit(1)

    path_to_data = sys.argv[1]
    simulation = sys.argv[2]

    bag_csv_path = path_to_data + "/data.csv"

    plot_experiment_data(bag_csv_path, path_to_data, simulation)


if __name__ == "__main__":
    main()