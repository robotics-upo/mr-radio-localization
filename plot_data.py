#!/usr/bin/env python3

import sys
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


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

        if latest_source_idx is not None:  # Ensure there's a valid matching source data
            # Get the corresponding source and t_That_s data
            source_row = source_gt_data_df.iloc[latest_source_idx]
            source_odom_row = source_odom_data_df.iloc[latest_source_idx]

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


def plot_3d_scatter(path, data_frame_ref, data_frame_ref_odom, data_frame_target, data_frame_target_odom, data_frame_opt, cols_ref, cols_ref_odom, cols_target, cols_target_odom, cols_opt):
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
    ax.plot(data_frame_ref_odom[cols_ref_odom[1]], data_frame_ref_odom[cols_ref_odom[2]], data_frame_ref_odom[cols_ref_odom[3]], c='r', label='odom source', linestyle='--', linewidth=2)


    # Plot ground truth target trajectory
    ax.plot(data_frame_target[cols_target[1]], data_frame_target[cols_target[2]], data_frame_target[cols_target[3]], c='g', label='GT Target', linewidth=2)
    ax.plot(data_frame_target_odom[cols_target_odom[1]], data_frame_target_odom[cols_target_odom[2]], data_frame_target_odom[cols_target_odom[3]], c='g', linestyle = '--', label='Odom Target', linewidth=2)


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

def plot_pose_temporal_evolution(path, df_ref, df_experiment, df_covariance, df_covariance_odom, df_odom, cols_ref, cols_experiment, cols_covariance, cols_covariance_odom, cols_odom, t0):
    

    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    plt.suptitle("Temporal Evolution of 3D Pose")

    # Subtract the first element of the timestamp column to start from 0
    #df_experiment[cols_experiment[0]] -= df_experiment[cols_experiment[0]].iloc[0]
    df_experiment[cols_experiment[0]] -= t0
    #df_covariance[cols_experiment[0]] -= df_covariance[cols_experiment[0]].iloc[0]
    df_covariance[cols_experiment[0]] -= t0

    df_ref[cols_ref[0]] -= t0
    df_odom[cols_odom[0]] -= t0

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
    axes[0].plot(np.array(df_odom[cols_odom[0]]), np.array(df_odom[cols_odom[1]]), c='r', linestyle= '--', label = 'target odom')

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
    axes[1].plot(np.array(df_odom[cols_odom[0]]), np.array(df_odom[cols_odom[2]]), c='r',linestyle= '--')

    axes[1].fill_between(np.array(df_experiment[cols_experiment[0]]),
                         np.array(df_experiment[cols_experiment[2]]) - 2.0*std_y,
                         np.array(df_experiment[cols_experiment[2]]) + 2.0*std_y,
                         color='blue', alpha=0.2, label='±2σ Uncertainty')
    axes[1].set_ylabel("Y(m)")
    axes[1].grid()

    axes[2].scatter(np.array(df_experiment[cols_experiment[0]]), np.array(abs(df_experiment[cols_experiment[3]])), c='b', alpha=0.4, s=10)
    axes[2].plot(np.array(df_ref[cols_ref[0]]), np.array(df_ref[cols_ref[3]]), c='r')
    axes[2].plot(np.array(df_odom[cols_odom[0]]), np.array(df_odom[cols_odom[3]]), c='r', linestyle= '--')

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

    print(f'RMSE position: {rmse_pos} m')

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


def plot_attitude_temporal_evolution(path, df_ref_rpy, df_opt_rpy, df_covariance, df_covariance_odom, df_odom_rpy, cols_experiment, cols_covariance, cols_covariance_odom, t0):
    

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
    axes[2].plot(np.array(df_odom_rpy[cols_experiment[0]]), np.array(df_odom_rpy[cols_experiment[3]]), c='r', linestyle='--')

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


def plot_experiment_data(path_experiment_data, path_figs, sim = "True"):

    
    print("Simulation set to: " + sim)
    #Get names of the topics we want to plot
    time_data_setpoint = '__time'
    
    if sim == "True":

        # Define the maximum allowable timestamp
        max_timestamp = 1000.0

        source_frame_id = 'agv_gt'
        target_frame_id = 'uav_opt'

        source_gt_x_data = '/tf/world/agv_gt/translation/x'
        source_gt_y_data = '/tf/world/agv_gt/translation/y'
        source_gt_z_data = '/tf/world/agv_gt/translation/z'
        source_gt_q0_data = '/tf/world/agv_gt/rotation/x'
        source_gt_q1_data = '/tf/world/agv_gt/rotation/y'
        source_gt_q2_data = '/tf/world/agv_gt/rotation/z'
        source_gt_q3_data = '/tf/world/agv_gt/rotation/w'

        source_odom_x_data = '/tf/world/agv_odom/translation/x'
        source_odom_y_data = '/tf/world/agv_odom/translation/y'
        source_odom_z_data = '/tf/world/agv_odom/translation/z'
        source_odom_q0_data = '/tf/world/agv_odom/rotation/x'
        source_odom_q1_data = '/tf/world/agv_odom/rotation/y'
        source_odom_q2_data = '/tf/world/agv_odom/rotation/z'
        source_odom_q3_data = '/tf/world/agv_odom/rotation/w'

        target_odom_x_data = '/tf/world/uav_odom/translation/x'
        target_odom_y_data = '/tf/world/uav_odom/translation/y'
        target_odom_z_data = '/tf/world/uav_odom/translation/z'
        target_odom_q0_data = '/tf/world/uav_odom/rotation/x'
        target_odom_q1_data = '/tf/world/uav_odom/rotation/y'
        target_odom_q2_data = '/tf/world/uav_odom/rotation/z'
        target_odom_q3_data = '/tf/world/uav_odom/rotation/w'

        target_odom_cov_x = '/uav/odom/pose/covariance/[0;0]'
        target_odom_cov_y = '/uav/odom/pose/covariance/[1;1]'
        target_odom_cov_z = '/uav/odom/pose/covariance/[2;2]'
        target_odom_cov_yaw = '/uav/odom/pose/covariance/[5;5]'
        
        target_gt_x_data = '/tf/world/uav_gt/translation/x'
        target_gt_y_data = '/tf/world/uav_gt/translation/y'
        target_gt_z_data = '/tf/world/uav_gt/translation/z'
        target_gt_q0_data = '/tf/world/uav_gt/rotation/x'
        target_gt_q1_data = '/tf/world/uav_gt/rotation/y'
        target_gt_q2_data = '/tf/world/uav_gt/rotation/z'
        target_gt_q3_data = '/tf/world/uav_gt/rotation/w'

        metrics_detR_data = '/optimization/metrics/data[0]'
        metrics_dett_data = '/optimization/metrics/data[1]'
        metrics_rmse_R_data = '/optimization/metrics/data[2]'
        metrics_rmse_t_data = '/optimization/metrics/data[3]'

    else:
        source_frame_id = 'arco/eliko'
        target_frame_id = 'base_link'

    opt_T_source_target_x_data = f'/tf/{source_frame_id}/{target_frame_id}/translation/x'
    opt_T_source_target_y_data = f'/tf/{source_frame_id}/{target_frame_id}/translation/y'
    opt_T_source_target_z_data = f'/tf/{source_frame_id}/{target_frame_id}/translation/z'
    opt_T_source_target_q0_data = f'/tf/{source_frame_id}/{target_frame_id}/rotation/x'
    opt_T_source_target_q1_data = f'/tf/{source_frame_id}/{target_frame_id}/rotation/y'
    opt_T_source_target_q2_data = f'/tf/{source_frame_id}/{target_frame_id}/rotation/z'
    opt_T_source_target_q3_data = f'/tf/{source_frame_id}/{target_frame_id}/rotation/w'
    opt_T_source_target_yaw_data = f'/tf/{source_frame_id}/{target_frame_id}/rotation/yaw'

    opt_T_target_source_x_data = '/eliko_optimization_node/optimized_T/transform/translation/x'
    opt_T_target_source_y_data = '/eliko_optimization_node/optimized_T/transform/translation/y'
    opt_T_target_source_z_data = '/eliko_optimization_node/optimized_T/transform/translation/z'
    opt_T_target_source_q0_data = '/eliko_optimization_node/optimized_T/transform/rotation/x'
    opt_T_target_source_q1_data = '/eliko_optimization_node/optimized_T/transform/rotation/y'
    opt_T_target_source_q2_data = '/eliko_optimization_node/optimized_T/transform/rotation/z'
    opt_T_target_source_q3_data = '/eliko_optimization_node/optimized_T/transform/rotation/w'
    opt_T_target_source_yaw_data = '/eliko_optimization_node/optimized_T/transform/rotation/yaw'

    covariance_x = '/eliko_optimization_node/covariance/matrix/data[0]'
    covariance_y = '/eliko_optimization_node/covariance/matrix/data[5]'
    covariance_z = '/eliko_optimization_node/covariance/matrix/data[10]'
    covariance_yaw = '/eliko_optimization_node/covariance/matrix/data[15]'

    columns_covariance = [time_data_setpoint, covariance_x, covariance_y, covariance_z, covariance_yaw]
    covariance_data_df = read_pandas_df(path_experiment_data, columns_covariance, 
                                        timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
    
    columns_odom_covariance = [time_data_setpoint, target_odom_cov_x, target_odom_cov_y, target_odom_cov_z, target_odom_cov_yaw]

    covariance_odom_data_df = read_pandas_df(path_experiment_data, columns_odom_covariance, 
                                        timestamp_col=time_data_setpoint, max_timestamp=max_timestamp)
    
    print(covariance_odom_data_df[columns_odom_covariance[1]])
    
    columns_t_That_s_data = [time_data_setpoint, opt_T_target_source_x_data, opt_T_target_source_y_data, opt_T_target_source_z_data, opt_T_target_source_q0_data, opt_T_target_source_q1_data, opt_T_target_source_q2_data, opt_T_target_source_q3_data, opt_T_target_source_yaw_data]

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
                        columns_target_odom_data, columns_w_That_t_data )
        
        t0 = target_gt_data_df[columns_target_gt_data[0]].iloc[0]
        
        # #Plot position vs time
        plot_pose_temporal_evolution(path_figs, target_gt_data_df, w_That_t_data_df, 
                                     covariance_data_df, covariance_odom_data_df, target_odom_data_df, columns_target_gt_data, 
                                     columns_w_That_t_data, columns_covariance, columns_odom_covariance, columns_target_odom_data, 
                                     t0)

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
                                         rpy_attitude_cols, columns_covariance, columns_odom_covariance, t0)

        #plot metrics
        plot_metrics(path_figs, metrics_df_data, columns_metrics, t0)

    # Plot transforms
    columns_t_T_s_data = [time_data_setpoint, "x", "y", "z", "qx", "qy", "qz", "qw", "yaw"]

    plot_transform(path_figs, t_That_s_data_df, t_T_s_data_df, covariance_data_df, columns_t_That_s_data, columns_t_T_s_data, columns_covariance, t0)


def read_pandas_df(path, columns, timestamp_col=None, max_timestamp=None):


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
            df = df.dropna()

            
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