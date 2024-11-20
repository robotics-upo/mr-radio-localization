#!/usr/bin/env python3

import sys
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
from scipy.spatial.transform import Rotation as R



def compute_transformation_errors(T_w_s, T_w_t, That_t_s):
    
        T_ts = np.linalg.inv(T_w_t) @ T_w_s
        # # Example converting a rotation matrix to Euler angles
        Te = np.linalg.inv(That_t_s) @ T_ts

        Re = R.from_matrix(Te[:3,:3])
        Re_rpy = Re.as_euler('zyx', degrees=True)
        te = Te[:3,3]
        dett = np.linalg.norm(te)
        detR = np.linalg.norm(Re_rpy)

        return detR, dett

def compute_transformation_errors_alt(T_w_s, T_w_t):
    
        Te = np.linalg.inv(T_w_t) @ T_w_s

        Re = R.from_matrix(Te[:3,:3])
        Re_rpy = Re.as_euler('zyx', degrees=True)
        te = Te[:3,3]
        dett = np.linalg.norm(te)
        detR = np.linalg.norm(Re_rpy)

        return detR, dett


def compute_That_w_t(timestamp, source_gt_data_df, target_gt_data_df, t_That_s_data_df, columns_source_gt_data, columns_target_gt_data, columns_t_That_s_data, metrics_cols):
    """
    Computes That_w_t for each row in t_That_s_data_df using the latest available source_gt_data_df.

    Parameters:
        source_gt_data_df (pd.DataFrame): DataFrame with ground truth source data. Must include timestamp column.
        t_That_s_data_df (pd.DataFrame): DataFrame with optimized That_t_s data. Must include timestamp column.

    Returns:
        pd.DataFrame: DataFrame containing That_w_t positions (x, y, z).
    """
    That_w_t_list = []
    metrics_list = []

    # Ensure data is sorted by timestamp
    source_gt_data_df = source_gt_data_df.sort_values(timestamp).reset_index(drop=True)
    t_That_s_data_df = t_That_s_data_df.sort_values(timestamp).reset_index(drop=True)

    for _, row in t_That_s_data_df.iterrows():
        # Find the latest source_gt_data_df row with a timestamp <= current t_That_s_data_df timestamp
        latest_source_idx = source_gt_data_df[source_gt_data_df[timestamp] <= row[timestamp]].index.max()

        if latest_source_idx is not None:  # Ensure there's a valid matching source data
            # Get the corresponding source and t_That_s data
            source_row = source_gt_data_df.iloc[latest_source_idx]
            target_row = target_gt_data_df.iloc[latest_source_idx]
            T_w_t = np.eye(4)
            T_w_s = np.eye(4)
            That_t_s = np.eye(4)

            # Populate T_w_s
            T_w_s[:3, 3] = source_row[[columns_source_gt_data[1], columns_source_gt_data[2], columns_source_gt_data[3]]].values
            q_w_s = source_row[[columns_source_gt_data[4], columns_source_gt_data[5], columns_source_gt_data[6], columns_source_gt_data[7]]].values
            T_w_s[:3, :3] = R.from_quat(q_w_s).as_matrix()


            # Populate T_w_t
            T_w_t[:3, 3] = target_row[[columns_target_gt_data[1], columns_target_gt_data[2], columns_target_gt_data[3]]].values
            q_w_t = target_row[[columns_target_gt_data[4], columns_target_gt_data[5], columns_target_gt_data[6], columns_target_gt_data[7]]].values
            T_w_t[:3, :3] = R.from_quat(q_w_t).as_matrix()

            # Populate That_t_s
            That_t_s[:3, 3] = row[[columns_t_That_s_data[1], columns_t_That_s_data[2], columns_t_That_s_data[3]]].values
            q_hat = row[[columns_t_That_s_data[4], columns_t_That_s_data[5], columns_t_That_s_data[6], columns_t_That_s_data[7]]].values
            That_t_s[:3, :3] = R.from_quat(q_hat).as_matrix()

            # Compute That_w_t
            That_w_t = T_w_s @ np.linalg.inv(That_t_s)
             # Compute That_w_t
            That_w_t = T_w_s @ np.linalg.inv(That_t_s)
            translation = That_w_t[:3, 3]  # Extract the translation part
            rotation = R.from_matrix(That_w_t[:3, :3]).as_quat()  # Extract rotation as quaternion

            That_w_t_list.append([row[timestamp], *translation, *rotation])  # Include timestamp, translation, and rotation

            #Compute metrics locally instead of using topic (should be same result)
            detR1, dett1 = compute_transformation_errors(T_w_s, 
                                                                T_w_t, 
                                                                That_t_s)
                
            detR2, dett2 = compute_transformation_errors_alt(T_w_t, That_w_t)

            #Select minimum error from the two methods
            dett = min(dett1, dett2)
            detR = min(detR1,detR2)

            metrics_list.append([row[timestamp], detR, dett])

    # Create a DataFrame for That_w_t and metrics
    That_w_t_df = pd.DataFrame(That_w_t_list, columns=[timestamp, "x", "y", "z", "qx", "qy", "qz", "qw"])
    metrics_df = pd.DataFrame(metrics_list, columns = [timestamp, metrics_cols[1], metrics_cols[2]])
    return That_w_t_df, metrics_df


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


def plot_3d_scatter(path, data_frame_ref, data_frame_target, data_frame_opt, cols_ref, cols_target, cols_opt):
    """
    Plots the 3D scatter including reference, target, and marker positions.
    
    Parameters:
        data_frame_ref (pd.DataFrame): DataFrame for reference trajectory.
        data_frame_target (pd.DataFrame): DataFrame for target trajectory.
        marker_positions (pd.DataFrame): DataFrame for marker positions.
        cols_ref (list): Columns for reference trajectory [x, y, z].
        cols_target (list): Columns for target trajectory [x, y, z].
    """
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    # Plot ground truth source trajectory
    ax.plot(data_frame_ref[cols_ref[1]], data_frame_ref[cols_ref[2]], data_frame_ref[cols_ref[3]], c='r', label='GT source', linewidth=2)

    # Plot ground truth target trajectory
    ax.plot(data_frame_target[cols_target[1]], data_frame_target[cols_target[2]], data_frame_target[cols_target[3]], c='g', label='GT Target', linewidth=2)

    # Plot markers
    ax.scatter(data_frame_opt[cols_opt[1]], data_frame_opt[cols_opt[2]], data_frame_opt[cols_opt[3]], c='b', marker='o', label='Optimized', alpha=0.4, s=10)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()

    plt.savefig(path + '/pose_3D.png', bbox_inches='tight')
    plt.show()

def plot_pose_temporal_evolution(path, df_ref, df_experiment, cols_ref, cols_experiment):
    

    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    plt.suptitle("Temporal Evolution of 3D Pose")

    # Subtract the first element of the timestamp column to start from 0
    df_experiment[cols_experiment[0]] -= df_experiment[cols_experiment[0]].iloc[0]
    df_ref[cols_ref[0]] -= df_ref[cols_ref[0]].iloc[0]

    # # Subtract the first element of the timestamp column to start from 0
    # df_experiment[cols_experiment[0]] *= 1e-6
    # df_ref[cols_ref[0]] *= 1e-6

    axes[0].scatter(np.array(df_experiment[cols_experiment[0]]), np.array(df_experiment[cols_experiment[1]]), c='b', label = 'pose_opt', alpha=0.4, s=10)
    axes[0].plot(np.array(df_ref[cols_ref[0]]), np.array(df_ref[cols_ref[1]]), c='r', label = 'reference')
    axes[0].legend()
    #plt.xlabel("Timestamp")
    axes[0].set_ylabel("X(m)")
    axes[0].grid()
    axes[1].scatter(np.array(df_experiment[cols_experiment[0]]), np.array(df_experiment[cols_experiment[2]]), c='b',alpha=0.4, s=10)
    axes[1].plot(np.array(df_ref[cols_ref[0]]), np.array(df_ref[cols_ref[2]]), c='r')
    axes[1].set_ylabel("Y(m)")
    axes[1].grid()

    axes[2].scatter(np.array(df_experiment[cols_experiment[0]]), np.array(abs(df_experiment[cols_experiment[3]])), c='b', alpha=0.4, s=10)
    axes[2].plot(np.array(df_ref[cols_ref[0]]), np.array(abs(df_ref[cols_ref[3]])), c='r')
    axes[2].set_ylabel("Z(m)")
    axes[2].grid()

    axes[2].set_xlabel("Time(s)")


    plt.savefig(path + '/pose_t.png', bbox_inches='tight')

    plt.show()

def plot_transform(path, df_experiment, cols_experiment, tf_name):

    fig, axes = plt.subplots(4, 1, figsize=(15, 15))
    plt.suptitle("Temporal Evolution of " + tf_name)

    # Subtract the first element of the timestamp column to start from 0
    df_experiment[cols_experiment[0]] -= df_experiment[cols_experiment[0]].iloc[0]

    # # Subtract the first element of the timestamp column to start from 0
    # df_experiment[cols_experiment[0]] *= 1e-6

    axes[0].plot(np.array(df_experiment[cols_experiment[0]]), np.array(df_experiment[cols_experiment[1]]), c='b', label = 'pose_opt')
    axes[0].legend()
    #plt.xlabel("Timestamp")
    axes[0].set_ylabel("X(m)")
    axes[0].grid()
    axes[1].plot(np.array(df_experiment[cols_experiment[0]]), np.array(df_experiment[cols_experiment[2]]), c='b')
    axes[1].set_ylabel("Y(m)")
    axes[1].grid()

    axes[2].plot(np.array(df_experiment[cols_experiment[0]]), np.array(abs(df_experiment[cols_experiment[3]])), c='b')
    axes[2].set_ylabel("Z(m)")
    axes[2].grid()

    axes[3].plot(np.array(df_experiment[cols_experiment[0]]), np.array(abs(df_experiment[cols_experiment[-1]])), c='b')
    axes[3].set_ylabel("Yaw(rad)")
    axes[3].grid()

    axes[3].set_xlabel("Time(s)")

    plt.savefig(path + '/' + tf_name + '.png', bbox_inches='tight')

    plt.show()




def plot_attitude_temporal_evolution(path, df_ref_rpy, df_opt_rpy, cols_experiment):
    

    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    plt.suptitle("Temporal Evolution of attitude")

    # Subtract the first element of the timestamp column to start from 0
    df_ref_rpy[cols_experiment[0]] -= df_ref_rpy[cols_experiment[0]].iloc[0]
    df_opt_rpy[cols_experiment[0]] -= df_opt_rpy[cols_experiment[0]].iloc[0]

    # # Subtract the first element of the timestamp column to start from 0
    # df_experiment[cols_experiment[0]] *= 1e-6
    # df_ref_rp[cols_ref_rp[0]] *= 1e-6
    # df_ref_yaw[cols_ref_yaw[0]] *= 1e-6

    axes[0].scatter(np.array(df_opt_rpy[cols_experiment[0]]), np.array(df_opt_rpy[cols_experiment[1]]), c='b', label = 'att_opt', alpha=0.4, s=10)
    axes[0].plot(np.array(df_ref_rpy[cols_experiment[0]]), np.array(df_ref_rpy[cols_experiment[1]]), c='r', label = 'reference')
    axes[0].legend()
    #plt.xlabel("Timestamp")
    axes[0].set_ylabel("Roll(rad)")
    axes[0].grid()
    axes[1].scatter(np.array(df_opt_rpy[cols_experiment[0]]), np.array(df_opt_rpy[cols_experiment[2]]), c='b', alpha=0.4, s=10)
    axes[1].plot(np.array(df_ref_rpy[cols_experiment[0]]), np.array(df_ref_rpy[cols_experiment[2]]), c='r')
    axes[1].set_ylabel("Pitch(rad)")
    axes[1].grid()

    axes[2].scatter(np.array(df_opt_rpy[cols_experiment[0]]), np.array(df_opt_rpy[cols_experiment[3]]), c='b',alpha=0.4, s=10)
    axes[2].plot(np.array(df_ref_rpy[cols_experiment[0]]), np.array(df_ref_rpy[cols_experiment[3]]), c='r')
    axes[2].set_ylabel("Yaw(rad)")
    axes[2].grid()

    axes[2].set_xlabel("Time(s)")

    plt.savefig(path + '/attitude_t.png', bbox_inches='tight')

    plt.show() 

def plot_metrics(path, df_metrics, cols_metrics):

    fig, axes = plt.subplots(2, 1, figsize=(15, 15))

    # Subtract the first element of the timestamp column to start from 0
    df_metrics[cols_metrics[0]] -= df_metrics[cols_metrics[0]].iloc[0]

    # # Subtract the first element of the timestamp column to start from 0
    # df_metrics[cols_metrics[0]] *= 1e-6

    axes[0].plot(np.array(df_metrics[cols_metrics[0]]), np.array(df_metrics[cols_metrics[1]]), c='b', label = 'pose_opt')
    axes[0].legend()
    #plt.xlabel("Timestamp")
    axes[0].set_ylabel("detR (º)")
    axes[0].grid()
    axes[1].plot(np.array(df_metrics[cols_metrics[0]]), np.array(df_metrics[cols_metrics[2]]), c='b')
    axes[1].set_ylabel("dett (m)")
    axes[1].grid()

    axes[1].set_xlabel("Time(s)")

    plt.savefig(path + '/metrics.png', bbox_inches='tight')

    plt.show()


def plot_experiment_data(path_experiment_data, path_figs, sim = "True"):

    
    print("Simulation set to: " + sim)
    #Get names of the topics we want to plot
    time_data_setpoint = '__time'
    
    if sim == "True":

        source_frame_id = 'ground_vehicle'
        target_frame_id = 'uav_opt'

        source_gt_x_data = '/tf/world/ground_vehicle/translation/x'
        source_gt_y_data = '/tf/world/ground_vehicle/translation/y'
        source_gt_z_data = '/tf/world/ground_vehicle/translation/z'
        source_gt_q0_data = '/tf/world/ground_vehicle/rotation/x'
        source_gt_q1_data = '/tf/world/ground_vehicle/rotation/y'
        source_gt_q2_data = '/tf/world/ground_vehicle/rotation/z'
        source_gt_q3_data = '/tf/world/ground_vehicle/rotation/w'


        target_gt_x_data = '/tf/world/uav_gt/translation/x'
        target_gt_y_data = '/tf/world/uav_gt/translation/y'
        target_gt_z_data = '/tf/world/uav_gt/translation/z'
        target_gt_q0_data = '/tf/world/uav_gt/rotation/x'
        target_gt_q1_data = '/tf/world/uav_gt/rotation/y'
        target_gt_q2_data = '/tf/world/uav_gt/rotation/z'
        target_gt_q3_data = '/tf/world/uav_gt/rotation/w'

        metrics_detR_data = '/optimization/metrics/data[0]'
        metrics_dett_data = '/optimization/metrics/data[1]'

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
    


    columns_s_That_t_data = [time_data_setpoint, opt_T_source_target_x_data, opt_T_source_target_y_data, opt_T_source_target_z_data, opt_T_source_target_q0_data, opt_T_source_target_q1_data, opt_T_source_target_q2_data, opt_T_source_target_q3_data, opt_T_source_target_yaw_data]
    columns_t_That_s_data = [time_data_setpoint, opt_T_target_source_x_data, opt_T_target_source_y_data, opt_T_target_source_z_data, opt_T_target_source_q0_data, opt_T_target_source_q1_data, opt_T_target_source_q2_data, opt_T_target_source_q3_data, opt_T_target_source_yaw_data]

    s_That_t_data_df = read_pandas_df(path_experiment_data, columns_s_That_t_data)
    t_That_s_data_df = read_pandas_df(path_experiment_data, columns_t_That_s_data)
    
    if sim == "True":
        # Plot 3D representation

        columns_source_gt_data = [time_data_setpoint, source_gt_x_data, source_gt_y_data, source_gt_z_data, source_gt_q0_data, source_gt_q1_data,source_gt_q2_data, source_gt_q3_data   ]
        columns_target_gt_data = [time_data_setpoint, target_gt_x_data, target_gt_y_data, target_gt_z_data, target_gt_q0_data , target_gt_q1_data ,target_gt_q2_data ,target_gt_q3_data ]

        columns_metrics = [time_data_setpoint, metrics_detR_data, metrics_dett_data]

        
        source_gt_data_df = read_pandas_df(path_experiment_data, columns_source_gt_data)
        target_gt_data_df = read_pandas_df(path_experiment_data, columns_target_gt_data)


        w_That_t_data_df, metrics_df_data = compute_That_w_t(time_data_setpoint, source_gt_data_df, target_gt_data_df, t_That_s_data_df, columns_source_gt_data, columns_target_gt_data, columns_t_That_s_data, columns_metrics)

        metrics_df_data = read_pandas_df(path_experiment_data, columns_metrics)

        
        columns_w_That_t_data = [time_data_setpoint, "x", "y", "z", "qx", "qy", "qz", "qw"]
        plot_3d_scatter(path_figs, source_gt_data_df, target_gt_data_df, w_That_t_data_df, columns_source_gt_data, columns_target_gt_data, columns_w_That_t_data )
        
        # #Plot position vs time
        plot_pose_temporal_evolution(path_figs, target_gt_data_df, w_That_t_data_df, columns_target_gt_data, columns_w_That_t_data)

        # #Plot attitude vs time

        rpy_attitude_opt_data_df = w_That_t_data_df[["qw", "qx", "qy", "qz"]].apply(lambda row: pd.Series(quaternion_to_euler_angles(row), index=['roll', 'pitch', 'yaw']), axis=1)    
        rpy_attitude_gt_data_df = target_gt_data_df[[target_gt_q3_data, target_gt_q0_data, target_gt_q1_data, target_gt_q2_data]].apply(lambda row: pd.Series(quaternion_to_euler_angles(row), index=['roll', 'pitch', 'yaw']), axis=1)    
        
        # Add the timestamp column to the resulting DataFrame
        rpy_attitude_opt_data_df[time_data_setpoint] = w_That_t_data_df[time_data_setpoint]
        rpy_attitude_gt_data_df[time_data_setpoint] = target_gt_data_df[time_data_setpoint]

        # Reorder the columns to make timestamp the first column
        rpy_attitude_cols = [time_data_setpoint, "roll", "pitch", "yaw"]
        
        rpy_attitude_opt_data_df = rpy_attitude_opt_data_df[rpy_attitude_cols]
        rpy_attitude_gt_data_df = rpy_attitude_gt_data_df[rpy_attitude_cols]

        plot_attitude_temporal_evolution(path_figs, rpy_attitude_gt_data_df, rpy_attitude_opt_data_df, rpy_attitude_cols)

        #plot metrics
        plot_metrics(path_figs, metrics_df_data, columns_metrics)

    # Plot transforms
    plot_transform(path_figs, s_That_t_data_df, columns_s_That_t_data, 's_That_t')
    plot_transform(path_figs, t_That_s_data_df, columns_t_That_s_data, 't_That_s')


def read_pandas_df(path, columns):

    try:
            # Read CSV file into a Pandas DataFrame
            df = pd.read_csv(path, usecols = columns)
            df = df.dropna()

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