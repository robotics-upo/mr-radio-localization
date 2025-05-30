#!/usr/bin/env python3

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


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

            if dropna:
                # Drop rows where all columns except timestamp are NaN
                non_time_cols = [col for col in columns if col != timestamp_col]
                df = df.dropna(subset=non_time_cols, how='all')
            
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

def plot_ranges(csv_path):
    # Columns to read
    time_data_setpoint = '__time'
    raw_range_topics = [
        '/eliko/Distance/A0x0009D6/T0x001155/data',
        '/eliko/Distance/A0x0009D6/T0x001397/data',
        '/eliko/Distance/A0x0009E5/T0x001155/data',
        '/eliko/Distance/A0x0009E5/T0x001397/data',
        '/eliko/Distance/A0x0016CF/T0x001155/data',
        '/eliko/Distance/A0x0016CF/T0x001397/data',
        '/eliko/Distance/A0x0016FA/T0x001155/data',
        '/eliko/Distance/A0x0016FA/T0x001397/data',
    ]
    est_range_topics = [
        '/eliko_optimization_node/range_estimation/data[0]',
        '/eliko_optimization_node/range_estimation/data[1]'
    ]
    all_columns = [time_data_setpoint] + raw_range_topics + est_range_topics

    # Read CSV using consistent method
    df = read_pandas_df(csv_path, all_columns, timestamp_col=time_data_setpoint, dropna=True)

    df[all_columns[0]]-= df[all_columns[0]].iloc[0]
    print(f"Timestamps range from {df[time_data_setpoint].min():.2f} to {df[time_data_setpoint].max():.2f} seconds")

    # Plot
    plt.figure(figsize=(15, 6))

    for i, topic in enumerate(raw_range_topics):
        if i == 0:
            plt.plot(np.array(df[all_columns[0]]), df[topic], 'k.', alpha=0.3, label='Range measurements')
        else:
            plt.plot(np.array(df[all_columns[0]]), df[topic], 'k.', alpha=0.3)

    plt.plot(np.array(df[all_columns[0]]), df[est_range_topics[0]], 'r.-', label='Ground Truth Ranges')
    plt.plot(np.array(df[all_columns[0]]), df[est_range_topics[1]], 'b.-', label='Estimated Ranges')

    plt.xlabel("Time (s)")
    plt.ylabel("Range (cm)")
    plt.title("UWB Ranges Over Time")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("uwb_ranges.png", bbox_inches='tight')
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 plot_ranges.py <csv_file_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    plot_ranges(csv_path)

if __name__ == "__main__":
    main()
