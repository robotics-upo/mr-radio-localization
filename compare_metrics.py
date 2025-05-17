#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ─── EDIT THESE TO MATCH YOUR CSV COLUMN NAMES ────────────────────────────────
TIMESTAMP_COL   = '__time'          # name of your time column
ROT_ERROR_COL   = '/optimization/metrics/data[2]'     # name of your rotational‐error column
TRANS_ERROR_COL = '/optimization/metrics/data[3]'  # name of your translational‐error column
# ───────────────────────────────────────────────────────────────────────────────


def read_and_validate(path):
    df_cols = [TIMESTAMP_COL, ROT_ERROR_COL, TRANS_ERROR_COL]
    df = pd.read_csv(path, usecols = df_cols)
    for col in df_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {path}")
    return df.sort_values(TIMESTAMP_COL)

def plot_comparison(csv1, csv2, label1='Exp 1', label2='Exp 2'):
    df1 = read_and_validate(csv1)
    df2 = read_and_validate(csv2)

    # normalize each to its own first timestamp
    df1['t'] = df1[TIMESTAMP_COL] - df1[TIMESTAMP_COL].iloc[0]
    df2['t'] = df2[TIMESTAMP_COL] - df2[TIMESTAMP_COL].iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("Relative Transform RMSE")

    # Rotational error
    axes[0].plot(np.array(df1['t']), np.array(df1[ROT_ERROR_COL]), '-o', label=label1)
    axes[0].plot(np.array(df2['t']), np.array(df2[ROT_ERROR_COL]), '-o', label=label2)
    # axes[0].set_yscale('log')

    axes[0].set_ylabel("Rotational error (°)")  # or radians
    axes[0].grid(True)
    axes[0].legend()

    # Translational error
    axes[1].plot(np.array(df1['t']), np.array(df1[TRANS_ERROR_COL]), '-o', label=label1)
    axes[1].plot(np.array(df2['t']), np.array(df2[TRANS_ERROR_COL]), '-o', label=label2)
    # axes[1].set_yscale('log')

    axes[1].set_ylabel("Translational error (m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig('metrics_comparison.svg', format = 'svg', bbox_inches='tight')
    plt.savefig('metrics_comparison.png', format = 'png', bbox_inches='tight')

    plt.show()

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_metrics.py <exp1.csv> <exp2.csv> [label1 label2]")
        sys.exit(1)

    csv1, csv2 = sys.argv[1], sys.argv[2]
    if len(sys.argv) >= 5:
        label1, label2 = sys.argv[3], sys.argv[4]
    else:
        label1, label2 = 'Odom 0%', 'Odom 2%'

    plot_comparison(csv1, csv2, label1, label2)

if __name__ == "__main__":
    main()