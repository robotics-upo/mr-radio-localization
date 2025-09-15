import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import re
import sys

from scipy.optimize import least_squares

# === for paper‐quality bigger fonts ===
mpl.rcParams.update({
    'font.size':         28,   # base font size
    'axes.labelsize':    28,   # x/y label
    'axes.titlesize':    28,   # subplot title
    'xtick.labelsize':   28,   # tick numbers
    'ytick.labelsize':   28,
    'legend.fontsize':   28,
    'figure.titlesize':  28    # overall figure title
})
 # -------------------------------------------------

# Anchors (AGV frame)
anchors = {
    "0x0009D6": [-0.31,  0.32, 0.875],   # a1
    "0x0009E5": [ 0.26, -0.32, 0.875],   # a2
    "0x0016FA": [ 0.24,  0.32, 0.33 ],   # a3
    "0x0016CF": [-0.24, -0.32, 0.33 ],   # a4
}
# Tags (UAV frame)
tags = {
    "0x001155": [-0.24, -0.24, -0.06],   # t1
    "0x001397": [ 0.24,  0.24, -0.06],   # t2
}

def pose_to_matrix(pose):
    x, y, z, roll, pitch, yaw = pose
    rot = R.from_euler('zyx', [yaw, pitch, roll])
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = [x, y, z]
    return T

def transform_point(T, point):
    p_h = np.array([*point, 1.0])   # homogeneous
    return (T @ p_h)[:3]

def quat_to_T(x, y, z, qx, qy, qz, qw):
    T = np.eye(4)
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T

def compute_gt_series(df, agv_cols, uav_cols, anchors_local, tags_local, tag_id, time_col="__time", max_dt=0.05):
    """
    Build per-row GT distances (cm) by time-aligning AGV and UAV pose tracks.
    For each df[time_col], we pick the nearest AGV pose and nearest UAV pose
    within ±max_dt seconds. If either is missing, GT at that row is NaN.
    """
    # Ensure time sorted and numeric
    df = df.copy()
    if time_col not in df.columns:
        raise ValueError(f"time column '{time_col}' not found")
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df = df.sort_values(time_col, kind="mergesort")

    # Build clean AGV/UAV pose tracks (rows where all 7 fields are present)
    agv_track = df[[time_col] + agv_cols].dropna()
    uav_track = df[[time_col] + uav_cols].dropna()

    agv_track = agv_track.drop_duplicates(subset=time_col).sort_values(time_col, kind="mergesort")
    uav_track = uav_track.drop_duplicates(subset=time_col).sort_values(time_col, kind="mergesort")

    # Target timeline = original df times (after your 5s crop)
    timeline = df[[time_col]].drop_duplicates().sort_values(time_col, kind="mergesort")

    # Nearest-neighbor time alignment (within tolerance)
    tol = pd.Timedelta(seconds=max_dt)
    # Convert to datetime-ish index by treating seconds as timedeltas from zero for asof to accept tolerance
    base = pd.Timestamp("1970-01-01")
    def to_dt(s): return (base + pd.to_timedelta(s, unit="s"))

    timeline_dt         = timeline.copy();          timeline_dt["__dt"] = to_dt(timeline_dt[time_col])
    agv_track_dt        = agv_track.copy();         agv_track_dt["__dt"] = to_dt(agv_track_dt[time_col])
    uav_track_dt        = uav_track.copy();         uav_track_dt["__dt"] = to_dt(uav_track_dt[time_col])

    # asof for AGV
    agv_aligned = pd.merge_asof(
        left=timeline_dt.sort_values("__dt"),
        right=agv_track_dt.sort_values("__dt"),
        on="__dt",
        direction="nearest",
        tolerance=tol,
        suffixes=("","_agv"),
    )
    # asof for UAV
    uav_aligned = pd.merge_asof(
        left=timeline_dt.sort_values("__dt"),
        right=uav_track_dt.sort_values("__dt"),
        on="__dt",
        direction="nearest",
        tolerance=tol,
        suffixes=("","_uav"),
    )

    # Combine aligned poses onto the original timeline
    aligned = timeline_dt[[time_col, "__dt"]].merge(
        agv_aligned.drop(columns=[time_col]), on="__dt", how="left"
    ).merge(
        uav_aligned.drop(columns=[time_col]), on="__dt", how="left", suffixes=("","")
    )

    # Extract numeric arrays
    def colv(name): return pd.to_numeric(aligned[name], errors="coerce").to_numpy()

    ax, ay, az, aqx, aqy, aqz, aqw = [colv(c) for c in agv_cols]
    ux, uy, uz, uqx, uqy, uqz, uqw = [colv(c) for c in uav_cols]

    N = len(aligned)
    out = {aid: np.full(N, np.nan, dtype=float) for aid in anchors_local}
    anchor_points = {aid: np.array([*p, 1.0]) for aid, p in anchors_local.items()}
    tag_point = np.array([*tags_local[tag_id], 1.0])

    def safe_T(x, y, z, qx, qy, qz, qw):
        qn = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if not np.isfinite(qn) or qn < 1e-9: return None
        Rm = R.from_quat([qx/qn, qy/qn, qz/qn, qw/qn]).as_matrix()
        T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = [x, y, z]
        return T

    valid_rows = 0
    for i in range(N):
        if any(np.isnan(v) for v in (ax[i], ay[i], az[i], aqx[i], aqy[i], aqz[i], aqw[i],
                                     ux[i], uy[i], uz[i], uqx[i], uqy[i], uqz[i], uqw[i])):
            continue
        T_agv = safe_T(ax[i], ay[i], az[i], aqx[i], aqy[i], aqz[i], aqw[i])
        T_uav = safe_T(ux[i], uy[i], uz[i], uqx[i], uqy[i], uqz[i], uqw[i])
        if T_agv is None or T_uav is None:
            continue
        tag_g = (T_uav @ tag_point)[:3]
        for aid, a_local in anchor_points.items():
            a_g = (T_agv @ a_local)[:3]
            out[aid][i] = 100.0 * np.linalg.norm(a_g - tag_g)  # cm
        valid_rows += 1

    print(f"[GT] time-aligned with tolerance ±{max_dt*1000:.0f} ms → {valid_rows}/{N} rows matched.")
    return out

def real_topics_by_anchor_for_tag(tag_id: str):
    """Build REAL distance topic names for all anchors for a given tag id ('0x001155' or '0x001397')."""
    def a(aid_hex): return f"/eliko/Distance/A{aid_hex}/T{tag_id}/data"
    return {
        "0x0009D6": a("0x0009D6"),
        "0x0009E5": a("0x0009E5"),
        "0x0016CF": a("0x0016CF"),
        "0x0016FA": a("0x0016FA"),
    }

def build_gt_pose_columns():
    """Return (agv_cols, uav_cols) for REAL GT pose columns (same names you already use)."""
    uav_gt_topic_name = '/dll_node/pose_estimation'
    agv_gt_topic_name = '/dll_node_arco/pose_estimation'
    agv_cols = [
        f'{agv_gt_topic_name}/pose/pose/position/x',
        f'{agv_gt_topic_name}/pose/pose/position/y',
        f'{agv_gt_topic_name}/pose/pose/position/z',
        f'{agv_gt_topic_name}/pose/pose/orientation/x',
        f'{agv_gt_topic_name}/pose/pose/orientation/y',
        f'{agv_gt_topic_name}/pose/pose/orientation/z',
        f'{agv_gt_topic_name}/pose/pose/orientation/w',
    ]
    uav_cols = [
        f'{uav_gt_topic_name}/pose/pose/position/x',
        f'{uav_gt_topic_name}/pose/pose/position/y',
        f'{uav_gt_topic_name}/pose/pose/position/z',
        f'{uav_gt_topic_name}/pose/pose/orientation/x',
        f'{uav_gt_topic_name}/pose/pose/orientation/y',
        f'{uav_gt_topic_name}/pose/pose/orientation/z',
        f'{uav_gt_topic_name}/pose/pose/orientation/w',
    ]
    return agv_cols, uav_cols

def plot_real_tags_side_by_side(real_csv_path: str, time_col="__time", max_seconds=25.0):
    """
    One figure, 4 rows x 2 cols:
      - Left col: Tag 1 (0x001155) vs all anchors
      - Right col: Tag 2 (0x001397) vs all anchors
    Measured series in default color, per-sample GT in RED.
    Computes RMSE & MBE per anchor-tag pair + averages across system (CSV only).
    Saves:
      - uwb_real_tags_t1_t2_<int(max_seconds)>s.png
      - uwb_real_tags_stats.csv
    """
    # anchor_order  = ["0x0009D6", "0x0009E5", "0x0016FA", "0x0016CF"]
    anchor_order  = ["0x0009D6", "0x0009E5"]

    anchor_labels = {
        "0x0009D6": "Anchor 1",
        "0x0009E5": "Anchor 2",
        # "0x0016FA": "Anchor 3",
        # "0x0016CF": "Anchor 4"
    }

    # Map tag IDs to aliases
    tag_map = {"0x001155": "Tag 1", "0x001397": "Tag 2"}
    tag_left, tag_right = "0x001155", "0x001397"

    topics_left  = real_topics_by_anchor_for_tag(tag_left)
    topics_right = real_topics_by_anchor_for_tag(tag_right)
    agv_cols, uav_cols = build_gt_pose_columns()

    # Columns needed (REAL distances for both tags + GT poses)
    real_columns = [time_col] + list(topics_left.values()) + list(topics_right.values()) + agv_cols + uav_cols

    # Read & crop
    print("reading real csv")
    df = read_pandas_df(real_csv_path, real_columns, timestamp_col=time_col, dropna=True)
    df[time_col] -= df[time_col].iloc[0]
    df = df[df[time_col] <= max_seconds].copy()

    # Per-sample GT series
    gt_left  = compute_gt_series(df, agv_cols, uav_cols, anchors, tags, tag_left,  time_col=time_col, max_dt=0.10)
    gt_right = compute_gt_series(df, agv_cols, uav_cols, anchors, tags, tag_right, time_col=time_col, max_dt=0.10)

    # Stats container
    stats = []

    def compute_stats(vals, gts):
        n = min(len(vals), len(gts))
        vals = vals[:n].astype(float)
        gts  = gts[:n].astype(float)
        mask = ~np.isnan(vals) & ~np.isnan(gts)
        if not np.any(mask):
            return None
        v, g = vals[mask], gts[mask]
        resid = v - g
        return dict(
            samples=len(v),
            rmse_cm=float(np.sqrt(np.nanmean(resid**2))),
            mbe_cm=float(np.nanmean(resid)),
            stdev_cm=float(np.nanstd(resid, ddof=1)) if len(resid) > 1 else 0.0
        )

    # Plot grid
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14,7), sharex=False, sharey=False)
    plotted_any = False

    for r, aid in enumerate(anchor_order):
        # Left column: Tag 1
        axL = axes[r, 0]
        topicL = topics_left[aid]
        if topicL in df.columns:
            vals = pd.to_numeric(df[topicL], errors='coerce').to_numpy(dtype=float)
            gts  = gt_left[aid]
            s = compute_stats(vals, gts)
            if s:
                idx = np.arange(min(len(vals), len(gts)))
                axL.plot(idx, vals[:len(idx)], '.', alpha=0.9, ms=15)
                axL.plot(idx, gts[:len(idx)],  '-', alpha=0.9, color='r', linewidth=2.5)
                axL.set_title(f"{anchor_labels[aid]} — {tag_map[tag_left]}")
                # axL.set_xlabel("Samples")
                # axL.set_ylabel("Range (cm)")
                axL.grid(alpha=0.3)
                stats.append({"anchor": aid, "tag": tag_map[tag_left], **s})
                plotted_any = True
            else:
                axL.set_visible(False)
        else:
            axL.set_visible(False)

        # Right column: Tag 2
        axR = axes[r, 1]
        topicR = topics_right[aid]
        if topicR in df.columns:
            vals = pd.to_numeric(df[topicR], errors='coerce').to_numpy(dtype=float)
            gts  = gt_right[aid]
            s = compute_stats(vals, gts)
            if s:
                idx = np.arange(min(len(vals), len(gts)))
                axR.plot(idx, vals[:len(idx)], '.', alpha=0.9, ms=15)
                axR.plot(idx, gts[:len(idx)],  '-', alpha=0.9, color='r', linewidth=2.5)
                axR.set_title(f"{anchor_labels[aid]} — {tag_map[tag_right]}")
                # axR.set_xlabel("Samples")
                # axR.set_ylabel("Range (cm)")
                axR.grid(alpha=0.3)
                stats.append({"anchor": aid, "tag": tag_map[tag_right], **s})
                plotted_any = True
            else:
                axR.set_visible(False)
        else:
            axR.set_visible(False)
        
        # Only show x labels on the bottom row
        for ax in axes[-1, :]:
            ax.set_xlabel("Samples")

        # Only show y labels on the left column
        for ax in axes[:, 0]:
            ax.set_ylabel("Range (cm)")

    # # Shared legend
    # handles = [
    #     plt.Line2D([0],[0], marker='.', linestyle='None', label='Measured'),
    #     plt.Line2D([0],[0], linestyle='-', color='r', label='Reference'),
    # ]
    # fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False)

    fig.tight_layout()

    out_fig = f"uwb_real_tags_t1_t2_{int(max_seconds)}s.png"
    if plotted_any:
        fig.savefig(out_fig, bbox_inches='tight')
        print(f"Saved {out_fig}")
    else:
        print("[WARN] No valid data to plot for either tag.")
    plt.close(fig)

    # ---------- Stats output (CSV only) ----------
    if stats:
        df_stats = pd.DataFrame(stats)
        # Compute system-wide averages
        avg_rmse  = df_stats['rmse_cm'].mean()
        avg_mbe   = df_stats['mbe_cm'].mean()
        avg_stdev = df_stats['stdev_cm'].mean()
        df_stats.loc[len(df_stats)] = {
            "anchor": "ALL", "tag": "ALL",
            "samples": df_stats['samples'].sum(),
            "rmse_cm": avg_rmse, "mbe_cm": avg_mbe, "stdev_cm": avg_stdev
        }
        out_csv = "uwb_real_tags_stats.csv"
        df_stats.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")


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



def main():
    # Usage:
    #   python3 plot_ranges.py <real_csv_path>
    args = sys.argv
    if len(args) == 2:
        plot_real_tags_side_by_side(args[1], time_col="__time", max_seconds=15.0)

    else:
        print("Usage:")
        print("  python3 plot_ranges.py <real_csv_path>")
        sys.exit(1)

if __name__ == "__main__":
    main()