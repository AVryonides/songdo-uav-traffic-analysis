#pipeline.py
"""
Main analysis pipeline for the Q intersection.

Orchestrates data loading, lane inference, vehicle grouping,
space-time diagram generation, and all ancillary plots.

Usage (identical to the old monolithic script):
    python -m songdo_q --traj 2022-10-04_Q_PM5.csv --seg Q.csv --out outputs_Q
"""

from __future__ import annotations

import csv as _csv
import json
import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tool_songdo_M import Master
import tool_songdo as tool

from .config import (
    VERBOSE,
    log,
    dbg,
    parse_args,
    _save_fig_formats,
    _THESIS_FONT,
    infer_intersection_id_from_filename,
    patch_matplotlib,
    patch_od_pairs,
)
from .geometry import compute_bbox_latlon, compute_center_latlon, trim_to_intersection
from .lane_inference import (
    summarize_lane_numbers,
    print_lane_summary,
    build_vehicle_feature_table,
    fit_vehicle_knn_with_thresholds,
    predict_missing_lanes_vehicle_knn,
    assign_lane_final_per_vehicle,
    print_lane_coverage_before_after,
)
from .vehicle_grouping import (
    group_vehicles_by_sections,
    build_movement_to_ids_for_workset,
    print_section_group_report_limited,
    filter_df_by_vehicle_ids,
)
from .events import (
    compute_movement_arrival_departure_times,
    compute_lane_arrival_departure_times,
)
from .plotting import (
    plot_lane_map,
    plot_cumulative_by_movement,
    plot_cumulative_by_lane,
    plot_general_inputs_outputs_per_time,
    plot_turning_arrow_diagram,
    plot_intersection_layout,
    plot_rejected_vehicle_maps,
    plot_recording_timeline,
    plot_headway_distributions,
    save_g_maps_into_run_dir,
)
from .space_time import generate_space_time_diagram


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_recording_segments(
    df: pd.DataFrame,
    local_time_col: str = "Local_Time",
    time_col: str = "time_s",
    gap_threshold_minutes: float = 1.5,
):
    """Detect recording segments from Local_Time gaps.

    A new segment begins whenever two consecutive minute-timestamps are more
    than *gap_threshold_minutes* apart.  The default of 1.5 min is chosen so
    that every real inter-recording gap in the M dataset (≥ 2 min) is detected
    while no intra-recording gap (≤ 1 min between consecutive frames) creates a
    false split.

    Returns list of (label, t_start_s, t_end_s) sorted by time.
    label is 'HH:MM-HH:MM' using the first and last minute of each segment.
    """
    from datetime import datetime, timedelta

    raw = df[local_time_col].dropna().astype(str)

    def _parse_clock(s):
        try:
            parts = s.strip().split(":")
            return datetime(2000, 1, 1, int(parts[0]), int(parts[1]))
        except Exception:
            return None

    clock_series = raw.map(_parse_clock)
    valid_mask = clock_series.notna()
    if valid_mask.sum() == 0:
        return []

    df_work = df.loc[valid_mask].copy()
    df_work["_clock"] = clock_series[valid_mask]
    unique_minutes = sorted(df_work["_clock"].unique())

    seg_starts = [unique_minutes[0]]
    seg_ends_list = []
    for i in range(1, len(unique_minutes)):
        delta = (unique_minutes[i] - unique_minutes[i - 1]).total_seconds() / 60.0
        if delta > gap_threshold_minutes:
            seg_ends_list.append(unique_minutes[i - 1])
            seg_starts.append(unique_minutes[i])
    seg_ends_list.append(unique_minutes[-1])

    segments = []
    for s_dt, e_dt in zip(seg_starts, seg_ends_list):
        window_end = e_dt + timedelta(minutes=1)
        mask = (df_work["_clock"] >= s_dt) & (df_work["_clock"] <= window_end)
        seg_df = df.loc[df_work.index[mask]]
        if seg_df.empty:
            continue
        t_start_s = float(seg_df[time_col].min())
        t_end_s = float(seg_df[time_col].max())
        label = f"{s_dt.strftime('%H:%M')}-{e_dt.strftime('%H:%M')}"
        segments.append((label, t_start_s, t_end_s))

    return segments


def _load_green_intervals(general_dir: str) -> dict:
    """Read results_table.csv and return {movement: [(t_start, t_end), ...]} for green phases.

    Called after generate_space_time_diagram so the CSV is guaranteed to exist.
    Returns an empty dict (no filtering) if the file is missing or unreadable.
    """
    import csv as _csv

    results_csv = os.path.join(general_dir, "results_table.csv")
    if not os.path.exists(results_csv):
        print(f"[WARN] results_table.csv not found; departure green-filtering disabled.")
        return {}

    def _dir_to_mov(d: str):
        parts = d.replace("L_", "").split()
        if len(parts) >= 3 and parts[1] == "to":
            return parts[0][0] + parts[2][0]
        return None

    intervals: dict = {}
    try:
        with open(results_csv, newline="") as f:
            for row in _csv.DictReader(f, delimiter=";"):
                if row.get("phase") == "green":
                    mov = _dir_to_mov(row.get("direction", ""))
                    if mov:
                        intervals.setdefault(mov, []).append(
                            (float(row["start"]), float(row["end"]))
                        )
        total = sum(len(v) for v in intervals.values())
        print(f"[INFO] Green intervals loaded: {total} intervals across {len(intervals)} movements.")
    except Exception as e:
        print(f"[WARN] Could not load green intervals: {e}")
    return intervals


def parse_local_time_series(series):
    """Parse a Local_Time-like series into float seconds (zero-referenced).

    Tries multiple heuristics in order:
    1. If values contain ':' assume HH:MM:SS[.ms] strings and use pd.to_timedelta
    2. If numeric, try interpreting as seconds, milliseconds, microseconds, nanoseconds
       (we pick the first interpretation that yields a reasonable span)
    3. If numeric but none of the above work, assume a frame index and convert using
       a default FPS (29.97) as a last resort.

    Returns a numpy array of float seconds (series.min() -> 0.0).
    """
    s = series
    # If object/string-like and contains ':' try direct timedelta parse
    try:
        if s.dtype == object or str(s.dtype).startswith('str'):
            if s.str.contains(':').any():
                try:
                    times = pd.to_timedelta(s)
                    secs = (times - times.min()).dt.total_seconds().to_numpy(dtype=float)
                    if np.nanmax(secs) - np.nanmin(secs) > 1e-6:
                        return secs
                except Exception:
                    pass
    except Exception:
        pass

    # Try numeric interpretations: seconds, ms, us, ns
    try:
        num = pd.to_numeric(s, errors='coerce')
    except Exception:
        num = None

    if num is not None:
        units_try = ['s', 'ms', 'us', 'ns']
        for unit in units_try:
            try:
                times = pd.to_timedelta(num, unit=unit)
                span = (times.max() - times.min()).total_seconds()
                if span >= 0.001:
                    secs = (times - times.min()).dt.total_seconds().to_numpy(dtype=float)
                    print(f"[INFO] Parsed Local_Time as unit='{unit}' (span={span:.3f}s)")
                    return secs
            except Exception:
                continue

        # fallback: treat numeric as frame index -> convert using FPS
        fps = 29.97
        try:
            arr = num.to_numpy(dtype=float)
            secs = (arr - arr.min()) / float(fps)
            print(f"[WARN] Local_Time appears numeric but no time unit matched; assuming frame index with fps={fps}")
            return secs
        except Exception:
            pass

    # Last resort: enumerate rows as frames at default fps
    fps = 29.97
    n = len(series)
    print("[WARN] Could not parse Local_Time; falling back to row-index / fps conversion")
    return np.arange(n, dtype=float) / float(fps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the full Q-intersection analysis pipeline."""

    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    TRAJ_CSV = args.traj
    INTERSECTION_ID = infer_intersection_id_from_filename(TRAJ_CSV)

    SEG_CSV = (
        args.seg
        if args.seg is not None
        else os.path.join(args.seg_dir, f"{INTERSECTION_ID}.csv")
    )

    OUTPUT_PARENT = args.out
    os.makedirs(OUTPUT_PARENT, exist_ok=True)

    # ---- Monkey-patches ----
    patch_od_pairs()
    restore_mpl, _original_close = patch_matplotlib()

    # ---- Load trajectory CSV ----
    df = pd.read_csv(TRAJ_CSV, low_memory=False)
    lane_counts_all, lane_to_ids_all, missing_ids_all = summarize_lane_numbers(df)
    vehicles_total = df["Vehicle_ID"].nunique()
    labeled_vehicle_ids = set()
    for _lane, vids in lane_to_ids_all.items():
        labeled_vehicle_ids.update(vids)
    vehicles_with_lane = len(labeled_vehicle_ids)

    # ---- Parse time ----
    df["time_s"] = parse_local_time_series(df["Local_Time"])

    # ==========================
    # KNN lane assignment
    # ==========================
    feat_df = build_vehicle_feature_table(
        df,
        vehicle_id_col="Vehicle_ID",
        lane_col="Lane_Number",
        time_col="time_s",
        x_col="Ortho_X",
        y_col="Ortho_Y",
        speed_col="Vehicle_Speed",
        n_edge=10,
    )

    K = 9
    scaler, knn_model, T95, feature_cols, col_means = fit_vehicle_knn_with_thresholds(
        feat_df, k=K, lane_label_col="lane_mode",
    )

    pred_df = predict_missing_lanes_vehicle_knn(
        feat_df,
        scaler=scaler, knn=knn_model, T95=T95,
        feature_cols=feature_cols, col_means=col_means,
        k=K, conf_thresh=0.67, margin_thresh=0.22,
    )

    pred_df_for_assign = pred_df[["Vehicle_ID", "Lane_Inferred", "Lane_Confidence"]].copy()
    df = assign_lane_final_per_vehicle(
        df, pred_df_for_assign,
        vehicle_id_col="Vehicle_ID", lane_col="Lane_Number",
    )
    print_lane_coverage_before_after(df)

    n_missing_raw = int(feat_df["lane_mode"].isna().sum())
    n_assigned_knn = int(pred_df["Lane_Inferred"].notna().sum())
    n_rejected = n_missing_raw - n_assigned_knn

    if VERBOSE:
        print("\n=== FEATURE HEALTH CHECK ===")
        print("Total vehicles:", len(feat_df))
        print("Labeled vehicles:", feat_df["lane_mode"].notna().sum())
        print("Unlabeled vehicles:", feat_df["lane_mode"].isna().sum())

        u = feat_df[feat_df["lane_mode"].isna()]
        print("\nNaN counts in UNLABELED feature rows:")
        print(u.isna().sum().sort_values(ascending=False).to_string())

        l = feat_df[feat_df["lane_mode"].notna()]
        cols = ["x_start", "y_start", "x_end", "y_end",
                "x_mean", "y_mean", "dx", "dy", "T", "v_mean", "v_p90"]
        print("\nLabeled feature ranges:")
        print(l[cols].describe().loc[["min", "max"]].to_string())
        print("\nUnlabeled feature ranges:")
        print(u[cols].describe().loc[["min", "max"]].to_string())
        print("=================================\n")

        print("\nExamples (accepted):")
        print(pred_df[pred_df["reject_reason"] == "accepted"][
            ["Vehicle_ID", "Lane_Inferred", "Lane_Confidence", "dk"]
        ].head(10).to_string(index=False))
        print("\nExamples (rejected):")
        print(pred_df[pred_df["reject_reason"] != "accepted"][
            ["Vehicle_ID", "Lane_Inferred", "Lane_Confidence", "dk", "reject_reason"]
        ].head(10).to_string(index=False))
        print("===============================================\n")

    print(f"\n=== KNN LANE INFERENCE ===")
    print(f"Missing in raw CSV: {n_missing_raw} → KNN assigned: {n_assigned_knn}, Rejected: {n_rejected}")
    print(f"OOD threshold T95: {T95:.4f}")
    print(f"==========================\n")

    assigned_vehicle_ids = set(
        df.dropna(subset=["Lane_Final"])["Vehicle_ID"].astype(int).unique()
    )

    # ==========================
    # Build raw_data for tool_songdo
    # ==========================
    df_sorted = df.sort_values(["Vehicle_ID", "time_s"])

    ids, vtypes, xs, ys, ts = [], [], [], [], []
    for vid, g in df_sorted.groupby("Vehicle_ID"):
        ids.append(int(vid))
        vtypes.append(str(g["Vehicle_Class"].iloc[0]))
        xs.append(g["Ortho_X"].tolist())
        ys.append(g["Ortho_Y"].tolist())
        ts.append(g["time_s"].tolist())

    raw_data = {
        "id": ids, "vtype": vtypes,
        "x": xs, "y": ys, "time": ts,
        "speed": [[] for _ in xs],
    }

    time_axis = sorted(df_sorted["time_s"].unique())

    bbox = compute_bbox_latlon(df, lat_col="Ortho_Y", lon_col="Ortho_X")
    intersection_center = compute_center_latlon(df, lat_col="Ortho_Y", lon_col="Ortho_X")

    if bbox is None or intersection_center is None:
        raise ValueError(
            "Could not compute bbox/center from Latitude/Longitude. "
            "Check that these columns exist and are non-empty."
        )

    bbox_xy = [(p[1], p[0]) for p in bbox]
    center_xy = (intersection_center[1], intersection_center[0])

    spatio_temporal_info = {
        "bbox": bbox_xy,
        "intersection_center": center_xy,
        "time_axis": time_axis,
    }

    tool_master = Master()
    loader = tool_master.dataloader(raw_data, spatio_temporal_info)
    data_filtered = loader.get_filtered_data()
    data = data_filtered

    g_df = pd.read_csv(SEG_CSV)

    # ==========================
    # Trim trajectories to intersection boundary
    # ==========================
    df = trim_to_intersection(
        df, g_df, x_col="Ortho_X", y_col="Ortho_Y",
        vehicle_id_col="Vehicle_ID", margin=50.0,
    )

    # ==========================
    # Group vehicles by sections
    # ==========================
    allowed_ids = set(map(int, data["id"]))

    groups, start_end_section, start_end_laneidx, vehicle_to_movement, vehicle_state_sequence = \
        group_vehicles_by_sections(
            df=df, g_df=g_df,
            allowed_vehicle_ids=allowed_ids,
            time_col="time_s", x_col="Ortho_X", y_col="Ortho_Y",
            N=30, contain_radius=3.0,
            snap_dist_thresh=120.0, closest_fallback_thresh=200.0,
            grouping_mode=getattr(args, 'grouping_mode', 'rules'),
            manual_mapping_file=getattr(args, 'manual_mapping_file', None),
            export_sections_path=getattr(args, 'export_sections', None),
        )

    print_section_group_report_limited(groups, start_end_section, start_end_laneidx, max_ids=25)

    # ==========================
    # Free observation mode
    # ==========================
    work_vehicle_ids = sorted(allowed_ids)

    # Filter stationary vehicles
    def compute_displacement_m_obs(vid):
        vid_data = df[df["Vehicle_ID"] == vid]
        if len(vid_data) < 2:
            return 0.0
        xs_v = vid_data["Ortho_X"].values
        ys_v = vid_data["Ortho_Y"].values
        start = (ys_v[0], xs_v[0])
        end = (ys_v[-1], xs_v[-1])
        return tool_master.distances(start, end).get_distance()

    displacements = {vid: compute_displacement_m_obs(vid) for vid in work_vehicle_ids}
    THRESH_MOVE_M = 8.0

    moving_vehicle_ids = [vid for vid in work_vehicle_ids if displacements[vid] >= THRESH_MOVE_M]
    stationary_vehicle_ids = [vid for vid in work_vehicle_ids if displacements[vid] < THRESH_MOVE_M]
    work_vehicle_ids = sorted(moving_vehicle_ids)

    print(f"\n=== VEHICLE FILTERING ===")
    print(f"Total vehicles: {len(allowed_ids)} → Moving: {len(moving_vehicle_ids)}, Stationary: {len(stationary_vehicle_ids)} (< {THRESH_MOVE_M}m)")
    print(f"=========================\n")

    df_work = filter_df_by_vehicle_ids(df, work_vehicle_ids)
    df_by_move = {"ALL_VEHICLES_OBSERVED": df_work}

    # Filter data to moving vehicles only
    moving_vehicle_set = set(work_vehicle_ids)
    idx_keep = [i for i, vid in enumerate(data["id"]) if int(vid) in moving_vehicle_set]
    data = {k: [data[k][i] for i in idx_keep] for k in data.keys()}

    excluded_vehicle_ids_in_filtered = sorted(set(map(int, data["id"])) - set(work_vehicle_ids))
    excluded_vehicle_ids_overall = sorted(
        set(df["Vehicle_ID"].astype(int).unique()) - set(map(int, data["id"]))
    )

    work_set = set(work_vehicle_ids)
    idx_keep = [i for i, vid in enumerate(data["id"]) if int(vid) in work_set]
    data = {k: [data[k][i] for i in idx_keep] for k in data.keys()}

    # ==========================
    # tool_songdo analysis/viz
    # ==========================
    analysis = tool_master.analysis(data, spatio_temporal_info)
    visualization = tool_master.visualization(data, spatio_temporal_info)

    output_parent = OUTPUT_PARENT
    os.makedirs(output_parent, exist_ok=True)

    dataset_name = os.path.splitext(os.path.basename(TRAJ_CSV))[0]

    # Debug directory
    debug_dir = os.path.join(output_parent, f"DEBUG_{dataset_name}")
    os.makedirs(debug_dir, exist_ok=True)
    work_ids_path = os.path.join(debug_dir, "work_vehicle_ids.txt")
    excluded_ids_path = os.path.join(debug_dir, "excluded_vehicle_ids.txt")
    excluded_vehicle_ids = excluded_vehicle_ids_in_filtered
    with open(work_ids_path, "w") as f:
        for vid in work_vehicle_ids:
            f.write(f"{vid}\n")
    with open(excluded_ids_path, "w") as f:
        for vid in excluded_vehicle_ids:
            f.write(f"{vid}\n")

    random_vehicle_id = random.choice(work_vehicle_ids)

    run_folder_name = f"vehicle_{random_vehicle_id}_{dataset_name}"
    run_dir = os.path.join(output_parent, run_folder_name)
    os.makedirs(run_dir, exist_ok=True)

    # Structured output folders
    general_dir = os.path.join(run_dir, "general")
    vehicle_stats_dir = os.path.join(run_dir, f"vehicle_{random_vehicle_id}_statistics")

    os.makedirs(general_dir, exist_ok=True)
    os.makedirs(vehicle_stats_dir, exist_ok=True)

    # Dump observation workset
    try:
        dump_json = os.path.join(debug_dir, "observation_workset.json")
        with open(dump_json, "w") as jf:
            json.dump({"ALL_VEHICLES_OBSERVED": work_vehicle_ids}, jf, indent=2)
        dump_csv = os.path.join(debug_dir, "observation_workset.csv")
        with open(dump_csv, "w", newline='') as cf:
            w = _csv.writer(cf)
            w.writerow(["observation_group", "vehicle_id"])
            for vid in work_vehicle_ids:
                w.writerow(["ALL_VEHICLES_OBSERVED", vid])
        dbg(f"Wrote observation workset: {dump_json} and {dump_csv}")
    except Exception:
        pass

    # G-maps, lane maps, trajectories, route map
    print("\n=== GENERATING SUPPORTING PLOTS ===")
    g_lane_path, g_section_path = save_g_maps_into_run_dir(
        run_dir=general_dir, g_csv_path=SEG_CSV,
        df_points=df, allowed_vehicle_ids=work_vehicle_ids,
    )

    # tool_songdo visualizations
    def draw_and_save(draw_func, path, *a, **kw):
        with np.errstate(divide="ignore", invalid="ignore"):
            draw_func(*a, **kw)
        fig = plt.gcf()
        fig.savefig(path, dpi=200)
        _original_close(fig)

    fig_route = visualization.draw_trajectories_od()
    route_map_path = os.path.join(general_dir, "route_map.png")
    fig_route.savefig(route_map_path, dpi=200)
    _original_close(fig_route)
    print("  Segmentation maps, lane maps, trajectory maps saved")

    # ==========================
    # Turning Arrow Diagram
    # ==========================
    print("\n=== GENERATING CUMULATIVE & TURNING ARROW DIAGRAMS ===")

    movement_to_ids, unassigned_ids = build_movement_to_ids_for_workset(
        work_vehicle_ids=work_vehicle_ids,
        start_end_section=start_end_section,
        vehicle_to_movement=vehicle_to_movement,
    )

    movement_to_ids_valid = {k: v for k, v in movement_to_ids.items() if k != "UNASSIGNED"}

    # Export movement mapping for animation script
    try:
        movement_json_path = os.path.join(general_dir, "movement_to_ids.json")
        movement_export = {k: [int(v) for v in vids] for k, vids in movement_to_ids.items()}
        with open(movement_json_path, "w") as jf:
            json.dump(movement_export, jf, indent=2)
        dbg(f"  Movement mapping exported: {movement_json_path}")
    except Exception:
        pass

    base_order = [
        'SN', 'SW', 'SE', 'SS',
        'NS', 'NN', 'NW', 'NE',
        'WN', 'WE', 'WS',
        'EN', 'EW', 'ES',
    ]
    movement_plot_order = [m for m in base_order if m in movement_to_ids_valid]

    # --- Per-segment analysis ---
    recording_segments = _detect_recording_segments(df)
    print(f"\nDetected {len(recording_segments)} recording segments: {[s[0] for s in recording_segments]}")

    all_reject_rows = []   # collect from all segments for rejected maps
    all_unknown_lane_vids = []

    for seg_label, t_start_s, t_end_s in recording_segments:
        print(f"\n{'='*70}")
        print(f"=== SEGMENT: {seg_label}  ({t_start_s:.1f}s – {t_end_s:.1f}s) ===")
        print(f"{'='*70}")

        seg_mask = (df["time_s"] >= t_start_s) & (df["time_s"] <= t_end_s)
        df_seg = df[seg_mask].copy()
        if df_seg.empty:
            print(f"  [SKIP] No data in segment {seg_label}")
            continue

        seg_vids_in_window = set(df_seg["Vehicle_ID"].astype(int).unique())
        seg_work_vids = [v for v in work_vehicle_ids if v in seg_vids_in_window]
        if not seg_work_vids:
            print(f"  [SKIP] No work vehicles in segment {seg_label}")
            continue

        # Build segment directory structure
        # signal_timing_diagrams/ is created inside seg_dir by generate_space_time_diagram
        # (it uses general_dir as its base, so we pass seg_dir as general_dir)
        seg_dir = os.path.join(run_dir, f"segment_{seg_label}")
        seg_st_dir      = os.path.join(seg_dir, "space_time")
        seg_cum_mov_dir = os.path.join(seg_dir, "cumulative_by_movement")
        seg_cum_lane_dir = os.path.join(seg_dir, "cumulative_by_lane")
        seg_hw_dir      = os.path.join(seg_dir, "headway")
        seg_hw_dist_dir = os.path.join(seg_hw_dir, "headway_distributions")

        for d in [seg_st_dir, seg_cum_mov_dir, seg_cum_lane_dir,
                  seg_hw_dir, seg_hw_dist_dir]:
            os.makedirs(d, exist_ok=True)

        # Space-time (writes results_table.csv + signal_timing_diagrams/ into seg_dir)
        print(f"\n  Space-time diagrams...")
        seg_st = generate_space_time_diagram(
            df=df_seg,
            out_path=os.path.join(seg_st_dir, "space_time.png"),
            g_df=g_df,
            allowed_vehicle_ids=seg_work_vids,
            title=f"{INTERSECTION_ID} - Space-Time — {seg_label}",
            x_col="Ortho_X", y_col="Ortho_Y",
            time_col="time_s", vehicle_id_col="Vehicle_ID",
            vehicle_state_sequence=vehicle_state_sequence,
            general_dir=seg_dir,          # signal_timing_diagrams/ and results_table.csv go here
            headway_dir=seg_hw_dir,
            vehicle_to_movement=vehicle_to_movement,
        )

        # Collect reject rows for the global rejected maps
        if seg_st is not None and hasattr(seg_st, "reject_rows"):
            all_reject_rows.extend(seg_st.reject_rows)
            if seg_st.plot_df is not None and not seg_st.plot_df.empty:
                unk = seg_st.plot_df[seg_st.plot_df["lane"] == "unknown"]["vid"].unique().tolist()
                all_unknown_lane_vids.extend(unk)

        # Green intervals for this segment
        seg_green = _load_green_intervals(seg_dir)

        # Movement-to-IDs filtered to this segment
        seg_movement_to_ids = {
            mov: [v for v in vids if v in seg_vids_in_window]
            for mov, vids in movement_to_ids_valid.items()
        }
        seg_movement_to_ids = {k: v for k, v in seg_movement_to_ids.items() if v}

        # Cumulative by movement
        print(f"  Cumulative by movement...")
        seg_mov_event_times = compute_movement_arrival_departure_times(
            df=df_seg, movement_to_ids=seg_movement_to_ids,
            vehicle_id_col="Vehicle_ID", time_col="time_s",
            speed_col="Vehicle_Speed", section_col="Road_Section",
            x_col="Ortho_X", y_col="Ortho_Y",
            g_df=g_df, stop_speed_kmh=10.0,
            green_intervals_by_movement=seg_green,
        )
        seg_mov_order = [m for m in movement_plot_order if m in seg_mov_event_times]
        plot_cumulative_by_movement(
            movement_event_times=seg_mov_event_times,
            out_dir=seg_cum_mov_dir,
            movement_order=seg_mov_order,
            green_intervals_by_movement=seg_green,
        )

        # Cumulative by lane
        print(f"  Cumulative by lane...")
        seg_lane_event_times = compute_lane_arrival_departure_times(
            df=df_seg, movement_to_ids=seg_movement_to_ids,
            vehicle_id_col="Vehicle_ID", time_col="time_s",
            speed_col="Vehicle_Speed", section_col="Road_Section",
            lane_col="Lane_Number",
            x_col="Ortho_X", y_col="Ortho_Y",
            g_df=g_df, stop_speed_kmh=10.0,
            green_intervals_by_movement=seg_green,
        )
        plot_cumulative_by_lane(
            lane_event_times=seg_lane_event_times,
            out_dir=seg_cum_lane_dir,
            green_intervals_by_movement=seg_green,
        )

        # Headway distributions
        print(f"  Headway distributions...")
        try:
            plot_headway_distributions(
                lane_event_times=seg_lane_event_times,
                out_dir=seg_hw_dist_dir,
            )
        except Exception as e:
            print(f"  [ERROR] Headway distributions: {e}")

        print(f"  Segment {seg_label} complete.")

    # ── Full-recording signal timing diagrams ─────────────────────────────
    # Run inference on the full dataset so the signal timing shows all 4
    # recording segments combined.  The space-time PNG is written to a
    # temporary path inside general_dir and deleted afterwards; only the
    # signal_timing_diagrams/ subfolder (renamed to
    # signal_timing_full_recording/) is kept.
    print("\n=== GENERATING FULL-RECORDING SIGNAL TIMING ===")
    import shutil as _shutil
    _full_st_tmp = os.path.join(general_dir, "_full_recording_space_time_tmp.png")
    try:
        generate_space_time_diagram(
            df=df,
            out_path=_full_st_tmp,
            g_df=g_df,
            allowed_vehicle_ids=work_vehicle_ids,
            title=f"{INTERSECTION_ID} - Space-Time — Full Recording",
            x_col="Ortho_X", y_col="Ortho_Y",
            time_col="time_s", vehicle_id_col="Vehicle_ID",
            vehicle_state_sequence=vehicle_state_sequence,
            general_dir=general_dir,
            headway_dir=None,
            vehicle_to_movement=vehicle_to_movement,
        )
        # Rename signal_timing_diagrams → signal_timing_full_recording
        _src = os.path.join(general_dir, "signal_timing_diagrams")
        _dst = os.path.join(general_dir, "signal_timing_full_recording")
        if os.path.isdir(_src):
            if os.path.isdir(_dst):
                _shutil.rmtree(_dst)
            os.rename(_src, _dst)
            print(f"  Full-recording signal timing saved → general/signal_timing_full_recording/")
    except Exception as _e:
        print(f"  [ERROR] Full-recording signal timing: {_e}")
    finally:
        if os.path.exists(_full_st_tmp):
            os.remove(_full_st_tmp)

    # ── Build rejected vehicle groups for diagnostic maps ────────
    # Uses combined reject data from all segment runs
    from collections import defaultdict

    rejected_groups = {}

    # Group 1: stationary
    rejected_groups["stationary"] = sorted(stationary_vehicle_ids)

    # Groups from space-time quality filter (all segments combined)
    if all_reject_rows:
        reason_to_vids = defaultdict(list)
        for row in all_reject_rows:
            reason_to_vids[row["reason"]].append(row["vid"])

        # Map internal reason codes to user-facing labels (matching the funnel table)
        _REASON_ORDER = [
            ("silently_skipped_unk_no_origin", "silently_skipped_UNK_no_origin"),
            ("silently_skipped_no_origin_set", "silently_skipped_UNK_no_origin"),
            ("no_upstream_approach", "no_upstream_approach"),
            ("too_many_breaks", "too_many_breaks"),
            ("no_downstream_departure", "no_downstream_departure"),
            ("too_short_in_time", "too_short_in_time"),
            ("too_few_points", "too_few_points"),
            ("no_clean_stopline_crossing", "no_clean_stopline_crossing"),
            ("large_internal_gap", "large_internal_gap"),
            ("single_point_after_clip", "single_point_after_clip"),
            ("small_y_range", "small_y_range"),
        ]
        # Merge reasons that map to the same label
        merged = defaultdict(list)
        for internal_key, label in _REASON_ORDER:
            if internal_key in reason_to_vids:
                merged[label].extend(reason_to_vids[internal_key])
        for label, vids in merged.items():
            rejected_groups[label] = sorted(set(vids))

    # Unknown lane (passed quality but couldn't be assigned a lane)
    if all_unknown_lane_vids:
        rejected_groups["unknown_lane"] = sorted(set(int(v) for v in all_unknown_lane_vids))

    # Plot the rejection maps
    rejected_dir = os.path.join(run_dir, "rejected")
    rejected_paths = plot_rejected_vehicle_maps(
        df=df, rejected_groups=rejected_groups, g_df=g_df,
        out_dir=rejected_dir,
        x_col="Ortho_X", y_col="Ortho_Y",
        vehicle_id_col="Vehicle_ID",
    )
    print(f"  Rejected vehicle maps: {len(rejected_paths)} diagrams in rejected/")

    # Turning arrow diagram + intersection layout
    arrow_path = os.path.join(general_dir, "turning_arrow_diagram.png")
    try:
        _original_close("all")
        plot_turning_arrow_diagram(
            movement_to_ids,
            out_path=arrow_path,
            title="M Intersection - Detected Vehicle Movements",
        )
    except Exception as e:
        print(f"[ERROR] Turning arrow diagram: {e}")

    layout_path = os.path.join(general_dir, "intersection_layout.png")
    try:
        plot_intersection_layout(
            out_path=layout_path,
            title=f"{INTERSECTION_ID} Intersection — Layout & Movement Directions",
        )
    except Exception as e:
        print(f"[ERROR] Intersection layout: {e}")
    print("  Turning arrow diagram + intersection layout saved")

    # Recording timeline (show recording segments & gaps)
    timeline_path = os.path.join(general_dir, "recording_timeline.png")
    try:
        plot_recording_timeline(
            df,
            out_path=timeline_path,
            local_time_col="Local_Time",
            vehicle_id_col="Vehicle_ID",
            gap_threshold_minutes=1.5,
            title=f"{INTERSECTION_ID} Intersection - Recording Timeline & Gaps",
        )
    except Exception as e:
        print(f"[ERROR] Recording timeline: {e}")

    # Per-vehicle statistics
    fig_dist = visualization.draw_distance_travelled(random_vehicle_id)
    distance_path = os.path.join(vehicle_stats_dir, "vehicle_distance.png")
    fig_dist.savefig(distance_path, dpi=200)
    _original_close(fig_dist)

    fig_speed = visualization.draw_speed(random_vehicle_id)
    speed_path = os.path.join(vehicle_stats_dir, "vehicle_speed.png")
    fig_speed.savefig(speed_path, dpi=200)
    _original_close(fig_speed)

    fig_acc = visualization.draw_acceleration(random_vehicle_id)
    acc_path = os.path.join(vehicle_stats_dir, "vehicle_acceleration.png")
    fig_acc.savefig(acc_path, dpi=200)
    _original_close(fig_acc)
    print(f"  Sample vehicle stats (ID={random_vehicle_id}): distance, speed, acceleration")

    print("\n" + "=" * 80)
    print("Analysis complete! Check outputs folder for results.")
    print("=" * 80)

    # Restore matplotlib patches
    restore_mpl()
    plt.close = _original_close
    tool.plt.close = _original_close
