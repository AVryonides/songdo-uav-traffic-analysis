#plotting.py
import os
import re
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon
from typing import Optional

from .config import _THESIS_FONT, _save_fig_formats


def plot_lane_map(
    df: pd.DataFrame,
    x_col: str = "Local_X",
    y_col: str = "Local_Y",
    vehicle_id_col: str = "Vehicle_ID",
    lane_col: str = "Lane_Number",
    allowed_vehicle_ids=None,
    kind: str = "points",
    max_points: int = 60000,
    vehicles_per_lane: int = 25,
    out_path: str = "lane_map.png",
    plot_missing: bool = True,
):
    """
    Draw a 2D map using coordinates (x_col,y_col) colored by Lane_Number.

    - kind="points": scatter a (downsampled) set of points per lane.
    - kind="trajectories": plot a small number of vehicle paths per lane.

    Saves to out_path and returns out_path.
    """

    d = df.copy()

    if allowed_vehicle_ids is not None:
        allowed_set = set(allowed_vehicle_ids)
        d = d[d[vehicle_id_col].isin(allowed_set)]


    for c in [x_col, y_col, vehicle_id_col, lane_col]:
        if c not in d.columns:
            raise ValueError(f"Column '{c}' not found in df. Available: {list(d.columns)}")

    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d[lane_col] = pd.to_numeric(d[lane_col], errors="coerce")

    d = d.dropna(subset=[x_col, y_col])

    d_lane = d.dropna(subset=[lane_col]).copy()
    d_missing = d[d[lane_col].isna()].copy()

    d_lane[lane_col] = d_lane[lane_col].astype(int)

    fig, ax = plt.subplots(figsize=(10, 8))

    if kind == "points":

        if len(d_lane) > max_points:
            d_lane_plot = d_lane.sample(n=max_points, random_state=0)
        else:
            d_lane_plot = d_lane

        for lane in sorted(d_lane_plot[lane_col].unique()):
            s = d_lane_plot[d_lane_plot[lane_col] == lane]
            ax.scatter(s[x_col], s[y_col], s=4, alpha=0.6, label=f"Lane {lane}")


        if plot_missing and len(d_missing) > 0:
            d_missing_plot = d_missing.sample(n=min(len(d_missing), max_points // 5), random_state=0)
            ax.scatter(d_missing_plot[x_col], d_missing_plot[y_col], s=3, alpha=0.2, label="Lane missing")


    elif kind == "trajectories":

        for lane in sorted(d_lane[lane_col].unique()):
            vids = d_lane.loc[d_lane[lane_col] == lane, vehicle_id_col].unique().tolist()
            random.Random(0).shuffle(vids)
            vids = vids[:vehicles_per_lane]

            for vid in vids:
                g = d_lane[(d_lane[vehicle_id_col] == vid) & (d_lane[lane_col] == lane)]

                if "time_s" in g.columns:
                    g = g.sort_values("time_s")
                ax.plot(g[x_col].values, g[y_col].values, linewidth=1, alpha=0.7)


            ax.plot([], [], linewidth=3, label=f"Lane {lane}")

    else:
        raise ValueError("kind must be 'points' or 'trajectories'")

    ax.set_title(f"Lane map using {x_col} vs {y_col} ({kind})")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.axis("equal")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _plot_per_cycle_departures(
    departures: np.ndarray,
    green_intervals: list,
    out_path: str,
    title: str,
) -> str:
    """Per-cycle cumulative departure plot.

    Departure count resets to 0 at each green onset.  Each cycle's staircase
    is drawn independently on an absolute time x-axis.
    """
    sorted_intervals = sorted(green_intervals, key=lambda x: x[0])
    if not sorted_intervals:
        return None

    with plt.rc_context(_THESIS_FONT):
        fig, ax = plt.subplots(figsize=(12, 5))

        first_label = True
        for t0, t1 in sorted_intervals:
            cycle_deps = np.sort(departures[(departures >= t0) & (departures <= t1)])
            n = len(cycle_deps)
            if n == 0:
                # Keep the cycle baseline visible even with no departures.
                ax.plot([t0, t1], [0, 0], color="tab:orange", linewidth=1.5,
                        alpha=0.4, zorder=2)
                continue

            # Staircase starting from (t0, 0)
            xs = np.concatenate([[t0], cycle_deps])
            ys = np.concatenate([[0], np.arange(1, n + 1)])
            label = "Stop-line crossing (green only)" if first_label else "_nolegend_"
            ax.step(xs, ys, where="post", color="tab:orange",
                    linewidth=2.0, label=label, zorder=3)
            first_label = False
            # Count annotation at end of green window
            ax.text(t1 + 0.5, n, str(n), va="center", ha="left",
                    fontsize=8, color="tab:orange", zorder=4)

        legend_handles = [
            plt.Line2D([], [], color="tab:orange", linewidth=2,
                       label="Stop-line crossing (green only)"),
        ]
        ax.legend(handles=legend_handles, loc="upper left")

        all_t = [t for t0, t1 in sorted_intervals for t in (t0, t1)]
        if departures.size > 0:
            all_t += list(departures)
        ax.set_xlim(min(all_t) - 5, max(all_t) + 20)
        ax.set_ylim(bottom=-0.3)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative Departures (per cycle)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        _save_fig_formats(fig, out_path)
        plt.close(fig)
        return out_path


def plot_cumulative_by_movement(
    movement_event_times: dict,
    out_dir: str,
    movement_order=None,
    green_intervals_by_movement: dict = None,
) -> list:
    """
    Saves THREE cumulative plots per movement:
      - arrivals only
      - departures only  (per-cycle: resets to 0 at each green onset)
      - combined arrivals + departures (departures green-phase filtered)
    Skips movements with no events at all.
    """
    with plt.rc_context(_THESIS_FONT):
        _MOVEMENT_LABELS = {
            "NS": "N to S", "NW": "N to W", "NE": "N to E", "NN": "N to N",
            "SN": "S to N", "SW": "S to W", "SE": "S to E", "SS": "S to S",
            "WN": "W to N", "WE": "W to E", "WS": "W to S",
            "EN": "E to N", "EW": "E to W", "ES": "E to S",
        }

        def _movement_label(movement) -> str:
            return _MOVEMENT_LABELS.get(str(movement), str(movement))

        def _movement_folder(movement) -> str:
            return _movement_label(movement).replace(" ", "_")

        def _set_compressed_time_ticks(ax, change_times: np.ndarray, max_ticks: int = 12):
            x_event_idx = np.arange(len(change_times), dtype=int)
            if len(change_times) <= max_ticks:
                tick_idx = x_event_idx
            else:
                tick_idx = np.linspace(0, len(change_times) - 1, max_ticks, dtype=int)
                tick_idx = np.unique(tick_idx)
            tick_labels = [f"{change_times[i]:.1f}" for i in tick_idx]
            ax.set_xticks(tick_idx)
            ax.set_xticklabels(tick_labels, rotation=30, ha="right")

        def _plot_compressed_cumulative_single(
            event_times: np.ndarray,
            out_path: str,
            title: str,
            series_label: str,
            color: str,
        ) -> str:
            t = np.sort(np.asarray(event_times, dtype=float))
            if len(t) == 0:
                return None  # skip empty plots entirely
            fig, ax = plt.subplots(figsize=(10, 6))

            change_times = np.unique(t)
            x_event_idx = np.arange(len(change_times), dtype=int)
            y_cum = np.searchsorted(t, change_times, side="right")

            ax.plot(x_event_idx, y_cum, linewidth=2.5, label=series_label, color=color)
            _set_compressed_time_ticks(ax, change_times)
            ax.legend(loc="best")
            ax.set_ylim(bottom=-0.5, top=y_cum.max() + 0.5)

            ax.set_title(title)
            ax.set_xlabel("Event Index (tick labels = time in s)")
            ax.set_ylabel("Cumulative Vehicles")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            _save_fig_formats(fig, out_path)
            plt.close(fig)
            return out_path

        os.makedirs(out_dir, exist_ok=True)
        if movement_order is None:
            movement_order = sorted(movement_event_times.keys())

        saved = []
        for movement in movement_order:
            if movement not in movement_event_times:
                continue
            ev = movement_event_times[movement]
            arrivals = np.sort(np.asarray(ev.get("arrivals", []), dtype=float))
            departures = np.sort(np.asarray(ev.get("departures", []), dtype=float))

            # Skip movements with zero events on both sides
            if len(arrivals) == 0 and len(departures) == 0:
                print(f"[INFO] Skipping cumulative plot for {movement}: no arrivals or departures.")
                continue

            movement_label = _movement_label(movement)
            movement_dir = os.path.join(out_dir, _movement_folder(movement))
            os.makedirs(movement_dir, exist_ok=True)

            arr_path = os.path.join(movement_dir, "cumulative_arrival.png")
            dep_path = os.path.join(movement_dir, "cumulative_departure.png")
            comb_path = os.path.join(movement_dir, "cumulative_arrival_departure.png")

            p = _plot_compressed_cumulative_single(
                event_times=arrivals,
                out_path=arr_path,
                title=f"Cumulative Approach Arrival: {movement_label}",
                series_label="Approach arrival",
                color="tab:blue",
            )
            if p:
                saved.append(p)

            # Departure plot: per-cycle (resets at each green onset) when
            # green intervals are available; otherwise fall back to compressed.
            mov_green = (green_intervals_by_movement or {}).get(movement, [])
            if mov_green and len(departures) > 0:
                p = _plot_per_cycle_departures(
                    departures=departures,
                    green_intervals=mov_green,
                    out_path=dep_path,
                    title=f"Cumulative Stop-line Crossing (per cycle): {movement_label}",
                )
            else:
                p = _plot_compressed_cumulative_single(
                    event_times=departures,
                    out_path=dep_path,
                    title=f"Cumulative Stop-line Crossing: {movement_label}",
                    series_label="Stop-line crossing",
                    color="tab:orange",
                )
            if p:
                saved.append(p)

            # Combined plot — only if at least one series has data
            fig, ax = plt.subplots(figsize=(10, 6))
            if len(arrivals) > 0 and len(departures) > 0:
                change_times = np.unique(np.concatenate([arrivals, departures]))
            elif len(arrivals) > 0:
                change_times = np.unique(arrivals)
            else:
                change_times = np.unique(departures)

            x_event_idx = np.arange(len(change_times), dtype=int)
            
            # Initialize arrays so we can compute the difference (Queue)
            arr_cum = np.zeros(len(change_times))
            dep_cum = np.zeros(len(change_times))

            if len(arrivals) > 0:
                arr_cum = np.searchsorted(arrivals, change_times, side="right")
                ax.plot(x_event_idx, arr_cum, linewidth=2.5, label="Approach arrival", color="tab:blue")
            if len(departures) > 0:
                dep_cum = np.searchsorted(departures, change_times, side="right")
                ax.plot(x_event_idx, dep_cum, linewidth=2.5, label="Stop-line crossing", color="tab:orange")

            # --- Q ANNOTATION LOGIC (thesis style) ---
            if len(arrivals) > 0 and len(departures) > 0:
                queue_lengths = arr_cum - dep_cum  # positive = vehicles queued

                # --- Average arrival rate dashed line (reference image style) ---
                if len(arrivals) >= 2:
                    avg_slope = arr_cum[-1] / max(len(x_event_idx) - 1, 1)
                    avg_line = avg_slope * x_event_idx
                    ax.plot(x_event_idx, avg_line, linestyle='--', linewidth=1.2,
                            color='gray', alpha=0.6, label=r"Avg. arrival rate $q$", zorder=1)

                # --- Find local Q peaks (one per "hump" in the queue curve) ---
                # A peak is where queue goes from rising to falling
                q_peaks = []
                min_q_display = 3  # show Q if at least 3 vehicles queued
                for j in range(1, len(queue_lengths) - 1):
                    q_val = queue_lengths[j]
                    if q_val < min_q_display:
                        continue
                    # Local max: higher than or equal to neighbors, and strictly
                    # higher than at least one neighbor
                    if (q_val >= queue_lengths[j - 1] and
                            q_val >= queue_lengths[j + 1] and
                            (q_val > queue_lengths[j - 1] or q_val > queue_lengths[j + 1])):
                        q_peaks.append((j, int(q_val)))

                # Also check endpoints
                if len(queue_lengths) > 0 and queue_lengths[0] >= min_q_display:
                    q_peaks.insert(0, (0, int(queue_lengths[0])))
                if (len(queue_lengths) > 1 and
                        queue_lengths[-1] >= min_q_display and
                        queue_lengths[-1] >= queue_lengths[-2]):
                    q_peaks.append((len(queue_lengths) - 1, int(queue_lengths[-1])))

                # Merge nearby peaks: keep only the tallest within a window
                if len(q_peaks) > 1:
                    merge_window = max(5, len(x_event_idx) // 15)
                    merged = [q_peaks[0]]
                    for idx_q, val_q in q_peaks[1:]:
                        if idx_q - merged[-1][0] < merge_window:
                            # Keep the taller one
                            if val_q > merged[-1][1]:
                                merged[-1] = (idx_q, val_q)
                        else:
                            merged.append((idx_q, val_q))
                    q_peaks = merged

                # Limit to top 4 peaks to avoid clutter
                if len(q_peaks) > 4:
                    q_peaks = sorted(q_peaks, key=lambda p: p[1], reverse=True)[:4]
                    q_peaks = sorted(q_peaks, key=lambda p: p[0])  # re-sort by time

                # --- Draw each Q annotation ---
                q_color = 'forestgreen'
                text_side_toggle = 1  # alternate annotation sides to avoid overlap
                for k, (peak_idx, peak_val) in enumerate(q_peaks):
                    y_top_q = arr_cum[peak_idx]
                    y_bot_q = dep_cum[peak_idx]

                    # Vertical line between arrival & departure curves
                    ax.vlines(x=peak_idx, ymin=y_bot_q, ymax=y_top_q,
                              color=q_color, linestyle='-', linewidth=2.0, zorder=4)

                    # Small horizontal ticks at top and bottom (reference image style)
                    tick_half = max(1, len(x_event_idx) // 80)
                    ax.hlines(y=y_top_q, xmin=peak_idx - tick_half, xmax=peak_idx + tick_half,
                              color=q_color, linewidth=1.5, zorder=4)
                    ax.hlines(y=y_bot_q, xmin=peak_idx - tick_half, xmax=peak_idx + tick_half,
                              color=q_color, linewidth=1.5, zorder=4)

                    # Label: Q^i style for multiple, plain Q for single
                    y_center = y_bot_q + (peak_val / 2.0)
                    if len(q_peaks) > 1:
                        label_text = f"$Q^{{{k + 1}}}={peak_val}$"
                    else:
                        label_text = f"$Q={peak_val}$"

                    # Alternate text offset left/right to reduce overlap
                    x_offset = 35 * text_side_toggle
                    text_side_toggle *= -1

                    ax.annotate(
                        label_text,
                        xy=(peak_idx, y_center),
                        xytext=(x_offset, 0),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color=q_color, lw=1.2),
                        color=q_color, fontsize=11, fontweight='bold',
                        ha='left' if x_offset > 0 else 'right', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                  ec=q_color, lw=1, alpha=0.9),
                        zorder=5,
                    )
            # --- END Q ANNOTATION LOGIC ---

            _set_compressed_time_ticks(ax, change_times)
            ax.legend(loc="best")
            y_top = max(
                arr_cum.max() if len(arrivals) > 0 else 0,
                dep_cum.max() if len(departures) > 0 else 0,
            )
            ax.set_ylim(bottom=-0.5, top=y_top + 0.5)

            ax.set_title(f"Cumulative Arrivals/Departures: {movement_label}")
            ax.set_xlabel("Event Index (tick labels = time in s)")
            ax.set_ylabel("Cumulative Vehicles")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            _save_fig_formats(fig, comb_path)
            plt.close(fig)
            saved.append(comb_path)

    return saved


def plot_cumulative_by_lane(
    lane_event_times: dict,
    out_dir: str,
    green_intervals_by_movement: dict = None,
) -> list:
    """
    Saves THREE cumulative plots per (movement, lane):
      - arrivals only
      - departures only  (per-cycle when green intervals available)
      - combined arrivals + departures (green phase shading)
    Uses SCATTER POINTS (not continuous lines) since data is discrete.
    Uses real time on x-axis.  Skips lanes with no events.
    """
    with plt.rc_context(_THESIS_FONT):

        def _plot_cumulative_scatter_single(
            event_times: np.ndarray,
            out_path: str,
            title: str,
            series_label: str,
            color: str,
        ) -> str:
            t = np.sort(np.asarray(event_times, dtype=float))
            if len(t) == 0:
                return None  # skip empty plots entirely
            fig, ax = plt.subplots(figsize=(10, 6))

            y_cum = np.arange(1, len(t) + 1)
            ax.scatter(t, y_cum, s=40, label=series_label, color=color, zorder=3)
            t_range = t[-1] - t[0] if len(t) > 1 else max(t[0], 1.0)
            ax.set_xlim(left=-t_range * 0.03)
            ax.set_ylim(bottom=-0.5, top=len(t) + 0.5)

            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Cumulative Vehicles")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()
            _save_fig_formats(fig, out_path)
            plt.close(fig)
            return out_path

        # Descriptive direction labels for folder naming (M intersection movements)
        _MOVEMENT_DESCRIPTIONS = {
            'SN': 'South_to_North', 'SW': 'South_to_West',
            'SE': 'South_to_East',  'SS': 'South_to_South',
            'NS': 'North_to_South', 'NN': 'North_to_North',
            'NW': 'North_to_West',  'NE': 'North_to_East',
            'WN': 'West_to_North',  'WE': 'West_to_East',
            'WS': 'West_to_South',
            'EN': 'East_to_North',  'EW': 'East_to_West',
            'ES': 'East_to_South',
        }

        os.makedirs(out_dir, exist_ok=True)

        # Sort keys for deterministic output, skip UNASSIGNED and unknown lanes
        sorted_keys = sorted(
            [k for k in lane_event_times.keys()
             if k[0] != "UNASSIGNED" and k[1] != "unknown"],
            key=lambda k: (str(k[0]), str(k[1])),
        )

        # ---- Write summary table (CSV) with vehicle counts per lane ----
        import csv as _csv
        table_path = os.path.join(out_dir, "lane_vehicle_table.csv")
        with open(table_path, "w", newline="") as csvf:
            writer = _csv.writer(csvf)
            writer.writerow(["movement", "direction", "lane", "event_type", "vehicle_count"])
            for (movement, lane) in sorted_keys:
                dir_desc = _MOVEMENT_DESCRIPTIONS.get(movement, movement)
                ev = lane_event_times[(movement, lane)]
                n_arr = len(ev.get("arrival_details", []))
                n_dep = len(ev.get("departure_details", []))
                writer.writerow([movement, dir_desc, lane, "arrival", n_arr])
                writer.writerow([movement, dir_desc, lane, "departure", n_dep])
        # table_path saved silently

        saved = []
        for (movement, lane) in sorted_keys:
            dir_desc = _MOVEMENT_DESCRIPTIONS.get(movement, movement)
            ev = lane_event_times[(movement, lane)]
            arrivals = np.sort(np.asarray(ev.get("arrivals", []), dtype=float))
            departures = np.sort(np.asarray(ev.get("departures", []), dtype=float))

            # Skip lanes with too few vehicles (sum of arrivals + departures < 5)
            if len(arrivals) + len(departures) < 5:
                continue

            lane_dir = os.path.join(out_dir, f"{dir_desc}_lane_{lane}")
            os.makedirs(lane_dir, exist_ok=True)

            title_prefix = f"{movement} ({dir_desc.replace('_', ' ')}) Lane {lane}"

            arr_path = os.path.join(lane_dir, "cumulative_arrival.png")
            dep_path = os.path.join(lane_dir, "cumulative_departure.png")
            comb_path = os.path.join(lane_dir, "cumulative_arrival_departure.png")

            p = _plot_cumulative_scatter_single(
                event_times=arrivals,
                out_path=arr_path,
                title=f"Cumulative Approach Arrival: {title_prefix}",
                series_label="Approach arrival",
                color="tab:blue",
            )
            if p:
                saved.append(p)

            lane_green = (green_intervals_by_movement or {}).get(movement, [])
            if lane_green and len(departures) > 0:
                p = _plot_per_cycle_departures(
                    departures=departures,
                    green_intervals=lane_green,
                    out_path=dep_path,
                    title=f"Cumulative Stop-line Crossing (per cycle): {title_prefix}",
                )
            else:
                p = _plot_cumulative_scatter_single(
                    event_times=departures,
                    out_path=dep_path,
                    title=f"Cumulative Stop-line Crossing: {title_prefix}",
                    series_label="Stop-line crossing",
                    color="tab:orange",
                )
            if p:
                saved.append(p)

            # Combined plot — only if at least one series has data
            fig, ax = plt.subplots(figsize=(10, 6))

            if len(arrivals) > 0:
                y_arr = np.arange(1, len(arrivals) + 1)
                ax.scatter(arrivals, y_arr, s=40, label="Approach arrival", color="tab:blue", zorder=3)
            if len(departures) > 0:
                y_dep = np.arange(1, len(departures) + 1)
                ax.scatter(departures, y_dep, s=40, label="Stop-line crossing", color="tab:orange", zorder=3)

            # --- NEW Q ANNOTATION LOGIC ---
            if len(arrivals) > 0 and len(departures) > 0:
                # Calculate queue at all event times
                all_unique_times = np.unique(np.concatenate([arrivals, departures]))
                arr_cum = np.searchsorted(arrivals, all_unique_times, side="right")
                dep_cum = np.searchsorted(departures, all_unique_times, side="right")
                queue_lengths = arr_cum - dep_cum
                
                max_q_idx = int(np.argmax(queue_lengths))
                max_q_val = queue_lengths[max_q_idx]
                max_q_time = all_unique_times[max_q_idx]
                
                if max_q_val >= 3:
                    y_top = arr_cum[max_q_idx]
                    y_bot = dep_cum[max_q_idx]
                    
                    ax.vlines(x=max_q_time, ymin=y_bot, ymax=y_top,
                              color='forestgreen', linestyle='-', linewidth=2.5, zorder=4)
                    
                    y_center = y_bot + (max_q_val / 2.0)
                    ax.annotate(f"$Q={int(max_q_val)}$",
                                xy=(max_q_time, y_center),
                                xytext=(40, 0),
                                textcoords='offset points',
                                arrowprops=dict(arrowstyle='-', color='forestgreen', lw=1.5),
                                color='forestgreen', fontsize=12, fontweight='bold',
                                ha='left', va='center',
                                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='forestgreen', lw=1, alpha=0.9, zorder=5))
            # --- END NEW LOGIC ---

            all_times = np.concatenate([arrivals, departures])
            t_range = all_times.max() - all_times.min() if len(all_times) > 1 else max(all_times.max(), 1.0)
            ax.set_xlim(left=-t_range * 0.03)
            y_max = max(len(arrivals), len(departures))
            ax.set_ylim(bottom=-0.5, top=y_max + 0.5)
            ax.set_title(f"Cumulative Arrivals/Departures: {title_prefix}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Cumulative Vehicles")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()
            _save_fig_formats(fig, comb_path)
            plt.close(fig)
            saved.append(comb_path)

        return saved


def plot_rejected_vehicle_maps(
    df: pd.DataFrame,
    rejected_groups: dict,
    g_df: pd.DataFrame,
    out_dir: str,
    x_col: str = "Ortho_X",
    y_col: str = "Ortho_Y",
    vehicle_id_col: str = "Vehicle_ID",
    invert_y: bool = True,
) -> list:
    """Plot one trajectory map per rejection reason showing where rejected vehicles drove.

    Parameters
    ----------
    df : DataFrame
        Full trajectory data with x/y columns.
    rejected_groups : dict
        ``{reason_label: [vehicle_id, ...]}`` — one entry per rejection category.
    g_df : DataFrame
        Segmentation CSV (Section, Lane, tlx, tly, …).
    out_dir : str
        Directory to write plots into (created if needed).

    Returns a list of saved file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    # Pre-build segmentation polygons once
    seg_polys = []
    if g_df is not None:
        for _, r in g_df.iterrows():
            pts = [
                (float(r["tlx"]), float(r["tly"])),
                (float(r["trx"]), float(r["try"])),
                (float(r["brx"]), float(r["bry"])),
                (float(r["blx"]), float(r["bly"])),
            ]
            seg_polys.append((str(r["Section"]), pts))

    # Colour palette for distinct rejection categories
    _COLORS = [
        "#E53935", "#1E88E5", "#43A047", "#FB8C00",
        "#8E24AA", "#00ACC1", "#6D4C41",
    ]

    with plt.rc_context(_THESIS_FONT):
        for idx, (reason, vids) in enumerate(rejected_groups.items()):
            if not vids:
                continue

            vid_set = set(int(v) for v in vids)
            sub = df[df[vehicle_id_col].astype(int).isin(vid_set)]

            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 8))

            # Draw segmentation polygons
            for sec_name, pts in seg_polys:
                poly = Polygon(pts, closed=True, fill=False,
                               linewidth=0.8, edgecolor="#999999", zorder=1)
                ax.add_patch(poly)
                cx = sum(p[0] for p in pts) / 4.0
                cy = sum(p[1] for p in pts) / 4.0
                ax.text(cx, cy, sec_name, ha="center", va="center",
                        fontsize=6, color="#AAAAAA", zorder=2)

            # Plot rejected vehicle trajectories
            color = _COLORS[idx % len(_COLORS)]
            n_plotted = 0
            max_plot = 200  # cap per-vehicle trajectories for readability
            for vid in sorted(vid_set):
                vdata = sub[sub[vehicle_id_col].astype(int) == vid]
                if vdata.empty:
                    continue
                xv = vdata[x_col].values.astype(float)
                yv = vdata[y_col].values.astype(float)
                ax.plot(xv, yv, linewidth=0.4, alpha=0.5, color=color, zorder=3)
                n_plotted += 1
                if n_plotted >= max_plot:
                    break

            ax.set_title(f"Rejected: {reason}  ({len(vid_set)} vehicles)", fontsize=12)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.axis("equal")
            if invert_y:
                ax.invert_yaxis()

            fig.tight_layout()
            safe_name = reason.replace(" ", "_").replace("/", "_")
            out_path = os.path.join(out_dir, f"rejected_{safe_name}.png")
            _save_fig_formats(fig, out_path)
            plt.close(fig)
            saved.append(out_path)

    return saved


def plot_general_inputs_outputs_per_time(
    movement_event_times: dict,
    out_dir: str,
) -> str:
    """
    Saves ONE general plot (aggregated across movements) with:
      - general inputs (arrivals)
      - general outputs (departures)
    Uses compressed event-time x-axis so idle gaps are removed.
    """
    def _set_compressed_time_ticks(ax, change_times: np.ndarray, max_ticks: int = 12):
        x_event_idx = np.arange(len(change_times), dtype=int)
        if len(change_times) <= max_ticks:
            tick_idx = x_event_idx
        else:
            tick_idx = np.linspace(0, len(change_times) - 1, max_ticks, dtype=int)
            tick_idx = np.unique(tick_idx)
        tick_labels = [f"{change_times[i]:.1f}" for i in tick_idx]
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right")

    os.makedirs(out_dir, exist_ok=True)
    arrivals_all = []
    departures_all = []
    for ev in movement_event_times.values():
        arr = np.asarray(ev.get("arrivals", []), dtype=float)
        dep = np.asarray(ev.get("departures", []), dtype=float)
        if len(arr) > 0:
            arrivals_all.append(arr)
        if len(dep) > 0:
            departures_all.append(dep)

    arr = np.concatenate(arrivals_all) if arrivals_all else np.array([], dtype=float)
    dep = np.concatenate(departures_all) if departures_all else np.array([], dtype=float)

    out_path = os.path.join(out_dir, "general_inputs_outputs_per_time.png")
    fig, ax = plt.subplots(figsize=(12, 6))

    if len(arr) == 0 and len(dep) == 0:
        ax.text(0.5, 0.5, "No input/output events available", ha="center", va="center", transform=ax.transAxes)
    else:
        if len(arr) > 0 and len(dep) > 0:
            change_times = np.unique(np.concatenate([arr, dep]))
        elif len(arr) > 0:
            change_times = np.unique(arr)
        else:
            change_times = np.unique(dep)

        x_event_idx = np.arange(len(change_times), dtype=int)
        arr_cum = np.searchsorted(np.sort(arr), change_times, side="right")
        dep_cum = np.searchsorted(np.sort(dep), change_times, side="right")

        # Smooth (non-stair) general cumulative lines.
        ax.plot(x_event_idx, arr_cum, linewidth=2.5, label="General approach arrival", color="tab:blue")
        ax.plot(x_event_idx, dep_cum, linewidth=2.5, label="General Stop-line Crossing", color="tab:orange")
        _set_compressed_time_ticks(ax, change_times)
        ax.legend(loc="best")

    ax.set_title("General Approach Arrival / Stop-line Crossing per Time (compressed idle time)")
    ax.set_xlabel("Event Index (tick labels = time in s)")
    ax.set_ylabel("Cumulative Vehicles")
    ax.set_ylim(bottom=-0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig_formats(fig, out_path)
    plt.close(fig)
    return out_path


def plot_turning_arrow_diagram(direction_to_ids: dict, out_path: str, title: str = "Movements"):
    """
    Boxed-arrow diagram showing ALL detected M-intersection movements in ONE image.

    Each cell contains a turning-movement arrow that matches the physical
    road geometry of the M intersection (4-arm cross: N, S, W, E).
    Style mirrors the Q-intersection turning arrow diagram.
    """
    # Fixed 4×4 grid — 14 movements always shown, 2 placeholder slots (None).
    # Rows top→bottom: N arm (4), S arm (4), W arm (3+gap), E arm (3+gap).
    # direction_to_ids is used only to look up vehicle counts.
    _GRID = [
        # row 3 (top): N arm
        'NS', 'NW', 'NE', 'NN',
        # row 2: S arm
        'SN', 'SW', 'SE', 'SS',
        # row 1: W arm  (last slot empty)
        'WN', 'WE', 'WS', None,
        # row 0 (bottom): E arm  (last slot empty)
        'EN', 'EW', 'ES', None,
    ]

    def parse_move_key(k: str):
        s = str(k).strip()
        s = s.replace("→", "_to_").replace("->", "_to_")
        m = re.match(r"^([NSEW])_to_([NSEW])$", s)
        if m:
            return m.group(1), m.group(2)
        m2 = re.match(r"^([NSEW])[_\-]?([NSEW])$", s)
        if m2:
            return m2.group(1), m2.group(2)
        return None

    # Build count lookup from direction_to_ids
    counts: dict = {}
    for k, vids in (direction_to_ids or {}).items():
        parsed = parse_move_key(k)
        if parsed:
            counts[parsed[0] + parsed[1]] = (
                len(vids) if hasattr(vids, "__len__") else 0
            )

    cols, rows = 4, 4

    fig_w = cols * 2.2
    fig_h = rows * 2.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.add_patch(Rectangle((0, 0), cols, rows,
                            fill=False, linewidth=2, edgecolor="black"))
    for x in range(1, cols):
        ax.plot([x, x], [0, rows], linewidth=1, color="black")
    for y in range(1, rows):
        ax.plot([0, cols], [y, y], linewidth=1, color="black")

    def cell_center(col, row):
        return (col + 0.5, row + 0.5)

    def draw_arrow(start, end):
        arr = FancyArrowPatch(start, end, arrowstyle='-|>',
                              mutation_scale=18, linewidth=2, color="black")
        ax.add_patch(arr)

    def draw_cell_label(col, row, text):
        ax.text(col + 0.5, row + 0.10, text,
                ha="center", va="center", fontsize=12, color="black")

    def sign(dir_):
        return {"E": 1, "W": -1, "N": 1, "S": -1}[dir_]

    def draw_movement_in_cell(col, row, origin, dest):
        cx, cy = cell_center(col, row)
        margin = 0.35
        elbow = 0.05

        is_uturn = (origin == dest)

        if is_uturn:
            # Draw a U-shaped arc returning to the same side
            ax.annotate(
                "", xy=(cx + 0.18, cy),
                xytext=(cx - 0.18, cy),
                arrowprops=dict(
                    arrowstyle="-|>", color="black",
                    mutation_scale=16, lw=2,
                    connectionstyle="arc3,rad=-0.75",
                ),
            )
        else:
            if origin in ("E", "W"):
                start = (cx + sign(origin) * margin, cy)
                elbow_pt = (cx + sign(origin) * elbow, cy)
            else:
                start = (cx, cy + sign(origin) * margin)
                elbow_pt = (cx, cy + sign(origin) * elbow)

            if dest in ("E", "W"):
                end = (cx + sign(dest) * margin, elbow_pt[1])
            else:
                end = (elbow_pt[0], cy + sign(dest) * margin)

            opposite = {("E", "W"), ("W", "E"), ("N", "S"), ("S", "N")}
            if (origin, dest) in opposite:
                draw_arrow(start, end)
            else:
                ax.plot([start[0], elbow_pt[0]], [start[1], elbow_pt[1]],
                        linewidth=2, color="black")
                draw_arrow(elbow_pt, end)

        draw_cell_label(col, row, f"{origin}→{dest}")

    for idx, mv in enumerate(_GRID):
        col = idx % cols
        row = rows - 1 - (idx // cols)
        if mv is None:
            continue
        draw_movement_in_cell(col, row, mv[0], mv[1])

    ax.set_xlim(-0.05, cols + 0.05)
    ax.set_ylim(-0.05, rows + 0.05)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13, color="black")

    fig.tight_layout()
    _save_fig_formats(fig, out_path)
    plt.close(fig)

def plot_intersection_layout(out_path: str, title: str = "M Intersection — Layout & Movement Directions"):
    import matplotlib.transforms as mtransforms

    # ── movement colour palette (14 movements) ───────────────────
    C = {
        'SN': '#E53935',  # red
        'SW': '#FB8C00',  # orange
        'SE': '#FDD835',  # yellow
        'SS': '#B71C1C',  # dark red  (U-turn)
        'NS': '#1565C0',  # blue
        'NN': '#0D47A1',  # dark blue (U-turn)
        'NW': '#00897B',  # teal
        'NE': '#2E7D32',  # green
        'WN': '#8E24AA',  # purple
        'WE': '#3949AB',  # indigo
        'WS': '#D81B60',  # pink
        'EN': '#6D4C41',  # brown
        'EW': '#546E7A',  # blue-grey
        'ES': '#AD1457',  # dark pink
    }

    with plt.rc_context(_THESIS_FONT):
        fig, ax = plt.subplots(figsize=(18, 14))

        lw  = 0.80    # base lane width
        arm = 5.0     # arm length beyond intersection edge
        gap = 0.10    # gap between median and stop line

        # ── perfectly flush bounding coordinates ──────────────────
        # Maximum extents to ensure zero gaps in the central cross
        xL = -7 * lw
        xR = +6 * lw
        yT = +5 * lw
        yB = -5 * lw

        xWL = xL - arm - 1.5   
        xER = xR + arm + 1.5   

        rc  = "#E8E8E8"   # road surface
        ic  = "#DADADA"   # intersection body
        med = "#FFD54F"   # median / centre line

        # ── flush road surfaces (arms trace directly to corners) ──
        ax.fill([xL, xR, xR, xL], [yT, yT, yT + arm, yT + arm], color=rc, zorder=0) # North
        ax.fill([xL, xR, xR, xL], [yB - arm, yB - arm, yB, yB], color=rc, zorder=0) # South
        ax.fill([xWL, xL, xL, xWL], [yB, yB, yT, yT], color=rc, zorder=0)           # West
        ax.fill([xR, xER, xER, xR], [yB, yB, yT, yT], color=rc, zorder=0)           # East
        ax.fill([xL, xR, xR, xL], [yT, yT, yB, yB], color=ic, zorder=1)             # Center Core

        # ── median / centre lines (solid yellow) ─────────────────
        ckw = dict(ls="-", color=med, lw=2.5, zorder=3)
        ax.plot([0, 0], [yT + gap, yT + arm - 0.25], **ckw)        # N
        ax.plot([0, 0], [yB - gap, yB - arm + 0.25], **ckw)        # S
        ax.plot([xWL + 0.25, xL - gap], [0, 0], **ckw)             # W
        ax.plot([xR + gap, xER - 0.25], [0, 0], **ckw)             # E

        # ── outer lane boundaries ────────────────────────────────
        bkw = dict(ls="-", color="white", lw=2.2, zorder=3)
        ax.plot([xL, xL], [yT, yT + arm], **bkw)
        ax.plot([xR, xR], [yT, yT + arm], **bkw)
        ax.plot([xL, xL], [yB - arm, yB], **bkw)
        ax.plot([xR, xR], [yB - arm, yB], **bkw)
        ax.plot([xWL, xL], [yT, yT], **bkw)
        ax.plot([xWL, xL], [yB, yB], **bkw)
        ax.plot([xR, xER], [yT, yT], **bkw)
        ax.plot([xR, xER], [yB, yB], **bkw)

        # ── lane dividers (dashed white, scaled for fit) ─────────
        dkw = dict(ls=(0, (4, 3)), color="white", lw=0.9, zorder=3)
        # North
        for i in range(1, 7): ax.plot([-i * lw] * 2, [yT + gap, yT + arm - 0.25], **dkw)
        for i in range(1, 4): ax.plot([i * 1.5 * lw] * 2, [yT + gap, yT + arm - 0.25], **dkw)
        # South
        for i in range(1, 4): ax.plot([-i * 1.75 * lw] * 2, [yB - gap, yB - arm + 0.25], **dkw)
        for i in range(1, 6): ax.plot([i * lw] * 2, [yB - gap, yB - arm + 0.25], **dkw)
        # West
        for i in range(1, 4): ax.plot([xWL + 0.25, xL - gap], [i * 1.25 * lw] * 2, **dkw)
        for i in range(1, 5): ax.plot([xWL + 0.25, xL - gap], [-i * lw] * 2, **dkw)
        # East
        for i in range(1, 5): ax.plot([xR + gap, xER - 0.25], [i * lw] * 2, **dkw)
        for i in range(1, 4): ax.plot([xR + gap, xER - 0.25], [-i * 1.25 * lw] * 2, **dkw)

        # ── group-boundary dividers (solid white, thicker) ───────
        gkw = dict(ls="-", color="white", lw=2.2, zorder=3)
        # North
        ax.plot([-6 * lw] * 2, [yT + gap, yT + arm - 0.25], **gkw)
        ax.plot([-3 * lw] * 2, [yT + gap, yT + arm - 0.25], **gkw)
        ax.plot([+4.5 * lw] * 2, [yT + gap, yT + arm - 0.25], **gkw)
        # South
        ax.plot([-5.25 * lw] * 2, [yB - gap, yB - arm + 0.25], **gkw)
        ax.plot([+2 * lw] * 2, [yB - gap, yB - arm + 0.25], **gkw)
        ax.plot([+5 * lw] * 2, [yB - gap, yB - arm + 0.25], **gkw)
        # West
        ax.plot([xWL + 0.25, xL - gap], [+3.75 * lw] * 2, **gkw) # EW/NW boundary
        ax.plot([xWL + 0.25, xL - gap], [-1 * lw] * 2, **gkw)
        ax.plot([xWL + 0.25, xL - gap], [-4 * lw] * 2, **gkw)
        # East
        ax.plot([xR + gap, xER - 0.25], [+1 * lw] * 2, **gkw)
        ax.plot([xR + gap, xER - 0.25], [+4 * lw] * 2, **gkw)
        ax.plot([xR + gap, xER - 0.25], [-3.75 * lw] * 2, **gkw)

        # ── stop lines ───────────────────────────────────────────
        skw = dict(color="#FFC107", lw=4.0, zorder=4, solid_capstyle="butt")
        ax.plot([gap, xR],   [yT, yT],  **skw)   # N incoming
        ax.plot([gap, xR],   [yB, yB],  **skw)   # S incoming 
        ax.plot([xL, xL],    [-gap, yB], **skw)  # W incoming (South side y < 0)
        ax.plot([xR, xR],    [-gap, yB], **skw)  # E incoming (South side y < 0)

        # ── arm labels ───────────────────────────────────────────
        lbl_s = dict(fontsize=20, fontweight="bold", ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                               edgecolor="black", alpha=0.95, lw=1.5))
        ax.text(0,         yT + arm + 0.9, "NORTH  (N)",  **lbl_s, zorder=6)
        ax.text(0,         yB - arm - 0.9, "SOUTH  (S)",  **lbl_s, zorder=6)
        ax.text(xWL - 2.0, 0,              "WEST  (W)",   **lbl_s, zorder=6)
        ax.text(xER + 2.0, 0,              "EAST  (E)",   **lbl_s, zorder=6)

        # ── lane-group colour bands ───────────────────────────────
        band_alpha = 0.12

        def _vband(x0, x1, ytop, ybot, col):
            ax.fill([x0, x1, x1, x0], [ybot, ybot, ytop, ytop], color=col, alpha=band_alpha, zorder=2)

        def _hband(xl, xr, y0, y1, col):
            ax.fill([xl, xr, xr, xl], [y0, y0, y1, y1], color=col, alpha=band_alpha, zorder=2)

        ytop_N = yT + arm
        ybot_S = yB - arm

        # North arm bands
        _vband(-7*lw, -6*lw,    ytop_N, yT, C['NW'])   
        _vband(-6*lw, -3*lw,    ytop_N, yT, C['NS'])   
        _vband(-3*lw, 0,        ytop_N, yT, C['NE'])   
        _vband(0, 4.5*lw,       ytop_N, yT, C['SN'])   
        _vband(4.5*lw, 6*lw,    ytop_N, yT, C['EN'])   

        # South arm bands
        _vband(-7*lw, -5.25*lw, yB, ybot_S, C['WS'])    
        _vband(-5.25*lw, 0,     yB, ybot_S, C['NS'])   
        _vband(0, 2*lw,         yB, ybot_S, C['SW'])   
        _vband(2*lw, 5*lw,      yB, ybot_S, C['SN'])   
        _vband(5*lw, 6*lw,      yB, ybot_S, C['SE'])   

        # West arm bands
        _hband(xWL, xL, 0, 3.75*lw,     C['EW'])   
        _hband(xWL, xL, 3.75*lw, 5*lw,  C['NW'])   
        _hband(xWL, xL, 0, -1*lw,       C['WN'])   
        _hband(xWL, xL, -1*lw, -4*lw,   C['WE'])   
        _hband(xWL, xL, -4*lw, -5*lw,   C['WS'])   

        # East arm bands
        _hband(xR, xER, 0, 1*lw,        C['ES'])   
        _hband(xR, xER, 1*lw, 4*lw,     C['EW'])   
        _hband(xR, xER, 4*lw, 5*lw,     C['EN'])   
        _hband(xR, xER, 0, -3.75*lw,    C['WE'])   
        _hband(xR, xER, -3.75*lw, -5*lw, C['SE'])   

        # ── GEOMETRIC ARROWS & TEXT PLACEMENT ────────────────────
        def _road_arrow(cx, cy, kind, rot=0, scale=0.30, color="#333333"):
            """Helper to draw physical arrows on the road surface."""
            s = scale
            t = mtransforms.Affine2D().rotate_deg(rot).translate(cx, cy)
            kw = dict(color=color, lw=1.8, zorder=5, clip_on=False, transform=t + ax.transData)
            if "straight" in kind:
                ax.plot([0, 0],             [s * 1.2, -s * .6],  **kw)
                ax.plot([0, -s * .4], [-s * .6, -s * .1],  **kw)
                ax.plot([0,  s * .4], [-s * .6, -s * .1],  **kw)
            if "right" in kind:
                by = s * .3 if "straight" in kind else 0
                ax.plot([0, 0],            [s * 1.2, by],         **kw)
                ax.plot([0, s * .7],       [by, by],               **kw)
                ax.plot([s * .7, s * .35], [by, by + s * .35],    **kw)
                ax.plot([s * .7, s * .35], [by, by - s * .35],    **kw)
            if "left" in kind:
                by = s * .3 if "straight" in kind else 0
                ax.plot([0, 0],              [s * 1.2, by],        **kw)
                ax.plot([0, -s * .7],        [by, by],              **kw)
                ax.plot([-s * .7, -s * .35], [by, by + s * .35],  **kw)
                ax.plot([-s * .7, -s * .35], [by, by - s * .35],  **kw)

        def draw_lane(cx, cy, text, color, kind, rot):
            """Combines drawing the geometric arrow and properly offset text."""
            _road_arrow(cx, cy, kind, rot=rot, color=color)
            
            tx, ty = cx, cy
            offset = 1.6
            if rot == 0:       ty += offset   
            elif rot == 180:   ty -= offset   
            elif rot == 90:    tx -= offset   
            elif rot == -90:   tx += offset   
                
            ax.text(tx, ty, text, color=color, fontsize=11, fontweight="bold", ha="center", va="center", zorder=6)

        ay_N = yT + arm * 0.55
        ay_S = yB - arm * 0.55
        ax_W = xWL + arm * 0.55 + 0.3
        ax_E = xR  + arm * 0.55 + 1.5

        # North arm
        draw_lane(-6.5 * lw, ay_N, "NW (1)", C['NW'], "left", 0)     
        draw_lane(-4.5 * lw, ay_N, "NS (3)", C['NS'], "straight", 0)
        draw_lane(-1.5 * lw, ay_N, "NE (3)", C['NE'], "right", 0)    
        draw_lane(+2.25 * lw, ay_N, "WN+SN (3)", C['SN'], "straight", 180) 
        draw_lane(+5.25 * lw, ay_N, "EN (1)", C['EN'], "left", 180)   

        # South arm
        draw_lane(-6.125 * lw, ay_S, "WS (1)", C['WS'], "left", 0)         # Changed geometric arrow to 'left' for the ↵ shape
        draw_lane(-2.625 * lw, ay_S, "ES+NS (3)", C['NS'], "straight", 0)  
        draw_lane(+1.0 * lw, ay_S, "SW (2)", C['SW'], "right", 180)  
        draw_lane(+3.5 * lw, ay_S, "SN (3)", C['SN'], "straight", 180)
        draw_lane(+5.5 * lw, ay_S, "SE (1)", C['SE'], "left", 180)   

        # West arm
        draw_lane(ax_W, +4.375 * lw, "NW (1)", C['NW'], "left", -90)       
        draw_lane(ax_W, +1.875 * lw, "SW+EW (3)", C['EW'], "straight", -90) 
        draw_lane(ax_W, -0.5 * lw, "WN (1)", C['WN'], "right", 90)   
        draw_lane(ax_W, -2.5 * lw, "WE (3)", C['WE'], "straight", 90)
        draw_lane(ax_W, -4.5 * lw, "WS (1)", C['WS'], "left", 90)    

        # East arm
        draw_lane(ax_E, +4.5 * lw, "EN (1)", C['EN'], "left", -90)    
        draw_lane(ax_E, +2.5 * lw, "EW (3)", C['EW'], "straight", -90)
        draw_lane(ax_E, +0.5 * lw, "ES (1)", C['ES'], "right", -90)   
        draw_lane(ax_E, -1.875 * lw, "NE+WE (3)", C['WE'], "straight", 90) 
        draw_lane(ax_E, -4.375 * lw, "SE (1)", C['SE'], "left", 90)    

        # ── INCOMING / OUTGOING side labels ──────────────────────
        side = dict(fontsize=8, ha="center", va="center", color="#777777", zorder=5, rotation=90)
        ax.text(xL - 0.25, yT + arm / 2, "OUTGOING (7 lanes)", **side)
        ax.text(xR + 0.25, yT + arm / 2, "INCOMING (4 lanes)", **side)
        ax.text(xL - 0.25, yB - arm / 2, "INCOMING (4 lanes)", **side)
        ax.text(xR + 0.25, yB - arm / 2, "OUTGOING (6 lanes)", **side)

        side_h = dict(fontsize=8, ha="center", va="center", color="#777777", zorder=5)
        ax.text((xWL + xL) / 2, yT + 0.20, "INCOMING (4 lanes)", **side_h)
        ax.text((xWL + xL) / 2, yB - 0.20, "OUTGOING (5 lanes)", **side_h)
        ax.text((xR + xER) / 2, yT + 0.20, "OUTGOING (5 lanes)", **side_h)
        ax.text((xR + xER) / 2, yB - 0.20, "INCOMING (4 lanes)", **side_h)

        # ── final axis settings ──────────────────────────────────
        ax.set_xlim(xWL - 3.5, xER + 3.5)
        ax.set_ylim(yB - arm - 2.5, yT + arm + 2.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.tight_layout()
        _save_fig_formats(fig, out_path)
        plt.close(fig)


def plot_g_lane_section_map(
    g_df: pd.DataFrame,
    out_path: str,
    label_mode: str = "lane",
    df_points: Optional[pd.DataFrame] = None,
    x_col: str = "Ortho_X",
    y_col: str = "Ortho_Y",
    allowed_vehicle_ids=None,
    max_points: int = 20000,
    invert_y: bool = True,
):
    """
    Draw polygons from a segmentation CSV and label them.

    Segmentation CSV columns:
      Section, Lane, tlx,tly, blx,bly, brx,bry, trx,try

    label_mode:
      - "lane": label each polygon with its Lane number
      - "section": label each Section once (at section centroid)

    If df_points is provided, overlays (downsampled) Ortho_X/Ortho_Y points
    to visually verify alignment with the polygons.
    """

    required_cols = {"Section", "Lane", "tlx", "tly", "blx", "bly", "brx", "bry", "trx", "try"}
    missing = required_cols - set(g_df.columns)
    if missing:
        raise ValueError(f"Segmentation CSV is missing columns: {missing}")

    fig, ax = plt.subplots(figsize=(10, 8))

    # --- draw all lane polygons ---
    for _, r in g_df.iterrows():
        pts = [
            (float(r["tlx"]), float(r["tly"])),
            (float(r["trx"]), float(r["try"])),
            (float(r["brx"]), float(r["bry"])),
            (float(r["blx"]), float(r["bly"])),
        ]
        poly = Polygon(pts, closed=True, fill=False, linewidth=1)
        ax.add_patch(poly)

        if label_mode == "lane":
            cx = sum(p[0] for p in pts) / 4.0
            cy = sum(p[1] for p in pts) / 4.0
            ax.text(cx, cy, str(int(r["Lane"])), ha="center", va="center", fontsize=9, fontweight="bold")

    # --- label each section once ---
    if label_mode == "section":
        for sec, g in g_df.groupby("Section"):
            xs = []
            ys = []
            for _, r in g.iterrows():
                xs += [r["tlx"], r["trx"], r["brx"], r["blx"]]
                ys += [r["tly"], r["try"], r["bry"], r["bly"]]
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            ax.text(cx, cy, str(sec), ha="center", va="center", fontsize=10, fontweight="bold")

    # --- optional overlay of vehicle points to validate geometry alignment ---
    if df_points is not None:
        d = df_points.copy()

        if allowed_vehicle_ids is not None:
            allowed_set = set(allowed_vehicle_ids)
            d = d[d["Vehicle_ID"].isin(allowed_set)]

        for c in [x_col, y_col]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=[x_col, y_col])

        if len(d) > max_points:
            d = d.sample(n=max_points, random_state=0)

        ax.scatter(d[x_col], d[y_col], s=2, alpha=0.25, label="vehicle points")

    ax.set_title(f"Segmentation map labeled by: {label_mode}")
    ax.set_xlabel("Ortho_X")
    ax.set_ylabel("Ortho_Y")
    ax.axis("equal")

    if invert_y:
        ax.invert_yaxis()

    if df_points is not None:
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_g_maps_into_run_dir(
    run_dir: str,
    g_csv_path: str = "segmentation.csv",
    df_points: Optional[pd.DataFrame] = None,
    allowed_vehicle_ids=None,
):
    """
    Convenience wrapper: creates TWO maps inside run_dir:
      1) lane labels on each polygon
      2) section labels (one per section)
    """
    g_df = pd.read_csv(g_csv_path)

    out_lane = os.path.join(run_dir, "segmentation_map_label_LANE.png")
    out_sec  = os.path.join(run_dir, "segmentation_map_label_SECTION.png")

    plot_g_lane_section_map(
        g_df,
        out_path=out_lane,
        label_mode="lane",
        df_points=df_points,                   # overlay points (optional but useful)
        allowed_vehicle_ids=allowed_vehicle_ids
    )
    plot_g_lane_section_map(
        g_df,
        out_path=out_sec,
        label_mode="section",
        df_points=df_points,                   # overlay points (optional but useful)
        allowed_vehicle_ids=allowed_vehicle_ids
    )

    return out_lane, out_sec


# ------------------------------------------------------------------
# Headway distribution plots (ECDF + Exponential fit) per lane
# ------------------------------------------------------------------

def plot_headway_distributions(
    lane_event_times: dict,
    out_dir: str,
) -> list:
    """
    For each (movement, lane) pair, produce two headway distribution plots:

    1. **Arrival headways** – ECDF of inter-arrival times with fitted
       exponential CDF overlay.  Shape: logarithmic rise (1 - e^{-λx}).
    2. **Departure headways** – PDF histogram of inter-departure times
       with fitted exponential PDF overlay.  Shape: decaying e^{-λx}.

    Both use only the Poisson/Exponential distribution.

    Parameters
    ----------
    lane_event_times : dict
        {(movement, lane): {"arrivals": np.ndarray, "departures": np.ndarray}}
    out_dir : str
        Root directory for output.  Sub-folders per movement/lane are created.

    Returns
    -------
    list of saved file paths.
    """
    from scipy.stats import expon, kstest

    _MOVEMENT_DESCRIPTIONS = {
        'SN': 'South_to_North', 'SW': 'South_to_West',
        'SE': 'South_to_East',  'SS': 'South_to_South',
        'NS': 'North_to_South', 'NN': 'North_to_North',
        'NW': 'North_to_West',  'NE': 'North_to_East',
        'WN': 'West_to_North',  'WE': 'West_to_East',
        'WS': 'West_to_South',
        'EN': 'East_to_North',  'EW': 'East_to_West',
        'ES': 'East_to_South',
    }

    os.makedirs(out_dir, exist_ok=True)
    saved = []

    sorted_keys = sorted(
        [k for k in lane_event_times.keys()
         if k[0] != "UNASSIGNED" and k[1] != "unknown"],
        key=lambda k: (str(k[0]), str(k[1])),
    )

    with plt.rc_context(_THESIS_FONT):
        for movement, lane in sorted_keys:
            ev = lane_event_times[(movement, lane)]
            arrivals = np.sort(np.asarray(ev.get("arrivals", []), dtype=float))
            departures = np.sort(np.asarray(ev.get("departures", []), dtype=float))

            # Pre-compute headways to decide if this lane is worth plotting
            arr_headways = None
            dep_headways = None

            if len(arrivals) >= 3:
                _ah = np.diff(arrivals)
                _ah = _ah[_ah > 0]
                if len(_ah) >= 2:
                    arr_headways = _ah

            if len(departures) >= 3:
                _dh = np.diff(departures)
                _dh = _dh[_dh > 0]
                if len(_dh) >= 2:
                    dep_headways = _dh

            # Skip this lane entirely — no folder created
            if arr_headways is None and dep_headways is None:
                continue

            mov_desc = _MOVEMENT_DESCRIPTIONS.get(movement, movement)
            lane_dir = os.path.join(out_dir, f"{mov_desc}_lane{lane}")
            os.makedirs(lane_dir, exist_ok=True)

            # --- Arrival headways (ECDF + exponential CDF) ---
            if arr_headways is not None:
                path = _plot_headway_ecdf(
                    headways=arr_headways,
                    out_path=os.path.join(lane_dir, "arrival_headway_ecdf.png"),
                    title=f"Arrival | {movement} Lane {lane} "
                          f"($n$={len(arr_headways)})",
                    xlabel="Inter-arrival time (s)",
                )
                if path:
                    saved.append(path)

            # --- Departure headways (PDF histogram + exponential PDF) ---
            if dep_headways is not None:
                path = _plot_headway_pdf(
                    headways=dep_headways,
                    out_path=os.path.join(lane_dir, "departure_headway_pdf.png"),
                    title=f"Departure | {movement} Lane {lane} "
                          f"($n$={len(dep_headways)})",
                    xlabel="Inter-departure time (s)",
                )
                if path:
                    saved.append(path)

    print(f"  Headway distributions: {len(saved)} plots saved")
    return saved


def _plot_headway_ecdf(
    headways: np.ndarray,
    out_path: str,
    title: str,
    xlabel: str = "Inter-event time (s)",
) -> str:
    """ECDF of headways with exponential CDF fit (logarithmic rise shape).

    Matches the attached reference image style:
      - Solid blue line for empirical ECDF
      - Dashed colored line for exponential CDF fit
      - KS statistic in legend
    """
    from scipy.stats import expon, kstest

    h = np.sort(headways)
    n = len(h)

    # Empirical CDF: F(x_i) = i / n
    ecdf_y = np.arange(1, n + 1) / n

    # Fit exponential distribution (MLE: λ = 1/mean)
    loc, scale = expon.fit(h, floc=0)  # force loc=0
    lam = 1.0 / scale

    # KS test
    ks_stat, ks_p = kstest(h, "expon", args=(loc, scale))

    # Theoretical CDF on a smooth x grid
    x_fit = np.linspace(0, h[-1] * 1.1, 300)
    cdf_fit = expon.cdf(x_fit, loc=loc, scale=scale)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(h, ecdf_y, linewidth=2.5, color="tab:blue", label="Empirical ECDF", zorder=3)
    ax.plot(x_fit, cdf_fit, linewidth=2.0, linestyle="--", color="tab:orange",
            label=rf"Exponential ($\lambda$={lam:.3f}) | KS={ks_stat:.3f}", zorder=2)

    # Light fill under ECDF for visual depth
    ax.fill_between(h, 0, ecdf_y, alpha=0.08, color="tab:blue")

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("ECDF", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    _save_fig_formats(fig, out_path)
    plt.close(fig)
    return out_path


def _plot_headway_pdf(
    headways: np.ndarray,
    out_path: str,
    title: str,
    xlabel: str = "Inter-event time (s)",
) -> str:
    """Histogram of headways with exponential PDF fit (e^{-λx} decay shape).

    Produces the classic decaying exponential look the user expects
    for departure headways.
    """
    from scipy.stats import expon, kstest

    h = np.sort(headways)

    # Fit
    loc, scale = expon.fit(h, floc=0)
    lam = 1.0 / scale
    ks_stat, ks_p = kstest(h, "expon", args=(loc, scale))

    x_fit = np.linspace(0, h[-1] * 1.1, 300)
    pdf_fit = expon.pdf(x_fit, loc=loc, scale=scale)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Histogram (density-normalized so it overlays with the PDF)
    n_bins = min(max(10, len(h) // 5), 40)
    ax.hist(h, bins=n_bins, density=True, alpha=0.45, color="tab:blue",
            edgecolor="white", linewidth=0.5, label="Observed headways", zorder=2)

    ax.plot(x_fit, pdf_fit, linewidth=2.5, color="tab:orange", linestyle="-",
            label=rf"Exponential PDF ($\lambda$={lam:.3f}) | KS={ks_stat:.3f}", zorder=3)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    _save_fig_formats(fig, out_path)
    plt.close(fig)
    return out_path


def plot_recording_timeline(
    df: pd.DataFrame,
    out_path: str,
    local_time_col: str = "Local_Time",
    vehicle_id_col: str = "Vehicle_ID",
    gap_threshold_minutes: float = 1.5,
    intra_gap_threshold_seconds: int = 30,
    title: str = "Recording Timeline & Gaps",
):
    """Draw a timeline showing recording segments, gaps, and vehicle counts.

    Each continuous recording block is drawn as a filled rectangle.
    Inside each rectangle, vehicle dots are scattered to give a sense
    of density.  Gaps between blocks are annotated with their duration.

    Parameters
    ----------
    df : DataFrame with at least *local_time_col* and *vehicle_id_col*.
    out_path : Where to save the figure.
    gap_threshold_minutes : Minimum gap (in whole minutes) between
        consecutive minutes with data to consider them separate segments.
    intra_gap_threshold_seconds : Minimum gap (seconds) inside a segment
        to be reported as an internal gap in the console output.
    """
    import matplotlib.patches as mpatches
    from matplotlib.dates import DateFormatter, MinuteLocator
    from datetime import datetime, timedelta

    # --- 1. Parse Local_Time to clock times (HH:MM level) ----------------
    raw = df[local_time_col].dropna().astype(str)

    # Minute-level parser (used for segment detection & diagram)
    def _parse_clock(s):
        """Return datetime with clock time from *s* at minute precision."""
        try:
            parts = s.strip().split(":")
            h = int(parts[0])
            m = int(parts[1])
            return datetime(2000, 1, 1, h, m)
        except Exception:
            return None

    # Second-level parser (used for intra-segment gap analysis)
    def _parse_clock_s(s):
        """Return datetime with clock time from *s* at second precision."""
        try:
            parts = s.strip().split(":")
            h = int(parts[0])
            m = int(parts[1])
            sec = int(float(parts[2])) if len(parts) >= 3 else 0
            return datetime(2000, 1, 1, h, m, sec)
        except Exception:
            return None

    clock_series = raw.map(_parse_clock)
    valid_mask = clock_series.notna()
    if valid_mask.sum() == 0:
        print("[WARN] Could not parse any Local_Time values for recording timeline")
        return out_path

    df_work = df.loc[valid_mask, [vehicle_id_col]].copy()
    df_work["clock_minute"] = clock_series[valid_mask]

    # Second-level series for intra-segment analysis
    clock_series_s = raw.map(_parse_clock_s)
    valid_mask_s = clock_series_s.notna()
    df_work_s = df.loc[valid_mask_s, [vehicle_id_col]].copy()
    df_work_s["clock_second"] = clock_series_s[valid_mask_s]

    # --- 2. Detect recording segments ------------------------------------
    # Get sorted unique minutes
    unique_minutes = sorted(df_work["clock_minute"].unique())

    segments = []  # list of (start_dt, end_dt, set_of_vehicle_ids)
    seg_start = unique_minutes[0]
    seg_end = unique_minutes[0]
    seg_vids = set(df_work.loc[df_work["clock_minute"] == seg_start, vehicle_id_col])

    for dt in unique_minutes[1:]:
        delta_min = (dt - seg_end).total_seconds() / 60.0
        if delta_min <= gap_threshold_minutes:
            # Continue current segment
            seg_end = dt
            seg_vids |= set(df_work.loc[df_work["clock_minute"] == dt, vehicle_id_col])
        else:
            # Close current segment, start new one
            segments.append((seg_start, seg_end, seg_vids))
            seg_start = dt
            seg_end = dt
            seg_vids = set(df_work.loc[df_work["clock_minute"] == dt, vehicle_id_col])
    segments.append((seg_start, seg_end, seg_vids))

    # --- 2b. Intra-segment gap analysis (console report) -----------------
    print("\n--- Recording Timestamp Gap Analysis ---")
    for seg_idx, (seg_start_dt, seg_end_dt, _) in enumerate(segments, 1):
        start_str = seg_start_dt.strftime("%H:%M")
        end_str = seg_end_dt.strftime("%H:%M")

        # Collect unique seconds within this segment window (add 1 min buffer on end)
        window_end = seg_end_dt + timedelta(minutes=1)
        mask_seg = (
            (df_work_s["clock_second"] >= seg_start_dt) &
            (df_work_s["clock_second"] < window_end)
        )
        seg_seconds = sorted(df_work_s.loc[mask_seg, "clock_second"].unique())

        if len(seg_seconds) < 2:
            print(f"  Timestamp {seg_idx}: {start_str} - {end_str}  (insufficient second-level data)")
            continue

        intra_gaps = []
        for i in range(len(seg_seconds) - 1):
            delta_s = (seg_seconds[i + 1] - seg_seconds[i]).total_seconds()
            if delta_s > intra_gap_threshold_seconds:
                g_start = seg_seconds[i].strftime("%H:%M:%S")
                g_end = seg_seconds[i + 1].strftime("%H:%M:%S")
                intra_gaps.append(f"{g_start}–{g_end} ({int(delta_s)}s)")

        if intra_gaps:
            print(f"  Timestamp {seg_idx}: {start_str} - {end_str}  gaps in: {', '.join(intra_gaps)}")
        else:
            print(f"  Timestamp {seg_idx}: {start_str} - {end_str}  no gaps")
    print("--- End Gap Analysis ---\n")

    # --- 3. Draw the diagram ---------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))

    block_color = "#4A90D9"
    block_alpha = 0.25
    dot_color = "#2C5F8A"
    gap_color = "#D94A4A"
    y_block_lo = 0.0
    y_block_hi = 1.0
    block_height = y_block_hi - y_block_lo

    for i, (s_start, s_end, vids) in enumerate(segments):
        # Rectangle spans from start-0.5min to end+0.5min for visual width
        rect_left = s_start - timedelta(minutes=0.5)
        rect_right = s_end + timedelta(minutes=0.5)
        width_minutes = (rect_right - rect_left).total_seconds() / 60.0

        # Draw rectangle
        rect = Rectangle(
            (rect_left, y_block_lo),
            rect_right - rect_left,  # matplotlib handles timedelta for date axes
            block_height,
            linewidth=2,
            edgecolor=block_color,
            facecolor=block_color,
            alpha=block_alpha,
            zorder=2,
        )
        ax.add_patch(rect)

        # Scatter vehicle dots inside the block
        n_vids = len(vids)
        if n_vids > 0:
            # Distribute dots in a grid-like pattern inside the rectangle
            n_cols = max(1, int(np.ceil(np.sqrt(n_vids * width_minutes / block_height))))
            n_rows = max(1, int(np.ceil(n_vids / n_cols)))

            dot_xs = []
            dot_ys = []
            margin_x = timedelta(minutes=0.3)
            margin_y = 0.06
            for idx in range(n_vids):
                col = idx % n_cols
                row = idx // n_cols
                # Spread within block bounds
                frac_x = (col + 0.5) / n_cols
                frac_y = (row + 0.5) / max(n_rows, 1)
                x_dt = rect_left + margin_x + timedelta(
                    seconds=(rect_right - rect_left - 2 * margin_x).total_seconds() * frac_x
                )
                y_val = y_block_lo + margin_y + (block_height - 2 * margin_y) * frac_y
                dot_xs.append(x_dt)
                dot_ys.append(y_val)

            ax.scatter(dot_xs, dot_ys, s=8, color=dot_color, alpha=0.6, zorder=3, edgecolors="none")

        # Label vehicle count
        mid_dt = s_start + (s_end - s_start) / 2
        ax.text(
            mid_dt, y_block_hi + 0.06, f"{n_vids} vehicles",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color=block_color,
        )

        # Time labels below block
        s_label = s_start.strftime("%H:%M")
        e_label = s_end.strftime("%H:%M")
        duration_min = (s_end - s_start).total_seconds() / 60.0 + 1  # inclusive
        ax.text(
            rect_left + timedelta(minutes=0.3), y_block_lo - 0.06,
            s_label, ha="left", va="top", fontsize=9, color="#333",
        )
        ax.text(
            rect_right - timedelta(minutes=0.3), y_block_lo - 0.06,
            e_label, ha="right", va="top", fontsize=9, color="#333",
        )
        ax.text(
            mid_dt, y_block_lo - 0.14,
            f"({int(duration_min)} min)", ha="center", va="top", fontsize=8, color="#666",
        )

    # --- 4. Annotate gaps ------------------------------------------------
    for i in range(len(segments) - 1):
        _, end_prev, _ = segments[i]
        start_next, _, _ = segments[i + 1]
        gap_start = end_prev + timedelta(minutes=0.5)
        gap_end = start_next - timedelta(minutes=0.5)
        gap_mid = end_prev + (start_next - end_prev) / 2
        gap_duration = (start_next - end_prev).total_seconds() / 60.0 - 1  # exclude boundary minutes

        # Draw gap annotation
        ax.annotate(
            "", xy=(gap_end, 0.5), xytext=(gap_start, 0.5),
            arrowprops=dict(arrowstyle="<->", color=gap_color, lw=1.5),
            zorder=4,
        )
        ax.text(
            gap_mid, 0.58,
            f"GAP\n~{int(round(gap_duration))} min",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=gap_color,
        )

    # --- 5. Style the axes -----------------------------------------------
    ax.set_ylim(-0.3, 1.35)
    ax.set_yticks([])
    ax.set_ylabel("")

    # X-axis as clock times
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(MinuteLocator(interval=2))
    ax.tick_params(axis="x", rotation=0, labelsize=9)

    # Expand x limits slightly
    if segments:
        x_lo = segments[0][0] - timedelta(minutes=2)
        x_hi = segments[-1][1] + timedelta(minutes=2)
        ax.set_xlim(x_lo, x_hi)

    ax.set_xlabel("Local Time (HH:MM)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    # Total summary
    total_vids = df[vehicle_id_col].nunique()
    total_rec_min = sum((s[1] - s[0]).total_seconds() / 60.0 + 1 for s in segments)
    total_gap_min = sum(
        (segments[i + 1][0] - segments[i][1]).total_seconds() / 60.0 - 1
        for i in range(len(segments) - 1)
    )
    summary = (
        f"Total: {total_vids} vehicles | "
        f"{len(segments)} recording segments ({int(total_rec_min)} min) | "
        f"{len(segments) - 1} gaps (~{int(round(total_gap_min))} min)"
    )
    ax.text(
        0.5, -0.18, summary, transform=ax.transAxes,
        ha="center", va="top", fontsize=10, color="#444",
    )

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#999")

    fig.tight_layout()
    _save_fig_formats(fig, out_path)
    plt.close(fig)
    print(f"  Recording timeline saved: {out_path}")
    return out_path
