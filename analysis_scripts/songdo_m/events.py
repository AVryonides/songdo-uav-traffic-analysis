# events.py
import numpy as np
import pandas as pd
from typing import Tuple

from .m_rules import M_ARM_APPROACH, MOVEMENTS


def _in_green(t: float, movement: str, green_intervals_by_movement: dict) -> bool:
    """Return True if time *t* falls within any green interval for *movement*."""
    for t0, t1 in green_intervals_by_movement.get(movement, []):
        if t0 <= t <= t1:
            return True
    return False


def _base_section(section_value) -> str:
    """Strip lane suffix from values like '3_4_2' -> '3_4'."""
    if section_value is None:
        return ""
    parts = str(section_value).strip().split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else str(section_value).strip()


def _compute_vehicle_arrival_departure_times(
    df: pd.DataFrame,
    allowed_vehicle_ids=None,
    vehicle_id_col: str = "Vehicle_ID",
    time_col: str = "time_s",
    speed_col: str = "Vehicle_Speed",
    section_col: str = "Road_Section",
    x_col: str = "Ortho_X",
    y_col: str = "Ortho_Y",
    g_df: pd.DataFrame = None,
    movement_to_ids: dict = None,
    stop_speed_kmh: float = 10.0,
    green_intervals_by_movement: dict = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Upgraded arrival/departure computation:
      - Arrival  = first time the vehicle is observed in its origin approach.
      - Departure = time the vehicle crosses the stop line
                    (transitions from approach section to intersection).
                    Only recorded when the crossing falls within a green phase
                    (if green_intervals_by_movement is provided).
    """
    # Movement -> approach (origin) sections for M intersection.
    # M uses compass movement names (NS, SW, etc.), not R-style 1->2 keys.
    arm_origin_sections = {
        mov: set(M_ARM_APPROACH[mov[0]])
        for mov in MOVEMENTS
        if mov and mov[0] in M_ARM_APPROACH
    }

    cols = [vehicle_id_col, time_col]
    if speed_col in df.columns:
        cols.append(speed_col)
    if section_col in df.columns:
        cols.append(section_col)
    if x_col in df.columns:
        cols.append(x_col)
    if y_col in df.columns:
        cols.append(y_col)

    d = df[cols].copy()
    d[vehicle_id_col] = pd.to_numeric(d[vehicle_id_col], errors="coerce")
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    if speed_col in d.columns:
        d[speed_col] = pd.to_numeric(d[speed_col], errors="coerce")
    d = d.dropna(subset=[vehicle_id_col, time_col])
    d[vehicle_id_col] = d[vehicle_id_col].astype(int)

    if allowed_vehicle_ids is not None:
        allowed_set = set(map(int, allowed_vehicle_ids))
        d = d[d[vehicle_id_col].isin(allowed_set)]

    # Build vid -> movement lookup
    vid_to_movement = {}
    if movement_to_ids:
        for mov, vids in movement_to_ids.items():
            for v in vids:
                vid_to_movement[int(v)] = mov

    # Build polygon lookup for section detection (fallback when Road_Section missing)
    poly_lookup = None
    if g_df is not None and section_col not in d.columns:
        import matplotlib.path as mplPath
        poly_lookup = []
        for _, row in g_df.iterrows():
            pts = [
                (row['tlx'], row['tly']),
                (row['trx'], row['try']),
                (row['brx'], row['bry']),
                (row['blx'], row['bly'])
            ]
            poly_lookup.append((str(row['Section']), mplPath.Path(pts)))

    arrival_times = {}
    departure_times = {}

    for vid, g in d.groupby(vehicle_id_col):
        g = g.sort_values(time_col)
        vid_int = int(vid)
        mov = vid_to_movement.get(vid_int)

        # --- Arrival/departure: approach entry and stop-line crossing time ---
        if mov and mov in arm_origin_sections:
            origin_set = arm_origin_sections[mov]

            if section_col in g.columns:
                sections = np.array([_base_section(v) for v in g[section_col].values])
            elif poly_lookup is not None and x_col in g.columns and y_col in g.columns:
                xs = g[x_col].values.astype(float)
                ys = g[y_col].values.astype(float)
                sections = []
                for px, py in zip(xs, ys):
                    found = None
                    for sid, path in poly_lookup:
                        if path.contains_point((px, py)):
                            found = sid
                            break
                    sections.append(found if found else "")
                sections = np.array(sections)
            else:
                continue

            times = g[time_col].values.astype(float)

            in_origin = np.array([s in origin_set for s in sections])
            origin_indices = np.where(in_origin)[0]
            if len(origin_indices) > 0:
                arrival_times[vid_int] = float(times[origin_indices[0]])
                last_origin_idx = origin_indices[-1]
                if last_origin_idx < len(times) - 1:
                    t_last_in = times[last_origin_idx]
                    t_first_out = times[last_origin_idx + 1]
                    t_dep = float(0.5 * (t_last_in + t_first_out))
                    if (green_intervals_by_movement is None
                            or _in_green(t_dep, mov, green_intervals_by_movement)):
                        departure_times[vid_int] = t_dep

    arrivals = pd.Series(arrival_times, dtype=float)
    arrivals.index.name = vehicle_id_col
    departures = pd.Series(departure_times, dtype=float)
    departures.index.name = vehicle_id_col
    return arrivals, departures


def compute_movement_arrival_departure_times(
    df: pd.DataFrame,
    movement_to_ids: dict,
    vehicle_id_col: str = "Vehicle_ID",
    time_col: str = "time_s",
    speed_col: str = "Vehicle_Speed",
    section_col: str = "Road_Section",
    x_col: str = "Ortho_X",
    y_col: str = "Ortho_Y",
    g_df: pd.DataFrame = None,
    stop_speed_kmh: float = 10.0,
    green_intervals_by_movement: dict = None,
) -> dict:
    """
    Returns:
      {movement: {"arrivals": np.ndarray, "departures": np.ndarray}}
    """
    all_ids = sorted({int(v) for vids in movement_to_ids.values() for v in vids})
    arrivals_by_vid, departures_by_vid = _compute_vehicle_arrival_departure_times(
        df,
        allowed_vehicle_ids=all_ids,
        vehicle_id_col=vehicle_id_col,
        time_col=time_col,
        speed_col=speed_col,
        section_col=section_col,
        x_col=x_col,
        y_col=y_col,
        g_df=g_df,
        movement_to_ids=movement_to_ids,
        stop_speed_kmh=stop_speed_kmh,
        green_intervals_by_movement=green_intervals_by_movement,
    )

    out = {}
    for movement, vids in movement_to_ids.items():
        vids_i = [int(v) for v in vids]
        arr = arrivals_by_vid.reindex(vids_i).dropna().to_numpy(dtype=float)
        dep = departures_by_vid.reindex(vids_i).dropna().to_numpy(dtype=float)
        out[movement] = {
            "arrivals": np.sort(arr),
            "departures": np.sort(dep),
        }

    return out


def compute_lane_arrival_departure_times(
    df: pd.DataFrame,
    movement_to_ids: dict,
    vehicle_id_col: str = "Vehicle_ID",
    time_col: str = "time_s",
    speed_col: str = "Vehicle_Speed",
    section_col: str = "Road_Section",
    lane_col: str = "Lane_Number",
    x_col: str = "Ortho_X",
    y_col: str = "Ortho_Y",
    g_df: pd.DataFrame = None,
    stop_speed_kmh: float = 10.0,
    green_intervals_by_movement: dict = None,
) -> dict:
    """
    Returns:
      {(movement, lane): {"arrivals": np.ndarray, "departures": np.ndarray}}
    """
    all_ids = sorted({int(v) for vids in movement_to_ids.values() for v in vids})
    arrivals_by_vid, departures_by_vid = _compute_vehicle_arrival_departure_times(
        df,
        allowed_vehicle_ids=all_ids,
        vehicle_id_col=vehicle_id_col,
        time_col=time_col,
        speed_col=speed_col,
        section_col=section_col,
        x_col=x_col,
        y_col=y_col,
        g_df=g_df,
        movement_to_ids=movement_to_ids,
        stop_speed_kmh=stop_speed_kmh,
        green_intervals_by_movement=green_intervals_by_movement,
    )

    d = df[[vehicle_id_col, lane_col]].copy()
    d[vehicle_id_col] = pd.to_numeric(d[vehicle_id_col], errors="coerce")
    d[lane_col] = pd.to_numeric(d[lane_col], errors="coerce")
    d = d.dropna(subset=[vehicle_id_col])
    d[vehicle_id_col] = d[vehicle_id_col].astype(int)
    d = d[d[vehicle_id_col].isin(set(all_ids))]
    per_vid_lane = d.groupby(vehicle_id_col)[lane_col].agg(
        lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan
    )

    out = {}
    for movement, vids in movement_to_ids.items():
        vids_i = [int(v) for v in vids]
        for vid in vids_i:
            lane_val = per_vid_lane.get(vid)
            if pd.isna(lane_val):
                continue
            else:
                lane_label = int(lane_val)
            key = (movement, lane_label)
            if key not in out:
                out[key] = {"arrival_vids": [], "departure_vids": []}
            if vid in arrivals_by_vid.index:
                out[key]["arrival_vids"].append(vid)
            if vid in departures_by_vid.index:
                out[key]["departure_vids"].append(vid)

    result = {}
    for key, info in out.items():
        arr_series = arrivals_by_vid.reindex(info["arrival_vids"]).dropna()
        dep_series = departures_by_vid.reindex(info["departure_vids"]).dropna()
        arr_detail = sorted(zip(arr_series.index.astype(int), arr_series.values.astype(float)), key=lambda x: x[1])
        dep_detail = sorted(zip(dep_series.index.astype(int), dep_series.values.astype(float)), key=lambda x: x[1])
        result[key] = {
            "arrivals": np.sort(arr_series.to_numpy(dtype=float)),
            "departures": np.sort(dep_series.to_numpy(dtype=float)),
            "arrival_details": arr_detail,
            "departure_details": dep_detail,
        }

    return result
