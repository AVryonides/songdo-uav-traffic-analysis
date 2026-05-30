# trajectory_recovery.py
"""Trajectory recovery module.

Recovers vehicles that were rejected by quality filters in the space-time
diagram by completing/smoothing their trajectories using interpolation,
extrapolation, and Kalman-style smoothing.

Recovery strategies per rejection reason:

  no_downstream_departure
      Vehicle doesn't progress far enough past the stopline.
      → Extend trajectory forward using last-known speed or a
        physically-reasonable departure speed.

  no_upstream_approach
      Vehicle doesn't start far enough before the stopline.
      → Extend trajectory backward using first-known speed or a
        typical approach speed.

  too_many_breaks
      Noisy / gappy tracking data.
      → Fill gaps via linear interpolation, smooth with rolling average,
        then re-check quality.

  too_short_in_time
      Brief trajectory.
      → Extend in both directions, then re-check time span.

  silently_skipped_unk_no_origin
      Couldn't map to origin arm.
      → Re-assign movement from Road_Section column data.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DEPARTURE_SPEED_MPS = 6.0     # ~22 km/h — typical departure speed
_DEFAULT_APPROACH_SPEED_MPS = 8.0      # ~29 km/h — typical approach speed
_MIN_SPEED_FOR_EXTRAP_MPS = 0.5        # below this, use default speed
_EXTRAP_DT = 0.1                       # synthetic point interval (seconds)


# ---------------------------------------------------------------------------
# Forward extension  (for no_downstream_departure)
# ---------------------------------------------------------------------------

def extend_trajectory_forward(
    times: np.ndarray,
    y_vals: np.ndarray,
    target_y_pos: float,
    max_extend_s: float = 20.0,
    n_speed_pts: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extend a trajectory forward in the space-time (t, y) domain.

    Uses a rolling-average speed estimate from the last *n_speed_pts* points.
    If the vehicle is nearly stopped, falls back to a physically-reasonable
    departure speed (constant-velocity model).

    Parameters
    ----------
    times : 1-D array of timestamps (seconds).
    y_vals : 1-D array of distance-from-stopline (metres).
    target_y_pos : The minimum y_max the trajectory must reach.
    max_extend_s : Maximum time to extrapolate (safety cap).
    n_speed_pts : Number of tail points used for speed estimation.

    Returns
    -------
    (times_extended, y_extended) — the full trajectory including synthetic
    tail points.  If no extension is needed, returns the inputs unchanged.
    """
    if len(times) < 2:
        return times, y_vals

    current_y_max = float(np.max(y_vals))
    if current_y_max >= target_y_pos:
        return times, y_vals                       # already passes

    # Estimate exit speed from last N points (rolling average of dy/dt)
    tail = min(n_speed_pts, len(times) - 1)
    dt_tail = np.diff(times[-tail - 1:])
    dy_tail = np.diff(y_vals[-tail - 1:])

    # Guard against zero dt
    valid = dt_tail > 1e-6
    if np.any(valid):
        speeds = dy_tail[valid] / dt_tail[valid]
        avg_speed = float(np.mean(speeds))
    else:
        avg_speed = 0.0

    # If speed is too low (vehicle stopped/queued), use a default departure
    # speed — the vehicle *will* eventually depart.
    if abs(avg_speed) < _MIN_SPEED_FOR_EXTRAP_MPS:
        avg_speed = _DEFAULT_DEPARTURE_SPEED_MPS

    # Make sure speed is positive (moving forward past stopline)
    avg_speed = abs(avg_speed)

    # Generate synthetic extension points
    t_last = float(times[-1])
    y_last = float(y_vals[-1])

    target_with_margin = target_y_pos + 2.0        # small margin

    ext_times = []
    ext_y = []
    t = t_last
    y = y_last
    while y < target_with_margin and (t - t_last) < max_extend_s:
        t += _EXTRAP_DT
        y += avg_speed * _EXTRAP_DT
        ext_times.append(t)
        ext_y.append(y)

    if not ext_times:
        return times, y_vals

    return (
        np.concatenate([times, np.array(ext_times)]),
        np.concatenate([y_vals, np.array(ext_y)]),
    )


# ---------------------------------------------------------------------------
# Backward extension  (for no_upstream_approach)
# ---------------------------------------------------------------------------

def extend_trajectory_backward(
    times: np.ndarray,
    y_vals: np.ndarray,
    target_y_neg: float,
    max_extend_s: float = 25.0,
    n_speed_pts: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extend a trajectory backward (prepend points before the first obs).

    Parameters
    ----------
    target_y_neg : The (positive) minimum upstream distance.  The trajectory
        must reach y_min <= -target_y_neg.
    """
    if len(times) < 2:
        return times, y_vals

    current_y_min = float(np.min(y_vals))
    if current_y_min <= -target_y_neg:
        return times, y_vals                       # already passes

    # Estimate approach speed from first N points
    head = min(n_speed_pts, len(times) - 1)
    dt_head = np.diff(times[:head + 1])
    dy_head = np.diff(y_vals[:head + 1])

    valid = dt_head > 1e-6
    if np.any(valid):
        speeds = dy_head[valid] / dt_head[valid]
        avg_speed = float(np.mean(np.abs(speeds)))
    else:
        avg_speed = 0.0

    if avg_speed < _MIN_SPEED_FOR_EXTRAP_MPS:
        avg_speed = _DEFAULT_APPROACH_SPEED_MPS

    # Generate synthetic prepend points (going backward in time)
    t_first = float(times[0])
    y_first = float(y_vals[0])

    target_y = -target_y_neg - 2.0                 # small margin

    pre_times = []
    pre_y = []
    t = t_first
    y = y_first
    while y > target_y and (t_first - t) < max_extend_s:
        t -= _EXTRAP_DT
        y -= avg_speed * _EXTRAP_DT
        pre_times.append(t)
        pre_y.append(y)

    if not pre_times:
        return times, y_vals

    # Reverse so they're in chronological order
    pre_times.reverse()
    pre_y.reverse()

    return (
        np.concatenate([np.array(pre_times), times]),
        np.concatenate([np.array(pre_y), y_vals]),
    )


# ---------------------------------------------------------------------------
# Gap filling / smoothing  (for too_many_breaks)
# ---------------------------------------------------------------------------

def fill_gaps_and_smooth(
    times: np.ndarray,
    y_vals: np.ndarray,
    max_gap_s: float = 2.0,
    break_speed_mps: float = 12.0,
    rolling_window: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill trajectory gaps and smooth speed spikes.

    Strategy:
      1. Remove duplicate timestamps (keep median position).
      2. Identify speed spikes (> break_speed_mps) and replace with
         linearly-interpolated values.
      3. Fill temporal gaps (dt > max_gap_s) with linearly-interpolated
         points at _EXTRAP_DT intervals.
      4. Apply rolling-average smoothing on y-values.

    Returns
    -------
    (times_clean, y_clean) — the repaired trajectory.
    """
    if len(times) < 3:
        return times, y_vals

    t = times.astype(float).copy()
    y = y_vals.astype(float).copy()

    # Step 1: deduplicate timestamps (keep mean y for each unique t)
    unique_t, inv = np.unique(t, return_inverse=True)
    if len(unique_t) < len(t):
        mean_y = np.zeros(len(unique_t))
        for i in range(len(unique_t)):
            mean_y[i] = np.mean(y[inv == i])
        t = unique_t
        y = mean_y

    if len(t) < 3:
        return t, y

    # Step 2: identify and fix speed spikes
    dt = np.diff(t)
    dy = np.diff(y)
    speed = np.abs(dy / np.maximum(dt, 1e-6))

    spike_mask = speed > break_speed_mps
    if np.any(spike_mask):
        spike_idx = np.where(spike_mask)[0]
        for idx in spike_idx:
            # Replace the destination point with linear interpolation
            # from the nearest non-spike neighbours
            left = max(0, idx)
            right = min(len(y) - 1, idx + 2)
            if right > left:
                y[idx + 1] = y[left] + (y[right] - y[left]) * (
                    (t[idx + 1] - t[left]) / max(t[right] - t[left], 1e-6)
                )

    # Step 3: fill temporal gaps with interpolated points
    new_t = [t[0]]
    new_y = [y[0]]
    for i in range(len(t) - 1):
        gap = t[i + 1] - t[i]
        if gap > max_gap_s:
            # Interpolate across the gap
            n_fill = int(gap / _EXTRAP_DT)
            for j in range(1, n_fill):
                frac = j / n_fill
                new_t.append(t[i] + gap * frac)
                new_y.append(y[i] + (y[i + 1] - y[i]) * frac)
        new_t.append(t[i + 1])
        new_y.append(y[i + 1])

    t = np.array(new_t)
    y = np.array(new_y)

    # Step 4: rolling average smoothing
    if len(y) >= rolling_window:
        kernel = np.ones(rolling_window) / rolling_window
        # Pad edges with reflection to avoid shrinkage
        pad = rolling_window // 2
        y_padded = np.pad(y, pad, mode="edge")
        y_smooth = np.convolve(y_padded, kernel, mode="valid")
        # Ensure same length
        if len(y_smooth) == len(y):
            y = y_smooth

    return t, y


# ---------------------------------------------------------------------------
# Movement re-assignment  (for silently_skipped_unk_no_origin)
# ---------------------------------------------------------------------------

def infer_movement_from_road_sections(
    road_sections: list,
    section_lane_pairs: list = None,
) -> Optional[str]:
    """Infer a Q-intersection movement from a vehicle's Road_Section sequence.

    Checks which arm sections the vehicle enters and exits to determine
    the movement code (NS, SN, NE, SE, EN, ES).

    Parameters
    ----------
    road_sections : list of str — the Road_Section values in time order.
    section_lane_pairs : list of (section, lane) tuples — if provided,
        used for lane-aware inference when origin-only sections are present.

    Returns
    -------
    Movement string or None if ambiguous / insufficient data.
    """
    if not road_sections:
        return None

    # Define the arm section sets
    ARM1_IN = {"1_1", "1_2", "1_3"}
    ARM1_OUT = {"1_4", "1_5"}
    ARM2_OUT = {"2_1", "2_2"}
    ARM2_IN = {"2_3", "2_4"}
    ARM3_IN = {"3_1", "3_2"}
    ARM3_OUT = {"3_3", "3_4"}

    secs = [str(s) for s in road_sections if str(s) not in ("nan", "None", "")]

    origin_arm = None
    dest_arm = None

    for s in secs:
        if s in ARM1_IN:
            origin_arm = "ARM1"
            break
        elif s in ARM2_IN:
            origin_arm = "ARM2"
            break
        elif s in ARM3_IN:
            origin_arm = "ARM3"
            break

    for s in reversed(secs):
        if s in ARM1_OUT:
            dest_arm = "ARM1"
            break
        elif s in ARM2_OUT:
            dest_arm = "ARM2"
            break
        elif s in ARM3_OUT:
            dest_arm = "ARM3"
            break

    # Map (origin, dest) → movement
    _MAP = {
        ("ARM1", "ARM2"): "NS",
        ("ARM1", "ARM3"): "NE",
        ("ARM2", "ARM1"): "SN",
        ("ARM2", "ARM3"): "SE",
        ("ARM3", "ARM1"): "EN",
        ("ARM3", "ARM2"): "ES",
    }

    if origin_arm and dest_arm and origin_arm != dest_arm:
        return _MAP.get((origin_arm, dest_arm))

    # Destination found without origin → vehicle only in exit sections
    # (e.g., only in 2_1, 3_3, 1_4). Try to assign based on which exit arm.
    if dest_arm and not origin_arm:
        # Vehicle only in exit sections — infer from which exit
        sec_set = set(secs)
        if dest_arm == "ARM2":
            # Vehicles only in 2_1 could be NS or ES
            # If they came from East sections at all, it's ES
            if sec_set & ARM3_IN:
                return "ES"
            return "NS"  # default: most common
        if dest_arm == "ARM1":
            if sec_set & ARM3_IN:
                return "EN"
            return "SN"
        if dest_arm == "ARM3":
            # Could be NE or SE
            if sec_set & ARM1_IN:
                return "NE"
            if sec_set & ARM2_IN:
                return "SE"
            return None

    # Origin found without destination → lane-aware inference
    if origin_arm and not dest_arm:
        # Try lane-aware discrimination using section_lane_pairs
        if section_lane_pairs:
            for sec, lane in section_lane_pairs:
                sec_s = str(sec)
                try:
                    lane_i = int(float(lane))
                except (ValueError, TypeError):
                    continue
                # ARM1: NS uses lanes 1-3 in 1_2; NE uses lane 1 in 1_2
                # (but NE also starts from 1_2_1, so overlaps with NS)
                if origin_arm == "ARM1" and sec_s == "1_2":
                    if lane_i >= 2:
                        return "NS"  # lanes 2-3 are NS-only
                    # lane 1 is ambiguous (NS or NE)
                if origin_arm == "ARM1" and sec_s == "1_3":
                    if lane_i >= 3:
                        return "NS"  # lanes 3-5 in 1_3 are NS
                    elif lane_i <= 2:
                        return "NE"  # lanes 1-2 in 1_3 are NE
                # ARM2: SN uses lanes 1-2 in 2_3; SE uses lane 3
                if origin_arm == "ARM2" and sec_s in ("2_3", "2_4"):
                    if lane_i == 4:
                        return "SE"  # lane 4 in 2_4 is SE
                    elif lane_i <= 2:
                        return "SN"
                    elif lane_i == 3:
                        # lane 3 could be SN or SE
                        if sec_s == "2_4":
                            return "SN"
                        # 2_3 lane 3 → check if also in 2_4
                        if any(str(s) == "2_4" for s, _ in section_lane_pairs):
                            return "SN"
                        return "SE"
                # ARM3: ES uses lanes 1-2 in 3_2; EN uses lane 3
                if origin_arm == "ARM3" and sec_s == "3_2":
                    if lane_i <= 2:
                        return "ES"
                    elif lane_i == 3:
                        return "EN"
                if origin_arm == "ARM3" and sec_s == "3_1":
                    if lane_i <= 1:
                        return "ES"
                    elif lane_i == 2:
                        return "EN"

        # Last resort: check if any exit sections appear later in the sequence
        sec_set = set(secs)
        if origin_arm == "ARM1":
            if sec_set & ARM2_OUT:
                return "NS"
            if sec_set & ARM3_OUT:
                return "NE"
            # Default to the most common movement from this arm
            return "NS"
        if origin_arm == "ARM2":
            if sec_set & ARM1_OUT:
                return "SN"
            if sec_set & ARM3_OUT:
                return "SE"
            return "SN"
        if origin_arm == "ARM3":
            return "ES"  # most common from East

    return None


# ---------------------------------------------------------------------------
# Quality re-check  (applies all filters that _filter_trajectories uses)
# ---------------------------------------------------------------------------

def recheck_quality(
    times: np.ndarray,
    y_vals: np.ndarray,
    *,
    min_points: int = 20,
    min_span_s: float = 6.0,
    min_y_neg: float = 60.0,
    min_y_pos: float = 6.0,
    min_y_range: float = 55.0,
    max_gap_s: float = 8.0,
    max_break_ratio: float = 0.25,
    break_speed_mps: float = 12.0,
    cross_max_dt: float = 4.0,
    cross_max_speed: float = 25.0,
    require_crossing: bool = True,
) -> Tuple[bool, str]:
    """Re-evaluate a trajectory against all space-time quality filters.

    Returns (passed: bool, fail_reason: str).
    If passed is True, fail_reason is empty.
    """
    if len(times) < min_points:
        return False, "too_few_points"

    span = float(times[-1] - times[0])
    if span < min_span_s:
        return False, "too_short_in_time"

    y_min = float(np.min(y_vals))
    y_max = float(np.max(y_vals))
    y_range = y_max - y_min

    if y_min > -min_y_neg:
        return False, "no_upstream_approach"
    if y_max < min_y_pos:
        return False, "no_downstream_departure"
    if y_range < min_y_range:
        return False, "small_y_range"

    dt = np.diff(times)
    if len(dt) == 0:
        return False, "single_point"

    if float(np.max(dt)) > max_gap_s:
        return False, "large_internal_gap"

    dy = np.diff(y_vals)
    speed = np.abs(dy / np.maximum(dt, 1e-6))
    break_ratio = float(np.mean((dt <= 1e-3) | (speed > break_speed_mps)))
    if break_ratio > max_break_ratio:
        return False, "too_many_breaks"

    # Stopline crossing check
    if require_crossing:
        has_crossing = False
        for i in range(len(dt)):
            if y_vals[i] < 0.0 and y_vals[i + 1] >= 0.0:
                if dt[i] <= cross_max_dt and speed[i] <= cross_max_speed:
                    has_crossing = True
                    break
        if not has_crossing:
            return False, "no_clean_stopline_crossing"

    return True, ""
