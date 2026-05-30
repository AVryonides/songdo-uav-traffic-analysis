"""Signal inference engine for space-time diagram signal phase detection.

Hybrid approach: stop-to-start event detection + modulo-histogram cycle
estimation, replacing the old rate-threshold method.

Algorithm overview:
  1. Detect **stopline crossings** — vehicles whose y-coordinate goes from
     negative to positive.
  2. Detect **stop-to-start events** — vehicles that were stopped near the
     stopline and then began moving.  Each event marks a green onset.
  3. Estimate **cycle length** via modulo-histogram on pooled crossings
     (shared across all movements that call set_global_cycle).
  4. For each movement, find its green/red phase positions within that
     shared cycle using the movement's own crossing times.
  5. Tile phase windows across the observation period, refined by
     stop-to-start events.
  6. Split red into **certain** (vehicles waiting on approach) vs
     **uncertain** (no vehicles visible).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd


class SignalInferenceEngine:
    """Hybrid stop-to-start + modulo-histogram signal phase inference.

    All threshold constants are stored as instance attributes so they can be
    overridden per-instance via ``**config_overrides`` passed to ``__init__``.
    """

    def __init__(self, **config_overrides):
        # Observation window parameters
        self.OBS_GAP_S: float = config_overrides.get("OBS_GAP_S", 12.0)
        self.OBS_MIN_WINDOW_S: float = config_overrides.get("OBS_MIN_WINDOW_S", 3.0)
        self.MAX_STEP_SPEED_FOR_CROSS: float = config_overrides.get("MAX_STEP_SPEED_FOR_CROSS", 20.0)

        # Stop-to-start detection
        self.STOP_SPEED_THRESH: float = config_overrides.get("STOP_SPEED_THRESH", 1.0)
        self.START_SPEED_THRESH: float = config_overrides.get("START_SPEED_THRESH", 2.0)
        self.STOPLINE_PROXIMITY: float = config_overrides.get("STOPLINE_PROXIMITY", 30.0)
        self.MIN_STOP_DURATION_S: float = config_overrides.get("MIN_STOP_DURATION_S", 2.0)

        # Cycle length estimation
        self.CYCLE_MIN_S: float = config_overrides.get("CYCLE_MIN_S", 60.0)
        self.CYCLE_MAX_S: float = config_overrides.get("CYCLE_MAX_S", 200.0)
        self.CYCLE_STEP_S: float = config_overrides.get("CYCLE_STEP_S", 1.0)
        self.HISTOGRAM_BINS: int = config_overrides.get("HISTOGRAM_BINS", 60)
        self.MIN_CROSSINGS_FOR_CYCLE: int = config_overrides.get("MIN_CROSSINGS_FOR_CYCLE", 15)
        self.MIN_CROSSINGS_FOR_PHASE: int = config_overrides.get("MIN_CROSSINGS_FOR_PHASE", 10)

        # Green / red duration constraints
        self.MIN_GREEN_DURATION_S: float = config_overrides.get("MIN_GREEN_DURATION_S", 8.0)
        self.MIN_RED_DURATION_S: float = config_overrides.get("MIN_RED_DURATION_S", 8.0)
        self.GREEN_MERGE_GAP_S: float = config_overrides.get("GREEN_MERGE_GAP_S", 5.0)

        # Reaction-time correction applied to the behavioral green onset.
        # Shifts every detected green start earlier by this many seconds.
        # Tune visually: increase if bar still starts late, decrease if it
        # starts before any visible vehicle movement.
        self.GREEN_ONSET_CORRECTION_S: float = config_overrides.get("GREEN_ONSET_CORRECTION_S", 4.0)

        # Tight-proximity onset refinement.
        # After the 30m behavioural detection finds approximate green starts,
        # a second pass looks only at vehicles within ONSET_TIGHT_PROXIMITY_M
        # of the stop line — these are front-of-queue vehicles that react
        # almost immediately to green.  Their first movement is a precise
        # onset marker, shifted back by ONSET_REACTION_S (pure reaction time,
        # no queue-discharge delay).
        self.ONSET_TIGHT_PROXIMITY_M: float = config_overrides.get("ONSET_TIGHT_PROXIMITY_M", 10.0)
        self.ONSET_REACTION_S: float        = config_overrides.get("ONSET_REACTION_S", 2.0)
        self.ONSET_SNAP_WINDOW_S: float     = config_overrides.get("ONSET_SNAP_WINDOW_S", 15.0)

        # Minimum duration a green (or red) interval must have to be shown.
        # Any interval shorter than this is treated as an algorithm artefact
        # (noise spike) and removed before the results are returned.
        # Increase this value if short spikes still appear in the diagrams.
        self.MIN_PHASE_DISPLAY_S: float = config_overrides.get("MIN_PHASE_DISPLAY_S", 5.0)

        # Crossing-corroboration filter for short green intervals.
        # A green interval shorter than MAX_UNCORROBORATED_GREEN_S that has
        # NO stop-line crossings within GREEN_CROSSING_CORROBORATE_S seconds
        # of it is treated as a false positive caused by arriving vehicles
        # moving through the approach zone during red.  Real greens always
        # produce crossings as the queue discharges.
        self.MAX_UNCORROBORATED_GREEN_S: float  = config_overrides.get("MAX_UNCORROBORATED_GREEN_S", 20.0)
        self.GREEN_CROSSING_CORROBORATE_S: float = config_overrides.get("GREEN_CROSSING_CORROBORATE_S", 30.0)

        # Minimum physically plausible red gap BETWEEN two green intervals.
        # If two detected green intervals are separated by a red gap shorter
        # than this, the red is a false split caused by arriving vehicles
        # moving through the approach zone during red (they look "moving"
        # to the behavioral detector before they stop and queue).
        # In a real signal cycle the conflicting phases need at least this
        # many seconds, so a shorter inter-green red is impossible.
        self.SHORT_RED_MERGE_S: float = config_overrides.get("SHORT_RED_MERGE_S", 15.0)

        # Proximity used for the "definitive red" stopped-interval enforcement.
        # Only vehicles within this distance of the stop line can generate a
        # stopped-interval that overrides/trims a detected green interval.
        # Keep this MUCH smaller than STOPLINE_PROXIMITY so that back-of-queue
        # vehicles (still stationary in the first 2–3 s of actual green) do not
        # push the green onset forward and undo reaction-time corrections.
        self.STOPPED_ENFORCE_PROXIMITY_M: float = config_overrides.get("STOPPED_ENFORCE_PROXIMITY_M", 8.0)

        # Vehicle presence detection (for red_certain vs red_uncertain)
        self.PRESENCE_BIN_S: float = config_overrides.get("PRESENCE_BIN_S", 1.0)
        self.PRESENCE_SMOOTH_WINDOW_S: int = config_overrides.get("PRESENCE_SMOOTH_WINDOW_S", 5)
        self.PRESENCE_THRESH: float = config_overrides.get("PRESENCE_THRESH", 0.3)

        # Fallback rate-based parameters
        self.RATE_BIN_S: float = config_overrides.get("RATE_BIN_S", 1.0)
        self.RATE_SMOOTH_WINDOW_S: int = config_overrides.get("RATE_SMOOTH_WINDOW_S", 7)
        self.GREEN_RATE_THRESH: float = config_overrides.get("GREEN_RATE_THRESH", 0.08)
        self.RED_RATE_THRESH: float = config_overrides.get("RED_RATE_THRESH", 0.05)
        self.RED_MERGE_GAP_S: float = config_overrides.get("RED_MERGE_GAP_S", 4.0)

        # Global cycle length (shared across movements, set once)
        self._global_cycle: Optional[float] = None

    # ------------------------------------------------------------------
    # Global cycle management
    # ------------------------------------------------------------------

    def set_global_cycle(self, all_crossings: List[float]) -> Optional[float]:
        """Estimate and store a single global cycle length from pooled crossings.

        Should be called ONCE with crossings from all movements combined
        (or from the strongest movements) before calling infer_signal_intervals
        for individual movements.
        """
        self._global_cycle = self._estimate_cycle_length(all_crossings)
        return self._global_cycle

    def get_global_cycle(self) -> Optional[float]:
        return self._global_cycle

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def merge_intervals(
        self, intervals: List[Tuple[float, float]], max_join_gap: float = 0.0
    ) -> List[Tuple[float, float]]:
        if not intervals:
            return []
        arr = sorted((float(a), float(b)) for a, b in intervals if b > a)
        if not arr:
            return []
        out = [arr[0]]
        for s, e in arr[1:]:
            ps, pe = out[-1]
            if s <= pe + max_join_gap:
                out[-1] = (ps, max(pe, e))
            else:
                out.append((s, e))
        return out

    def clip_intervals_to_observed(
        self,
        intervals: List[Tuple[float, float]],
        observed_intervals: List[Tuple[float, float]],
        min_len: float = 1.0,
    ) -> List[Tuple[float, float]]:
        clipped: List[Tuple[float, float]] = []
        for s, e in intervals:
            for os_, oe in observed_intervals:
                a = max(float(s), float(os_))
                b = min(float(e), float(oe))
                if (b - a) >= float(min_len):
                    clipped.append((a, b))
        return self.merge_intervals(clipped, max_join_gap=0.75)

    # ------------------------------------------------------------------
    # Observation windows
    # ------------------------------------------------------------------

    def compute_observed_intervals(
        self,
        phase_df: pd.DataFrame,
        max_gap_s: Optional[float] = None,
        min_window_s: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        if max_gap_s is None:
            max_gap_s = self.OBS_GAP_S
        if min_window_s is None:
            min_window_s = self.OBS_MIN_WINDOW_S

        if phase_df.empty:
            return []
        tvals = np.sort(np.unique(phase_df["t"].astype(float).to_numpy()))
        if len(tvals) == 0:
            return []
        windows: List[Tuple[float, float]] = []
        s = float(tvals[0])
        prev = float(tvals[0])
        for t in tvals[1:]:
            t = float(t)
            if (t - prev) > float(max_gap_s):
                if (prev - s) >= float(min_window_s):
                    windows.append((s, prev))
                s = t
            prev = t
        if (prev - s) >= float(min_window_s):
            windows.append((s, prev))
        return self.merge_intervals(windows, max_join_gap=0.5)

    # ------------------------------------------------------------------
    # Vehicle presence
    # ------------------------------------------------------------------

    def compute_vehicle_presence(
        self,
        phase_df: pd.DataFrame,
        observed_intervals: List[Tuple[float, float]],
    ) -> Callable[[float, float], bool]:
        if phase_df.empty or not observed_intervals:
            return lambda s, e: False

        approach_df = phase_df[phase_df["y"] < 0.0]
        if approach_df.empty:
            return lambda s, e: False

        PRESENCE_BIN_S = self.PRESENCE_BIN_S
        PRESENCE_SMOOTH_WINDOW_S = self.PRESENCE_SMOOTH_WINDOW_S
        PRESENCE_THRESH = self.PRESENCE_THRESH

        t_min = float(observed_intervals[0][0])
        t_max = float(observed_intervals[-1][1])
        n_bins = int(np.ceil((t_max - t_min) / PRESENCE_BIN_S)) + 1

        presence = np.zeros(n_bins, dtype=float)
        for _, vg in approach_df.groupby("vid"):
            t_arr = vg["t"].values.astype(float)
            vt_min, vt_max = float(t_arr.min()), float(t_arr.max())
            b_start = max(0, int((vt_min - t_min) / PRESENCE_BIN_S))
            b_end = min(n_bins - 1, int((vt_max - t_min) / PRESENCE_BIN_S))
            presence[b_start:b_end + 1] += 1.0

        hw = PRESENCE_SMOOTH_WINDOW_S // 2
        kernel = np.ones(2 * hw + 1) / (2 * hw + 1)
        presence_smooth = np.convolve(presence, kernel, mode="same")

        def is_present(t_start: float, t_end: float) -> bool:
            b0 = max(0, int((t_start - t_min) / PRESENCE_BIN_S))
            b1 = min(n_bins - 1, int((t_end - t_min) / PRESENCE_BIN_S))
            if b0 > b1:
                return False
            return float(np.mean(presence_smooth[b0:b1 + 1])) >= PRESENCE_THRESH
        return is_present

    # ------------------------------------------------------------------
    # Crossing extraction (unchanged interface)
    # ------------------------------------------------------------------

    def extract_crossing_times(self, phase_df: pd.DataFrame) -> List[float]:
        MAX_STEP_SPEED_FOR_CROSS = self.MAX_STEP_SPEED_FOR_CROSS

        crossing_times: List[float] = []
        for _, g in phase_df.groupby("vid"):
            gg = g.sort_values("t")
            gg = gg.groupby("t", sort=True, as_index=False).agg({"y": "median"})
            t_arr = gg["t"].values.astype(float)
            y_arr = gg["y"].values.astype(float)
            if len(t_arr) < 2:
                continue

            dt = np.diff(t_arr)
            dy = np.diff(y_arr)
            speed = np.abs(dy / np.maximum(dt, 1e-6))

            for i in range(len(dt)):
                if dt[i] <= 0.05 or dt[i] > 4.0:
                    continue
                if speed[i] > MAX_STEP_SPEED_FOR_CROSS:
                    continue
                y0 = y_arr[i]
                y1 = y_arr[i + 1]
                if not (y0 < 0.0 and y1 >= 0.0):
                    continue
                if y1 == y0:
                    continue
                frac = (0.0 - y0) / (y1 - y0)
                crossing_times.append(float(t_arr[i] + frac * (t_arr[i + 1] - t_arr[i])))

        return sorted(crossing_times)

    def extract_crossing_events(
        self, phase_df: pd.DataFrame
    ) -> List[Tuple[float, int]]:
        """Return list of (t_cross, vid) for every valid stopline crossing."""
        MAX_STEP_SPEED_FOR_CROSS = self.MAX_STEP_SPEED_FOR_CROSS

        events: List[Tuple[float, int]] = []
        for vid, g in phase_df.groupby("vid"):
            gg = g.sort_values("t")
            gg = gg.groupby("t", sort=True, as_index=False).agg({"y": "median"})
            t_arr = gg["t"].values.astype(float)
            y_arr = gg["y"].values.astype(float)
            if len(t_arr) < 2:
                continue

            dt = np.diff(t_arr)
            dy = np.diff(y_arr)
            speed = np.abs(dy / np.maximum(dt, 1e-6))

            for i in range(len(dt)):
                if dt[i] <= 0.05 or dt[i] > 4.0:
                    continue
                if speed[i] > MAX_STEP_SPEED_FOR_CROSS:
                    continue
                y0 = y_arr[i]
                y1 = y_arr[i + 1]
                if not (y0 < 0.0 and y1 >= 0.0):
                    continue
                if y1 == y0:
                    continue
                frac = (0.0 - y0) / (y1 - y0)
                t_cross = float(t_arr[i] + frac * (t_arr[i + 1] - t_arr[i]))
                events.append((t_cross, int(vid)))
                break

        events.sort(key=lambda x: x[0])
        return events

    # ------------------------------------------------------------------
    # Stop-to-start event detection
    # ------------------------------------------------------------------

    def _extract_stop_to_start_events(
        self, phase_df: pd.DataFrame
    ) -> List[float]:
        """Detect green-onset events: vehicles near the stopline that
        transition from stopped to moving.

        Returns a sorted list of timestamps when vehicles start moving.
        """
        events: List[float] = []

        for _, g in phase_df.groupby("vid"):
            gg = g.sort_values("t")
            t_arr = gg["t"].values.astype(float)
            y_arr = gg["y"].values.astype(float)
            if len(t_arr) < 4:
                continue

            dt = np.diff(t_arr)
            dy = np.diff(y_arr)
            speed = np.abs(dy / np.maximum(dt, 1e-6))

            near_stopline = (y_arr[:-1] < 0.0) & (y_arr[:-1] > -self.STOPLINE_PROXIMITY)
            is_stopped = (speed < self.STOP_SPEED_THRESH) & near_stopline
            is_moving = speed >= self.START_SPEED_THRESH

            i = 0
            while i < len(is_stopped):
                if not is_stopped[i]:
                    i += 1
                    continue

                stop_start = i
                while i < len(is_stopped) and is_stopped[i]:
                    i += 1
                stop_end = i

                stop_duration = t_arr[stop_end] - t_arr[stop_start] if stop_end < len(t_arr) else 0.0
                if stop_duration < self.MIN_STOP_DURATION_S:
                    continue

                for j in range(stop_end, min(stop_end + 10, len(is_moving))):
                    if is_moving[j]:
                        events.append(float(t_arr[j]))
                        break

        return sorted(events)

    def _extract_stopped_intervals(self, phase_df: pd.DataFrame) -> List[Tuple[float, float]]:
        """Detect definitively RED intervals: vehicles stopped very close to the stopline.

        Uses STOPPED_ENFORCE_PROXIMITY_M (default 8 m) — much tighter than
        STOPLINE_PROXIMITY (30 m) — so that back-of-queue vehicles still
        stationary in the first 2–3 s of actual green are NOT captured here
        and cannot push the reaction-time-corrected green onset forward.
        """
        stopped_intervals = []
        for _, g in phase_df.groupby("vid"):
            gg = g.sort_values("t")
            t_arr = gg["t"].values.astype(float)
            y_arr = gg["y"].values.astype(float)
            if len(t_arr) < 4:
                continue
            dt = np.diff(t_arr)
            dy = np.diff(y_arr)
            speed = np.abs(dy / np.maximum(dt, 1e-6))
            near_stopline = (y_arr[:-1] < 0.0) & (y_arr[:-1] > -self.STOPPED_ENFORCE_PROXIMITY_M)
            is_stopped = (speed < self.STOP_SPEED_THRESH) & near_stopline
            
            i = 0
            while i < len(is_stopped):
                if not is_stopped[i]:
                    i += 1
                    continue
                start_i = i
                while i < len(is_stopped) and is_stopped[i]:
                    i += 1
                end_i = i
                
                stop_duration = t_arr[end_i] - t_arr[start_i] if end_i < len(t_arr) else 0.0
                if stop_duration >= self.MIN_STOP_DURATION_S:
                    stopped_intervals.append((float(t_arr[start_i]), float(t_arr[end_i])))
        
        # Merge overlapping stopped intervals across all vehicles
        return self.merge_intervals(stopped_intervals, max_join_gap=2.0)

    # ------------------------------------------------------------------
    # Modulo-histogram cycle length estimation
    # ------------------------------------------------------------------

    def _estimate_cycle_length(
        self, crossing_times: List[float]
    ) -> Optional[float]:
        """Estimate cycle length using modulo-histogram analysis.

        For each candidate cycle length, project all crossing times using
        modulo and build a histogram.  The correct cycle length produces a
        histogram with the sharpest gap (= red phase where no crossings).
        """
        if len(crossing_times) < self.MIN_CROSSINGS_FOR_CYCLE:
            return None

        ct = np.array(crossing_times, dtype=float)
        n_bins = self.HISTOGRAM_BINS

        best_cycle = None
        best_score = -1.0

        candidates = np.arange(
            self.CYCLE_MIN_S, self.CYCLE_MAX_S + self.CYCLE_STEP_S, self.CYCLE_STEP_S
        )

        for cycle in candidates:
            residuals = ct % cycle
            hist, _ = np.histogram(residuals, bins=n_bins, range=(0.0, cycle))

            # Lower threshold to detect sparse crossing phases
            threshold = max(0.5, len(ct) / (n_bins * 3))
            is_empty = hist <= threshold

            # Longest consecutive empty run (circular)
            max_run = 0
            run = 0
            extended = np.concatenate([is_empty, is_empty])
            for val in extended:
                if val:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 0
            max_run = min(max_run, n_bins)

            gap_fraction = max_run / n_bins
            if gap_fraction < 0.10 or gap_fraction > 0.70:
                continue

            filled_bins = hist[~is_empty]
            if len(filled_bins) == 0:
                continue
            contrast = float(np.mean(filled_bins)) / max(float(np.mean(hist)) + 1e-9, 1e-9)

            score = gap_fraction * contrast

            if score > best_score:
                best_score = score
                best_cycle = float(cycle)

        return best_cycle

    # ------------------------------------------------------------------
    # Find per-movement phase positions within the global cycle
    # ------------------------------------------------------------------

    def _find_phase_positions_in_cycle(
        self, crossing_times: List[float], cycle_length: float
    ) -> Tuple[float, float, float, float]:
        """Given crossing times and cycle length, find green/red positions.

        Returns (green_start_offset, green_end_offset, red_start_offset, red_end_offset)
        where offsets are within [0, cycle_length).
        """
        ct = np.array(crossing_times, dtype=float)
        n_bins = self.HISTOGRAM_BINS

        residuals = ct % cycle_length
        bin_width = cycle_length / n_bins
        hist, _ = np.histogram(residuals, bins=n_bins, range=(0.0, cycle_length))

        kernel = np.ones(3) / 3.0
        hist_smooth = np.convolve(hist, kernel, mode="same")

        threshold = max(1, len(ct) / (n_bins * 2))
        is_empty = hist_smooth <= threshold

        # Find longest gap (consecutive low bins) in circular histogram
        extended = np.concatenate([is_empty, is_empty])
        best_run_start = 0
        best_run_len = 0
        run_start = 0
        run_len = 0
        for i, val in enumerate(extended):
            if val:
                if run_len == 0:
                    run_start = i
                run_len += 1
                if run_len > best_run_len:
                    best_run_len = run_len
                    best_run_start = run_start
            else:
                run_len = 0

        best_run_len = min(best_run_len, n_bins)

        gap_start_bin = best_run_start % n_bins
        gap_end_bin = (best_run_start + best_run_len) % n_bins

        red_start = gap_start_bin * bin_width
        red_end = gap_end_bin * bin_width

        green_start = red_end
        green_end = red_start
        if green_end <= green_start:
            green_end += cycle_length

        return green_start, green_end, red_start, red_end

    # ------------------------------------------------------------------
    # Tile phase windows across observation period
    # ------------------------------------------------------------------

    def _tile_phases(
        self,
        cycle_length: float,
        green_start_offset: float,
        green_end_offset: float,
        observed_intervals: List[Tuple[float, float]],
        crossing_times: List[float],
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Tile green and red intervals across the observation period."""
        if not crossing_times or not observed_intervals:
            return [], []

        t_min = float(observed_intervals[0][0])
        t_max = float(observed_intervals[-1][1])

        # Green duration within one cycle
        green_duration = green_end_offset - green_start_offset
        if green_duration <= 0:
            green_duration += cycle_length
        red_duration = cycle_length - green_duration

        # FIX: The old logic aligned the cycle incorrectly.
        # Anchor: find the cycle offset that aligns with crossing data.
        # The median crossing residual should fall within the green window.
        ct = np.array(crossing_times, dtype=float)
        residuals = ct % cycle_length
        
        # Sort residuals and find the middle one, AND subtract its exact value
        # to ensure anchor is a perfect multiple of cycle_length
        sorted_ct = np.sort(ct)
        median_crossing = sorted_ct[len(sorted_ct) // 2]
        anchor = median_crossing - (median_crossing % cycle_length)

        # Tile cycles
        green_intervals: List[Tuple[float, float]] = []
        red_intervals: List[Tuple[float, float]] = []

        n_before = int(np.ceil((anchor - t_min) / cycle_length)) + 1
        cycle_start = anchor - n_before * cycle_length

        while cycle_start < t_max + cycle_length:
            # Green window
            gs = cycle_start + green_start_offset
            ge = cycle_start + green_end_offset
            if ge <= gs:
                ge += cycle_length
            green_intervals.append((gs, ge))

            # Red window = the complement within this cycle
            # Red goes from green_end to next green_start
            rs = ge
            re = gs + cycle_length
            if (re - rs) >= self.MIN_RED_DURATION_S * 0.3:
                red_intervals.append((rs, re))

            cycle_start += cycle_length

        # Clip to observed windows
        green_intervals = self.clip_intervals_to_observed(
            green_intervals, observed_intervals, min_len=self.MIN_GREEN_DURATION_S * 0.3
        )
        red_intervals = self.clip_intervals_to_observed(
            red_intervals, observed_intervals, min_len=self.MIN_RED_DURATION_S * 0.3
        )

        # Remove any red that overlaps with green
        red_clean: List[Tuple[float, float]] = []
        for rs, re in red_intervals:
            if not any(rs < ge and re > gs for gs, ge in green_intervals):
                red_clean.append((rs, re))

        return green_intervals, red_clean

    # ------------------------------------------------------------------
    # Fallback: rate-based method
    # ------------------------------------------------------------------

    def _infer_rate_based(
        self,
        phase_df: pd.DataFrame,
        observed_intervals: List[Tuple[float, float]],
        crossing_times: List[float],
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Original rate-threshold method as fallback."""
        if not crossing_times or not observed_intervals:
            return [], []

        t_min = float(observed_intervals[0][0])
        t_max = float(observed_intervals[-1][1])
        n_bins = int(np.ceil((t_max - t_min) / self.RATE_BIN_S)) + 1

        counts = np.zeros(n_bins, dtype=float)
        for tc in crossing_times:
            b = int((tc - t_min) / self.RATE_BIN_S)
            if 0 <= b < n_bins:
                counts[b] += 1.0
        rate = counts / self.RATE_BIN_S

        hw = self.RATE_SMOOTH_WINDOW_S // 2
        kernel = np.ones(2 * hw + 1) / (2 * hw + 1)
        rate_smooth = np.convolve(rate, kernel, mode="same")

        # Green
        green_mask = rate_smooth >= self.GREEN_RATE_THRESH
        green_cands: List[Tuple[float, float]] = []
        in_g = False
        gs = 0.0
        for i in range(n_bins):
            t_bin = t_min + i * self.RATE_BIN_S
            if green_mask[i] and not in_g:
                gs = t_bin
                in_g = True
            elif not green_mask[i] and in_g:
                if (t_bin - gs) >= self.MIN_GREEN_DURATION_S:
                    green_cands.append((gs, t_bin))
                in_g = False
        if in_g:
            ge = t_min + n_bins * self.RATE_BIN_S
            if (ge - gs) >= self.MIN_GREEN_DURATION_S:
                green_cands.append((gs, ge))

        green_ints = self.merge_intervals(green_cands, max_join_gap=self.GREEN_MERGE_GAP_S)
        green_ints = self.clip_intervals_to_observed(
            green_ints, observed_intervals, min_len=self.MIN_GREEN_DURATION_S
        )

        # Red
        red_mask = rate_smooth < self.RED_RATE_THRESH
        red_cands: List[Tuple[float, float]] = []
        in_r = False
        rs = 0.0
        for i in range(n_bins):
            t_bin = t_min + i * self.RATE_BIN_S
            if red_mask[i] and not in_r:
                rs = t_bin
                in_r = True
            elif not red_mask[i] and in_r:
                if (t_bin - rs) >= self.MIN_RED_DURATION_S:
                    red_cands.append((rs, t_bin))
                in_r = False
        if in_r:
            re = t_min + n_bins * self.RATE_BIN_S
            if (re - rs) >= self.MIN_RED_DURATION_S:
                red_cands.append((rs, re))

        red_ints = self.merge_intervals(red_cands, max_join_gap=self.RED_MERGE_GAP_S)
        red_ints = self.clip_intervals_to_observed(
            red_ints, observed_intervals, min_len=self.MIN_RED_DURATION_S
        )

        red_clean: List[Tuple[float, float]] = []
        for rs, re in red_ints:
            if not any(rs < ge and re > gs for gs, ge in green_ints):
                red_clean.append((rs, re))

        return green_ints, red_clean

    # ------------------------------------------------------------------
    # Refine phase boundaries with stop-to-start events
    # ------------------------------------------------------------------

    def _refine_with_stop_start(
        self,
        green_intervals: List[Tuple[float, float]],
        red_intervals: List[Tuple[float, float]],
        stop_start_events: List[float],
        observed_intervals: List[Tuple[float, float]],
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Refine green onset times using stop-to-start events.

        If a cluster of stop-to-start events falls just before a green
        interval starts, snap the green start to the earliest event in
        that cluster.
        """
        if not stop_start_events or not green_intervals:
            return green_intervals, red_intervals

        SNAP_WINDOW = 10.0

        refined_green = list(green_intervals)

        for event_t in stop_start_events:
            for idx, (gs, ge) in enumerate(refined_green):
                if 0.0 < (gs - event_t) <= SNAP_WINDOW:
                    refined_green[idx] = (event_t, ge)
                    break
                if abs(gs - event_t) <= 2.0:
                    break

        refined_green = self.merge_intervals(refined_green, max_join_gap=3.0)

        # Rebuild red intervals as gaps between green intervals
        if len(refined_green) >= 2:
            new_red: List[Tuple[float, float]] = []
            for i in range(len(refined_green) - 1):
                gap_s = refined_green[i][1]
                gap_e = refined_green[i + 1][0]
                if (gap_e - gap_s) >= self.MIN_RED_DURATION_S * 0.3:
                    new_red.append((gap_s, gap_e))
            # Also check gap before first green and after last green
            if observed_intervals:
                obs_start = observed_intervals[0][0]
                obs_end = observed_intervals[-1][1]
                if refined_green[0][0] - obs_start >= self.MIN_RED_DURATION_S * 0.3:
                    new_red.insert(0, (obs_start, refined_green[0][0]))
                if obs_end - refined_green[-1][1] >= self.MIN_RED_DURATION_S * 0.3:
                    new_red.append((refined_green[-1][1], obs_end))

            new_red = self.clip_intervals_to_observed(
                new_red, observed_intervals, min_len=self.MIN_RED_DURATION_S * 0.3
            )
            if new_red:
                red_intervals = new_red

        return refined_green, red_intervals

    # ------------------------------------------------------------------
    # Rebuild red as complement of green within observed windows
    # ------------------------------------------------------------------

    def _rebuild_red_from_green(
        self,
        green_intervals: List[Tuple[float, float]],
        observed_intervals: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Return red intervals = observed time minus green time.

        Every second inside an observed window that is NOT covered by a
        green interval becomes part of a red interval.  Tiny red slivers
        (< MIN_RED_DURATION_S * 0.3) are discarded.
        """
        if not observed_intervals:
            return []

        red: List[Tuple[float, float]] = []
        min_red = self.MIN_RED_DURATION_S * 0.3

        for obs_s, obs_e in observed_intervals:
            # Collect green intervals that overlap this observed window
            greens_in_obs = []
            for gs, ge in green_intervals:
                # Clip to observed window
                cs = max(gs, obs_s)
                ce = min(ge, obs_e)
                if ce > cs:
                    greens_in_obs.append((cs, ce))
            greens_in_obs.sort()

            # Red = gaps between consecutive greens (and at boundaries)
            cursor = obs_s
            for gs, ge in greens_in_obs:
                if gs - cursor >= min_red:
                    red.append((cursor, gs))
                cursor = max(cursor, ge)
            if obs_e - cursor >= min_red:
                red.append((cursor, obs_e))

        return red

    # ------------------------------------------------------------------
    # Behavioral phase detection (primary method)
    # ------------------------------------------------------------------

    def _infer_from_vehicle_behavior(
        self,
        phase_df: pd.DataFrame,
        observed_intervals: List[Tuple[float, float]],
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Infer green/red phases directly from vehicle behavior near the stop line.

        RED  — vehicles near the stop line are predominantly stopped
               (near-horizontal trajectories in the space-time diagram).
        GREEN — vehicles near the stop line are predominantly moving
               (diagonal trajectories: either queue discharging or free flow).

        Green onset is shifted back by GREEN_ONSET_CORRECTION_S seconds to
        account for the reaction time between the actual signal turning green
        and vehicles visibly starting to move.

        Returns (green_intervals, red_intervals), or ([], []) if there is
        insufficient approach-side data (caller should fall back to histogram).
        """
        # Only vehicles on the approach side, within proximity of the stop line
        near = phase_df[
            (phase_df["y"] < 0.0) & (phase_df["y"] > -self.STOPLINE_PROXIMITY)
        ].copy()

        if near.empty:
            return [], []

        BIN            = 1.0                      # 1-second time bins
        STOPPED_THRESH = self.STOP_SPEED_THRESH   # m/s — below this = stopped
        MIN_VEH        = 2                        # min vehicles per bin to trust
        SMOOTH_W       = 3                        # smoothing window (bins)
        RED_FRAC       = 0.55                     # fraction stopped → RED
        GREEN_FRAC     = 0.30                     # fraction stopped → GREEN

        t_min = float(observed_intervals[0][0])
        t_max = float(observed_intervals[-1][1])
        n_bins = int(np.ceil((t_max - t_min) / BIN)) + 1

        stopped_count = np.zeros(n_bins, dtype=float)
        total_count   = np.zeros(n_bins, dtype=float)

        for vid, g in near.groupby("vid"):
            g = g.sort_values("t")
            t_arr = g["t"].values.astype(float)
            y_arr = g["y"].values.astype(float)
            if len(t_arr) < 2:
                continue
            dt  = np.diff(t_arr)
            dy  = np.diff(y_arr)
            spd = np.abs(dy / np.maximum(dt, 1e-6))
            for i in range(len(dt)):
                if dt[i] <= 0.0 or dt[i] > 4.0:
                    continue
                t_mid = float(0.5 * (t_arr[i] + t_arr[i + 1]))
                b = int((t_mid - t_min) / BIN)
                if 0 <= b < n_bins:
                    total_count[b] += 1.0
                    if spd[i] < STOPPED_THRESH:
                        stopped_count[b] += 1.0

        # Fraction of stopped vehicles per bin; NaN where fewer than MIN_VEH
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(total_count >= MIN_VEH,
                            stopped_count / np.maximum(total_count, 1.0),
                            np.nan)

        # Need enough valid bins to be useful
        valid = ~np.isnan(frac)
        if valid.sum() < SMOOTH_W * 2:
            return [], []

        # Smooth (fill NaN→0 for convolution, restore NaN mask after)
        kernel = np.ones(SMOOTH_W) / SMOOTH_W
        frac_filled  = np.where(np.isnan(frac), 0.0, frac)
        frac_smoothed = np.full(n_bins, np.nan)
        frac_smoothed[valid] = np.convolve(frac_filled, kernel, mode="same")[valid]

        # State per bin: 1=RED, 0=GREEN, -1=UNKNOWN
        state = np.full(n_bins, -1, dtype=int)
        state[~np.isnan(frac_smoothed) & (frac_smoothed >= RED_FRAC)]   = 1
        state[~np.isnan(frac_smoothed) & (frac_smoothed <= GREEN_FRAC)] = 0

        # Forward-fill unknown bins from the previous known state
        prev = -1
        for i in range(n_bins):
            if state[i] != -1:
                prev = state[i]
            elif prev != -1:
                state[i] = prev

        # If we never settled on a state the data is too sparse — fall back
        if (state == -1).all():
            return [], []

        # Build intervals by scanning state transitions
        green_intervals: List[Tuple[float, float]] = []
        red_intervals:   List[Tuple[float, float]] = []

        cur_state  = state[0]
        seg_start  = t_min

        for i in range(1, n_bins):
            if state[i] == cur_state:
                continue
            seg_end = t_min + i * BIN
            if cur_state == 0:   # GREEN segment ended
                # Shift onset earlier by reaction-time correction
                g_start = max(seg_start - self.GREEN_ONSET_CORRECTION_S,
                              observed_intervals[0][0])
                green_intervals.append((g_start, seg_end))
            elif cur_state == 1:  # RED segment ended
                red_intervals.append((seg_start, seg_end))
            cur_state = state[i]
            seg_start = seg_end

        # Close the final segment
        seg_end = t_min + n_bins * BIN
        if cur_state == 0:
            g_start = max(seg_start - self.GREEN_ONSET_CORRECTION_S,
                          observed_intervals[0][0])
            green_intervals.append((g_start, seg_end))
        elif cur_state == 1:
            red_intervals.append((seg_start, seg_end))

        # Clip, merge, enforce minimum durations
        green_intervals = self.clip_intervals_to_observed(
            green_intervals, observed_intervals,
            min_len=self.MIN_GREEN_DURATION_S * 0.3,
        )
        green_intervals = self.merge_intervals(
            green_intervals, max_join_gap=self.GREEN_MERGE_GAP_S
        )
        red_intervals = self._rebuild_red_from_green(green_intervals, observed_intervals)

        return green_intervals, red_intervals

    def _refine_green_onsets_from_queue_discharge(
        self,
        green_intervals: List[Tuple[float, float]],
        phase_df: pd.DataFrame,
        observed_intervals: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Refine each green start using genuine queue-discharge events.

        Searches ALL approach vehicles (any distance from stop line) for
        vehicles that were definitively stopped (speed < STOP_SPEED_THRESH for
        at least MIN_STOP_DURATION_S) and then started moving.  These are
        queued vehicles responding to green — not vehicles merely approaching.

        For each green interval, finds the EARLIEST such discharge event
        within [gs - ONSET_SNAP_WINDOW_S, gs + 3 s].  Snaps the green start
        to that time minus ONSET_REACTION_S (pure reaction-time offset).
        Only applied when the result would be earlier than the current start.

        This is more effective than a tight-proximity scan for long-queue
        movements (NS/SN/NE) where queued vehicles sit 30–80 m from the
        stop line and would be outside any fixed proximity window.
        """
        approach = phase_df[phase_df["y"] < 0.0].copy()
        if approach.empty:
            return green_intervals

        # Collect (t_first_move) for every vehicle that was genuinely stopped
        discharge_events: List[float] = []

        for vid, g in approach.groupby("vid"):
            g = g.sort_values("t")
            t_arr = g["t"].values.astype(float)
            y_arr = g["y"].values.astype(float)
            if len(t_arr) < 4:
                continue

            dt  = np.diff(t_arr)
            dy  = np.diff(y_arr)
            spd = np.abs(dy / np.maximum(dt, 1e-6))

            is_stopped = spd < self.STOP_SPEED_THRESH

            i = 0
            while i < len(is_stopped):
                if not is_stopped[i]:
                    i += 1
                    continue

                stop_start = i
                while i < len(is_stopped) and is_stopped[i]:
                    i += 1
                stop_end = i

                if stop_end >= len(t_arr):
                    break

                stop_dur = t_arr[stop_end] - t_arr[stop_start]
                if stop_dur < self.MIN_STOP_DURATION_S:
                    continue

                # Find the first step after the stop where speed exceeds START_SPEED_THRESH
                for j in range(stop_end, min(stop_end + 10, len(spd))):
                    if spd[j] >= self.START_SPEED_THRESH:
                        discharge_events.append(float(t_arr[j + 1]))
                        break

        if not discharge_events:
            return green_intervals

        events_arr = np.array(sorted(discharge_events))
        refined = list(green_intervals)

        for idx, (gs, ge) in enumerate(refined):
            search_start = gs - self.ONSET_SNAP_WINDOW_S
            # Look for genuine queue-discharge events just before this green start
            # (+3 s buffer handles the case where behavioral detection is very close)
            candidates = events_arr[
                (events_arr >= search_start) & (events_arr <= gs + 3.0)
            ]
            if len(candidates) == 0:
                continue

            earliest = float(candidates[0])
            candidate_gs = max(earliest - self.ONSET_REACTION_S,
                               observed_intervals[0][0])
            if candidate_gs < gs:
                refined[idx] = (candidate_gs, ge)

        return self.merge_intervals(refined, max_join_gap=self.GREEN_MERGE_GAP_S)

    # ------------------------------------------------------------------
    # Main inference entry point (same signature as before)
    # ------------------------------------------------------------------

    def infer_signal_intervals(
        self, phase_df: pd.DataFrame
    ) -> Tuple[
        List[Tuple[float, float]],
        List[Tuple[float, float]],
        List[Tuple[float, float]],
        List[Tuple[float, float]],
        List[float],
    ]:
        """Signal phase inference.

        Returns (green_intervals, red_certain_intervals, red_uncertain_intervals,
                 observed_intervals, crossing_times).

        Algorithm:
          1. Compute observation windows.
          2. Extract stopline crossings.
          3. PRIMARY — behavioral detection: read phase state directly from
             the stopped/moving fraction of approach-zone vehicles per 1 s bin.
             Green onset is shifted back by GREEN_ONSET_CORRECTION_S (reaction
             time between actual green and first vehicle movement).
          4. FALLBACK — if behavioral detection returns no phases (too few
             approach vehicles): crossing-based histogram + tiling, or
             rate-based if global cycle is unavailable.
          5. Enforce no green where vehicles are definitively stopped.
          6. Split red into certain vs uncertain.
        """
        observed_intervals = self.compute_observed_intervals(phase_df)
        if not observed_intervals:
            return [], [], [], observed_intervals, []

        crossing_times = self.extract_crossing_times(phase_df)
        if not crossing_times:
            return [], [], [], observed_intervals, crossing_times

        # --- Primary: behavioral detection from approach-zone speed profile ---
        # Uses what vehicles are actually doing (stopped vs moving near the
        # stop line) rather than when crossings occur.  Reaction-time correction
        # (GREEN_ONSET_CORRECTION_S) is built in.  Falls back to the
        # crossing-based histogram when approach data is too sparse.
        green_intervals, red_intervals = self._infer_from_vehicle_behavior(
            phase_df, observed_intervals
        )

        if green_intervals:
            # Queue-discharge refinement: find the earliest vehicle that was
            # genuinely stopped then started moving, across ALL approach
            # depths.  This catches long-queue movements (NS/SN/NE) where
            # queued vehicles sit beyond any fixed proximity window.
            green_intervals = self._refine_green_onsets_from_queue_discharge(
                green_intervals, phase_df, observed_intervals
            )
            red_intervals = self._rebuild_red_from_green(green_intervals, observed_intervals)

        if not green_intervals:
            # --- Fallback: crossing-based histogram + tiling ---
            print("    [signal] behavioral detection found no phases — falling back to histogram")
            cycle = self._global_cycle
            use_cycle = (
                cycle is not None
                and len(crossing_times) >= self.MIN_CROSSINGS_FOR_PHASE
            )
            if use_cycle:
                g_start, g_end, r_start, r_end = self._find_phase_positions_in_cycle(
                    crossing_times, cycle
                )
                green_intervals, red_intervals = self._tile_phases(
                    cycle, g_start, g_end, observed_intervals, crossing_times
                )
            else:
                green_intervals, red_intervals = self._infer_rate_based(
                    phase_df, observed_intervals, crossing_times
                )

        # --- Close small intra-phase gaps ---
        # Within each observed window, if two green intervals (or two red
        # intervals) are separated by a gap smaller than PHASE_GAP_CLOSE_S,
        # merge them.  This prevents nonsensical 1–2 s "phase switches".
        PHASE_GAP_CLOSE_S = 3.0
        green_intervals = self.merge_intervals(green_intervals,
                                               max_join_gap=PHASE_GAP_CLOSE_S)

        # Ensure RED is respected: if there are stopped vehicles queued, 
        # truncate any overlapping GREEN intervals.
        is_present = self.compute_vehicle_presence(phase_df, observed_intervals)
        
        stopped_intervals = self._extract_stopped_intervals(phase_df)
        
        corrected_green = []
        for gs, ge in green_intervals:
            # Find if this green interval overlaps with any definitive stopped intervals
            current_gs = gs
            current_ge = ge
            valid_greens = [(current_gs, current_ge)]
            
            for ss, se in stopped_intervals:
                new_greens = []
                for c_gs, c_ge in valid_greens:
                    if ss < c_ge and se > c_gs:
                        # Overlap exists — decide how to handle it.
                        if ss <= c_gs and se >= c_ge:
                            # Green is entirely inside a stopped interval -> remove it entirely
                            pass
                        elif ss <= c_gs:
                            # Stopped interval predates or straddles the green onset.
                            # These are vehicles that were queued during red and started
                            # moving as green began — a normal pattern.  Do NOT push the
                            # reaction-time-corrected green onset forward; keep it intact.
                            new_greens.append((c_gs, c_ge))
                        elif se >= c_ge:
                            # Stopped interval covers the end of the green -> pull green end forward
                            if ss - c_gs >= self.MIN_GREEN_DURATION_S * 0.3:
                                new_greens.append((c_gs, ss))
                        else:
                            # Stopped interval is in the middle of green -> split green
                            if ss - c_gs >= self.MIN_GREEN_DURATION_S * 0.3:
                                new_greens.append((c_gs, ss))
                            if c_ge - se >= self.MIN_GREEN_DURATION_S * 0.3:
                                new_greens.append((se, c_ge))
                    else:
                        new_greens.append((c_gs, c_ge))
                valid_greens = new_greens
            
            corrected_green.extend(valid_greens)

        # Rebuild red as the complement of green within observed windows
        red_intervals = self._rebuild_red_from_green(
            corrected_green, observed_intervals
        )

        # ── Final spike filter ────────────────────────────────────────────
        # Step 1: remove green intervals shorter than MIN_PHASE_DISPLAY_S.
        # These are algorithm artefacts — noise transitions that slipped
        # through all the earlier filters.
        min_disp = self.MIN_PHASE_DISPLAY_S
        corrected_green = [
            (gs, ge) for gs, ge in corrected_green if (ge - gs) >= min_disp
        ]

        # Step 2: merge green intervals whose inter-green red gap is shorter
        # than SHORT_RED_MERGE_S.  Such short reds are physically impossible
        # (conflicting movements would need more time) and are caused by
        # arriving vehicles moving through the approach zone during red —
        # they look "moving" to the behavioural detector before they stop.
        # Example: vehicle arrives at t=454 while moving → false 20 s green
        #          → 9 s false red → real green.  Merging collapses this into
        #          one clean green interval.
        corrected_green = self.merge_intervals(
            corrected_green, max_join_gap=self.SHORT_RED_MERGE_S
        )

        # Step 3: crossing-corroboration filter.
        # Short green intervals with no stop-line crossings nearby are false
        # positives from vehicles arriving on the approach during red (they
        # look "moving" to the behavioural detector before they queue up).
        # Only intervals shorter than MAX_UNCORROBORATED_GREEN_S are eligible;
        # long greens are kept even if crossing data is sparse.
        if crossing_times:
            ct_arr = np.array(crossing_times, dtype=float)
            buf = self.GREEN_CROSSING_CORROBORATE_S
            max_uncorr = self.MAX_UNCORROBORATED_GREEN_S
            corroborated = []
            for gs, ge in corrected_green:
                if (ge - gs) >= max_uncorr:
                    corroborated.append((gs, ge))  # long green — keep unconditionally
                    continue
                # Short green: keep only if at least one crossing falls within
                # [gs - buf, ge + buf]
                nearby = ct_arr[(ct_arr >= gs - buf) & (ct_arr <= ge + buf)]
                if len(nearby) > 0:
                    corroborated.append((gs, ge))
                # else: no corroborating crossings → drop silently
            corrected_green = corroborated

        # Merge again in case removing uncorroborated greens left adjacencies
        corrected_green = self.merge_intervals(
            corrected_green, max_join_gap=self.GREEN_MERGE_GAP_S
        )

        # Step 4: rebuild red from the cleaned green, then drop short slivers
        red_intervals = self._rebuild_red_from_green(corrected_green, observed_intervals)
        red_intervals = [
            (rs, re) for rs, re in red_intervals if (re - rs) >= min_disp
        ]

        # Split red into certain vs uncertain
        red_certain: List[Tuple[float, float]] = []
        red_uncertain: List[Tuple[float, float]] = []
        for rs, re in red_intervals:
            if is_present(rs, re):
                red_certain.append((rs, re))
            else:
                red_uncertain.append((rs, re))

        return corrected_green, red_certain, red_uncertain, observed_intervals, crossing_times