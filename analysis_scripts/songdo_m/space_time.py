# space_time.py
"""Space-Time Diagram generation module.
"""

from __future__ import annotations

import os
import traceback
import warnings
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from .config import _THESIS_FONT, _save_fig_formats, log, dbg
from .signal_inference import SignalInferenceEngine
from .m_rules import infer_m_lane_from_sequence as infer_q_lane_from_sequence, get_lane_path_description


# ---------------------------------------------------------------------------
# SpaceTimeDiagram class
# ---------------------------------------------------------------------------

class SpaceTimeDiagram:
    """Generates space-time diagrams with signal phase inference.

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory DataFrame (one row per observation).
    out_path : str
        File path for the main diagram image.
    g_df : pd.DataFrame, optional
        Section geometry (polygon corners) for per-point section inference.
    allowed_vehicle_ids : iterable, optional
        If given, only these vehicle IDs are kept.
    title : str
        Plot title.
    x_col, y_col, time_col, vehicle_id_col : str
        Column names in *df*.
    vehicle_state_sequence : dict, optional
        ``{int(vid): [state_list]}`` for per-vehicle lane inference.
    general_dir : str, optional
        Directory for results table, headway analysis, and signal timing
        diagrams.  When ``None`` those outputs are skipped.
    """

    # ── Movement / arm constants (M intersection — 4 arms, 14 movements) ──
    # Arm 1 = North, Arm 2 = West, Arm 3 = South, Arm 4 = East
    # Approach sections (vehicles entering the intersection)
    _ARM1_IN = {"1_1", "1_2", "1_3"}   # North approach
    _ARM2_IN = {"2_2", "2_3"}           # West approach
    _ARM3_IN = {"3_4", "3_5", "3_6"}   # South approach
    _ARM4_IN = {"4_1", "4_2"}           # East approach

    # Departure sections (vehicles leaving the intersection)
    _TO_ARM1 = {"1_4", "1_5"}          # North departure
    _TO_ARM2 = {"2_1"}                  # West departure
    _TO_ARM3 = {"3_1", "3_2", "3_3"}   # South departure
    _TO_ARM4 = {"4_3", "4_4"}           # East departure

    _MOVEMENTS = [
        "NS", "NW", "NE", "NN",
        "SN", "SW", "SE", "SS",
        "WN", "WE", "WS",
        "EN", "EW", "ES",
    ]

    _MOV_ARM_MAP = {
        "NS": ("N", "S"), "NW": ("N", "W"), "NE": ("N", "E"), "NN": ("N", "N"),
        "SN": ("S", "N"), "SW": ("S", "W"), "SE": ("S", "E"), "SS": ("S", "S"),
        "WN": ("W", "N"), "WE": ("W", "E"), "WS": ("W", "S"),
        "EN": ("E", "N"), "EW": ("E", "W"), "ES": ("E", "S"),
    }

    # Signal phase grouping for inheritance fallback.
    _COMMON_GREEN_GROUPS = {
        "PHASE_N": ["NS", "NW", "NE", "NN"],
        "PHASE_S": ["SN", "SW", "SE", "SS"],
        "PHASE_W": ["WN", "WE", "WS"],
        "PHASE_E": ["EN", "EW", "ES"],
    }

    # No ground-truth phase data available for M; use empty list (inferred only).
    _OBSERVED_PHASES = []

    # Zoom windows not pre-calibrated for M — empty dict (no zoomed views generated).
    _ZOOM_WINDOWS_BY_MOVEMENT = {}

    _ZOOM_WINDOWS_BY_ARM_SIGNAL = {
        "ARM1_SIGNAL": [],
        "ARM2_SIGNAL": [],
        "ARM3_SIGNAL": [],
        "ARM4_SIGNAL": [],
    }

    # Trajectory quality-filter thresholds
    _TRAJ_MIN_POINTS = 20
    _TRAJ_MIN_SPAN_S = 6.0
    _TRAJ_MIN_Y_NEG_M = 60.0
    _TRAJ_MIN_Y_POS_M = 6.0
    _TRAJ_MIN_Y_RANGE_M = 55.0
    _TRAJ_MIN_Y_NEG_M_BY_ARM = {
        "ARM1_SIGNAL": 60.0, "ARM2_SIGNAL": 60.0,
        "ARM3_SIGNAL": 30.0, "ARM4_SIGNAL": 60.0,
    }
    _TRAJ_MIN_Y_POS_M_BY_ARM = {
        "ARM1_SIGNAL": 6.0, "ARM2_SIGNAL": 6.0,
        "ARM3_SIGNAL": 4.0, "ARM4_SIGNAL": 6.0,
    }
    _TRAJ_MIN_Y_RANGE_M_BY_ARM = {
        "ARM1_SIGNAL": 55.0, "ARM2_SIGNAL": 55.0,
        "ARM3_SIGNAL": 40.0, "ARM4_SIGNAL": 55.0,
    }
    _TRAJ_MAX_GAP_S = 2.0
    _TRAJ_BREAK_SPEED_MPS = 12.0
    _TRAJ_MAX_BREAK_RATIO = 0.25
    _CROSS_MAX_STEP_DT_S = 2.5
    _CROSS_MAX_STEP_SPEED_MPS = 20.0

    # Signal-timing diagram colours
    _PHASE_COLORS = {
        "green": "#4CAF50",
        "red_certain": "#E53935",
        "red_uncertain": "#42A5F5",
        "unknown": "#BDBDBD",
    }

    _MOV_TO_ARM_LABEL = {
        "NS": "North Arm", "NW": "North Arm", "NE": "North Arm", "NN": "North Arm",
        "SN": "South Arm", "SW": "South Arm", "SE": "South Arm", "SS": "South Arm",
        "WN": "West Arm",  "WE": "West Arm",  "WS": "West Arm",
        "EN": "East Arm",  "EW": "East Arm",  "ES": "East Arm",
    }

    # Movements observed as permissive in the UAV footage.
    _PERMISSIVE_MOVS = frozenset({"SS", "NN", "WS", "EN", "NW"})

    _DIR_GROUPS = {
        "N": ["NS", "NW", "NE", "NN"],
        "S": ["SN", "SW", "SE", "SS"],
        "W": ["WN", "WE", "WS"],
        "E": ["EN", "EW", "ES"],
    }
    _DIR_FULL_NAMES = {"N": "North", "S": "South", "W": "West", "E": "East"}

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        df: pd.DataFrame,
        out_path: str,
        g_df: pd.DataFrame = None,
        allowed_vehicle_ids=None,
        title: str = "Space-Time Diagram",
        x_col: str = "Ortho_X",
        y_col: str = "Ortho_Y",
        time_col: str = "time_s",
        vehicle_id_col: str = "Vehicle_ID",
        vehicle_state_sequence: dict = None,
        general_dir: str = None,
        headway_dir: str = None,
        vehicle_to_movement: dict = None,
    ):
        self.signal_engine = SignalInferenceEngine()

        self.out_path = out_path
        self.g_df = g_df
        self.title = title
        self.x_col = x_col
        self.y_col = y_col
        self.time_col = time_col
        self.vehicle_id_col = vehicle_id_col
        self.vehicle_state_sequence = vehicle_state_sequence
        self.general_dir = general_dir
        self.headway_dir = headway_dir
        self.vehicle_to_movement = vehicle_to_movement or {}

        # Will be populated by _filter_trajectories
        self.plot_df: pd.DataFrame = pd.DataFrame()
        self.processed_count: int = 0
        self.reject_counts: Counter = Counter()
        self.reject_rows: list = []

        # Pre-filter input dataframe
        self.df = df.copy() if df is not None else pd.DataFrame()
        if self.df.empty:
            return
        if allowed_vehicle_ids is not None:
            allowed_set = set(allowed_vehicle_ids)
            self.df = self.df[self.df[vehicle_id_col].isin(allowed_set)].copy()
        if self.df.empty:
            return

        # Build polygon lookup from g_df if needed
        section_cols = [c for c in ("Road_Section", "Section", "Section_ID") if c in self.df.columns]
        self.section_cols = section_cols

        self.poly_lookup = None
        if g_df is not None and not section_cols:
            import matplotlib.path as mplPath
            self.poly_lookup = []
            for _, row in g_df.iterrows():
                pts = [
                    (row['tlx'], row['tly']),
                    (row['trx'], row['try']),
                    (row['brx'], row['bry']),
                    (row['blx'], row['bly']),
                ]
                self.poly_lookup.append((str(row['Section']), mplPath.Path(pts)))

        # Run trajectory filtering
        self._filter_trajectories()

        # Attempt to recover rejected vehicles
        self._attempt_recovery()

        # Signal estimation must happen AFTER recovery so recovered
        # vehicles contribute to phase detection.
        if not self.plot_df.empty:
            self._estimate_global_cycle()
            self._precompute_arm_phases()

    # ------------------------------------------------------------------
    # Movement / arm helpers
    # ------------------------------------------------------------------

    def _movement_from_sections(self, origin, dest):
        # M intersection: 4 arms → 14 movements (including U-turns NN, SS)
        _arm_in = {
            "N": self._ARM1_IN, "W": self._ARM2_IN,
            "S": self._ARM3_IN, "E": self._ARM4_IN,
        }
        _arm_out = {
            "N": self._TO_ARM1, "W": self._TO_ARM2,
            "S": self._TO_ARM3, "E": self._TO_ARM4,
        }
        for o_dir, o_set in _arm_in.items():
            if origin in o_set:
                for d_dir, d_set in _arm_out.items():
                    if dest in d_set:
                        return f"{o_dir}{d_dir}"
        return "UNK"

    def _origin_arm_set(self, mov):
        _map = {
            "N": self._ARM1_IN, "W": self._ARM2_IN,
            "S": self._ARM3_IN, "E": self._ARM4_IN,
        }
        m = str(mov) if mov else ""
        if len(m) >= 2:
            o_dir = m[0]
            return _map.get(o_dir, set())
        return set()

    def _origin_arm_label_from_section(self, section_value):
        s = str(section_value) if section_value is not None else ""
        if s in self._ARM1_IN:
            return "ARM1_SIGNAL"
        if s in self._ARM2_IN:
            return "ARM2_SIGNAL"
        if s in self._ARM3_IN:
            return "ARM3_SIGNAL"
        if s in self._ARM4_IN:
            return "ARM4_SIGNAL"
        return None

    @staticmethod
    def _dir_to_mov(direction_str):
        """Extract movement code from direction like 'N to E L_1' -> 'NE'."""
        parts = direction_str.replace("L_", "").split()
        if len(parts) >= 3 and parts[1] == "to":
            return parts[0][0] + parts[2][0]
        return direction_str

    @staticmethod
    def _lane_suffix_alpha(lane_label) -> str:
        suffix = str(lane_label).replace("lane_", "")
        if suffix.isdigit():
            idx = int(suffix)
            if 1 <= idx <= 26:
                return chr(ord("A") + idx - 1)
        return suffix

    @classmethod
    def _display_lane_label(cls, mov, lane_label) -> str:
        return f"lane_{cls._lane_suffix_alpha(lane_label)}"

    @classmethod
    def _display_direction(cls, mov, lane_label) -> str:
        orig_arm, dest_arm = cls._MOV_ARM_MAP.get(mov, (str(mov)[:1], str(mov)[1:]))
        return f"{orig_arm} to {dest_arm} L_{cls._lane_suffix_alpha(lane_label)}"

    @classmethod
    def _display_lane_maps(cls, mov, lane_labels):
        ordered = sorted(
            [str(l) for l in lane_labels if str(l) != "unknown"],
            key=lambda x: (0, int(x.replace("lane_", ""))) if x.replace("lane_", "").isdigit() else (1, x),
        )
        suffixes = [chr(ord("A") + i) for i in range(len(ordered))]
        label_map = {raw: f"lane_{suffix}" for raw, suffix in zip(ordered, suffixes)}
        direction_map = {}
        orig_arm, dest_arm = cls._MOV_ARM_MAP.get(mov, (str(mov)[:1], str(mov)[1:]))
        for raw, suffix in zip(ordered, suffixes):
            direction_map[raw] = f"{orig_arm} to {dest_arm} L_{suffix}"
        return label_map, direction_map

    # ------------------------------------------------------------------
    # Trajectory filtering
    # ------------------------------------------------------------------

    def _filter_trajectories(self):
        """Filter and resample trajectories; populate self.plot_df."""
        df = self.df
        vehicle_id_col = self.vehicle_id_col
        time_col = self.time_col
        x_col = self.x_col
        y_col = self.y_col
        section_cols = self.section_cols
        poly_lookup = self.poly_lookup

        print(f"[INFO] Generating Space-Time Diagram for {len(df[vehicle_id_col].unique())} vehicles...")

        out_rows: list = []
        processed_count = 0
        reject_counts: Counter = Counter()
        reject_rows: list = []

        for vid, group in df.groupby(vehicle_id_col):
            g = group.sort_values(time_col).reset_index(drop=True)
            if len(g) < 5:
                reject_counts["too_few_points"] += 1
                reject_rows.append({"vid": int(vid), "movement": "UNK", "arm_signal": None, "reason": "too_few_points"})
                continue

            # Aggregate / resample trajectory
            try:
                g[time_col] = g[time_col].astype(float)
            except Exception:
                pass
            g = g.sort_values(time_col).reset_index(drop=True)

            aggreg_time = 0.1
            try:
                g['time_q'] = (np.round(g[time_col].astype(float) / aggreg_time) * aggreg_time).astype(float)
                agg_map = {x_col: 'median', y_col: 'median'}
                if section_cols:
                    sc = section_cols[0]

                    def mode_or_none(s):
                        s2 = s.dropna()
                        if s2.empty:
                            return None
                        return s2.mode().iloc[0]

                    agg_map[sc] = mode_or_none

                g = g.groupby('time_q', sort=True, as_index=False).agg(agg_map)
                g[time_col] = g['time_q']
                g = g.drop(columns=['time_q'])
            except Exception:
                g = g.sort_values(time_col).reset_index(drop=True)

            if len(g) < 5:
                reject_counts["too_few_points"] += 1
                reject_rows.append({"vid": int(vid), "movement": "UNK", "arm_signal": None, "reason": "too_few_points"})
                continue

            # Choose XY columns for distance computation
            if "Local_X" in g.columns and "Local_Y" in g.columns:
                xs = g["Local_X"].values.astype(float)
                ys = g["Local_Y"].values.astype(float)
                meters = True
            else:
                xs = g[x_col].values.astype(float)
                ys = g[y_col].values.astype(float)
                meters = False

            # cumulative traveled distance s(t)
            dx = np.diff(xs)
            dy = np.diff(ys)
            ds = np.sqrt(dx * dx + dy * dy)
            s = np.concatenate(([0.0], np.cumsum(ds)))
            if not meters:
                s = s * 0.03

            # infer origin / destination section
            N = min(8, len(g))
            origin_sec = None
            dest_sec = None
            if section_cols:
                sc = section_cols[0]
                try:
                    origin_sec = g.loc[:N - 1, sc].mode().iloc[0]
                    dest_sec = g.loc[len(g) - N:, sc].mode().iloc[0]
                except Exception:
                    origin_sec = None
                    dest_sec = None
            elif poly_lookup is not None:
                def vote_section_for_points(idx_slice):
                    votes = []
                    for i in idx_slice:
                        px, py = xs[i], ys[i]
                        for sid, path in poly_lookup:
                            if path.contains_point((px, py)):
                                votes.append(sid)
                                break
                    return votes

                origin_votes = vote_section_for_points(range(0, N))
                dest_votes = vote_section_for_points(range(len(g) - N, len(g)))
                if origin_votes:
                    origin_sec = Counter(origin_votes).most_common(1)[0][0]
                if dest_votes:
                    dest_sec = Counter(dest_votes).most_common(1)[0][0]

            mov = self._movement_from_sections(
                str(origin_sec) if origin_sec is not None else None,
                str(dest_sec) if dest_sec is not None else None,
            )
            origin_arm_label = self._origin_arm_label_from_section(origin_sec)

            # Fallback: use pipeline's movement classification when local
            # section-based classifier fails (UNK with no origin arm).
            if mov == "UNK" and origin_arm_label is None:
                pipeline_mov = self.vehicle_to_movement.get(int(vid))
                if pipeline_mov and pipeline_mov not in ("UNK", "UNASSIGNED"):
                    mov = pipeline_mov
                    origin_set = self._origin_arm_set(mov)
                    if origin_set:
                        # Derive ARM_SIGNAL label from movement directional code
                        # N→Arm1, W→Arm2, S→Arm3, E→Arm4
                        _dir_to_arm_num = {"N": "1", "W": "2", "S": "3", "E": "4"}
                        o_dir = str(mov)[0] if len(str(mov)) >= 2 else ""
                        o_arm_num = _dir_to_arm_num.get(o_dir, "")
                        origin_arm_label = f"ARM{o_arm_num}_SIGNAL" if o_arm_num else None

            if mov == "UNK":
                _arm_signal_map = {
                    "ARM1_SIGNAL": self._ARM1_IN, "ARM2_SIGNAL": self._ARM2_IN,
                    "ARM3_SIGNAL": self._ARM3_IN, "ARM4_SIGNAL": self._ARM4_IN,
                }
                if origin_arm_label in _arm_signal_map:
                    origin_set = _arm_signal_map[origin_arm_label]
                else:
                    reject_counts["silently_skipped_unk_no_origin"] += 1
                    reject_rows.append({"vid": int(vid), "movement": mov, "arm_signal": origin_arm_label, "reason": "silently_skipped_unk_no_origin"})
                    continue
            else:
                origin_set = self._origin_arm_set(mov)
                if not origin_set:
                    reject_counts["silently_skipped_no_origin_set"] += 1
                    reject_rows.append({"vid": int(vid), "movement": mov, "arm_signal": origin_arm_label, "reason": "silently_skipped_no_origin_set"})
                    continue

            # find last index still in origin_set
            stop_idx = -1
            if section_cols:
                sc = section_cols[0]
                mask = g[sc].astype(str).isin(origin_set).values
                idxs = np.where(mask)[0]
                if len(idxs) > 0:
                    stop_idx = int(idxs.max())
            elif poly_lookup is not None:
                last_in = None
                for i in range(len(g)):
                    px, py = xs[i], ys[i]
                    for sid, path in poly_lookup:
                        if sid in origin_set and path.contains_point((px, py)):
                            last_in = i
                            break
                if last_in is not None:
                    stop_idx = int(last_in)

            if stop_idx < 0 or stop_idx >= len(s):
                reject_counts["silently_skipped_unk_no_origin"] += 1
                reject_rows.append({"vid": int(vid), "movement": mov, "arm_signal": origin_arm_label, "reason": "silently_skipped_unk_no_origin"})
                continue

            s0 = float(s[stop_idx])
            y_vals = s - s0

            clip_min, clip_max = -200.0, 80.0
            keep_mask = (y_vals >= clip_min) & (y_vals <= clip_max)
            if not np.any(keep_mask):
                reject_counts["silently_skipped_unk_no_origin"] += 1
                reject_rows.append({"vid": int(vid), "movement": mov, "arm_signal": origin_arm_label, "reason": "silently_skipped_unk_no_origin"})
                continue

            times = g[time_col].values.astype(float)
            times_keep = times[keep_mask]
            y_keep = y_vals[keep_mask].astype(float)

            if len(times_keep) < self._TRAJ_MIN_POINTS:
                reject_counts["too_few_points"] += 1
                reject_rows.append({"vid": int(vid), "movement": mov, "arm_signal": origin_arm_label, "reason": "too_few_points",
                                    "_times": times_keep.copy(), "_y": y_keep.copy()})
                continue

            span_s = float(times_keep[-1] - times_keep[0])
            if span_s < self._TRAJ_MIN_SPAN_S:
                reject_counts["too_short_in_time"] += 1
                reject_rows.append({"vid": int(vid), "movement": mov, "arm_signal": origin_arm_label, "reason": "too_short_in_time", "span_s": span_s,
                                    "_times": times_keep.copy(), "_y": y_keep.copy()})
                continue

            y_min = float(np.min(y_keep))
            y_max = float(np.max(y_keep))
            y_range = float(y_max - y_min)
            min_y_neg_req = float(self._TRAJ_MIN_Y_NEG_M_BY_ARM.get(origin_arm_label, self._TRAJ_MIN_Y_NEG_M))
            min_y_pos_req = float(self._TRAJ_MIN_Y_POS_M_BY_ARM.get(origin_arm_label, self._TRAJ_MIN_Y_POS_M))
            min_y_range_req = float(self._TRAJ_MIN_Y_RANGE_M_BY_ARM.get(origin_arm_label, self._TRAJ_MIN_Y_RANGE_M))

            if y_min > -min_y_neg_req:
                reject_counts["no_upstream_approach"] += 1
                reject_rows.append({
                    "vid": int(vid), "movement": mov, "arm_signal": origin_arm_label,
                    "reason": "no_upstream_approach", "y_min": y_min, "min_y_neg_req": min_y_neg_req,
                    "_times": times_keep.copy(), "_y": y_keep.copy(),
                })
                continue
            if y_max < min_y_pos_req:
                reject_counts["no_downstream_departure"] += 1
                reject_rows.append({
                    "vid": int(vid), "movement": mov, "arm_signal": origin_arm_label,
                    "reason": "no_downstream_departure", "y_max": y_max, "min_y_pos_req": min_y_pos_req,
                    "_times": times_keep.copy(), "_y": y_keep.copy(),
                })
                continue
            if y_range < min_y_range_req:
                reject_counts["small_y_range"] += 1
                reject_rows.append({
                    "vid": int(vid), "movement": mov, "arm_signal": origin_arm_label,
                    "reason": "small_y_range", "y_range": y_range, "min_y_range_req": min_y_range_req,
                    "_times": times_keep.copy(), "_y": y_keep.copy(),
                })
                continue

            dt_keep = np.diff(times_keep)
            if len(dt_keep) == 0:
                reject_counts["single_point_after_clip"] += 1
                reject_rows.append({"vid": int(vid), "movement": mov, "arm_signal": origin_arm_label, "reason": "single_point_after_clip"})
                continue
            if float(np.max(dt_keep)) > self._TRAJ_MAX_GAP_S:
                reject_counts["large_internal_gap"] += 1
                reject_rows.append({
                    "vid": int(vid), "movement": mov, "arm_signal": origin_arm_label,
                    "reason": "large_internal_gap", "max_gap_s": float(np.max(dt_keep)),
                    "_times": times_keep.copy(), "_y": y_keep.copy(),
                })
                continue

            dy_keep = np.diff(y_keep)
            speed_keep = np.abs(dy_keep / np.maximum(dt_keep, 1e-6))
            break_ratio = float(np.mean((dt_keep <= 1e-3) | (speed_keep > self._TRAJ_BREAK_SPEED_MPS)))
            if break_ratio > self._TRAJ_MAX_BREAK_RATIO:
                reject_counts["too_many_breaks"] += 1
                reject_rows.append({
                    "vid": int(vid), "movement": mov, "arm_signal": origin_arm_label,
                    "reason": "too_many_breaks", "break_ratio": break_ratio,
                    "_times": times_keep.copy(), "_y": y_keep.copy(),
                })
                continue

            has_good_crossing = False
            for i in range(len(dt_keep)):
                y0 = y_keep[i]
                y1 = y_keep[i + 1]
                if y0 < 0.0 and y1 >= 0.0:
                    if dt_keep[i] <= self._CROSS_MAX_STEP_DT_S and speed_keep[i] <= self._CROSS_MAX_STEP_SPEED_MPS:
                        has_good_crossing = True
                        break
            if not has_good_crossing:
                reject_counts["no_clean_stopline_crossing"] += 1
                reject_rows.append({"vid": int(vid), "movement": mov, "arm_signal": origin_arm_label, "reason": "no_clean_stopline_crossing",
                                    "_times": times_keep.copy(), "_y": y_keep.copy()})
                continue

            # Determine per-lane label
            vid_lane = "unknown"
            if self.vehicle_state_sequence is not None:
                seq = self.vehicle_state_sequence.get(int(vid), [])
                vid_lane = infer_q_lane_from_sequence(seq, mov)

            for ti, yi in zip(times_keep, y_keep):
                out_rows.append({
                    'vid': vid,
                    't': ti,
                    'y': float(yi),
                    'movement': mov,
                    'arm_signal': origin_arm_label,
                    'lane': vid_lane,
                })

            processed_count += 1

        self.processed_count = processed_count
        self.reject_counts = reject_counts
        self.reject_rows = reject_rows

        if not out_rows:
            print("[WARNING] No valid trajectories found for space-time diagram.")
            return

        self.plot_df = pd.DataFrame(out_rows)

        dropped_count = int(sum(reject_counts.values()))
        print(
            f"[INFO] Trajectory quality filter kept {processed_count} vehicles and dropped "
            f"{dropped_count} incomplete/noisy trajectories."
        )
        if dropped_count > 0:
            print("[INFO] Drop reasons:", dict(reject_counts))

        # Debug counts
        try:
            m_counts = self.plot_df["movement"].value_counts(dropna=False).to_dict()
            a_counts = self.plot_df["arm_signal"].value_counts(dropna=False).to_dict()
            print("[INFO] Accepted trajectories by movement:", m_counts)
            print("[INFO] Accepted trajectories by arm signal:", a_counts)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Trajectory recovery (second pass over rejected vehicles)
    # ------------------------------------------------------------------

    _RECOVERABLE_REASONS = {
        "no_downstream_departure",
        "no_upstream_approach",
        "too_many_breaks",
        "too_short_in_time",
        "silently_skipped_unk_no_origin",
        "silently_skipped_no_origin_set",
        "no_clean_stopline_crossing",
        "large_internal_gap",
        "small_y_range",
    }

    def _attempt_recovery(self):
        """Second pass: try to recover rejected vehicles via trajectory
        completion, gap-filling, movement re-assignment, and — for vehicles
        that were skipped before space-time transform — full reprocessing.

        Modifies self.plot_df (appending recovered rows) and
        self.reject_rows / self.reject_counts (removing recovered).
        """
        from .trajectory_recovery import (
            extend_trajectory_forward,
            extend_trajectory_backward,
            fill_gaps_and_smooth,
            recheck_quality,
            infer_movement_from_road_sections,
        )

        if not self.reject_rows:
            return

        recovered_rows = []
        still_rejected = []
        recovered_count = 0
        recovered_by_reason: Counter = Counter()

        # ── Pass A: vehicles WITH cached space-time data ─────────
        needs_reprocess = []  # for pass B (silently_skipped without cache)

        for row in self.reject_rows:
            reason = row["reason"]
            vid = row["vid"]
            mov = row.get("movement", "UNK")
            arm = row.get("arm_signal")
            t_cached = row.get("_times")
            y_cached = row.get("_y")

            # Skip non-recoverable
            if reason not in self._RECOVERABLE_REASONS:
                still_rejected.append(row)
                continue

            # No cached data → queue for reprocessing (pass B)
            if t_cached is None or y_cached is None or len(t_cached) < 3:
                needs_reprocess.append(row)
                continue

            times = t_cached.astype(float)
            y_vals = y_cached.astype(float)

            # Arm-specific thresholds
            min_y_neg = float(self._TRAJ_MIN_Y_NEG_M_BY_ARM.get(arm, self._TRAJ_MIN_Y_NEG_M))
            min_y_pos = float(self._TRAJ_MIN_Y_POS_M_BY_ARM.get(arm, self._TRAJ_MIN_Y_POS_M))
            min_y_range = float(self._TRAJ_MIN_Y_RANGE_M_BY_ARM.get(arm, self._TRAJ_MIN_Y_RANGE_M))

            # Decide whether this reason gets relaxed quality checks.
            # Vehicles that execute full movements (proven by section
            # traversals) are accepted with relaxed thresholds.
            use_relaxed = reason in (
                "no_upstream_approach",
                "too_many_breaks",
                "too_short_in_time",
                "no_clean_stopline_crossing",
                "large_internal_gap",
            )

            # ── Apply recovery strategy based on reason ──────────

            if reason == "no_downstream_departure":
                times, y_vals = extend_trajectory_forward(
                    times, y_vals, target_y_pos=min_y_pos)

            elif reason == "no_upstream_approach":
                # Extend backward as far as possible
                times, y_vals = extend_trajectory_backward(
                    times, y_vals, target_y_neg=min_y_neg,
                    max_extend_s=40.0)

            elif reason == "too_many_breaks":
                # Smooth and fill gaps
                times, y_vals = fill_gaps_and_smooth(
                    times, y_vals,
                    break_speed_mps=self._TRAJ_BREAK_SPEED_MPS)

            elif reason in ("too_short_in_time", "small_y_range"):
                times, y_vals = extend_trajectory_backward(
                    times, y_vals, target_y_neg=min_y_neg,
                    max_extend_s=40.0)
                times, y_vals = extend_trajectory_forward(
                    times, y_vals, target_y_pos=min_y_pos)

            elif reason in ("no_clean_stopline_crossing", "large_internal_gap"):
                times, y_vals = fill_gaps_and_smooth(
                    times, y_vals,
                    break_speed_mps=self._TRAJ_BREAK_SPEED_MPS)

            # ── Re-check quality (relaxed for proven movements) ──
            if use_relaxed:
                # Relaxed: minimal thresholds — these vehicles have proven
                # movements from section traversals
                passed, fail_reason = recheck_quality(
                    times, y_vals,
                    min_points=5,
                    min_span_s=3.0,
                    min_y_neg=0.0,
                    min_y_pos=0.0,
                    min_y_range=0.0,
                    max_gap_s=self._TRAJ_MAX_GAP_S * 2.0,
                    max_break_ratio=1.0,          # accept any break ratio
                    break_speed_mps=50.0,          # effectively disabled
                    cross_max_dt=self._CROSS_MAX_STEP_DT_S,
                    cross_max_speed=self._CROSS_MAX_STEP_SPEED_MPS,
                    require_crossing=False,
                )
            else:
                passed, fail_reason = recheck_quality(
                    times, y_vals,
                    min_points=self._TRAJ_MIN_POINTS,
                    min_span_s=self._TRAJ_MIN_SPAN_S,
                    min_y_neg=min_y_neg,
                    min_y_pos=min_y_pos,
                    min_y_range=min_y_range,
                    max_gap_s=self._TRAJ_MAX_GAP_S,
                    max_break_ratio=self._TRAJ_MAX_BREAK_RATIO,
                    break_speed_mps=self._TRAJ_BREAK_SPEED_MPS,
                    cross_max_dt=self._CROSS_MAX_STEP_DT_S,
                    cross_max_speed=self._CROSS_MAX_STEP_SPEED_MPS,
                    require_crossing=True,
                )

            if not passed:
                still_rejected.append(row)
                continue

            # ── Recovery succeeded — assign movement + lane ────────
            # Re-infer movement from state sequence if still UNK
            from .m_rules import infer_m_movement_from_lane_sequence as infer_q_movement_from_lane_sequence
            vid_lane = "unknown"
            if self.vehicle_state_sequence is not None:
                seq = self.vehicle_state_sequence.get(int(vid), [])
                if mov == "UNK" and seq:
                    inferred = infer_q_movement_from_lane_sequence(seq)
                    if inferred != "UNASSIGNED":
                        mov = inferred
                        arm = self._origin_arm_label_from_section(
                            seq[0].rsplit("_", 1)[0] if seq else None
                        )
                vid_lane = infer_q_lane_from_sequence(seq, mov)

            for ti, yi in zip(times, y_vals):
                recovered_rows.append({
                    'vid': vid,
                    't': float(ti),
                    'y': float(yi),
                    'movement': mov,
                    'arm_signal': arm,
                    'lane': vid_lane,
                    'recovered': True,
                })

            recovered_count += 1
            recovered_by_reason[reason] += 1
            self.reject_counts[reason] -= 1

        # ── Pass B: reprocess vehicles without cached space-time ──
        # These were skipped before y-value computation (silently_skipped).
        # Re-derive movement from Road_Section, then run the full
        # space-time transform from XY coordinates.
        if needs_reprocess:
            rp_count = self._reprocess_skipped_vehicles(
                needs_reprocess, recovered_rows, still_rejected,
                recovered_by_reason,
            )
            recovered_count += rp_count

        # Update reject_rows
        self.reject_rows = still_rejected

        # Append recovered trajectories to plot_df
        if recovered_rows:
            rec_df = pd.DataFrame(recovered_rows)
            if self.plot_df is not None and not self.plot_df.empty:
                # Ensure 'recovered' column exists in original
                if 'recovered' not in self.plot_df.columns:
                    self.plot_df['recovered'] = False
                self.plot_df = pd.concat([self.plot_df, rec_df], ignore_index=True)
            else:
                self.plot_df = rec_df

            self.processed_count += recovered_count

        if recovered_count > 0:
            print(f"[RECOVERY] Recovered {recovered_count} vehicles: {dict(recovered_by_reason)}")
        else:
            print("[RECOVERY] No vehicles could be recovered.")

    def _reprocess_skipped_vehicles(
        self,
        needs_reprocess: list,
        recovered_rows: list,
        still_rejected: list,
        recovered_by_reason: Counter,
    ) -> int:
        """Re-derive space-time data from scratch for vehicles that were
        skipped before y-value computation (no cached _times/_y).

        These vehicles typically have valid Road_Section data but couldn't
        be mapped to an origin arm during the first pass.  We re-infer the
        movement from their section sequence, compute the cumulative
        distance, find the stopline, and apply relaxed quality checks.

        Returns the number of successfully recovered vehicles.
        """
        from .trajectory_recovery import (
            infer_movement_from_road_sections,
            extend_trajectory_forward,
            extend_trajectory_backward,
            fill_gaps_and_smooth,
        )

        time_col = self.time_col
        x_col = self.x_col
        y_col = self.y_col
        section_cols = self.section_cols
        poly_lookup = self.poly_lookup
        count = 0

        for row in needs_reprocess:
            vid = row["vid"]
            reason = row["reason"]

            vdata = self.df[self.df[self.vehicle_id_col] == vid]
            if vdata.empty or len(vdata) < 5:
                still_rejected.append(row)
                continue

            # Build Road_Section list (may be empty for NO_SECTION vehicles)
            road_secs = []
            section_lane_pairs = []
            if "Road_Section" in vdata.columns:
                road_secs = vdata["Road_Section"].dropna().astype(str).tolist()
                road_secs = [s for s in road_secs if s not in ("nan", "None", "")]
                # Build section+lane pairs for lane-aware inference
                if "Lane_Number" in vdata.columns:
                    for _, r in vdata.dropna(subset=["Road_Section"]).iterrows():
                        sec = str(r["Road_Section"])
                        if sec not in ("nan", "None", ""):
                            section_lane_pairs.append((sec, r.get("Lane_Number")))

            # If no road_section data, try polygon-based section detection
            if not road_secs and self.poly_lookup is not None:
                g = vdata.sort_values(self.time_col).reset_index(drop=True)
                xs = g[self.x_col].values.astype(float)
                ys = g[self.y_col].values.astype(float)
                for i in range(len(g)):
                    px, py = xs[i], ys[i]
                    for sid, path in self.poly_lookup:
                        if path.contains_point((px, py)):
                            road_secs.append(sid)
                            break
                road_secs = [s for s in road_secs if s not in ("nan", "None", "")]

            mov = infer_movement_from_road_sections(
                road_secs, section_lane_pairs=section_lane_pairs)
            if not mov:
                still_rejected.append(row)
                continue

            arm = self._origin_arm_label_from_section(None)
            origin_set = self._origin_arm_set(mov)
            if not origin_set:
                still_rejected.append(row)
                continue

            # Determine arm label from origin sections
            for s in road_secs:
                arm = self._origin_arm_label_from_section(s)
                if arm is not None:
                    break

            # Prepare trajectory data (same as _filter_trajectories)
            g = vdata.sort_values(time_col).reset_index(drop=True)
            try:
                g[time_col] = g[time_col].astype(float)
            except Exception:
                still_rejected.append(row)
                continue

            # Aggregate
            aggreg_time = 0.1
            try:
                g['time_q'] = (np.round(g[time_col].astype(float) / aggreg_time) * aggreg_time).astype(float)
                agg_map = {x_col: 'median', y_col: 'median'}
                if section_cols:
                    sc = section_cols[0]
                    def mode_or_none(s):
                        s2 = s.dropna()
                        return s2.mode().iloc[0] if not s2.empty else None
                    agg_map[sc] = mode_or_none
                g = g.groupby('time_q', sort=True, as_index=False).agg(agg_map)
                g[time_col] = g['time_q']
                g = g.drop(columns=['time_q'])
            except Exception:
                g = g.sort_values(time_col).reset_index(drop=True)

            if len(g) < 5:
                still_rejected.append(row)
                continue

            # XY for distance
            if "Local_X" in g.columns and "Local_Y" in g.columns:
                xs = g["Local_X"].values.astype(float)
                ys = g["Local_Y"].values.astype(float)
                meters = True
            else:
                xs = g[x_col].values.astype(float)
                ys = g[y_col].values.astype(float)
                meters = False

            dx = np.diff(xs)
            dy = np.diff(ys)
            ds = np.sqrt(dx * dx + dy * dy)
            s = np.concatenate(([0.0], np.cumsum(ds)))
            if not meters:
                s = s * 0.03

            # Find stopline (last point in origin sections)
            stop_idx = -1
            if section_cols:
                sc = section_cols[0]
                mask = g[sc].astype(str).isin(origin_set).values
                idxs = np.where(mask)[0]
                if len(idxs) > 0:
                    stop_idx = int(idxs.max())
            elif poly_lookup is not None:
                last_in = None
                for i in range(len(g)):
                    px, py = xs[i], ys[i]
                    for sid, path in poly_lookup:
                        if sid in origin_set and path.contains_point((px, py)):
                            last_in = i
                            break
                if last_in is not None:
                    stop_idx = int(last_in)

            if stop_idx < 0 or stop_idx >= len(s):
                still_rejected.append(row)
                continue

            s0 = float(s[stop_idx])
            y_vals = s - s0

            clip_min, clip_max = -200.0, 80.0
            keep_mask = (y_vals >= clip_min) & (y_vals <= clip_max)
            if not np.any(keep_mask):
                still_rejected.append(row)
                continue

            times = g[time_col].values.astype(float)
            times_keep = times[keep_mask]
            y_keep = y_vals[keep_mask].astype(float)

            if len(times_keep) < 5:
                still_rejected.append(row)
                continue

            # Smooth if needed
            times_keep, y_keep = fill_gaps_and_smooth(
                times_keep, y_keep,
                break_speed_mps=self._TRAJ_BREAK_SPEED_MPS)

            # Extend if needed
            min_y_neg = float(self._TRAJ_MIN_Y_NEG_M_BY_ARM.get(arm, self._TRAJ_MIN_Y_NEG_M))
            min_y_pos = float(self._TRAJ_MIN_Y_POS_M_BY_ARM.get(arm, self._TRAJ_MIN_Y_POS_M))
            times_keep, y_keep = extend_trajectory_backward(
                times_keep, y_keep, target_y_neg=min_y_neg, max_extend_s=40.0)
            times_keep, y_keep = extend_trajectory_forward(
                times_keep, y_keep, target_y_pos=min_y_pos)

            # Relaxed quality check — these vehicles have proven movements
            if len(times_keep) < 5:
                still_rejected.append(row)
                continue

            span_s = float(times_keep[-1] - times_keep[0])
            if span_s < 2.0:
                still_rejected.append(row)
                continue

            # Accept: add to recovered rows
            vid_lane = "unknown"
            if self.vehicle_state_sequence is not None:
                seq = self.vehicle_state_sequence.get(int(vid), [])
                vid_lane = infer_q_lane_from_sequence(seq, mov)

            for ti, yi in zip(times_keep, y_keep):
                recovered_rows.append({
                    'vid': vid,
                    't': float(ti),
                    'y': float(yi),
                    'movement': mov,
                    'arm_signal': arm,
                    'lane': vid_lane,
                    'recovered': True,
                })

            count += 1
            recovered_by_reason[reason] += 1
            self.reject_counts[reason] -= 1

        return count

    # ------------------------------------------------------------------
    # Global cycle estimation
    # ------------------------------------------------------------------

    def _estimate_global_cycle(self):
        """Pool crossings from the two strongest movements to estimate a
        single global cycle length shared by the whole intersection."""
        if self.plot_df.empty:
            return

        # Collect crossings per movement
        mov_crossings = {}
        for mov in self._MOVEMENTS:
            sub = self.plot_df[self.plot_df["movement"] == mov]
            if sub.empty:
                continue
            ct = self.signal_engine.extract_crossing_times(sub)
            if ct:
                mov_crossings[mov] = ct

        if not mov_crossings:
            return

        # Pool crossings from the two movements with the most data
        sorted_movs = sorted(mov_crossings.keys(), key=lambda m: len(mov_crossings[m]), reverse=True)
        pooled = []
        for mov in sorted_movs[:2]:
            pooled.extend(mov_crossings[mov])
        pooled.sort()

        cycle = self.signal_engine.set_global_cycle(pooled)
        if cycle is not None:
            print(f"[SIGNAL] Global cycle length estimated: {cycle:.0f}s "
                  f"(from {sorted_movs[0]}+{sorted_movs[1] if len(sorted_movs) > 1 else 'N/A'}, "
                  f"{len(pooled)} crossings pooled)")
        else:
            print("[SIGNAL] Could not estimate global cycle length — using rate-based fallback")

    # ------------------------------------------------------------------
    # Pre-computed arm-level signal phases
    # ------------------------------------------------------------------

    def _precompute_arm_phases(self):
        """Compute signal phases per MOVEMENT.

        Movements in the same ``_COMMON_GREEN_GROUPS`` entry have
        identical green/red schedules, so their crossings are **pooled**
        for a single, stronger detection and the result is shared.

        Single-movement groups detect independently.  Any movement that
        still fails to detect (< 2 green intervals) inherits from the
        strongest movement in its group.

        Stores results in ``self._movement_phases``: a dict mapping
        movement name (e.g. ``"NS"``) to the 5-tuple returned by
        ``infer_signal_intervals``.
        """
        self._movement_phases: dict = {}
        if self.plot_df.empty:
            return

        for group_label, movs in self._COMMON_GREEN_GROUPS.items():
            # Collect trajectories for all movements in this group
            group_movs_present = [
                m for m in movs
                if not self.plot_df[self.plot_df["movement"] == m].empty
            ]
            if not group_movs_present:
                continue

            if len(group_movs_present) > 1:
                # Pool crossings from all movements with identical
                # green patterns for a stronger detection.
                sub = self.plot_df[
                    self.plot_df["movement"].isin(group_movs_present)
                ]
                phases = self.signal_engine.infer_signal_intervals(sub)
                green_ints, red_cert, red_uncert, observed, crossings = phases

                if len(green_ints) >= 2:
                    for m in group_movs_present:
                        self._movement_phases[m] = phases
                    print(
                        f"[SIGNAL] {group_label} ({'+'.join(group_movs_present)}): "
                        f"{len(crossings)} pooled crossings, "
                        f"{len(green_ints)} green, {len(red_cert)} red_certain, "
                        f"{len(red_uncert)} red_uncertain"
                    )
                else:
                    # Pooled detection too weak — try individually
                    for m in group_movs_present:
                        sub_m = self.plot_df[self.plot_df["movement"] == m]
                        p = self.signal_engine.infer_signal_intervals(sub_m)
                        if len(p[0]) >= 2:
                            self._movement_phases[m] = p
                            print(
                                f"[SIGNAL] {m}: {len(p[4])} crossings, "
                                f"{len(p[0])} green (individual fallback)"
                            )
                        else:
                            print(
                                f"[SIGNAL] {m}: {len(p[4])} crossings — "
                                f"too few for reliable phase detection"
                            )
            else:
                # Single movement — detect independently
                m = group_movs_present[0]
                sub = self.plot_df[self.plot_df["movement"] == m]
                phases = self.signal_engine.infer_signal_intervals(sub)
                green_ints, red_cert, red_uncert, observed, crossings = phases

                if len(green_ints) >= 2:
                    self._movement_phases[m] = phases
                    print(
                        f"[SIGNAL] {m}: {len(crossings)} crossings, "
                        f"{len(green_ints)} green, {len(red_cert)} red_certain, "
                        f"{len(red_uncert)} red_uncertain"
                    )
                else:
                    print(
                        f"[SIGNAL] {m}: {len(crossings)} crossings — "
                        f"too few for reliable phase detection"
                    )

        # Fallback: movements without phases inherit from the strongest
        # movement in their group
        for group_label, movs in self._COMMON_GREEN_GROUPS.items():
            best_mov = None
            best_count = 0
            for m in movs:
                if m in self._movement_phases:
                    phases = self._movement_phases[m]
                    c = len(phases[4])  # crossing count
                    if c > best_count:
                        best_count = c
                        best_mov = m

            if best_mov is None:
                continue

            for m in movs:
                if m not in self._movement_phases:
                    has_data = not self.plot_df[
                        self.plot_df["movement"] == m
                    ].empty
                    if has_data:
                        self._movement_phases[m] = self._movement_phases[best_mov]
                        print(
                            f"[SIGNAL] {m}: inheriting phases from {best_mov} "
                            f"({best_count} crossings)"
                        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _write_diagnostics(self):
        """Persist plot_df, diagnostics CSV, and rejected-trajectory CSV."""
        run_dir = os.path.dirname(self.out_path)

        # Save plot dataframe
        try:
            self.plot_df.to_csv(os.path.join(run_dir, 'space_time_plot_df.csv'), index=False)
        except Exception:
            pass

        # Jump / duplicate-time diagnostics
        diag_rows: list = []
        report_speed_thresh = 30.0
        report_dy_thresh = 5.0
        for vid, g in self.plot_df.groupby('vid'):
            gg = g.sort_values('t').reset_index(drop=True)
            tvals = gg['t'].values.astype(float)
            yvals = gg['y'].values.astype(float)
            if len(tvals) < 2:
                continue
            unique, counts = np.unique(tvals, return_counts=True)
            dup_times = unique[counts > 1]
            for dtval in dup_times:
                idxs = np.where(tvals == dtval)[0]
                ys = yvals[idxs]
                dy_range = float(np.max(ys) - np.min(ys))
                if dy_range >= report_dy_thresh:
                    diag_rows.append({
                        'vid': int(vid), 'issue': 'duplicate_time',
                        't': float(dtval), 'dt': 0.0, 'dy': dy_range,
                        'implied_speed': np.inf,
                    })
            dt = np.diff(tvals)
            dy = np.diff(yvals)
            implied_speed = np.abs(dy / np.maximum(dt, 1e-6))
            bad_idx = np.where((implied_speed >= report_speed_thresh) & (np.abs(dy) >= report_dy_thresh))[0]
            for i in bad_idx:
                diag_rows.append({
                    'vid': int(vid), 'issue': 'high_speed',
                    't': float(tvals[i]), 'dt': float(dt[i]),
                    'dy': float(dy[i]), 'implied_speed': float(implied_speed[i]),
                })

        try:
            diag_path = os.path.join(run_dir, 'space_time_diagnostics.csv')
            if diag_rows:
                pd.DataFrame(diag_rows).to_csv(diag_path, index=False)
            else:
                pd.DataFrame(columns=['vid', 'issue', 't', 'dt', 'dy', 'implied_speed']).to_csv(diag_path, index=False)
        except Exception:
            pass

        # Save rejected trajectories
        try:
            rej_path = os.path.join(run_dir, "space_time_rejected_trajectories.csv")
            if self.reject_rows:
                pd.DataFrame(self.reject_rows).to_csv(rej_path, index=False)
            else:
                pd.DataFrame(columns=["vid", "movement", "arm_signal", "reason"]).to_csv(rej_path, index=False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Trajectory interpolation / smoothing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_local_speed(t_arr, y_arr, idx, n_pts=5, direction="backward"):
        """Estimate average speed from *n_pts* points around *idx*.

        Parameters
        ----------
        direction : "backward" — use points *before* idx (for forward
                    extrapolation at gap start).
                    "forward" — use points *after* idx (for backward
                    extrapolation at gap end).
        Returns speed in m/s (signed, positive = increasing y).
        """
        if direction == "backward":
            lo = max(0, idx - n_pts)
            hi = idx + 1
        else:
            lo = idx
            hi = min(len(t_arr), idx + n_pts + 1)

        seg_t = t_arr[lo:hi]
        seg_y = y_arr[lo:hi]
        if len(seg_t) < 2:
            return 0.0
        dt_total = seg_t[-1] - seg_t[0]
        if dt_total < 1e-6:
            return 0.0
        return float((seg_y[-1] - seg_y[0]) / dt_total)

    @staticmethod
    def _speed_based_interp(t_arr, y_arr, i, gap, speed_left, speed_right,
                            step=0.1):
        """Fill a temporal gap using speed-based extrapolation from both
        ends, blended with a cubic-Hermite spline.

        The left endpoint extrapolates forward at *speed_left* and the
        right endpoint extrapolates backward at *speed_right*.  A
        blending weight smoothly transitions from left-dominant near the
        start of the gap to right-dominant near the end.
        """
        n_fill = max(1, int(gap / step))
        fill_t = []
        fill_y = []
        t0, y0 = t_arr[i], y_arr[i]
        t1, y1 = t_arr[i + 1], y_arr[i + 1]
        for j in range(1, n_fill):
            frac = j / n_fill
            t_new = t0 + gap * frac
            # Hermite basis: smooth blend that matches position and
            # velocity at both endpoints.
            h00 = (1 + 2 * frac) * (1 - frac) ** 2
            h10 = frac * (1 - frac) ** 2
            h01 = frac ** 2 * (3 - 2 * frac)
            h11 = frac ** 2 * (frac - 1)
            y_new = (h00 * y0 + h10 * gap * speed_left
                     + h01 * y1 + h11 * gap * speed_right)
            fill_t.append(t_new)
            fill_y.append(y_new)
        return fill_t, fill_y

    def _plot_trajectories(self, ax_local, sub_df: pd.DataFrame):
        traj_color = "mediumblue"
        time_eps = 1e-3
        break_speed_thresh = 15.0   # only break trajectory if speed > this
        min_seg_points = 6
        min_seg_span_s = 1.0
        min_seg_range_m = 8.0
        max_interp_gap_s = 8.0      # interpolate gaps up to 8s
        max_interp_speed = 15.0     # max plausible speed for interpolation
        speed_est_pts = 5            # points used to estimate local speed
        savgol_window = 7            # Savitzky-Golay smoothing window
        savgol_poly = 2              # Savitzky-Golay polynomial order

        def _plot_if_valid(st, sy):
            if len(st) < min_seg_points:
                return
            if float(st[-1] - st[0]) < min_seg_span_s:
                return
            if float(np.max(sy) - np.min(sy)) < min_seg_range_m:
                return
            ax_local.plot(st, sy, color=traj_color, linewidth=0.6, alpha=0.5)

        for _, g in sub_df.groupby("vid"):
            gg = g.sort_values("t")
            gg = gg.groupby("t", sort=True, as_index=False).agg({"y": "median"})
            t_arr = gg["t"].values.astype(float)
            y_arr = gg["y"].values.astype(float)

            if len(t_arr) < 2:
                continue

            # --- Speed-based interpolation for temporal gaps ---
            new_t = list(t_arr)
            new_y = list(y_arr)
            insert_offset = 0
            for i in range(len(t_arr) - 1):
                gap = t_arr[i + 1] - t_arr[i]
                if gap <= time_eps or gap > max_interp_gap_s:
                    continue

                # Estimate speed at both sides of the gap
                spd_left = self._estimate_local_speed(
                    t_arr, y_arr, i, speed_est_pts, "backward")
                spd_right = self._estimate_local_speed(
                    t_arr, y_arr, i + 1, speed_est_pts, "forward")

                # Sanity: implied linear speed across the gap
                implied = abs(y_arr[i + 1] - y_arr[i]) / gap
                if implied > max_interp_speed:
                    continue  # gap will cause a break

                # For small gaps (≤ 3s), simple linear is fine
                if gap <= 3.0:
                    n_fill = max(1, int(gap / 0.1))
                    for j in range(1, n_fill):
                        frac = j / n_fill
                        ins_idx = (i + 1) + insert_offset
                        new_t.insert(ins_idx, t_arr[i] + gap * frac)
                        new_y.insert(ins_idx,
                                     y_arr[i] + (y_arr[i + 1] - y_arr[i]) * frac)
                        insert_offset += 1
                else:
                    # Larger gaps: Hermite spline using local speeds
                    fill_t, fill_y = self._speed_based_interp(
                        t_arr, y_arr, i, gap, spd_left, spd_right)
                    for ft, fy in zip(fill_t, fill_y):
                        ins_idx = (i + 1) + insert_offset
                        new_t.insert(ins_idx, ft)
                        new_y.insert(ins_idx, fy)
                        insert_offset += 1

            t_arr = np.array(new_t)
            y_arr = np.array(new_y)

            # --- Savitzky-Golay smoothing ---
            if len(y_arr) >= savgol_window:
                try:
                    from scipy.signal import savgol_filter
                    y_arr = savgol_filter(y_arr, savgol_window, savgol_poly)
                except ImportError:
                    # Fallback: rolling median
                    y_arr = pd.Series(y_arr).rolling(
                        window=5, center=True, min_periods=1
                    ).median().values

            # --- Detect remaining breaks (only real problems) ---
            dt = np.diff(t_arr)
            dy = np.diff(y_arr)
            implied_speed = np.abs(dy / np.maximum(dt, 1e-6))

            bad_dt = np.where(dt <= time_eps)[0]
            bad_speed = np.where(implied_speed > break_speed_thresh)[0]
            large_gaps = np.where(dt > max_interp_gap_s)[0]
            breaks = np.unique(np.concatenate((bad_dt, bad_speed, large_gaps)))

            if breaks.size == 0:
                _plot_if_valid(t_arr, y_arr)
            else:
                split_idx = (breaks + 1).tolist()
                seg_t = np.split(t_arr, split_idx)
                seg_y = np.split(y_arr, split_idx)
                for st, sy in zip(seg_t, seg_y):
                    _plot_if_valid(st, sy)

    def _draw_phase_bars(
        self,
        ax_local,
        observed_intervals: List[Tuple[float, float]],
        red_certain_intervals: List[Tuple[float, float]],
        red_uncertain_intervals: List[Tuple[float, float]],
        green_intervals: List[Tuple[float, float]],
    ):
        # Draw red bars (both certain and uncertain shown as red)
        for s, e in red_certain_intervals:
            ax_local.hlines(0.0, s, e, colors=self._PHASE_COLORS["red_certain"], linewidth=8)
        for s, e in red_uncertain_intervals:
            ax_local.hlines(0.0, s, e, colors=self._PHASE_COLORS["red_certain"], linewidth=8)
        # Draw green bars on top
        for s, e in green_intervals:
            ax_local.hlines(0.0, s, e, colors=self._PHASE_COLORS["green"], linewidth=8)

        # Clean legend: only Green and Red
        from matplotlib.patches import Patch as _Patch
        legend_patches = [
            _Patch(facecolor=self._PHASE_COLORS["green"], label="Green"),
            _Patch(facecolor=self._PHASE_COLORS["red_certain"], label="Red"),
        ]
        existing_handles, existing_labels = ax_local.get_legend_handles_labels()
        if "Green (crossing)" not in existing_labels:
            ax_local.legend(
                handles=legend_patches, loc="upper right",
                fontsize=8, framealpha=0.8,
            )

    @staticmethod
    def _interval_duration(intervals: List[Tuple[float, float]]) -> float:
        return float(sum(max(0.0, float(b) - float(a)) for a, b in intervals))

    @staticmethod
    def _clip_intervals_to_windows(
        intervals: List[Tuple[float, float]],
        windows: List[Tuple[float, float]],
        min_len: float = 0.0,
    ) -> List[Tuple[float, float]]:
        clipped = []
        for s, e in intervals:
            for ws, we in windows:
                cs = max(float(s), float(ws))
                ce = min(float(e), float(we))
                if ce - cs >= min_len:
                    clipped.append((cs, ce))
        return clipped

    @staticmethod
    def _crossing_supported_green_intervals(
        crossing_times,
        observed_windows: List[Tuple[float, float]],
        max_gap_s: float = 10.0,
        pre_buffer_s: float = 4.0,
        post_buffer_s: float = 4.0,
        min_unique_crossings: int = 2,
    ) -> List[Tuple[float, float]]:
        """Build green evidence windows from dense stop-line crossing clusters."""
        if crossing_times is None or not observed_windows:
            return []

        ct = np.asarray(crossing_times, dtype=float)
        ct = ct[np.isfinite(ct)]
        if len(ct) == 0:
            return []

        # Remove duplicate interpolated crossings at the same instant.
        ct = np.unique(np.round(np.sort(ct), 3))
        if len(ct) < min_unique_crossings:
            return []

        clusters = []
        start = prev = float(ct[0])
        count = 1
        for val in ct[1:]:
            cur = float(val)
            if cur - prev <= max_gap_s:
                prev = cur
                count += 1
            else:
                if count >= min_unique_crossings:
                    clusters.append((start - pre_buffer_s, prev + post_buffer_s))
                start = prev = cur
                count = 1
        if count >= min_unique_crossings:
            clusters.append((start - pre_buffer_s, prev + post_buffer_s))

        return SpaceTimeDiagram._clip_intervals_to_windows(clusters, observed_windows, min_len=0.1)

    def _normalise_display_phases(
        self,
        mov: str,
        green_ints: List[Tuple[float, float]],
        red_cert_ints: List[Tuple[float, float]],
        red_uncert_ints: List[Tuple[float, float]],
        observed_ints: List[Tuple[float, float]],
        crossing_times,
    ) -> Tuple[
        List[Tuple[float, float]],
        List[Tuple[float, float]],
        List[Tuple[float, float]],
        List[Tuple[float, float]],
    ]:
        """Clean signal phases for display/results without changing raw inference.

        This removes extrapolated negative-time phases and short/no-evidence
        green spikes. Permissive movements are shown as green for the observed
        windows instead of being interpreted as protected red/green cycles.
        """
        observed_clean = [
            (max(0.0, float(s)), float(e))
            for s, e in observed_ints
            if float(e) > max(0.0, float(s))
        ]
        if not observed_clean:
            return [], [], [], []

        if mov in self._PERMISSIVE_MOVS:
            return observed_clean, [], [], observed_clean

        min_green = max(10.0, float(getattr(self.signal_engine, "MIN_PHASE_DISPLAY_S", 5.0)))
        green_clean = self._clip_intervals_to_windows(green_ints, observed_clean, min_len=min_green)

        ct_arr = np.asarray(crossing_times, dtype=float)
        if len(ct_arr) == 0:
            green_clean = []
        elif green_clean:
            evidence_buf = 1.0
            green_clean = [
                (gs, ge)
                for gs, ge in green_clean
                if np.any((ct_arr >= gs - evidence_buf) & (ct_arr <= ge + evidence_buf))
            ]

        crossing_green = self._crossing_supported_green_intervals(
            crossing_times, observed_clean,
        )
        if crossing_green:
            green_clean = green_clean + crossing_green

        if green_clean:
            max_red_sliver = float(getattr(self.signal_engine, "SHORT_RED_MERGE_S", 15.0))
            green_clean = self.signal_engine.merge_intervals(
                green_clean, max_join_gap=max_red_sliver,
            )

        red_rebuilt = self.signal_engine._rebuild_red_from_green(green_clean, observed_clean)
        red_rebuilt = [
            (rs, re)
            for rs, re in red_rebuilt
            if (re - rs) >= min_green
        ]
        return green_clean, red_rebuilt, [], observed_clean

    # ------------------------------------------------------------------
    # Start-up lost time
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_startup_delays(
        green_ints: List[Tuple[float, float]],
        crossing_times,
    ) -> List[Tuple[float, float, float]]:
        """Return [(t_green_start, t_first_crossing, delay_s), ...] per green cycle.

        Only cycles where a vehicle crosses within the green window are included.
        A sanity cap of 30 s is applied — delays beyond that are artefacts.
        """
        sorted_ct = np.sort(np.asarray(crossing_times, dtype=float))
        results = []
        for t0, t1 in sorted(green_ints, key=lambda x: x[0]):
            lo = int(np.searchsorted(sorted_ct, float(t0), side="left"))
            hi = int(np.searchsorted(sorted_ct, float(t1), side="right"))
            if lo < hi:
                t_first = float(sorted_ct[lo])
                delay = t_first - float(t0)
                if 0.0 <= delay <= 30.0:
                    results.append((float(t0), t_first, delay))
        return results

    def _annotate_startup_delays(
        self,
        ax,
        startup_delays: List[Tuple[float, float, float]],
    ):
        """Draw start-up lost time dimension lines on the space-time diagram.

        A bracket is drawn at y = +6 m (just past the stop line) from the
        green onset to the first vehicle departure, with a Δ_L label above.
        """
        if not startup_delays:
            return

        ann_y = 6.0          # metres above stop line
        tick_h = 1.2         # vertical tick half-height
        color = "#E65100"    # deep orange — distinct from green/red phase colours

        for t_green, t_first, delay in startup_delays:
            # Vertical ticks at each boundary
            ax.plot([t_green, t_green], [ann_y - tick_h, ann_y + tick_h],
                    color=color, linewidth=1.0, zorder=6)
            ax.plot([t_first, t_first], [ann_y - tick_h, ann_y + tick_h],
                    color=color, linewidth=1.0, zorder=6)
            # Horizontal dimension line with arrowheads
            ax.annotate(
                "", xy=(t_first, ann_y), xytext=(t_green, ann_y),
                arrowprops=dict(arrowstyle="<->", color=color,
                                lw=1.2, mutation_scale=6),
                zorder=6,
            )
            # Label centred above the bracket
            mid_t = 0.5 * (t_green + t_first)
            ax.text(
                mid_t, ann_y + tick_h + 0.4,
                f"$\\Delta_L$={delay:.1f}s",
                ha="center", va="bottom",
                fontsize=7, color=color, fontweight="bold", zorder=7,
            )

    def _report_startup_delays(
        self,
        movement: str,
        startup_delays: List[Tuple[float, float, float]],
    ):
        """Print per-cycle start-up lost time to the console and append to CSV."""
        if not startup_delays:
            print(f"  [{movement}] Start-up lost time: no cycles with data")
            return

        delays = [d for _, _, d in startup_delays]
        print(f"\n  === Start-up lost time: {movement} ===")
        for i, (t_g, t_f, d) in enumerate(startup_delays, 1):
            print(f"    Cycle {i:2d}: green onset {t_g:.1f}s → "
                  f"first departure {t_f:.1f}s   Δ_L = {d:.2f}s")
        print(f"    Mean Δ_L = {np.mean(delays):.2f}s  "
              f"(σ={np.std(delays):.2f}s, n={len(delays)} cycles)")

        # Append to a shared CSV in the signal_timing_diagrams folder
        if self.general_dir:
            import csv as _csv
            csv_path = os.path.join(
                self.general_dir, "signal_timing_diagrams", "startup_lost_time.csv"
            )
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                w = _csv.writer(f)
                if write_header:
                    w.writerow(["movement", "cycle", "t_green_start_s",
                                "t_first_crossing_s", "delay_s"])
                for i, (t_g, t_f, d) in enumerate(startup_delays, 1):
                    w.writerow([movement, i, f"{t_g:.3f}", f"{t_f:.3f}", f"{d:.3f}"])

    @staticmethod
    def _inside_any_interval(x: float, intervals: List[Tuple[float, float]]) -> bool:
        xv = float(x)
        for a, b in intervals:
            if float(a) <= xv <= float(b):
                return True
        return False

    def _audit_green_inference(self, label: str, phase_df: pd.DataFrame) -> dict:
        if phase_df is None or phase_df.empty:
            return {
                "label": label,
                "vehicles": 0, "crossings": 0, "greens": 0, "reds": 0,
                "observed_windows": 0, "obs_duration_s": 0.0,
                "green_duration_s": 0.0, "red_duration_s": 0.0,
                "cross_in_green": 0, "cross_in_red": 0, "cross_unassigned": 0,
                "pct_cross_in_green": np.nan, "pct_cross_in_red": np.nan,
                "green_cross_rate": np.nan, "red_cross_rate": np.nan,
                "green_to_red_rate_ratio": np.nan, "has_full_cycle": 0,
            }

        greens, reds_cert, reds_uncert, observed, crossings = self.signal_engine.infer_signal_intervals(phase_df)
        reds = reds_cert + reds_uncert
        n_cross = len(crossings)
        in_green = sum(1 for x in crossings if self._inside_any_interval(x, greens))
        in_red = sum(1 for x in crossings if self._inside_any_interval(x, reds))
        in_obs = sum(1 for x in crossings if self._inside_any_interval(x, observed))
        unassigned = max(0, n_cross - in_green - in_red)
        d_green = self._interval_duration(greens)
        d_red = self._interval_duration(reds)
        d_obs = self._interval_duration(observed)

        rate_green = (in_green / d_green) if d_green > 0 else np.nan
        rate_red = (in_red / d_red) if d_red > 0 else np.nan
        rate_ratio = (rate_green / rate_red) if (np.isfinite(rate_green) and np.isfinite(rate_red) and rate_red > 0) else np.nan

        return {
            "label": label,
            "vehicles": int(phase_df["vid"].nunique()),
            "crossings": int(n_cross),
            "greens": int(len(greens)),
            "reds": int(len(reds)),
            "observed_windows": int(len(observed)),
            "obs_duration_s": float(d_obs),
            "green_duration_s": float(d_green),
            "red_duration_s": float(d_red),
            "cross_in_green": int(in_green),
            "cross_in_red": int(in_red),
            "cross_unassigned": int(unassigned),
            "cross_in_observed": int(in_obs),
            "pct_cross_in_green": float(100.0 * in_green / n_cross) if n_cross > 0 else np.nan,
            "pct_cross_in_red": float(100.0 * in_red / n_cross) if n_cross > 0 else np.nan,
            "green_cross_rate": float(rate_green) if np.isfinite(rate_green) else np.nan,
            "red_cross_rate": float(rate_red) if np.isfinite(rate_red) else np.nan,
            "green_to_red_rate_ratio": float(rate_ratio) if np.isfinite(rate_ratio) else np.nan,
            "has_full_cycle": int(len(greens) >= 2),
        }

    # ------------------------------------------------------------------
    # Subset renderers
    # ------------------------------------------------------------------

    def _render_space_time_subset(
        self,
        sub_df: pd.DataFrame,
        save_path: str,
        title_local: str,
        zoom_windows=None,
        phase_source_df: pd.DataFrame = None,
        phase_label: str = None,
        precomputed_phases: tuple = None,
    ):
        if sub_df.empty:
            return

        fig2, ax2 = plt.subplots(figsize=(12, 8), dpi=150)
        self._plot_trajectories(ax2, sub_df)

        if precomputed_phases is not None:
            green_ints, red_cert_ints, red_uncert_ints, observed_ints, crossing_times = precomputed_phases
        else:
            if phase_source_df is None or phase_source_df.empty:
                phase_source_df = sub_df
            green_ints, red_cert_ints, red_uncert_ints, observed_ints, crossing_times = \
                self.signal_engine.infer_signal_intervals(phase_source_df)

        mov_for_display = None
        if phase_label:
            mov_for_display = str(phase_label).split("_lane_")[0]
        green_ints, red_cert_ints, red_uncert_ints, observed_ints = self._normalise_display_phases(
            mov_for_display or "", green_ints, red_cert_ints, red_uncert_ints,
            observed_ints, crossing_times,
        )

        self._draw_phase_bars(ax2, observed_ints, red_cert_ints, red_uncert_ints, green_ints)

        # Start-up lost time — Δ_L visual annotation removed (thesis style).
        # Console/CSV reporting is retained via _report_startup_delays.
        startup_delays = self._compute_startup_delays(green_ints, crossing_times)

        if len(green_ints) < 2:
            src = phase_label if phase_label is not None else "self"
            print(
                f"[INFO] Phase UNKNOWN for {os.path.basename(save_path)} "
                f"(source={src}, crossings={len(crossing_times)})."
            )

        min_t = float(sub_df["t"].min())
        max_t = float(sub_df["t"].max())
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Distance (m) — stop line at 0")
        ax2.set_title(title_local)
        ax2.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        fig2.savefig(save_path)

        if zoom_windows:
            base2, ext2 = os.path.splitext(save_path)
            for t0, t1 in zoom_windows:
                ax2.set_xlim(t0, t1)
                zoom_path = f"{base2}_zoom_{t0}_{t1}{ext2}"
                fig2.savefig(zoom_path)
            ax2.set_xlim(min_t, max_t)

        plt.close(fig2)

    def _render_empty_space_time(self, save_path: str, title_local: str, zoom_windows=None):
        fig_e, ax_e = plt.subplots(figsize=(12, 8), dpi=150)
        ax_e.hlines(0.0, 0.0, 1.0, colors="0.70", linewidth=6, alpha=0.9)
        ax_e.text(0.5, 0.55, "No valid trajectories after filtering",
                  ha="center", va="center", transform=ax_e.transAxes)
        ax_e.set_xlim(0.0, 1.0)
        ax_e.set_ylim(-200.0, 80.0)
        ax_e.set_xlabel("Time (s)")
        ax_e.set_ylabel("Distance (m) — stop line at 0")
        ax_e.set_title(title_local)
        ax_e.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        fig_e.savefig(save_path)
        print(f"[INFO] Saved empty chart: {save_path}")

        if zoom_windows:
            base2, ext2 = os.path.splitext(save_path)
            for t0, t1 in zoom_windows:
                ax_e.set_xlim(float(t0), float(t1))
                zoom_path = f"{base2}_zoom_{t0}_{t1}{ext2}"
                fig_e.savefig(zoom_path)
                print(f"[INFO] Saved empty zoom chart: {zoom_path}")
            ax_e.set_xlim(0.0, 1.0)

        plt.close(fig_e)

    # ------------------------------------------------------------------
    # Paper method
    # ------------------------------------------------------------------

    def _render_paper_method_for_arm(
        self,
        sub_df: pd.DataFrame,
        phase_source_df: pd.DataFrame,
        out_dir: str,
        phase_group: str,
        title_local: str,
        zoom_windows=None,
    ):
        """Paper-inspired method: infer green cycles, estimate free-flow speed,
        assign trajectories to cycles, compute density."""
        os.makedirs(out_dir, exist_ok=True)

        def _first_clean_crossing_time(one_df: pd.DataFrame):
            gg = one_df.sort_values("t")
            gg = gg.groupby("t", sort=True, as_index=False).agg({"y": "median"})
            t_arr = gg["t"].values.astype(float)
            y_arr = gg["y"].values.astype(float)
            if len(t_arr) < 2:
                return np.nan
            dt = np.diff(t_arr)
            dy = np.diff(y_arr)
            speed = np.abs(dy / np.maximum(dt, 1e-6))
            for i in range(len(dt)):
                if dt[i] <= 0.05 or dt[i] > 4.0:
                    continue
                if speed[i] > 25.0:
                    continue
                y0 = y_arr[i]
                y1 = y_arr[i + 1]
                if y0 < 0.0 and y1 >= 0.0 and y1 != y0:
                    frac = (0.0 - y0) / (y1 - y0)
                    return float(t_arr[i] + frac * (t_arr[i + 1] - t_arr[i]))
            return np.nan

        def _estimate_freeflow_speed_mps(one_df: pd.DataFrame):
            speeds = []
            for _, gg0 in one_df.groupby("vid"):
                gg = gg0.sort_values("t")
                gg = gg.groupby("t", sort=True, as_index=False).agg({"y": "median"})
                t_arr = gg["t"].values.astype(float)
                y_arr = gg["y"].values.astype(float)
                if len(t_arr) < 3:
                    continue
                dt = np.diff(t_arr)
                dy = np.diff(y_arr)
                valid = (
                    (dt > 0.05) &
                    (dt < 2.5) &
                    (dy > 0.0) &
                    (y_arr[:-1] < -30.0) &
                    (y_arr[1:] < -5.0)
                )
                v = (dy[valid] / np.maximum(dt[valid], 1e-6))
                v = v[(v >= 2.0) & (v <= 30.0)]
                if len(v) > 0:
                    speeds.extend(v.tolist())
            if len(speeds) == 0:
                return 12.0
            vff = float(np.percentile(np.array(speeds, dtype=float), 85))
            return float(np.clip(vff, 6.0, 22.0))

        if sub_df is None or sub_df.empty:
            veh_csv = os.path.join(out_dir, "paper_method_vehicle_assignment.csv")
            cyc_csv = os.path.join(out_dir, "paper_method_cycle_density.csv")
            pd.DataFrame(columns=["vid", "cross_time", "earliest_departure_time", "cycle_idx"]).to_csv(veh_csv, index=False)
            pd.DataFrame(columns=[
                "phase_group", "cycle_idx", "green_start", "green_end", "next_green_start",
                "window_len_s", "n_first", "n_last", "density_first", "density_last", "state",
            ]).to_csv(cyc_csv, index=False)
            fig_path = os.path.join(out_dir, f"paper_method_{phase_group}.png")
            self._render_empty_space_time(fig_path, f"{title_local} | paper-method", zoom_windows=zoom_windows)
            return

        if phase_source_df is None or phase_source_df.empty:
            phase_source_df = sub_df

        green_ints, red_cert_ints, red_uncert_ints, obs_ints, _ = \
            self.signal_engine.infer_signal_intervals(phase_source_df)
        cycle_starts = [float(gs) for gs, _ in green_ints]
        v_ff = _estimate_freeflow_speed_mps(sub_df)

        veh_rows = []
        for vid, g in sub_df.groupby("vid"):
            gg = g.sort_values("t")
            gg = gg.groupby("t", sort=True, as_index=False).agg({"y": "median"})
            t_arr = gg["t"].values.astype(float)
            y_arr = gg["y"].values.astype(float)
            if len(t_arr) < 2:
                continue
            cross_t = _first_clean_crossing_time(gg)
            if not np.isfinite(cross_t):
                continue
            neg_mask = y_arr < 0.0
            if not np.any(neg_mask):
                continue
            earliest_dep = float(np.min(t_arr[neg_mask] - (y_arr[neg_mask] / max(v_ff, 1e-6))))
            if len(cycle_starts) > 0:
                cidx = int(np.searchsorted(np.array(cycle_starts), earliest_dep, side="right") - 1)
            else:
                cidx = -1
            veh_rows.append({
                "vid": int(vid),
                "cross_time": float(cross_t),
                "earliest_departure_time": earliest_dep,
                "cycle_idx": cidx,
            })

        veh_df = pd.DataFrame(veh_rows)
        veh_csv = os.path.join(out_dir, "paper_method_vehicle_assignment.csv")
        if veh_df.empty:
            pd.DataFrame(columns=["vid", "cross_time", "earliest_departure_time", "cycle_idx"]).to_csv(veh_csv, index=False)
        else:
            veh_df.to_csv(veh_csv, index=False)

        cycle_rows = []
        density_window_s = 8.0
        if (not veh_df.empty) and (len(green_ints) >= 2):
            for i in range(len(green_ints) - 1):
                gs, ge = green_ints[i]
                next_gs = cycle_starts[i + 1]
                if (next_gs - gs) <= 0.0 or (ge - gs) <= 2.0:
                    continue

                win_len = float(min(density_window_s, max(3.0, 0.35 * (ge - gs))))
                f0, f1 = float(gs), float(min(gs + win_len, ge))
                l0, l1 = float(max(ge - win_len, gs)), float(ge)

                cyc = veh_df[veh_df["cycle_idx"] == i]
                n_first = int(((cyc["cross_time"] >= f0) & (cyc["cross_time"] < f1)).sum())
                n_last = int(((cyc["cross_time"] >= l0) & (cyc["cross_time"] <= l1)).sum())
                d_first = float(n_first / max(f1 - f0, 1e-6))
                d_last = float(n_last / max(l1 - l0, 1e-6))
                state = "OVER_SATURATED" if d_last > d_first else "UNDER_SATURATED"

                cycle_rows.append({
                    "phase_group": phase_group,
                    "cycle_idx": int(i),
                    "green_start": float(gs),
                    "green_end": float(ge),
                    "next_green_start": float(next_gs),
                    "window_len_s": win_len,
                    "n_first": n_first,
                    "n_last": n_last,
                    "density_first": d_first,
                    "density_last": d_last,
                    "state": state,
                })

        cyc_df = pd.DataFrame(cycle_rows)
        cyc_csv = os.path.join(out_dir, "paper_method_cycle_density.csv")
        if cyc_df.empty:
            pd.DataFrame(columns=[
                "phase_group", "cycle_idx", "green_start", "green_end", "next_green_start",
                "window_len_s", "n_first", "n_last", "density_first", "density_last", "state",
            ]).to_csv(cyc_csv, index=False)
        else:
            cyc_df.to_csv(cyc_csv, index=False)

        fig_p, ax_p = plt.subplots(figsize=(12, 8), dpi=150)
        self._plot_trajectories(ax_p, sub_df)
        self._draw_phase_bars(ax_p, obs_ints, red_cert_ints, red_uncert_ints, green_ints)

        y_min = float(sub_df["y"].min()) if len(sub_df) else -200.0
        y_min = min(y_min, -120.0)
        for cstart in cycle_starts:
            t_up = float(cstart + y_min / max(v_ff, 1e-6))
            ax_p.plot([t_up, cstart], [y_min, 0.0], color="red", linewidth=2.0, alpha=0.9)

        ax_p.set_xlabel("Time (s)")
        ax_p.set_ylabel("Distance (m) — stop line at 0")
        ax_p.set_title(f"{title_local} | paper-method (v_ff={v_ff:.2f} m/s)")
        ax_p.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        fig_path = os.path.join(out_dir, f"paper_method_{phase_group}.png")
        fig_p.savefig(fig_path)

        if zoom_windows:
            basep, extp = os.path.splitext(fig_path)
            for t0, t1 in zoom_windows:
                ax_p.set_xlim(float(t0), float(t1))
                zpath = f"{basep}_zoom_{t0}_{t1}{extp}"
                fig_p.savefig(zpath)

        plt.close(fig_p)

    # ------------------------------------------------------------------
    # Main diagram plot
    # ------------------------------------------------------------------

    def _plot_main_diagram(self):
        """Plot all valid trajectories with signal phase bars."""
        plot_df = self.plot_df
        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        self._plot_trajectories(ax, plot_df)

        green_main, red_cert_main, red_uncert_main, observed_main, crossing_main = \
            self.signal_engine.infer_signal_intervals(plot_df)
        self._draw_phase_bars(ax, observed_main, red_cert_main, red_uncert_main, green_main)

        if len(green_main) < 2:
            print(
                "[INFO] Signal phases left UNKNOWN for main plot (insufficient full-cycle evidence). "
                f"Crossings detected: {len(crossing_main)}"
            )

        min_time = float(plot_df["t"].min())
        max_time = float(plot_df["t"].max())
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance (m) — stop line at 0")
        ax.set_title(self.title)
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        fig.savefig(self.out_path)

        zoom_windows = [
            (0, 210),
            (450, 650),
            (950, 1150),
            (1400, 1600),
        ]
        base, ext = os.path.splitext(self.out_path)
        for t0, t1 in zoom_windows:
            ax.set_xlim(t0, t1)
            zoom_path = f"{base}_zoom_{t0}_{t1}{ext}"
            fig.savefig(zoom_path)

        ax.set_xlim(min_time, max_time)
        plt.close(fig)
        print(f"  Main diagram + {len(zoom_windows)} zooms saved ({self.processed_count} vehicles)")

    # ------------------------------------------------------------------
    # Green inference audit
    # ------------------------------------------------------------------

    def _write_audit(self):
        run_dir = os.path.dirname(self.out_path)
        try:
            audit_rows = []
            audit_rows.append(self._audit_green_inference("MAIN_ALL", self.plot_df))
            for phase_label, phase_movs in sorted(self._COMMON_GREEN_GROUPS.items()):
                audit_rows.append(self._audit_green_inference(
                    phase_label,
                    self.plot_df[self.plot_df["movement"].isin(phase_movs)].copy(),
                ))
            for mov_label in self._MOVEMENTS:
                audit_rows.append(self._audit_green_inference(
                    mov_label,
                    self.plot_df[self.plot_df["movement"] == mov_label].copy(),
                ))
            audit_df = pd.DataFrame(audit_rows)
            audit_path = os.path.join(run_dir, "green_inference_audit.csv")
            audit_df.to_csv(audit_path, index=False)
            dbg(f"  Green inference audit: {audit_path}")
        except Exception as e:
            print(f"[WARNING] Failed to write green inference audit: {e}")

    # ------------------------------------------------------------------
    # Per-movement / per-lane diagrams
    # ------------------------------------------------------------------

    def _plot_per_movement_per_lane(self):
        plot_df = self.plot_df
        run_dir = os.path.dirname(self.out_path)
        _, ext_local = os.path.splitext(self.out_path)
        if not ext_local:
            ext_local = ".png"

        total_lane_diagrams = 0
        for mov in self._MOVEMENTS:
            sub_mov = plot_df[plot_df["movement"] == mov].copy()
            if sub_mov.empty:
                continue

            # Use pre-computed movement-level phases (same Q/R structure).
            mov_phases = self._movement_phases.get(mov)

            lanes_in_mov = sorted([l for l in sub_mov["lane"].unique() if l != "unknown"])
            display_lane_map, _display_direction_map = self._display_lane_maps(mov, lanes_in_mov)
            lane_summaries = []

            for lane_label in lanes_in_mov:
                sub_lane = sub_mov[sub_mov["lane"] == lane_label].copy()
                if sub_lane.empty:
                    continue

                n_veh = sub_lane["vid"].nunique()
                display_lane_label = display_lane_map.get(str(lane_label), self._display_lane_label(mov, lane_label))
                lane_dir = os.path.join(run_dir, f"space_time_{mov}_{display_lane_label}")
                os.makedirs(lane_dir, exist_ok=True)
                lane_out = os.path.join(lane_dir, f"space_time_{mov}_{display_lane_label}{ext_local}")
                mov_zooms = self._ZOOM_WINDOWS_BY_MOVEMENT.get(mov, None)
                path_desc = get_lane_path_description(mov, lane_label)
                if path_desc:
                    lane_title = f"{self.title}: {mov}_{display_lane_label} ({path_desc}) {n_veh} veh"
                else:
                    lane_title = f"{self.title}: {mov}_{display_lane_label} {n_veh} veh"
                self._render_space_time_subset(
                    sub_lane, lane_out, lane_title,
                    zoom_windows=mov_zooms,
                    precomputed_phases=mov_phases,
                    phase_label=f"{mov}_{lane_label}",
                )
                lane_summaries.append(f"{display_lane_label}={n_veh}")
                total_lane_diagrams += 1

            if lane_summaries:
                print(f"  {mov}: {', '.join(lane_summaries)}")
                # Report start-up lost time once per movement (phases are shared across lanes)
                if mov_phases is not None and mov not in self._PERMISSIVE_MOVS:
                    green_ints_mov, _, _, _, crossing_times_mov = mov_phases
                    startup_delays_mov = self._compute_startup_delays(
                        green_ints_mov, crossing_times_mov
                    )
                    self._report_startup_delays(mov, startup_delays_mov)

        print(f"  Total: {total_lane_diagrams} per-lane diagrams generated")

    # ------------------------------------------------------------------
    # Gap-filling helper for the results table
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_phase_gaps(phase_list, observed_ints, mov_zooms, min_gap=4.0):
        """Ensure every second inside recording windows is covered.

        When zoom windows are available they are used directly as the
        authoritative time boundaries (they represent the known
        recording segments).  This avoids the problem where
        ``observed_ints`` splits at short data-inherent gaps (e.g. a
        15 s red phase with no crossings) and leaves those gaps
        unfilled.

        min_gap=4.0 matches Q: gaps shorter than 4 s between green intervals
        are left as green (data noise), not filled as red_certain.

        Returns the augmented and sorted phase_list.
        """
        if not observed_ints and not mov_zooms:
            return phase_list

        # Use zoom windows as the authority when available.
        # They define the actual recording segments and should be
        # fully covered by green or red — no blank gaps.
        if mov_zooms is not None and len(mov_zooms) > 0:
            cover_windows = list(mov_zooms)
        elif observed_ints:
            cover_windows = list(observed_ints)
        else:
            return phase_list

        if not cover_windows:
            return phase_list

        # Sort existing phases
        phase_list = sorted(phase_list, key=lambda x: x[0])

        # Also filter phase_list to only keep entries inside cover windows
        filtered = []
        for s, e, phase in phase_list:
            for ws, we in cover_windows:
                cs = max(s, ws)
                ce = min(e, we)
                if ce - cs > 0:
                    filtered.append((cs, ce, phase))
        filtered.sort(key=lambda x: x[0])

        # For each cover window, walk through and fill uncovered spans
        filled = list(filtered)
        for ws, we in cover_windows:
            # Collect phases within this window
            in_win = [(s, e, p) for s, e, p in filtered if e > ws and s < we]
            in_win.sort()

            cursor = ws
            for s, e, p in in_win:
                cs = max(s, ws)
                if cs - cursor >= min_gap:
                    filled.append((cursor, cs, "red_certain"))
                cursor = max(cursor, min(e, we))
            if we - cursor >= min_gap:
                filled.append((cursor, we, "red_certain"))

        filled.sort(key=lambda x: x[0])
        return filled

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------

    def _build_results_table(self):
        if self.general_dir is None:
            return

        plot_df = self.plot_df

        try:
            results_rows = []
            for mov in self._MOVEMENTS:
                sub_mov = plot_df[plot_df["movement"] == mov].copy()
                if sub_mov.empty:
                    continue

                lanes_in_mov = sorted([l for l in sub_mov["lane"].unique() if l != "unknown"])
                _display_lane_map, display_direction_map = self._display_lane_maps(mov, lanes_in_mov)
                mov_phases = self._movement_phases.get(mov)

                for lane_label in lanes_in_mov:
                    sub_lane = sub_mov[sub_mov["lane"] == lane_label].copy()
                    if sub_lane.empty:
                        continue

                    direction_str = display_direction_map.get(str(lane_label), self._display_direction(mov, lane_label))

                    if mov_phases is not None:
                        green_ints, red_cert_ints, red_uncert_ints, observed_ints, crossing_times = mov_phases
                    else:
                        green_ints, red_cert_ints, red_uncert_ints, observed_ints, crossing_times = \
                            self.signal_engine.infer_signal_intervals(sub_lane)

                    green_ints, red_cert_ints, red_uncert_ints, observed_ints = self._normalise_display_phases(
                        mov, green_ints, red_cert_ints, red_uncert_ints,
                        observed_ints, crossing_times,
                    )

                    phase_list = []
                    for s, e in green_ints:
                        phase_list.append((s, e, "green"))
                    for s, e in red_cert_ints:
                        phase_list.append((s, e, "red_certain"))
                    for s, e in red_uncert_ints:
                        phase_list.append((s, e, "red_uncertain"))
                    phase_list.sort(key=lambda x: x[0])

                    mov_zooms = self._ZOOM_WINDOWS_BY_MOVEMENT.get(mov, None)

                    # --- Fill gaps within observed windows ---
                    # Any time within an observed window that is not
                    # covered by green/red_certain/red_uncertain gets
                    # filled with red_certain so the diagram has no
                    # blank regions inside recording segments.
                    phase_list = self._fill_phase_gaps(
                        phase_list, observed_ints, mov_zooms,
                    )

                    if not phase_list:
                        t_min = float(sub_lane["t"].min())
                        t_max = float(sub_lane["t"].max())
                        results_rows.append({
                            "direction": direction_str,
                            "lane_id": lane_label,
                            "phase": "unknown",
                            "start": round(t_min, 2),
                            "end": round(t_max, 2),
                        })
                    else:
                        for s, e, phase in phase_list:
                            results_rows.append({
                                "direction": direction_str,
                                "lane_id": lane_label,
                                "phase": phase,
                                "start": round(s, 2),
                                "end": round(e, 2),
                            })

            if results_rows:
                results_df = pd.DataFrame(results_rows,
                    columns=["direction", "lane_id", "phase", "start", "end"])
                results_csv = os.path.join(self.general_dir, "results_table.csv")
                results_df.to_csv(results_csv, sep=";", index=False)
                print(f"  Results table: {len(results_df)} rows")

                self._phase_consistency_checks(results_df)
                self._render_signal_timing_diagrams(results_df)
            else:
                print("[WARNING] No results rows generated for results table.")
        except Exception as e:
            print(f"[ERROR] Failed to build results table: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Phase consistency checks
    # ------------------------------------------------------------------

    def _phase_consistency_checks(self, results_df: pd.DataFrame):
        print("\n=== TRAFFIC LIGHT PHASE CONSISTENCY CHECKS ===")
        check_rows = []
        for _, row in results_df.iterrows():
            duration = row["end"] - row["start"]
            issue = None
            if row["phase"] in ("green", "red_certain", "red_uncertain") and duration < 5.0:
                issue = f"VERY_SHORT_PHASE ({duration:.1f}s)"
            elif row["phase"] in ("green", "red_certain", "red_uncertain") and duration < 10.0:
                issue = f"SHORT_PHASE ({duration:.1f}s)"
            if issue:
                check_rows.append({
                    "direction": row["direction"],
                    "lane_id": row["lane_id"],
                    "phase": row["phase"],
                    "start": row["start"],
                    "end": row["end"],
                    "duration_s": round(duration, 2),
                    "issue": issue,
                })

        # Check alternation
        for (dir_label, lane_label), grp in results_df.groupby(["direction", "lane_id"]):
            grp_sorted = grp.sort_values("start")
            phases_seq = grp_sorted["phase"].tolist()
            starts = grp_sorted["start"].tolist()
            for i in range(1, len(phases_seq)):
                if phases_seq[i] == phases_seq[i - 1] and phases_seq[i] != "unknown":
                    check_rows.append({
                        "direction": dir_label,
                        "lane_id": lane_label,
                        "phase": phases_seq[i],
                        "start": starts[i],
                        "end": grp_sorted["end"].iloc[i],
                        "duration_s": round(grp_sorted["end"].iloc[i] - starts[i], 2),
                        "issue": f"REPEATED_PHASE (consecutive {phases_seq[i]})",
                    })

            # Check cycle length consistency
            green_starts = [s for s, p in zip(starts, phases_seq) if p == "green"]
            if len(green_starts) >= 3:
                cycle_lengths = [green_starts[j + 1] - green_starts[j] for j in range(len(green_starts) - 1)]
                median_cycle = float(np.median(cycle_lengths))
                for j, cl in enumerate(cycle_lengths):
                    if median_cycle > 0 and (cl < 0.4 * median_cycle or cl > 2.5 * median_cycle):
                        check_rows.append({
                            "direction": dir_label,
                            "lane_id": lane_label,
                            "phase": "green",
                            "start": green_starts[j],
                            "end": green_starts[j + 1],
                            "duration_s": round(cl, 2),
                            "issue": f"IRREGULAR_CYCLE_LENGTH ({cl:.1f}s vs median {median_cycle:.1f}s)",
                        })

        # Cross-lane check
        for dir_label, dir_grp in results_df.groupby("direction"):
            lanes = sorted(dir_grp["lane_id"].unique())
            if len(lanes) < 2:
                continue
            for phase_type in ("green", "red_certain", "red_uncertain"):
                phase_data = {}
                for lane_label, lane_grp in dir_grp.groupby("lane_id"):
                    ph = lane_grp[lane_grp["phase"] == phase_type].sort_values("start")
                    phase_data[lane_label] = list(zip(ph["start"].tolist(), ph["end"].tolist()))
                counts = {l: len(v) for l, v in phase_data.items()}
                if len(set(counts.values())) > 1:
                    check_rows.append({
                        "direction": dir_label,
                        "lane_id": "ALL",
                        "phase": phase_type,
                        "start": 0,
                        "end": 0,
                        "duration_s": 0,
                        "issue": f"CROSS_LANE_PHASE_COUNT_MISMATCH ({phase_type} counts: {counts})",
                    })

        if check_rows:
            checks_df = pd.DataFrame(check_rows,
                columns=["direction", "lane_id", "phase", "start", "end", "duration_s", "issue"])
            checks_csv = os.path.join(self.general_dir, "phase_consistency_checks.csv")
            checks_df.to_csv(checks_csv, sep=";", index=False)
            print(f"[WARNING] Found {len(checks_df)} phase consistency issues!")
            for issue_type in checks_df["issue"].str.extract(r'^(\w+)')[0].unique():
                count = checks_df["issue"].str.startswith(issue_type).sum()
                print(f"  - {issue_type}: {count} occurrences")
            for dir_label in checks_df["direction"].unique():
                dir_issues = checks_df[checks_df["direction"] == dir_label]
                print(f"  [{dir_label}] {len(dir_issues)} issues")
            print(f"  Full report: {checks_csv}")
        else:
            print("[OK] All phase checks passed — no issues detected.")

    # ------------------------------------------------------------------
    # Signal timing diagrams
    # ------------------------------------------------------------------

    def _render_signal_timing_diagrams(self, results_df: pd.DataFrame):
        try:
            signal_timing_dir = os.path.join(self.general_dir, "signal_timing_diagrams")
            os.makedirs(signal_timing_dir, exist_ok=True)

            # Full timeline overview
            t_global_min = float(results_df["start"].min())
            t_global_max = float(results_df["end"].max())
            self._render_signal_timing_chart(
                results_df, t_global_min, t_global_max,
                os.path.join(signal_timing_dir, "signal_timing_full.png"),
                "Inferred Signal Timing — Full Timeline",
            )

            # Protected movements only (NS, SN, NE, ES)
            protected_movs = set(self._MOVEMENTS) - self._PERMISSIVE_MOVS
            df_protected = results_df[
                results_df["direction"].apply(self._dir_to_mov).isin(protected_movs)
            ]
            self._render_signal_timing_chart(
                df_protected, t_global_min, t_global_max,
                os.path.join(signal_timing_dir, "signal_timing_protected.png"),
                "Inferred Signal Timing — Protected Movements",
            )

            # Permissive movements only (EN, SE)
            df_permissive = results_df[
                results_df["direction"].apply(self._dir_to_mov).isin(self._PERMISSIVE_MOVS)
            ]
            self._render_signal_timing_chart(
                df_permissive, t_global_min, t_global_max,
                os.path.join(signal_timing_dir, "signal_timing_permissive.png"),
                "Inferred Signal Timing — Permissive Movements",
                permissive_label=True,
            )

            # Phase summary table: which movements are active per phase
            self._generate_phase_summary(results_df, signal_timing_dir)

        except Exception as e:
            print(f"[ERROR] Failed to generate signal timing diagrams: {e}")
            traceback.print_exc()

    def _generate_phase_summary(self, results_df, out_dir):
        """Create a summary table + diagram showing the observed signal phases.

        Uses the ground-truth ``_OBSERVED_PHASES`` definition (3 phases)
        rather than trying to discover phases from noisy per-movement
        green interval boundaries.
        """
        try:
            import csv as _csv

            all_movs = self._MOVEMENTS  # ["NS", "NE", "SN", "SE", "EN", "ES"]

            # --- 1. Write CSV summary ---
            csv_path = os.path.join(out_dir, "phase_summary.csv")
            with open(csv_path, "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["phase_id", "description", "green_movements",
                            "red_movements"])
                for phase_id, desc, green_movs in self._OBSERVED_PHASES:
                    green_set = set(green_movs)
                    red_movs = [m for m in all_movs if m not in green_set]
                    w.writerow([
                        phase_id,
                        desc,
                        ", ".join(green_movs),
                        ", ".join(red_movs),
                    ])
            print(f"  Phase summary CSV: {csv_path}")

            # --- 2. Render phase summary diagram ---
            self._render_phase_summary_diagram(out_dir)

        except Exception as e:
            print(f"[ERROR] Phase summary: {e}")
            traceback.print_exc()

    def _render_phase_summary_diagram(self, out_dir):
        """Draw a clean diagram showing each signal phase and which
        movements are green/red, based on ground-truth observed phases."""
        from matplotlib.patches import FancyBboxPatch

        phases = self._OBSERVED_PHASES
        n_phases = len(phases)
        if n_phases == 0:
            return

        all_movs = self._MOVEMENTS

        with plt.rc_context(_THESIS_FONT):
            fig_w = max(8, 3.2 * n_phases + 1)
            fig_h = max(4, 0.5 * len(all_movs) + 3)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            col_width = 1.0
            row_height = 0.5
            x_gap = 0.6

            for col, (phase_id, desc, green_movs) in enumerate(phases):
                x_left = col * (col_width + x_gap)
                green_set = set(green_movs)

                # Phase header
                ax.text(
                    x_left + col_width / 2, len(all_movs) * row_height + 0.4,
                    f"Phase {phase_id}\n({desc})",
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                )

                for row, mov in enumerate(all_movs):
                    y_bot = (len(all_movs) - 1 - row) * row_height

                    is_green = mov in green_set
                    color = "#4CAF50" if is_green else "#E53935"
                    alpha = 0.85

                    rect = FancyBboxPatch(
                        (x_left, y_bot), col_width, row_height * 0.85,
                        boxstyle="round,pad=0.02",
                        facecolor=color, edgecolor="white",
                        linewidth=1.5, alpha=alpha, zorder=2,
                    )
                    ax.add_patch(rect)

                    label = "GO" if is_green else "STOP"
                    ax.text(
                        x_left + col_width / 2, y_bot + row_height * 0.42,
                        label, ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white", zorder=3,
                    )

            # Row labels (movement names) on the left
            for row, mov in enumerate(all_movs):
                y_bot = (len(all_movs) - 1 - row) * row_height
                is_perm = mov in self._PERMISSIVE_MOVS
                ax.text(
                    -0.15, y_bot + row_height * 0.55,
                    mov, ha="right", va="center",
                    fontsize=10, fontweight="bold",
                )
                if is_perm:
                    ax.text(
                        -0.15, y_bot + row_height * 0.1,
                        "(Perm.)", ha="right", va="center",
                        fontsize=7, color="#757575", style="italic",
                    )

            total_w = n_phases * (col_width + x_gap)
            ax.set_xlim(-1.1, total_w + 0.3)
            ax.set_ylim(-0.6, len(all_movs) * row_height + 1.2)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(
                "Signal Phase Summary — Active Movements",
                fontsize=14, fontweight="bold", pad=15,
            )

            # Footnote explaining permissive label
            ax.text(
                0, -0.45, "* Permissive: movement operates on permissive green throughout all phases.",
                ha="left", va="bottom", fontsize=7, color="#757575", style="italic",
                transform=ax.transData,
            )

            fig.tight_layout()
            out_path = os.path.join(out_dir, "phase_summary.png")
            _save_fig_formats(fig, out_path)
            plt.close(fig)
            print(f"  Phase summary diagram: {out_path}")

    def _render_signal_timing_chart(self, df_phases, t_start, t_end, out_path, title,
                                     permissive_label=False):
        """Render a single signal timing Gantt chart.

        permissive_label: when True, append '(Perm.)' to y-tick labels for
        movements in _PERMISSIVE_MOVS.
        """
        df_win = df_phases[
            (df_phases["end"] > t_start) & (df_phases["start"] < t_end)
        ].copy()
        if df_win.empty:
            return

        df_win["mov"] = df_win["direction"].apply(self._dir_to_mov)
        df_win["arm_label"] = df_win["mov"].map(self._MOV_TO_ARM_LABEL).fillna("Other")

        arm_order = ["North Arm", "South Arm", "West Arm", "East Arm"]
        y_labels = []
        seen = set()
        for arm in arm_order:
            arm_rows = df_win[df_win["arm_label"] == arm].sort_values(["direction", "lane_id"])
            for _, r in arm_rows.iterrows():
                key = (r["direction"], r["lane_id"])
                if key not in seen:
                    seen.add(key)
                    y_labels.append(key)

        if not y_labels:
            return

        with plt.rc_context(_THESIS_FONT):
            n_rows = len(y_labels)
            fig_height = max(4, 0.55 * n_rows + 2.5)
            fig, ax = plt.subplots(figsize=(16, fig_height))

            label_to_y = {lbl: i for i, lbl in enumerate(y_labels)}

            for _, row in df_win.iterrows():
                key = (row["direction"], row["lane_id"])
                if key not in label_to_y:
                    continue
                y_pos = label_to_y[key]
                bar_start = max(float(row["start"]), t_start)
                bar_end = min(float(row["end"]), t_end)
                bar_width = bar_end - bar_start
                if bar_width <= 0:
                    continue
                phase = row["phase"]
                # Merge red_uncertain into red_certain for clean display
                if phase == "red_uncertain":
                    phase = "red_certain"
                color = self._PHASE_COLORS.get(phase, "#BDBDBD")
                ax.barh(y_pos, bar_width, left=bar_start, height=0.7,
                        color=color, edgecolor="white", linewidth=0.5)

            ax.set_yticks(range(n_rows))
            display_labels = []
            for d, l in y_labels:
                mov = self._dir_to_mov(d)
                if permissive_label or mov in self._PERMISSIVE_MOVS:
                    display_labels.append(f"{d} (Perm.)")
                else:
                    display_labels.append(d)
            ax.set_yticklabels(display_labels)
            ax.invert_yaxis()

            ax.set_xlim(t_start, t_end)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
            ax.tick_params(axis="x", rotation=45)
            ax.set_xlabel("Time (s)", labelpad=10)
            ax.set_title(title, fontweight="bold", pad=12)
            ax.grid(True, axis="x", alpha=0.3)

            # Add arm group separators
            prev_arm = None
            for i, (d, l) in enumerate(y_labels):
                mov = self._dir_to_mov(d)
                arm = self._MOV_TO_ARM_LABEL.get(mov, "Other")
                if prev_arm is not None and arm != prev_arm:
                    ax.axhline(i - 0.5, color="black", linewidth=1.2, linestyle="-")
                prev_arm = arm

            legend_elements = [
                Patch(facecolor="#4CAF50", edgecolor="black", label="Green"),
                Patch(facecolor="#E53935", edgecolor="black", label="Red"),
            ]
            ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

            fig.tight_layout()
            fig.subplots_adjust(bottom=0.18)
            _save_fig_formats(fig, out_path)
            plt.close(fig)
            dbg(f"  Signal timing diagram: {out_path}")

    def _render_signal_timing_compressed(
        self, df_phases, direction_movs, out_path, title,
        gap_thresh=30.0, dir_key=None,
    ):
        """Render signal timing for one direction with large empty gaps cut out."""
        df_dir = df_phases[
            df_phases["direction"].apply(self._dir_to_mov).isin(direction_movs)
        ].copy()
        if df_dir.empty:
            return

        all_segments = []
        for _, r in df_dir.iterrows():
            all_segments.append((float(r["start"]), float(r["end"])))
        all_segments.sort()

        merged = [all_segments[0]]
        for s, e in all_segments[1:]:
            ps, pe = merged[-1]
            if s <= pe + gap_thresh:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))

        pad = 5.0
        windows = [(max(0, s - pad), e + pad) for s, e in merged]

        y_labels = []
        seen = set()
        for _, r in df_dir.sort_values(["direction", "lane_id"]).iterrows():
            key = (r["direction"], r["lane_id"])
            if key not in seen:
                seen.add(key)
                y_labels.append(key)
        if not y_labels:
            return

        with plt.rc_context(_THESIS_FONT):
            n_rows = len(y_labels)
            n_windows = len(windows)
            label_to_y = {lbl: i for i, lbl in enumerate(y_labels)}

            widths = [we - ws for ws, we in windows]
            total_w = sum(widths)
            fig_width = max(18, min(28, total_w / 18.0))
            fig_height = max(6, 0.7 * n_rows + 4.0)

            fig, axes = plt.subplots(
                1, n_windows,
                sharey=True,
                figsize=(fig_width, fig_height),
                gridspec_kw={"width_ratios": widths, "wspace": 0.08},
                squeeze=False,
            )
            axes = axes[0]

            for w_idx, (ws, we) in enumerate(windows):
                ax = axes[w_idx]
                for _, row in df_dir.iterrows():
                    key = (row["direction"], row["lane_id"])
                    if key not in label_to_y:
                        continue
                    y_pos = label_to_y[key]
                    bar_start = max(float(row["start"]), ws)
                    bar_end = min(float(row["end"]), we)
                    bar_width = bar_end - bar_start
                    if bar_width <= 0:
                        continue
                    phase = row["phase"]
                    if phase == "red_uncertain":
                        phase = "red_certain"
                    color = self._PHASE_COLORS.get(phase, "#BDBDBD")
                    ax.barh(y_pos, bar_width, left=bar_start, height=0.7,
                            color=color, edgecolor="white", linewidth=0.5)

                ax.set_xlim(ws, we)

                # East direction, last window: show only start & end tick
                if dir_key == "E" and w_idx == n_windows - 1:
                    win_phases = df_dir[
                        (df_dir["start"] < we) & (df_dir["end"] > ws)
                    ]
                    if not win_phases.empty:
                        phase_start = max(float(win_phases["start"].min()), ws)
                        phase_end = min(float(win_phases["end"].max()), we)
                        ax.set_xticks([int(round(phase_start)), int(round(phase_end))])
                    else:
                        ax.set_xticks([int(round(ws)), int(round(we))])
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

                ax.grid(True, axis="x", alpha=0.3)

                if w_idx == 0:
                    ax.set_yticks(range(n_rows))
                    display_labels = [d for d, l in y_labels]
                    ax.set_yticklabels(display_labels)
                    ax.invert_yaxis()
                else:
                    ax.tick_params(left=False)

                # Break marks between subplots
                if w_idx > 0:
                    d_size = 0.015
                    kwargs = dict(transform=ax.transAxes, color='k',
                                  clip_on=False, linewidth=1.2)
                    ax.plot((-d_size, d_size), (1 - d_size, 1 + d_size), **kwargs)
                    ax.plot((-d_size, d_size), (-d_size, d_size), **kwargs)
                if w_idx < n_windows - 1:
                    d_size = 0.015
                    kwargs = dict(transform=ax.transAxes, color='k',
                                  clip_on=False, linewidth=1.2)
                    ax.plot((1 - d_size, 1 + d_size), (1 - d_size, 1 + d_size), **kwargs)
                    ax.plot((1 - d_size, 1 + d_size), (-d_size, d_size), **kwargs)

                if w_idx > 0:
                    ax.spines["left"].set_visible(False)
                if w_idx < n_windows - 1:
                    ax.spines["right"].set_visible(False)

            fig.suptitle(title, fontweight="bold")

            legend_elements = [
                Patch(facecolor="#4CAF50", edgecolor="black", label="Green"),
                Patch(facecolor="#E53935", edgecolor="black", label="Red"),
            ]
            axes[-1].legend(handles=legend_elements, loc="upper right", framealpha=0.9)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fig.tight_layout()
            fig.text(0.5, 0.005, "Time (s)", ha="center", va="bottom", fontsize=14)
            fig.subplots_adjust(bottom=0.12)
            _save_fig_formats(fig, out_path)
            plt.close(fig)
            dbg(f"  Signal timing diagram: {out_path}")

    # ------------------------------------------------------------------
    # Headway analysis
    # ------------------------------------------------------------------

    def _headway_analysis(self):
        # Use dedicated headway_dir if set, otherwise fall back to general_dir
        base_dir = self.headway_dir if self.headway_dir else self.general_dir
        if base_dir is None:
            return

        plot_df = self.plot_df

        try:
            headway_rows = []
            for mov in self._MOVEMENTS:
                if mov in self._PERMISSIVE_MOVS:
                    continue
                sub_mov = plot_df[plot_df["movement"] == mov].copy()
                if sub_mov.empty:
                    continue

                lanes_in_mov = sorted([l for l in sub_mov["lane"].unique() if l != "unknown"])
                _display_lane_map, display_direction_map = self._display_lane_maps(mov, lanes_in_mov)
                mov_phases = self._movement_phases.get(mov)

                for lane_label in lanes_in_mov:
                    sub_lane = sub_mov[sub_mov["lane"] == lane_label].copy()
                    if sub_lane.empty:
                        continue

                    if mov_phases is not None:
                        green_ints, red_cert_ints, red_uncert_ints, observed_ints, crossing_times = mov_phases
                    else:
                        green_ints, red_cert_ints, red_uncert_ints, observed_ints, crossing_times = \
                            self.signal_engine.infer_signal_intervals(sub_lane)
                    green_ints, red_cert_ints, red_uncert_ints, observed_ints = self._normalise_display_phases(
                        mov, green_ints, red_cert_ints, red_uncert_ints,
                        observed_ints, crossing_times,
                    )
                    if not green_ints:
                        continue

                    mov_zooms = self._ZOOM_WINDOWS_BY_MOVEMENT.get(mov, None)

                    crossing_events = self.signal_engine.extract_crossing_events(sub_lane)
                    if len(crossing_events) < 2:
                        continue

                    # Build per-vehicle trajectory lookup
                    vid_trajectories = {}
                    for vid, vg in sub_lane.groupby("vid"):
                        vg_s = vg.sort_values("t")
                        vid_trajectories[int(vid)] = (
                            vg_s["t"].values.astype(float),
                            vg_s["y"].values.astype(float),
                        )

                    direction_str = display_direction_map.get(str(lane_label), self._display_direction(mov, lane_label))

                    skipped_no_queue = 0
                    kept_cycles = 0
                    dropped_close_crossings = 0
                    for gs, ge in green_ints:
                        if mov_zooms is not None:
                            in_zoom = any(gs < z1 and ge > z0 for z0, z1 in mov_zooms)
                            if not in_zoom:
                                continue

                        # Only process cycles where a standing queue was present.
                        # A queue exists when a red_certain interval immediately precedes
                        # this green (its end falls within 12 s of the green start).
                        # If no such red_certain interval exists, the cycle is excluded:
                        # it may be red_uncertain -> green or simply unobserved, so the
                        # first crossing may be free-flow rather than queued discharge.
                        preceded_by_queue = any(
                            abs(rc_end - gs) <= 12.0
                            for _rc_start, rc_end in red_cert_ints
                        )
                        if not preceded_by_queue:
                            skipped_no_queue += 1
                            continue

                        green_crossings = [(tc, vid) for tc, vid in crossing_events if gs <= tc <= ge]
                        if len(green_crossings) < 1:
                            continue

                        # Remove duplicate / same-instant crossings inside a
                        # lane-cycle. These are interpolation or resampling
                        # artefacts; physically, two queued vehicles cannot
                        # depart the stop line with a zero headway.
                        MIN_DEPARTURE_HEADWAY_S = 0.3
                        clean_crossings = []
                        for tc, vid in green_crossings:
                            if clean_crossings and tc - clean_crossings[-1][0] < MIN_DEPARTURE_HEADWAY_S:
                                dropped_close_crossings += 1
                                continue
                            clean_crossings.append((tc, vid))
                        green_crossings = clean_crossings
                        if len(green_crossings) < 1:
                            continue

                        # Truncate to the initial queue discharge cluster.
                        # Once the gap between successive crossings exceeds
                        # QUEUE_DISCHARGE_GAP_S the initial queue has cleared;
                        # anything after is a free-flow arrival and must NOT
                        # contribute to h_sat.
                        QUEUE_DISCHARGE_GAP_S = 4.0
                        queue_crossings = [green_crossings[0]]
                        for _k in range(1, len(green_crossings)):
                            gap = green_crossings[_k][0] - green_crossings[_k - 1][0]
                            if gap > QUEUE_DISCHARGE_GAP_S:
                                break
                            queue_crossings.append(green_crossings[_k])

                        if len(queue_crossings) < 2:
                            continue

                        kept_cycles += 1

                        # Position 1: headway = time from green start to first crossing.
                        t_first, vid_first = queue_crossings[0]
                        headway_rows.append({
                            "direction": direction_str,
                            "lane_id": lane_label,
                            "green_start": round(gs, 2),
                            "green_end": round(ge, 2),
                            "queue_position": 1,
                            "vehicle_id": vid_first,
                            "leader_id": None,
                            "crossing_time": round(t_first, 2),
                            "time_headway_s": round(t_first - gs, 2),
                            "space_headway_m": np.nan,
                        })

                        # Positions 2+: successive inter-departure headways
                        # within the discharge cluster only.
                        for pos_idx in range(1, len(queue_crossings)):
                            t_follower = queue_crossings[pos_idx][0]
                            vid_follower = queue_crossings[pos_idx][1]
                            t_leader = queue_crossings[pos_idx - 1][0]
                            vid_leader = queue_crossings[pos_idx - 1][1]

                            time_headway = t_follower - t_leader

                            space_headway = np.nan
                            if vid_leader in vid_trajectories:
                                t_arr_l, y_arr_l = vid_trajectories[vid_leader]
                                if len(t_arr_l) >= 2 and t_arr_l[0] <= t_follower <= t_arr_l[-1]:
                                    y_leader_at_tf = float(np.interp(t_follower, t_arr_l, y_arr_l))
                                    if y_leader_at_tf > 0:
                                        space_headway = y_leader_at_tf

                            headway_rows.append({
                                "direction": direction_str,
                                "lane_id": lane_label,
                                "green_start": round(gs, 2),
                                "green_end": round(ge, 2),
                                "queue_position": pos_idx + 1,
                                "vehicle_id": vid_follower,
                                "leader_id": vid_leader,
                                "crossing_time": round(t_follower, 2),
                                "time_headway_s": round(time_headway, 2),
                                "space_headway_m": round(space_headway, 2) if not np.isnan(space_headway) else np.nan,
                            })

                    if skipped_no_queue or kept_cycles or dropped_close_crossings:
                        print(
                            f"  [headway] {mov} {lane_label}: "
                            f"{kept_cycles} cycle(s) kept (red_certain queue), "
                            f"{skipped_no_queue} skipped (no standing queue), "
                            f"{dropped_close_crossings} duplicate/near-zero crossing(s) dropped"
                        )

            if headway_rows:
                headway_df = pd.DataFrame(headway_rows)
                headway_dir = os.path.join(base_dir, "headway_analysis")
                os.makedirs(headway_dir, exist_ok=True)

                # Drop space headway
                if "space_headway_m" in headway_df.columns:
                    headway_df = headway_df.drop(columns=["space_headway_m"])

                headway_df["leader_id"] = headway_df["leader_id"].astype("Int64")

                headway_csv = os.path.join(headway_dir, "headway_table.csv")
                headway_df.to_csv(headway_csv, sep=";", index=False)
                print(f"  Headway table: {len(headway_df)} rows")

                MAX_QUEUE_POS = 8
                pos_counts = headway_df.groupby("queue_position").size()
                valid_positions = pos_counts[pos_counts >= 2].index
                hw_plot = headway_df[
                    (headway_df["queue_position"] <= MAX_QUEUE_POS) &
                    (headway_df["queue_position"].isin(valid_positions))
                ].copy()

                self._plot_headway_graphs(headway_df, hw_plot, headway_dir, MAX_QUEUE_POS)
            else:
                print("[INFO] No headway data computed (need >=2 crossings in a green phase).")

        except Exception as e:
            print(f"[ERROR] Failed headway analysis: {e}")
            traceback.print_exc()

    def _plot_headway_graphs(self, headway_df, hw_plot, headway_dir, MAX_QUEUE_POS):
        with plt.rc_context(_THESIS_FONT):

            # GRAPH 1: Time Headway vs. Queue Position (with error bars)
            fig_pos, ax = plt.subplots(figsize=(10, 7))
            for (direction, lane), grp in hw_plot.groupby(["direction", "lane_id"]):
                label = f"{direction} {lane}"
                ax.scatter(grp["queue_position"], grp["time_headway_s"],
                           s=50, alpha=0.5, label=label, zorder=3)

            pos_stats = hw_plot.groupby("queue_position")["time_headway_s"].agg(["mean", "std", "count"])
            pos_stats["se"] = pos_stats["std"] / np.sqrt(pos_stats["count"])
            ax.errorbar(pos_stats.index, pos_stats["mean"], yerr=pos_stats["se"],
                        fmt="ko-", linewidth=2, markersize=7, capsize=4,
                        label="Mean ± SE", zorder=5)

            sat_vals = hw_plot[hw_plot["queue_position"] >= 4]["time_headway_s"]
            if len(sat_vals) > 0:
                sat_h = sat_vals.mean()
                ax.axhline(sat_h, color="red", linestyle="--", linewidth=1.5,
                           label=f"Saturation ≈ {sat_h:.1f} s", zorder=4)

            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Queue Position")
            ax.set_ylabel("Time Headway (s)")
            ax.set_title("Discharge Time Headway vs. Queue Position")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            max_pos = int(hw_plot["queue_position"].max()) if not hw_plot.empty else MAX_QUEUE_POS
            ax.set_xticks(range(0, max_pos + 2))

            fig_pos.tight_layout()
            pos_path = os.path.join(headway_dir, "headway_vs_queue_position.png")
            _save_fig_formats(fig_pos, pos_path)
            plt.close(fig_pos)
            dbg(f"  Headway plots saved")


            # GRAPH 3: Combined all-lanes headway vs queue position
            # + formal startup lost time (L) computation per cycle
            fig_comb, ax = plt.subplots(figsize=(10, 7))
            lane_groups = list(hw_plot.groupby(["direction", "lane_id"]))
            colors = plt.cm.tab10.colors
            for idx, ((direction, lane), grp) in enumerate(lane_groups):
                c = colors[idx % len(colors)]
                label = f"{direction} {lane}"
                ax.scatter(grp["queue_position"], grp["time_headway_s"],
                           s=50, alpha=0.5, color=c, label=label, zorder=3)
                pos_m = grp.groupby("queue_position")["time_headway_s"].mean()
                ax.plot(pos_m.index, pos_m.values, "-o", color=c,
                        linewidth=1.5, markersize=5, alpha=0.8, zorder=4)

            if not hw_plot.empty:
                pos_stats_all = hw_plot.groupby("queue_position")["time_headway_s"].agg(["mean", "std", "count"])
                pos_stats_all["se"] = pos_stats_all["std"] / np.sqrt(pos_stats_all["count"])
                ax.errorbar(pos_stats_all.index, pos_stats_all["mean"],
                            yerr=pos_stats_all["se"],
                            fmt="ks--", linewidth=2.5, markersize=8, capsize=5,
                            label="Overall Mean ± SE", zorder=6)

            # Saturation headway: mean of all headways at queue positions >= 5.
            # Positions 1-4 are the "startup" zone where drivers are still
            # reacting and accelerating; from position 5 onward flow is saturated.
            SAT_POSITION = 5
            sat_vals_c = hw_plot[hw_plot["queue_position"] >= SAT_POSITION]["time_headway_s"]
            sat_c = sat_vals_c.mean() if len(sat_vals_c) > 0 else None

            if sat_c is not None:
                ax.axhline(sat_c, color="red", linestyle="--", linewidth=1.5,
                           label=f"$h_{{sat}}$ ≈ {sat_c:.2f} s", zorder=5)

                # ── Per-lane-cycle startup lost time ────────────────────────
                # Group by (direction, lane_id, green_start) so that each
                # row in the output is one lane in one signal cycle.
                # Grouping by green_start alone would pool all lanes that
                # share a phase, summing their L values and producing a
                # number 4-5× too large.
                startup_rows = []
                group_keys = ["direction", "lane_id", "green_start"]
                for (dir_val, lane_val, gs_val), cycle_grp in hw_plot.groupby(group_keys):
                    early = cycle_grp[cycle_grp["queue_position"] <= 4].sort_values("queue_position")
                    if early.empty:
                        continue
                    # Need at least positions 1 and 2 to compute a meaningful L
                    if len(early) < 2:
                        continue

                    headways_early = early["time_headway_s"].values
                    L_k = float(np.sum(headways_early - sat_c))

                    # t_until_sat: crossing time of the last captured early
                    # vehicle (position 4 if available, else highest present)
                    # minus the green start.
                    last_early = early.iloc[-1]
                    t_until_sat = float(last_early["crossing_time"] - gs_val)

                    # One column per position (h_1 … h_4); missing positions get NaN
                    h_by_pos = {}
                    for _, r in early.iterrows():
                        h_by_pos[f"h_{int(r['queue_position'])}"] = round(r["time_headway_s"], 3)

                    startup_rows.append({
                        "direction": dir_val,
                        "lane_id": lane_val,
                        "green_start": round(gs_val, 2),
                        "h_sat": round(sat_c, 3),
                        **h_by_pos,
                        "L": round(L_k, 3),
                        "t_until_saturation_s": round(t_until_sat, 2),
                    })

                if startup_rows:
                    startup_df = pd.DataFrame(startup_rows)

                    # Enforce column order: direction, lane_id, green_start,
                    # h_sat, h_1, h_2, h_3, h_4, … h_N, L, t_until_saturation_s
                    h_cols = sorted(
                        [c for c in startup_df.columns if c.startswith("h_") and c != "h_sat"],
                        key=lambda x: int(x.split("_")[1]),
                    )
                    ordered_cols = ["direction", "lane_id", "green_start", "h_sat"] + h_cols + ["L", "t_until_saturation_s"]
                    startup_df = startup_df[ordered_cols]

                    startup_csv = os.path.join(headway_dir, "startup_lost_time_per_cycle.csv")
                    startup_df.to_csv(startup_csv, sep=";", index=False)

                    mean_L = startup_df["L"].mean()
                    mean_t_sat = startup_df["t_until_saturation_s"].dropna().mean()
                    print(
                        f"  [startup lost time] {len(startup_df)} cycle(s): "
                        f"mean L = {mean_L:.2f} s, "
                        f"h_sat = {sat_c:.2f} s, "
                        f"mean t_until_sat = {mean_t_sat:.1f} s"
                    )

                    # ── Shade lost-time area on the plot ────────────────────
                    # For each queue position 1-4, shade between the per-position
                    # mean headway and the sat_c line, showing the excess time
                    # lost to startup delay.
                    mean_by_pos = hw_plot[hw_plot["queue_position"] <= 4].groupby(
                        "queue_position"
                    )["time_headway_s"].mean()

                    shade_positions = sorted(mean_by_pos.index)
                    shade_means = [mean_by_pos[p] for p in shade_positions]

                    if shade_positions:
                        ax.fill_between(
                            shade_positions, shade_means, sat_c,
                            where=[m > sat_c for m in shade_means],
                            alpha=0.18, color="red", zorder=2,
                            label=f"Startup lost time L = {mean_L:.2f} s",
                        )

                    # Annotation box — lower-right so it does not clash with
                    # the legend (upper-right).  Shows the three headline
                    # thesis numbers directly on the figure.
                    memo_lines = [
                        f"$h_{{sat}}$ = {sat_c:.2f} s",
                        f"$L$ = {mean_L:.2f} s",
                        f"$t_{{sat}}$ = {mean_t_sat:.1f} s",
                    ]
                    ax.text(
                        0.97, 0.03,
                        "\n".join(memo_lines),
                        transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                                  edgecolor="red", linewidth=1.2, alpha=0.9),
                    )

            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Queue Position")
            ax.set_ylabel("Time Headway (s)")
            ax.set_title("Discharge Headway — All Lanes Combined")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3)
            max_pos_c = int(hw_plot["queue_position"].max()) if not hw_plot.empty else MAX_QUEUE_POS
            ax.set_xticks(range(0, max_pos_c + 2))

            fig_comb.tight_layout()
            comb_path = os.path.join(headway_dir, "headway_all_lanes_combined.png")
            _save_fig_formats(fig_comb, comb_path)
            plt.close(fig_comb)
            dbg(f"  Combined headway plot saved")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(self):
        """Main entry point -- calls all sub-methods in order."""
        if self.plot_df.empty:
            print("DataFrame is empty or no valid trajectories. Skipping space-time diagram.")
            return

        # 1. Write diagnostics / rejected trajectories
        self._write_diagnostics()

        # 2. Main diagram
        #self._plot_main_diagram()

        # 3. Green inference audit
        self._write_audit()

        # 4. Per-movement / per-lane diagrams
        self._plot_per_movement_per_lane()

        # 5. Results table (includes phase consistency checks + signal timing diagrams)
        self._build_results_table()

        # 6. Headway analysis
        self._headway_analysis()


# ---------------------------------------------------------------------------
# Backward-compatible bridge function
# ---------------------------------------------------------------------------

def generate_space_time_diagram(
    df: pd.DataFrame,
    out_path: str,
    g_df: pd.DataFrame = None,
    allowed_vehicle_ids=None,
    title: str = "Space-Time Diagram",
    x_col: str = "Ortho_X",
    y_col: str = "Ortho_Y",
    time_col: str = "time_s",
    vehicle_id_col: str = "Vehicle_ID",
    vehicle_state_sequence: dict = None,
    general_dir: str = None,
    headway_dir: str = None,
    vehicle_to_movement: dict = None,
):
    """Backward-compatible wrapper around :class:`SpaceTimeDiagram`.

    Generates a Space-Time diagram where:
    - X-axis: Time (s)
    - Y-axis: Distance relative to Stop Line (m)
      (Negative = approaching, 0 = at stop line, Positive = inside/past intersection)

    The Stop Line is defined as the point where the vehicle exits the approach section
    and enters the intersection (or turns).
    """
    if df is None or df.empty:
        print("DataFrame is empty. Skipping space-time diagram.")
        return

    diagram = SpaceTimeDiagram(
        df=df,
        out_path=out_path,
        g_df=g_df,
        allowed_vehicle_ids=allowed_vehicle_ids,
        title=title,
        x_col=x_col,
        y_col=y_col,
        time_col=time_col,
        vehicle_id_col=vehicle_id_col,
        vehicle_state_sequence=vehicle_state_sequence,
        general_dir=general_dir,
        headway_dir=headway_dir,
        vehicle_to_movement=vehicle_to_movement,
    )
    diagram.generate()
    return diagram
