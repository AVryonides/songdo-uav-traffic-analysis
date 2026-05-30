# vehicle_grouping.py
"""
Vehicle grouping / section-based classification functions extracted from uavsongdopie_Q.py.
"""

import itertools
import json
import re

import numpy as np
import pandas as pd
from collections import Counter
from typing import Optional, Tuple
from matplotlib.path import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def build_movement_to_ids_for_workset(
    work_vehicle_ids,
    start_end_section: dict,
    vehicle_to_movement: dict = None,
) -> Tuple[dict, list]:
    """
    Build {movement: [vehicle_ids]} for the current workset.
    """
    # Import here to avoid circular imports at module level
    from .r_rules import infer_r_movement as infer_q_movement

    movement_to_ids = {}
    unassigned_ids = []
    vehicle_to_movement = vehicle_to_movement or {}

    for vehicle_id in work_vehicle_ids:
        movement = vehicle_to_movement.get(vehicle_id, None)
        if movement is not None:
            movement_to_ids.setdefault(movement, []).append(vehicle_id)
            continue

        if vehicle_id not in start_end_section:
            unassigned_ids.append(vehicle_id)
            continue

        start_sec, end_sec = start_end_section[vehicle_id]
        movement = infer_q_movement(start_sec, end_sec)
        movement_to_ids.setdefault(movement, []).append(vehicle_id)

    return movement_to_ids, unassigned_ids


def pick_k_by_silhouette(X: np.ndarray, k_min: int = 3, k_max: int = 5):
    best = None
    # Need at least k+1 points for silhouette; keep safe bounds
    max_k = min(k_max, max(2, len(X) - 1))
    for k in range(k_min, max_k + 1):
        if len(X) <= k:
            continue
        km = KMeans(n_clusters=k, n_init=20, random_state=0)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        s = silhouette_score(X, labels)
        if best is None or s > best[0]:
            best = (s, k, km)
    return best  # (score, k, model) or None

def label_clusters_compass(section_df: pd.DataFrame, center_xy: Tuple[float, float]) -> dict:
    # Ortho coords typically have y increasing downward => positive dy is 'South'
    out = section_df.copy()
    out["dx"] = out["cx"] - center_xy[0]
    out["dy"] = out["cy"] - center_xy[1]
    cluster_dirs = {}
    for c, g in out.groupby("cluster"):
        mx = float(g["dx"].mean())
        my = float(g["dy"].mean())
        if abs(mx) >= abs(my):
            cluster_dirs[int(c)] = "E" if mx > 0 else "W"
        else:
            cluster_dirs[int(c)] = "S" if my > 0 else "N"
    return cluster_dirs


def assign_clusters_to_compass_opt(section_df: pd.DataFrame, center_xy: Tuple[float, float], k: int) -> dict:
    """Assign k clusters (k=3 or 4) to unique compass labels among {N,E,S,W} by maximizing alignment."""
    # y increases downward in orthophoto coords => South is +dy, North is -dy
    compass_vecs = {
        "N": np.array([0.0, -1.0]),
        "E": np.array([1.0,  0.0]),
        "S": np.array([0.0,  1.0]),
        "W": np.array([-1.0, 0.0]),
    }

    dirs = {}
    for cid, g in section_df.groupby("cluster"):
        dx = float((g["cx"] - center_xy[0]).mean())
        dy = float((g["cy"] - center_xy[1]).mean())
        v = np.array([dx, dy], dtype=float)
        n = float(np.linalg.norm(v))
        dirs[int(cid)] = v / n if n > 1e-9 else np.array([0.0, 0.0])

    clusters = sorted(dirs.keys())
    labels = ["N", "E", "S", "W"]

    best_score = None
    best_map = None

    if k == 4:
        for perm in itertools.permutations(labels, 4):
            score = 0.0
            mapping = {}
            for cid, lab in zip(clusters, perm):
                score += float(np.dot(dirs[cid], compass_vecs[lab]))
                mapping[cid] = lab
            if best_score is None or score > best_score:
                best_score, best_map = score, mapping
    else:
        # k==3: choose best 3-of-4 label set and best assignment
        for subset in itertools.combinations(labels, 3):
            for perm in itertools.permutations(subset, 3):
                score = 0.0
                mapping = {}
                for cid, lab in zip(clusters, perm):
                    score += float(np.dot(dirs[cid], compass_vecs[lab]))
                    mapping[cid] = lab
                if best_score is None or score > best_score:
                    best_score, best_map = score, mapping

    return best_map if best_map is not None else {cid: f"A{cid}" for cid in clusters}

def build_section_to_approach_map(seg_df: pd.DataFrame):
    """
    DEPRECATED: Direction/approach mapping has been disabled for generic operation.
    This function now returns empty results to indicate no direction-based grouping.
    """
    from .geometry import compute_intersection_center_ortho
    center_xy = compute_intersection_center_ortho(seg_df)
    return center_xy, {}

def group_vehicles_by_sections(
    df: pd.DataFrame,
    g_df: pd.DataFrame,
    allowed_vehicle_ids,
    time_col: str = "time_s",
    x_col: str = "Ortho_X",
    y_col: str = "Ortho_Y",
    N: int = 30,                 # bigger vote window
    contain_radius: float = 3.0, # polygon tolerance in pixels
    snap_dist_thresh: float = 40.0,  # snap fallback threshold (pixels)
    closest_fallback_thresh: float = 200.0, # guarded closest-section fallback (pixels)
    grouping_mode: str = "rules",  # 'rules'|'auto'|'manual'
    manual_mapping_file: str = None, # JSON file path when grouping_mode=='manual'
    export_sections_path: str = None, # optional path to export section centroids for inspection

):
    """
    For each vehicle:
      - vote start_section from first N points
      - vote end_section from last N points
    Improvements:
      - larger N
      - contains_point radius tolerance
      - snap-to-nearest polygon if containment fails
    """
    from .geometry import _build_g_section_polys, compute_section_centroids
    from .r_rules import infer_r_movement as infer_q_movement, infer_r_movement_from_lane_sequence as infer_q_movement_from_lane_sequence, R_MOVEMENT_RULES as Q_MOVEMENT_RULES

    polys = _build_g_section_polys(g_df)

    d = df[df["Vehicle_ID"].isin(set(allowed_vehicle_ids))].copy()
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=[time_col, x_col, y_col])
    d = d.sort_values(["Vehicle_ID", time_col])

    # --- helpers for snap distance ---
    def point_to_segment_dist(px, py, ax, ay, bx, by):
        abx, aby = (bx - ax), (by - ay)
        apx, apy = (px - ax), (py - ay)
        denom = abx*abx + aby*aby
        if denom == 0:
            return float(np.hypot(apx, apy))
        t = (apx*abx + apy*aby) / denom
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        cx, cy = ax + t*abx, ay + t*aby
        return float(np.hypot(px - cx, py - cy))

    def point_to_quad_dist(px, py, quad):
        # 0 if inside, else distance to boundary
        path = Path(quad)
        if path.contains_point((px, py), radius=contain_radius):
            return 0.0
        dmin = 1e18
        for i in range(4):
            ax, ay = quad[i]
            bx, by = quad[(i+1) % 4]
            dmin = min(dmin, point_to_segment_dist(px, py, ax, ay, bx, by))
        return float(dmin)

    def point_to_section_lane_single(x: float, y: float) -> Tuple[Optional[str], Optional[int]]:
        # 1) containment
        for p in polys:
            if p["path"].contains_point((x, y), radius=contain_radius):
                return p["section_id"], int(p["lane_idx"])

        # 2) snap nearest
        best = None
        best_d = 1e18
        for p in polys:
            dquad = point_to_quad_dist(x, y, p["verts"])
            if dquad < best_d:
                best_d = dquad
                best = (p["section_id"], int(p["lane_idx"]))

        if best is None:
            return None, None
        if best_d <= snap_dist_thresh:
            return best
        if best_d <= closest_fallback_thresh:
            return best
        return None, None

    def build_vehicle_state_sequence(gveh: pd.DataFrame, max_points: int = 240) -> list:
        # Sample trajectory points to keep runtime bounded.
        if len(gveh) > max_points:
            idx = np.linspace(0, len(gveh) - 1, max_points, dtype=int)
            gs = gveh.iloc[idx]
        else:
            gs = gveh

        states = []
        for _, r in gs.iterrows():
            x = float(r[x_col]); y = float(r[y_col])
            sec, li = point_to_section_lane_single(x, y)
            if sec is None or li is None:
                continue
            st = f"{sec}_{int(li)}"
            if not states or states[-1] != st:
                states.append(st)
        return states

    def vote_section_lane(points_df):
        """
        Returns:
          sec, li: voted section and lane_idx (or None, None)
          method: "hit" if containment votes used, "snap" if snap votes used, "none" if nothing
          min_best_d: the smallest distance-to-polygon-boundary seen during snap search in this window
        """
        hits = []
        snaps = []
        min_best_d = 1e18

        for _, r in points_df.iterrows():
            x = float(r[x_col]); y = float(r[y_col])

            # 1) containment with radius tolerance
            found = None
            for p in polys:
                if p["path"].contains_point((x, y), radius=contain_radius):
                    found = (p["section_id"], p["lane_idx"])
                    break
            if found is not None:
                hits.append(found)
                continue

            # 2) snap fallback: nearest polygon boundary within threshold
            best = None
            best_d = 1e18
            for p in polys:
                dquad = point_to_quad_dist(x, y, p["verts"])
                if dquad < best_d:
                    best_d = dquad
                    best = (p["section_id"], p["lane_idx"])

            if best_d < min_best_d:
                min_best_d = best_d

            if best is not None and best_d <= snap_dist_thresh:
                snaps.append(best)

        # --- decide votes AFTER scanning the whole window ---
        use = hits if hits else snaps

        # 3) guarded "closest polygon" fallback (only if still empty)
        if not use:
            # Use first point of the window (matches your "closest to first/last coordinate" idea)
            x_rep = float(points_df[x_col].iloc[0])
            y_rep = float(points_df[y_col].iloc[0])

            best = None
            best_d = 1e18
            for p in polys:
                dquad = point_to_quad_dist(x_rep, y_rep, p["verts"])
                if dquad < best_d:
                    best_d = dquad
                    best = (p["section_id"], p["lane_idx"])

            if best is not None and best_d <= closest_fallback_thresh:
                # method is "snap" conceptually (closest boundary), but mark it separately if you want
                if min_best_d == 1e18:
                    min_best_d = best_d
                return best[0], best[1], "snap", min_best_d

            if min_best_d == 1e18:
                min_best_d = None
            return None, None, "none", min_best_d

        (sec, li), _ = Counter(use).most_common(1)[0]
        method = "hit" if hits else "snap"
        if min_best_d == 1e18:
            min_best_d = None
        return sec, li, method, min_best_d



    # --- helper: extract start/end section from CSV Road_Section column ---
    road_sec_col = "Road_Section"
    lane_num_col = "Lane_Number"
    has_road_section = road_sec_col in d.columns

    def _csv_start_end(g_vehicle: pd.DataFrame):
        """
        Read the first non-blank and last non-blank Road_Section from the CSV.
        Returns (start_sec, start_lane, end_sec, end_lane) or Nones.
        """
        if not has_road_section:
            return None, None, None, None

        rs = g_vehicle[road_sec_col].astype(str).str.strip()
        valid_mask = rs.ne("") & rs.ne("nan") & rs.notna()
        valid_rows = g_vehicle[valid_mask]
        if valid_rows.empty:
            return None, None, None, None

        first_row = valid_rows.iloc[0]
        last_row = valid_rows.iloc[-1]

        s_sec = str(first_row[road_sec_col]).strip()
        e_sec = str(last_row[road_sec_col]).strip()
        s_sec = s_sec if s_sec and s_sec != "nan" else None
        e_sec = e_sec if e_sec and e_sec != "nan" else None

        # Lane from same rows
        s_lane = first_row.get(lane_num_col)
        e_lane = last_row.get(lane_num_col)
        try:
            s_lane = int(float(s_lane)) if pd.notna(s_lane) else None
        except (ValueError, TypeError):
            s_lane = None
        try:
            e_lane = int(float(e_lane)) if pd.notna(e_lane) else None
        except (ValueError, TypeError):
            e_lane = None

        return s_sec, s_lane, e_sec, e_lane

    start_end_section = {}
    start_end_laneidx = {}
    vehicle_state_sequence = {}
    vehicle_to_movement = {}

    # --- diagnostics counters ---
    start_method_counts = Counter()  # "csv" | "hit" | "snap" | "none"
    end_method_counts = Counter()
    none_start_min_d = []
    none_end_min_d = []

    for vid, g in d.groupby("Vehicle_ID"):
        # --- Primary: use CSV Road_Section if it yields a valid movement ---
        csv_s_sec, csv_s_lane, csv_e_sec, csv_e_lane = _csv_start_end(g)
        csv_valid = False

        if csv_s_sec is not None and csv_e_sec is not None:
            # Check if this start→end pair matches any movement rule
            csv_movement = infer_q_movement(csv_s_sec, csv_e_sec)
            if csv_movement != "UNASSIGNED":
                s_sec, e_sec = csv_s_sec, csv_e_sec
                s_li, e_li = csv_s_lane, csv_e_lane
                s_method, e_method = "csv", "csv"
                csv_valid = True

        if not csv_valid:
            # --- Fallback: polygon voting ---
            g_head = g.head(N)
            g_tail = g.tail(N)
            s_sec, s_li, s_method, s_min_d = vote_section_lane(g_head)
            e_sec, e_li, e_method, e_min_d = vote_section_lane(g_tail)

            if s_method == "none":
                none_start_min_d.append(s_min_d)
            if e_method == "none":
                none_end_min_d.append(e_min_d)

        start_method_counts[s_method] += 1
        end_method_counts[e_method] += 1

        start_end_section[int(vid)] = (s_sec, e_sec)
        start_end_laneidx[int(vid)] = (s_li, e_li)
        vehicle_state_sequence[int(vid)] = build_vehicle_state_sequence(g)

    # --- print diagnostics summary ---
    def _summarize_min_d(arr):
        vals = [v for v in arr if (v is not None and v != 1e18)]
        if not vals:
            return "no distances"
        vals = np.array(vals, dtype=float)
        return (f"min={vals.min():.1f}, p25={np.percentile(vals,25):.1f}, "
                f"median={np.percentile(vals,50):.1f}, p75={np.percentile(vals,75):.1f}, "
                f"p90={np.percentile(vals,90):.1f}")

    n_attempted = len(start_end_section)
    if n_attempted > 0:
        print("\n=== SECTION ASSIGNMENT DIAGNOSTICS (start/end) ===")
        print(f"Road_Section column available: {has_road_section}")
        print(f"Polygon fallback — N: {N}, contain_radius: {contain_radius}, snap_dist_thresh: {snap_dist_thresh}")

        print("\nStart methods:")
        for method in ["csv", "hit", "snap", "none"]:
            cnt = start_method_counts.get(method, 0)
            pct = 100 * cnt / n_attempted if n_attempted else 0
            print(f"  {method:4s}: {cnt} ({pct:.1f}%)")
        if none_start_min_d:
            print(f"  none best_d stats: {_summarize_min_d(none_start_min_d)}")

        print("\nEnd methods:")
        for method in ["csv", "hit", "snap", "none"]:
            cnt = end_method_counts.get(method, 0)
            pct = 100 * cnt / n_attempted if n_attempted else 0
            print(f"  {method:4s}: {cnt} ({pct:.1f}%)")
        if none_end_min_d:
            print(f"  none best_d stats: {_summarize_min_d(none_end_min_d)}")
        print("=================================================\n")

    # ========================================================================================
    # Q INTERSECTION: Movement-based classification for section transitions
    # ========================================================================================

    # Optionally export section centroids for manual inspection/labeling
    if export_sections_path is not None and not g_df.empty:
        try:
            sec_cent = compute_section_centroids(g_df)
            sec_cent.to_csv(export_sections_path, index=False)
            print(f"Wrote section centroids for inspection: {export_sections_path}")
        except Exception as e:
            print(f"Failed to export sections to {export_sections_path}: {e}")

    # Classify each vehicle by movement based on lane-aware section-lane sequence.
    groups = {}
    for vid, (start_sec, end_sec) in start_end_section.items():
        seq = vehicle_state_sequence.get(int(vid), [])
        movement = infer_q_movement_from_lane_sequence(seq)
        if movement == 'UNASSIGNED':
            # Keep section-based fallback when sequence is too sparse/noisy.
            movement = infer_q_movement(start_sec, end_sec)
        if movement not in groups:
            groups[movement] = []
        groups[movement].append(vid)
        vehicle_to_movement[int(vid)] = movement

    # Ensure all movement categories exist (even if empty)
    for _, _, movement_name in Q_MOVEMENT_RULES:
        if movement_name not in groups:
            groups[movement_name] = []

    return groups, start_end_section, start_end_laneidx, vehicle_to_movement, vehicle_state_sequence

def print_section_group_report_limited(
    groups: dict,
    start_end_section: dict,
    start_end_laneidx: dict,
    max_ids: int = 20
):
    print("\n=== MOVEMENT CLASSIFICATION ===")
    movement_order = ['1->2','1->3','1->4','2->1','2->3','2->4','3->1','3->2','3->4','4->1','4->2','4->3']
    total_classified = 0
    for movement in movement_order:
        if movement not in groups:
            continue
        vids = groups[movement]
        total_classified += len(vids)
        print(f"  {movement}: {len(vids)} vehicles")
    unassigned = len(start_end_section) - total_classified
    if unassigned > 0:
        print(f"  UNASSIGNED: {unassigned} vehicles")
    print(f"  Total classified: {total_classified}/{len(start_end_section)}")
    print("================================\n")

def build_direction_worksets(groups: dict) -> dict:
    """
    Returns movement groups organized by direction.
    Returns a dict with:
      - direction_to_ids: {movement: [Vehicle_IDs]}
      - work_vehicle_ids: sorted unique list of all vehicles
    """
    direction_to_ids = {k: sorted(v) for k, v in groups.items()}
    work_vehicle_ids = sorted({vid for vids in groups.values() for vid in vids})
    return {
        "direction_to_ids": direction_to_ids,
        "work_vehicle_ids": work_vehicle_ids,
    }

def filter_df_by_vehicle_ids(df: pd.DataFrame, vehicle_ids, vehicle_id_col: str = "Vehicle_ID") -> pd.DataFrame:
    """Convenience filter for distributions."""
    keep = set(map(int, vehicle_ids))
    d = df.copy()
    d[vehicle_id_col] = pd.to_numeric(d[vehicle_id_col], errors="coerce")
    d = d.dropna(subset=[vehicle_id_col])
    d[vehicle_id_col] = d[vehicle_id_col].astype(int)
    return d[d[vehicle_id_col].isin(keep)].copy()
