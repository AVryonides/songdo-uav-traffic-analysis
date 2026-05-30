# geometry.py
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from typing import Optional, Tuple, List
from collections import Counter


def compute_bbox_latlon(df: pd.DataFrame, lat_col: str = "Latitude", lon_col: str = "Longitude", q: float = 0.005, pad: float = 0.00005):
    d = df[[lat_col, lon_col]].dropna()
    if d.empty:
        return None
    lat_lo, lat_hi = d[lat_col].quantile([q, 1 - q]).tolist()
    lon_lo, lon_hi = d[lon_col].quantile([q, 1 - q]).tolist()
    lat_lo -= pad; lat_hi += pad
    lon_lo -= pad; lon_hi += pad
    return [(lat_lo, lon_lo), (lat_lo, lon_hi), (lat_hi, lon_hi), (lat_hi, lon_lo)]

def compute_center_latlon(df: pd.DataFrame, lat_col: str = "Latitude", lon_col: str = "Longitude"):
    d = df[[lat_col, lon_col]].dropna()
    if d.empty:
        return None
    return (float(d[lat_col].median()), float(d[lon_col].median()))


def _build_g_section_polys(g_df: pd.DataFrame):
    required_cols = {"Section", "Lane", "tlx", "tly", "blx", "bly", "brx", "bry", "trx", "try"}
    missing = required_cols - set(g_df.columns)
    if missing:
        raise ValueError(f"Segmentation CSV is missing columns: {missing}")

    polys = []
    for _, r in g_df.iterrows():
        section_id = str(r["Section"]).strip()
        lane_idx = int(r["Lane"])

        verts = np.array([
            (float(r["tlx"]), float(r["tly"])),
            (float(r["trx"]), float(r["try"])),
            (float(r["brx"]), float(r["bry"])),
            (float(r["blx"]), float(r["bly"])),
        ], dtype=float)

        polys.append({
            "section_id": section_id,
            "lane_idx": lane_idx,
            "verts": verts,
            "path": MplPath(verts),
        })

    return polys


def _point_to_section_and_lane(x: float, y: float, polys) -> Tuple[Optional[str], Optional[int]]:
    """
    For a point (x,y), returns:
      (section_id, lane_idx) e.g. ("1_4", 3)
    or (None, None) if not inside any polygon.
    """
    for p in polys:
        if p["path"].contains_point((x, y)):
            return p["section_id"], p["lane_idx"]
    return None, None


def trim_to_intersection(
    df: pd.DataFrame,
    g_df: pd.DataFrame,
    x_col: str = "Ortho_X",
    y_col: str = "Ortho_Y",
    vehicle_id_col: str = "Vehicle_ID",
    margin: float = 50.0,
) -> pd.DataFrame:
    """Remove trajectory rows that fall outside the intersection boundary.

    Instead of discarding entire vehicles, this trims each vehicle's
    trajectory to only the timestamps where it is within (or near)
    the segmentation polygons.

    Parameters
    ----------
    df : Full trajectory DataFrame.
    g_df : Segmentation CSV DataFrame (Section, Lane, tlx, tly, …).
    margin : Extra buffer (in coordinate units) around the bounding box
        of all segmentation polygons.  Points within this expanded bbox
        are kept, so vehicles approaching the intersection aren't
        clipped too aggressively.

    Returns
    -------
    Trimmed DataFrame (rows outside the intersection removed).
    """
    polys = _build_g_section_polys(g_df)
    if not polys:
        return df

    # Build the bounding box of all segmentation polygons + margin
    all_verts = np.vstack([p["verts"] for p in polys])
    x_min = float(all_verts[:, 0].min()) - margin
    x_max = float(all_verts[:, 0].max()) + margin
    y_min = float(all_verts[:, 1].min()) - margin
    y_max = float(all_verts[:, 1].max()) + margin

    xs = df[x_col].values.astype(float)
    ys = df[y_col].values.astype(float)

    # Fast bbox pre-filter
    in_bbox = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)

    n_before = len(df)
    n_vids_before = df[vehicle_id_col].nunique()

    df_trimmed = df[in_bbox].copy()

    n_after = len(df_trimmed)
    n_vids_after = df_trimmed[vehicle_id_col].nunique()
    n_vids_removed = n_vids_before - n_vids_after

    print(f"[TRIM] Intersection boundary filter: {n_before} → {n_after} rows "
          f"({n_before - n_after} outside, {n_vids_removed} vehicles fully outside)")

    return df_trimmed


def _quad_centroid(row: pd.Series) -> Tuple[float, float]:
    xs = np.array([row["tlx"], row["blx"], row["brx"], row["trx"]], dtype=float)
    ys = np.array([row["tly"], row["bly"], row["bry"], row["try"]], dtype=float)
    return float(xs.mean()), float(ys.mean())

def compute_section_centroids(seg_df: pd.DataFrame) -> pd.DataFrame:
    tmp = seg_df.copy()
    tmp[["cx", "cy"]] = tmp.apply(lambda r: pd.Series(_quad_centroid(r)), axis=1)
    sec = tmp.groupby("Section")[["cx", "cy"]].mean().reset_index()
    return sec

def compute_intersection_center_ortho(seg_df: pd.DataFrame) -> Tuple[float, float]:
    sec = compute_section_centroids(seg_df)
    return float(sec["cx"].median()), float(sec["cy"].median())


def _point_to_segment_distance(px, py, ax, ay, bx, by):
    """Euclidean distance from point P to segment AB (all floats)."""
    abx, aby = (bx - ax), (by - ay)
    apx, apy = (px - ax), (py - ay)
    denom = abx*abx + aby*aby
    if denom == 0:
        return float(np.hypot(apx, apy))
    t = (apx*abx + apy*aby) / denom
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    cx, cy = ax + t*abx, ay + t*aby
    return float(np.hypot(px - cx, py - cy))

def _point_to_quad_distance(px, py, quad_xy):
    """Distance from point to the boundary of a quadrilateral (0 if inside)."""
    path = MplPath(quad_xy)
    if path.contains_point((px, py)):
        return 0.0
    # distance to edges
    dmin = 1e18
    for i in range(4):
        ax, ay = quad_xy[i]
        bx, by = quad_xy[(i+1) % 4]
        d = _point_to_segment_distance(px, py, ax, ay, bx, by)
        if d < dmin:
            dmin = d
    return float(dmin)

def build_g_lane_geoms(g_df: pd.DataFrame):
    """Build geometry objects for each lane polygon in G.csv."""
    geoms = []
    for _, row in g_df.iterrows():
        quad = np.array(
            [
                (float(row['tlx']), float(row['tly'])),
                (float(row['blx']), float(row['bly'])),
                (float(row['brx']), float(row['bry'])),
                (float(row['trx']), float(row['try'])),
            ],
            dtype=float,
        )
        geoms.append(
            {
                'section': str(row['Section']),
                'lane_idx': int(row['Lane']),   # lane index inside section
                'quad': quad,
                'path': MplPath(quad),
            }
        )
    return geoms

def infer_vehicle_sectionlane_from_g(
    vehicle_points_xy: np.ndarray,
    g_geoms,
    *,
    contain_radius: float = 0.0,
    snap_dist_thresh: float = 120.0,
    closest_fallback_thresh: float = 200.0,
):
    """
    Returns:
      (section, lane_idx) or (None, None),
      confidence in [0,1],
      method: 'contain' | 'snap' | 'none'
    Strategy:
      - First: strict containment (with optional radius tolerance).
      - If zero containment hits: snap each point to nearest polygon within snap_dist_thresh and vote.
    """
    contain_hits = []
    snap_hits = []

    # 1) containment vote
    for x, y in vehicle_points_xy:
        if np.isnan(x) or np.isnan(y):
            continue
        found = None
        for g in g_geoms:
            if g['path'].contains_point((float(x), float(y)), radius=contain_radius):
                found = (g['section'], g['lane_idx'])
                break
        if found is not None:
            contain_hits.append(found)

    if contain_hits:
        counts = Counter(contain_hits)
        (sec, li), cmax = counts.most_common(1)[0]
        conf = cmax / len(contain_hits)
        return sec, li, float(conf), 'contain'

    # 2) snap-to-nearest (only if within threshold)
    for x, y in vehicle_points_xy:
        if np.isnan(x) or np.isnan(y):
            continue
        best = None
        best_d = 1e18
        for g in g_geoms:
            d = _point_to_quad_distance(float(x), float(y), g['quad'])
            if d < best_d:
                best_d = d
                best = (g['section'], g['lane_idx'])
        if best is not None and best_d <= snap_dist_thresh:
            snap_hits.append(best)

    if snap_hits:
        counts = Counter(snap_hits)
        (sec, li), cmax = counts.most_common(1)[0]
        conf = cmax / len(snap_hits)
        return sec, li, float(conf), 'snap'

    return None, None, 0.0, 'none'


def _time_uniform_sample(g: pd.DataFrame, n: int, time_col: str = "time_s") -> pd.DataFrame:
    """
    Sample n rows from a vehicle trajectory roughly uniformly over time.
    Falls back to random if time_col is missing.
    """
    if len(g) <= n:
        return g

    if time_col in g.columns:
        g = g.sort_values(time_col)
        idx = np.linspace(0, len(g) - 1, n).round().astype(int)
        return g.iloc[idx]
    else:
        return g.sample(n=n, random_state=0)


def build_sectionlane_to_lane_number_map(
    df: pd.DataFrame,
    g_geoms,
    *,
    vehicle_id_col: str = 'Vehicle_ID',
    lane_col_original: str = 'Lane_Number',
    x_col: str = 'Ortho_X',
    y_col: str = 'Ortho_Y',
    time_col: str = 'time_s',
    sample_points_per_vehicle: int = 200,
    contain_radius: float = 0.0,
):
    """Learn mapping: (Section, LaneIdx in G.csv) -> dataset Lane_Number using already-labeled vehicles."""
    mapping_records = []

    # per-vehicle lane label (mode) from dataset
    d0 = df[[vehicle_id_col, lane_col_original]].copy()
    d0[lane_col_original] = pd.to_numeric(d0[lane_col_original], errors='coerce')

    def pick_lane(series: pd.Series):
        s = series.dropna()
        if s.empty:
            return np.nan
        m = s.mode()
        return float(m.iloc[0]) if len(m) else float(s.iloc[0])

    lane_by_vehicle = d0.groupby(vehicle_id_col)[lane_col_original].apply(pick_lane)

    labeled_vids = lane_by_vehicle[~lane_by_vehicle.isna()].index.astype(int).tolist()
    if not labeled_vids:
        return {}

    dxy = df[[vehicle_id_col, x_col, y_col, time_col]].copy()
    dxy[x_col] = pd.to_numeric(dxy[x_col], errors='coerce')
    dxy[y_col] = pd.to_numeric(dxy[y_col], errors='coerce')
    dxy = dxy.dropna(subset=[x_col, y_col])

    for vid in labeled_vids:
        gveh = dxy[dxy[vehicle_id_col] == vid]
        if gveh.empty:
            continue
        g_s = _time_uniform_sample(gveh, sample_points_per_vehicle, time_col=time_col)
        pts = g_s[[x_col, y_col]].to_numpy(dtype=float)

        # vote section/lane per point (containment only for mapping)
        for x, y in pts:
            sec = li = None
            for gg in g_geoms:
                if gg['path'].contains_point((float(x), float(y)), radius=contain_radius):
                    sec, li = gg['section'], gg['lane_idx']
                    break
            if sec is None:
                continue
            mapping_records.append((sec, li, int(lane_by_vehicle.loc[vid])))

    if not mapping_records:
        return {}

    rec_df = pd.DataFrame(mapping_records, columns=['Section', 'LaneIdx', 'LaneNumber'])
    # mode LaneNumber per (Section, LaneIdx)
    sl_to_ln = rec_df.groupby(['Section', 'LaneIdx'])['LaneNumber'].agg(lambda s: s.mode().iloc[0]).to_dict()
    return {(k[0], int(k[1])): int(v) for k, v in sl_to_ln.items()}
