# lane_inference.py
"""
Lane inference functions extracted from uavsongdopie_Q.py.
"""

import random

import numpy as np
import pandas as pd
from collections import Counter
from matplotlib.path import Path as MplPath
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


def summarize_lane_numbers(
    df: pd.DataFrame,
    vehicle_id_col: str = "Vehicle_ID",
    lane_col: str = "Lane_Number",
    allowed_vehicle_ids=None,
    lane_pick: str = "mode",
):
    """
    Returns:
      lane_counts: {lane_int: count_of_vehicles}
      lane_to_vehicle_ids: {lane_int: [Vehicle_IDs...]}
      missing_lane_vehicle_ids: [Vehicle_IDs with no lane anywhere]
    """

    d = df[[vehicle_id_col, lane_col]].copy()

    if allowed_vehicle_ids is not None:
        allowed_set = set(allowed_vehicle_ids)
        d = d[d[vehicle_id_col].isin(allowed_set)]

    d[lane_col] = pd.to_numeric(d[lane_col], errors="coerce")

    def pick_lane(series: pd.Series):
        s = series.dropna()
        if s.empty:
            return np.nan

        if lane_pick == "first":
            return float(s.iloc[0])

        m = s.mode()
        return float(m.iloc[0]) if len(m) else float(s.iloc[0])

    per_vehicle_lane = d.groupby(vehicle_id_col)[lane_col].apply(pick_lane)

    missing_ids = per_vehicle_lane[per_vehicle_lane.isna()].index.astype(int).tolist()

    per_vehicle_lane_nonnull = per_vehicle_lane.dropna().astype(int)
    lane_to_ids = (
        per_vehicle_lane_nonnull.groupby(per_vehicle_lane_nonnull)
        .apply(lambda s: sorted(s.index.astype(int).tolist()))
        .to_dict()
    )

    lane_counts = {int(lane): len(ids) for lane, ids in lane_to_ids.items()}

    return lane_counts, lane_to_ids, missing_ids


def print_lane_summary(title, lane_counts, lane_to_ids, missing_ids, max_ids_per_lane: int = 15):
    """Compact lane summary (doesn't spam the console with huge ID lists)."""
    print(f"\n=== {title} ===")
    if not lane_counts:
        print("No lane data found (all missing).")
    else:
        for lane in sorted(lane_counts.keys()):
            ids = lane_to_ids[lane]
            if max_ids_per_lane <= 0:
                print(f"Lane {lane}: {lane_counts[lane]} vehicles (ids hidden)")
            else:
                shown = ids[:max_ids_per_lane]
                extra = "" if len(ids) <= max_ids_per_lane else f" ... (+{len(ids)-max_ids_per_lane} more)"
                print(f"Lane {lane}: {lane_counts[lane]} vehicles")
                print(f"  Vehicle_IDs: {shown}{extra}")

    if missing_ids:
        if max_ids_per_lane <= 0:
            print(f"\nVehicles with NO lane label: {len(missing_ids)} (ids hidden)")
        else:
            shown = missing_ids[:max_ids_per_lane]
            extra = "" if len(missing_ids) <= max_ids_per_lane else f" ... (+{len(missing_ids)-max_ids_per_lane} more)"
            print(f"\nVehicles with NO lane label: {len(missing_ids)}")
            print(f"  Vehicle_IDs: {shown}{extra}")

    print("================================\n")


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


def build_vehicle_feature_table(
    df: pd.DataFrame,
    vehicle_id_col: str = "Vehicle_ID",
    lane_col: str = "Lane_Number",
    time_col: str = "time_s",
    x_col: str = "Ortho_X",
    y_col: str = "Ortho_Y",
    speed_col: str = "Vehicle_Speed",
    n_edge: int = 10,
) -> pd.DataFrame:
    """
    One row per vehicle with features + per-vehicle lane label (mode).
    Features (reviewer-friendly):
      start(x,y), end(x,y), mean(x,y), displacement(dx,dy), duration(T),
      speed mean + p90 (if available),
      heading encoded as cos/sin from displacement.
    """
    d = df[[vehicle_id_col, lane_col, time_col, x_col, y_col] + ([speed_col] if speed_col in df.columns else [])].copy()

    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d[lane_col] = pd.to_numeric(d[lane_col], errors="coerce")
    if speed_col in d.columns:
        d[speed_col] = pd.to_numeric(d[speed_col], errors="coerce")

    d = d.dropna(subset=[vehicle_id_col, time_col, x_col, y_col])
    d = d.sort_values([vehicle_id_col, time_col])

    rows = []
    for vid, g in d.groupby(vehicle_id_col):
        g = g.sort_values(time_col)

        # per-vehicle lane label (mode) from original Lane_Number (may be NaN)
        lane_vals = g[lane_col].dropna()
        if lane_vals.empty:
            lane_mode = np.nan
        else:
            m = lane_vals.mode()
            lane_mode = float(m.iloc[0]) if len(m) else float(lane_vals.iloc[0])

        # edge means
        g_head = g.head(n_edge)
        g_tail = g.tail(n_edge)

        x_start = float(g_head[x_col].mean())
        y_start = float(g_head[y_col].mean())
        x_end   = float(g_tail[x_col].mean())
        y_end   = float(g_tail[y_col].mean())

        x_mean = float(g[x_col].mean())
        y_mean = float(g[y_col].mean())

        dx = x_end - x_start
        dy = y_end - y_start

        t0 = float(g[time_col].iloc[0])
        t1 = float(g[time_col].iloc[-1])
        T = max(0.0, t1 - t0)

        # heading encoding (stable)
        theta = float(np.arctan2(dy, dx)) if (dx != 0.0 or dy != 0.0) else 0.0
        hcos = float(np.cos(theta))
        hsin = float(np.sin(theta))

        # optional speed stats
        v_mean = np.nan
        v_p90 = np.nan
        if speed_col in g.columns:
            v = g[speed_col].dropna()
            if len(v) > 0:
                v_mean = float(v.mean())
                v_p90 = float(np.percentile(v, 90))

        rows.append({
            vehicle_id_col: int(vid),
            "lane_mode": lane_mode,
            "x_start": x_start, "y_start": y_start,
            "x_end": x_end, "y_end": y_end,
            "x_mean": x_mean, "y_mean": y_mean,
            "dx": dx, "dy": dy,
            "T": T,
            "v_mean": v_mean, "v_p90": v_p90,
            "hcos": hcos, "hsin": hsin,
        })

    feat = pd.DataFrame(rows)
    return feat


def fit_vehicle_knn_with_thresholds(
    feat_df: pd.DataFrame,
    k: int = 9,
    lane_label_col: str = "lane_mode",
):
    """
    Trains KNN on labeled vehicles only.
    Computes an out-of-distribution threshold T95 using k-th neighbor distance
    among labeled vehicles (excluding itself).
    """
    labeled = feat_df.dropna(subset=[lane_label_col]).copy()
    labeled[lane_label_col] = labeled[lane_label_col].astype(int)

    feature_cols = [c for c in labeled.columns if c not in ["Vehicle_ID", lane_label_col]]
    X = labeled[feature_cols].to_numpy(dtype=float)
    y = labeled[lane_label_col].to_numpy(dtype=int)

    # fill NaN speed stats with column means (simple + defensible)
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(Xs, y)

    # OOD threshold: 95th percentile of k-th neighbor distance among labeled vehicles
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean")
    nn.fit(Xs)
    dists, _ = nn.kneighbors(Xs)      # first neighbor is itself (dist=0)
    dk = dists[:, -1]                 # (k+1)th incl self => k-th other neighbor
    T95 = float(np.percentile(dk, 99))

    return scaler, knn, T95, feature_cols, col_means


def predict_missing_lanes_vehicle_knn(
    feat_df: pd.DataFrame,
    scaler,
    knn,
    T95: float,
    feature_cols,
    col_means,
    k: int = 9,
    conf_thresh: float = 0.56,     # >= 6/9 neighbors agree
    margin_thresh: float = 0.12,   # top - second >= 0.22
):
    """
    Returns per-vehicle predictions for vehicles where lane_mode is NaN.
    Applies:
      - vote confidence threshold
      - margin threshold
      - OOD reject via k-th neighbor distance > T95
    """
    missing = feat_df[feat_df["lane_mode"].isna()].copy()
    if missing.empty:
        return pd.DataFrame(columns=["Vehicle_ID", "Lane_Inferred", "Lane_Confidence", "dk", "reject_reason"])

    X = missing[feature_cols].to_numpy(dtype=float)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    Xs = scaler.transform(X)

    # neighbor labels for vote distribution
    neigh_dist, neigh_idx = knn.kneighbors(Xs, n_neighbors=k, return_distance=True)
    neigh_labels = knn._y[neigh_idx]  # labels of neighbors

    preds = []
    for i in range(len(missing)):
        labs = neigh_labels[i]
        counts = Counter(labs)
        top, c1 = counts.most_common(1)[0]
        p1 = c1 / k
        p2 = 0.0
        if len(counts) > 1:
            p2 = counts.most_common(2)[1][1] / k

        dk = float(neigh_dist[i][-1])
        reason = None

        if dk > T95 and p1 < 0.85:
            reason = "ood_distance"
        elif p1 < conf_thresh:
            reason = "low_conf"
        elif (p1 - p2) < margin_thresh:
            reason = "low_margin"

        if reason is None:
            lane_hat = int(top)
        else:
            lane_hat = np.nan

        preds.append({
            "Vehicle_ID": int(missing.iloc[i]["Vehicle_ID"]),
            "Lane_Inferred": lane_hat,
            "Lane_Confidence": float(p1),
            "dk": dk,
            "reject_reason": reason if reason is not None else "accepted",
        })

    return pd.DataFrame(preds)


def build_lane_training_set(
    df: pd.DataFrame,
    vehicle_id_col: str = "Vehicle_ID",
    lane_col: str = "Lane_Number",
    x_col: str = "Local_X",
    y_col: str = "Local_Y",
    points_per_vehicle: int = 150,
):
    """
    Build a balanced training set where each LABELED vehicle contributes
    up to points_per_vehicle points.
    Returns X (N,2), y (N,)
    """
    d = df[[vehicle_id_col, lane_col, x_col, y_col, "time_s"]].copy()

    d[lane_col] = pd.to_numeric(d[lane_col], errors="coerce")
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")


    d = d.dropna(subset=[lane_col, x_col, y_col])
    d[lane_col] = d[lane_col].astype(int)

    X_list = []
    y_list = []

    for vid, g in d.groupby(vehicle_id_col):
        g_s = _time_uniform_sample(g, points_per_vehicle, time_col="time_s")
        X_list.append(g_s[[x_col, y_col]].to_numpy(dtype=float))
        y_list.append(g_s[lane_col].to_numpy(dtype=int))

    if not X_list:
        raise ValueError("No labeled lane points found to train on.")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def train_lane_knn(
    df: pd.DataFrame,
    k: int = 25,
    points_per_vehicle: int = 150,
    x_col: str = "Local_X",
    y_col: str = "Local_Y",
    vehicle_id_col: str = "Vehicle_ID",
    lane_col: str = "Lane_Number",
):
    """
    Train KNN model to map (Local_X, Local_Y) -> lane.
    """
    X, y = build_lane_training_set(
        df,
        vehicle_id_col=vehicle_id_col,
        lane_col=lane_col,
        x_col=x_col,
        y_col=y_col,
        points_per_vehicle=points_per_vehicle,
    )

    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(X, y)
    return knn


def validate_lane_knn_by_vehicle(
    df: pd.DataFrame,
    k: int = 25,
    points_per_vehicle: int = 150,
    test_frac: float = 0.2,
    seed: int = 0,
    x_col: str = "Local_X",
    y_col: str = "Local_Y",
    vehicle_id_col: str = "Vehicle_ID",
    lane_col: str = "Lane_Number",
):
    """
    Validation split by VEHICLE (not by row), so you can defend it in thesis.
    Prints accuracy and confusion matrix.
    """
    d = df[[vehicle_id_col, lane_col, x_col, y_col, "time_s"]].copy()
    d[lane_col] = pd.to_numeric(d[lane_col], errors="coerce")
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=[lane_col, x_col, y_col])
    d[lane_col] = d[lane_col].astype(int)

    labeled_vids = d[vehicle_id_col].unique().tolist()
    if len(labeled_vids) < 5:
        print("[WARN] Not enough labeled vehicles for validation.")
        return

    rng = random.Random(seed)
    rng.shuffle(labeled_vids)
    n_test = max(1, int(len(labeled_vids) * test_frac))
    test_vids = set(labeled_vids[:n_test])
    train_vids = set(labeled_vids[n_test:])

    df_train = d[d[vehicle_id_col].isin(train_vids)]
    df_test = d[d[vehicle_id_col].isin(test_vids)]

    X_train, y_train = build_lane_training_set(
        df_train,
        vehicle_id_col=vehicle_id_col,
        lane_col=lane_col,
        x_col=x_col,
        y_col=y_col,
        points_per_vehicle=points_per_vehicle,
    )
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(X_train, y_train)

    X_test = df_test[[x_col, y_col]].to_numpy(dtype=float)
    y_true = df_test[lane_col].to_numpy(dtype=int)
    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(y_true)))

    print("\n=== KNN LANE VALIDATION (split by vehicle) ===")
    print(f"Test vehicles: {len(test_vids)} / {len(labeled_vids)}")
    print(f"Point-level accuracy: {acc:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print("=============================================\n")


def infer_missing_vehicle_lanes(
    df: pd.DataFrame,
    knn,
    confidence_thresh: float = 0.70,
    sample_points_per_vehicle: int = 200,
    x_col: str = "Local_X",
    y_col: str = "Local_Y",
    vehicle_id_col: str = "Vehicle_ID",
    lane_col: str = "Lane_Number",
):
    """
    Infer lane per vehicle for vehicles that have NO lane anywhere.
    Returns a DataFrame with:
      Vehicle_ID, Lane_Inferred, Lane_Confidence
    """
    d = df[[vehicle_id_col, lane_col, x_col, y_col, "time_s"]].copy()
    d[lane_col] = pd.to_numeric(d[lane_col], errors="coerce")
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=[x_col, y_col])

    has_lane = d.groupby(vehicle_id_col)[lane_col].apply(lambda s: s.notna().any())
    vids_missing = has_lane[~has_lane].index.tolist()

    records = []

    for vid in vids_missing:
        g = d[d[vehicle_id_col] == vid]
        g_s = _time_uniform_sample(g, sample_points_per_vehicle, time_col="time_s")

        X = g_s[[x_col, y_col]].to_numpy(dtype=float)
        if len(X) == 0:
            records.append((int(vid), np.nan, 0.0))
            continue

        pred = knn.predict(X).astype(int)
        counts = Counter(pred)
        lane_hat, cmax = counts.most_common(1)[0]
        conf = cmax / len(pred)

        if conf < confidence_thresh:
            lane_hat = np.nan

        records.append((int(vid), lane_hat, conf))

    out = pd.DataFrame(records, columns=[vehicle_id_col, "Lane_Inferred", "Lane_Confidence"])
    return out


def assign_lane_final_per_vehicle(
    df: pd.DataFrame,
    inferred_df: pd.DataFrame,
    vehicle_id_col: str = "Vehicle_ID",
    lane_col: str = "Lane_Number",
):
    """
    Create per-vehicle Lane_Final, Lane_Source, Lane_Confidence.
    - If a vehicle has any lane info, use its mode lane as original.
    - Otherwise use Lane_Inferred (if not NaN).
    Then push Lane_Final to ALL rows of that vehicle.
    """
    d = df[[vehicle_id_col, lane_col]].copy()
    d[lane_col] = pd.to_numeric(d[lane_col], errors="coerce")

    def pick_lane(series: pd.Series):
        s = series.dropna()
        if s.empty:
            return np.nan
        m = s.mode()
        return float(m.iloc[0]) if len(m) else float(s.iloc[0])

    original_lane = d.groupby(vehicle_id_col)[lane_col].apply(pick_lane)

    inf_map = inferred_df.set_index(vehicle_id_col)

    lane_final_vehicle = {}
    lane_source_vehicle = {}
    lane_conf_vehicle = {}

    for vid, orig in original_lane.items():
        if not np.isnan(orig):
            lane_final_vehicle[int(vid)] = int(orig)
            lane_source_vehicle[int(vid)] = "original"
            lane_conf_vehicle[int(vid)] = 1.0
        else:
            if int(vid) in inf_map.index:
                lane_inf = inf_map.loc[int(vid), "Lane_Inferred"]
                conf = float(inf_map.loc[int(vid), "Lane_Confidence"])
                if pd.isna(lane_inf):
                    lane_final_vehicle[int(vid)] = np.nan
                    lane_source_vehicle[int(vid)] = "unknown"
                    lane_conf_vehicle[int(vid)] = conf
                else:
                    lane_final_vehicle[int(vid)] = int(lane_inf)
                    lane_source_vehicle[int(vid)] = "inferred"
                    lane_conf_vehicle[int(vid)] = conf
            else:
                lane_final_vehicle[int(vid)] = np.nan
                lane_source_vehicle[int(vid)] = "unknown"
                lane_conf_vehicle[int(vid)] = 0.0

    df["Lane_Final"] = df[vehicle_id_col].astype(int).map(lane_final_vehicle)
    df["Lane_Source"] = df[vehicle_id_col].astype(int).map(lane_source_vehicle)
    df["Lane_Confidence"] = df[vehicle_id_col].astype(int).map(lane_conf_vehicle)

    return df


# ==========================
# CLASSIC LANE ASSIGNMENT FROM G.csv POLYGONS (Ortho_X/Ortho_Y)
# ==========================

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

def assign_lane_final_using_g_polygons(
    df: pd.DataFrame,
    g_csv_path: str = 'segmentation.csv',
    vehicle_id_col: str = 'Vehicle_ID',
    lane_col_original: str = 'Lane_Number',
    x_col: str = 'Ortho_X',
    y_col: str = 'Ortho_Y',
    time_col: str = 'time_s',
    sample_points_per_vehicle: int = 200,
    confidence_thresh: float = 0.55,
    contain_radius: float = 0.0,
    snap_dist_thresh: float = 120.0,
    DEBUG_LANE_ASSIGN: bool = True,
):
    """
    Fills missing lane numbers using G.csv polygons in Ortho pixel coordinates.

    Outputs new columns:
      - Lane_Final: dataset-style lane number (1..4) where possible
      - Lane_Source: 'original' | 'g_contain' | 'g_snap' | 'unknown'
      - Lane_Confidence: vote share (1.0 for original)
      - Section_Final, LaneIdx_Final: the (Section, lane index) we voted for from G.csv

    Key idea:
      1) Use already-labeled vehicles to learn mapping from (Section, LaneIdx) -> Lane_Number.
      2) For missing-lane vehicles, vote their dominant (Section, LaneIdx) using polygons, then map to Lane_Number.
    """
    out = df.copy()

    # --- Load lane polygons
    g_df = pd.read_csv(g_csv_path)
    g_geoms = build_g_lane_geoms(g_df)

    # --- Build mapping from section-lane to dataset lane number (using labeled vehicles)
    sl_to_lane_number = build_sectionlane_to_lane_number_map(
        out,
        g_geoms,
        vehicle_id_col=vehicle_id_col,
        lane_col_original=lane_col_original,
        x_col=x_col,
        y_col=y_col,
        time_col=time_col,
        sample_points_per_vehicle=sample_points_per_vehicle,
        contain_radius=contain_radius,
    )

    # --- Per-vehicle original lane (mode)
    d0 = out[[vehicle_id_col, lane_col_original]].copy()
    d0[lane_col_original] = pd.to_numeric(d0[lane_col_original], errors='coerce')

    def pick_lane(series: pd.Series):
        s = series.dropna()
        if s.empty:
            return np.nan
        m = s.mode()
        return float(m.iloc[0]) if len(m) else float(s.iloc[0])

    original_lane = d0.groupby(vehicle_id_col)[lane_col_original].apply(pick_lane)
    vids_missing = original_lane[original_lane.isna()].index.astype(int).tolist()

    # --- Prep point table
    dxy = out[[vehicle_id_col, x_col, y_col, time_col]].copy()
    dxy[x_col] = pd.to_numeric(dxy[x_col], errors='coerce')
    dxy[y_col] = pd.to_numeric(dxy[y_col], errors='coerce')
    dxy = dxy.dropna(subset=[x_col, y_col])

    # --- Output dicts
    lane_final_vehicle = {}
    lane_source_vehicle = {}
    lane_conf_vehicle = {}
    sec_final_vehicle = {}
    li_final_vehicle = {}

    # fill originals
    for vid, orig in original_lane.items():
        if not np.isnan(orig):
            lane_final_vehicle[int(vid)] = int(orig)
            lane_source_vehicle[int(vid)] = 'original'
            lane_conf_vehicle[int(vid)] = 1.0
            sec_final_vehicle[int(vid)] = None
            li_final_vehicle[int(vid)] = None

    # infer missings
    src_counts = Counter()
    nohit_examples = []
    for vid in vids_missing:
        gveh = dxy[dxy[vehicle_id_col] == vid]
        if gveh.empty:
            lane_final_vehicle[int(vid)] = np.nan
            lane_source_vehicle[int(vid)] = 'unknown'
            lane_conf_vehicle[int(vid)] = 0.0
            sec_final_vehicle[int(vid)] = None
            li_final_vehicle[int(vid)] = None
            src_counts['unknown'] += 1
            continue

        g_s = _time_uniform_sample(gveh, sample_points_per_vehicle, time_col=time_col)
        pts = g_s[[x_col, y_col]].to_numpy(dtype=float)

        sec, li, conf, method = infer_vehicle_sectionlane_from_g(
            pts,
            g_geoms,
            contain_radius=contain_radius,
            snap_dist_thresh=snap_dist_thresh,
        )

        if method == 'none' or conf < confidence_thresh or sec is None:
            lane_final_vehicle[int(vid)] = np.nan
            lane_source_vehicle[int(vid)] = 'unknown'
            lane_conf_vehicle[int(vid)] = float(conf)
            sec_final_vehicle[int(vid)] = sec
            li_final_vehicle[int(vid)] = li
            src_counts['unknown'] += 1
            # track a couple examples
            if len(nohit_examples) < 10:
                nohit_examples.append((int(vid), float(conf)))
            continue

        # map (Section, LaneIdx) -> dataset lane number if mapping exists
        lane_num = sl_to_lane_number.get((sec, int(li)))
        if lane_num is None:
            # fallback: use lane_idx as lane number (last resort)
            lane_num = int(li)

        lane_final_vehicle[int(vid)] = int(lane_num)
        lane_source_vehicle[int(vid)] = 'g_contain' if method == 'contain' else 'g_snap'
        lane_conf_vehicle[int(vid)] = float(conf)
        sec_final_vehicle[int(vid)] = sec
        li_final_vehicle[int(vid)] = int(li)
        src_counts[lane_source_vehicle[int(vid)]] += 1

    out['Lane_Final'] = out[vehicle_id_col].astype(int).map(lane_final_vehicle)
    out['Lane_Source'] = out[vehicle_id_col].astype(int).map(lane_source_vehicle)
    out['Lane_Confidence'] = out[vehicle_id_col].astype(int).map(lane_conf_vehicle)
    out['Section_Final'] = out[vehicle_id_col].astype(int).map(sec_final_vehicle)
    out['LaneIdx_Final'] = out[vehicle_id_col].astype(int).map(li_final_vehicle)

    if DEBUG_LANE_ASSIGN:
        print("\n=== LANE ASSIGNMENT DEBUG SUMMARY ===")
        print(f"Mapping learned for (Section,LaneIdx) -> Lane_Number: {len(sl_to_lane_number)} keys")
        print(f"Vehicles missing Lane_Number in raw CSV: {len(vids_missing)}")
        print(f"Assigned via containment: {src_counts.get('g_contain', 0)}")
        print(f"Assigned via snap-to-nearest (<= {snap_dist_thresh}px): {src_counts.get('g_snap', 0)}")
        print(f"Still unknown: {src_counts.get('unknown', 0)}")
        if nohit_examples:
            print("Examples still unknown (Vehicle_ID, confidence):", nohit_examples[:10])
        print("=====================================\n")

    return out

def print_missing_lane_inference_report(
    df: pd.DataFrame,
    inferred_lanes_df: pd.DataFrame,
    vehicle_id_col: str = "Vehicle_ID",
    lane_col_original: str = "Lane_Number",
    lane_col_final: str = "Lane_Final",
    sort_desc: bool = True,
    only_unknown: bool = False,
):
    """
    Prints details for vehicles that have NO original Lane_Number anywhere:
      - Vehicle_ID
      - Lane_Inferred
      - Lane_Confidence (from inference)
      - Lane_Final (after thresholding)
      - Lane_Source
      - Lane_Confidence_final (per-vehicle confidence stored in df)

    only_unknown=True -> prints only those whose Lane_Final is NaN (low confidence / unknown).
    """

    d0 = df[[vehicle_id_col, lane_col_original]].copy()
    d0[lane_col_original] = pd.to_numeric(d0[lane_col_original], errors="coerce")
    has_lane = d0.groupby(vehicle_id_col)[lane_col_original].apply(lambda s: s.notna().any())
    missing_vids = set(has_lane[~has_lane].index.astype(int).tolist())

    rep = inferred_lanes_df.copy()
    rep[vehicle_id_col] = rep[vehicle_id_col].astype(int)
    rep = rep[rep[vehicle_id_col].isin(missing_vids)].copy()

    if "Lane_Confidence" in rep.columns:
        rep = rep.rename(columns={"Lane_Confidence": "Lane_Confidence_inferred"})

    per_vehicle_final = (
        df[[vehicle_id_col, lane_col_final, "Lane_Source", "Lane_Confidence"]]
        .drop_duplicates(subset=[vehicle_id_col])
        .copy()
    )
    per_vehicle_final[vehicle_id_col] = per_vehicle_final[vehicle_id_col].astype(int)
    per_vehicle_final = per_vehicle_final.rename(columns={"Lane_Confidence": "Lane_Confidence_final"})

    rep = rep.merge(
        per_vehicle_final,
        on=vehicle_id_col,
        how="left",
    )

    if only_unknown:
        rep = rep[rep[lane_col_final].isna()].copy()

    sort_col = "Lane_Confidence_final" if "Lane_Confidence_final" in rep.columns else "Lane_Confidence_inferred"
    rep = rep.sort_values(sort_col, ascending=not sort_desc)

    print("\n=== MISSING Lane_Number VEHICLES: inference details ===")
    print(f"Total vehicles with NO original Lane_Number: {len(missing_vids)}")
    print(f"Rows in inferred_lanes_df for those: {len(rep)}")
    if only_unknown:
        print("Showing ONLY vehicles where Lane_Final is unknown (NaN).")
    print(f"Sorted by: {sort_col}")
    print("------------------------------------------------------")

    for _, row in rep.iterrows():
        vid = int(row[vehicle_id_col])

        lane_inf = row.get("Lane_Inferred", np.nan)
        lane_final = row.get(lane_col_final, np.nan)

        conf_inf = float(row.get("Lane_Confidence_inferred", np.nan)) if "Lane_Confidence_inferred" in row else float("nan")
        conf_final = float(row.get("Lane_Confidence_final", np.nan)) if "Lane_Confidence_final" in row else float("nan")

        src = row.get("Lane_Source", "unknown")

        lane_inf_str = "NaN" if pd.isna(lane_inf) else str(int(lane_inf))
        lane_final_str = "NaN" if pd.isna(lane_final) else str(int(lane_final))

        print(
            f"Vehicle {vid:4d} | inferred={lane_inf_str:>3s} | final={lane_final_str:>3s} | "
            f"conf_inf={conf_inf:0.3f} | conf_final={conf_final:0.3f} | source={src}"
        )

    print("======================================================\n")


def print_lane_coverage_before_after(df: pd.DataFrame):
    """
    Prints BEFORE (Lane_Number) and AFTER (Lane_Final) lane coverage at vehicle level.
    """
    vehicles_total = int(df["Vehicle_ID"].nunique())

    # BEFORE: original Lane_Number coverage
    d0 = df[["Vehicle_ID", "Lane_Number"]].copy()
    d0["Lane_Number"] = pd.to_numeric(d0["Lane_Number"], errors="coerce")
    has_orig = d0.groupby("Vehicle_ID")["Lane_Number"].apply(lambda s: s.notna().any())
    before_with = int(has_orig.sum())
    before_without = vehicles_total - before_with

    print(f"\n=== LANE COVERAGE ===")
    print(f"Total vehicles: {vehicles_total}")
    print(f"Before KNN: {before_with} labeled, {before_without} missing")

    # AFTER: Lane_Final coverage (only if Lane_Final exists)
    if "Lane_Final" not in df.columns:
        print("[WARN] Lane_Final not found yet. Run KNN + assign_lane_final_per_vehicle first.")
        return

    d1 = df[["Vehicle_ID", "Lane_Final", "Lane_Source"]].drop_duplicates("Vehicle_ID").copy()
    d1["Lane_Final"] = pd.to_numeric(d1["Lane_Final"], errors="coerce")

    after_with = int(d1["Lane_Final"].notna().sum())
    after_without = vehicles_total - after_with

    gained = after_with - before_with

    print(f"After KNN:  {after_with} labeled, {after_without} missing (+{gained} inferred)")
    print(f"=====================\n")
