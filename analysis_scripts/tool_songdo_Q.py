# tool_songdo.py
"""
Adapted traffic analysis tool for Songdo dataset.
Built upon the original tool.py, adapted for Songdo's characteristics:
- Multiple coordinate systems (orthophoto, local planar, WGS84)
- Pre-computed vehicle metadata (lane assignment, speed, acceleration)
- Higher temporal resolution (29.97 FPS vs typical GPS ~1 Hz)
- Already-segmented lane/section information
"""

from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans


class Master:
    """
    Master class for Songdo traffic analysis.
    Provides interfaces to various analysis tools adapted for Songdo dataset.
    """

    def __init__(self):
        """Initialization of Master class"""
        pass

    def distances(self, initial_coordinates: tuple, final_coordinates: tuple, coord_system: str = "local"):
        """
        Calculate distances between two points.

        Parameters
        ----------
        initial_coordinates : tuple
            (x, y) or (lat, lon) of starting point
        final_coordinates : tuple
            (x, y) or (lat, lon) of ending point
        coord_system : str
            "local" - planar coordinates in meters (Local_X, Local_Y)
            "ortho" - orthophoto pixels (Ortho_X, Ortho_Y)
            "latlon" - geographic coordinates (Latitude, Longitude)

        Returns
        -------
        _distances
            Distance calculator object
        """
        return self._distances(self, initial_coordinates, final_coordinates, coord_system)

    def dataloader(self, raw_data: dict, spatio_temporal_info: dict):
        """
        Load and prepare trajectory data.

        Parameters
        ----------
        raw_data : dict
            Dictionary with trajectory data (id, x, y, time, vehicle_class, lane, section, visibility)
        spatio_temporal_info : dict
            Dictionary with spatial/temporal metadata (bbox, intersection_center, time_axis)

        Returns
        -------
        _dataloader
            Data loader object
        """
        return self._dataloader(self, raw_data, spatio_temporal_info)

    def analysis(self, data: dict, spatio_temporal_info: dict):
        """
        Perform traffic analysis on trajectory data.

        Parameters
        ----------
        data : dict
            Processed trajectory data
        spatio_temporal_info : dict
            Spatial/temporal metadata

        Returns
        -------
        _analysis
            Analysis object
        """
        return self._analysis(self, data, spatio_temporal_info)

    def visualization(self, data: dict, spatio_temporal_info: dict):
        """
        Visualize trajectory data.

        Parameters
        ----------
        data : dict
            Processed trajectory data
        spatio_temporal_info : dict
            Spatial/temporal metadata

        Returns
        -------
        _visualization
            Visualization object
        """
        return self._visualization(self, data, spatio_temporal_info)

    class _distances:
        """
        Distance calculation utility for Songdo coordinates.
        Supports multiple coordinate systems used in Songdo dataset.
        """

        def __init__(self, master: 'Master', initial_coordinates: tuple, final_coordinates: tuple, 
                     coord_system: str = "local"):
            """
            Initialize distance calculator.

            Parameters
            ----------
            master : Master
                Reference to Master instance
            initial_coordinates : tuple
                (x, y) starting coordinates
            final_coordinates : tuple
                (x, y) ending coordinates
            coord_system : str
                "local" - planar KGD2002 coordinates (meters)
                "ortho" - orthophoto pixels
                "latlon" - WGS84 geographic coordinates
            """
            self.master = master
            self.x_i, self.y_i = initial_coordinates
            self.x_f, self.y_f = final_coordinates
            self.coord_system = coord_system
            
            # Earth circumference for lat/lon conversion
            self.earth_radius = 6371000  # meters
            self.factor = 2 * np.pi * self.earth_radius / 360

        def get_dx(self) -> float:
            """
            Calculate longitudinal distance along x/longitude axis.

            Returns
            -------
            float
                Longitudinal distance in appropriate units (meters or pixels)
            """
            if self.coord_system in ["local", "ortho"]:
                return float(self.x_f - self.x_i)
            elif self.coord_system == "latlon":
                # Longitude distance (x-axis in lat/lon terms)
                return self.factor * np.cos(np.deg2rad(self.y_i)) * (self.x_f - self.x_i)
            return 0.0

        def get_dy(self) -> float:
            """
            Calculate latitudinal distance along y/latitude axis.

            Returns
            -------
            float
                Latitudinal distance in appropriate units (meters or pixels)
            """
            if self.coord_system in ["local", "ortho"]:
                return float(self.y_f - self.y_i)
            elif self.coord_system == "latlon":
                # Latitude distance (y-axis in lat/lon terms)
                return self.factor * (self.y_f - self.y_i)
            return 0.0

        def get_distance(self) -> float:
            """
            Calculate Euclidean distance between two points.

            Returns
            -------
            float
                Distance in appropriate units (meters for local/latlon, pixels for ortho)
            """
            dx = self.get_dx()
            dy = self.get_dy()
            return np.sqrt(dx**2 + dy**2)

    class _dataloader:
        """
        Data loader for Songdo trajectory data.
        Handles filtering and bounding of vehicle trajectories.
        """

        def __init__(self, master: 'Master', raw_data: dict, spatio_temporal_info: dict):
            """
            Initialize data loader.

            Parameters
            ----------
            master : Master
                Reference to Master instance
            raw_data : dict
                Raw trajectory data with keys:
                - id: vehicle IDs
                - x: x-coordinates (Local_X or Ortho_X)
                - y: y-coordinates (Local_Y or Ortho_Y)
                - time: timestamps
                - speed: vehicle speeds (already provided in Songdo)
                - vehicle_class: vehicle type (0-3)
                - lane: lane numbers (optional, may be NaN)
                - section: road section IDs (optional, may be NaN)
                - visibility: visibility status (0/1)
            spatio_temporal_info : dict
                Metadata with keys:
                - bbox: bounding box of interest area
                - intersection_center: center coordinates
                - time_axis: time reference info
                - coord_system: "local" or "ortho"
            """
            self.master = master
            needed_keys_data = ['id', 'x', 'y', 'time', 'speed']
            needed_keys_info = ['bbox', 'intersection_center', 'time_axis']

            if any(key not in raw_data.keys() for key in needed_keys_data):
                raise KeyError(f'data dictionary needs keys {needed_keys_data} to work!')
            if any(key not in spatio_temporal_info.keys() for key in needed_keys_info):
                raise KeyError(f'spatio temporal info dictionary needs keys {needed_keys_info} to work!')

            self.spatio_temporal_info = spatio_temporal_info
            self.coord_system = spatio_temporal_info.get('coord_system', 'local')
            
            # Extract required fields
            base_keys = ['id', 'x', 'y', 'time', 'speed']
            self.vehicle_id, self.x, self.y, self.t, self.u = itemgetter(*base_keys)(raw_data)
            
            # Extract optional fields (lane, section, visibility, vehicle_class)
            self.lane = raw_data.get('lane', [[] for _ in self.x])
            self.section = raw_data.get('section', [[] for _ in self.x])
            self.visibility = raw_data.get('visibility', [[] for _ in self.x])
            self.vehicle_class = raw_data.get('vehicle_class', [[] for _ in self.x])
            
            self.bbox, self.intersection_center, self.time_axis = itemgetter(
                'bbox', 'intersection_center', 'time_axis'
            )(spatio_temporal_info)
            
            # If speed information is missing or empty, compute it using analysis (behavior from original tool.py)
            if (('speed' not in raw_data) or (raw_data.get('speed') is None) or (raw_data.get('speed') == []) ):
                raw_data['speed'] = []
                # compute speeds from trajectory using analysis helper (km/h)
                try:
                    computed_u = self.master.analysis(raw_data, spatio_temporal_info).get_speed()
                    raw_data['speed'] = computed_u
                    self.u = computed_u
                except Exception:
                    # fallback: leave speeds as empty lists
                    raw_data['speed'] = []
                    self.u = raw_data.get('speed', [])
            else:
                self.u = raw_data.get('speed', [])

            self.raw_data = raw_data

        def get_data(self) -> dict:
            """
            Get raw data without spatial filtering.

            Returns
            -------
            dict
                Original dataset with all available fields
            """
            return self.raw_data

        def get_bounded_data(self) -> dict:
            """
            Filter trajectories to keep only points within bounding box.
            Extracts continuous segments of each trajectory that pass through bbox.

            Returns
            -------
            dict
                Filtered dataset containing only bounded trajectory segments
            """
            box = Polygon(self.bbox)
            id_, x_, y_, t_, u_, lane_, section_, visibility_, vclass_ = [], [], [], [], [], [], [], [], []
            
            for i, vec_x in enumerate(self.x):
                flag = False
                index_start = None

                # Find first point inside bbox
                for j in range(len(vec_x)):
                    if box.contains(Point(self.x[i][j], self.y[i][j])):
                        flag = True
                        index_start = j
                        break

                if flag:
                    # Find last point inside bbox
                    index_end = index_start
                    for k in range(index_start + 1, len(vec_x)):
                        if box.contains(Point(self.x[i][k], self.y[i][k])):
                            index_end = k
                        else:
                            break

                    # Keep segment if it has multiple points
                    if len(self.t[i][index_start:index_end + 1]) > 1:
                        id_.append(self.vehicle_id[i])
                        x_.append(self.x[i][index_start:index_end + 1])
                        y_.append(self.y[i][index_start:index_end + 1])
                        t_.append(self.t[i][index_start:index_end + 1])
                        u_.append(self.u[i][index_start:index_end + 1])
                        
                        # Handle optional fields
                        if self.lane and i < len(self.lane) and self.lane[i]:
                            lane_.append(self.lane[i][index_start:index_end + 1])
                        else:
                            lane_.append([])
                            
                        if self.section and i < len(self.section) and self.section[i]:
                            section_.append(self.section[i][index_start:index_end + 1])
                        else:
                            section_.append([])
                            
                        if self.visibility and i < len(self.visibility) and self.visibility[i]:
                            visibility_.append(self.visibility[i][index_start:index_end + 1])
                        else:
                            visibility_.append([])
                            
                        if self.vehicle_class and i < len(self.vehicle_class) and self.vehicle_class[i]:
                            vclass_.append(self.vehicle_class[i][index_start:index_end + 1])
                        else:
                            vclass_.append([])

            bounded_data = {
                'id': id_, 'x': x_, 'y': y_, 'time': t_, 'speed': u_,
                'lane': lane_, 'section': section_, 'visibility': visibility_, 
                'vehicle_class': vclass_
            }
            return bounded_data

        def get_filtered_data(self, cursed_ids=None, min_trajectory_length: int = 2, 
                            immobility_threshold: float = 0.95) -> dict:
            """
            Filter out parked/stationary vehicles and unwanted IDs.

            Parameters
            ----------
            cursed_ids : list, optional
                Vehicle IDs to exclude from analysis
            min_trajectory_length : int
                Minimum number of points in a valid trajectory
            immobility_threshold : float
                Maximum fraction of stationary points (< 1e-4 m distance)
                before vehicle is considered parked

            Returns
            -------
            dict
                Filtered dataset excluding parked vehicles and specified IDs
            """
            if cursed_ids is None:
                cursed_ids = []

            intersection_data = self.get_bounded_data()
            id_, x_, y_, t_, u_, lane_, section_, visibility_, vclass_ = [], [], [], [], [], [], [], [], []
            
            vehicle_id, x, y, t, u = itemgetter(
                'id', 'x', 'y', 'time', 'speed'
            )(intersection_data)
            lane = intersection_data.get('lane', [])
            section = intersection_data.get('section', [])
            visibility = intersection_data.get('visibility', [])
            vclass = intersection_data.get('vehicle_class', [])

            for i, vec in enumerate(x):
                # Count stationary points (distance < 1e-4 m)
                immobility_count = 0
                for j in range(1, len(vec)):
                    dist = self.master.distances(
                        initial_coordinates=(x[i][j-1], y[i][j-1]),
                        final_coordinates=(x[i][j], y[i][j]),
                        coord_system=self.coord_system
                    ).get_distance()
                    if dist < 1e-4:
                        immobility_count += 1

                immobility_ratio = immobility_count / len(vec) if len(vec) > 0 else 0

                # Keep vehicle if not in cursed list and not too stationary
                if (vehicle_id[i] not in cursed_ids and 
                    immobility_ratio <= immobility_threshold and
                    len(vec) >= min_trajectory_length):
                    id_.append(vehicle_id[i])
                    x_.append(x[i])
                    y_.append(y[i])
                    t_.append(t[i])
                    u_.append(u[i])
                    
                    if lane and i < len(lane):
                        lane_.append(lane[i])
                    else:
                        lane_.append([])
                        
                    if section and i < len(section):
                        section_.append(section[i])
                    else:
                        section_.append([])
                        
                    if visibility and i < len(visibility):
                        visibility_.append(visibility[i])
                    else:
                        visibility_.append([])
                        
                    if vclass and i < len(vclass):
                        vclass_.append(vclass[i])
                    else:
                        vclass_.append([])

            filtered_data = {
                'id': id_, 'x': x_, 'y': y_, 'time': t_, 'speed': u_,
                'lane': lane_, 'section': section_, 'visibility': visibility_,
                'vehicle_class': vclass_
            }
            return filtered_data

    class _analysis:
        """
        Analysis class for Songdo trajectory data.
        Computes traffic metrics from pre-processed and filtered trajectories.
        """

        def __init__(self, master: 'Master', data: dict, spatio_temporal_info: dict):
            """
            Initialize analysis class.

            Parameters
            ----------
            master : Master
                Reference to Master instance
            data : dict
                Processed trajectory data (from dataloader)
            spatio_temporal_info : dict
                Spatial/temporal metadata
            """
            self.master = master
            needed_keys_data = ['id', 'x', 'y', 'time', 'speed']
            needed_keys_info = ['bbox', 'intersection_center', 'time_axis']

            if any(key not in data.keys() for key in needed_keys_data):
                raise KeyError(f'data dictionary needs keys {needed_keys_data} to work!')
            if any(key not in spatio_temporal_info.keys() for key in needed_keys_info):
                raise KeyError(f'spatio temporal info dictionary needs keys {needed_keys_info} to work!')

            self.data = data
            self.spatio_temporal_info = spatio_temporal_info
            self.coord_system = spatio_temporal_info.get('coord_system', 'local')
            
            # Extract core fields
            base_keys = ['id', 'x', 'y', 'time', 'speed']
            self.vehicle_id, self.x, self.y, self.t, self.u = itemgetter(*base_keys)(data)
            
            # Extract optional fields
            self.lane = data.get('lane', [[] for _ in self.x])
            self.section = data.get('section', [[] for _ in self.x])
            self.visibility = data.get('visibility', [[] for _ in self.x])
            self.vehicle_class = data.get('vehicle_class', [[] for _ in self.x])
            
            self.bbox, self.center, self.time_axis = itemgetter(
                'bbox', 'intersection_center', 'time_axis'
            )(spatio_temporal_info)
            
            self.x_center, self.y_center = self.center

        def get_distance_travelled(self) -> list:
            """
            Calculate cumulative distance travelled per vehicle.

            Returns
            -------
            list[list[float]]
                Cumulative distance in meters for each vehicle at each timestep
            """
            distance_travelled = []

            for i in range(len(self.x)):
                temp_distance_travelled = []
                temp_sum = 0
                for j in range(len(self.x[i])):
                    if j == 0:
                        temp_distance_travelled.append(0)
                    else:
                        dist = self.master.distances(
                            initial_coordinates=(self.x[i][j-1], self.y[i][j-1]),
                            final_coordinates=(self.x[i][j], self.y[i][j]),
                            coord_system=self.coord_system
                        ).get_distance()
                        temp_sum += dist
                        temp_distance_travelled.append(temp_sum)
                distance_travelled.append(temp_distance_travelled)

            return distance_travelled

        def get_speed_from_trajectory(self, km_per_h: bool = True) -> list:
            """
            Calculate speed from trajectory data.
            
            Note: Songdo dataset provides pre-computed Vehicle_Speed column.
            This method recalculates from raw trajectory if needed for validation.

            Parameters
            ----------
            km_per_h : bool
                If True, return speed in km/h; if False, in m/s

            Returns
            -------
            list[list[float]]
                Speed for each vehicle at each timestep
            """
            distance_travelled = self.get_distance_travelled()
            multiplier = 3.6 if km_per_h else 1.0
            
            u = []
            for i in range(len(distance_travelled)):
                speed = np.gradient(distance_travelled[i], self.t[i]) * multiplier
                u.append(speed.tolist())
            
            # Apply smoothing
            smoothing_factor = 2
            u_smooth = [gaussian_filter(np.array(vec), sigma=smoothing_factor).tolist() for vec in u]
            return u_smooth

        def get_acceleration_from_trajectory(self) -> list:
            """
            Calculate acceleration from trajectory data.
            
            Note: Songdo dataset provides pre-computed Vehicle_Acceleration column.
            This method recalculates from raw trajectory if needed for validation.

            Returns
            -------
            list[list[float]]
                Acceleration in m/s² for each vehicle at each timestep
            """
            u = self.get_speed_from_trajectory(km_per_h=False)
            
            acc = []
            for i in range(len(u)):
                acceleration = np.gradient(u[i], self.t[i])
                acc.append(acceleration.tolist())
            
            # Apply smoothing
            smoothing_factor = 2
            a_smooth = [gaussian_filter(np.array(vec), sigma=smoothing_factor).tolist() for vec in acc]
            return a_smooth

        def get_section_distribution(self) -> dict:
            """
            Get distribution of vehicles across road sections.

            Returns
            -------
            dict
                Mapping of section IDs to list of vehicle IDs present in that section
            """
            section_dist = {}
            
            for i, vec_section in enumerate(self.section):
                if vec_section:  # If section data exists
                    unique_sections = set([s for s in vec_section if s is not None and str(s) != 'nan'])
                    for section in unique_sections:
                        if section not in section_dist:
                            section_dist[section] = []
                        section_dist[section].append(self.vehicle_id[i])
            
            return section_dist

        def get_lane_distribution(self) -> dict:
            """
            Get distribution of vehicles across lanes.

            Returns
            -------
            dict
                Mapping of lane numbers to list of vehicle IDs present in that lane
            """
            lane_dist = {}
            
            for i, vec_lane in enumerate(self.lane):
                if vec_lane:  # If lane data exists
                    unique_lanes = set([l for l in vec_lane if l is not None and str(l) != 'nan'])
                    for lane in unique_lanes:
                        if lane not in lane_dist:
                            lane_dist[lane] = []
                        lane_dist[lane].append(self.vehicle_id[i])
            
            return lane_dist

        # Compatibility methods matching original tool.py API
        def get_speed(self, km_per_h: bool = True) -> list:
            """
            Compatibility wrapper: returns speed per-vehicle per-timestep.
            """
            return self.get_speed_from_trajectory(km_per_h=km_per_h)

        def get_acceleration(self) -> list:
            """
            Compatibility wrapper: returns acceleration per-vehicle per-timestep in m/s^2.
            """
            return self.get_acceleration_from_trajectory()

        def get_vehicle_class_distribution(self) -> dict:
            """
            Get distribution of vehicles by class.
            Vehicle classes: 0=car/van, 1=bus, 2=truck, 3=motorcycle

            Returns
            -------
            dict
                Mapping of class to list of vehicle IDs
            """
            class_dist = {0: [], 1: [], 2: [], 3: []}
            
            for i, vec_class in enumerate(self.vehicle_class):
                if vec_class:  # If class data exists
                    # Use first non-NaN class value
                    for c in vec_class:
                        if c is not None and str(c) != 'nan':
                            if int(c) not in class_dist:
                                class_dist[int(c)] = []
                            class_dist[int(c)].append(self.vehicle_id[i])
                            break
            
            return class_dist

        def get_visibility_stats(self) -> dict:
            """
            Get visibility statistics for trajectories.

            Returns
            -------
            dict
                Statistics on vehicle visibility (fully visible vs partially visible)
            """
            fully_visible = []
            partially_visible = []
            
            for i, vec_vis in enumerate(self.visibility):
                if vec_vis:
                    vis_sum = sum([1 for v in vec_vis if v == 1])
                    if vis_sum == len(vec_vis):
                        fully_visible.append(self.vehicle_id[i])
                    elif vis_sum > 0:
                        partially_visible.append(self.vehicle_id[i])
            
            return {
                'fully_visible': fully_visible,
                'partially_visible': partially_visible,
                'count_fully_visible': len(fully_visible),
                'count_partially_visible': len(partially_visible)
            }

        def get_od_pairs(self) -> list:
            """
            Infer origin-destination (OD) sectors for each vehicle based on
            the angular sector around the intersection center.

            Returns
            -------
            list
                List of tuples (origin, destination) where each is an int in 1..4.
            """
            import math

            def sector(x, y):
                ang = math.atan2(y - self.y_center, x - self.x_center)
                # Map angle to 4 sectors (clockwise-ish):
                # sector 1: (-pi, -pi/2]
                # sector 2: (-pi/2, 0]
                # sector 3: (0, pi/2]
                # sector 4: (pi/2, pi]
                if ang <= -math.pi/2:
                    return 1
                if ang <= 0:
                    return 2
                if ang <= math.pi/2:
                    return 3
                return 4

            od_pairs = []
            for v in range(len(self.x)):
                if not self.x[v] or not self.y[v]:
                    od_pairs.append((0, 0))
                    continue
                o_x, o_y = self.x[v][0], self.y[v][0]
                d_x, d_y = self.x[v][-1], self.y[v][-1]
                origin = sector(o_x, o_y)
                destination = sector(d_x, d_y)
                od_pairs.append((origin, destination))

            return od_pairs

    class _visualization:
        """
        Visualization utilities for Songdo trajectory data.
        """

        def __init__(self, master: 'Master', data: dict, spatio_temporal_info: dict):
            """
            Initialize visualization class.

            Parameters
            ----------
            master : Master
                Reference to Master instance
            data : dict
                Processed trajectory data
            spatio_temporal_info : dict
                Spatial/temporal metadata
            """
            self.master = master
            self.data = data
            self.spatio_temporal_info = spatio_temporal_info
            self.vehicle_id, self.x, self.y, self.t, self.u = itemgetter(
                'id', 'x', 'y', 'time', 'speed'
            )(data)
            self.lane = data.get('lane', [])
            self.section = data.get('section', [])

        def plot_trajectories(self, out_path: str = "trajectories.png", 
                            color_by_lane: bool = True, max_vehicles: int = 50):
            """
            Plot vehicle trajectories.

            Parameters
            ----------
            out_path : str
                Path to save output image
            color_by_lane : bool
                If True, color trajectories by lane; otherwise by vehicle ID
            max_vehicles : int
                Maximum number of trajectories to plot (for clarity)
            """
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Limit number of vehicles to plot
            num_vehicles = min(max_vehicles, len(self.x))
            selected_indices = np.random.choice(len(self.x), num_vehicles, replace=False)
            
            for idx in selected_indices:
                if color_by_lane and self.lane and idx < len(self.lane) and self.lane[idx]:
                    # Color by first lane
                    lane_val = next((l for l in self.lane[idx] if l is not None and str(l) != 'nan'), None)
                    color = plt.cm.tab20(int(lane_val) % 20) if lane_val else 'gray'
                else:
                    # Color by vehicle ID
                    color = plt.cm.hsv(self.vehicle_id[idx] % 256 / 256)
                
                ax.plot(self.x[idx], self.y[idx], alpha=0.6, linewidth=1, color=color)
            
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_title('Vehicle Trajectories')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            
            return out_path

        # --- Compatibility alias methods expected by uavsongdopie.py ---
        def draw_trajectories(self, title=True, scale=8, dpi=100):
            # Similar to plot_trajectories but leave figure open for outer saver
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.set_title('Trajectories' if title else '')
            for idx in range(len(self.x)):
                ax.plot(self.x[idx], self.y[idx], alpha=0.6, linewidth=1)
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig

        def draw_trajectories_od(self, title=True, scale=6, dpi=100):
            # Simple OD plot: draw all trajectories and the bbox/center
            fig, ax = plt.subplots(figsize=(8, 8))
            if title:
                ax.set_title('Route map')
            for xi, yi in zip(self.x, self.y):
                ax.plot(xi, yi, color='k', linewidth=0.5, alpha=0.5)
            # draw bbox if present
            try:
                bx = [p[0] for p in self.master.spatio_temporal_info.get('bbox', [])]
                by = [p[1] for p in self.master.spatio_temporal_info.get('bbox', [])]
            except Exception:
                bx, by = [], []
            if bx and by:
                ax.plot(bx + [bx[0]], by + [by[0]], color='blue', linewidth=0.5)
            plt.tight_layout()
            return fig

        def draw_distance_travelled(self, vehicle_id: int):
            # Match styling from original tool.py
            ana = self.master.analysis(self.data, self.spatio_temporal_info)
            dists = ana.get_distance_travelled()
            try:
                idx = list(self.vehicle_id).index(vehicle_id)
            except ValueError:
                raise ValueError(f'Vehicle {vehicle_id} not found')
            t = self.t[idx]
            y = dists[idx]
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(t, y, color='k', linewidth=2, label=f'Vehicle ID: {vehicle_id}')
            ax.set_xlabel('t (s)')
            ax.set_ylabel('Distance Travelled (m)')
            ax.set_title('Distance Travalled vs Time')
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            return fig

        def draw_speed(self, vehicle_id: int):
            # Match styling and normalization from original tool.py
            try:
                idx = list(self.vehicle_id).index(vehicle_id)
            except ValueError:
                raise ValueError(f'Vehicle {vehicle_id} not found')
            # prefer provided speed if available, else compute from trajectory
            if self.u and idx < len(self.u) and self.u[idx]:
                speed_data = list(map(float, self.u[idx]))
            else:
                ana = self.master.analysis(self.data, self.spatio_temporal_info)
                speed_data = ana.get_speed_from_trajectory(km_per_h=True)[idx]

            # Min-max normalization
            speed_min = min(speed_data) if len(speed_data) else 0.0
            speed_max = max(speed_data) if len(speed_data) else 0.0
            if speed_max > speed_min:
                normalized_speed = [(s - speed_min) / (speed_max - speed_min) for s in speed_data]
            else:
                normalized_speed = [0.0] * len(speed_data)

            t = self.t[idx]
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(t, normalized_speed, color='blue', linewidth=2, label=f'Vehicle ID: {vehicle_id}')
            ax.set_xlabel('t (s)')
            ax.set_ylabel('Speed (normalized)')
            ax.set_title('Speed vs Time (Normalized)')
            ax.set_ylim([-0.05, 1.05])
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            return fig

        def draw_acceleration(self, vehicle_id: int):
            # Match layout from original tool.py
            ana = self.master.analysis(self.data, self.spatio_temporal_info)
            acc = ana.get_acceleration_from_trajectory()
            try:
                idx = list(self.vehicle_id).index(vehicle_id)
            except ValueError:
                raise ValueError(f'Vehicle {vehicle_id} not found')
            t = self.t[idx]
            y = acc[idx]
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(t, y, color='red', linewidth=2, label=f'Vehicle ID: {vehicle_id}')
            ax.set_xlabel('t (s)')
            ax.set_ylabel(r'Acceleration $(m/s^2)$')
            ax.set_title('Acceleration vs Time')
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            return fig

        def plot_speed_distribution(self, out_path: str = "speed_distribution.png"):
            """
            Plot distribution of vehicle speeds.

            Parameters
            ----------
            out_path : str
                Path to save output image
            """
            all_speeds = []
            for vec_speed in self.u:
                all_speeds.extend(vec_speed)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(all_speeds, bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Speed (km/h)')
            ax.set_ylabel('Frequency')
            ax.set_title('Speed Distribution')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            
            return out_path
