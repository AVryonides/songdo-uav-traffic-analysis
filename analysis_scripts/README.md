# Analysis Scripts

This folder contains the Python scripts used for the thesis analysis. They are
included as adaptable research code rather than a polished Python package.

The scripts are organized by intersection:

```text
songdo_q/   Q intersection pipeline and analysis tools
songdo_m/   M intersection pipeline and analysis tools
songdo_r/   R intersection pipeline and analysis tools
```

The lightweight runner files are:

```text
uavsongdopie_Q.py
uavsongdopie_M.py
uavsongdopie_R.py
```

The `tool_songdo*.py` files provide supporting geometry/utility functionality
used by the intersection-specific pipelines.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

## Expected Inputs

The scripts expect UAV trajectory CSV files and intersection geometry CSV files.
The raw data are not included in this public repository.

Typical trajectory columns used by the pipeline include:

```text
Vehicle_ID
Local_Time
Ortho_X
Ortho_Y
Local_X
Local_Y
Latitude
Longitude
Vehicle_Class
Vehicle_Speed
Vehicle_Acceleration
Road_Section
Lane_Number
Visibility
```

Geometry files describe the intersection road sections used to assign vehicles
to movements and lanes.

## Example Usage

Place your trajectory and geometry CSV files in a local working folder, then run
the matching intersection pipeline from this directory:

```bash
cd analysis_scripts
python -m songdo_q --traj /path/to/trajectory_Q.csv --seg /path/to/Q.csv --out /path/to/outputs_Q
```

For M and R:

```bash
python -m songdo_m --traj /path/to/trajectory_M.csv --seg /path/to/M.csv --out /path/to/outputs_M
python -m songdo_r --traj /path/to/trajectory_R.csv --seg /path/to/R.csv --out /path/to/outputs_R
```

## Adapting to a New Intersection

To reuse the workflow on another intersection, the following usually need to be
adjusted:

1. Road-section geometry and stop-line definitions.
2. Movement rules mapping origins to destinations.
3. Lane labels and lane grouping logic.
4. Protected versus permissive movement definitions.
5. Time windows used for visual inspection and composite diagrams.
6. Signal inference thresholds if the trajectory quality or traffic behavior is
   different.

The Q implementation is the most complete reference version. M and R extend the
same logic to larger non-T-junction intersections.
