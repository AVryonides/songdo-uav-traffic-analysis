# Output Guide

This document explains the selected outputs included in the public repository.

## Figures

| Figure | File | Purpose |
| --- | --- | --- |
| Q intersection layout | `assets/figures/q_intersection_layout.png` | Shows the analyzed Q intersection geometry |
| Turning movements | `assets/figures/turning_movements.png` | Shows the movement directions used in the analysis |
| Recording timeline | `assets/figures/recording_timeline.png` | Shows the analyzed recording windows |
| Route map | `assets/figures/route_map.png` | Shows vehicle routes/movements in the Q case study |
| Protected signal timing | `assets/figures/protected_signal_timing.png` | Shows inferred protected-movement signal intervals |
| Permissive signal timing | `assets/figures/permissive_signal_timing.png` | Shows permissive movement timing behavior |
| Composite space-time | `assets/figures/composite_space_time_all.png` | Stitched clean windows from selected lanes and time ranges |
| Cumulative ES | `assets/figures/cumulative_es_1740_1744.png` | Cumulative arrival/departure example for East to South movement |
| Headway model | `assets/figures/headway_vehicle_order.png` | Lane-level headway versus vehicle order in queue |
| Validation plot | `assets/figures/model_validation_pairwise.png` | Observed versus modelled vehicle counts |

## Tables

| Table | File | Purpose |
| --- | --- | --- |
| Lane saturation model | `results/lane_saturation_model_table.csv` | Lane-level `h_sat`, `t_C`, and `r_sat` |
| Saturation summary | `results/lane_saturation_model_summary.csv` | Summary of protected lane rates |
| Cycle validation | `results/traffic_light_capped_median_validation_table.csv` | Lane-cycle model validation rows |
| Validation summary | `results/traffic_light_validation_summary.csv` | Error metrics, including MAE |

## Privacy and Data Scope

Only processed figures and compact result tables are included. The full UAV
trajectory dataset and full generated output directories are excluded. The
thesis PDF is included separately in `thesis/thesis.pdf`.
