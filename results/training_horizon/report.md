# Measured evaluation report

Descriptive episode means ± sample SD; paired seeds, not independent training runs.
Assembly SAC results are zero-shot generalisation. Candidate training provenance is unverified.

| Trajectory | Condition | Controller | Mean error (cm) ± SD | RMSE (cm) | Within 3 cm (%) | Command Δ (rad/step) |
|---|---|---|---:|---:|---:|---:|
| circle | clean | IK | 0.98 ± 0.00 | 0.99 | 100.0 | 0.00099 |
| circle | clean | Clean SAC | 0.02 ± 0.00 | 0.02 | 100.0 | 0.00223 |
| circle | clean | SAC candidate (unverified) | 0.54 ± 0.00 | 0.57 | 100.0 | 0.00533 |

## Data-driven observations

- circle/clean: Clean SAC has the lowest mean error (0.02 cm) among evaluated controllers.

These rankings are descriptive, not significance tests. For unreachable references, inspect joint/command
limit occupancy, speed, action clipping and command second differences in summary.csv alongside distance.
A small command difference alone can also indicate a stuck controller, not useful tracking.
