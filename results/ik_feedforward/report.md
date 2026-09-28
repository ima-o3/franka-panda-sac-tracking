# Measured evaluation report

Descriptive episode means ± sample SD; paired seeds, not independent training runs.
Assembly SAC results are zero-shot generalisation. Candidate training provenance is unverified.

| Trajectory | Condition | Controller | Mean error (cm) ± SD | RMSE (cm) | Within 3 cm (%) | Command Δ (rad/step) |
|---|---|---|---:|---:|---:|---:|
| circle | clean | IK | 0.06 ± 0.00 | 0.07 | 100.0 | 0.00101 |
| circle | observation_noise | IK | 0.07 ± 0.00 | 0.08 | 100.0 | 0.00105 |
| circle | action_noise | IK | 0.57 ± 0.05 | 0.62 | 100.0 | 0.00520 |
| circle | noise | IK | 0.57 ± 0.04 | 0.63 | 100.0 | 0.00521 |
| circle | unreachable | IK | 22.29 ± 0.00 | 25.90 | 0.0 | 0.00177 |
| circle | stress | IK | 20.69 ± 3.22 | 24.60 | 0.0 | 0.00624 |
| assembly | clean | IK | 0.18 ± 0.00 | 0.21 | 100.0 | 0.00104 |
| assembly | observation_noise | IK | 0.19 ± 0.00 | 0.22 | 100.0 | 0.00114 |
| assembly | action_noise | IK | 0.59 ± 0.06 | 0.64 | 100.0 | 0.00528 |
| assembly | noise | IK | 0.59 ± 0.06 | 0.64 | 100.0 | 0.00529 |
| assembly | unreachable | IK | 22.05 ± 0.00 | 25.38 | 0.0 | 0.00187 |
| assembly | stress | IK | 20.44 ± 3.22 | 24.07 | 0.0 | 0.00625 |

## Data-driven observations

- circle/clean: IK has the lowest mean error (0.06 cm) among evaluated controllers.
- circle/observation_noise: IK has the lowest mean error (0.07 cm) among evaluated controllers.
- circle/action_noise: IK has the lowest mean error (0.57 cm) among evaluated controllers.
- circle/noise: IK has the lowest mean error (0.57 cm) among evaluated controllers.
- circle/unreachable: IK has the lowest mean error (22.29 cm) among evaluated controllers.
- circle/stress: IK has the lowest mean error (20.69 cm) among evaluated controllers.
- assembly/clean: IK has the lowest mean error (0.18 cm) among evaluated controllers.
- assembly/observation_noise: IK has the lowest mean error (0.19 cm) among evaluated controllers.
- assembly/action_noise: IK has the lowest mean error (0.59 cm) among evaluated controllers.
- assembly/noise: IK has the lowest mean error (0.59 cm) among evaluated controllers.
- assembly/unreachable: IK has the lowest mean error (22.05 cm) among evaluated controllers.
- assembly/stress: IK has the lowest mean error (20.44 cm) among evaluated controllers.

These rankings are descriptive, not significance tests. For unreachable references, inspect joint/command
limit occupancy, speed, action clipping and command second differences in summary.csv alongside distance.
A small command difference alone can also indicate a stuck controller, not useful tracking.
