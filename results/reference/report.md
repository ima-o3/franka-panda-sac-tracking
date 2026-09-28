# Measured evaluation report

Descriptive episode means ± sample SD; paired seeds, not independent training runs.
Assembly SAC results are zero-shot generalisation. Candidate training provenance is unverified.

| Trajectory | Condition | Controller | Mean error (cm) ± SD | RMSE (cm) | Within 3 cm (%) | Command Δ (rad/step) |
|---|---|---|---:|---:|---:|---:|
| circle | clean | IK | 1.00 ± 0.00 | 1.00 | 100.0 | 0.00099 |
| circle | clean | Clean SAC | 0.17 ± 0.00 | 0.28 | 100.0 | 0.00245 |
| circle | clean | SAC candidate (unverified) | 3.55 ± 0.00 | 5.32 | 62.3 | 0.00906 |
| circle | observation_noise | IK | 1.00 ± 0.00 | 1.00 | 100.0 | 0.00103 |
| circle | observation_noise | Clean SAC | 0.31 ± 0.01 | 0.38 | 100.0 | 0.01996 |
| circle | observation_noise | SAC candidate (unverified) | 3.43 ± 0.05 | 5.16 | 62.6 | 0.01864 |
| circle | action_noise | IK | 1.15 ± 0.04 | 1.19 | 100.0 | 0.00520 |
| circle | action_noise | Clean SAC | 0.18 ± 0.01 | 0.28 | 100.0 | 0.00670 |
| circle | action_noise | SAC candidate (unverified) | 3.47 ± 0.05 | 5.17 | 62.5 | 0.01406 |
| circle | noise | IK | 1.14 ± 0.04 | 1.19 | 100.0 | 0.00521 |
| circle | noise | Clean SAC | 0.31 ± 0.01 | 0.39 | 100.0 | 0.02092 |
| circle | noise | SAC candidate (unverified) | 3.63 ± 0.11 | 5.63 | 62.7 | 0.01957 |
| circle | unreachable | IK | 22.32 ± 0.00 | 25.92 | 0.0 | 0.00176 |
| circle | unreachable | Clean SAC | 45.70 ± 0.00 | 46.33 | 0.0 | 0.00802 |
| circle | unreachable | SAC candidate (unverified) | 28.46 ± 0.00 | 31.00 | 0.0 | 0.02116 |
| circle | stress | IK | 20.71 ± 3.22 | 24.62 | 0.0 | 0.00624 |
| circle | stress | Clean SAC | 44.99 ± 2.09 | 45.58 | 0.0 | 0.01892 |
| circle | stress | SAC candidate (unverified) | 28.28 ± 4.98 | 31.29 | 0.0 | 0.02272 |
| assembly | clean | IK | 1.07 ± 0.00 | 1.42 | 97.4 | 0.00099 |
| assembly | clean | Clean SAC | 0.79 ± 0.00 | 0.86 | 100.0 | 0.00230 |
| assembly | clean | SAC candidate (unverified) | 6.94 ± 0.00 | 7.58 | 10.2 | 0.00568 |
| assembly | observation_noise | IK | 1.08 ± 0.00 | 1.42 | 97.5 | 0.00109 |
| assembly | observation_noise | Clean SAC | 0.86 ± 0.00 | 0.92 | 100.0 | 0.01952 |
| assembly | observation_noise | SAC candidate (unverified) | 6.95 ± 0.01 | 7.58 | 10.2 | 0.01620 |
| assembly | action_noise | IK | 1.29 ± 0.05 | 1.54 | 96.0 | 0.00526 |
| assembly | action_noise | Clean SAC | 0.80 ± 0.01 | 0.87 | 100.0 | 0.00669 |
| assembly | action_noise | SAC candidate (unverified) | 6.94 ± 0.02 | 7.57 | 10.2 | 0.01154 |
| assembly | noise | IK | 1.29 ± 0.05 | 1.54 | 96.0 | 0.00527 |
| assembly | noise | Clean SAC | 0.87 ± 0.01 | 0.93 | 100.0 | 0.02052 |
| assembly | noise | SAC candidate (unverified) | 6.95 ± 0.02 | 7.58 | 10.2 | 0.01837 |
| assembly | unreachable | IK | 22.13 ± 0.00 | 25.44 | 0.0 | 0.00187 |
| assembly | unreachable | Clean SAC | 43.53 ± 0.00 | 44.01 | 0.0 | 0.01906 |
| assembly | unreachable | SAC candidate (unverified) | 28.53 ± 0.00 | 30.05 | 0.0 | 0.02346 |
| assembly | stress | IK | 20.53 ± 3.22 | 24.13 | 0.0 | 0.00625 |
| assembly | stress | Clean SAC | 42.71 ± 2.25 | 43.16 | 0.0 | 0.02438 |
| assembly | stress | SAC candidate (unverified) | 25.96 ± 4.84 | 27.18 | 0.0 | 0.02447 |

## Data-driven observations

- circle/clean: Clean SAC has the lowest mean error (0.17 cm) among evaluated controllers.
- circle/observation_noise: Clean SAC has the lowest mean error (0.31 cm) among evaluated controllers.
- circle/action_noise: Clean SAC has the lowest mean error (0.18 cm) among evaluated controllers.
- circle/noise: Clean SAC has the lowest mean error (0.31 cm) among evaluated controllers.
- circle/unreachable: IK has the lowest mean error (22.32 cm) among evaluated controllers.
- circle/stress: IK has the lowest mean error (20.71 cm) among evaluated controllers.
- assembly/clean: Clean SAC has the lowest mean error (0.79 cm) among evaluated controllers.
- assembly/observation_noise: Clean SAC has the lowest mean error (0.86 cm) among evaluated controllers.
- assembly/action_noise: Clean SAC has the lowest mean error (0.80 cm) among evaluated controllers.
- assembly/noise: Clean SAC has the lowest mean error (0.87 cm) among evaluated controllers.
- assembly/unreachable: IK has the lowest mean error (22.13 cm) among evaluated controllers.
- assembly/stress: IK has the lowest mean error (20.53 cm) among evaluated controllers.

These rankings are descriptive, not significance tests. For unreachable references, inspect joint/command
limit occupancy, speed, action clipping and command second differences in summary.csv alongside distance.
A small command difference alone can also indicate a stuck controller, not useful tracking.
