| Trajectory | Condition | IK feedback | IK + feedforward | Clean SAC | SAC candidate* |
|---|---|---:|---:|---:|---:|
| circle | clean | 1.00 ± 0.00 | 0.06 ± 0.00 | 0.17 ± 0.00 | 3.55 ± 0.00 |
| circle | observation_noise | 1.00 ± 0.00 | 0.07 ± 0.00 | 0.31 ± 0.01 | 3.43 ± 0.05 |
| circle | action_noise | 1.15 ± 0.04 | 0.57 ± 0.05 | 0.18 ± 0.01 | 3.47 ± 0.05 |
| circle | noise | 1.14 ± 0.04 | 0.57 ± 0.04 | 0.31 ± 0.01 | 3.63 ± 0.11 |
| circle | unreachable | 22.32 ± 0.00 | 22.29 ± 0.00 | 45.70 ± 0.00 | 28.46 ± 0.00 |
| circle | stress | 20.71 ± 3.22 | 20.69 ± 3.22 | 44.99 ± 2.09 | 28.28 ± 4.98 |
| assembly | clean | 1.07 ± 0.00 | 0.18 ± 0.00 | 0.79 ± 0.00 | 6.94 ± 0.00 |
| assembly | observation_noise | 1.08 ± 0.00 | 0.19 ± 0.00 | 0.86 ± 0.00 | 6.95 ± 0.01 |
| assembly | action_noise | 1.29 ± 0.05 | 0.59 ± 0.06 | 0.80 ± 0.01 | 6.94 ± 0.02 |
| assembly | noise | 1.29 ± 0.05 | 0.59 ± 0.06 | 0.87 ± 0.01 | 6.95 ± 0.02 |
| assembly | unreachable | 22.13 ± 0.00 | 22.05 ± 0.00 | 43.53 ± 0.00 | 28.53 ± 0.00 |
| assembly | stress | 20.53 ± 3.22 | 20.44 ± 3.22 | 42.71 ± 2.25 | 25.96 ± 4.84 |

Mean Cartesian error in cm ± sample SD across episode means. *Training provenance unverified.
Assembly SAC is zero-shot; IK feedforward receives analytic reference velocity.
