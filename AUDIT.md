# Repository audit (28 September 2026)

The supplied ZIP was extracted without replacing workspace files. All historical
checkpoints, PNGs, TensorBoard logs and evaluation NPZ files are retained. The ZIP
contains no AGENTS.md and no .gitignore, despite the original README listing one.

## Original architecture and findings

| Component | Status before changes |
|---|---|
| `Original_SAC/panda_circle_sac_env.py` | 32-dimensional observations, 7 incremental position commands, hand-body origin tracking, 5 physics steps/action, 500-step horizon. Model path depended on folder depth. |
| `Noisy_SAC/panda_circle_sac_env_uncertain.py` | Copy of clean environment with Gaussian noise on all 32 observation entries, normalized action noise, per-episode unreachable circle and jitter. Jitter is once per episode, contrary to training comment. |
| `circle_ik_panda.py` | Interactive DLS controller with analytic feedforward, 30-second run and one physics step/update. Not a matched comparison with SAC. |
| Clean training | SAC MLP defaults, 1M requested steps, evaluations every 10k, checkpoints every 25k, no seed; saves relative to current directory. |
| Noisy training | Same algorithm, observation std .002, action std .05, unreachable probability .20, centre jitter .03. Writes the same relative models/logs paths as clean training. |
| Playback scripts | Usable with matching dependencies/model paths, but depend on current working directory. Noisy playback defaults all disturbances to zero. |
| `Original_SAC/inspect_best_eval.py` | Reads cwd-relative evaluation log; reward inspection only. |
| `Testing/` | Earlier 20-input environment and manual viewer/random-action checks; incompatible with current 32-input saved policies. Reset does not set actual arm qpos to home. Not an automated test suite. |
| `backup_original_sac/` | Historical duplicate of clean SAC; environment differs only in model path. Retained as requested. |
| `models/` | Root best checkpoint differs from clean best; training provenance cannot be established. User confirmed uncertainty. Not labeled disturbance-trained. |
| README | Escaped Markdown, obsolete file layout, reward formula inconsistent with both supplied environments, no final quantitative comparison. |
| requirements | Full historical environment freeze; preserved as requirements-original.txt. |

The clean best model metadata records 640,000 training steps; root candidate records
160,000. Both were saved with SB3 2.8.0 and no training seed. The clean best and backup
best are byte-identical. `artifact_inventory.json` records every supplied checkpoint
and historical result hash. Neither filenames nor differing weights prove noisy
training. Historical plots are evidence of earlier demos, not matched experiments.

## Minimum implementation

1. Add `trajectories.py`: one circle definition and a configurable quintic waypoint trajectory.
2. Add `benchmark.py`: subclass the clean environment, retain action/reward/observation contract,
   share the plant with IK; use a scratch state for IK's measured joint positions.
3. Add `evaluate_controllers.py` and `plot_results.py`: paired seeded rollouts, metrics,
   version/hash manifest, raw traces, CSVs, PNG/PDF plots and a generated report.
4. Add numerical/simulation tests and JSON trajectory configuration.
5. Add optional model path to the clean constructor, without changing legacy defaults.
6. Replace README with accurate methods/results and add ignores and tested dependencies.

The benchmark refreshes MuJoCo kinematics after stepping, correcting the legacy
one-physics-step Cartesian sampling lag. Legacy demos and training remain unchanged
apart from the optional clean model-path parameter. They are historical entry points;
the documented evaluator is the supported comparison path. No files were deleted,
no policies were trained, and no historical checkpoint/result was overwritten.

## Validation scope

See README and results/reference/manifest.json for the tested runtime, model assets,
commands and measured comparison. Legacy viewers/training runs were inspected, not
rerun; they are not claimed to be validated in the new runtime. Exact historical
MuJoCo asset revision and training runtime are not recoverable from the archive.
