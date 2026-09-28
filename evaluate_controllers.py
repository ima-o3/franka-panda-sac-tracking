"""Headless, paired IK/SAC evaluation. Never trains or saves a policy."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
from benchmark import ROOT, TrackingEnv, IKController, metrics, resolve_model

CONDITIONS = ("clean", "observation_noise", "action_noise", "noise", "unreachable", "stress")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rollout(env, policy, controller, seed, condition, obs_std=.002, action_std=.05):
    # Independent streams ensure controllers see identical draws regardless of policy.
    streams = np.random.SeedSequence(seed).spawn(3)
    obs_rng, action_rng, target_rng = [np.random.default_rng(s) for s in streams]
    shift = np.zeros(3)
    if condition in ("unreachable", "stress"):
        # Same circle centre as legacy uncertain environment; translated assembly.
        env.reset(seed=seed)
        origin = env.trajectory.home - [env.trajectory.radius, 0, 0]
        shift = np.array([1.05, 0., .45]) - origin
        if condition == "stress":
            shift += target_rng.normal(0, .03, 3)  # fixed per episode, not per step
    obs, _ = env.reset(seed=seed, options={"target_shift": shift})
    use_obs = condition in ("observation_noise", "noise", "stress")
    use_action = condition in ("action_noise", "noise", "stress")
    trace = {k: [] for k in ("time", "target", "actual", "error", "action_change",
        "command_change", "command_second_difference", "joint_speed", "joint_limit",
        "command_limit", "action_clip", "reward", "inference_ms", "phase", "qpos", "q_cmd",
        "raw_action", "executed_action")}
    previous_delta = np.zeros(7)
    for _ in range(env.max_episode_steps):
        measured = obs + obs_rng.normal(0, obs_std if use_obs else 0., 32).astype(np.float32)
        start = time.perf_counter()
        if controller == "ik":
            action = policy.predict(measured)
        else:
            action, _ = policy.predict(measured, deterministic=True)
        inference = (time.perf_counter()-start)*1000
        raw = np.clip(np.asarray(action, dtype=float), -1, 1)
        noisy = raw + action_rng.normal(0, action_std if use_action else 0., 7)
        executed = np.clip(noisy, -1, 1)
        previous_command = env.q_cmd.copy()
        obs, reward, terminated, truncated, info = env.step(executed)
        delta = env.q_cmd-previous_command
        values = dict(time=env.data.time, target=info["target_pos"], actual=info["ee_pos"],
            error=info["tracking_error"], action_change=info["action_change"],
            command_change=np.linalg.norm(delta),
            command_second_difference=np.linalg.norm(delta-previous_delta),
            joint_speed=info["joint_speed"],
            joint_limit=np.mean((env.data.qpos[:7] <= env.lower_limits+.01) |
                                (env.data.qpos[:7] >= env.upper_limits-.01)),
            command_limit=np.mean((env.q_cmd <= env.lower_limits+.01) |
                                  (env.q_cmd >= env.upper_limits-.01)),
            action_clip=np.mean(np.abs(noisy) > 1), reward=reward, inference_ms=inference,
            phase=info["phase"], qpos=env.data.qpos[:7].copy(), q_cmd=env.q_cmd.copy(),
            raw_action=raw, executed_action=executed)
        if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
            raise RuntimeError(f"Nonfinite simulation: {controller}/{condition}/{seed}")
        for key, value in values.items():
            trace[key].append(value)
        previous_delta = delta
        if terminated or truncated:
            break
    return {key: np.asarray(value) for key, value in trace.items()}


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controllers", nargs="+", choices=("ik", "clean_sac", "candidate_sac", "disturbed_sac"),
                        default=["ik", "clean_sac", "candidate_sac"])
    parser.add_argument("--trajectories", nargs="+", choices=("circle", "assembly"), default=["circle", "assembly"])
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--obs-std", type=float, default=.002)
    parser.add_argument("--action-std", type=float, default=.05)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--trajectory-config", type=Path, help="JSON keyword arguments for Trajectory")
    parser.add_argument("--clean-model", type=Path, default=ROOT/"Original_SAC/models_clean/best_model/best_model.zip")
    parser.add_argument("--candidate-model", type=Path, default=ROOT/"models/best_model/best_model.zip")
    parser.add_argument("--disturbed-model", type=Path, help="Explicit checkpoint with verified disturbance training")
    parser.add_argument("--ik-feedforward", action="store_true", help="Give IK privileged analytic target velocity")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0 or min(args.obs_std, args.action_std) < 0:
        parser.error("Positive episode count and nonnegative seed/noise required")
    if "disturbed_sac" in args.controllers and args.disturbed_model is None:
        parser.error("--disturbed-model is required; unverified models are never auto-labeled trained with disturbances")
    config = json.loads(args.trajectory_config.read_text()) if args.trajectory_config else {}
    model_path = resolve_model(args.model_path)
    policies, model_metadata = {}, {}
    if any(c != "ik" for c in args.controllers):
        import torch
        from stable_baselines3 import SAC
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        for name in args.controllers:
            if name == "ik":
                continue
            path = {"clean_sac": args.clean_model, "candidate_sac": args.candidate_model,
                    "disturbed_sac": args.disturbed_model}[name]
            policies[name] = SAC.load(path, device="cpu")
            if policies[name].observation_space.shape != (32,) or policies[name].action_space.shape != (7,):
                raise ValueError(f"Incompatible model: {path}")
            model_metadata[name] = {"filename": path.name, "sha256": sha256(path),
                "training_steps": policies[name].num_timesteps,
                "provenance": "unverified" if name == "candidate_sac" else "user-specified or archive clean"}
    output = args.output or ROOT/"results"/datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier experiment.
    (output/"traces").mkdir()
    packages = {p: importlib.metadata.version(p) for p in ("numpy", "mujoco", "gymnasium", "matplotlib")}
    if policies:
        packages.update({p: importlib.metadata.version(p) for p in ("torch", "stable-baselines3")})
    asset_hashes = {str(p.relative_to(model_path.parent)): sha256(p)
                   for p in sorted(model_path.parent.rglob("*")) if p.is_file() and ".git" not in p.parts}
    metadata = {"arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "trajectory_config": config, "models": model_metadata, "packages": packages,
        "python": platform.python_version(), "platform": platform.platform(), "assets_sha256": asset_hashes,
        "source_sha256": {p.name: sha256(p) for p in [Path(__file__), ROOT/"benchmark.py", ROOT/"trajectories.py",
            ROOT/"plot_results.py", ROOT/"Original_SAC/panda_circle_sac_env.py"]}}
    rows, examples = [], {}
    for kind in args.trajectories:
        for condition in args.conditions:
            for controller in args.controllers:
                env = TrackingEnv(kind, args.duration, model_path, args.frame_skip, config)
                metadata["control_dt_s"] = env.dt
                metadata["simulation_dt_s"] = env.model.opt.timestep
                metadata["steps_per_episode"] = env.max_episode_steps
                policy = IKController(env, feedforward=args.ik_feedforward) if controller == "ik" else policies[controller]
                try:
                    for episode in range(args.episodes):
                        seed = args.seed+episode
                        trace = rollout(env, policy, controller, seed, condition, args.obs_std, args.action_std)
                        row = dict(controller=controller, trajectory=kind, condition=condition, episode=episode, seed=seed,
                            generalisation="zero-shot" if kind == "assembly" and controller != "ik" else "nominal",
                            **metrics(trace, controller))
                        rows.append(row)
                        np.savez_compressed(output/"traces"/f"{kind}_{condition}_{controller}_{seed}.npz", **trace)
                        if episode == 0:
                            examples[(kind, condition, controller)] = trace
                        print(f"{kind:8} {condition:17} {controller:14} seed={seed}: mean={row['mean_error_m']:.4f} m", flush=True)
                        save_csv(output/"episodes.csv", rows)
                finally:
                    env.close()
    (output/"manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    from plot_results import make_report
    make_report(output, rows, examples)
    print(f"Saved results to {output.resolve()}")


if __name__ == "__main__":
    main()
