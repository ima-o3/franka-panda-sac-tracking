"""Common plant, observation contract and classical controller for evaluation."""
import os
from pathlib import Path
import mujoco
import numpy as np
from Original_SAC.panda_circle_sac_env import PandaCircleSACEnv
from trajectories import Trajectory

ROOT = Path(__file__).resolve().parent


def resolve_model(path=None):
    if path or os.environ.get("PANDA_MODEL_PATH"):
        result = Path(path or os.environ["PANDA_MODEL_PATH"]).expanduser().resolve()
        if result.is_file():
            return result
        raise FileNotFoundError(f"Panda scene not found: {result}")
    for parent in (ROOT, ROOT.parent):
        result = parent / "mujoco_menagerie/franka_emika_panda/scene.xml"
        if result.is_file():
            return result
    raise FileNotFoundError("Install MuJoCo Menagerie beside this project or pass --model-path scene.xml")


class TrackingEnv(PandaCircleSACEnv):
    def __init__(self, trajectory="circle", duration=20., model_path=None, frame_skip=5,
                 trajectory_config=None):
        if frame_skip < 1 or duration <= 0:
            raise ValueError("Positive duration and frame_skip required")
        self.trajectory = None
        self.kind = trajectory
        self.duration = duration
        self.trajectory_config = trajectory_config or {}
        super().__init__(model_path=resolve_model(model_path))
        self.radius = self.trajectory_config.get("radius", self.radius)
        self.omega = self.trajectory_config.get("omega", self.omega)
        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip
        self.max_episode_steps = int(round(duration / self.dt))
        if self.max_episode_steps < 1 or not np.isclose(self.max_episode_steps*self.dt, duration):
            raise ValueError("Duration must be an integer multiple of the control timestep")

    def reset(self, *, seed=None, options=None):
        self.trajectory = None
        super().reset(seed=seed)
        self.trajectory = Trajectory(self.data.xpos[self.ee_body_id].copy(),
            kind=self.kind, duration=self.duration, **self.trajectory_config)
        self.target_shift = np.asarray((options or {}).get("target_shift", [0., 0., 0.]))
        return self._get_obs(), {}

    def _target_position(self):
        if self.trajectory is None:
            return super()._target_position()
        return self.trajectory.sample(self.data.time)[0] + self.target_shift

    def step(self, action):
        # Reuse legacy action integration, limits and stepping. Refresh kinematics
        # after mj_step so metrics pair the final qpos and Cartesian position.
        _, _, terminated, truncated, info = super().step(action)
        mujoco.mj_forward(self.model, self.data)
        ee = self.data.xpos[self.ee_body_id].copy()
        target = self._target_position()
        error = float(np.linalg.norm(target-ee))
        reward = (-10*error - .01*info["action_effort"] - .02*info["action_change"]
                  - .001*info["joint_speed"] + (.5 if error < .03 else 0))
        info.update(ee_pos=ee, target_pos=target, tracking_error=error,
                    phase=self.trajectory.sample(self.data.time)[2])
        return self._get_obs(), reward, terminated, truncated, info


class IKController:
    """Original DLS law; measured q used in a separate kinematic scratch state.

    Feedforward is opt-in because the existing SAC policy has no velocity input.
    The original 1 rad/s internal velocity cap is retained; plant bounds are shared.
    """
    def __init__(self, env, kp=4., damping=.05, max_joint_speed=1., feedforward=False):
        self.env = env
        self.kp, self.damping = kp, damping
        self.max_joint_speed, self.feedforward = max_joint_speed, feedforward
        self.scratch = mujoco.MjData(env.model)

    def predict(self, obs):
        env = self.env
        self.scratch.qpos[:] = env.data.qpos
        self.scratch.qpos[:7] = obs[:7]
        mujoco.mj_forward(env.model, self.scratch)
        jac = np.zeros((3, env.model.nv))
        mujoco.mj_jacBody(env.model, self.scratch, jac, None, env.ee_body_id)
        desired = self.kp * np.asarray(obs[20:23], dtype=float)
        if self.feedforward:
            desired += env.trajectory.sample(env.data.time)[1]
        desired *= min(1., .35/max(np.linalg.norm(desired), 1e-12))
        J = jac[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + self.damping**2*np.eye(3), desired)
        return np.clip(dq, -self.max_joint_speed, self.max_joint_speed)*env.dt/env.action_scale


def metrics(trace, controller):
    e = trace["error"]
    return {
        "mean_error_m": float(e.mean()), "rmse_m": float(np.sqrt(np.mean(e**2))),
        "max_error_m": float(e.max()), "p95_error_m": float(np.percentile(e, 95)),
        "within_3cm_pct": float(100*np.mean(e < .03)),
        "within_5cm_pct": float(100*np.mean(e < .05)),
        "mean_action_change": float(trace["action_change"].mean()),
        "mean_command_change_rad": float(trace["command_change"].mean()),
        "mean_command_second_difference_rad": float(trace["command_second_difference"].mean()),
        "mean_joint_speed_rad_s": float(trace["joint_speed"].mean()),
        "max_joint_speed_rad_s": float(trace["joint_speed"].max()),
        "command_limit_pct": float(100*trace["command_limit"].mean()),
        "joint_limit_pct": float(100*trace["joint_limit"].mean()),
        "action_clip_pct": float(100*trace["action_clip"].mean()),
        "closest_distance_m": float(e.min()),
        "total_reward": float(trace["reward"].sum()) if controller != "ik" else "",
        "mean_inference_ms": float(trace["inference_ms"].mean()),
    }
