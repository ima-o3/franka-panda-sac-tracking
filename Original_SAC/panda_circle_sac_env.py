from pathlib import Path
import time

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np


class PandaCircleSACEnv(gym.Env):
    """
    Franka Panda circular end-effector tracking environment.

    Action:
        7D continuous action in [-1, 1].
        Each action changes the commanded joint target slightly.

    Observation:
        qpos[7], qvel[7], end-effector position[3],
        target position[3], tracking error[3],
        sin/cos trajectory phase[2], commanded joint targets[7].

    Reward:
        Dense reward based mainly on end-effector tracking error,
        with small penalties for large/sudden actions and joint velocity.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, max_episode_steps=500):
        super().__init__()

        self.model_path = (
            Path(__file__).resolve().parents[2]
            / "mujoco_menagerie"
            / "franka_emika_panda"
            / "scene.xml"
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Could not find model at:\n{self.model_path}\n\n"
                "Expected folder structure:\n"
                "mujoco_projects/\n"
                "  mujoco_menagerie/\n"
                "    franka_emika_panda/\n"
                "      scene.xml\n"
                "  panda_tracking_rl/\n"
                "    panda_circle_sac_env.py\n"
            )

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None

        self.n_arm_joints = 7
        self.frame_skip = 5
        self.dt = self.model.opt.timestep * self.frame_skip

        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        self.ee_body_name = "hand"
        self.ee_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.ee_body_name,
        )

        if self.ee_body_id == -1:
            print("Available bodies:")
            for i in range(self.model.nbody):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                print(i, name)
            raise ValueError("Could not find end-effector body named 'hand'.")

        # Franka-ish home pose.
        self.q_home = np.array([0.0, -0.7, 0.0, -2.0, 0.0, 1.6, 0.7], dtype=np.float64)

        self.lower_limits, self.upper_limits = self._get_joint_limits()

        # Circle settings.
        self.radius = 0.08      # metres
        self.omega = 0.5        # rad/s
        self.circle_center = None

        # Action is normalised. Actual joint target delta = action * action_scale.
        self.action_scale = 0.04  # rad per environment step

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )

        # qpos 7 + qvel 7 + ee 3 + target 3 + error 3 + phase 2 + q_cmd 7 = 32
        obs_dim = 32
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.q_cmd = self.q_home.copy()
        self.prev_action = np.zeros(7, dtype=np.float64)

    def _get_joint_limits(self):
        """
        Get joint limits from the MuJoCo model.
        If unavailable, use approximate Franka Panda limits.
        """
        fallback_lower = np.array(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            dtype=np.float64,
        )
        fallback_upper = np.array(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            dtype=np.float64,
        )

        try:
            lower = self.model.jnt_range[:7, 0].copy()
            upper = self.model.jnt_range[:7, 1].copy()

            if np.all(upper > lower):
                return lower, upper
        except Exception:
            pass

        return fallback_lower, fallback_upper

    def _target_position(self):
        """
        Circular target trajectory in the XY plane.
        """
        t = self.data.time

        return self.circle_center + np.array(
            [
                self.radius * np.cos(self.omega * t),
                self.radius * np.sin(self.omega * t),
                0.0,
            ],
            dtype=np.float64,
        )

    def _get_obs(self):
        qpos = self.data.qpos[:7].copy()
        qvel = self.data.qvel[:7].copy()

        ee_pos = self.data.xpos[self.ee_body_id].copy()
        target_pos = self._target_position()
        error = target_pos - ee_pos

        phase = self.omega * self.data.time
        phase_features = np.array([np.sin(phase), np.cos(phase)], dtype=np.float64)

        obs = np.concatenate(
            [
                qpos,
                qvel,
                ee_pos,
                target_pos,
                error,
                phase_features,
                self.q_cmd,
            ]
        )

        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self.step_count = 0
        self.q_cmd = self.q_home.copy()
        self.prev_action = np.zeros(7, dtype=np.float64)

        # Set arm joint positions.
        self.data.qpos[:7] = self.q_home

        # Open gripper slightly if the model has finger joints.
        if self.model.nq >= 9:
            self.data.qpos[7:9] = np.array([0.04, 0.04])

        # Set actuator targets.
        self.data.ctrl[:7] = self.q_cmd

        mujoco.mj_forward(self.model, self.data)

        # Make the circle start at the current end-effector location.
        ee_start = self.data.xpos[self.ee_body_id].copy()
        self.circle_center = ee_start + np.array([-self.radius, 0.0, 0.0])

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        # Convert normalised SAC action into a small joint target change.
        self.q_cmd = self.q_cmd + action * self.action_scale
        self.q_cmd = np.clip(self.q_cmd, self.lower_limits, self.upper_limits)

        self.data.ctrl[:7] = self.q_cmd

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        ee_pos = self.data.xpos[self.ee_body_id].copy()
        target_pos = self._target_position()
        error_vec = target_pos - ee_pos

        tracking_error = np.linalg.norm(error_vec)
        action_effort = np.linalg.norm(action)
        action_change = np.linalg.norm(action - self.prev_action)
        joint_speed = np.linalg.norm(self.data.qvel[:7])

        # Main objective: minimise tracking error.
        # The other terms discourage aggressive/jerky motion.
        reward = (
            -10.0 * tracking_error
            -0.01 * action_effort
            -0.02 * action_change
            -0.001 * joint_speed
        )

        # Small bonus for being close to the moving target.
        if tracking_error < 0.03:
            reward += 0.5

        self.prev_action = action.copy()

        terminated = False
        truncated = self.step_count >= self.max_episode_steps

        info = {
            "tracking_error": tracking_error,
            "ee_pos": ee_pos,
            "target_pos": target_pos,
            "action_effort": action_effort,
            "action_change": action_change,
            "joint_speed": joint_speed,
            "sim_time": self.data.time,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None