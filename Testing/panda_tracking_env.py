from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class PandaTrackingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        self.model_path = Path("../mujoco_menagerie/franka_emika_panda/scene.xml")
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None

        self.ee_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "hand",
        )

        if self.ee_body_id == -1:
            raise ValueError("Could not find body named 'hand'. Print body names and choose another one.")

        self.num_arm_joints = 7
        self.t = 0.0
        self.max_steps = 500
        self.step_count = 0

        # Action = small changes to 7 joint position targets.
        self.action_space = spaces.Box(
            low=-0.05,
            high=0.05,
            shape=(7,),
            dtype=np.float32,
        )

        # Observation = qpos(7) + qvel(7) + ee_pos(3) + target_pos(3)
        obs_dim = 7 + 7 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.ctrl = np.array([0.0, -0.7, 0.0, -2.0, 0.0, 1.6, 0.7], dtype=np.float64)

    def target_position(self):
        radius = 0.10
        omega = 0.5
        center = np.array([0.5, 0.0, 0.4])

        return np.array([
            center[0] + radius * np.cos(omega * self.t),
            center[1] + radius * np.sin(omega * self.t),
            center[2],
        ])

    def get_obs(self):
        qpos = self.data.qpos[:7].copy()
        qvel = self.data.qvel[:7].copy()
        ee_pos = self.data.xpos[self.ee_body_id].copy()
        target = self.target_position()

        obs = np.concatenate([qpos, qvel, ee_pos, target])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self.ctrl = np.array([0.0, -0.7, 0.0, -2.0, 0.0, 1.6, 0.7], dtype=np.float64)
        self.data.ctrl[:7] = self.ctrl

        self.t = 0.0
        self.step_count = 0

        mujoco.mj_forward(self.model, self.data)

        return self.get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)

        # Update joint target controls.
        self.ctrl += action
        self.data.ctrl[:7] = self.ctrl

        # Step several small physics steps per RL action.
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        self.t = self.data.time
        self.step_count += 1

        ee_pos = self.data.xpos[self.ee_body_id]
        target = self.target_position()

        tracking_error = np.linalg.norm(ee_pos - target)
        action_penalty = np.linalg.norm(action)

        reward = -tracking_error - 0.01 * action_penalty

        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "tracking_error": tracking_error,
            "ee_pos": ee_pos.copy(),
            "target_pos": target.copy(),
        }

        return self.get_obs(), reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        if self.viewer is not None:
            self.viewer.close()