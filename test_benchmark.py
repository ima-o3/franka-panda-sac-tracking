"""Numerical contract checks: python -m unittest -v test_benchmark."""
import unittest
import numpy as np
from trajectories import Trajectory
from benchmark import TrackingEnv, IKController, metrics
from evaluate_controllers import rollout


class TrajectoryTests(unittest.TestCase):
    def test_circle_matches_legacy(self):
        home = np.array([.4, 0., .5])
        tr = Trajectory(home)
        for t in [0., .01, 5., 20.]:
            expected = home + [-.08, 0, 0] + .08*np.array([np.cos(.5*t), np.sin(.5*t), 0])
            np.testing.assert_allclose(tr.sample(t)[0], expected, atol=1e-14)

    def test_assembly_continuity_and_velocity(self):
        tr = Trajectory(np.array([.4, 0., .5]), kind="assembly")
        for t in tr.times:
            np.testing.assert_allclose(tr.sample(t-1e-7)[0], tr.sample(t+1e-7)[0], atol=1e-10)
            np.testing.assert_allclose(tr.sample(t)[1], 0, atol=1e-10)
        np.testing.assert_allclose(tr.sample(20)[0], tr.home, atol=1e-14)
        for t in np.linspace(.1, 19.9, 50):
            finite_difference = (tr.sample(t+1e-5)[0]-tr.sample(t-1e-5)[0])/2e-5
            np.testing.assert_allclose(tr.sample(t)[1], finite_difference, atol=1e-8)
        self.assertLess(np.linalg.norm(tr.sample(13.)[1]), .4)


class SimulationTests(unittest.TestCase):
    def test_shared_target_and_seeded_rollout(self):
        env = TrackingEnv(duration=.2)
        try:
            a = rollout(env, IKController(env), "ik", 42, "stress")
            b = rollout(env, IKController(env), "ik", 42, "stress")
            for key in a:
                if key != "inference_ms":
                    np.testing.assert_array_equal(a[key], b[key])
            c = rollout(env, IKController(env), "ik", 43, "stress")
            self.assertFalse(np.array_equal(a["executed_action"], c["executed_action"]))
            self.assertTrue(np.all(a["q_cmd"] >= env.lower_limits))
            self.assertTrue(np.all(a["q_cmd"] <= env.upper_limits))
            self.assertAlmostEqual(a["time"][-1], .2)
            self.assertEqual(len(a["time"]), env.max_episode_steps)
        finally:
            env.close()

    def test_gym_contract_and_exact_position_time(self):
        from stable_baselines3.common.env_checker import check_env
        import mujoco
        env = TrackingEnv(trajectory="assembly", duration=.2)
        try:
            check_env(env, warn=True)
            env.reset(seed=42)
            obs, _, _, _, info = env.step(np.ones(7)*.1)
            scratch = mujoco.MjData(env.model)
            scratch.qpos[:] = env.data.qpos
            mujoco.mj_forward(env.model, scratch)
            np.testing.assert_allclose(info["ee_pos"], scratch.xpos[env.ee_body_id])
            np.testing.assert_allclose(obs[17:20], info["target_pos"], atol=1e-7)
        finally:
            env.close()

    def test_paired_target_and_action_noise(self):
        class Constant:
            def predict(self, obs, deterministic=True):
                return np.zeros(7), None
        env = TrackingEnv(duration=.2)
        try:
            a = rollout(env, Constant(), "clean_sac", 9, "stress")
            b = rollout(env, Constant(), "candidate_sac", 9, "stress")
            np.testing.assert_array_equal(a["target"], b["target"])
            np.testing.assert_array_equal(a["executed_action"], b["executed_action"])
            result = metrics(a, "clean_sac")
            self.assertAlmostEqual(result["rmse_m"]**2, np.mean(a["error"]**2))
            self.assertAlmostEqual(result["total_reward"], sum(a["reward"]))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
