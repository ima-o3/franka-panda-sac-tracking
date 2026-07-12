import time

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from panda_circle_sac_env_uncertain import PandaCircleSACEnv


def main():
    model_path = "./models/best_model/best_model"

    env = PandaCircleSACEnv(render_mode="human", max_episode_steps=1000)
    model = SAC.load(model_path)

    obs, info = env.reset()

    errors = []
    ee_positions = []
    target_positions = []

    total_reward = 0.0

    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        errors.append(info["tracking_error"])
        ee_positions.append(info["ee_pos"])
        target_positions.append(info["target_pos"])

        if step % 20 == 0:
            print(
                f"step={step:4d} | "
                f"reward={reward: .3f} | "
                f"tracking_error={info['tracking_error']:.4f} m"
            )

        time.sleep(env.dt)

        if terminated or truncated:
            break

    env.close()

    errors = np.array(errors)
    ee_positions = np.array(ee_positions)
    target_positions = np.array(target_positions)

    print("\nEvaluation finished.")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Mean tracking error: {errors.mean():.4f} m")
    print(f"Final tracking error: {errors[-1]:.4f} m")

    plt.figure()
    plt.plot(target_positions[:, 0], target_positions[:, 1], label="Target circle")
    plt.plot(ee_positions[:, 0], ee_positions[:, 1], label="SAC end-effector")
    plt.xlabel("x position [m]")
    plt.ylabel("y position [m]")
    plt.title("SAC Panda Circular Tracking")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(errors)
    plt.xlabel("Step")
    plt.ylabel("Tracking error [m]")
    plt.title("SAC Tracking Error")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()