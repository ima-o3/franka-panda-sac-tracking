from panda_tracking_env import PandaTrackingEnv


env = PandaTrackingEnv()
obs, info = env.reset()

total_reward = 0.0

for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if step % 20 == 0:
        print(
            f"step={step}, "
            f"reward={reward:.3f}, "
            f"tracking_error={info['tracking_error']:.3f}"
        )

    if terminated or truncated:
        break

print("Total reward:", total_reward)
env.close()