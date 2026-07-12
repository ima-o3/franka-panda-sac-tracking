import numpy as np
from pathlib import Path


eval_file = Path("logs/eval/evaluations.npz")

if not eval_file.exists():
    raise FileNotFoundError(
        f"Could not find {eval_file}. "
        "Check that EvalCallback used log_path='./logs/eval/'."
    )

data = np.load(eval_file)

timesteps = data["timesteps"]
results = data["results"]       # shape: [num_evals, n_eval_episodes]
ep_lengths = data["ep_lengths"]

mean_rewards = results.mean(axis=1)
std_rewards = results.std(axis=1)
mean_lengths = ep_lengths.mean(axis=1)

best_idx = np.argmax(mean_rewards)

print("Evaluation summary")
print("------------------")
print(f"Number of evaluations: {len(timesteps)}")
print(f"Best evaluation index: {best_idx}")
print(f"Best timestep: {timesteps[best_idx]}")
print(f"Best mean reward: {mean_rewards[best_idx]:.3f}")
print(f"Reward std at best: {std_rewards[best_idx]:.3f}")
print(f"Mean episode length at best: {mean_lengths[best_idx]:.1f}")

print("\nLast 10 evaluations:")
for t, mean_r, std_r in zip(timesteps[-10:], mean_rewards[-10:], std_rewards[-10:]):
    marker = "<-- best" if t == timesteps[best_idx] else ""
    print(f"timestep={t:>8} | mean_reward={mean_r:>9.3f} | std={std_r:>8.3f} {marker}")