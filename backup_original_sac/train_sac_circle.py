import os

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from panda_circle_sac_env import PandaCircleSACEnv


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # First, sanity-check the environment.
    check_env(PandaCircleSACEnv(render_mode=None), warn=True)

    train_env = Monitor(PandaCircleSACEnv(render_mode=None))
    eval_env = Monitor(PandaCircleSACEnv(render_mode=None))

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        tensorboard_log="./logs/sac_tensorboard/",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path="./models/",
        name_prefix="sac_panda_circle_checkpoint",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model/",
        log_path="./logs/eval/",
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    # Start with 50_000 for a smoke test.
    # Increase to 300_000 or more once the script works.
    model.learn(
        total_timesteps=1_000_000,
        callback=[checkpoint_callback, eval_callback],
    )

    model.save("./models/sac_panda_circle_final")

    train_env.close()
    eval_env.close()

    print("Training finished.")
    print("Saved model to: ./models/sac_panda_circle_final.zip")


if __name__ == "__main__":
    main()