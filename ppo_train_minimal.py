import os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from ai import DoppelkopfEnv
from agents import ExpectiMaxAgent

MODELS_DIR = Path("models")
SAVE_NAME = "dk_ppo_minimal_vec"
TOTAL_TIMESTEPS = 100_000
LOAD_PATH = None  # e.g. MODELS_DIR / "dk_ppo_minimal_vec_1M.zip"

def make_env(expectimax_prob: float = 0.7, seed: int | None = None):
    """
    creates ONE DoppelkopfEnv instance.
    expectimax_prob: (if your env supports it) probability of using the
                     ExpectiMax partner vs random or something else.
    seed:            base RNG seed for this worker.
    """
    def _init():
        env = DoppelkopfEnv(
            "ALICE",
            expectimax_prob=expectimax_prob  # <- actually use it if supported
        )
        # Make sure env knows about the partner agent
        env.agent = ExpectiMaxAgent("ALICE")

        if seed is not None:
            env.reset(seed=seed)

        return env

    return _init


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Use all but one CPU, but at least 1
    num_cpu = max(1, (os.cpu_count() or 1) - 1)
    expectimax_prob = 0.7

    # Different seed per worker
    env_fns = [
        make_env(expectimax_prob, seed=1000 + i)
        for i in range(num_cpu)
    ]

    train_env = SubprocVecEnv(env_fns)
    train_env = VecMonitor(train_env)

    if LOAD_PATH:
        model = PPO.load(LOAD_PATH, env=train_env)
        print(f"Loaded checkpoint {LOAD_PATH}, continuing training…")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            gamma=1.0,          # full 12-trick episode, no discount
            n_steps=128,
            batch_size=256,
            learning_rate=3e-4,
            ent_coef=0.01,
            verbose=1,
            device="auto",
            # tensorboard_log="./ppo_tensorboard/",
        )

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
    except KeyboardInterrupt:
        print("Training interrupted, saving current model…")

    save_path = MODELS_DIR / f"{SAVE_NAME}.zip"
    model.save(save_path)
    print(f"Model saved to {save_path}")

    train_env.close()


if __name__ == "__main__":
    main()
