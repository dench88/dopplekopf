# ppo_train_minimal.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from ai import DoppelkopfEnv
from agents import ExpectiMaxAgent

load_path = None
# load_path = 'models/dk_ppo_minimal.zip'
def make_env(expectimax_prob: float):
    def _init():
        env = DoppelkopfEnv("ALICE")
        # Attach a rule‐based agent so obs/_compute_reward see correct team flags
        env.agent = ExpectiMaxAgent("ALICE")
        return env
    return _init

if __name__ == "__main__":
    num_cpu = max(1, os.cpu_count() - 1)
    expectimax_prob = 0.7
    train_env_fns = [make_env(expectimax_prob) for _ in range(num_cpu)]
    train_env = SubprocVecEnv(train_env_fns)
    train_env = VecMonitor(train_env)

    if load_path:
        model = PPO.load(load_path, env=train_env)
        print(f"Loaded checkpoint {load_path}, continuing training…")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            gamma=1.0,          # no discounting across the 12‐trick episode
            n_steps=128,
            batch_size=256,
            learning_rate=3e-4,
            ent_coef=0.01,
            verbose=1,          # prints training progress
            device="auto"
            # tensorboard_log="./ppo_tensorboard/",  # optional
        )

    model.learn(total_timesteps=100_000)
    model.save("models/dk_ppo_VEC_61a_minimal_04R")
    train_env.close()


