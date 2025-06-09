# ppo_train.py

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from ai import DoppelkopfEnv
from agents import ExpectiMaxAgent
import constants

def make_env(rank: int, expectimax_prob: float):
    def _init():
        env = DoppelkopfEnv("ALICE", expectimax_prob=expectimax_prob)
        env.agent = ExpectiMaxAgent("ALICE")
        return env
    return _init

class EvalCallback(BaseCallback):
    def __init__(self, eval_env_fns, eval_freq: int, n_eval_episodes: int, verbose=1):
        super().__init__(verbose)
        # build pure-EM vecenv
        self.eval_env = SubprocVecEnv(eval_env_fns)
        self.eval_env = VecMonitor(self.eval_env)
        self.eval_freq = eval_freq
        self.n_eval   = n_eval_episodes
        self.steps    = 0

    def _on_step(self) -> bool:
        self.steps += 1
        if self.steps >= self.eval_freq:
            self.steps = 0
            wins, total = 0, 0
            for _ in range(self.n_eval):
                obs, _ = self.eval_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, rewards, dones, _, _ = self.eval_env.step(action)
                    done = dones[0]
                if rewards[0] > 0:
                    wins += 1
                total += 1

            win_rate = wins / total * 100.0
            print(f"==== Eval @ {self.num_timesteps} steps: Win‐rate = {win_rate:.1f}% over {total} hands")
        return True

    def _on_training_end(self) -> None:
        self.eval_env.close()

def main():
    num_cpu        = 8
    total_timesteps= 1_000_000
    expectimax_prob= 0.9  # 90% EM / 10% random

    # 1) training env
    train_env_fns = [make_env(i, expectimax_prob) for i in range(num_cpu)]
    train_env     = SubprocVecEnv(train_env_fns)
    train_env     = VecMonitor(train_env)

    # 2) PPO instantiation
    model = PPO(
        "MlpPolicy",
        train_env,
        gamma=1.0,
        n_steps=128,
        batch_size=256,
        learning_rate=3e-4,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/",
        policy_kwargs=dict(net_arch=[256, 256]),
        device="auto",
    )

    # 3) evaluation callback (pure EM)
    eval_env_fns = [make_env(i, 1.0) for i in range(4)]
    eval_cb      = EvalCallback(eval_env_fns, eval_freq=100_000, n_eval_episodes=50)

    # 4) train
    model.learn(total_timesteps=total_timesteps, callback=eval_cb)

    # 5) save
    model.save("doppelkopf_ppo_1M_shaped")
    train_env.close()
    print("Training complete; saved as doppelkopf_ppo_1M_shaped.zip")

if __name__ == "__main__":
    main()
