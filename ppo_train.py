# ppo_train.py

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from ai import DoppelkopfEnv, TYPE_TO_IDX
from agents import ExpectiMaxAgent
import constants

# ————————————————
# 1) “make_env” factory for SubprocVecEnv
# ————————————————
def make_env(rank: int, expectimax_prob: float):
    """
    Returns a thunk that creates one DoppelkopfEnv instance.
    Each env will control "ALICE" vs opponents that
    are ExpectiMax with probability expectimax_prob,
    else RandomAgent (handled internally by DoppelkopfEnv).
    We also attach a rule‐based ExpectiMaxAgent as env.agent
    so that reward shaping and obs‐encoding see correct team flags.
    """
    def _init():
        env = DoppelkopfEnv("ALICE", expectimax_prob=expectimax_prob)
        # Tell the env’s internal logic which agent to use for team flags.
        env.agent = ExpectiMaxAgent("ALICE")
        return env
    return _init

# ————————————————
# 2) Callback to evaluate win‐rate periodically
# ————————————————
class EvalCallback(BaseCallback):
    """
    After every `eval_freq` timesteps, run N eval episodes
    (with the current policy against pure EM) and print win-rate.
    """
    def __init__(self, eval_env_fns, eval_freq: int, n_eval_episodes: int, verbose=1):
        super().__init__(verbose)
        # We’ll build a separate VecEnv for evaluation, using pure‐EM opponents:
        self.eval_env = SubprocVecEnv(eval_env_fns)
        self.eval_env = VecMonitor(self.eval_env)  # to keep episode lengths
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.steps_since_last = 0

    def _on_step(self) -> bool:
        self.steps_since_last += 1
        if self.steps_since_last >= self.eval_freq:
            self.steps_since_last = 0
            wins = 0
            total = 0

            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                while not done:
                    # PPO’s predict requires a single-env observation; we have a VecEnv
                    # so take obs[0] (first env in the vector), but we still call predict on the full batch:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, rewards, dones, infos = self.eval_env.step(action)
                    # If any env signals done, we break. Using VecEnv, we just look at dones[0].
                    done = dones[0]

                # At terminal, rewards[0] is the final team‐point diff for env#0
                if rewards[0] > 0:
                    wins += 1
                total += 1

            win_rate = wins / total * 100.0
            print(f"==== Eval @ {self.num_timesteps} steps: Win‐rate = {win_rate:.1f}% over {total} hands")
        return True

    def _on_training_end(self) -> None:
        self.eval_env.close()

# ————————————————
# 3) Main training routine
# ————————————————
def main():
    num_cpu = 8                  # number of parallel environments
    total_timesteps = 1_000_000  # adjust as desired

    # 3.1) Build the vectorized “training” env: 8 copies with 80% EM, 20% Random
    expectimax_prob = 0.9  # 90% ExpectiMax, 10% RandomAgent
    # This will be used to control the opponents in the training envs.
    train_env_fns = [make_env(i, expectimax_prob) for i in range(num_cpu)]
    # Create a vectorized environment with 8 parallel envs
    # using SubprocVecEnv for better performance with multiple CPU cores.
    train_env = SubprocVecEnv(train_env_fns)
    # Wrap it with VecMonitor to automatically record episode lengths and returns
    # (VecMonitor will also handle the logging of rewards and lengths).
    train_env = VecMonitor(train_env)  # auto‐record lengths & returns

    # 3.2) Build the PPO model
    #     We keep gamma=1.0 (no discount), and use a moderate learning rate
    model = PPO(
        "MlpPolicy",
        train_env,
        gamma=1.0,
        n_steps=128,          # each env will run 128 steps before each update
        batch_size=256,       # batch from all envs (128*8 = 1024 per update, then subsample to 256)
        learning_rate=3e-4,
        ent_coef=0.01,        # small entropy bonus to maintain exploration
        verbose=1,
        tensorboard_log="./ppo_tensorboard/",
        policy_kwargs=dict(net_arch=[256, 256]),
        device="auto",        # use GPU if available
    )

    # 3.3) Build a small “evaluation” vectorized env (4 parallel) for pure-EM testing
    eval_env_fns = [make_env(i, 1.0) for i in range(4)]  # 4 envs all with expectimax_prob=1.0
    eval_cb = EvalCallback(eval_env_fns, eval_freq=100_000, n_eval_episodes=50)

    # 3.4) Train for 1 000 000 timesteps, printing PPO logs and EvalCallback results
    model.learn(total_timesteps=total_timesteps, callback=eval_cb)

    # 3.5) Save the final policy
    model.save("doppelkopf_ppo_1M_shaped")
    train_env.close()
    print("Training complete; saved as doppelkopf_ppo_1M_shaped.zip")

if __name__ == "__main__":
    main()
