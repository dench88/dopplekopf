# ppo_train.py

import os
import argparse
from functools import partial

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from ai import DoppelkopfEnv
from agents import ExpectiMaxAgent, RandomAgent, HeuristicRandomAgent
import constants

# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, choices=[1,2], required=True,
                   help="1: from scratch vs weak; 2: warm-start vs mixed")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
def make_env(rank: int, mix: dict[str,float]):
    """
    Returns a thunk that creates one DoppelkopfEnv instance.
    mix maps agent‐type→fraction, e.g. {"random":0.5, "heur":0.5, "em":0.0}.
    """
    types, freqs = zip(*mix.items())
    cum = np.cumsum(freqs)

    def _init():
        env = DoppelkopfEnv("ALICE", expectimax_prob=0.0)
        env.agent = ExpectiMaxAgent("ALICE")  # needed for team flags

        # assign opponent agents
        env.opponent_agents.clear()
        for p in constants.players:
            if p == "ALICE":
                continue
            r = np.random.rand()
            if r < cum[0] and types[0]=="random":
                env.opponent_agents[p] = RandomAgent()
            elif r < cum[0] and types[0]=="heur":
                env.opponent_agents[p] = HeuristicRandomAgent(p)
            else:
                # find which bucket
                for t, c in zip(types, cum):
                    if r < c:
                        if t=="random":
                            env.opponent_agents[p] = RandomAgent()
                        elif t=="heur":
                            env.opponent_agents[p] = HeuristicRandomAgent(p)
                        else:
                            env.opponent_agents[p] = ExpectiMaxAgent(p)
                        break
        return env

    return _init

# ─────────────────────────────────────────────────────────────────────────────
class EvalCallback(BaseCallback):
    def __init__(self, eval_env_fns, eval_freq: int, n_eval_episodes: int, verbose=1):
        super().__init__(verbose)
        self.eval_env = SubprocVecEnv(eval_env_fns)
        self.eval_env = VecMonitor(self.eval_env)
        self.eval_freq = eval_freq
        self.n_eval   = n_eval_episodes
        self.step_cnt = 0

    def _on_step(self) -> bool:
        self.step_cnt += 1
        if self.step_cnt >= self.eval_freq:
            self.step_cnt = 0
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
            self.logger.record("eval/win_rate", win_rate)
            print(f"==== Eval @ {self.num_timesteps} steps: Win‐rate = {win_rate:.1f}% over {total} hands")
        return True

    def _on_training_end(self) -> None:
        self.eval_env.close()

# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Curriculum settings
    if args.phase == 1:
        # Phase 1: scratch, 100k, 50% random / 50% heuristic
        load_path    = None
        save_path    = "ppo_phase1"
        timesteps    = 100_000
        opponent_mix = {"random":0.5, "heur":0.5, "em":0.0}
    else:
        # Phase 2: warm start from Phase 1, 200k, 20% random / 40% heur / 40% EM
        load_path    = "ppo_phase1.zip"
        save_path    = "ppo_phase2"
        timesteps    = 200_000
        opponent_mix = {"random":0.2, "heur":0.4, "em":0.4}

    num_cpu = max(1, os.cpu_count() - 1)

    # 1) Training env
    train_env_fns = [make_env(i, opponent_mix) for i in range(num_cpu)]
    train_env     = SubprocVecEnv(train_env_fns)
    train_env     = VecMonitor(train_env)

    # 2) Instantiate or load model
    if load_path:
        model = PPO.load(load_path, env=train_env)
        print(f"Loaded checkpoint {load_path}, continuing training…")
    else:
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
            policy_kwargs=dict(net_arch=[256,256]),
            device="auto",
        )
        print("Starting training from scratch…")

    # 3) Eval callback (100% EM)
    eval_env_fns = [make_env(i, {"random":0.0,"heur":0.0,"em":1.0}) for i in range(4)]
    eval_cb      = EvalCallback(eval_env_fns, eval_freq=100_000, n_eval_episodes=50)

    # 4) Optional checkpointing every 100k timesteps
    ckpt_cb = CheckpointCallback(
        save_freq = 100_000 // num_cpu,
        save_path = "./checkpoints/",
        name_prefix = f"ppo_phase{args.phase}"
    )



    # 5) Train & save partial only on exception
    completed = False
    try:
        model.learn(total_timesteps=timesteps, callback=[eval_cb, ckpt_cb])
        completed = True
    except Exception as e:
        # Training crashed—save a partial checkpoint
        model.save(f"{save_path}_partial")
        print(f"Training aborted early, saved partial model: {save_path}_partial.zip")
        # Re‐raise so you still see the error
        raise
    # 6) If we got here without exception, do the normal final save
    if completed:
        model.save(save_path)
        print(f"Training complete; saved final model: {save_path}.zip")

    train_env.close()




if __name__ == "__main__":
    main()
