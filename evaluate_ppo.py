
# evaluate_ppo.py  (parallel version)

import random
import numpy as np
from multiprocessing import Pool, cpu_count

from stable_baselines3 import PPO
from ai import DoppelkopfEnv, TYPE_TO_IDX
from agents import ExpectiMaxAgent, RandomAgent
import constants

RL_SEAT      = "ALICE"
NUM_EPISODES = 100
EXPECTIMAX_PCT = 1.0
MODEL_PATH   = "ppo_phase1.zip"

model: PPO = PPO.load(MODEL_PATH)

def run_one_episode(_):
    # 1) fresh env
    env = DoppelkopfEnv(RL_SEAT, expectimax_prob=EXPECTIMAX_PCT)
    env.agent = ExpectiMaxAgent(RL_SEAT, verbose=False)
    obs, _ = env.reset()

    # 2) play the full hand
    while True:
        current = env.state.next_player
        if current == RL_SEAT:
            idx, _ = model.predict(obs, deterministic=True)
            # map to a legal Card
            chosen = None
            for c in env.state.legal_actions():
                if TYPE_TO_IDX[(c.type, c.suit)] == int(idx):
                    chosen = c; break
            if chosen is None:
                chosen = random.choice(env.state.legal_actions())
        else:
            chosen = env.opponent_agents[current].choose(env.state)

        obs, reward, done, truncated, _ = env.step(TYPE_TO_IDX[(chosen.type, chosen.suit)])
        if done or truncated:
            break

    # 3) return final reward and win-flag
    win = 1 if reward > 0 else 0
    env.close()
    return reward, win

def evaluate_parallel(n_episodes=NUM_EPISODES):
    procs = max(1, cpu_count() - 1)
    with Pool(processes=procs) as pool:
        results = pool.map(run_one_episode, range(n_episodes))

    rewards, wins = zip(*results)
    avg_reward = sum(rewards) / n_episodes
    win_rate   = sum(wins) / n_episodes
    return avg_reward, win_rate

if __name__ == "__main__":
    avg_r, win_rt = evaluate_parallel()
    print(f"\nModel used: {MODEL_PATH}")
    print(f"Over {NUM_EPISODES} hands vs. {EXPECTIMAX_PCT*100:.0f}% EM:")
    print(f"  → Average (team‐pt diff) = {avg_r:.2f}")
    print(f"  → Win rate (rewards>0)  = {win_rt*100:.1f}%\n")



