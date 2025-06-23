# evaluate_ppo_vs_ppo.py

import random
import numpy as np
from multiprocessing import Pool, cpu_count

from stable_baselines3 import PPO
from ai import DoppelkopfEnv, TYPE_TO_IDX
from game_state import GameState
import constants

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
NUM_EPISODES    = 10
# model filepaths:
MODEL_B_PATH    = "ppo_phase1.zip"
MODEL_A_PATH    = "ppo_phase_21D.zip"
# which seats does each control?
MODEL_A_SEATS   = ["RUSTY"]
MODEL_B_SEATS   = ["HARLEM", "ALICE", "SUSIE"]

# ─────────────────────────────────────────────────────────────────────────────
# Load both policies
# ─────────────────────────────────────────────────────────────────────────────
model_a = PPO.load(MODEL_A_PATH)
model_b = PPO.load(MODEL_B_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Helper to run one hand with A controlling MODEL_A_SEATS, B controlling MODEL_B_SEATS
# ─────────────────────────────────────────────────────────────────────────────
def run_one_episode(_):
    # create a fresh env each episode
    env = DoppelkopfEnv("ALICE", expectimax_prob=0.0)
    # override env.agent so team‐flags still work
    env.agent = None

    # assign each seat to the correct model
    agents = {}
    for seat in constants.players:
        if seat in MODEL_A_SEATS:
            agents[seat] = ("A", model_a)
        else:
            agents[seat] = ("B", model_b)

    obs, _ = env.reset()

    while True:
        current = env.state.next_player
        side, mdl = agents[current]

        # build obs for this seat
        batch = obs[None]

        idx, _ = mdl.predict(batch, deterministic=True)
        idx = int(idx.item())

        # map to a legal card
        chosen = None
        for c in env.state.legal_actions():
            if TYPE_TO_IDX[(c.type, c.suit)] == idx:
                chosen = c
                break
        if chosen is None:
            chosen = random.choice(env.state.legal_actions())

        obs, reward, done, truncated, _ = env.step(TYPE_TO_IDX[(chosen.type, chosen.suit)])
        if done or truncated:
            break

    # Determine which two players ended up on the same team (the two Q-club holders)
    final = env.state
    holders = {
        p for trick in final.trick_history
        for p, c in trick
        if c.identifier == "Q-clubs"
    }
    # compute total points for holders vs others
    pts_hldr = sum(final.points[p] for p in holders)
    pts_othr = sum(v for p, v in final.points.items() if p not in holders)
    diff = pts_hldr - pts_othr

    # Was model A controlling both holders?
    if holders.issubset(set(MODEL_A_SEATS)):
        reward_A = diff
    elif holders.issubset(set(MODEL_B_SEATS)):
        reward_A = -diff
    else:
        # Mixed team: one seat each → compare each full team total instead
        pts_A = sum(final.points[p] for p in MODEL_A_SEATS)
        pts_B = sum(final.points[p] for p in MODEL_B_SEATS)
        reward_A = pts_A - pts_B

    win_A = 1 if reward_A > 0 else 0

    env.close()
    return reward_A, win_A


# ─────────────────────────────────────────────────────────────────────────────
# Parallel evaluation harness
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_parallel(n_episodes=NUM_EPISODES):
    procs = max(1, cpu_count() - 1)
    with Pool(processes=procs) as pool:
        results = pool.map(run_one_episode, range(n_episodes))

    rewards, wins = zip(*results)
    avg_reward = sum(rewards) / n_episodes
    win_rate   = sum(wins) / n_episodes
    return avg_reward, win_rate


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    avg_r, win_rt = evaluate_parallel()
    print(f"\nModel A seats: {MODEL_A_SEATS}, path: {MODEL_A_PATH}")
    print(f"Model B seats: {MODEL_B_SEATS}, path: {MODEL_B_PATH}")
    print(f"Over {NUM_EPISODES} hands (2v2):")
    print(f"  → Average (A_pts – B_pts) = {avg_r:.2f}")
    print(f"  → Model A win rate       = {win_rt*100:.1f}%\n")
