
# evaluate_random.py

import numpy as np

from ai import DoppelkopfEnv, TYPE_TO_IDX
from agents import ExpectiMaxAgent, RandomAgent
from game_state import GameState
import constants

# ————————— Configuration —————————
RL_SEAT      = "ALICE"         # we’ll put a RandomAgent here
NUM_EPISODES = 100             # number of hands to simulate
MODEL_PATH   = None            # not used here
# force all opponents to be ExpectiMax
EXPECTIMAX_PCT = 1.0           # 1.0 means “100% ExpectiMax for each opponent”

# —————— Agents setup ——————
# We’ll create one RandomAgent for ALICE,
# and three ExpectiMaxAgent for the others. 
rand_agent = RandomAgent()

# The other seats (RUSTY, SUSIE, HARLEM) are always ExpectiMax:
opp_seats = [p for p in constants.players if p != RL_SEAT]
expecti_agents = {seat: ExpectiMaxAgent(seat) for seat in opp_seats}


def evaluate_random(num_episodes=NUM_EPISODES):
    """
    Play `num_episodes` hands where ALICE is RandomAgent and
    the other three players are always ExpectiMaxAgent.
    Returns (average_team_point_diff, win_rate).
    """
    rewards = []
    win_count = 0

    for ep in range(num_episodes):
        # 1) Build a fresh env with all opponents = ExpectiMax
        env = DoppelkopfEnv(RL_SEAT, expectimax_prob=EXPECTIMAX_PCT)
        # Attach a “rule-based” copy of ExpectiMax to env.agent
        # so that _encode() can fill in team-flag bits correctly.
        env.agent = ExpectiMaxAgent(RL_SEAT)

        # 2) Reset to get initial obs (the obs won’t matter for RandomAgent)
        obs, _ = env.reset()

        # 3) Play out one full hand
        while True:
            current = env.state.next_player
            legal_cards = env.state.legal_actions()

            if current == RL_SEAT:
                # ALICE’s turn → RandomAgent
                chosen_card = rand_agent.choose(env.state)

            else:
                # One of the three ExpectiMax opponents
                chosen_card = expecti_agents[current].choose(env.state)

            # 4) Convert chosen_card → action_int
            action_int = TYPE_TO_IDX[(chosen_card.type, chosen_card.suit)]
            next_obs, reward, terminated, truncated, _ = env.step(action_int)

            obs = next_obs
            if terminated or truncated:
                break

        # 5) At hand’s end, record the final reward (team‐pt diff)
        #    (env.step(...)’s return `reward` is already the final team diff.)
        rewards.append(reward)
        if reward > 0:
            win_count += 1

        env.close()

    avg_reward = sum(rewards) / len(rewards)
    win_rate = win_count / len(rewards)
    return avg_reward, win_rate


if __name__ == "__main__":
    avg_r, win_rt = evaluate_random()
    print(f"\nOver {NUM_EPISODES} hands (Random vs. 3×ExpectiMax):")
    print(f"  → Average (team‐pt diff)  = {avg_r:.2f}")
    print(f"  → Win rate (rewards>0)   = {win_rt*100:.1f}%\n")
