# evaluate_ppo.py

import numpy as np
from stable_baselines3 import PPO

from ai import DoppelkopfEnv, TYPE_TO_IDX
from agents import ExpectiMaxAgent, RandomAgent
import constants

# ————————————————
# Configuration
# ————————————————
RL_SEAT         = "ALICE"                    # Seat controlled by our trained agent
NUM_EPISODES    = 30                        # Number of hands to simulate
EXPECTIMAX_PCT  = 1.0                       # Probability each opponent is ExpectiMax (else Random)
MODEL_PATH      = "doppelkopf_ppo_1M_shaped.zip"  # Path to your PPO checkpoint

# ————————————————
# Load the trained model
# ————————————————
model: PPO = PPO.load(MODEL_PATH)


def evaluate(num_episodes=NUM_EPISODES):
    """
    Plays out `num_episodes` full hands.  Each hand:
      • Creates a fresh DoppelkopfEnv with the specified mix of ExpectiMax vs Random opponents.
      • Attaches a rule-based ExpectiMaxAgent to env.agent (so the obs‐encoding and reward‐shaping see correct team‐flags).
      • Resets the env, then loops until done:
         – If it's RL_SEAT's turn → model.predict(obs) to pick a card.
         – Otherwise → ask env.opponent_agents[current] for a card.
         – Step the env, gather obs/reward/done.
      • At the end of the hand, record the final reward (team‐point differential).
      • Count a “win” if final reward > 0 (ALICE’s team outscored opponents).
    Returns (avg_reward, win_rate).
    """
    rand_agent = RandomAgent()

    rewards = []
    win_count = 0

    for ep in range(num_episodes):
        # 1) Create a fresh env with a new random draw & opponent assignment
        env = DoppelkopfEnv(RL_SEAT, expectimax_prob=EXPECTIMAX_PCT)

        # 2) Attach a rule‐based ExpectiMaxAgent so _encode/_compute_reward can see team flags
        env.agent = ExpectiMaxAgent(RL_SEAT, verbose=False)

        # 3) Reset to get initial observation (and deal a new hand)
        obs, _ = env.reset()

        # 4) Play out one full hand
        while True:
            current = env.state.next_player

            if current == RL_SEAT:
                # → RL uses PPO's policy to select an action (deterministic)
                action_idx, _ = model.predict(obs, deterministic=True)

                # Find the actual Card in ALICE’s hand whose index matches action_idx
                chosen_card = None
                for c in env.state.legal_actions():
                    if TYPE_TO_IDX[(c.type, c.suit)] == action_idx:
                        chosen_card = c
                        break

                # If the network picked an illegal action (rare), fallback to uniform random
                if chosen_card is None:
                    chosen_card = np.random.choice(env.state.legal_actions())

            else:
                # → Opponent’s turn: use ExpectiMaxAgent or RandomAgent
                opp_agent = env.opponent_agents[current]
                chosen_card = opp_agent.choose(env.state)

            # Convert chosen_card → its integer index, then step the env
            action_int = TYPE_TO_IDX[(chosen_card.type, chosen_card.suit)]
            next_obs, reward, terminated, truncated, _ = env.step(action_int)

            obs = next_obs
            if terminated or truncated:
                break

        # 5) At hand’s end, record the final reward (team‐pt diff)
        #    — _compute_reward returns the team differential at the terminal step
        rewards.append(reward)
        if reward > 0:
            win_count += 1

        # 6) Clean up this env before the next episode
        env.close()

    avg_reward = sum(rewards) / len(rewards)
    win_rate = win_count / len(rewards)
    return avg_reward, win_rate


if __name__ == "__main__":
    avg_r, win_rt = evaluate()
    print(f"\nModel used: {MODEL_PATH}")
    print(f"Over {NUM_EPISODES} hands vs. mixed opponents ({EXPECTIMAX_PCT*100:.0f}% EM / {100-EXPECTIMAX_PCT*100:.0f}% Random):")
    print(f"  → Average (team‐pt diff)  = {avg_r:.2f}")
    print(f"  → Win rate (rewards>0)   = {win_rt*100:.1f}%\n")
