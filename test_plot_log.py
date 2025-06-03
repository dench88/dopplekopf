import numpy as np
from stable_baselines3 import DQN

from ai import DoppelkopfEnv, TYPE_TO_IDX, CARD_TYPES
from agents import ExpectiMaxAgent, RandomAgent
from game_state import GameState
import constants

# --- Configuration ---
RL_SEAT         = "ALICE"
NUM_EPISODES    = 300
# EXPECTIMAX_PCT  = 0.8   # 80% of opponents will be ExpectiMax; 20% Random
EXPECTIMAX_PCT  = 1.0
MODEL_PATH      = "doppelkopf_dqn_phase4_mixed.zip"  # points to doppelkopf_dqn.zip

# --- Load the trained model ---
model: DQN = DQN.load(MODEL_PATH)

def evaluate(num_episodes=NUM_EPISODES):
    rewards = []
    win_count = 0

    for ep in range(num_episodes):
        # 1) Create a fresh env for this episode, with a new mix of opponents
        env = DoppelkopfEnv(RL_SEAT, expectimax_prob=EXPECTIMAX_PCT)

        # 2) Attach a "rule-based" copy of ExpectiMax to env.agent so _encode() can fill team flags
        env.agent = ExpectiMaxAgent(RL_SEAT)

        # 3) Reset to get initial obs
        obs, _ = env.reset()

        # 4) Play out one full hand, always sampling RL actions deterministically
        while True:
            current = env.state.next_player

            if current == RL_SEAT:
                # RL chooses its action
                action_idx, _ = model.predict(obs, deterministic=True)
                chosen_card = None
                # find a matching legal card
                for c in env.state.legal_actions():
                    if TYPE_TO_IDX[(c.type, c.suit)] == action_idx:
                        chosen_card = c
                        break
                # fallback if the chosen card wasn’t legal (should be rare if policy is sane)
                if chosen_card is None:
                    chosen_card = np.random.choice(env.state.legal_actions())

            else:
                # Opponent’s move: pick ExpectiMax or Random based on env.opponent_agents dict
                agent = env.opponent_agents[current]
                chosen_card = agent.choose(env.state)

            # Convert chosen_card → action_int, then step
            action_int = TYPE_TO_IDX[(chosen_card.type, chosen_card.suit)]
            next_obs, reward, terminated, truncated, _ = env.step(action_int)

            obs = next_obs
            if terminated or truncated:
                break

        # 5) At episode end, record the reward
        #    (our _compute_reward already returned team-based differential at terminal)
        #    reward is what was returned by the final step()
        rewards.append(reward)

        # 6) Count a “win” if reward > 0 (your team outscored opponents)
        if reward > 0:
            win_count += 1

        # 7) Clean up
        env.close()

    avg_reward = sum(rewards) / len(rewards)
    win_rate = win_count / len(rewards)
    return avg_reward, win_rate

if __name__ == "__main__":
    avg_r, win_rt = evaluate()
    print(f"\nModel used: {MODEL_PATH}")
    print(f"Over {NUM_EPISODES} hands vs. mixed opponents:")
    # print(f"\nOver {NUM_EPISODES} hands vs. EM opponents:")
    print(f"  → Average (team‐pt diff)  = {avg_r:.2f}")
    print(f"  → Win rate (rewards>0)   = {win_rt*100:.1f}%\n")
