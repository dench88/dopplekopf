# test_env.py
import random
# from ai import DoppelkopfEnv
# test_env.py
# import random
from ai import DoppelkopfEnv, TYPE_TO_IDX, CARD_TYPES


def run_random_episodes(n_episodes=5, max_steps=50):
    env = DoppelkopfEnv("ALICE")
    for ep in range(n_episodes):
        obs = env.reset()
        print(f"\n=== EPISODE {ep+1} ===")
        for t in range(max_steps):
            # pick a random legal action
            legal = env.state.legal_actions()
            # map from legal cards to their idx in action_space
            legal_idxs = [TYPE_TO_IDX[(c.type, c.suit)] for c in legal]
            action = random.choice(legal_idxs)

            obs, reward, done, info = env.step(action)
            print(f"Step {t+1}: action={action}, reward={reward}, done={done}")
            # print("  Hand counts:", obs[env.hand_offset : env.hand_offset + len(legal_idxs)])
            hand_size = len(TYPE_TO_IDX)  # ==24
            print("  Hand counts:", obs[env.hand_offset : env.hand_offset + hand_size])
            for i, (r, s) in enumerate(CARD_TYPES):
                print(f"{i:2d}: {r}-{s} → count {obs[env.hand_offset + i]}")

            print("  Suit lengths:", obs[env.suit_offset : env.suit_offset + 3])
            print("  Team flags:", obs[env.team_flag_offset : env.team_flag_offset+1], 
                  obs[env.partner_offset : env.partner_offset + (len(env.opponent_agents))])
            print("  GameState:", env.state, "\n")

            if done:
                break

if __name__ == "__main__":
    run_random_episodes()
