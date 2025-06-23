
# evaluate_ppo.py  (parallel version)

import random
from multiprocessing import Pool, cpu_count
from stable_baselines3 import PPO
from ai import DoppelkopfEnv, TYPE_TO_IDX
from agents import ExpectiMaxAgent
import constants

RL_SEAT = "ALICE"
NUM_EPISODES = 200
EXPECTIMAX_PCT = 1.0
MODEL_PATH = "ppo_phase_17D.zip"
# MODEL_PATH = "doppelkopf_ppo_1M_shaped_A.zip"
model: PPO = PPO.load(MODEL_PATH)

def run_one_episode(_):
    env = DoppelkopfEnv(RL_SEAT, expectimax_prob=EXPECTIMAX_PCT)
    env.agent = ExpectiMaxAgent(RL_SEAT, verbose=False)
    obs, _ = env.reset()

    while True:
        current = env.state.next_player
        if current == RL_SEAT:
            idx, _ = model.predict(obs, deterministic=True)
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

    # --- DEBUG PRINTS BEGIN HERE ---
    final = env.state
    print("Final points:")
    for p in constants.players:
        print(f"  {p}: {final.points[p]}")

    # Q-club holders ("real team")—search both hands and trick history
    qclub_holders = set()
    for p, hand in final.hands.items():
        for c in hand:
            if c.identifier == "Q-clubs":
                qclub_holders.add(p)
    for trick in final.trick_history:
        for p, c in trick:
            if c.identifier == "Q-clubs":
                qclub_holders.add(p)
    print("Q-club holders (teammates):", sorted(qclub_holders))

    # Show what agent is in each seat
    for p in constants.players:
        if p == RL_SEAT:
            print(f"  {p}: PPO Model")
        else:
            print(f"  {p}: {type(env.opponent_agents[p]).__name__}")

    # Who was my team at the end?
    env.agent.update_team_info(final, force=True)
    my_team = env.agent.get_team_members(final)
    print("My team according to agent:", sorted(my_team))

    opp_team = [p for p in final.points if p not in my_team]
    my_team_pts = sum(final.points[p] for p in my_team)
    opp_team_pts = sum(final.points[p] for p in opp_team)
    print(f"Team points: {my_team_pts}, Opponent points: {opp_team_pts}")

    if RL_SEAT in my_team:
        print("→ PPO Model's team won!" if my_team_pts > opp_team_pts else "→ PPO Model's team lost.")
    else:
        print("→ PPO Model was not on the team.")

    print("-" * 50)
    # --- END DEBUG PRINTS ---

    win = 1 if my_team_pts > opp_team_pts else 0
    env.close()
    return (my_team_pts - opp_team_pts), win



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
    print(f"  → Win rate (team pts - opponent team points)  = {win_rt*100:.1f}%\n")



