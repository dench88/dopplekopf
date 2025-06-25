import sys
from main import make_initial_state, play_game, RLWrapper
from input_utils import get_qc_split_and_points
from agents import ExpectiMaxAgent, HeuristicRandomAgent, RandomAgent
from ai import DoppelkopfEnv
from stable_baselines3 import PPO
import constants

# Set up agents as you wish
dummy_env = DoppelkopfEnv("ALICE", expectimax_prob=1.0)
# ppo_model = PPO.load("ppo_phase_17D.zip")
ppo_model = PPO.load("doppelkopf_ppo_minimal.zip")

agents = {
    "ALICE": RLWrapper(ppo_model, "ALICE", dummy_env),
    "RUSTY": ExpectiMaxAgent("RUSTY"),
    "HARLEM": HeuristicRandomAgent("HARLEM"),
    "SUSIE": RandomAgent()
}

NUM_GAMES = 100
win_counts = {p: 0 for p in constants.players}

# open a text file for writing in UTF-8
with open("tournament_results.txt", "w", encoding="utf-8") as fout:
    def log(*args, **kwargs):
        # mirror to console
        print(*args, **kwargs)
        # also write to file
        line = " ".join(str(a) for a in args)
        # respect any custom end= in print
        end = kwargs.get("end", "\n")
        fout.write(line + end)

    for game_idx in range(1, NUM_GAMES + 1):
        state = make_initial_state()
        final_state = play_game(state, agents)

        qc_team, non_qc_team, qc_pts, non_qc_pts = get_qc_split_and_points(final_state)

        # write per-game summary
        log(f"\nGame {game_idx}:")
        log(f"  Q-club team      = {qc_team}   -> {qc_pts} points")
        log(f"  non-Q-club team  = {non_qc_team} -> {non_qc_pts} points")

        # decide winners
        if qc_pts > non_qc_pts:
            winners = qc_team
        else:
            winners = non_qc_team

        for p in winners:
            win_counts[p] += 1

    # after all games, print aggregate win-rates
    log("\n=== Tournament results ===")
    for p in constants.players:
        pct = win_counts[p] / NUM_GAMES * 100
        log(f"  {p}: {win_counts[p]} wins ({pct:.1f}%)")