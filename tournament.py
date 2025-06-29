
from main import make_initial_state, play_game, RLWrapper
from input_utils import get_qc_split_and_points
from agents import ExpectiMaxAgent, HeuristicRandomAgent, RandomAgent
from ai import DoppelkopfEnv
# from ai_old_version_fixed import DoppelkopfEnv
from stable_baselines3 import PPO
import constants
import sys
from datetime import datetime

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = f"tournament_{ts}.log"

sys.stdout = open(logfile, "w", encoding="utf-8")
sys.stderr = sys.stdout  # funnel errors into the same file

print(f"Logging to {logfile!r}…")

# Set up agents as you wish
dummy_env = DoppelkopfEnv("RUSTY", expectimax_prob=1.0)
# ppo_model = PPO.load("ppo_phase_17D.zip")
# ppo_model = PPO.load("models/dk_ppo_VEC_61a_minimal_03R.zip")
ppo_model = PPO.load("models/dk_ppo_VEC_61a_minimal_04R.zip")

agents = {
    # "ALICE": RLWrapper(ppo_model, "ALICE", dummy_env),
    "RUSTY": RLWrapper(ppo_model, "RUSTY", dummy_env),
    # "HARLEM": RLWrapper(ppo_model, "HARLEM", dummy_env),
    # "ALICE": RLWrapper(ppo_model, "ALICE", dummy_env),
    # "RUSTY": ExpectiMaxAgent("RUSTY"),
    # "HARLEM": HeuristicRandomAgent("HARLEM"),
    "SUSIE": ExpectiMaxAgent("SUSIE"),
    # "HARLEM": ExpectiMaxAgent("HARLEM"),
    # "ALICE": ExpectiMaxAgent("ALICE")
    "HARLEM": RandomAgent(),
    "ALICE": RandomAgent()
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