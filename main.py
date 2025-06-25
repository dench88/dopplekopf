import random
from form_deck import create_deck
from game_state import GameState
from input_utils import get_qc_split_and_points
import constants
from agents import HumanAgent, RandomAgent, MinimaxAgent, ExpectiMaxAgent, HeuristicRandomAgent
from determinized_mcts_agent import DeterminizedMCTSAgent
from stable_baselines3 import PPO
from ai import DoppelkopfEnv, TYPE_TO_IDX
import time

# —————————————————————————————————————————————————————
# 0) Load your PPO checkpoint
# —————————————————————————————————————————————————————
# PPO_MODEL_PATH = "doppelkopf_ppo_1M_shaped_A.zip"


# —————————————————————————————————————————————————————
# 1) A tiny wrapper so that we can do model.predict(obs) → Card
# —————————————————————————————————————————————————————
class RLWrapper:
    def __init__(self, model, seat_name, env):
        self.model    = model
        self.seat     = seat_name
        self.env      = env       # store the env directly

    def choose(self, state: GameState):
        # build the obs vector from our stored env
        obs_vec = self.env._encode(state)[None]

        action_arr, _ = self.model.predict(obs_vec, deterministic=True)
        action_idx    = int(action_arr.item())

        for c in state.legal_actions():
            if TYPE_TO_IDX[(c.type, c.suit)] == action_idx:
                return c

        return random.choice(state.legal_actions())


# agents = {
#     # "RUSTY": HumanAgent(),
#     # "SUSIE": RandomAgent(),
#     "RUSTY": ExpectiMaxAgent("RUSTY"),
#     # "HARLEM": DeterminizedMCTSAgent("HARLEM", simulations=1500),
# }

def find_first_player():
    return random.choice(list(constants.players.keys()))


def make_initial_state():
    while True:
        hands = {player: [] for player in constants.players}
        deck = create_deck()
        for player in hands:
            hands[player] = [deck.pop() for _ in range(12)]
        # Check that all player hands have more than 2 trumps
        if all(sum(1 for card in hand if card.category == 'trumps') > 2 for hand in hands.values()):
            break  # Exit loop only if condition is met

    # Convert hands to immutable form (tuples)
    frozen_hands = {player: tuple(hand) for player, hand in hands.items()}
    # Determine who goes first
    first = find_first_player()
    # Create and return the initial game state
    return GameState(hands=frozen_hands, next_player=first)


def render(state, last_trick):
    # Whose turn
    # print(f"Next to play: {state.next_player}")
    # Public Q-clubs: only those who've actually played Q-clubs so far
    qc_public = [
        player
        for trick in list(state.trick_history) + [state.current_trick]
        for player, card in trick
        if card.identifier == 'Q-clubs'
    ]

    print("Team Q-clubs (public):", qc_public)

    # Current trick so far
    if state.current_trick:
        cards = [card.identifier for _, card in state.current_trick]
        print("Current trick:", cards)
        winner, winning_card = max(state.current_trick, key=lambda pc: pc[1].power)
        print(f"{winner} winning with {winning_card.identifier}")
    # Current hand (sorted)
    hand = sorted(state.hands[state.next_player], key=lambda c: c.power)
    print("Current hand:", [c.identifier for c in hand])
    # Playable cards
    playable = sorted(state.legal_actions(), key=lambda c: c.power)
    print("Playable cards:", " ".join(c.identifier for c in playable))



def play_game(state: GameState, agents: dict[str, any], render_func=None):

    last_trick = None
    while not state.is_terminal():
        print(f"TRICK {len(state.trick_history)+1}; Ply {len(state.current_trick)+1} of 4")
        current_player = state.next_player
        print(f"{current_player} to play!")
        agent  = agents[current_player]

        # render if human…
        if isinstance(agent, HumanAgent) and render_func:
            render_func(state, last_trick)

        # ask agent for move
        action = agent.choose(state)

        # show partial trick
        if state.current_trick:
            cards = [c.identifier for _, c in state.current_trick]
            print(f"Current trick: {cards}")

        # show hand & play
        hand = sorted(state.hands[current_player], key=lambda c: c.power)
        print(f"       {current_player} hand: {[c.identifier for c in hand]}")
        if not isinstance(agent, HumanAgent):
            print(f"{current_player} played {action.identifier}")

        # 1) update state with card just played.
        state = state.apply_action(action)
        # if that was the 4th card in the trick, this 'state' now contains an empty current trick

        # 2) now if Q-club is in the new state, force everyone to refresh
        if action.identifier == "Q-clubs":
            for ag in agents.values():
                if hasattr(ag, "update_team_info"):
                    ag.update_team_info(state, force=True)

        # 3) show team status
        if hasattr(agent, "is_team_playing"):
            agent_team_members = getattr(agent, "team_members", None)
            if agent.is_team_playing and agent_team_members:
                print(f"    {current_player} is team playing with {agent_team_members}")
            else:
                print(f"    {current_player} is not team playing")

        # 4) trick-complete reporting
        if not state.current_trick:
            last_trick = state.trick_history[-1]
            cards = [c.identifier for _, c in last_trick]
            winner, _ = max(last_trick, key=lambda pc: pc[1].power)
            pts = sum(c.points for _, c in last_trick)
            print("Trick done:", cards)
            print(f"************{winner} won {pts} points; totals: {state.points}")
        else:
            last_trick = None

        print("=================================")

    return state




if __name__ == "__main__":
    env = DoppelkopfEnv("ALICE", expectimax_prob=1.0)
    PPO_MODEL_PATH = ("ppo_phase4D.zip")
    ppo_model = PPO.load(PPO_MODEL_PATH)
    rl_agent = RLWrapper(ppo_model, "ALICE", env)
    agents = {
        # "RUSTY": HumanAgent(),
        "SUSIE": ExpectiMaxAgent("SUSIE"),
        "RUSTY": ExpectiMaxAgent("RUSTY"),
        "HARLEM": ExpectiMaxAgent("HARLEM"),
        # "ALICE": HeuristicRandomAgent("ALICE"),
        # "RUSTY": rl_agent,
        # "SUSIE": rl_agent,
        # "HARLEM": rl_agent,
        "ALICE": rl_agent
    }
    state = make_initial_state()
    print("Initial hands:")
    for player, hand in state.hands.items():
        sorted_ids = [c.identifier for c in sorted(hand, key=lambda c: c.power)]
        print(f"  {player}: {sorted_ids}")
    print("====================================================")

    start = time.time()
    final_state = play_game(state, agents, render)
    end = time.time()

    print("\nGame over! Final points:", final_state.points)
    print(f"Game runtime: {end - start:.2f} seconds\n")

    # Compute final teams and totals as before:
    qc_team, non_qc_team, qc_pts, non_qc_pts = get_qc_split_and_points(final_state)

    print(f"Team Q-clubs ({', '.join(qc_team)}): Total points: {qc_pts}")
    print(f"Team non-Q-clubs ({', '.join(non_qc_team)}): Total points: {non_qc_pts}")

    print("\nGame summary:")
    for i, trick in enumerate(final_state.trick_history, 1):
        trick_summary = {player: card.identifier for player, card in trick}
        print(f"Trick {i}: {trick_summary}")