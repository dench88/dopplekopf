import random
from form_deck import create_deck
from game_state import GameState
import constants
from agents import HumanAgent, RandomAgent, MinimaxAgent, ExpectiMaxAgent
from determinized_mcts_agent import DeterminizedMCTSAgent
import time


agents = {
    # "RUSTY": HumanAgent(),
    # "ALICE": HumanAgent(),
    # "HARLEM": HumanAgent(),
    # "RUSTY": RandomAgent(),
    # "ALICE": RandomAgent(),
    # "SUSIE": RandomAgent(),
    # "HARLEM": MinimaxAgent(depth=10),
    # "HARLEM": MinimaxAgent("HARLEM", depth=5),
    # "RUSTY": ExpectiMaxAgent("RUSTY", samples=5, depth=12),
    # "SUSIE": ExpectiMaxAgent("SUSIE", samples=5, depth=12),
    # "HARLEM": ExpectiMaxAgent("HARLEM", samples=5, depth=12),
    # "ALICE": ExpectiMaxAgent("ALICE", samples=5, depth=12),
    "RUSTY": ExpectiMaxAgent("RUSTY"),
    "SUSIE": ExpectiMaxAgent("SUSIE"),
    "HARLEM": ExpectiMaxAgent("HARLEM"),
    "ALICE": ExpectiMaxAgent("ALICE"),
    # "HARLEM": ExpectiMaxAgent("HARLEM", samples=10, depth=8),
    # "ALICE": ExpectiMaxAgent("ALICE", samples=10, depth=8),
    # "SUSIE": ExpectiMaxAgent("SUSIE", samples=10, depth=8),
    # "RUSTY": ExpectiMaxAgent("RUSTY", samples=10, depth=8),
    # "HARLEM": DeterminizedMCTSAgent("HARLEM", simulations=1500),
    # "SUSIE": MinimaxAgent("SUSIE", depth=5),
    # "SUSIE": DeterminizedMCTSAgent("SUSIE", simulations=1500),
    # "RUSTY": DeterminizedMCTSAgent("RUSTY", simulations=1500),
    # "ALICE": DeterminizedMCTSAgent("ALICE", simulations=1500),
    # "ALICE": MinimaxAgent("ALICE", depth=5),
    # "RUSTY": MinimaxAgent("RUSTY", depth=5),
}

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


def play_game(state: GameState, render_func=None):
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
    state = make_initial_state()
    # — ADDED: print every player’s opening hand once —
    print("Initial hands:")
    for player, hand in state.hands.items():
        # sort by power so it’s readable
        sorted_ids = [c.identifier for c in sorted(hand, key=lambda c: c.power)]
        print(f"  {player}: {sorted_ids}")
    print("====================================================")
    # Start timer
    start = time.time()

    final_state = play_game(state, render)
    print("\nGame over! Final points:", final_state.points)
    # End timer
    end = time.time()
    print(f"\nGame runtime: {end - start:.2f} seconds")

    print("\nGame over! Final points:", final_state.points)

    # After printing final_state.points:
    # Compute team totals from the completed tricks
    qc_public = [
        player
        for trick in final_state.trick_history
        for player, card in trick
        if card.identifier == 'Q-clubs'
    ]
    # Dedupe in play order
    qc_public = list(dict.fromkeys(qc_public))
    team_no_qc = [p for p in final_state.points if p not in qc_public]
    qc_pts = sum(final_state.points[p] for p in qc_public)
    no_qc_pts = sum(final_state.points[p] for p in team_no_qc)

    # figure out who the two Q-club holders are
    qc_members = {
        player
        for trick in final_state.trick_history
        for player, card in trick
        if card.identifier == 'Q-clubs'
    }
    # the other two are the non-holders
    non_qc_members = [p for p in constants.players if p not in qc_members]

    # print with membership
    print(f"Team Q-clubs ({', '.join(qc_members)}): Total points: {qc_pts}")
    print(f"Team non-Q-clubs ({', '.join(non_qc_members)}): Total points: {no_qc_pts}")

    print("\nGame summary:")
    for i, trick in enumerate(final_state.trick_history, 1):
        trick_summary = {player: card.identifier for player, card in trick}
        print(f"Trick {i}: {trick_summary}")

