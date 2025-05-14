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
    # "HARLEM": ExpectiMaxAgent("HARLEM", samples=10, depth=8),
    "ALICE": ExpectiMaxAgent("ALICE"),
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
    print(f"Next to play: {state.next_player}")
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
        print(f"Currently winning: {winner} with {winning_card.identifier}")
    # Current hand (sorted)
    hand = sorted(state.hands[state.next_player], key=lambda c: c.power)
    print("Current hand:", [c.identifier for c in hand])
    # Playable cards
    playable = sorted(state.legal_actions(), key=lambda c: c.power)
    print("Playable cards:", " ".join(c.identifier for c in playable))


def play_game(state: GameState, render_func=None):
    last_trick = None
    while not state.is_terminal():
        current = state.next_player
        agent = agents[current]

        if (isinstance(agent, HumanAgent) or isinstance(agent, DeterminizedMCTSAgent)) and render_func:
            render_func(state, last_trick)

        action = agent.choose(state)
        hand = sorted(state.hands[current], key=lambda c: c.power)
        print(f"       {current} hand:", [card.identifier for card in hand])
        if not isinstance(agent, HumanAgent):
            print(f"{current} played {action.identifier}")

        # 1) Apply the play
        state = state.apply_action(action)

        # 2) Immediate team‐switch logic on Q-clubs
        if action.identifier == 'Q-clubs':
            # who’s already played a Q-club (public set)
            played = {
                         p
                         for trick in state.trick_history
                         for p, c in trick
                         if c.identifier == 'Q-clubs'
                     } | {
                         p for p, c in state.current_trick if c.identifier == 'Q-clubs'
                     }

            if len(played) == 1:
                # first club → only holders learn
                for name, ag in agents.items():
                    if hasattr(ag, '_check_team_switch') \
                            and any(c.identifier == 'Q-clubs' for c in state.hands[name]):
                        ag._check_team_switch(state, force=True)

            elif len(played) == 2:
                # second club → everyone learns
                for ag in agents.values():
                    if hasattr(ag, '_check_team_switch'):
                        ag._check_team_switch(state, force=True)

        # 3) Now check for trick completion
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


    qc_pts    = sum(final_state.points[p] for p in qc_public)
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

