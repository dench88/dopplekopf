import random
import time
from typing import List, Optional
from cards import Card
from game_state import GameState


def human_play_logic(player_hand: List[Card], trick_type: Optional[str]) -> Card:
    """
    Prompt the human player to choose a card.
    - Follows suit if possible
    - 'x' for random play
    """

    while True:
        # Show full current hand to the human
        sorted_hand = sorted(player_hand, key=lambda c: c.power)
        # print("Your hand:", [c.identifier for c in sorted_hand])
        playable = [c for c in player_hand if trick_type is None or c.category == trick_type]
        if not playable:
            playable = player_hand

        # 2) If only one option, auto-play it
        if len(playable) == 1:
            must_play = playable[0]
            print(f"\nOnly one card you can play: {must_play.identifier}\n")
            time.sleep(2)
            return must_play

        # 4) Ask for your choice
        choice = input("\nWhat card do you play? ").strip().lower()
        if choice == 'x':
            return random.choice(playable)

        # 5) Match by prefix
        matched = next((c for c in playable if c.identifier.lower().startswith(choice)), None)
        if matched:
            return matched

        print("Invalid choice, try again.")

def get_qc_split_and_points(state: GameState):
    """
    Returns:
      qc_team     – list of players on the Q-clubs side (1 or 2 names)
      non_qc_team – list of players on the other side
      qc_pts      – sum of points scored by qc_team
      non_qc_pts  – sum of points scored by non_qc_team
    """
    # 1) Who played Q-clubs?
    played = {
        p
        for trick in state.trick_history
        for p, c in trick
        if c.identifier == "Q-clubs"
    }
    # 2) Who still holds Q-clubs in hand?
    held = {
        p
        for p, hand in state.hands.items()
        if any(c.identifier == "Q-clubs" for c in hand)
    }
    qc_holders = played | held

    # Decide qc_team
    if len(qc_holders) == 2:
        qc_team = sorted(qc_holders)
    elif len(qc_holders) == 1:
        qc_team = [next(iter(qc_holders))]
    else:
        qc_team = []  # shouldn't really happen in standard Doppelkopf

    # Build the complementary team
    all_players = list(state.points.keys())
    non_qc_team = [p for p in all_players if p not in qc_team]

    # Compute their point sums
    qc_pts     = sum(state.points[p] for p in qc_team)
    non_qc_pts = sum(state.points[p] for p in non_qc_team)

    return qc_team, non_qc_team, qc_pts, non_qc_pts
