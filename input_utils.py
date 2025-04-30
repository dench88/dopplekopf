import random
import time
from typing import List, Optional
from form_deck import Card

def play_logic(player_hand: List[Card], trick_type: Optional[str]) -> Card:
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

        # # 3) Show the options
        # print("\nPlayable cards:")
        # if trick_type is None or playable == player_hand:
        #     print("Any card...")
        # else:
        #     print("; ".join(c.identifier for c in playable))

        # 4) Ask for your choice
        choice = input("\nWhat card do you play? ").strip().lower()
        if choice == 'x':
            return random.choice(playable)

        # 5) Match by prefix
        matched = next((c for c in playable if c.identifier.lower().startswith(choice)), None)
        if matched:
            return matched

        print("Invalid choice, try again.")

