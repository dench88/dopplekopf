from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import constants
from form_deck import Card

@dataclass(frozen=True)
class GameState:
    # Each player's hand: immutable tuple of Card
    hands: Dict[str, Tuple[Card, ...]]
    # Completed tricks: tuple of tuples of (player, Card)
    trick_history: Tuple[Tuple[Tuple[str, Card], ...], ...] = field(default_factory=tuple)
    # Current trick in progress: tuple of (player, Card)
    current_trick: Tuple[Tuple[str, Card], ...] = field(default_factory=tuple)
    # Accumulated points per player
    points: Dict[str, int] = field(default_factory=lambda: dict(constants.player_points))
    # Who plays next
    next_player: str = ""

    def is_terminal(self) -> bool:
        # Game ends when no cards remain
        return all(len(hand) == 0 for hand in self.hands.values())

    def legal_actions(self) -> List[Card]:
        # Determine suit/trick_type
        if not self.current_trick:
            suit = None
        else:
            # category of first card in trick
            suit = self.current_trick[0][1].category
        # current player's hand
        hand = list(self.hands[self.next_player])
        # follow suit if possible
        if suit is None:
            return hand
        follow = [c for c in hand if c.category == suit]
        return follow if follow else hand

    def apply_action(self, card: Card) -> "GameState":
        # Mutable copies
        current_hands = {p: list(h) for p, h in self.hands.items()}
        current_trick_history = [list(tr) for tr in self.trick_history]
        the_current_trick = list(self.current_trick)
        current_trick_points = dict(self.points)

        # remove card from hand and add to trick
        current_hands[self.next_player].remove(card)
        the_current_trick.append((self.next_player, card))

        # determine next player index
        players = list(constants.players.keys())
        idx = players.index(self.next_player)
        next_idx = (idx + 1) % len(players)
        next_player = players[next_idx]

        # if trick complete (4 cards)
        if len(the_current_trick) == len(players):
            # find winning card by power
            # determine trick suit from the very first card
            trick_suit = the_current_trick[0][1].category
            # “strength” function: only same-suit or trumps ever count
            def strength(pc):
                card = pc[1]
                if card.category == trick_suit or card.category == 'trumps':
                    return card.power
                # any off-suit, non-trump color card is powerless
                return -1

            # pick the trick-winner by adjusted strength
            winner, winning_card = max(the_current_trick, key=strength)

            # sum trick points
            trick_pts = sum(c.points for _, c in the_current_trick)
            current_trick_points[winner] += trick_pts
            # record trick
            current_trick_history.append(the_current_trick)
            # clear current trick and set next player to winner
            the_current_trick = []
            next_player = winner

        # freeze data structures
        frozen_hands = {p: tuple(h) for p, h in current_hands.items()}
        frozen_history = tuple(tuple(tr) for tr in current_trick_history)
        frozen_current = tuple(the_current_trick)

        # return new GameState
        return GameState(
            hands=frozen_hands,
            trick_history=frozen_history,
            current_trick=frozen_current,
            points=current_trick_points,
            next_player=next_player
        )