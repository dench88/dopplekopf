from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import constants
from form_deck import Card, create_deck

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

    
    @classmethod
    def random_deal(cls):
        # 1) build & shuffle the deck
        deck = create_deck()  # returns 48 shuffled Card objects

        # 2) split into equal hands (12 cards each if 4 players)
        players  = list(constants.players)              # e.g. ["ALICE","BOB","CAROL","DAVE"]
        hand_size = len(deck) // len(players)

        # simple slice-based deal
        hands = {
            p: tuple(deck[i*hand_size : (i+1)*hand_size])
            for i, p in enumerate(players)
        }

        # 3) initialize scores to zero
        points = {p: 0 for p in players}

        # 4) choose the first leader (you can randomize or cycle)
        next_player = players[0]

        # 5) return a fresh GameState
        return cls(
            hands=hands,
            trick_history=(),
            current_trick=(),
            points=points,
            next_player=next_player,
        )

    
    def apply_action(self, card: Card) -> "GameState":
        # 1) Mutable copies of everything
        players = list(constants.players)
        hands   = {p: list(h)             for p, h in self.hands.items()}
        history = [list(tr)              for tr     in self.trick_history]
        trick   = [*self.current_trick, (self.next_player, card)]
        points  = dict(self.points)

        # remove card from hand
        hands[self.next_player].remove(card)

        # compute the “default” next player
        idx      = players.index(self.next_player)
        next_plr = players[(idx + 1) % len(players)]

        # 2) If trick is complete, score it
        if len(trick) == len(players):
            lead_suit = trick[0][1].category
            def strength(pc):
                cat = pc[1].category
                return pc[1].power if (cat == lead_suit or cat == "trumps") else -1

            winner, _ = max(trick, key=strength)
            points[winner] += sum(c.points for _, c in trick)
            history.append(trick)
            trick = []
            next_plr = winner

        # 3) Freeze and return new state
        return GameState(
            hands={p: tuple(h) for p, h in hands.items()},
            trick_history=tuple(tuple(tr) for tr in history),
            current_trick=tuple(trick),
            points=points,
            next_player=next_plr
        )
