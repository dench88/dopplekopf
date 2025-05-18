import random
import math
from typing import List
from game_state import GameState
from form_deck import Card
from input_utils import human_play_logic
from heuristics import evaluate
import constants
from tqdm import tqdm
from typing import Optional


class HumanAgent:
    def choose(self, state: GameState) -> Card:
        trick_type = None
        if state.current_trick:
            trick_type = state.current_trick[0][1].category
        hand = list(state.hands[state.next_player])
        return human_play_logic(hand, trick_type)

class RandomAgent:
    def choose(self, state: GameState) -> Card:
        return random.choice(state.legal_actions())


def card_strength(card: Card, trick_suit: str) -> int:
    """
    Shared strength logic: only cards of the trick suit or trumps count, others are powerless.
    """
    if card.category == trick_suit or card.category == 'trumps':
        return card.power
    return -1



class DuckFeedMixin:
    """
    Shared logic for mid-trick duck/feed decision, plus team-aware last-player override.
    """
    def _duck_or_feed(self, state: GameState, legal: List[Card]) -> Optional[Card]:
        num_players = len(constants.players)
        # determine public Q-club info
        qc_public = {p for t in state.trick_history for p, c in t if c.identifier == 'Q-clubs'}
        qc_public |= {p for p, c in state.current_trick if c.identifier == 'Q-clubs'}
        if self.is_team_playing or len(qc_public) == 2:
            qc_public |= {p for p, h in state.hands.items() if any(c.identifier == 'Q-clubs' for c in h)}
        
        # mid-trick strength tracking
        trick_suit = None
        current_winner = None
        current_strength = -1
        if state.current_trick:
            trick_suit = state.current_trick[0][1].category
            current_winner, current_card = max(state.current_trick, key=lambda pc: pc[1].power)
            current_strength = card_strength(current_card, trick_suit)
        
        # last-player override: feed or win
        if state.current_trick and len(state.current_trick) == num_players - 1 and self.is_team_playing:
            winners = [c for c in legal if card_strength(c, trick_suit) > current_strength]
            losers  = [c for c in legal if card_strength(c, trick_suit) <= current_strength]
            partner_wins = ((current_winner in qc_public) == (self.name in qc_public))
            if winners:
                if partner_wins:
                    pool = [c for c in losers if c.identifier != '10-hearts'] or losers
                    choice = max(pool, key=lambda c: (c.points, c.power))
                    print(f"{self.name} (last-player) partner winning, feed with {choice.identifier}")
                    return choice
                pool = [c for c in winners if c.identifier != '10-hearts'] or winners
                choice = max(pool, key=lambda c: (c.points, c.power))
                print(f"{self.name} (last-player) win trick with {choice.identifier}")
                return choice
            if partner_wins:
                pool = [c for c in losers if c.identifier != '10-hearts'] or losers
                choice = max(pool, key=lambda c: (c.points, c.power))
                print(f"{self.name} (last-player) partner winning, feed with {choice.identifier}")
                return choice
            choice = min(losers, key=lambda c: (c.points, c.power))
            print(f"{self.name} (last-player) duck cheaply with {choice.identifier}")
            return choice

        # mid-trick duck if cannot win
        if state.current_trick:
            winners = [c for c in legal if card_strength(c, trick_suit) > current_strength]
            if not winners:
                losers = [c for c in legal if card_strength(c, trick_suit) <= current_strength]
                partner_wins = ((current_winner in qc_public) == (self.name in qc_public))
                if partner_wins and self.is_team_playing:
                    pool = [c for c in losers if c.identifier != '10-hearts'] or losers
                    choice = max(pool, key=lambda c: (c.points, c.power))
                    print(f"{self.name} mid-trick: partner winning, feed with {choice.identifier}")
                    return choice
                choice = min(losers, key=lambda c: (c.points, c.power))
                print(f"{self.name} mid-trick: duck cheaply with {choice.identifier}")
                return choice
        return None




class MinimaxAgent(DuckFeedMixin):
    """
    Perfect-information minimax with alpha-beta pruning, maximizing team or individual score.
    """
    def __init__(self, name: str, depth: int = None):
        self.name = name
        # If no depth given, search to end of current trick
        self.depth = depth
        self.is_team_playing = False  # add this line!

    def _check_team_switch(self, state: GameState, *, force=False):
        # 1) Pre-trick boundary guard (unless forced)
        if state.current_trick and not force:
            return

        # 2) Who’s played a Q-club so far?
        played = {
                     p for trick in state.trick_history for p, card in trick
                     if card.identifier == 'Q-clubs'
                 } | {
                     p for p, card in state.current_trick if card.identifier == 'Q-clubs'
                 }

        # 3) Who still holds one in hand?
        holders = {
            p for p, h in state.hands.items()
            if any(c.identifier == 'Q-clubs' for c in h)
        }

        # 4) Should I switch now?
        first_club = (len(played) == 1 and self.name in holders)
        second_club = (len(played) == 2)

        if (first_club or second_club) and not self.is_team_playing:
            # Build the full team set
            qc_public = set(played)
            if second_club or self.name in holders:
                # now reveal both holders
                qc_public |= {
                    p for p, h in state.hands.items()
                    if any(c.identifier == 'Q-clubs' for c in h)
                }

            # Your team is qc_public if you’re in it, else the complement
            if self.name in qc_public:
                team = qc_public
            else:
                team = set(state.hands.keys()) - qc_public

            partners = sorted(team - {self.name})
            partner_str = ", ".join(partners)

            print(f"{self.name} now knows their team (with {partner_str}) and switches to team play!")
            self.is_team_playing = True


    def choose(self, state: GameState) -> Card:
        legal = state.legal_actions()
        # team switch before any pick if start of trick
        if not state.current_trick and hasattr(self, '_check_team_switch'):
            self._check_team_switch(state)
        # duck/feed override
        choice = self._duck_or_feed(state, legal)
        if choice:
            return choice
        # minimax search reach end of current trick by default
        depth = self.depth if self.depth is not None else (len(constants.players) - len(state.current_trick))
        move, _ = self._minimax(state, depth, True, -math.inf, math.inf)
        return move


    def _minimax(self, state: GameState, depth: int, maximizing: bool,
                 alpha: float, beta: float) -> (Card, float):
        if depth == 0 or state.is_terminal():
            return None, evaluate(state, me=self.name, agent=self)

        best_move = None
        if maximizing:
            max_eval = -math.inf
            for action in state.legal_actions():
                _, eval_val = self._minimax(state.apply_action(action), depth-1,
                                             False, alpha, beta)
                if eval_val > max_eval:
                    max_eval, best_move = eval_val, action
                alpha = max(alpha, eval_val)
                if alpha >= beta:
                    break
            return best_move, max_eval
        else:
            min_eval = math.inf
            for action in state.legal_actions():
                _, eval_val = self._minimax(state.apply_action(action), depth-1,
                                             True, alpha, beta)
                if eval_val < min_eval:
                    min_eval, best_move = eval_val, action
                beta = min(beta, eval_val)
                if alpha >= beta:
                    break
            return best_move, min_eval

class ExpectiMaxAgent(DuckFeedMixin):
    """
    Sampling-based imperfect-information agent: for each candidate move, samples 10 possible hypothetical deals
    and runs a depth-limited perfect-information minimax, averaging scores to pick the best.
    """
    def __init__(self, name: str, samples: int = 10, depth: int = None):
        self.name = name
        self.samples = samples
        self.depth = depth
        self.is_team_playing = False  # initially selfish

    def _check_team_switch(self, state: GameState, *, force=False):
        # 1) Pre-trick boundary guard (unless forced)
        if state.current_trick and not force:   
            return

        # 2) Who’s played a Q-club so far?
        played = {
                     p for trick in state.trick_history for p, card in trick
                     if card.identifier == 'Q-clubs'
                 } | {
                     p for p, card in state.current_trick if card.identifier == 'Q-clubs'
                 }

        # 3) Who still holds one in hand?
        holders = {
            p for p, h in state.hands.items()
            if any(c.identifier == 'Q-clubs' for c in h)
        }

        # 4) Should I switch now?
        first_club = (len(played) == 1 and self.name in holders)
        second_club = (len(played) == 2)

        if (first_club or second_club) and not self.is_team_playing:
            # Build the full team set
            qc_public = set(played)
            if second_club or self.name in holders:
                # now reveal both holders
                qc_public |= {
                    p for p, h in state.hands.items()
                    if any(c.identifier == 'Q-clubs' for c in h)
                }

            # Your team is qc_public if you’re in it, else the complement
            if self.name in qc_public:
                team = qc_public
            else:
                team = set(state.hands.keys()) - qc_public

            partners = sorted(team - {self.name})
            partner_str = ", ".join(partners)

            print(f"{self.name} now knows their team (with {partner_str}) and switches to team play!")
            self.is_team_playing = True

    def choose(self, state: GameState) -> Card:
        legal = state.legal_actions()
        # team switch at trick start
        if not state.current_trick:
            self._check_team_switch(state)
        # duck/feed override
        choice = self._duck_or_feed(state, legal)
        if choice:
            return choice
        # sampling + minimax
        depth = self.depth if self.depth is not None else (len(constants.players) - len(state.current_trick))
        best_moves, best_score = [], -math.inf
        seen = set()
        for action in tqdm(state.legal_actions(), desc="Root actions", leave=True, disable=True):
            if action.identifier in seen:
                continue
            seen.add(action.identifier)
            scores = []
            for _ in range(self.samples):
                sampled_state = self._sample_hidden(state)
                _, sc = MinimaxAgent(self.name, depth)._minimax(
                    sampled_state.apply_action(action), depth, True, -math.inf, math.inf
                )
                scores.append(sc)
            avg = sum(scores) / len(scores)
            tqdm.write(f"  → {action.identifier}: used {len(scores)}/{self.samples} samples, avg {avg:.2f}")
            if avg > best_score:
                best_score, best_moves = avg, [action]
            elif avg == best_score:
                best_moves.append(action)
        final_choice = random.choice(best_moves)
        print(f"[INFO] Final choice: {final_choice.identifier}")
        return final_choice

    def _sample_hidden(self, state: GameState) -> GameState:
        # Gather known identifiers: played cards + my hand
        known = {c.identifier for _, c in state.current_trick}
        known |= {c.identifier for c in state.hands[self.name]}
        for trick in state.trick_history:
            known |= {c.identifier for _, c in trick}
        # Build unseen Card objects
        unseen = []
        for t in constants.types:
            for s in constants.suits:
                ident = f"{t}-{s}"
                if ident not in known:
                    unseen.append(Card(t, s))
        random.shuffle(unseen)
        # Start with current hands
        sampled = {p: list(state.hands[p]) for p in state.hands}
        # Distribute unseen among opponents only
        opponents = [p for p in sampled if p != self.name]
        for idx, card in enumerate(unseen):
            sampled[opponents[idx % len(opponents)]].append(card)
        frozen = {p: tuple(cards) for p, cards in sampled.items()}
        return GameState(
            hands=frozen,
            next_player=state.next_player,
            trick_history=state.trick_history,
            current_trick=state.current_trick,
            points=state.points
        )

# def debug_one_sample_trick(agent, state: GameState, action: Card):
#     """
#     Sample hidden hands once, apply `action`, then finish off
#     the rest of this one trick at random, printing each play
#     and the final trick point tally.
#     """
#     # 1) create a fully‐dealt sample
#     sampled: GameState = agent._sample_hidden(state)
#     print("\n--- DEBUG: one sampled deal ---")
#     for p, hand in sampled.hands.items():
#         print(f"  {p} holds: {[c.identifier for c in hand]}")
#     print(f"\nLead: {state.next_player} plays {action.identifier}")
    
#     # 2) play the lead
#     s = sampled.apply_action(action)
    
#     # 3) continue this trick until 4 cards are down
#     plays: List[Tuple[str, str]] = [(state.next_player, action.identifier)]
#     while s.current_trick:
#         nxt = s.next_player
#         legal = s.legal_actions()
#         # pick randomly among legal for this debug
#         choice = random.choice(legal)
#         print(f"      {nxt} plays {choice.identifier}")
#         plays.append((nxt, choice.identifier))
#         s = s.apply_action(choice)
    
#     # 4) trick is complete
#     trick = s.trick_history[-1]
#     pts = sum(c.points for _, c in trick)
#     print("\nTrick finished:")
#     for p, c in trick:
#         print(f"    {p}: {c.identifier} ({c.points} pts)")
#     print(f"Total trick points → {pts}\n")
#     print("--- end DEBUG sample ---\n")