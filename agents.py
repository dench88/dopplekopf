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
        # mid-trick leader
        current_winner, current_card = (None, None)
        if state.current_trick:
            current_winner, current_card = max(state.current_trick, key=lambda pc: pc[1].power)
            trick_suit = state.current_trick[0][1].category
            current_strength = card_strength(current_card, trick_suit)
        # last-player override
        if len(state.current_trick) == num_players - 1 and self.is_team_playing:
            trick_suit = state.current_trick[0][1].category
            # candidates that can win
            winners = [c for c in legal if card_strength(c, trick_suit) > current_strength]
            losers = [c for c in legal if card_strength(c, trick_suit) <= current_strength]
            partner_wins = ((current_winner in qc_public) == (self.name in qc_public))
            if winners:
                # if partner already winning, feed partner by throwing high points
                if partner_wins:
                    pool = [c for c in losers if c.identifier != '10-hearts'] or losers
                    choice = max(pool, key=lambda c: (c.points, c.power))
                    print(f"{self.name} (last-player) partner winning, feed with {choice.identifier}")
                    return choice
                # otherwise win with high point card
                pool = [c for c in winners if c.identifier != '10-hearts'] or winners
                choice = max(pool, key=lambda c: (c.points, c.power))
                print(f"{self.name} (last-player) win trick with {choice.identifier}")
                return choice
            # no winners: if partner winning, throw highest loser, else lowest
            if partner_wins:
                pool = [c for c in losers if c.identifier != '10-hearts'] or losers
                choice = max(pool, key=lambda c: (c.points, c.power))
                print(f"{self.name} (last-player) partner winning, feed with {choice.identifier}")
                return choice
            # standard duck cheaply
            choice = min(losers, key=lambda c: (c.points, c.power))
            print(f"{self.name} (last-player) can't win so duck cheaply with {choice.identifier}")
            return choice

        # -- Mid-trick duck logic when no chance to win --
        if not state.current_trick:
            return None
        trick_suit = state.current_trick[0][1].category
        winner, win_card = max(state.current_trick, key=lambda pc: pc[1].power)
        strength_win = card_strength(win_card, trick_suit)
        losers = [c for c in legal if card_strength(c, trick_suit) <= strength_win]
        if len(losers) != len(legal):
            return None  # can win, defer to search
        partner_wins = ((winner in qc_public) == (self.name in qc_public))
        if partner_wins and self.is_team_playing:
            pool = [c for c in losers if c.identifier != '10-hearts'] or losers
            choice = max(pool, key=lambda c: (c.points, c.power))
            print(f"{self.name} mid-trick: partner winning, feed with {choice.identifier}")
            return choice
        # otherwise standard duck cheaply
        choice = min(losers, key=lambda c: (c.points, c.power))
        print(f"{self.name} mid-trick: duck cheaply with {choice.identifier}")
        return choice



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
        # only check at the start of a trick
        if not state.current_trick:
            self._check_team_switch(state)

        if len(legal) == 1:
            return legal[0]
        # duck/feed override
        card = self._duck_or_feed(state, legal)
        if card:
            return card

        # **ensure we search to the end of this trick**
        players   = list(constants.players.keys())
        remaining = len(players) - len(state.current_trick)

        if self.depth is None:
            depth = remaining
        else:
            # bump up user‐requested depth if it’s too small
            depth = max(self.depth, remaining)

        best_move, _ = self._minimax(state, depth, True, -math.inf, math.inf)
        # print(f"    [DEBUG] {self.name} final choice: {best_move.identifier}")

        return best_move


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
        # team-check at trick start
        if not state.current_trick:
            self._check_team_switch(state)
        # 1. single move trivial
        if len(legal) == 1:
            print(f"{self.name} has only one move: {legal[0].identifier}")
            return legal[0]
        # 2. fast heuristic for first trick
        if not state.trick_history and not state.current_trick:
            hand = list(state.hands[state.next_player])
            acards = [c for c in hand if c.identifier in ("A-spades","A-clubs")]
            if acards:
                spades = [c for c in hand if c.category=="spades" and c.identifier not in constants.trumps]
                clubs  = [c for c in hand if c.category=="clubs"  and c.identifier not in constants.trumps]
                if len(spades)<=len(clubs):
                    for c in acards:
                        if c.category=="spades":
                            print(f"{self.name} used fast rule: A-spades")
                            return c
                for c in acards:
                    if c.category=="clubs":
                        print(f"{self.name} used fast rule: A-clubs")
                        return c
            a_hearts = [c for c in hand if c.identifier=="A-hearts"]
            colour_hearts = [c for c in hand if c.category=="hearts" and c.identifier not in constants.trumps]
            if a_hearts and len(colour_hearts)<=2:
                print(f"{self.name} used fast rule: A-hearts")
                return a_hearts[0]
            jq = [c for c in hand if c.type in ("J","Q")]
            if 3<=len(jq)<=5:
                sorted_jq = sorted(jq, key=lambda c:c.power, reverse=True)
                print(f"{self.name} used fast rule: Holds 3–5 Js or Qs")
                return sorted_jq[2]
            if len(jq)==2:
                sorted_jq = sorted(jq, key=lambda c:c.power, reverse=True)
                print(f"{self.name} used fast rule: 2 JQ")
                return sorted_jq[1]

        # duck/feed override
        card = self._duck_or_feed(state, legal)
        if card:
            return card

        # 4. sample+minimax
        players = list(constants.players.keys())
        remaining = len(players) - len(state.current_trick)
        depth = self.depth if self.depth is not None else remaining
        best_moves, best_score = [], -math.inf

        seen = set()
        # for action in tqdm(state.legal_actions(), desc="Root actions", leave=True, disable=True):
        for action in state.legal_actions():
            if action.identifier in seen:
                continue  # Skip duplicate card evaluations
            seen.add(action.identifier)
            scores=[]
            samples_used = 0

            for i in range(self.samples):
                samples_used = i + 1
                sampled = self._sample_hidden(state)

                _, sc = MinimaxAgent(self.name, depth)._minimax(
                    sampled.apply_action(action), depth, True, -math.inf, math.inf
                )

                scores.append(sc)
                if i>=5 and sum(scores)/len(scores) < best_score-10:
                    break
            avg = sum(scores)/len(scores)

            tqdm.write(f"  → {action.identifier}: used {samples_used}/{self.samples} samples, avg {avg:.2f}")

            if avg>best_score:
                best_score, best_moves = avg, [action]
            elif avg==best_score:
                best_moves.append(action)

        self.is_final_choice = True
        if best_moves:
            final_action = random.choice(best_moves)
        else:
            print("    WARNING: No best move found, defaulting to random choice.")
            final_action = random.choice(state.legal_actions())
        self.is_final_choice = False
        # print(f"    [DEBUG] {self.name} final choice: {final_action.identifier}")

        return final_action

        # return random.choice(best_moves)

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