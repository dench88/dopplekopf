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

def fast_opening_play(agent_name: str,
                      hands: dict[str, tuple[Card, ...]],
                      trick_history: tuple,
                      current_trick: tuple) -> Optional[Card]:
    """
    If we are literally at the very first play of the very first trick, 
    apply your fast heuristics.  Otherwise return None.
    """
    # only at absolute game start
    if trick_history or current_trick:
        return None

    hand = list(hands[agent_name])

    # — Priority 1: A-spades or A-clubs ——
    acards = [c for c in hand if c.identifier in ("A-spades", "A-clubs")]
    if acards:
        spades = [c for c in hand if c.category == "spades" and c.identifier not in constants.trumps]
        clubs  = [c for c in hand if c.category == "clubs"  and c.identifier not in constants.trumps]
        # if fewer safe spades than clubs, lead A-spades, else A-clubs
        if len(spades) <= len(clubs):
            for c in acards:
                if c.category == "spades":
                    return c
        for c in acards:
            if c.category == "clubs":
                return c

    # — Priority 2: A-hearts if few hearts left ——
    a_hearts     = [c for c in hand if c.identifier == "A-hearts"]
    colour_hearts = [c for c in hand if c.category == "hearts" and c.identifier not in constants.trumps]
    if a_hearts and len(colour_hearts) <= 2:
        return a_hearts[0]

    # — Priority 3: if I hold 3–5 J/Q, play the 3rd strongest ——
    jq = [c for c in hand if c.type in ("J", "Q")]
    if 3 <= len(jq) <= 5:
        jq_sorted = sorted(jq, key=lambda c: c.power, reverse=True)
        return jq_sorted[2]

    # — Priority 4: if I hold exactly 2 J/Q, play the 2nd strongest ——
    if len(jq) == 2:
        jq_sorted = sorted(jq, key=lambda c: c.power, reverse=True)
        return jq_sorted[1]

    return None


class TeamMixin:
    def update_team_info(self, state: GameState, *, force=False):
        # only at trick boundary unless forced
        if state.current_trick and not force:
            return

        # 1) who’s played a Q-clubs?
        played = {
            p
            for trick in state.trick_history
            for p, c in trick
            if c.identifier == 'Q-clubs'
        } | {
            p
            for p, c in state.current_trick
            if c.identifier == 'Q-clubs'
        }

        # 2) who still holds one?
        still_holds = {
            p
            for p, h in state.hands.items()
            if any(c.identifier == 'Q-clubs' for c in h)
        }

        original_holders = played | still_holds

        # A) no Qs seen → nobody knows
        if len(played) == 0:
            self.is_team_playing = False
            self.team_members = None
            return

        # B) exactly one seen → only the *other* holder (still_holds) learns
        if len(played) == 1:
            # if *I* still hold a Q (i.e. I didn't just play it),
            # then I now know my partner
            if self.name in still_holds:
                self.is_team_playing = True
                self.team_members = sorted(original_holders)
            return

        # C) both Qs seen → everyone learns
        if len(played) == 2:
            self.is_team_playing = True
            non_holders = set(state.hands) - original_holders
            if self.name in original_holders:
                self.team_members = sorted(original_holders)
            else:
                self.team_members = sorted(non_holders)
            
            return

    def get_team_members(self, state: GameState) -> List[str]:
        """
        Returns the two names on my side once I know my team,
        or None if I’m still in “selfish” (pre-team) mode.
        """
        # who’s played Q-clubs so far?
        qc_public = {
            p
            for trick in state.trick_history
            for p, c in trick
            if c.identifier == "Q-clubs"
        } | {
            p
            for p, c in state.current_trick
            if c.identifier == "Q-clubs"
        }

        # who still holds a Q-clubs in hand?
        holders = {
            p
            for p, hand in state.hands.items()
            if any(c.identifier == "Q-clubs" for c in hand)
        }

        # once I’m in holders or both seen, reveal full teams
        if self.name in holders or len(qc_public) == 2:
            qc_public |= holders

        # if I haven’t switched to team play yet, nobody’s shown
        if not getattr(self, "is_team_playing", False):
            return []

        # pick my side
        all_players = set(state.hands)
        if self.name in qc_public:
            return sorted(qc_public)
        else:
            return sorted(all_players - qc_public)

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
            cards_that_are_winners = [c for c in legal if card_strength(c, trick_suit) > current_strength]
            losers  = [c for c in legal if card_strength(c, trick_suit) <= current_strength]
            partner_wins = ((current_winner in qc_public) == (self.name in qc_public))
            if cards_that_are_winners:
                if partner_wins:
                    # try to feed with a low card that isn’t 10-hearts,
                    # but if there are no losers at all fall back to winners
                    feed_pool = [c for c in losers if c.identifier != '10-hearts']
                    pool = feed_pool or cards_that_are_winners
                    return max(pool, key=lambda c: (c.points, c.power))
                    # pool = [c for c in losers if c.identifier != '10-hearts'] or losers
                    # choice = max(pool, key=lambda c: (c.points, c.power))
                    print(f"    **{self.name} is last-player. Partner winning. Feed with {choice.identifier}")
                    # return choice
                pool = [c for c in cards_that_are_winners if c.identifier != '10-hearts'] or cards_that_are_winners
                choice = max(pool, key=lambda c: (c.points, c.power))
                print(f"    **{self.name} (last-player) win trick with {choice.identifier}")
                return choice
            if partner_wins:
                pool = [c for c in losers if c.identifier != '10-hearts'] or losers
                choice = max(pool, key=lambda c: (c.points, c.power))
                print(f"    **{self.name} (last-player) partner winning, feed with {choice.identifier}")
                return choice
            choice = min(losers, key=lambda c: (c.points, c.power))
            print(f"    **{self.name} (last-player) duck cheaply with {choice.identifier}")
            return choice

        # mid-trick duck if cannot win
        if state.current_trick:
            cards_that_are_winners = [c for c in legal if card_strength(c, trick_suit) > current_strength]
            if not cards_that_are_winners:
                losers = [c for c in legal if card_strength(c, trick_suit) <= current_strength]
                partner_wins = ((current_winner in qc_public) == (self.name in qc_public))
                if partner_wins and self.is_team_playing:
                    pool = [c for c in losers if c.identifier != '10-hearts'] or losers
                    choice = max(pool, key=lambda c: (c.points, c.power))
                    print(f"    **{self.name} mid-trick: partner winning, feed with {choice.identifier}")
                    return choice
                choice = min(losers, key=lambda c: (c.points, c.power))
                print(f"    **{self.name} mid-trick: duck cheaply with {choice.identifier}")
                return choice
        return None


class MinimaxAgent(DuckFeedMixin, TeamMixin):
    """
    Perfect-information minimax with alpha-beta pruning, maximizing team or individual score.
    """
    def __init__(self, name: str, depth: int = None):
        self.name = name
        # If no depth given, search to end of current trick
        self.depth = depth
        self.is_team_playing = False  # add this line!

    def choose(self, state: GameState) -> Card:
        self.update_team_info(state)
        legal = state.legal_actions()
        # # team switch before any pick if start of trick
        # if not state.current_trick and hasattr(self, '_check_team_switch'):
        #     self._check_team_switch(state)
        # duck/feed override
        choice = self._duck_or_feed(state, legal)
        # 2) fast opening heuristic
        fast = fast_opening_play(self.name,
                             state.hands,
                             state.trick_history,
                             state.current_trick)
        if fast:
            print(f"{self.name} used fast opening rule: {fast.identifier}")
            return fast
    
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

class ExpectiMaxAgent(DuckFeedMixin, TeamMixin):
    """
    Sampling-based imperfect-information agent: for each candidate move, samples 10 possible hypothetical deals
    and runs a depth-limited perfect-information minimax, averaging scores to pick the best.
    """
    def __init__(self, name: str, samples: int = 10, depth: int = None):
        self.name = name
        self.samples = samples
        self.depth = depth
        self.is_team_playing = False  # initially selfish



    def choose(self, state: GameState) -> Card:
        self.update_team_info(state)
        legal = state.legal_actions()
        # 2) fast opening heuristic
        fast = fast_opening_play(self.name,
                             state.hands,
                             state.trick_history,
                             state.current_trick)
        if fast:
            print(f"{self.name} used fast opening rule: {fast.identifier}")
            return fast
        # duck/feed override
        choice = self._duck_or_feed(state, legal)
        if choice:
            return choice
        # sampling + minimax
        depth = self.depth if self.depth is not None else (len(constants.players) - len(state.current_trick))
        best_moves, best_score = [], -math.inf
        seen = set()
        # for action in tqdm(state.legal_actions(), desc="Root actions", leave=True, disable=True):
        #     if action.identifier in seen:
        #         continue
        #     seen.add(action.identifier)
        # 0) at the top of choose(), after you’ve determined depth/samples
        # grab the raw legal actions
        raw_actions = state.legal_actions()

        # apply your “no 10-hearts in tricks 1–6 when trick < 20 points” rule
        actions = []
        current_trick_pts = sum(c.points for _, c in state.current_trick)
        for c in raw_actions:
            if (
                c.identifier == "10-hearts"
                and len(state.trick_history) < 6
                and current_trick_pts < 20
            ):
                # skip it
                continue
            actions.append(c)

        # now use actions instead of state.legal_actions() in your sampling loop
        seen = set()
        for action in actions:
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
            tqdm.write(f"      → {action.identifier}: used {len(scores)}/{self.samples} samples, avg {avg:.2f}")
            if avg > best_score:
                best_score, best_moves = avg, [action]
            elif avg == best_score:
                best_moves.append(action)
        
        # if no best moves, return a random legal action
        if not best_moves:
            best_moves = legal
            print(f"[WARNING] No best moves found, using random legal action.")

        final_choice = random.choice(best_moves)
        # print(f"[INFO] Final choice: {final_choice.identifier}")
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

