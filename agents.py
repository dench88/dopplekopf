import random
import math
from typing import List
from game_state import GameState
from form_deck import Card
from input_utils import play_logic
from heuristics import evaluate
import constants
from tqdm import tqdm


class HumanAgent:
    def choose(self, state: GameState) -> Card:
        trick_type = None
        if state.current_trick:
            trick_type = state.current_trick[0][1].category
        hand = list(state.hands[state.next_player])
        return play_logic(hand, trick_type)

class RandomAgent:
    def choose(self, state: GameState) -> Card:
        return random.choice(state.legal_actions())

class MinimaxAgent:
    """
    Perfect-information minimax with alpha-beta pruning, maximizing team or individual score.
    """
    def __init__(self, name: str, depth: int = None):
        self.name = name
        # If no depth given, search to end of current trick
        self.depth = depth
        self.is_team_playing = False  # add this line!

    def _check_team_switch(self, state: GameState):
        # only run at trick boundary
        if state.current_trick:
            return

        # who’s revealed Q-clubs?
        qc_public = {
            p for trick in state.trick_history
            for p, card in trick
            if card.identifier == 'Q-clubs'
        }
        qc_public |= {
            p for p, card in state.current_trick
            if card.identifier == 'Q-clubs'
        }

        # determine if *I* know my team
        knows_team = (self.name in qc_public) or (len(qc_public) == 2)
        if knows_team and not self.is_team_playing:
            print(f"{self.name} now knows their team and switches to team play!")
            self.is_team_playing = True

    def choose(self, state: GameState) -> Card:
            legal = state.legal_actions()
            # only check at the start of a trick
            if not state.current_trick:
                self._check_team_switch(state)

            # Duck any losing trump trick (power ≤ current, prefer ≤9 pts)
            if state.current_trick:
                lead_cat = state.current_trick[0][1].category
                if lead_cat == "trumps" and all(c.category == "trumps" for c in legal):
                    current_win = max(card.power for _, card in state.current_trick)
                    losers = [c for c in legal if c.power <= current_win]
                    if losers:
                        losers.sort(key=lambda c: c.power)
                        # pick first with pts ≤ 9
                        duck = next((c for c in losers if c.points <= 9), losers[0])
                        print(f"{self.name} ducks with {duck.identifier} (losing trick cheaply)")
                        return duck

            if len(legal) == 1:
                return legal[0]

            # Determine search depth: finish current trick if not specified
            if self.depth is None:
                players = list(constants.players.keys())
                remaining = len(players) - len(state.current_trick)
                depth = remaining
            else:
                depth = self.depth

            best_move, _ = self._minimax(state, depth, True, -math.inf, math.inf)
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

class ExpectiMaxAgent:
    """
    Sampling-based imperfect-information agent: for each candidate move, samples 20 possible hypothetical deals
    and runs a depth-limited perfect-information minimax, averaging scores to pick the best.
    """
    def __init__(self, name: str, samples: int = 20, depth: int = None):
        self.name = name
        self.samples = samples
        self.depth = depth
        self.is_team_playing = False  # initially selfish

    def _check_team_switch(self, state: GameState):
        # only run at trick boundary
        if state.current_trick:
            return

        # who’s revealed Q-clubs?
        qc_public = {
            p for trick in state.trick_history
            for p, card in trick
            if card.identifier == 'Q-clubs'
        }
        qc_public |= {
            p for p, card in state.current_trick
            if card.identifier == 'Q-clubs'
        }

        # determine if *I* know my team
        knows_team = (self.name in qc_public) or (len(qc_public) == 2)
        if knows_team and not self.is_team_playing:
            print(f"{self.name} now knows their team and switches to team play!")
            self.is_team_playing = True

    def choose(self, state: GameState) -> Card:
        legal = state.legal_actions()
        # only check at the start of a trick
        if not state.current_trick:
            self._check_team_switch(state)

        # 1. Only one move? Just play it.
        if len(legal) == 1:
            print(f"{self.name} has only one move: {legal[0].identifier}")
            return legal[0]

        # 2. First move of game? Try fast heuristic
        if len(state.trick_history) == 0 and len(state.current_trick) == 0:
            hand = list(state.hands[state.next_player])

            # Priority 1: A-spades or A-clubs
            acards = [c for c in hand if c.identifier in ("A-spades", "A-clubs")]
            if acards:
                spades = [c for c in hand if c.category == "spades" and c.identifier not in constants.trumps]
                clubs = [c for c in hand if c.category == "clubs" and c.identifier not in constants.trumps]
                if len(spades) <= len(clubs):
                    for c in acards:
                        if c.category == "spades":
                            print(f"{self.name} used fast rule: A-spades")
                            return c
                for c in acards:
                    if c.category == "clubs":
                        print(f"{self.name} used fast rule: A-clubs")
                        return c

            # Priority 2: A-hearts and few hearts
            a_hearts = [c for c in hand if c.identifier == "A-hearts"]
            colour_hearts = [c for c in hand if c.category == "hearts" and c.identifier not in constants.trumps]
            if a_hearts and len(colour_hearts) <= 2:
                print(f"{self.name} used fast rule: A-hearts")
                return a_hearts[0]

            # Priority 3: 3–5 J/Q
            jq = [c for c in hand if c.type in ("J", "Q")]
            if 3 <= len(jq) <= 5:
                jq_sorted = sorted(jq, key=lambda c: c.power, reverse=True)
                print(f"{self.name} used fast rule: Holds 3–5 Js or Qs")
                return jq_sorted[2]

            # Priority 4: exactly 2 J/Q
            if len(jq) == 2:
                jq_sorted = sorted(jq, key=lambda c: c.power, reverse=True)
                print(f"{self.name} used fast rule: 2 JQ")
                return jq_sorted[1]

            # Duck any losing trump trick (power ≤ current, prefer ≤9 pts)
            if state.current_trick:
                lead_cat = state.current_trick[0][1].category
                legal = state.legal_actions()
                if lead_cat == "trumps" and all(c.category == "trumps" for c in legal):
                    current_win = max(card.power for _, card in state.current_trick)
                    losers = [c for c in legal if c.power <= current_win]
                    if losers:
                        losers.sort(key=lambda c: c.power)
                        # pick first with pts ≤ 9
                        duck = next((c for c in losers if c.points <= 9), losers[0])
                        print(f"{self.name} ducks with {duck.identifier} (losing trick cheaply)")
                        return duck

        # 3. Dynamic depth and sample control
        tricks_played = len(state.trick_history)
        if self.depth is not None:
            depth = self.depth
            samples = self.samples
        else:
            if tricks_played < 4:
                depth = 6
                samples = 20
            elif tricks_played < 8:
                depth = 8
                samples = 12
            else:
                depth = 10
                samples = 8

        # 4. Full Expectimax with sampling

        self.is_sampling = True

        best_moves = []
        best_score = -math.inf
        actions = state.legal_actions()

        self.is_sampling = True
        for action in tqdm(actions, desc="Root actions", leave=True, disable=True):
            scores = []
            for sample_idx in range(self.samples):
                sampled_state = self._sample_hidden(state)
                next_state = sampled_state.apply_action(action)
                _, score = MinimaxAgent(self.name, depth)._minimax(
                    next_state, depth, True, -math.inf, math.inf
                )
                scores.append(score)

                # ✅ Early stopping
                if sample_idx >= 5:
                    running_avg = sum(scores) / len(scores)
                    if running_avg < best_score - 10:
                        break

            avg_score = sum(scores) / len(scores)
            tqdm.write(f"Avg score for {action.identifier}: {avg_score:.2f}")

            if avg_score > best_score:
                best_score, best_moves = avg_score, [action]
            elif avg_score == best_score:
                best_moves.append(action)
        self.is_sampling = False

        return random.choice(best_moves)

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
