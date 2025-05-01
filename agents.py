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

    def _check_team_switch(self, state: GameState, *, force=False):
        # 1) pre‐trick boundary guard (unless forced)
        if state.current_trick and not force:
            return

        # 2) who’s played a Q-club so far?
        played = {
                     p for trick in state.trick_history
                     for p, card in trick
                     if card.identifier == 'Q-clubs'
                 } | {
                     p for p, card in state.current_trick
                     if card.identifier == 'Q-clubs'
                 }

        # 3) who still holds one in hand?
        holders = {
            p for p, h in state.hands.items()
            if any(c.identifier == 'Q-clubs' for c in h)
        }

        # 4) if I’m a holder & at least one club played, or both clubs played
        if ((self.name in holders and played) or len(played) == 2) \
                and not self.is_team_playing:
            print(f"{self.name} now knows their team and switches to team play!")
            self.is_team_playing = True

    def choose(self, state: GameState) -> Card:
            legal = state.legal_actions()
            # only check at the start of a trick
            if not state.current_trick:
                self._check_team_switch(state)

            if len(legal) == 1:
                return legal[0]

            if state.current_trick:
                # 1) who’s winning so far?
                winner, winning_card = max(state.current_trick, key=lambda pc: pc[1].power)

                # 2) legal set & which of mine would lose
                # legal = state.legal_actions()
                losing = [c for c in legal if c.power <= winning_card.power]
                if losing and len(losing) == len(legal):
                    # 3) figure out if winner is on my team
                    qc_public = {
                                    p for trick in state.trick_history for p, c in trick
                                    if c.identifier == 'Q-clubs'
                                } | {
                                    p for p, c in state.current_trick if c.identifier == 'Q-clubs'
                                }
                    if self.name in qc_public or len(qc_public) == 2:
                        qc_public |= {
                            p for p, h in state.hands.items()
                            if any(c.identifier == 'Q-clubs' for c in h)
                        }
                    partner_wins = (winner in qc_public) == (self.name in qc_public)

                    if partner_wins:
                        # feed partner: throw highest-point loser
                        duck = max(losing, key=lambda c: (c.points, c.power))
                        print(f"{self.name} feeds partner by playing {duck.identifier}")
                    else:
                        # standard duck: lowest-point loser
                        duck = min(losing, key=lambda c: (c.points, c.power))
                        print(f"{self.name} ducks with {duck.identifier} (no chance to win)")
                    return duck

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

    def _check_team_switch(self, state: GameState, *, force=False):
        # 1) pre‐trick boundary guard (unless forced)
        if state.current_trick and not force:
            return

        # 2) who’s played a Q-club so far?
        played = {
                     p for trick in state.trick_history
                     for p, card in trick
                     if card.identifier == 'Q-clubs'
                 } | {
                     p for p, card in state.current_trick
                     if card.identifier == 'Q-clubs'
                 }

        # 3) who still holds one in hand?
        holders = {
            p for p, h in state.hands.items()
            if any(c.identifier == 'Q-clubs' for c in h)
        }

        # 4) if I’m a holder & at least one club played, or both clubs played
        if ((self.name in holders and played) or len(played) == 2) \
                and not self.is_team_playing:
            print(f"{self.name} now knows their team and switches to team play!")
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

        # 3. duck/feed logic: if I cannot win mid-trick
        if state.current_trick:
            winner, winning_card = max(state.current_trick, key=lambda pc: pc[1].power)
            losing = [c for c in legal if c.power <= winning_card.power]
            if losing and len(losing) == len(legal):
                qc_public = {
                    p for trick in state.trick_history for p, c in trick
                    if c.identifier == 'Q-clubs'
                } | {
                    p for p, c in state.current_trick if c.identifier == 'Q-clubs'
                }
                if self.is_team_playing or len(qc_public) == 2:
                    qc_public |= {
                        p for p, h in state.hands.items()
                        if any(c.identifier == 'Q-clubs' for c in h)
                    }
                feed = self.is_team_playing and ((winner in qc_public) == (self.name in qc_public))
                if feed:
                    duck = max(losing, key=lambda c: (c.points, c.power))
                    print(f"{self.name} feeds partner by playing {duck.identifier}")
                else:
                    duck = min(losing, key=lambda c: (c.points, c.power))
                    print(f"{self.name} ducks with {duck.identifier} (no chance to win)")
                return duck

        # 4. sample+minimax
        players = list(constants.players.keys())
        remaining = len(players) - len(state.current_trick)
        depth = self.depth if self.depth is not None else remaining
        best_moves, best_score = [], -math.inf
        for action in tqdm(state.legal_actions(), desc="Root actions", leave=True, disable=True):
            scores=[]
            for i in range(self.samples):
                sampled = self._sample_hidden(state)
                _, sc = MinimaxAgent(self.name, depth)._minimax(
                    sampled.apply_action(action), depth, True, -math.inf, math.inf
                )
                scores.append(sc)
                if i>=5 and sum(scores)/len(scores) < best_score-10:
                    break
            avg = sum(scores)/len(scores)
            tqdm.write(f"Avg score for {action.identifier}: {avg:.2f}")
            if avg>best_score:
                best_score, best_moves = avg, [action]
            elif avg==best_score:
                best_moves.append(action)
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
