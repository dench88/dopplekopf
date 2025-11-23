import random
import math
from game_state import GameState
from cards import Card, create_deck
import constants

class DeterminizedMCTSAgent:
    def __init__(self, name: str, simulations: int = 500, C: float = math.sqrt(2)):
        self.name = name
        self.simulations = simulations
        self.C = C

    def choose(self, root_state: GameState) -> Card:
        # Initialize or reset root node
        root = None
        for _ in range(self.simulations):
            sampled = self._sample_hidden(root_state)
            if root is None:
                root = Node(sampled, action=None, parent=None)
            else:
                root.state = sampled
                root.children.clear()
                root.visits = 0
                root.total_reward = 0.0
            # Selection & expansion
            node = self._tree_policy(root)
            # Simulation
            reward = self._rollout(node.state)
            # Backpropagation
            self._backpropagate(node, reward)
        # Pick the most-visited child as the best move
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    def _sample_hidden(self, state: GameState) -> GameState:
        # Gather known card identifiers
        known = {card.identifier for _, card in state.current_trick}
        for h in state.hands.values():
            known |= {c.identifier for c in h}
        # Shuffle unseen cards
        deck = create_deck()
        unseen = [c for c in deck if c.identifier not in known]
        random.shuffle(unseen)
        # Deal unseen to other players
        sampled = {p: list(state.hands[p]) for p in state.hands}
        others = [p for p in sampled if p != self.name]
        for i, card in enumerate(unseen):
            sampled[others[i % len(others)]].append(card)
        frozen = {p: tuple(cards) for p, cards in sampled.items()}
        return GameState(
            hands=frozen,
            next_player=state.next_player,
            trick_history=state.trick_history,
            current_trick=state.current_trick,
            points=state.points
        )

    def _tree_policy(self, root: 'Node') -> 'Node':
        node = root
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return node.expand()
            node = node.best_uct_child(self.C)
        return node

    def _rollout(self, state: GameState) -> float:
        st = state
        # Random playout until terminal or stuck
        while not st.is_terminal():
            legal = st.legal_actions()
            if not legal:
                break
            st = st.apply_action(random.choice(legal))
        from heuristics import evaluate
        return evaluate(st, me=self.name)

    def _backpropagate(self, node: 'Node', reward: float):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

class Node:
    def __init__(self, state: GameState, action=None, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(self.state.legal_actions())

    def expand(self) -> 'Node':
        tried_ids = {child.action.identifier for child in self.children}
        for action in self.state.legal_actions():
            if action.identifier not in tried_ids:
                next_state = self.state.apply_action(action)
                child = Node(next_state, action=action, parent=self)
                self.children.append(child)
                return child
        raise RuntimeError("No valid actions to expand")

    def best_uct_child(self, C: float) -> 'Node':
        log_N = math.log(self.visits)
        def uct(ch):
            return (ch.total_reward / ch.visits) + C * math.sqrt(log_N / ch.visits)
        return max(self.children, key=uct)
