import gym
from gym import spaces
import constants
from game_state import GameState

class DoppelkopfEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, player_name):
        super().__init__()
        self.player = player_name
        # 1) action space = max number of legal cards (e.g. 24 distinct identifiers)
        self.action_space = spaces.Discrete(constants.total_cards)
        # 2) observation = a flat vector encoding:
        #    - your hand (one-hot per card)
        #    - current trick history summary (aggregate features)
        #    - points so far
        obs_size = constants.total_cards + 10 + len(constants.players)
        self.observation_space = spaces.Box(0, 1, shape=(obs_size,), dtype=int)
        self.state = None

    def reset(self):
        # deal fresh GameState
        self.state = GameState.deal_new()
        return self._encode(self.state)

    def step(self, action):
        legal = self.state.legal_actions()
        # map `action` (int) → actual Card, or penalize if illegal
        card = self._decode_action(action, legal)
        new_state = self.state.apply_action(card)
        reward = self._compute_reward(self.state, new_state)
        self.state = new_state

        done = new_state.is_terminal()
        obs = self._encode(new_state)
        return obs, reward, done, {}

    def render(self, mode="human"):
        # optional: print hands/tricks
        print(self.state)

    # — helpers below —
    def _encode(self, state: GameState):
        # e.g. one-hot your hand
        vec = [0]*self.observation_space.shape[0]
        for c in state.hands[self.player]:
            vec[c.index] = 1
        # … fill in trick points, etc …
        return np.array(vec, dtype=int)

    def _decode_action(self, idx, legal):
        # if idx not in legal_ids: pick random legal
        for c in legal:
            if c.index == idx:
                return c
        return random.choice(legal)

    def _compute_reward(self, old, new):
        # simple: final scoring only
        if new.is_terminal():
            pts = new.points[self.player]
            opp = sum(new.points[p] for p in new.points if p!=self.player)
            return pts - opp
        return 0
