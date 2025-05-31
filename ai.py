import gymnasium as gym
from gymnasium import spaces
import constants
from game_state import GameState
import numpy as np
import random
from agents import ExpectiMaxAgent, RandomAgent
import constants

# alias for clarity (optional)
RANKS = constants.types
SUITS = constants.suits

# build your 24 “card types” and index map
CARD_TYPES = [(r, s) for r in RANKS for s in SUITS]
TYPE_TO_IDX = {t: i for i, t in enumerate(CARD_TYPES)}

class DoppelkopfEnv(gym.Env):
    def __init__(self, player_name: str, expectimax_prob: float = 0.85):
        super().__init__()
        self.player = player_name
        self.agent = None
        # Expectimax opponents
        self.opponent_agents = {}
        self.expectimax_prob = expectimax_prob
        self._assign_opponents()

        # — your existing hand/action setup —
        # CARD_TYPES   = [(r,s) for r in RANKS for s in SUITS]   # 24 types
        hand_size    = len(CARD_TYPES)
        self.action_space = spaces.Discrete(hand_size)

        # record where each slice begins
        self.hand_offset  = 0
        self.seen_offset  = hand_size
        self.points_offset = hand_size + hand_size
        self.trick_offset  = self.points_offset + len(constants.players)
        #Remove trick suit lengths from obs vector
        # self.suit_offset   = self.trick_offset + 1

        # total obs = hand_counts + seen_counts + player_pts + trick_cnt + suit_lengths(3)
        prelim_obs_size = hand_size + hand_size + len(constants.players) + 1

        # Add team related flags after your suit-counts etc.
        self.team_flag_offset   = self.trick_offset + 1
        self.partner_offset     = self.team_flag_offset + 1
        n_other_players         = len(constants.players) - 1
        obs_size = prelim_obs_size + 1 + n_other_players

        self.observation_space = spaces.Box(
            low=0,
            high=2,                  # we’ll use 0–2 for counts
            shape=(obs_size,),
            dtype=np.int16
        )
        self.state = None

    def _assign_opponents(self):
        """(Re)randomize opponent types from scratch."""
        self.opponent_agents.clear()
        for p in constants.players:
            if p == self.player:
                continue
            if random.random() < self.expectimax_prob:
                self.opponent_agents[p] = ExpectiMaxAgent(p)
            else:
                self.opponent_agents[p] = RandomAgent()

    def reset(self, *, seed=None, options=None):
        # 1) If you want reproducible deals, you can optionally seed Python’s RNG here:
        if seed is not None:
            import random as _random
            _random.seed(seed)

        # 2) Deal a fresh GameState (using your random_deal helper or manual slicing)
        self.state = GameState.random_deal()

        # 3) (Re‐)assign opponents if you rotate them per episode
        if hasattr(self, "_assign_opponents"):
            self._assign_opponents()

        # 4) Force team‐info update if you attach a rule‐based agent
        if getattr(self, "agent", None):
            self.agent.update_team_info(self.state, force=True)

        # 5) Encode and return as (obs, info)
        obs = self._encode(self.state)
        return obs, {}  # empty info dict


    def step(self, action):
        # (1) RL’s move
        legal           = self.state.legal_actions()
        card            = self._decode_action(action, legal)
        state_after_me  = self.state.apply_action(card)

        # (2) Now have each expectimax opponent play in turn
        # for opp in [p for p in constants.players if p != self.player]:
        #     opp_card        = self.opponent_agents[opp].choose(state_after_me)
        #     state_after_me  = state_after_me.apply_action(opp_card)
        for opp_name, opp_agent in self.opponent_agents.items():
            opp_card = opp_agent.choose(state_after_me)
            state_after_me = state_after_me.apply_action(opp_card)


        # (3) Reward
        reward = self._compute_reward(self.state, state_after_me)

        # (4) Update state
        self.state = state_after_me

        # (5) Team flags
        if hasattr(self, "agent") and self.agent:
            self.agent.update_team_info(self.state)

        # (6) Return
        done = self.state.is_terminal()
        obs  = self._encode(self.state)
        return obs, reward, done, False, {}


    def render(self, mode="human"):
        # optional: print hands/tricks
        print(self.state)



    def _encode(self, state: GameState):
        vec = np.zeros(self.observation_space.shape[0], dtype=np.int16)
        
        # 1) Hand counts
        for c in state.hands[self.player]:
            idx = TYPE_TO_IDX[(c.type, c.suit)]
            vec[self.hand_offset + idx] += 1

        # 2) Seen-card counts
        for trick in (*state.trick_history, state.current_trick):
            for _, card in trick:
                idx = TYPE_TO_IDX[(card.type, card.suit)]
                vec[self.seen_offset + idx] += 1

        # 3) Points
        for i, p in enumerate(constants.players):
            vec[self.points_offset + i] = state.points[p]

        # 4) Trick count
        vec[self.trick_offset] = len(state.trick_history)

        # 5) Team-play flag (only if self.agent is not None)
        if getattr(self, "agent", None):
            vec[self.team_flag_offset] = int(self.agent.is_team_playing)
        else:
            vec[self.team_flag_offset] = 0

        # 6) Partner one-hot (only if self.agent is not None)
        if getattr(self, "agent", None):
            partner_list = self.agent.get_team_members(state)
            others = [p for p in constants.players if p != self.player]
            for i, p in enumerate(others):
                vec[self.partner_offset + i] = 1 if p in partner_list else 0
        # else: leave those partner slots as 0

        return vec


    def _decode_action(self, idx, legal):
        # if idx not in legal_ids: pick random legal
        for c in legal:
            if TYPE_TO_IDX[(c.type, c.suit)] == idx:
                return c
        return random.choice(legal)


    def _compute_reward(self, old: GameState, new: GameState) -> float:
        # only give reward at the end of the hand
        if not new.is_terminal():
            return 0

        # if we have a rule-based agent attached, refresh its team info
        team = []
        if hasattr(self, "agent") and self.agent:
            self.agent.update_team_info(new)
            team = self.agent.get_team_members(new)

        # --- Team-aware reward ---
        if len(team) == 2:
            # sum up points for my side vs theirs
            team_pts = sum(new.points[p] for p in team)
            opp_pts  = sum(v for p,v in new.points.items() if p not in team)
            # from my perspective: if I'm on the team, reward = (my team – opponents)
            # if somehow I'm not in the returned list, flip the sign
            if self.player in team:
                return team_pts - opp_pts
            else:
                return opp_pts - team_pts

        # --- Fallback to solo reward ---
        my_pts  = new.points[self.player]
        opp_pts = sum(v for p,v in new.points.items() if p != self.player)
        return my_pts - opp_pts

