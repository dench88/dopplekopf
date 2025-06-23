import gymnasium as gym
from gymnasium import spaces
import constants
from game_state import GameState
import numpy as np
import random
from agents import ExpectiMaxAgent, RandomAgent
import constants

# alias for clarity (optional)
TYPES = constants.types
SUITS = constants.suits
# A) the full set of card-identifiers (48 cards, but only 24 unique ids)
FULL_ID_SET = set(constants.CARD_POWER.keys())
# B) a quick lookup from identifier → power
ID_TO_POWER = constants.CARD_POWER  # already a dict identifier→int
# build your 24 “card types” and index map
CARD_TYPES = [(r, s) for r in TYPES for s in SUITS]
TYPE_TO_IDX = {t: i for i, t in enumerate(CARD_TYPES)}

class DoppelkopfEnv(gym.Env):
    def __init__(self, player_name: str, expectimax_prob: float = 0.85, custom_opponents: bool = False):
        super().__init__()
        self.player = player_name
        self.agent = None
        # Expectimax opponents
        self.opponent_agents = {}
        self.expectimax_prob = expectimax_prob
        # self._assign_opponents()
        self.custom_opponents = custom_opponents
        if not self.custom_opponents:
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
        n_other_players         = 3
        obs_size = prelim_obs_size + 1 + n_other_players

        self.pos_offset = obs_size
        self.winflag_offset = self.pos_offset + 1
        self.pts_curr_trick_offset = self.winflag_offset + 1
        self.strongest_offset = self.pts_curr_trick_offset + 1
        self.trump_power_offset = self.strongest_offset + 1
        self.color_offset = self.trump_power_offset + 1  # this one spans 3 dims
        obs_size = self.color_offset + 3
        self.observation_space = spaces.Box(0, 30, shape=(obs_size,), dtype=np.int16)

        self.state = None

    def _assign_opponents(self):
        """(Re)randomize opponent types from scratch."""
        self.opponent_agents.clear()
        for p in constants.players:
            if p == self.player:
                continue
            if random.random() < self.expectimax_prob:
                self.opponent_agents[p] = ExpectiMaxAgent(p, verbose=False)
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
            # self._assign_opponents()
            if not self.custom_opponents and hasattr(self, "_assign_opponents"):
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

        ### compute true_team exactly the same way reward does ###
        if getattr(self, "agent", None):
            # force team info up-to-date
            self.agent.update_team_info(state, force=True)
            true_team = set(self.agent.get_team_members(state))
        else:
            true_team = set()  # no team known

        # 7) Position in current trick
        pos = len(state.current_trick)
        vec[self.pos_offset] = pos

        # 8) Am I (or my team) winning so far?
        if pos > 0:
            lead_suit = state.current_trick[0][1].category

            def str_fn(pair):
                _, c = pair
                return c.power if c.category in (lead_suit, "trumps") else -1

            winner = max(state.current_trick, key=str_fn)[0]
            vec[self.winflag_offset] = 1 if (winner in true_team) else 0
        else:
            vec[self.winflag_offset] = 0

        # 9) Points in current trick
        vec[self.pts_curr_trick_offset] = sum(c.points for _, c in state.current_trick)

        # 10) Strongest card left in the unseen deck
        #    Build the set of all seen identifiers
        seen_ids = {
                       card.identifier
                       for trick in (*state.trick_history, state.current_trick)
                       for _, card in trick
                   } | {
                       card.identifier
                       for card in state.hands[self.player]
                   }
        #    The unseen identifiers are:
        remaining_ids = FULL_ID_SET - seen_ids
        #    Find the one with maximum power
        if remaining_ids:
            strongest_id = max(remaining_ids, key=lambda ident: ID_TO_POWER[ident])
            vec[self.strongest_offset] = ID_TO_POWER[strongest_id]
        else:
            vec[self.strongest_offset] = 0

        # 11) My remaining trump power
        my_trumps = [c.power for c in state.hands[self.player] if c.category == "trumps"]
        vec[self.trump_power_offset] = sum(my_trumps)

        # 12) How many “color” tricks completed per suit?
        counts = {"hearts": 0, "spades": 0, "clubs": 0}
        for trick in state.trick_history:
            lead = trick[0][1].category
            if lead in counts:
                counts[lead] += 1
        for i, suit in enumerate(("hearts", "spades", "clubs")):
            vec[self.color_offset + i] = counts[suit]

        return vec


    def _decode_action(self, idx, legal):
        # if idx not in legal_ids: pick random legal
        for c in legal:
            if TYPE_TO_IDX[(c.type, c.suit)] == idx:
                return c
        return random.choice(legal)


    def _compute_reward(self, old: GameState, new: GameState) -> float:
        """
        Gives:
        • A +X or –X reward immediately after each completed trick (X = trick’s point total).
        • At the end of the hand, gives the usual team‐point differential.

        old: the GameState before applying the last move (including opponents’ cards for that trick).
        new: the GameState after that move (so if a trick just finished, new.trick_history includes it).
        """

        # 1) If the hand is not yet over, see if a new trick just completed
        if not new.is_terminal():
            # Did we just finish a trick? Compare lengths of trick_history
            if len(new.trick_history) > len(old.trick_history):
                last_trick = new.trick_history[-1]

                # Sum up the points in that trick
                trick_pts = sum(card.points for _, card in last_trick)

                # Determine who won that trick
                suit = last_trick[0][1].category
                def strength(pair):
                    _, card = pair
                    # A card “counts” if it’s either the same suit or a trump
                    return card.power if card.category in (suit, "trumps") else -1

                winner = max(last_trick, key=strength)[0]

                # Figure out which two players are on ALICE’s team right now
                if getattr(self, "agent", None):
                    team = self.agent.get_team_members(new)
                    teams_known = getattr(self.agent, "is_team_playing", False)
                else:
                    team = []
                    teams_known = False

                # If the trick‐winner is on ALICE’s team, give +trick_pts; otherwise –trick_pts
                # return trick_pts if (winner in team) else -trick_pts

                # Trick won by my team
                if winner in team:
                    return trick_pts
                # Trick lost and teams known → regular penalty
                if teams_known:
                    return -trick_pts
                # Trick lost and teams unknown → expected penalty
                return -trick_pts / 3

            # If no trick was just completed (e.g. mid‐trick), give zero intermediate reward
            return 0

        # 2) If the entire hand is over
        #    Now give the final “team‐point differential” exactly as before.

        if getattr(self, "agent", None):
            # Force the agent to recompute team info now that all Q‐clubs have been played
            self.agent.update_team_info(new, force=True)
            team = self.agent.get_team_members(new)

            # Sum up final points for ALICE’s team vs. the opponents
            team_pts = sum(new.points[p] for p in team)
            opp_pts  = sum(v for p, v in new.points.items() if p not in team)

            # If (for some reason) ALICE is on that returned “team” list, reward = team_pts – opp_pts.
            # Otherwise (if ALICE somehow wasn’t included), flip the sign.
            return (team_pts - opp_pts) if (self.player in team) else (opp_pts - team_pts)

        else:
            # Fallback: treat it as a “solo” hand if there’s no rule‐based agent attached
            my_pts  = new.points[self.player]
            opp_pts = sum(v for p, v in new.points.items() if p != self.player)
            return my_pts - opp_pts
