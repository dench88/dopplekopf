from game_state import GameState
# from agents import ExpectiMaxAgent, MinimaxAgent

def evaluate(state: GameState, me: str, agent=None) -> float:
    # During sampling: selfish
    if agent and getattr(agent, "is_sampling", False):
        my_pts = state.points[me]
        other_pts = sum(v for p,v in state.points.items() if p != me)
        return my_pts - other_pts

    # Find public Q-clubs
    qc_public = {p for trick in state.trick_history for p,c in trick if c.identifier=='Q-clubs'}
    qc_public |= {p for p,c in state.current_trick if c.identifier=='Q-clubs'}

    # 2) If fewer than 2 Q-clubs are public AND I’m NOT a Q-holder → still individual
    if len(qc_public) < 2 and me not in qc_public:
        my_pts = state.points[me]
        other_pts = sum(v for p, v in state.points.items() if p != me)
        return my_pts - other_pts

    # Reveal full teams if holder or both revealed
    if me in qc_public or len(qc_public)==2:
        qc_public |= {p for p,hand in state.hands.items()
                      if any(c.identifier=='Q-clubs' for c in hand)}

    qc_pts = sum(state.points[p] for p in qc_public)
    other_pts = sum(v for p,v in state.points.items() if p not in qc_public)

    base_score = qc_pts - other_pts if me in qc_public else other_pts - qc_pts

    # If my last play was a trump that failed to win, crush its score
    if state.trick_history and state.trick_history[-1][-1][0] == me:
        last_trick = state.trick_history[-1]
        last_player, last_card = last_trick[-1]
        if last_player == me and last_card.category == "trumps":
            win_power = max(c.power for _, c in last_trick)
            if last_card.power <= win_power:
                # give a huge penalty so no search will pick this
                return -1e6

    # ——— incentive for saving strong cards, weighted by tricks left ———
    # number of future tricks = number of cards still in hand
    remaining_tricks = len(state.hands[me])
    # a small base factor (tweak 0.1 up/down to taste)
    base_factor = 0.1
    # scale it by the fraction of the game left (remaining_tricks / 12)
    scale = remaining_tricks / 12
    saved_power = sum(card.power for card in state.hands[me])
    # add the bonus
    base_score += base_factor * scale * saved_power



    return base_score
