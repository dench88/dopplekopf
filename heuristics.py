from game_state import GameState
# from agents import ExpectiMaxAgent, MinimaxAgent

def evaluate(state: GameState, me: str, agent=None) -> float:
    # During sampling: selfish
    if agent and getattr(agent, "is_sampling", False):
        my_pts = state.points[me]
        other_pts = sum(v for p,v in state.points.items() if p != me)
        return my_pts - other_pts

    # Find public Q-clubs
    qc_public = {p for trick in state.trick_history for p,c in trick if c.identifier =='Q-clubs'}
    qc_public |= {p for p,c in state.current_trick if c.identifier == 'Q-clubs'}

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

    # —— Penalty for losing a high-power card ——
    # Determine how many tricks remain (cards in hand)
    remaining_tricks = len(state.hands[me])
    scale = remaining_tricks / 12  # fraction of hand left
    base_factor = 0.9  # tweak to taste

    # Find the last card I played
    last_card = None
    if state.current_trick:
        # in the middle of a trick
        for player, card in reversed(state.current_trick):
            if player == me:
                last_card = card
                break
    elif state.trick_history:
        # at trick boundary, look at the final trick
        last_trick = state.trick_history[-1]
        for player, card in reversed(last_trick):
            if player == me:
                last_card = card
                break

    if last_card:
        # penalise the loss of that card’s power
        loss_penalty = base_factor * scale * last_card.power
        base_score -= loss_penalty

    return base_score



    return base_score
