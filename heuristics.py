from game_state import GameState
# from agents import ExpectiMaxAgent, MinimaxAgent

def evaluate(state: GameState, me: str, agent=None) -> float:
    # 1) During sampling: stay selfish and use the real state.points
    if agent and getattr(agent, "is_sampling", False):
        my_pts    = state.points[me]
        other_pts = sum(v for p, v in state.points.items() if p != me)
        return my_pts - other_pts

    # helper: “adjusted” points for lookahead, zeroing out any 10-hearts
    def adjusted_pts(player_name: str) -> int:
        pts = state.points[player_name]
        for trick in state.trick_history:
            # if that trick included a 10-hearts, then subtract 10 from whoever won it
            if any(c.identifier == "10-hearts" for _, c in trick):
                # replicate the same strength logic you use in apply_action():
                suit = trick[0][1].category
                def strength(pc):
                    card = pc[1]
                    return card.power if card.category in (suit, "trumps") else -1
                winner = max(trick, key=strength)[0]
                if winner == player_name:
                    pts -= 10
        return pts

    # 2) Find public Q-club holders
    qc_public = {
        p for trick in state.trick_history for p, c in trick
        if c.identifier == "Q-clubs"
    } | {
        p for p, c in state.current_trick if c.identifier == "Q-clubs"
    }

    # 3) If <2 clubs public AND I’m not one of them, fall back to selfish
    if len(qc_public) < 2 and me not in qc_public:
        my_pts    = adjusted_pts(me)
        other_pts = sum(adjusted_pts(p) for p in state.points if p != me)
        return my_pts - other_pts

    # 4) Otherwise reveal full teams
    if me in qc_public or len(qc_public) == 2:
        qc_public |= {
            p for p, hand in state.hands.items()
            if any(c.identifier == "Q-clubs" for c in hand)
        }

    team1 = qc_public
    team2 = set(state.points) - qc_public

    pts1 = sum(adjusted_pts(p) for p in team1)
    pts2 = sum(adjusted_pts(p) for p in team2)

    # 5) return from my perspective
    return (pts1 - pts2) if me in team1 else (pts2 - pts1)




