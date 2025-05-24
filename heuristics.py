from game_state import GameState
# from agents import ExpectiMaxAgent, MinimaxAgent

def evaluate(state: GameState, me: str, agent=None) -> float:
    # 1) During rollout sampling, stay selfish on raw points
    if agent and getattr(agent, "is_sampling", False):
        my_pts    = state.points[me]
        other_pts = sum(v for p, v in state.points.items() if p != me)
        return my_pts - other_pts

    # 2) Adjust for 10-hearts: deduct 10 from whoever won any trick that contained it
    def adjusted_pts(player_name: str) -> int:
        pts = state.points[player_name]
        for trick in state.trick_history:
            if any(c.identifier == "10-hearts" for _, c in trick):
                # decide winner exactly as in apply_action()
                lead = trick[0][1].category
                strength = lambda pc: pc[1].power if pc[1].category in (lead, "trumps") else -1
                winner = max(trick, key=strength)[0]
                if winner == player_name:
                    pts -= 10
        return pts

    # 3) Tell the agent to refresh its team info, then ask who its teammates are
    if agent:
        agent.update_team_info(state)
        team = set(agent.get_team_members(state))
    else:
        team = set()

    # 4) If I don’t yet know my partner, play selfishly with adjusted points
    if not team:
        my_pts    = adjusted_pts(me)
        other_pts = sum(adjusted_pts(p) for p in state.points if p != me)
        return my_pts - other_pts

    # 5) Otherwise, sum team vs. opponents
    all_players = set(state.points)
    opp = all_players - team

    team_pts = sum(adjusted_pts(p) for p in team)
    opp_pts  = sum(adjusted_pts(p) for p in opp)

    return (team_pts - opp_pts) if me in team else (opp_pts - team_pts)




