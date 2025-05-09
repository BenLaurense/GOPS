from GOPS_environment import GOPS
from agents.basic_agents import GOPSAgent
from utils import flip_state, sgn

"""
Helper functions for agents
"""


def collect_trajectories(
        game: GOPS,
        train_agent: GOPSAgent,
        opp_agent: GOPSAgent,
        num_traj: int,
        expl_rate: float=0.0):
    """
    Data collection loop
    :param game: GOPS game
    :param train_agent: agent being trained
    :param opp_agent: opponent agent
    :param expl_rate: exploration rate
    :param num_episodes: number of training episodes
    :return:
    """
    # For VPG, need to collect the episode scores, and the episode action logprobs
    logprobs, scores = [], []
    for _ in range(num_traj):
        done = False
        s_ = game.state.copy()
        traj_logprobs, traj_scores = [], []
        while not done:
            s = s_
            a1, a1_logprob = train_agent.get_action(s, expl_rate)
            a2, _ = opp_agent.get_action(flip_state(s), 0.0)
            s_temp, done = game.step(a1, a2)
            s_ = s_temp.copy() # Note the copy
            traj_logprobs.append(a1_logprob)
            score_diff = (s_[0, 0] - s_[0, 1]).item()
            traj_scores.append(score_diff)
        logprobs.append(traj_logprobs)
        scores.append(traj_scores)
        game.reset()
    return logprobs, scores


def profile(agent: GOPSAgent, agent2: GOPSAgent, num_games: int, num_cards: int) -> list[int]:
    """
    Plays two agents against each other and collects the wins
    :param agent:
    :param agent2:
    :param num_games:
    :param num_cards:
    :return:
    """
    G = GOPS(num_cards)
    scores = []
    for _ in range(num_games):
        done = False
        s_ = G.state.copy()
        while not done:
            s = s_
            a1, _ = agent.get_action(s)
            a2, _ = agent2.get_action(flip_state(s))
            s_, done = G.step(a1, a2)
        scores.append(sgn((s_[0, 0] - s_[0, 1]).item()))
        G.reset()
    return scores
