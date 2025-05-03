from GOPS_environment import GOPS
from BasicAgents import GOPSAgent
from utils import flip_state, sgn

"""
Helper functions for agents
"""


def collect_data(game: GOPS, train_agent: GOPSAgent, opp_agent: GOPSAgent, expl_rate: float, num_episodes: int):
    """
    Data collection loop
    :param game: GOPS game
    :param train_agent: agent being trained
    :param opp_agent: opponent agent
    :param expl_rate: exploration rate
    :param num_episodes: number of training episodes
    :return:
    """
    num_games = num_episodes//game.num_cards

    # For VPG, need to collect the episode final reward, and the episode action logprobs
    batch_logprobs = []
    batch_rewards = []

    for _ in range(num_games):
        done = False
        s_ = game.state.copy()
        r = []
        while not done:
            s = s_
            a1, a1_logprob = train_agent.get_action(s, expl_rate)
            a2, _ = opp_agent.get_action(flip_state(s), 0.0)
            s_, done = game.step(a1, a2)

            reward = s_[0, 0].item() - s_[0, 1].item()
            r.append(reward)
            batch_logprobs.append(a1_logprob)

        # Subject to tweaking lol
        tweaked_reward = list(map(lambda x: x + 10*sgn(reward), r))
        batch_rewards += tweaked_reward  # Or other functions?

        game.reset()
    return batch_logprobs, batch_rewards
