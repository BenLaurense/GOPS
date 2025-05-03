import numpy as np
from GOPS_environment import GOPS
from utils import flip_state, sgn, get_num_cards

"""
Helper functions for agents
"""

def get_legal_moves(state: np.ndarray, player: int) -> np.ndarray[int]:
    """
    Gets allowed actions for a state
    :param state: state to consider
    :param player: player taking the actions
    :param num_cards: number of cards in game
    :return: ndarray of values of cards that can be played
    """
    num_cards = get_num_cards(state)
    cards_idx = np.nonzero(state[0, num_cards*player+3:num_cards*(player+1)+3])[0]
    return cards_idx + 1


def collect_data(game: GOPS, train_agent, opp_agent, expl_rate: float, num_episodes: int):
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

            r.append(s_[0, 0] - s_[0, 1])
            batch_logprobs.append(a1_logprob)

        # Subject to tweaking lol
        tweaked_reward = list(map(lambda x: x + 100*sgn(s_[0, 0] - s_[0, 1]), r))
        batch_rewards += tweaked_reward  # Or other functions?

        game.reset()
    return batch_logprobs, batch_rewards
