import numpy as np

"""
Helper functions
"""

def sgn(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


def flip_state(state, num_cards):
    """
    Helper function for changing player1/player2
    :param state: state to flip (from player 1 perspective)
    :param num_cards: number of cards in game
    :return: flipped state
    """
    state[0, 0], state[0, 1] = state[0, 1], state[0, 0]
    state[0, num_cards + 3:num_cards*2 + 3], state[0, num_cards*2 + 3:] = \
        state[0, num_cards*2 + 3:], state[0, num_cards + 3:num_cards*2 + 3].copy()
    return state


def unpack_state(state, num_cards):
    s1, s2 = state[0, 0], state[0, 1]
    curr_card = state[0, 2]
    val_cards = state[0, 3:num_cards+3]
    cards_1, cards_2 = state[0, num_cards+3:num_cards*2+3], state[0, num_cards*2+3:num_cards*3+3]
    return {
        "score_1": s1, "score_2": s2,
        "curr_card": curr_card,
        "val_cards": val_cards,
        "cards_1": cards_1, "cards_2": cards_2
    }
