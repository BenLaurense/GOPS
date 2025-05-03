import numpy as np

"""
Helper functions
"""

def sgn(x: float | int) -> int:
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


def flip_state(state: np.ndarray) -> np.ndarray:
    """
    Helper function for changing player1/player2
    :param state: state to flip (from player 1 perspective)
    :param num_cards: number of cards in game
    :return: flipped state
    """
    num_cards = get_num_cards(state)
    state[0, 0], state[0, 1] = state[0, 1], state[0, 0]
    state[0, num_cards + 3:num_cards*2 + 3], state[0, num_cards*2 + 3:] = \
        state[0, num_cards*2 + 3:], state[0, num_cards + 3:num_cards*2 + 3].copy()
    return state


def get_num_cards(state: np.ndarray) -> int:
    """
    Gets the number of cards from a state
    :param state: state to consider
    :return: number of cards
    """
    return state.shape[1]//3 - 1


def unpack_state(state: np.ndarray) -> dict:
    num_cards = get_num_cards(state)
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
    return cards_idx + 1 # Convention
