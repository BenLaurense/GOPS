from abc import ABC
import numpy as np
from utils import unpack_state, get_legal_moves

"""
Simple agents to play the game
"""
# TODO: These should probably inherit from a base class with the correct method signatures


class GOPSAgent(ABC):
    def get_action(self, state: np.ndarray, *args) -> tuple[int, float | None]:
        pass


class HumanAgent(GOPSAgent):
    def __init__(self, *args):
        pass

    @staticmethod
    def get_action(state: np.ndarray, *args) -> tuple[int, float | None]:
        # TODO: pygame support?
        sd = unpack_state(state)
        print("="*40)
        print("Your score is {}. Opponent's score is {}".format(sd["score_1"], sd["score_2"]))
        print("Current value card is {}".format(sd["curr_card"]))
        value_cards = np.nonzero(sd["val_cards"])[0] + 1
        print("Remaining value cards: {}".format(value_cards))
        legal_moves = get_legal_moves(state, 1)
        print("Legal moves are {}".format(legal_moves))
        opp_legal_moves = get_legal_moves(state, 2)
        print("Opponent's legal moves are {}".format(opp_legal_moves))

        move = -1
        while move not in legal_moves:
            move = int(input("Choose a valid move:"))
        return move, None   # get_action returns move and logprob


class RandomAgent(GOPSAgent):
    def __init__(self, *args):
        pass

    @staticmethod
    def get_action(state: np.ndarray, *args) -> tuple[int, float | None]:
        legal_moves = get_legal_moves(state, 1)
        return np.random.choice(legal_moves), None  # get_action returns move and logprob
