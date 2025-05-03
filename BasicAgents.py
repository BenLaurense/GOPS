import numpy as np
from utils import unpack_state
from agent_utils import get_legal_moves

"""
Simple agents to play the game
"""
# TODO: These should probably inherit from a base class with the correct method signatures


class HumanAgent:
    def __init__(self, num_cards, player):
        self.num_cards = num_cards
        self.player = player

    def get_action(self, state, *args):
        # TODO: pygame support?
        sd = unpack_state(state)
        print(sd)
        print("="*40)
        if self.player == 1:
            print("Your score is {}. Opponent's score is {}".format(sd["score_1"], sd["score_2"]))
        print("Current value card is {}".format(sd["curr_card"]))
        value_cards = np.nonzero(sd["val_cards"])[0] + 1
        print("Remaining value cards: {}".format(value_cards))
        legal_moves = get_legal_moves(state, self.player)
        print("Legal moves are {}".format(legal_moves))
        opp_legal_moves = get_legal_moves(state, 3 - self.player)
        print("Opponent's legal moves are {}".format(opp_legal_moves))

        move = -1
        while move not in legal_moves:
            move = int(input("Choose a valid move:"))
        return move, None   # get_action returns move and logprob


class RandomAgent:
    def __init__(self, num_cards, player, *args):
        self.player = player
        self.num_cards = num_cards

    def get_action(self, state):
        legal_moves = get_legal_moves(state, self.player)
        return np.random.choice(legal_moves), None  # get_action returns move and logprob
