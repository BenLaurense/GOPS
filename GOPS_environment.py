import math
import numpy as np
from random import random


"""
Environment for GOPS
    - Players are numbered 1 and 2
    - State is represented by a 1xn binary array [scores | current value card | value cards | P1 cards | P2 cards]
    (it's a binary array except for current value card, which is an integer entry representing the card's value) 
    - Actions are represented by the VALUE of the card being played (from 1 to num_cards)
    - 
"""


class GOPS:
    def __init__(self):
        self.num_cards = 3
        self.size = 3*self.num_cards+3
        self.state = np.ones((1, self.size), dtype=int)
        self.reset()

    def reset(self):
        self.state = np.ones((1, self.size), dtype=int)
        self.state[0, 0:3] = np.zeros((1, 3), dtype=int) # Starting scores and value card are zero
        _ = self.draw_value_card()
        return

    def draw_value_card(self):
        remaining_value_cards = np.nonzero(self.state[0, 3:self.num_cards + 3])[0]
        if len(remaining_value_cards) == 0:
            self.state[0, 2] = 0
            return True

        chosen_value_card = np.random.choice(remaining_value_cards, 1)+1 # Index convention

        self.state[0, 2] = chosen_value_card
        self.state[0, chosen_value_card+2] = 0
        return False

    def step(self, action: int, action_opp: int):
        self.state[0, action+self.num_cards+2] = 0
        self.state[0, action_opp+self.num_cards*2+2] = 0

        self.state[0, 0:2] += self.state[0, 2]*np.array([action > action_opp, action < action_opp])

        done = self.draw_value_card()
        return self.state.copy(), done # Copy is necessary I think

    def get_legal_moves(self, player):
        cards_idx = np.nonzero(self.state[0, self.num_cards*player+3:self.num_cards*(player+1)+3])[0]
        return cards_idx + 1


def game_loop(game, agent1, agent2, num_games):
    for _ in range(num_games):
        done = False
        s_ = game.state.copy()
        while not done:
            s = s_
            a1, a2 = agent1.get_action(s, game.get_legal_moves(1)), \
                agent2.get_action(s, game.get_legal_moves(2))
            s_, done = game.step(a1, a2)
            print(s, a1, a2, s_, done)
        print('Final scores are {}, {}'.format(game.state[0, 0], game.state[0, 1]))
        game.reset()
    return


class HumanAgent:
    def __init__(self, player):
        self.player = player

    def get_action(self, state, legal_moves):
        print(state)
        move = int(input("Choose a valid move:"))
        return move


class RandomAgent:
    def __init__(self, player):
        self.player = player

    def get_action(self, state, legal_moves):
        return np.random.choice(legal_moves)


H1 = HumanAgent(1)
H2 = HumanAgent(2)
G = GOPS()
game_loop(G, H1, H2, 1)
