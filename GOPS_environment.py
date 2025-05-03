import numpy as np
from utils import flip_state, get_legal_moves


class GOPS:
    """
    Environment for GOPS
        - Players are numbered 1 and 2
        - State is represented by a 1xn binary array [scores | current value card | value cards | P1 cards | P2 cards]
        (it's a binary array except for current value card, which is an integer entry representing the card's value)
        - Actions are represented by the VALUE of the card being played (from 1 to num_cards)
    """
    def __init__(self, num_cards):
        self.num_cards = num_cards
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


def game_loop(game, agent1, agent2, num_games, num_cards):
    for _ in range(num_games):
        done = False
        s_ = game.state.copy()
        while not done:
            s = s_
            a1, _ = agent1.get_action(s)
            a2, _ = agent2.get_action(flip_state(s, num_cards))
            s_, done = game.step(a1, a2)
            # print(s, a1, a2, s_, done)
        s1, s2 = game.state[0, 0], game.state[0, 1]
        print('Final scores are {}, {}'.format(s1, s2))
        if s1 > s2:
            print("Player 1 won!")
        elif s1 < s2:
            print("Player 2 won!")
        else:
            print("Draw!")
        game.reset()
    return

