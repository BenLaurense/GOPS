import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from BasicAgents import GOPSAgent
from agent_utils import get_legal_moves

"""
Basic ML benchmark: vanilla policy gradient.
Idea is to learn a map state --> distribution over actions and directly perform GD on this map
Difficulty is sparse reward - we want to train the agent to win, but then it gets no reward feedback
until the end of the game. Several ways to propagate reward.
    - Easiest is a simple time-discount rate parameter;
    - Could integrate the current turn score
Choose the former for now
"""


class PolicyNetAgent(nn.Module, GOPSAgent):
    def __init__(self, num_cards: int, widths: list[int]=[10, 10, 4, 4], path=None):
        super().__init__()

        self.s_size, self.a_size = 3*num_cards+3, num_cards  # Player hands and value cards, and the current card and score

        self.layers = nn.Sequential(
            nn.Linear(self.s_size, widths[0]),
            nn.ReLU()
        )
        for i in range(len(widths) - 1):
            self.layers.append(nn.Linear(widths[i], widths[i + 1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(widths[-1], self.a_size))
        self.layers.append(nn.Softmax(dim=1))

        if path is not None:
            self.load_state_dict(torch.load(path))
        return

    def forward(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)  # Environment is numpy-based; convert
        action_probs = self.layers(state)
        return Categorical(probs=action_probs)

    def get_action(self, state, expl_rate=0):
        cat = self.forward(state)
        action_raw = cat.sample()
        action = action_raw.item() + 1    # Indexing convention
        legal_actions = get_legal_moves(state, 1)

        explore = np.random.uniform(0, 1)
        if explore < expl_rate:
            # Do a random action
            return np.random.choice(legal_actions), cat.log_prob(action_raw)
        if action in legal_actions:
            # print("blah blah blah {} {}".format(state, cat.probs))
            return action, cat.log_prob(action_raw)
        # If action was illegal, return a random legal action
        return np.random.choice(legal_actions), cat.log_prob(action_raw)
