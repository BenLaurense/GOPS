import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


"""
Basic ML benchmark: simple policy network

NN representing a function current state --> distribution over actions
"""


class PolicyNet(nn.Module):
    def __init__(self, num_cards, widths=[8, 8], path=None):
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
        state = torch.tensor(state, dtype=torch.float32)  # Environment is numpy-based; convert
        return self.layers(state)

    def get_action(self, state, legal_actions):
        action_probs = self.forward(state)
        cat = Categorical(probs=action_probs)

        action = cat.sample().item()
        if action in legal_actions:
            return action
        # If action was illegal, return a random legal action
        return np.random.choice(legal_actions)
