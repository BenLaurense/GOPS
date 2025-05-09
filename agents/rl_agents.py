import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from agents.basic_agents import GOPSAgent
from utils import get_legal_moves


class PolicyNetAgent(nn.Module, GOPSAgent):
    """
    Simple policy net, consisting of several fully-connected Tanh layers.
    Input is a vector representing game state, output is a torch.distributions.Categorical
    object representing a distribution over available moves
    """
    def __init__(self,
                 num_cards: int,
                 widths: list[int] = [10, 4],
                 path=None):
        super().__init__()
        self.s_size, self.a_size = 3 * num_cards + 3, num_cards  # Player hands and value cards, and the current card and score
        self.layers = nn.Sequential(
            nn.Linear(self.s_size, widths[0]),
            nn.Tanh()
        )
        for i in range(len(widths) - 1):
            self.layers.append(nn.Linear(widths[i], widths[i + 1]))
            self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(widths[-1], self.a_size))
        self.layers.append(nn.Softmax(dim=1))

        if path is not None:
            self.load_state_dict(torch.load(path))
        return

    def _forward(self, state: np.ndarray) -> Categorical:
        state = torch.as_tensor(state, dtype=torch.float32)  # Environment is numpy-based; convert
        action_probs = self.layers(state)
        return Categorical(probs=action_probs)

    def get_action(self, state: np.ndarray, expl_rate: float = 0.0) -> tuple[int, float]:
        """
        Given a game state and returns an action, randomly chosen according to the
        policy network.

        Note: if an illegal action is chosen, picks a random legal action instead
        :param state: game state
        :param expl_rate: rate at which a random move is selected
        :return: the action, and its log-probability
        """
        cat = self._forward(state)
        # print(state, cat.probs.detach().numpy())
        action = cat.sample() + 1  # Indexing convention
        legal_actions = get_legal_moves(state, 1)

        explore = np.random.uniform(0, 1)
        # If the action is legal and we DON'T explore, take that action
        if action.item() in legal_actions and explore > expl_rate:
            return action.item(), cat.log_prob(action - 1)
        # Otherwise, take a random legal action
        action = torch.tensor(np.random.choice(legal_actions))
        return action.item(), cat.log_prob(action - 1)
