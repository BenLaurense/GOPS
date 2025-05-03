import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm
from GOPS_environment import *


"""
Basic ML benchmark: vanilla policy gradient.
Idea is to learn a map state --> action and directly perform GD on this map
difficulty is sparse reward - we want to train the agent to win, but then it gets no reward feedback
until the end of the game. Several ways to propagate reward.
    - Easiest is a simple time-discount rate parameter;
    - Could integrate the current turn score
Choose the former for now

NN representing a function current state --> distribution over actions
"""


class PolicyNet(nn.Module):
    def __init__(self, num_cards, widths=[10, 10, 4, 4], path=None):
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

    def get_action(self, state, legal_actions, expl_rate=0):
        cat = self.forward(state)
        action_raw = cat.sample()
        action = action_raw.item()+1    # Indexing convention

        explore = np.random.uniform(0, 1)
        if explore < expl_rate:
            # Do a random action
            return np.random.choice(legal_actions), cat.log_prob(action_raw)
        if action in legal_actions:
            # print("blah blah blah {} {}".format(state, cat.probs))
            return action, cat.log_prob(action_raw)
        # If action was illegal, return a random legal action
        return np.random.choice(legal_actions), cat.log_prob(action_raw)


"""
Training loop
    - collect_data collects data from a pair of agents
    - train calls collect_data, performs a gradient step, and 
"""


def collect_data(game, agent1, agent2, expl_rate, num_episodes, num_cards):
    num_games = num_episodes//num_cards

    # For VPG, need to collect the episode final reward, the episode action logprobs,
    batch_logprobs = []
    batch_rewards = []

    for _ in range(num_games):
        done = False
        s_ = game.state.copy()
        r = []
        while not done:
            s = s_
            a1, a1_logprob = agent1.get_action(s, game.get_legal_moves(1), expl_rate)
            a2, _ = agent2.get_action(flip_state(s, num_cards), game.get_legal_moves(2), expl_rate)
            s_, done = game.step(a1, a2)

            r.append(s_[0, 0] - s_[0, 1])
            batch_logprobs.append(a1_logprob)

        # Subject to tweaking lol
        tweaked_reward = list(map(lambda x: x + 100*sgn(s_[0, 0] - s_[0, 1]), r))
        batch_rewards += tweaked_reward  # Or other functions?

        game.reset()
    return batch_logprobs, batch_rewards


def train(num_cards, num_epochs, max_lr, min_lr, lr_decay, max_expl, min_expl, expl_decay, start_path=None, end_path=None):
    # Training scheme with weight decay, and epsilon-greedy exploration

    G = GOPS(num_cards)
    A_active, A_opp = PolicyNet(num_cards, path=start_path), PolicyNet(num_cards, path=start_path)
    Profiling_opp = RandomAgent(2)
    profiling_scores = []

    optim = torch.optim.Adam(params=A_active.parameters(), lr=max_lr)
    expl_rate = max_expl

    for epoch in range(num_epochs+1):
        # Collect training data
        batch_logprobs, batch_rewards = collect_data(G, A_active, A_opp, expl_rate, 10, num_cards)   #PARAM??
        pseudo_loss = -(torch.concat(batch_logprobs, dim=0) *
                        torch.as_tensor(batch_rewards, dtype=torch.int32)).mean()

        # Step model
        optim.zero_grad()
        pseudo_loss.backward()
        optim.step()

        # Save model each epoch
        if end_path is not None:
            torch.save(A_active.state_dict(), end_path)

        # Update opponent model every x epochs, decay lr, decay exploration rate, and save model
        if not epoch % 200:
            A_opp.load_state_dict(torch.load(end_path))
            optim.lr = min_lr + (max_lr-min_lr)*math.exp(-lr_decay*epoch)
            expl_rate = min_expl + (max_expl-min_expl)*math.exp(-expl_decay*epoch)
            if end_path is not None:
                torch.save(A_active.state_dict(), end_path)

        # And profile every y epochs
        if not epoch % 200:
            profiling = profile(A_active, Profiling_opp, 100, num_cards)
            avg_score = np.average(profiling)
            print("Epoch {}: profiling score: {}".format(epoch, avg_score))
            print("Current lr: {}".format(optim.lr))
            print("Current expl rate: {}".format(expl_rate))
            profiling_scores.append(avg_score)

    return profiling_scores


"""
Profiling the model
"""


def profile(agent, agent2, num_games, num_cards):
    # agent plays games against random agent
    G = GOPS(num_cards)

    scores = []
    for _ in range(num_games):
        done = False
        s_ = G.state.copy()
        while not done:
            s = s_
            a1, _ = agent.get_action(s, G.get_legal_moves(1))
            a2, _ = agent2.get_action(flip_state(s, num_cards), G.get_legal_moves(2))
            s_, done = G.step(a1, a2)
        scores.append(sgn(s_[0, 0] - s_[0, 1]))
        G.reset()
    return scores



