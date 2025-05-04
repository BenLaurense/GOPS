import math
import numpy as np
import torch
from GOPS_environment import GOPS
from BasicAgents import GOPSAgent, RandomAgent
from RLAgents import PolicyNetAgent
from tqdm import tqdm
from training_utils import collect_trajectories, profile
from utils import sgn

"""
Basic ML benchmark: vanilla policy gradient.
Idea is to learn a map state --> distribution over actions and directly perform GD on this map
Difficulty is sparse reward - we want to train the agent to win, but then it gets no reward feedback
until the end of the game. Several ways to propagate reward.
    - Easiest is a simple time-discount rate parameter;
    - Could integrate the current turn score
Choose the former for now
"""

def vanilla_policy_gradient(
        num_cards: int,
        num_epochs: int,
        max_lr: float, min_lr: float, lr_decay: float,
        max_expl: float, min_expl: float, expl_decay: float,
        start_path: str = None, end_path: str = None
):
    """
    Training loop with learning rate deca, and epsilon-greedy exploration
    :param num_cards:
    :param num_epochs:
    :param max_lr:
    :param min_lr:
    :param lr_decay:
    :param max_expl:
    :param min_expl:
    :param expl_decay:
    :param start_path:
    :param end_path:
    :return:
    """
    env = GOPS(num_cards)
    agent, opp = PolicyNetAgent(num_cards, path=start_path), PolicyNetAgent(num_cards, path=start_path)
    profiling_agent = RandomAgent()
    avg_profiling_scores, avg_training_scores, rewards = [], [], []
    param_traces = [[], [], [], []]

    optim = torch.optim.SGD(params=agent.parameters(), lr=max_lr)
    expl_rate = max_expl
    for epoch in tqdm(range(1, num_epochs + 2)):
        # Collect training data
        batch_logprobs, batch_rewards = collect_trajectories(env, agent, opp, expl_rate, 5 * num_cards)
        # Compute pseudo-loss
        pseudo_loss = -(
                torch.concat(batch_logprobs, dim=0) * torch.as_tensor(batch_rewards, dtype=torch.int32)).mean()
        # Step model
        optim.zero_grad()
        pseudo_loss.backward()
        optim.step()

        # Save model each epoch
        if end_path is not None:
            torch.save(agent.state_dict(), end_path)

        # Update opponent model every x epochs, decay lr, decay exploration rate, and save model
        if not epoch % 200:
            opp.load_state_dict(torch.load(end_path))
            optim.param_groups[0]["lr"] = min_lr + (max_lr - min_lr) * math.exp(-lr_decay * epoch)
            expl_rate = min_expl + (max_expl - min_expl) * math.exp(-expl_decay * epoch)

        # Profile every y epochs
        if not epoch % 500:
            # Collect param traces
            p1 = agent.layers[0].state_dict()['weight'][0, 0].item()
            p2 = agent.layers[0].state_dict()['weight'][2, 4].item()
            p3 = agent.layers[2].state_dict()['bias'][1].item()
            p4 = agent.layers[2].state_dict()['weight'][1, 2].item()
            param_traces[0].append(p1)
            param_traces[1].append(p2)
            param_traces[2].append(p3)
            param_traces[3].append(p4)

            # Collect training rewards
            avg_reward = np.average(batch_rewards)
            rewards.append(avg_reward)

            # Profile against the opponent
            training_scores = profile(agent, opp, 100, num_cards)
            avg_training_score = np.average(training_scores)
            avg_training_scores.append(avg_training_score)

            # Profile against random agent
            profiling_scores = profile(agent, profiling_agent, 100, num_cards)
            avg_profiling_score = np.average(profiling_scores)
            avg_profiling_scores.append(avg_profiling_score)

            print("=" * 40)
            print("Epoch {}: avg reward: {}".format(epoch, avg_reward))
            print("Avg training score: {}. Avg profiling score: {}".format(avg_training_score, avg_profiling_score))
            print("Current lr: {}".format(optim.param_groups[0]["lr"]))
            print("Current expl rate: {}".format(expl_rate))

    return avg_profiling_scores, avg_training_scores, rewards, param_traces
