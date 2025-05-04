import math
import numpy as np
import torch
from GOPS_environment import GOPS
from BasicAgents import GOPSAgent, RandomAgent
from RLAgents import PolicyNetAgent
from tqdm import tqdm
from training_utils import collect_trajectories, profile
from utils import sgn


def naive_policy_gradient(
        num_cards: int,
        num_epochs: int,
        lr: float,
        start_path: str = None,
        end_path: str = None):
    """
    Naive policy gradient algorithm.

    Directly approximates policy gradient by
    $$\del U(\theta)\approx\frac1{N}\sum_\tau R(\tau)\sum_{i=1}^m\del_\theta\log\pi_\theta(a_i^\tau|s_i^\tau)$$
    Where the outer sum is over N trajectories $\tau$, $R(\tau)$ is the overall reward of that trajectory,
    and $\pi_\theta$ is the neural network approximation of the policy
    :return:
    """
    env = GOPS(num_cards)
    agent, opp = PolicyNetAgent(num_cards, path=start_path), RandomAgent()
    optim = torch.optim.SGD(params=agent.parameters(), lr=lr)
    N = 20

    test_scores, train_scores = [], []
    for epoch in tqdm(range(1, num_epochs + 1)):
        # Collect data
        logprobs, scores = collect_trajectories(env, agent, opp, N * num_cards, 0.0)
        traj_total_logprobs = [torch.concat(t, dim=0).sum() for t in logprobs]
        traj_rewards = [s[-1] + 10 * sgn(s[-1]) for s in scores] # Reward is score differential plus a bonus for winning

        # Compute policy gradient estimate
        g_est = -torch.tensor([lp * r for lp, r in zip(traj_total_logprobs, traj_rewards)]).mean()

        # Step model
        optim.zero_grad()
        g_est.backward()
        optim.step()

        # Profile model every 100 epochs
        if not epoch % 100:
            if end_path is not None:
                torch.save(agent.state_dict(), end_path)
            # 'Training reward'
            train_score = profile(agent, opp, 100, num_cards)
            train_scores.append(train_score)
            # 'Testing reward'
            test_score = profile(agent, RandomAgent(), 100, num_cards)
            test_scores.append(test_score)
            # Update opponent
            opp = PolicyNetAgent(num_cards, path=end_path)
    return train_scores, test_scores