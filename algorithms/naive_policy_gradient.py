import numpy as np
import torch
from GOPS_environment import GOPS
from agents.basic_agents import RandomAgent
from agents.rl_agents import PolicyNetAgent
from tqdm import tqdm
from algorithms.utils.training_utils import collect_trajectories, profile
from utils import sgn

NUM_MONTE_CARLO_SAMPLES = 20
PROFILING_FREQ = 50


def naive_policy_gradient(
        num_cards: int,
        num_epochs: int,
        lr: float,
        start_path: str = None,
        end_path: str = None,
        reward_lookback: bool = True,
        widths: list[int] = [20, 10]):
    """
    Naive policy gradient algorithm.

    Directly approximates policy gradient by
    $$\del U(\theta)\approx\frac1{N}\sum_\tau R(\tau)\sum_{i=1}^m\del_\theta\log\pi_\theta(a_i^\tau|s_i^\tau)$$
    Where the outer sum is over N trajectories $\tau$, $R(\tau)$ is the overall reward of that trajectory,
    and $\pi_\theta$ is the neural network approximation of the policy
    :return:
    """
    env = GOPS(num_cards)
    agent, opp = PolicyNetAgent(num_cards, widths=widths, path=start_path), RandomAgent()
    optim = torch.optim.SGD(params=agent.parameters(), lr=lr)
    print(agent.layers)

    test_scores, train_scores = [], []
    param_traces = [[], [], [], []]
    for epoch in tqdm(range(1, num_epochs + 1)):
        # Collect data
        logprobs, scores = collect_trajectories(env, agent, opp, NUM_MONTE_CARLO_SAMPLES * num_cards, 0.0)
        traj_total_logprobs = [torch.concat(t, dim=0).sum() for t in logprobs]
        if reward_lookback:
            traj_rewards = [simple_reward(s) for s in scores]
        else:
            traj_rewards = [no_lookback_reward(s) for s in scores]

        # Compute policy gradient estimator
        g_est = 0
        for i in range(NUM_MONTE_CARLO_SAMPLES):
            g_est -= traj_rewards[i] * traj_total_logprobs[i]
        g_est /= (NUM_MONTE_CARLO_SAMPLES * num_cards)
        print(g_est)

        # Step model
        optim.zero_grad()
        g_est.backward()
        optim.step()

        # Profile model every 100 epochs
        if not epoch % PROFILING_FREQ:
            if end_path is not None:
                torch.save(agent.state_dict(), end_path)

            with torch.no_grad():
                # 'Training reward'
                train_score = np.mean(profile(agent, opp, 100, num_cards))
                train_scores.append(train_score)
                # 'Testing reward'
                test_score = np.mean(profile(agent, RandomAgent(), 100, num_cards))
                test_scores.append(test_score)

            # Collect param traces
            p1 = agent.layers[0].state_dict()['weight'][0, 0].item()
            p2 = agent.layers[0].state_dict()['weight'][2, 4].item()
            p3 = agent.layers[2].state_dict()['bias'][1].item()
            p4 = agent.layers[2].state_dict()['weight'][1, 2].item()
            param_traces[0].append(p1)
            param_traces[1].append(p2)
            param_traces[2].append(p3)
            param_traces[3].append(p4)

            # Update opponent
            opp = PolicyNetAgent(num_cards, widths=widths, path=end_path)
    return train_scores, test_scores, param_traces
