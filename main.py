import math
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from GOPS_environment import GOPS,game_loop
from BasicAgents import HumanAgent, RandomAgent
from vanilla_policy_gradient import PolicyNetAgent
from agent_utils import collect_data
from utils import sgn, flip_state
from tqdm import tqdm


NUM_CARDS = 5
if __name__ == '__main__':
    def simple_train(
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
            batch_logprobs, batch_rewards = collect_data(env, agent, opp, expl_rate, 5*num_cards)
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


    def profile(agent, agent2, num_games, num_cards):
        """
        Plays two agents against each other and collects the wins
        :param agent:
        :param agent2:
        :param num_games:
        :param num_cards:
        :return:
        """
        G = GOPS(num_cards)
        scores = []
        for _ in range(num_games):
            done = False
            s_ = G.state.copy()
            while not done:
                s = s_
                a1, _ = agent.get_action(s)
                a2, _ = agent2.get_action(flip_state(s))
                s_, done = G.step(a1, a2)
            scores.append(sgn(s_[0, 0] - s_[0, 1]))
            G.reset()
        return scores


    # num_epochs = 10**4
    # max_lr, min_lr, lr_decay = 0.3, 0.01, 0.0001
    # max_expl, min_expl, expl_decay = 0.3, 0.01, 0.001
    end_path = "blah.pt"
    # p, t, r, pt = simple_train(NUM_CARDS, num_epochs, max_lr, min_lr, lr_decay, max_expl, min_expl, expl_decay, end_path=end_path)
    # # p, t, r = [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]
    # fig, ax = plt.subplots(2, 2)
    # l = range(len(p))
    # ax[0, 0].plot(l, p, label="Profiling", c='b')
    # ax[0, 0].plot(l, t, label="Training", c='r')
    # ax[0, 1].plot(l, r, label="Reward", c='m')
    # ax[1, 0].plot(l, pt[0], label="p1")
    # ax[1, 0].plot(l, pt[1], label="p2")
    # ax[1, 0].plot(l, pt[2], label="p3")
    # ax[1, 0].plot(l, pt[3], label="p4")
    # fig.legend()
    # plt.show(block=True)

    env = GOPS(NUM_CARDS)
    M = PolicyNetAgent(NUM_CARDS, path=end_path)
    H = HumanAgent()
    game_loop(env, H, M, 1)
    # x = M.layers[0].state_dict()
    # print(x)
    # for key, val in M.state_dict().items():
    #     print(key, val.shape)

