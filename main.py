import torch
import seaborn as sns
import matplotlib.pyplot as plt
from GOPS_environment import GOPS,game_loop
from BasicAgents import HumanAgent, RandomAgent


NUM_CARDS = 3
if __name__ == '__main__':
    # scores = train(NUM_CARDS, 10**4, max_lr=0.0001, min_lr=0.0001, lr_decay=10e-4,
    #                max_expl=1, min_expl=0.0001, expl_decay=10e-4,
    #                end_path='10card_naive_method3.pt')
    # plt.scatter(range(len(scores)), scores)
    # plt.ylim(-1, 1)
    # plt.show()

    G = GOPS(NUM_CARDS)
    A1 = HumanAgent(NUM_CARDS, 1)
    A2 = RandomAgent(NUM_CARDS,2)
    game_loop(G, A1, A2, 1, NUM_CARDS)

    # A2 = PolicyNet(NUM_CARDS)
    # bl, br = collect_data(G, A1, A2, 10, NUM_CARDS)
    # print(bl, len(bl))
    # print(br, len(br))
    #
    # optim = torch.optim.Adam(params=A1.parameters(), lr=0.1)
    #
    # pseudo_loss = -(torch.concat(bl, dim=0) * torch.as_tensor(br, dtype=torch.int32)).mean()
    # print(pseudo_loss)
    # optim.zero_grad()
    # pseudo_loss.backward()
