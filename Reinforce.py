
# def update(self):
#     """Updates the policy network's weights."""
#     running_g = 0
#     gs = []
#
#     # Discounted return (backwards) - [::-1] will return an array in reverse
#     for R in self.rewards[::-1]:
#         running_g = R + self.gamma * running_g
#         gs.insert(0, running_g)
#
#     deltas = torch.tensor(gs)
#
#     log_probs = torch.stack(self.probs)
#
#     # Calculate the mean of log probabilities for all actions in the episode
#     log_prob_mean = log_probs.mean()
#
#     # Update the loss with the mean log probability and deltas
#     # Now, we compute the correct total loss by taking the sum of the element-wise products.
#     loss = -torch.sum(log_prob_mean * deltas)
#
#     # Update the policy network
#     self.optimizer.zero_grad()
#     loss.backward()
#     self.optimizer.step()
#
#     # Empty / zero out all episode-centric/related variables
#     self.probs = []
#     self.rewards = []