import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
import pdb


"""
Alternating offers in a bargaining game with categorical actions a and outcomes for players 1, 2 (u1, u2). 
"""


def compute_loss(probs, rewards, gamma=0.99):
  tail = 0.
  scaled_rewards = []
  for reward in rewards[::-1]:
    tail = reward + gamma*tail
    scaled_rewards.insert(0, tail)
  loss = 0.
  for p, r in zip(probs, scaled_rewards):
    loss += p * r
  loss *= -1
  return loss


class PGLearner(nn.Module):
  """
  Stealing from http://bytepawn.com/solving-openai-gym-classic-control-problems-with-pytorch.html
  """
  def __init__(self, input_length, num_actions):
    super(PGLearner, self).__init__()
    self.num_actions = num_actions
    self.output_length = self.num_actions + 2
    self.fc1 = nn.Linear(input_length, 16)
    self.fc2 = nn.Linear(16, 16)
    self.fc3 = nn.Linear(16, self.output_length)

  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc3(x)
    x = F.softmax(x)
    return x

  def action(self, x):
    """
    :param x: array of form [(a^t, u1^t, u2^t)_{t=1}^T | offer]
    """
    x = T.from_numpy(x).double().unsqueeze(0)
    probs = self.forward(x)
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)


def train(counterpart_policy, n=2, T=100, num_episodes=1000):
  # Train PGLearner against counterpart_policy
  num_actions = 3
  expected_payoffs_1 = np.array([1, 2, 3])
  expected_payoffs_2 = np.array([5, 4, 3])

  input_size = n*3*3 + 1
  policy = PGLearner(input_size, len(expected_payoffs_1))
  policy.double()
  optimizer = optim.Adam(policy.parameters(), lr=0.01)
  episode_rewards = []
  loss = 0.
  for episode in range(num_episodes):
    probs, rewards, states = [], [], []
    offer = np.random.choice(range(num_actions), p=np.ones(num_actions)/num_actions)
    history = []
    for i in range(n):
      for j in range(3):
        history.append(j)
        history.append(expected_payoffs_1[j] + np.random.normal())
        history.append(expected_payoffs_2[j] + np.random.normal())

    done = False
    for t in range(T):
      state = np.concatenate((history, [offer]))
      if t % 2 == 0:  # PGLearner takes turns on even steps
        action, prob = policy.action(state)
        print(action)
      else:
        action, _ = counterpart_policy(state, policy.num_actions)

      if action == num_actions + 1:  # Accept
        reward = expected_payoffs_1[offer]
        done = True
      elif action == num_actions + 2:  # Reject
        reward = 0.
        done = True
      else:  # Counteroffer
        reward = 0.
      rewards.append(reward)
      probs.append(prob)

      if done:
        break
      else:
        offer = action

    episode_reward = np.sum(rewards)
    episode_rewards.append(episode_reward)
    loss += compute_loss(probs, rewards)
    if episode % 10 == 0:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss = 0
  return policy


if __name__ == "__main__":

  def dummy_counterpart_policy(x, num_actions):
    if x[-1] == 0:
      return num_actions + 1, 1.
    else:
      return num_actions + 2, 1.

  train(dummy_counterpart_policy, n=2, T=100, num_episodes=1000)













