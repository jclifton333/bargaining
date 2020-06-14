import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical
import pdb


"""
Alternating offers in a bargaining game with categorical actions a and outcomes for players 1, 2 (u1, u2). 
"""

# ToDo: next level of complication should have utilities depend on context (and therefore actions on
# ToDo: potentially high-dimensional regression model


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


# ToDo: PGLearner superclass
class WelfarePGLearner(nn.Module):
  def __init__(self, input_length, num_actions):
    super(WelfarePGLearner, self).__init__()
    self.num_actions = num_actions
    self.output_length = self.num_actions + 2
    self.fc1 = nn.Linear(input_length, 10)
    self.fc5 = nn.Linear(10, 10)
    self.fc2 = nn.Linear(10, num_actions)
    self.fc3 = nn.Linear(num_actions+1, 1)
    self.fc4 = nn.Linear(2, self.output_length)
    self.episode_rewards = []
    self.probs = []
    self.rewards = []

  def reset(self):
    self.probs = []
    self.rewards = []

  def forward(self, x):
    y = self.fc1(x)
    y = F.relu(y)
    y = self.fc5(y)
    y = F.relu(y)
    reward_model = self.fc2(y)
    reward_model = F.softmax(reward_model)
    y = T.cat((reward_model[0, :], x[:, -1]))
    y = self.fc3(y)
    y = F.relu(y)
    y = T.cat((y, x[:, -1]))
    y = self.fc4(y)
    y = F.softmax(y)
    return y

  def action(self, x):
    """
    :param x: array of form [(a^t, u1^t, u2^t)_{t=1}^T | offer]
    """
    x = T.from_numpy(x).double().unsqueeze(0)
    probs = self.forward(x)
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)

  def step(self, x, payoffs):
    action, p = self.action(x)
    offer = x[-1]
    if action == self.num_actions:  # Accept
      if offer < self.num_actions:
        reward = payoffs[int(offer)]
      else:
        reward = 0
      done = True
    elif action == self.num_actions + 1:  # Reject
      reward = 0.
      done = True
    else:  # Counteroffer
      reward = 0.
      done = False
    self.rewards.append(reward)
    self.probs.append(p)
    return action, p, reward, done


class PGLearner(nn.Module):
  def __init__(self, input_length, num_actions):
    super(PGLearner, self).__init__()
    self.num_actions = num_actions
    self.output_length = self.num_actions + 2
    self.fc1 = nn.Linear(input_length, 100)
    # self.fc1 = nn.Linear(num_actions, 48)
    self.fc2 = nn.Linear(100, 100)
    self.fc3 = nn.Linear(100, self.output_length)
    self.episode_rewards = []
    self.probs = []
    self.rewards = []

  def reset(self):
    self.probs = []
    self.rewards = []

  def forward(self, x):
    # x = T.from_numpy(util_each_action(x, self.num_actions))  # Give it some help
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

  def step(self, x, payoffs):
    action, p = self.action(x)
    offer = x[-1]
    if action == self.num_actions:  # Accept
      if offer < self.num_actions:
        reward = payoffs[int(offer)]
      else:
        reward = 0
      done = True
    elif action == self.num_actions + 1:  # Reject
      reward = 0.
      done = True
    else:  # Counteroffer
      reward = 0.
      done = False
    self.rewards.append(reward)
    self.probs.append(p)
    return action, p, reward, done


def train(policy_class=PGLearner, n=2, T=100, num_episodes=100000):
  # Train PGLearner against PGLearner
  num_actions = 3

  input_size = n*3*3 + 1
  policy1 = policy_class(input_size, num_actions)
  policy1.double()
  policy2 = policy_class(input_size, num_actions)
  policy2.double()

  optimizer1 = optim.Adam(policy1.parameters(), lr=0.01)
  optimizer2 = optim.Adam(policy2.parameters(), lr=0.01)
  efficient_list = []
  efficient_mean_list = []
  loss1 = 0
  loss2 = 0

  for episode in range(num_episodes):
    offer = num_actions + 1
    history = []
    expected_payoffs_1 = np.random.poisson(lam=5, size=num_actions)
    expected_payoffs_2 = np.random.poisson(lam=5, size=num_actions)
    for i in range(n):
      for j in range(num_actions):
        history.append(j)
        history.append(expected_payoffs_1[j] + np.random.normal(scale=2))
        history.append(expected_payoffs_2[j] + np.random.normal(scale=2))

    for t in range(T):
      state = np.concatenate((history, [offer]))
      if t % 2 == 0:  # Take turns
        action, p, reward, done = policy1.step(state, expected_payoffs_1)
      else:
        action, p, reward, done = policy2.step(state, expected_payoffs_2)
      if done:
        break
      else:
        offer = action

    if action == num_actions + 1 or offer > num_actions:
      efficient = False
    else:
      offer = int(offer)
      u1, u2 = expected_payoffs_1[offer], expected_payoffs_2[offer]
      for u1_, u2_ in zip(expected_payoffs_1, expected_payoffs_2):
        if u1_ > u1 and u2_ > u2:
          efficient = False
          break
        else:
          efficient = True
    efficient_list.append(efficient)
    efficient_pct = np.mean(efficient_list[-10:])
    efficient_mean_list.append(efficient_pct)
    print(t, efficient_pct)

    loss1 += compute_loss(policy1.probs, policy1.rewards)
    loss2 += compute_loss(policy2.probs, policy2.rewards)
    if episode % 10 == 0:
      optimizer1.zero_grad()
      if loss1 > 0:
        loss1.backward()
        optimizer1.step()
      loss1 = 0
      optimizer2.zero_grad()
      if loss2 > 0:
        loss2.backward()
        optimizer2.step()
      loss2 = 0
    policy1.reset()
    policy2.reset()

  return policy1, policy2, efficient_mean_list


def util_each_action(x, num_actions):
  util = np.zeros(num_actions)
  j = 0
  for i, z in enumerate(x[:-1]):
    if i % num_actions != 0:
      util[j] += z
      j += 1
    if j >= num_actions:
      j = 0
  return util


if __name__ == "__main__":
  np.random.seed(1)
  ts_naive_mean = []
  ts_welfare_mean = []
  for rep in range(10):
    # _, _, ts_naive = train(policy_class=PGLearner, n=200, T=100, num_episodes=1000)
    _, _, ts_welfare = train(policy_class=WelfarePGLearner, n=200, T=100, num_episodes=2000)
    # ts_naive_mean.append(ts_naive)
    ts_welfare_mean.append(ts_welfare)
  # ts_naive_mean = np.mean(ts_naive_mean, axis=0)
  ts_welfare_mean = np.mean(ts_welfare_mean, axis=0)
  # plt.plot(ts_naive_mean, label='naive')
  plt.plot(ts_welfare_mean, label='welfare')
  plt.legend()
  plt.show()













