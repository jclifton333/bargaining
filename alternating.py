import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
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
class BasicPGLearner(nn.Module):
  def __init__(self):
    super(BasicPGLearner, self).__init__()
    self.episode_rewards = []
    self.probs = []
    self.rewards = []

  def reset(self):
    self.probs = []
    self.rewards = []

  def step(self, x, payoffs):
    message_and_action, p = self.action(x)
    action = message_and_action[-1]
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
    return message_and_action, p, reward, done


class WelfarePGLearner(BasicPGLearner):
  def __init__(self, input_length, num_actions):
    super(WelfarePGLearner, self).__init__()
    self.num_actions = num_actions
    self.output_length = self.num_actions + 2
    self.fc1 = nn.Linear(2, 10)
    self.fc2 = nn.Linear(10, 10)
    self.fc3 = nn.Linear(10, self.output_length)

  def forward(self, x):
    y = self.fc1(x)
    y = F.relu(y)
    y = self.fc2(y)
    y = F.relu(y)
    y = self.fc3(y)
    y = F.softmax(y)
    return y

  def action(self, x):
    """
    :param x: array of form [(a^t, u1^t, u2^t)_{t=1}^T | offer]
    """
    argmax_utils = np.argmax(x[:-1])
    argmax_equals_offer = [argmax_utils == x[-1]]
    x = np.concatenate([argmax_equals_offer, [x[-1]]])
    x = T.from_numpy(x).double().unsqueeze(0)
    probs = self.forward(x)
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)


class ContinuousPGLearner(BasicPGLearner):
  """
  Output a continuous-valued message and a prob dbn over actions.
  """
  def __init__(self, message_length, num_actions):
    super(ContinuousPGLearner, self).__init__()
    self.num_actions = num_actions
    self.message_length = message_length
    self.fc1 = nn.Linear(self.message_length + 1, 100)
    self.fc2 = nn.Linear(100, 100)
    self.fc3 = nn.Linear(100, 2*self.message_length + self.num_actions + 2)

  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc3(x)
    message_means = x[:, :self.message_length]
    message_sds = x[:, self.message_length:(2*self.message_length)]
    message_sds = nn.Threshold(0.01, 0.01)(message_sds)
    action_logits = x[:, (2*self.message_length):]
    action_probs = F.softmax(action_logits)
    return message_means, message_sds, action_probs

  def action(self, x):
    # x = T.from_numpy(x).double().unsqueeze(0)
    x = x.double().unsqueeze(0)
    message_means, message_sds, action_probs = self.forward(x)
    action_dbn = Categorical(action_probs)
    action = action_dbn.sample()
    message_dbn = Normal(message_means, message_sds)
    message = message_dbn.sample()
    log_prob = action_dbn.log_prob(action) + message_dbn.log_prob(message).sum()
    x = T.cat((message[0, :], action.double()))
    return x, log_prob


class PGLearner(BasicPGLearner):
  def __init__(self, input_length, num_actions):
    super(PGLearner, self).__init__()
    self.num_actions = num_actions
    self.output_length = self.num_actions + 2
    self.fc1 = nn.Linear(num_actions+1, 100)
    # self.fc1 = nn.Linear(num_actions, 48)
    self.fc2 = nn.Linear(100, 100)
    self.fc3 = nn.Linear(100, self.output_length)

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


def util(u1, u2):
  return u1 + u2


def nash(u1, u2):
  return np.log(u1) + np.log(u2)


def welfare_wrapper(welfare_name, u1, u2):
  if welfare_name == 'util':
    return util(u1, u2)
  elif welfare_name == 'nash':
    return nash(u1, u2)
  elif welfare_name == 'random':
    welf = np.random.choice([util, nash])
    return welf(u1, u2)


def train(welfare1='util', welfare2='util', policy_class=PGLearner, policies=None, n=2, horizon=100,
          update_block_size=100, num_episodes=100000):
  # Train PGLearner against PGLearner
  num_actions = 3

  if policies is None:
    # input_size = n*num_actions*3 + 1
    message_length = num_actions
    policy1 = policy_class(message_length, num_actions)
    policy1.double()
    policy2 = policy_class(message_length, num_actions)
    policy2.double()

    optimizer1 = optim.Adam(policy1.parameters(), lr=0.01)
    optimizer2 = optim.Adam(policy2.parameters(), lr=0.01)
  else:
    policy1, policy2 = policies

  efficient_list = []
  efficient_mean_list = []
  cooperate_list = []
  cooperate_mean_list = []
  t_list = []
  loss1 = 0
  loss2 = 0

  for episode in range(num_episodes):
    offer = num_actions + 1.
    history = []
    welfare_means = np.zeros(num_actions)
    expected_payoffs_1 = np.random.poisson(lam=5, size=num_actions)
    expected_payoffs_2 = np.random.poisson(lam=5, size=num_actions)
    for i in range(n):
      for j in range(num_actions):
        history.append(j)
        payoff_1 = expected_payoffs_1[j] + np.random.normal(scale=2)
        payoff_2 = expected_payoffs_2[j] + np.random.normal(scale=2)
        history.append(payoff_1)
        history.append(payoff_2)
        welfare_means[j] += (payoff_1 + payoff_2)
    welfare_means /= n

    for t in range(horizon):
      # state = np.concatenate((history, [offer]))
      # state = np.concatenate((welfare_means, [offer]))
      welfare1_t = welfare_wrapper(welfare1, expected_payoffs_1, expected_payoffs_2)
      welfare2_t = welfare_wrapper(welfare2, expected_payoffs_1, expected_payoffs_2)
      # state1 = np.concatenate((welfare1_t, message_and_action))
      # state2 = np.concatenate((welfare2_t, message_and_action))
      message_and_action = T.ones(num_actions).double()
      state1 = T.cat((message_and_action, T.tensor([offer]).double()))
      state2 = T.cat((message_and_action, T.tensor([offer]).double()))
      # welfare_eq_1 = T.from_numpy(np.array([np.all(welfare1_t == message_and_action.detach().numpy())])).double()
      # state1 = T.cat((welfare_eq_1, message_and_action, T.tensor([offer]).double()))
      # welfare_eq_2 = T.from_numpy(np.array([np.all(welfare2_t == message_and_action.detach().numpy())])).double()
      # state2 = T.cat((welfare_eq_2, message_and_action, T.tensor([offer]).double()))
      if t % 2 == 0:  # Take turns
        message_and_action, p, reward, done = policy1.step(state1, expected_payoffs_1)
      else:
        message_and_action, p, reward, done = policy2.step(state2, expected_payoffs_2)
      if done:
        break
      else:
        offer = message_and_action[-1]

    if offer == num_actions + 1 or offer > num_actions or t == horizon-1:
      efficient = False
      cooperate = False
    else:
      cooperate = True
      offer = int(offer)
      u1, u2 = expected_payoffs_1[offer], expected_payoffs_2[offer]
      for u1_, u2_ in zip(expected_payoffs_1, expected_payoffs_2):
        if u1_ > u1 and u2_ > u2:
          efficient = False
          break
        else:
          efficient = True
    cooperate_list.append(cooperate)
    efficient_list.append(efficient)
    cooperate_pct = np.mean(cooperate_list[-10:])
    efficient_pct = np.mean(efficient_list[-10:])
    cooperate_mean_list.append(cooperate_pct)
    efficient_mean_list.append(efficient_pct)
    t_list.append(t)
    print(t, cooperate_pct)

    # Take turns updating policies
    if policies is None:
      if (episode // update_block_size) % 2 == 0:
        loss1 += compute_loss(policy1.probs, policy1.rewards)
        if episode % 10 == 0:
          optimizer1.zero_grad()
          if loss1 > 0:
            loss1.backward()
            optimizer1.step()
          loss1 = 0
      else:
        loss2 += compute_loss(policy2.probs, policy2.rewards)
        if episode % 10 == 0:
          optimizer2.zero_grad()
          if loss2 > 0:
            loss2.backward()
            optimizer2.step()
          loss2 = 0
    policy1.reset()
    policy2.reset()

  return policy1, policy2, cooperate_mean_list, t_list


if __name__ == "__main__":
  # ToDo: Implement messages
  # ToDo: reward agents for Pareto-optimal outcome?

  np.random.seed(1)
  n_rep = 10
  cooperate_ts_ur_train_mean = []
  cooperate_ts_nr_train_mean = []
  cooperate_ts_test_mean = []
  num_episodes = 1000
  for rep in range(n_rep):
    policy1_ur, policy2_ur, cooperate_ts_ur, _ = \
      train(policy_class=ContinuousPGLearner, n=100, horizon=100, update_block_size=300,
            num_episodes=num_episodes)
    cooperate_ts_ur_train_mean.append(cooperate_ts_ur)

  cooperate_ts_ur_train_mean = np.mean(cooperate_ts_ur_train_mean, axis=0)
  plt.plot(cooperate_ts_ur_train_mean, label='train_ur')
  plt.legend()
  plt.show()
