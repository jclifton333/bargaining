import numpy as np
import torch.nn as nn
import torch.nn.functional as F

"""
Alternating offers in a bargaining game with real-valued actions a and outcomes for players 1, 2 (x1, x2). 
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
  def __init__(self, input_length):
    super(PGLearner, self).__init__()
    self.fc1 = nn.Linear(input_length, 16)
    self.fc2 = nn.Linear(16, 16)
    self.fc3 = nn.Linear(16, 2)

  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc3(x)
    return x



