"""
Payoffs and probabilities for war probability model.

Bargaining utilities are a function of 1) military strength and 2) territory. Military strength is a function
of size of army and navy. Specifically, let r_i be player i's military strength and t_i a binary vector indicating
which territories are theirs. Then

u(r_1, r_2, t_1) = beta_11*r_1 + beta_12*(r_1 - r_2) + \sum_j beta_1j t_1j.


"""
import numpy as np
from scipy.optimize import Bounds, minimize


def military_strength(my_army, my_navy):
  """

  :param my_army: In number of 100's of thousands of troops (~2-10)
  :param my_navy: In number of ships (~1-30)
  :return:
  """
  return np.log(my_army) + 0.1*np.log(my_navy)


def bargaining_utility(my_army, your_army, my_navy, your_navy, my_territory, beta):
  """

  :param my_territory: Binary vector indicating posession of each territory (your_territory = 1 - my_territory)
  :return:
  """
  # Military strength component
  my_military_strength = military_strength(my_army, my_navy)
  your_military_strength = military_strength(your_army, your_navy)

  # Get features of utility function
  utility_features = np.concatenate(([my_military_strength, my_military_strength - your_military_strength],
                                      my_territory))

  return np.dot(utility_features, beta)


def nash_war_bargaining(max_army_1, max_army_2, max_navy_1, max_navy_2, disagreement_1, disagreement_2, beta_1, beta_2):
  """

  :param max_army_1:
  :param max_army_2:
  :param disagreement_1:
  :param disagreement_2:
  :param beta_1:
  :param beta_2:
  :return:
  """
  def nash_welfare(army_1, army_2, navy_1, navy_2, territory):
    utility_1 = bargaining_utility(army_1, army_2, navy_1, navy_2, territory, beta_1) - disagreement_1
    utility_2 = bargaining_utility(army_2, army_1, navy_2, navy_1, 1 - territory, beta_2) - disagreement_2
    return np.log(utility_1) + np.log(utility_2)

  def nash_welfare_wrapper(x):
    return nash_welfare(x[0], x[1], x[2], x[3], x[4])

  bounds = Bounds([0, max_army_1], [0, max_army_2], [0, max_navy_1], [0, max_navy_2], [0, 1], [0, 1], [0, 1])
  res = minimize(nash_welfare)



