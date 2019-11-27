"""
Payoffs and probabilities for war probability model.

Bargaining utilities are a function of 1) military strength and 2) territory. Military strength is a function
of size of army and navy. Specifically, let r_i be player i's military strength and t_i a binary vector indicating
which territories are theirs. Then

u(r_1, r_2, t_1) = beta_11*r_1 + beta_12*(r_1 - r_2) + \sum_j beta_1j t_1j.


"""
import numpy as np
from scipy.optimize import Bounds, minimize
import nashpy as nash


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
  x0 = np.array([max_army_1, max_army_2, max_navy_1, max_navy_2, 0.5, 0.5, 0.5])
  res = minimize(nash_welfare_wrapper, x0=x0, method='trust-constr', bounds=bounds)
  return res.x


def mixed_strategy_disagreement(max_army_1, max_army_2, max_navy_1, max_navy_2, territory, beta_1, beta_2,
                                loss_coef_1, loss_coef_2, mutual_defect_penalty_1, mutual_defect_penalty_2):
  """

  :param max_army_1:
  :param max_army_2:
  :param max_navy_1:
  :param max_navy_2:
  :return:
  """
  STRONGER_POWER_WIN_PROB = 0.64

  # Get win probs
  military_strength_1 = military_strength(max_army_1, max_navy_1)
  military_strength_2 = military_strength(max_army_2, max_navy_2)
  win_prob_1 = STRONGER_POWER_WIN_PROB*(military_strength_1 > military_strength_2) + \
               (1-STRONGER_POWER_WIN_PROB)*(military_strength_2 >= military_strength_1)
  win_prob_2 = 1 - win_prob_1

  # Get payoff matrix
  # Payoffs for mutual cooperation
  CC_1 = bargaining_utility(max_army_1, max_army_2, max_navy_1, max_navy_2, territory, beta_1)
  CC_2 = bargaining_utility(max_army_2, max_army_1, max_navy_2, max_navy_1, 1 - territory, beta_2)

  # Payoffs for 1 defect, 1 cooperate
  post_loss_diff_1 = loss_coef_1*military_strength_1 - military_strength_2
  post_loss_diff_2 = loss_coef_2*military_strength_2 - military_strength_1

  # Being defected against
  military_losses_1 = np.array([loss_coef_1*military_strength_1, post_loss_diff_1])
  military_losses_2 = np.array([loss_coef_2*military_strength_2, post_loss_diff_2])
  CD_1 = np.dot(beta_1[:2], military_losses_1)
  DC_2 = np.dot(beta_2[:2], military_losses_2)

  # Defecting against
  military_gains_2 = np.array([max_army_2, -post_loss_diff_1])
  military_gains_1 = np.array([max_army_1, -post_loss_diff_2])
  CD_2 = np.dot(beta_2[:2], military_gains_2)
  DC_1 = np.dot(beta_1[:2], military_gains_1)

  # Payoffs for mutual defection
  expected_military_power_1 = win_prob_1*military_gains_1 + (1 - win_prob_1)*military_losses_1
  expected_territory_1 = win_prob_1*np.ones(3)
  expected_military_power_2 = win_prob_2*military_gains_2 + (2 - win_prob_2)*military_losses_2
  expected_territory_2 = win_prob_2*np.ones(3)
  DD_1 = np.dot(beta_1, np.concatenate((expected_military_power_1, expected_territory_1))) + mutual_defect_penalty_1
  DD_2 = np.dot(beta_2, np.concatenate((expected_military_power_2, expected_territory_2))) + mutual_defect_penalty_2

  # Collect into payoff matrices
  player_1_payoffs = [[CC_1, CD_1], [DC_1, DD_1]]
  player_2_payoffs = [[CC_2, CD_2], [DC_2, DD_2]]
  display_matrix = [[(CC_1, CC_2), (CD_1, CD_2)], [(DC_1, DC_2), (DD_1, DD_2)]]
  print(display_matrix)

  # Compute and display equilibria
  game = nash.Game(player_1_payoffs, player_2_payoffs)

  # ToDo: assuming last eq is mixed!
  game_values = [game[eq] for eq in game.support_enumeration()][-1]
  game_profiles = [eq for eq in game.support_enumeration()]
  print(game_profiles)
  return game_values


if __name__ == "__main__":
  ALLIANCE_ARMY = 6 + 3.6 + 2.2
  ALLIANCE_NAVY = 24 + 7 + 3
  ENTENTE_ARMY = 11 + 2.1 + 5.8
  ENTENTE_NAVY = 17 + 39 + 22
  territory = np.ones(3)
  loss_coef_1 = 0.1
  loss_coef_2 = 0.1
  mutual_defect_penalty_1 = -30
  mutual_defect_penalty_2 = -30
  beta_1 = np.ones(5)
  beta_2 = np.ones(5)
  mixed_strategy_disagreement(ALLIANCE_ARMY, ENTENTE_ARMY, ALLIANCE_NAVY, ENTENTE_NAVY, territory, beta_1, beta_2,
                              loss_coef_1, loss_coef_2, mutual_defect_penalty_1, mutual_defect_penalty_2)

