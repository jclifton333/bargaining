"""
Model:

Two resources y and z, such that agent i's utility function is
u_i(y, z) = a^y_i * y + a^z_i * z.

Each agent can make twice as many units of the resource they value less. Each has (the same) budget B, such that
they can make x units of the resource they value more and 2*(B - x) units of the resource they value less,
for x in {0, 1, ..., B}.

The bargaining solution is given by
argmax_{y, z} \int log{ a^y_t1 * (y_1 + y_2) + a^z_t1 * (z1 + z2) } dP(t1) # ToDo: currently lacks disagreement pt.
            + \int log{ a^y_t2 * (y_1 + y_2) + a^z_t2 * (z1 + z2) } dP(t2)

where P is the prior over types.
"""
import numpy as np
import matplotlib.pyplot as plt
import pdb


def log_utility(yi, ymi, zi, zmi, ay_i, ay_mi, az_i, az_mi, budget):
  """
  Player i's utility at a given amount of y and z.

  :param y1: Units of y created by player 1.
  :param y2: Units of y created by player 2.
  :param z1: Units of z created by player 1.
  :param z2: Units of z created by player 2.
  :param ay_i: Coefficient of y in i's utility function.
  :param az_i: Coefficient of z in i's utility function.
  :param ay_mi: Coefficient of y in -i's utility function.
  :param az_mi: Coefficient of z in -i's utility function.
  :return:
  """
  i_prefers_y = ay_i > az_i
  mi_prefers_y = ay_mi > az_mi
  utility_ = ay_i*(yi + ymi + yi*i_prefers_y + ymi*mi_prefers_y) \
             + az_i*(zi + zmi + zi*(1-i_prefers_y) + zmi*(1-mi_prefers_y))

  # Get the best players could do without trade, to form disagreement point
  # (Recall that players can make twice as much of the resource that they value less)
  value_from_is_unilateral_action = \
    budget*max((ay_i + ay_i*(ay_i <= az_i), az_i + az_i*(az_i < ay_i)))
  if mi_prefers_y:
    mi_chooses_y = 2*az_mi <= ay_mi
    value_from_mis_unilateral_action = budget*(ay_i*mi_chooses_y + 2*az_i*(1-mi_chooses_y))
  else:
    mi_chooses_y = 2*ay_mi > az_mi
    value_from_mis_unilateral_action = budget*(2*ay_i*mi_chooses_y + az_i*(1-mi_chooses_y))

  # Compute disagreement and surplus
  disagreement_point = value_from_is_unilateral_action + value_from_mis_unilateral_action
  surplus_utility = utility_ - disagreement_point
  if surplus_utility > 0:
    return np.log(surplus_utility + 1)
  else:
    return 0


def independent_gaussian_nash_welfare(y1, y2, z1, z2, mu_ay_1, mu_ay_2, mu_az_1, mu_az_2, budget, num_draws=1000):
  """
  Compute expected welfare at a given allocation for a single agent, where prior over types is given by
  independent Gaussians with means mu_ay, mu_az over coefficients ay, az.

  :param y1:
  :param y2:
  :param z1:
  :param z2:
  :param mu_ay:
  :param mu_az:
  :param num_draws:
  :return:
  """
  ay_1_prior_draws = np.random.normal(loc=mu_ay_1, size=num_draws)
  az_1_prior_draws = np.random.normal(loc=mu_az_1, size=num_draws)
  ay_2_prior_draws = np.random.normal(loc=mu_ay_2 , size=num_draws)
  az_2_prior_draws = np.random.normal(loc=mu_az_2, size=num_draws)

  player_1_welfare_at_each_draw = np.array([log_utility(y1, y2, z1, z2, ay_1, ay_2, az_1, az_2, budget)
                                            for ay_1, ay_2, az_1, az_2
                                            in zip(ay_1_prior_draws, ay_2_prior_draws, az_1_prior_draws,
                                                   az_2_prior_draws)])
  player_2_welfare_at_each_draw = np.array([log_utility(y2, y1, z2, z1, ay_2, ay_1, az_2, az_1, budget)
                                            for ay_2, ay_1, az_2, az_1
                                            in zip(ay_2_prior_draws, ay_1_prior_draws, az_2_prior_draws,
                                                   az_1_prior_draws)])

  # Check that action is feasible
  # ToDo: not quite right, need to exclude actions that are not feasible for some type
  if np.sum(player_2_welfare_at_each_draw == 0) < num_draws and np.sum(player_1_welfare_at_each_draw == 0) < num_draws:
    return np.mean(player_1_welfare_at_each_draw[np.where(player_1_welfare_at_each_draw > 0)]) + \
           np.mean(player_2_welfare_at_each_draw[np.where(player_2_welfare_at_each_draw > 0)])
  else:
    return -float('inf')


def nbs_for_independent_gaussan_priors(mu_ay_1, mu_ay_2, mu_az_1, mu_az_2, budget=10, num_draws=1000):
  """
  Compute NBS for independent gaussian priors over utility coefficients, when each player can create
  integer-valued quantities of each resource up to a budget.

  :param mu_ay_1:
  :param mu_ay_2:
  :param mu_az_1:
  :param mu_az_2:
  :param budget:
  :param num_draws:
  :return:
  """
  optimal_welfare = -float('inf')
  y1_opt = None
  y2_opt = None

  # Search over all allocations up to budget for each player.
  for y1 in range(budget + 1):
    z1 = budget - y1
    for y2 in range(budget + 1):
      z2 = budget - y2
      nash_welfare_ = independent_gaussian_nash_welfare(y1, y2, z1, z2, mu_ay_1, mu_ay_2, mu_az_1, mu_az_2, budget,
                                                        num_draws=num_draws)
      if nash_welfare_ > optimal_welfare:
        optimal_welfare = nash_welfare_
        y1_opt = y1
        y2_opt = y2

  return y1_opt, y2_opt, budget - y1_opt, budget - y2_opt


def inefficiency_from_gaussian_prior_differences(budget=10, num_draws=1000):
  """
  Measure the inefficiencies that arise from players using the bargaining solutions corresponding to different priors,
  when priors are independent gaussian.

  Priors are structured like so:
    - Variance of all the gaussians is 1.
    - For each player, mean of prior on 1) player 1's z coef and 2) player 2's y coef are the same.
    - For each player i, mean of prior on player 1's y coef is equal to player i's prior mean on player 2's
      z coef.
    - Player 1's prior mean on ay_1 and az_2 is held fixed.
    - Player 2's prior mean on ay_1 and az_2 varies; thus this is the only respect in which the priors vary, and
      the difference between mu_ay_1_1 and mu_ay_1_2 controls the discrepancy between the priors.

  :param budget:
  :param num_draws:
  :return:
  """

  # Specify player 1's prior
  mu_ay_1_1, mu_az_1_1 = 10, 1
  mu_ay_2_1 = mu_az_1_1
  mu_az_2_1 = mu_ay_1_1

  # Bargaining solution for 1's prior
  y1_opt_1, z1_opt_1, y2_opt_1, z2_opt_1 = \
    nbs_for_independent_gaussan_priors(mu_ay_1_1, mu_ay_2_1, mu_az_1_1, mu_az_2_1, budget=budget, num_draws=num_draws)

  # Bargaining solution for 2's prior, as mu_ay_1_2 varies
  prior_differences = np.linspace(-2, 2, 5)
  true_welfares = []
  for diff in prior_differences:
    print(diff)
    mu_ay_1_2 = mu_ay_1_1 + diff
    mu_az_1_2 = 1
    mu_ay_2_2 = mu_az_1_2
    mu_az_2_2 = mu_ay_1_2
    y1_opt_2, y2_opt_2, z1_opt_2, z2_opt_2 = \
      nbs_for_independent_gaussan_priors(mu_ay_1_2, mu_ay_2_2, mu_az_1_2, mu_az_2_2, budget=budget, num_draws=num_draws)

    # Compute true welfare, using player 1's prior, at the bargaining solution computed by each player
    true_welfare = independent_gaussian_nash_welfare(y1_opt_1, y2_opt_2, z1_opt_1, z2_opt_2, mu_ay_1_1, mu_ay_2_1,
                                                     mu_az_1_1, mu_az_2_1, budget, num_draws=1000)
    true_welfares.append(true_welfare)

  # Display true welfare as a function of prior difference
  plt.plot(prior_differences, true_welfares)
  plt.show()

  return


if __name__ == "__main__":
  inefficiency_from_gaussian_prior_differences()








