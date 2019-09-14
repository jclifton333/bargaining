"""
Model:

Two resources y and z, such that agent i's utility function is
u_i(y, z) = a^y_i * y + a^z_i * z.

# ToDo: amount of units should depend on agents' utility functions.
Each agent can make U units total, so each agent's action is an allocation (y_i, z_i : y_i + z_i = U). The
NBS is given by

argmax_{y, z} \int log{ a^y_t1 * (y_1 + y_2) + a^z_t1 * (z1 + z2) } dP(t1) # ToDo: currently lacks disagreement pt.
            + \int log{ a^y_t2 * (y_1 + y_2) + a^z_t2 * (z1 + z2) } dP(t2)

where P is the prior over types.
"""
import numpy as np
import matplotlib.pyplot as plt


def utility(y1, y2, z1, z2, ay_i, ay_mi, az_i, az_2, budget):
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
  utility_ = ay_i*(y1 + y2) + az_i*(z1 + z2)

  # Get the best players could do without trade, to form disagreement point
  value_from_is_unilateral_action = budget*max((ay_i, az_i))  # ToDo: not accounting for ability to make diff. amounts of y,z
  value_from_mis_unilateral_action = budget*(ay_i*(ay_mi >= az_mi) + az_i*(az_mi > ay_mi))
  disagreement_point = value_from_is_unilateral_action + value_from_mis_unilateral_action

  surplus_utility = utility_ - disagreement_point
  return surplus_utility


def independent_gaussian_expected_welfare(y1, y2, z1, z2, mu_ay, mu_az, num_draws=1000):
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
  ay_prior_draws = np.random.normal(loc=mu_ay, size=num_draws)
  az_prior_draws = np.random.normal(loc=mu_az, size=num_draws)
  welfare_at_each_draw = np.array([np.log(utility(y1, y2, z1, z2, ay, az)) for ay, az in zip(ay_prior_draws,
                                                                                             az_prior_draws)])
  return np.mean(welfare_at_each_draw)


def independent_gaussian_nash_welfare(y1, y2, z1, z2, mu_ay_1, mu_ay_2, mu_az_1, mu_az_2, num_draws=1000):
  """
  Compute Nash welfare when priors over types are given by independent Gaussians over utility coefficients
  ay_i, az_i, i=1,2.

  :param y1:
  :param y2:
  :param z1:
  :param z2:
  :param mu_ay_1:
  :param mu_ay_2:
  :param mu_az_1:
  :param num_draws:
  :return:
  """
  expected_welfare_1 = independent_gaussian_expected_welfare(y1, y2, z1, z2, mu_ay_1, mu_az_1, num_draws=num_draws)
  expected_welfare_2 = independent_gaussian_expected_welfare(y1, y2, z1, z2, mu_ay_2, mu_az_2, num_draws=num_draws)
  nash_welfare = expected_welfare_1 + expected_welfare_2
  return nash_welfare


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
  # ToDo: assuming 1 prefers y to z, and 2 prefers z to y. This means that 1 will be able to create more z and
  # ToDo: 2 will be able to create more y.

  optimal_welfare = -float('inf')
  y1_opt = None
  y2_opt = None

  # Search over all allocations up to budget for each player.
  for y1 in range(budget + 1):
    z1 = budget - y1
    for y2 in range(budget + 1):
      z2 = budget - y2
      nash_welfare_ = independent_gaussian_nash_welfare(2*y1, y2, z1, 2*z2, mu_ay_1, mu_ay_2, mu_az_1, mu_az_2,
                                                        num_draws=num_draws)
      if nash_welfare_ > optimal_welfare:
        optimal_welfare = nash_welfare_
        y1_opt = y1
        y2_opt = y2

  return 2*y1_opt, y2_opt, budget - y1_opt, 2*(budget - y2_opt)


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
  prior_differences = np.linspace(-9, 9, 19)
  true_welfares = []
  for diff in prior_differences:
    mu_ay_1_2 = mu_ay_1_1 + diff
    mu_az_1_2 = 1
    mu_ay_2_2 = mu_az_1_2
    mu_az_2_2 = mu_ay_1_2
    y1_opt_2, y2_opt_2, z1_opt_2, z2_opt_2 = \
      nbs_for_independent_gaussan_priors(mu_ay_1_2, mu_ay_2_2, mu_az_1_2, mu_az_2_2, budget=budget, num_draws=num_draws)

    # Compute true welfare, using player 1's prior, at the bargaining solution computed by each player
    true_welfare = independent_gaussian_nash_welfare(y1_opt_1, y2_opt_2, z1_opt_1, z2_opt_2, mu_ay_1_1, mu_ay_2_1,
                                                     mu_az_1_1, mu_az_2_1, num_draws=1000)
    true_welfares.append(true_welfare)

  # Display true welfare as a function of prior difference
  plt.plot(prior_differences, true_welfares)
  plt.show()

  return


if __name__ == "__main__":
  inefficiency_from_gaussian_prior_differences()








