import numpy as np
from functools import partial
import nashpy
import matplotlib.pyplot as plt
import pdb


def complete_information_payoffs(threatener_type, target_type, p, c, low_cost, high_cost):
  # Not commit, Low
  if threatener_type == 0 and target_type == 0:
    u_threatener= np.array([[1, p], [1, p]])
    u_target= np.array([[0, 1-p], [0, 1-p]])

  # Not commit, high
  if threatener_type == 0 and target_type == 1:
    u_threatener = np.array([[1, p], [1, p]])
    u_target = np.array([[0, 1 - p], [0, 1 - p]])

  # Commit, low
  if threatener_type == 1 and target_type == 0:
    u_threatener = np.array([[1, p], [1, p*(1-c)]])
    u_target = np.array([[0, 1-p], [0, 1-p - p*low_cost]])

  # Commit, high
  if threatener_type == 1 and target_type == 1:
    u_threatener = np.array([[1, p], [1, p * (1 - c)]])
    u_target = np.array([[0, 1 - p], [0, 1 - p - p * high_cost]])

  return u_threatener, u_target


def get_payoffs(threatener_profile, target_profile, commit_prior_if_committed, commit_prior_if_not_committed,
                cost_prior, low_cost, high_cost, p, c):
  """
  Threatener actions {Not commit, Commit} indexed {0, 1}
  Target actions {Give in, Fight} indexed {0, 1}
  """
  complete_info_partial = partial(complete_information_payoffs, p=p, c=c, low_cost=low_cost, high_cost=high_cost)
  u_threatener = 0.
  u_target = 0.

  for threatener_type_ix, threatener_action in enumerate(threatener_profile):
    for target_type_ix, target_action in enumerate(target_profile):
      if threatener_action == 0:
        commit_prior = commit_prior_if_not_committed
      else:
        commit_prior = commit_prior_if_committed

      u_threatener_type, u_target_type = complete_info_partial(threatener_type_ix, target_type_ix)
      u_threatener += cost_prior[target_type_ix] * commit_prior[threatener_type_ix] * \
                      u_threatener_type[threatener_action, target_action]
      u_target += cost_prior[target_type_ix] * commit_prior[threatener_type_ix] * \
                      u_target_type[threatener_action, target_action]

  return u_threatener, u_target


def create_payoff_matrix(p, c, low_cost, high_cost, commit_prior_if_committed, commit_prior_if_not_committed,
                         cost_prior):
  """
  Target type is in {L, H} for Low/High cost of threat execution
  Threatener type is in {N, C} for able to Commit/not able to commit

  Target actions are in {Give in, Fight}
  Threatener actions are in {Not commit, Commit}

  Target strategies
    (L, H) -> (G, G)
    (L, H) -> (G, F)
    (L, H) -> (F, G)
    (L, H) -> (F, F)
  Threatener strategies
    (C, N) -> (C, N)
    (C, N) -> (N, N)
  """
  payoffs_threatener = np.zeros((2, 4))
  payoffs_target = np.zeros((2, 4))

  threatener_profiles = [(0, 0), (0, 1)]
  target_profiles = [(0, 0), (0, 1), (1, 0), (1, 1)]

  get_payoffs_partial = partial(get_payoffs, commit_prior_if_committed=commit_prior_if_committed,
                                commit_prior_if_not_committed=commit_prior_if_not_committed,
                                cost_prior=cost_prior, low_cost=low_cost,
                                high_cost=high_cost, p=p, c=c)
  for threatener_profile_ix, threatener_profile in enumerate(threatener_profiles):
    for target_profile_ix, target_profile in enumerate(target_profiles):
      u_threatener, u_target = get_payoffs_partial(threatener_profile, target_profile)
      payoffs_threatener[threatener_profile_ix, target_profile_ix] = u_threatener
      payoffs_target[threatener_profile_ix, target_profile_ix] = u_target

  return payoffs_threatener, payoffs_target


def threat_execution_probability_from_profile(threatener_strategy, target_strategy, commit_prior_if_committed,
                                              commit_prior_if_not_committed, cost_prior):
  # ToDo: think again about what priors to use to evaluate this; currently conditioning on commitment types
  threat_execution_probability = (threatener_strategy[1] == 1) * ((target_strategy[1] == 1) * cost_prior[1] + \
                                                                  (target_strategy[3] == 1))

  return threat_execution_probability


def get_equilibria(p, c, low_cost, high_cost, commit_prior_if_committed, commit_prior_if_not_committed, cost_prior):
  payoffs_threatener, payoffs_target = create_payoff_matrix(p, c, low_cost, high_cost, commit_prior_if_committed,
                                                            commit_prior_if_not_committed, cost_prior)
  game = nashpy.Game(payoffs_threatener, payoffs_target)
  eqs = list(game.support_enumeration())
  # ToDo: check for multiple equilibria?
  return game, eqs[0]


def cross_play(p, c, low_cost, high_cost, prior_on_commit_prior_if_committed, commit_prior_if_not_committed,
               prior_on_cost_prior, true_cost_prior, true_commit_prior_if_committed, reps=10):

  # true_game, _ = get_equilibria(p, c, low_cost, high_cost, true_commit_prior_if_committed, commit_prior_if_not_committed,
  #                               true_cost_prior)

  total_threat_execution_probability = 0.
  total_threat_made_probability = 0.
  prior_diffs = []
  for rep in range(reps):
    # Draw each player's priors
    threatener_cost_prior = prior_on_cost_prior()
    target_cost_prior = prior_on_cost_prior()
    threatener_commit_prior_if_committed = prior_on_commit_prior_if_committed()
    target_commit_prior_if_committed = prior_on_commit_prior_if_committed()
    prior_diffs.append(threatener_commit_prior_if_committed[0] - target_commit_prior_if_committed[0])

    # Get corresponding predictions
    _, threatener_eqs = get_equilibria(p, c, low_cost, high_cost, threatener_commit_prior_if_committed,
                                    commit_prior_if_not_committed, threatener_cost_prior)
    _, target_eqs = get_equilibria(p, c, low_cost, high_cost, target_commit_prior_if_committed,
                                commit_prior_if_not_committed, target_cost_prior)
    threat_execution_probability = threat_execution_probability_from_profile(threatener_eqs[0], target_eqs[1],
                                                                             true_commit_prior_if_committed,
                                                                             commit_prior_if_not_committed,
                                                                             true_cost_prior)
    total_threat_execution_probability += threat_execution_probability / reps
    total_threat_made_probability += threatener_eqs[0][1] / reps
  return total_threat_execution_probability, total_threat_made_probability, prior_diffs


def get_prior_over_priors(true_prior, variance_multiplier, size=1):
  """
  beta distribution over prior[0], centered at true_prior[0]
  """
  mean = true_prior[0]
  variance = variance_multiplier * mean * (1 - mean)
  alpha = mean**2 * ((1-mean) / variance - (1 / mean))
  beta = alpha * ((1/mean) - 1)

  def prior_over_priors():
    p = np.random.beta(a=alpha, b=beta, size=size)
    if size == 1:
      p = p[0]
    return [p, 1-p]

  return prior_over_priors


if __name__ == "__main__":
  true_commit_prior_if_committed = [0.5, 0.5]
  true_cost_prior = [0.9, 0.1]

  commit_prior_variance_multiplier = 0.5  # Must be < 1
  cost_prior_variance_multiplier = 0.5

  # Get distributions over credences
  prior_on_cost_prior = get_prior_over_priors(true_cost_prior, cost_prior_variance_multiplier)
  prior_on_commit_prior_if_committed = get_prior_over_priors(true_commit_prior_if_committed,
                                                             commit_prior_variance_multiplier)

  # ToDo: need to account for subgame perfection
  p = 0.5
  c = 0.01
  low_cost = 0.2
  high_cost = 10.
  commit_prior_if_not_committed = [0.9, 0.1]
  threat_prob, threat_made_prob, prior_diffs = cross_play(p, c, low_cost, high_cost, prior_on_commit_prior_if_committed,
                                                          commit_prior_if_not_committed, prior_on_cost_prior,
                                                          true_cost_prior,
                                                          true_commit_prior_if_committed, reps=100)
  print(threat_prob, threat_made_prob)
  # plt.hist(prior_diffs)
  # plt.show()


