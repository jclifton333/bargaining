import numpy as np
from functools import partial
import nashpy
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


def get_payoffs(threatener_profile, target_profile, commit_prior, cost_prior, low_cost, high_cost, p, c):
  """
  Threatener actions {Not commit, Commit} indexed {0, 1}
  Target actions {Give in, Fight} indexed {0, 1}
  """
  complete_info_partial = partial(complete_information_payoffs, p=p, c=c, low_cost=low_cost, high_cost=high_cost)
  u_threatener = 0.
  u_target = 0.

  for threatener_type_ix, threatener_action in enumerate(threatener_profile):
    for target_type_ix, target_action in enumerate(target_profile):
      u_threatener_type, u_target_type = complete_info_partial(threatener_type_ix, target_type_ix)
      u_threatener += cost_prior[target_type_ix] * commit_prior[threatener_type_ix] * \
                      u_threatener_type[threatener_action, target_action]
      u_target += cost_prior[target_type_ix] * commit_prior[threatener_type_ix] * \
                      u_target_type[threatener_action, target_action]

  return u_threatener, u_target


def create_payoff_matrix(p, c, low_cost, high_cost, commit_prior, cost_prior):
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

  get_payoffs_partial = partial(get_payoffs, commit_prior=commit_prior, cost_prior=cost_prior, low_cost=low_cost,
                                high_cost=high_cost, p=p, c=c)
  for threatener_profile_ix, threatener_profile in enumerate(threatener_profiles):
    for target_profile_ix, target_profile in enumerate(target_profiles):
      u_threatener, u_target = get_payoffs_partial(threatener_profile, target_profile)
      payoffs_threatener[threatener_profile_ix, target_profile_ix] = u_threatener
      payoffs_target[threatener_profile_ix, target_profile_ix] = u_target

  return payoffs_threatener, payoffs_target


def get_equilibria(p, c, low_cost, high_cost, commit_prior, cost_prior):
  payoffs_threatener, payoffs_target = create_payoff_matrix(p, c, low_cost, high_cost, commit_prior, cost_prior)
  game = nashpy.Game(payoffs_threatener, payoffs_target)
  eqs = list(game.support_enumeration())
  print(f'{payoffs_threatener}\n{payoffs_target}\n{eqs}')
  return


if __name__ == "__main__":
  p = 0.5
  c = 0.01
  low_cost = 0.1
  high_cost = 10.0
  commit_prior = [0.5, 0.5]
  cost_prior = [0.5, 0.5]
  get_equilibria(p, c, low_cost, high_cost, commit_prior, cost_prior)


