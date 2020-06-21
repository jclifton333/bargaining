import numpy as np
import nashpy as nash
from nash_unif import get_welfare_optimal_eq, expected_payoffs
import pdb


def get_welfare_optimal_profile(p1, p2):
  best_welfare = -float("inf")
  best_a1, best_a2 = None, None
  for i in range(p1.shape[0]):
    a1 = np.zeros(2)
    a1[i] = 1
    for j in range(p1.shape[1]):
      a2 = np.zeros(2)
      a2[i] = 1
      welfare = np.dot(a1, np.dot(p1, a2)) + np.dot(a2, np.dot(p2, a1))
      if welfare > best_welfare:
        best_a1, best_a2 = a1, a2
        best_welfare = welfare
  return best_a1, best_a2, best_welfare


def get_welfare_optimal_observation(x_1, x_2, public_1, public_2, n_public, x_1_true, x_2_true, used_ixs):
  best_ix = None
  best_payoff = -float('inf')
  ix = 0
  best_obs_1, best_obs_2 = None, None
  best_a1, best_a2 = None, None
  for obs_1, obs_2 in zip(x_1, x_2):
    new_public_1 = public_1 + (obs_1 - public_1) / (n_public + 1)
    new_public_2 = public_2 + (obs_2 - public_2) / (n_public + 1)
    if ix not in used_ixs:
      # a1, a2, _ = get_welfare_optimal_eq(nash.Game(new_public_1, new_public_2))
      a1, a2, _ = get_welfare_optimal_profile(new_public_1, new_public_2)
      u1_obs, _ = expected_payoffs(x_1_true, x_2_true, a1, a2)
      if u1_obs > best_payoff:
        best_payoff = u1_obs
        best_ix = ix
        best_obs_1, best_obs_2 = obs_1, obs_2
        best_a1, best_a2 = a1, a2
    ix += 1
  return best_payoff, best_obs_1, best_obs_2, best_ix, best_a1, best_a2


def alternating(n=5, sigma_u=1, sigma_x=20):
  # Parameters
  u1_mean = np.array([[-4, 0], [-5, -1]])
  u2_mean = np.array([[-6, -5], [0, -1]])
  sigma_x_mat = np.array([[sigma_x, sigma_x], [0., 0.]])

  # Generate true 2x2 payoffs
  u1 = np.random.normal(loc=u1_mean, scale=sigma_u, size=(2, 2))
  u2 = np.random.normal(loc=u2_mean, scale=sigma_u, size=(2, 2))

  # Generate obs
  x1_1 = np.random.normal(loc=u1, scale=sigma_x_mat, size=(n, 2, 2))
  x1_2 = np.random.normal(loc=u2, scale=sigma_x_mat, size=(n, 2, 2))
  x2_1 = np.random.normal(loc=u1, scale=sigma_x_mat, size=(n, 2, 2))
  x2_2 = np.random.normal(loc=u2, scale=sigma_x_mat, size=(n, 2, 2))
  x1_1_mean = x1_1.mean(axis=0)
  x1_2_mean = x1_2.mean(axis=0)
  x2_1_mean = x2_1.mean(axis=0)
  x2_2_mean = x2_2.mean(axis=0)
  # x1_1_std = x1_1.std(axis=0)
  # x1_2_std = x1_2.std(axis=0)
  # x2_1_std = x2_1.std(axis=0)
  # x2_2_std = x2_2.std(axis=0)

  # Greedy alternating offers
  public_mean_1 = np.zeros((2, 2))
  public_mean_2 = np.zeros((2, 2))
  best_payoff_1 = -np.float('inf')
  best_payoff_2 = -np.float('inf')
  ixs_1 = []
  ixs_2 = []
  i = 0
  # ToDo: check indices for nashpy
  # ToDo: rule for rejecting
  done = False
  t = 0
  while t < np.floor(n/2) and not done:
    best_payoff_1_t, best_obs_1, best_obs_2, best_ix_1, best_a1, best_a2 = \
      get_welfare_optimal_observation(x1_1, x1_2, public_mean_1, public_mean_2, len(ixs_1) + len(ixs_2), x1_1_mean,
                                      x1_2_mean, ixs_1)
    ixs_1.append(best_ix_1)
    public_mean_1 += (best_obs_1 - public_mean_1) / (len(ixs_1) + len(ixs_2))
    public_mean_2 += (best_obs_2 - public_mean_2) / (len(ixs_1) + len(ixs_2))
    # Other player naively incorporates new info
    x2_1_mean += (best_obs_1 - x2_1_mean) / (t + 1 + n)
    x2_2_mean += (best_obs_2 - x2_2_mean) / (t + 1 + n)

    i += 1
    best_payoff_2_t, best_obs_2, best_obs_1, best_ix_2, best_a2, best_a1 = \
      get_welfare_optimal_observation(x2_2, x2_1, public_mean_2, public_mean_1, len(ixs_1) + len(ixs_2), x2_2_mean,
                                      x2_1_mean, ixs_2)
    ixs_2.append(best_ix_2)
    public_mean_1 += (best_obs_1 - public_mean_1) / (len(ixs_1) + len(ixs_2))
    public_mean_2 += (best_obs_2 - public_mean_2) / (len(ixs_1) + len(ixs_2))
    # Other player naively incorporates new info
    x1_1_mean += (best_obs_1 - x1_1_mean) / (t + 1 + n)
    x1_2_mean += (best_obs_2 - x1_2_mean) / (t + 1 + n)
    t += 1
    i += 1

  a1_barg, a2_barg, _ = get_welfare_optimal_profile(public_mean_1, public_mean_2)
  a1_ind, _, _ = get_welfare_optimal_eq(nash.Game(x1_1.mean(axis=0), x1_2.mean(axis=0)))
  _, a2_ind, _ = get_welfare_optimal_eq(nash.Game(x2_1.mean(axis=0), x2_2.mean(axis=0)))
  # a1_ind, _, _ = get_welfare_optimal_profile(x1_1.mean(axis=0), x1_2.mean(axis=0))
  # _, a2_ind, _ = get_welfare_optimal_profile(x2_1.mean(axis=0), x2_2.mean(axis=0))
  u1_barg, u2_barg = expected_payoffs(u1, u2, a1_barg, a2_barg)
  u1_ind, u2_ind = expected_payoffs(u1, u2, a1_ind, a2_ind)
  return u1_barg - u1_ind, u2_barg - u2_ind, (i % 2 == 0)


if __name__ == "__main__":
  # ToDo: larger action spaces? Estimation accuracy probably much more important here
  diff_1_lst = []
  diff_2_lst = []
  even_lst = []
  for rep in range(1000):
    diff_1, diff_2, even = alternating(sigma_u=0, sigma_x=2, n=4)
    diff_1_lst.append(diff_1)
    diff_2_lst.append(diff_2)
    even_lst.append(even)
  print(np.mean(diff_1_lst), np.mean(diff_2_lst), np.mean(even_lst))


