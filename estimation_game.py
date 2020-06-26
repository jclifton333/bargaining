import numpy as np
import nashpy as nash
from nash_unif import get_welfare_optimal_eq, expected_payoffs
import pdb
import matplotlib.pyplot as plt
from scipy.stats import norm


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

def closed_form_bounds(sigma=1):
  u1_mean = np.array([[-10, 0], [-3, -1]])
  u2_mean = np.array([[-10, -3], [0, -1]])

  # lower bound on P{ independent Nash -> (0, 0)}
  # Prob Nash 
  p_ind_1 = norm.cdf(u1_mean[0, 1] - u1_mean[1, 1], loc=0, scale=sigma) * \
              norm.cdf(u2_mean[0, 1] - u2_mean[0, 0], loc=0, scale=sigma) * \
              norm.cdf(u1_mean[1, 0] - u1_mean[0, 0], loc=0, scale=sigma) * \
              norm.cdf(u2_mean[1, 0] - u2_mean[1, 1], loc=0, scale=sigma)
  p_ind_nash = p_ind_1 - p_ind_1**2
  # Prob welfare optimal
  p_ind_wo_01 = 1.
  p_ind_wo_10 = 1.
  for i in range(2):
    for j in range(2):
      if not (i == 0 and j == 1):
        welfare_diff_01_ij = u1_mean[i, j] + u2_mean[i, j] - u1_mean[0, 1] - \
          u2_mean[0, 1]
        p_ind_wo_01 *= 1 - norm.cdf(welfare_diff_01_ij, loc=0,
                                    scale=np.sqrt(2)*sigma)
      if not (i == 1 and j == 0):
        welfare_diff_10_ij = u1_mean[i, j] + u2_mean[i, j] - u1_mean[1, 0] - \
          u2_mean[1, 0]
        p_ind_wo_10 *= 1 - norm.cdf(welfare_diff_10_ij, loc=0,
                                    scale=np.sqrt(2)*sigma)
  # ToDo: not tight enough! Doesn't account for dependence of welfare optimal,
  # Nash; assumes welfare optimal, not welfare optimal among Nash equilibria
  p_ind = 2*p_ind_nash*p_ind_wo_10*p_ind_wo_01 - (p_ind_nash*p_ind_wo_10*p_ind_wo_01)**2

  # upper bound on P{ averaged Nash -> (0, 0) }
  # p{ one of the four possible averages is a NE
  p = norm.cdf(u1_mean[0, 0] - u1_mean[1, 0], loc=0, scale=sigma) * \
      norm.cdf(u2_mean[0, 0] - u2_mean[0, 1], loc=0, scale=sigma)
  p_avg = 4*p - 6*p**2 - 4*p**3 - p**4
  return p_ind, p_avg


def plot_closed_form_bounds():
  sigma_range = np.linspace(0.5, 5, 10)
  p_ind_lst = []
  p_avg_lst = []
  for sigma in sigma_range:
    p_ind_, p_avg_ = closed_form_bounds(sigma=sigma)
    p_ind_lst.append(p_ind_)
    p_avg_lst.append(p_avg_)
  plt.plot(sigma_range, p_ind_lst, label='ind')
  plt.plot(sigma_range, p_avg_lst, label='avg')
  plt.legend()
  plt.show()
  return


def random_nash(sigma_x=1, n=10):
  u1_mean = np.array([[-10, 0], [-3, -1]])
  u2_mean = np.array([[-10, -3], [0, -1]])
  sigma_x_mat = np.array([[sigma_x, sigma_x], [0., 0.]])
  u1_1 = np.random.normal(loc=u1_mean, scale=sigma_x_mat, size=(n, 2, 2))
  u1_2 = np.random.normal(loc=u2_mean, scale=sigma_x_mat, size=(n, 2, 2))
  u2_1 = np.random.normal(loc=u1_mean, scale=sigma_x_mat, size=(n, 2, 2))
  u2_2 = np.random.normal(loc=u2_mean, scale=sigma_x_mat, size=(n, 2, 2))
  crash_lst = []
  for i in range(n):
    u1_1_i, u1_2_i, u2_1_i, u2_2_i = u1_1[i], u1_2[i], u2_1[i], u2_2[i]
    a1_1, _, _ = get_welfare_optimal_eq(nash.Game(u1_1_i, u1_2_i))
    _, a2_2, _ = get_welfare_optimal_eq(nash.Game(u2_1_i, u2_2_i))
    crash = (a1_1[0] == a2_2[0] == 1)
    crash_lst.append(crash)
  print(np.mean(crash_lst))


def alternating(u1_mean=None, u2_mean=None, n=5, sigma_u=1, sigma_x=20):
  # Parameters
  if u1_mean is None:
    u1_mean = np.array([[-10, 0], [-3, -1]])
  if u2_mean is None:
    u2_mean = np.array([[-10, -3], [0, -1]])
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

  # a1_barg, a2_barg, _ = get_welfare_optimal_profile(public_mean_1, public_mean_2)
  a1_barg, a2_barg, _ = get_welfare_optimal_eq(nash.Game(public_mean_1, public_mean_2))
  a1_ind, _, _ = get_welfare_optimal_eq(nash.Game(x1_1.mean(axis=0), x1_2.mean(axis=0)))
  _, a2_ind, _ = get_welfare_optimal_eq(nash.Game(x2_1.mean(axis=0), x2_2.mean(axis=0)))
  # a1_ind, _, _ = get_welfare_optimal_profile(x1_1.mean(axis=0), x1_2.mean(axis=0))
  # _, a2_ind, _ = get_welfare_optimal_profile(x2_1.mean(axis=0), x2_2.mean(axis=0))
  u1_barg, u2_barg = expected_payoffs(u1, u2, a1_barg, a2_barg)
  u1_ind, u2_ind = expected_payoffs(u1, u2, a1_ind, a2_ind)
  return u1_barg - u1_ind, u2_barg - u2_ind, (i % 2 == 0)


if __name__ == "__main__":
  # ToDo: larger action spaces? Estimation accuracy probably much more important here
  sigma_list = np.linspace(1, 10, 11)
  improvement_lst = []
  improvement_se_lst = []
  n_rep = 1000
  n_model_draw = 10
  true_u1_mean = np.array([[-10, 0], [-3, -1]])
  true_u2_mean = np.array([[-10, -3], [0, -1]])
  u_scale = 2
  for sigma in sigma_list:
    improvement_draw_lst = []
    for draw in range(n_model_draw):
      u1_mean = true_u1_mean + np.random.normal(scale=u_scale, size=true_u1_mean.shape)
      u2_mean = true_u2_mean + np.random.normal(scale=u_scale, size=true_u2_mean.shape)
      diff_1_draw_lst = []
      diff_2_draw_lst = []
      even_lst = []
      for rep in range(n_rep):
        diff_1, diff_2, even = alternating(u1_mean=u1_mean, u2_mean=u2_mean, sigma_u=0, sigma_x=sigma, n=2)
        diff_1_draw_lst.append(diff_1)
        diff_2_draw_lst.append(diff_2)
        even_lst.append(even)
      diff_1_draw_mean = np.mean(diff_1_draw_lst)
      diff_2_draw_mean = np.mean(diff_2_draw_lst)
      improvement_draw = np.min((diff_1_draw_mean, diff_2_draw_mean))
      improvement_draw_lst.append(improvement_draw)
    improvement_lst.append(np.min(improvement_draw_lst))
  # plt.errorbar(sigma_list, improvement_lst, yerr=improvement_se_lst, ecolor='r', capsize=2)
  plt.plot(sigma_list, improvement_lst)
  plt.xlabel('sigma')
  plt.ylabel('mean improvement (std err)')
  plt.show()

    # print('mean sd 1: {}\nmean sd 2: {}'.format((np.mean(diff_1_lst), np.std(diff_1_lst) / np.sqrt(n_rep)),
    #                                             (np.mean(diff_2_lst), np.std(diff_2_lst) / np.sqrt(n_rep))))


