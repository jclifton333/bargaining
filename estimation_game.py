import numpy as np
import nashpy as nash
from nash_unif import get_welfare_optimal_eq, expected_payoffs, get_nash_welfare_optimal_eq
from simple import meta_ultimatum_game
from itertools import product
import pdb
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import norm
from scipy.special import expit
import seaborn as sns
np.set_printoptions(suppress=True)


def check_matrix_distances(m11, m12, m21, m22, sigma, sigma_tol):
  """
  Check that two pairs of matrices are sufficiently close, where they are
  assumed to be means of iid normal with diag(sigma) covariance.
  """
  dist_1 = np.mean((m11 - m21)**2)
  dist_2 = np.mean((m12 - m22)**2)
  if (dist_1 + dist_2) / 2 < sigma_tol * sigma:
    return True
  else:
    return False


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


def get_welfare_optimal_observation(x_1, x_2, public_1, public_2, n_public, x_1_true, x_2_true, used_ixs,
                                    sigma, sigma_tol=2, too_far_penalty=100):
  best_ix = None
  best_payoff = -float('inf')
  ix = 0
  best_obs_1, best_obs_2 = None, None
  best_a1, best_a2 = None, None
  for obs_1, obs_2 in zip(x_1, x_2):
    new_public_1 = public_1 + (obs_1 - public_1) / (n_public + 1)
    new_public_2 = public_2 + (obs_2 - public_2) / (n_public + 1)
    if ix not in used_ixs:
      close_enough = check_matrix_distances(public_1, public_2, obs_1, obs_2, sigma, sigma_tol)
      # a1, a2, _ = get_welfare_optimal_eq(nash.Game(new_public_1, new_public_2))
      a1, a2, _ = get_welfare_optimal_profile(new_public_1, new_public_2)
      u1_obs, _ = expected_payoffs(x_1_true, x_2_true, a1, a2)
      u1_obs -= (1 - close_enough)*too_far_penalty
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


def coop_bargaining(a1, a2, beta=1, sigma=1, tau=1, p_L=0.5, p_U=1.5, epsilon_1=0.50, epsilon_2=1.,
                    d1=0.1, d2=-10):
  """
  epsilon_2 > epsilon_1 by default represents exploitation of player 1 by player 2

  Cooperative bargaining model with utilities
  u_1(x, beta) = beta*x
  u_2(x, beta) = -x**2
  w(x, beta, p) = u_1(x, beta)**p + u_2(x, beta)**p

  And
    beta_i \iid LogNormal(mu, sigma)
    default iff beta1_tilde/beta2_tilde > tau
  """
  # ALPHA_1 = 0.05
  # ALPHA_2 = 0.15
  # z_1 = norm.ppf(1-ALPHA_1)
  # z_2 = norm.ppf(1-ALPHA_2)
  # epsilon_1 = np.exp(-z_1*sigma)
  # epsilon_2 = np.exp(-z_2*sigma)

  mu = np.log(beta) - sigma**2/2
  beta1_hat = np.random.lognormal(mean=mu, sigma=sigma)
  beta2_hat = np.random.lognormal(mean=mu, sigma=sigma)
  # ToDo: use BR to confidence bound instead?
  beta1_tilde = beta1_hat * np.sqrt(tau) * epsilon_1
  beta2_tilde = beta2_hat / (np.sqrt(tau) * epsilon_2)
  ratio = beta1_tilde / beta2_tilde
  close_enough = (ratio < tau)
  # ToDo: handle comparison to disagreement points?
  play_coop = (a1 == 1 and a1 == 1 and close_enough)
  if play_coop:
    betaHat = np.sqrt(beta1_tilde * beta2_tilde)
    xHat = betaHat / 2
    r1_coop = beta * xHat
    r2_coop = -xHat**2
    if r1_coop > d1 and r2_coop > d2:
      r1, r2 = r1_coop, r2_coop
    else:
      r1, r2, = d1, d2
  else:
    fair_lower = beta2_hat * np.power(2, -1/p_L)
    fair_upper = beta2_hat * np.power(2, -1/p_U)
    xHat = beta1_hat / 2
    if fair_lower < xHat < fair_upper:
      r1 = beta * xHat
      r2 = -xHat**2
    else:
      r1, r2 = d1, d2
  return r1, r2, close_enough


def monotone(u1=None, u2=None):
  if u1 is None:
    u1 = np.array([[-10, 0.5], [-3, -1]])
  if u2 is None:
    u2 = np.array([[-10, -3], [0, -1]])

  all_payoffs = []
  for rep in range(100):
    direction = np.random.uniform(low=-1, high=1, size=(2, 2))
    payoffs_1 = []
    magnitudes = np.linspace(0, 10, 100)
    for magnitude in magnitudes:
      u1_perturbed = u1 + direction*magnitude
      _, _, v1 = get_welfare_optimal_eq(nash.Game(u1_perturbed, u2))
      payoffs_1.append(v1)
    all_payoffs.append(payoffs_1)

  plt.plot(magnitudes, np.mean(all_payoffs, axis=0))
  plt.show()
  return


def alternating(a1, a2, u1_mean=None, u2_mean=None, bias_2_1=np.zeros((2, 2)), bias_2_2=np.zeros((2,2)), n=2, sigma_u=1,
                sigma_x=20, sigma_tol=0.1):
  """
  ai: int in {0, 1}, 0=collab and 1=independent.
  """
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
  x2_1 = np.random.normal(loc=u1 + bias_2_1, scale=sigma_x_mat, size=(n, 2, 2))
  x2_2 = np.random.normal(loc=u2 + bias_2_2, scale=sigma_x_mat, size=(n, 2, 2))
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
  if a1 == 1 and a2 == 1:
    while t < 1 and not done:
      # ToDo: passing known sigma_x to check_matrix_distances in get_welfare_optimal_observation, whereas in practice
      # ToDo: this is unknown
      best_payoff_1_t, best_obs_11, best_obs_12, best_ix_1, best_a1, best_a2 = \
        get_welfare_optimal_observation(x1_1, x1_2, public_mean_1, public_mean_2, len(ixs_1) + len(ixs_2), x1_1_mean,
                                        x1_2_mean, ixs_1, sigma_x / np.sqrt(n), sigma_tol=sigma_tol)
      ixs_1.append(best_ix_1)
      public_mean_1 += (best_obs_11 - public_mean_1) / (len(ixs_1) + len(ixs_2))
      public_mean_2 += (best_obs_12 - public_mean_2) / (len(ixs_1) + len(ixs_2))
      # # Other player naively incorporates new info
      # x2_1_mean += (best_obs_1 - x2_1_mean) / (t + 1 + n)
      # x2_2_mean += (best_obs_2 - x2_2_mean) / (t + 1 + n)

      i += 1
      x2_2 = [np.array([[-10, -3], [1., -1]])]
      x2_2.append(public_mean_2)
      # x2_1 = [x2_1_mean + np.random.uniform(low=-10, high=10, size=((2, 2))) for _ in range(n)]
      x2_1 = [np.array([[-10, -0.5], [-3, -1]])]
      x2_1.append(public_mean_1)
      best_payoff_2_t, best_obs_22, best_obs_21, best_ix_2, best_a2, best_a1 = \
        get_welfare_optimal_observation(x2_2, x2_1, public_mean_2, public_mean_1, len(ixs_1) + len(ixs_2), x2_2_mean,
                                        x2_1_mean, ixs_2, sigma_x / np.sqrt(n), sigma_tol=sigma_tol)
      ixs_2.append(best_ix_2)
      public_mean_1 += (best_obs_21 - public_mean_1) / (len(ixs_1) + len(ixs_2))
      public_mean_2 += (best_obs_22 - public_mean_2) / (len(ixs_1) + len(ixs_2))
      # # Other player naively incorporates new info
      # x1_1_mean += (best_obs_1 - x1_1_mean) / (t + 1 + n)
      # x1_2_mean += (best_obs_2 - x1_2_mean) / (t + 1 + n)
      t += 1
      i += 1

    close_enough = check_matrix_distances(best_obs_11, best_obs_12, best_obs_21, best_obs_22, sigma_x / np.sqrt(n),
                                          sigma_tol=sigma_tol)
    if close_enough:
      d1, d2, _ = get_welfare_optimal_eq(nash.Game(public_mean_1, public_mean_2))
    else:
      d1, _, _ = get_welfare_optimal_eq(nash.Game(x1_1_mean, x1_2_mean))
      _, d2, _ = get_welfare_optimal_eq(nash.Game(x2_1_mean, x2_2_mean))
  else:
    d1, _, _ = get_welfare_optimal_eq(nash.Game(x1_1_mean, x1_2_mean))
    _, d2, _ = get_welfare_optimal_eq(nash.Game(x2_1_mean, x2_2_mean))
    close_enough = False

  r1, r2 = expected_payoffs(u1, u2, d1, d2)
  return r1, r2, close_enough


def learn_conditional_expectation(epsilon_1, epsilon_2, env='coop', player=1, sigma_tol=2, sigma_upper=1., n_obs_per_strat=75):
  """
  Estimate expectation of coop and ind strategies given sigma.
  """
  X0 = np.zeros((0, 1))  # Will contain history of sigmas
  X1 = np.zeros((0, 1))
  y0 = np.zeros(0)
  y1 = np.zeros(0)
  lm0 = RandomForestRegressor(n_estimators=10, max_depth=2, min_samples_split=0.2)
  lm1 = RandomForestRegressor(n_estimators=10, max_depth=2, min_samples_split=0.2)
  # lm0 = Ridge()
  # lm1 = Ridge()
  is_player_1 = (player == 1)

  for obs in range(n_obs_per_strat):  # Collect cooperative strategy obs
    sigma = np.random.uniform(low=0., high=sigma_upper)
    if env == 'coop':
      beta = np.random.uniform(low=4., high=6.)
      r1, r2, close_enough_ = coop_bargaining(1, 1, beta=5, sigma=sigma,
                                              tau=sigma_tol,
                                              epsilon_1=epsilon_1,
                                              epsilon_2=epsilon_2)
    elif env == 'ug':
      alpha_1_over_alpha_0 = np.random.uniform(0, 1)
      alpha_1 = sigma*alpha_1_over_alpha_0
      alpha_2 = sigma - alpha_1
      prior_1 = np.random.dirichlet(np.array([alpha_1, alpha_2]))
      prior_2 = np.random.dirichlet(np.array([alpha_1, alpha_2]))
      r1, r2 = meta_ultimatum_game(1, 1, prior_1, prior_2, eps_1=epsilon_1,
                                  eps_2=epsilon_2, tau=sigma_tol)

    X1 = np.vstack((X1, [sigma]))
    r = r1*is_player_1 + r2*(1 - is_player_1)
    y1 = np.hstack((y1, r))

  for obs in range(n_obs_per_strat):  # Collect independent strategy obs
    if is_player_1:
      a1 = 0
      a2 = 1
    else:
      a1 = 1
      a2 = 0

    sigma = np.random.uniform(low=0., high=sigma_upper)
    if env == 'coop':
      beta = np.random.uniform(low=4., high=6.)
      r1, r2, close_enough_ = coop_bargaining(a1, a2, beta=beta, sigma=sigma,
                                              tau=sigma_tol,
                                              epsilon_1=epsilon_1,
                                              epsilon_2=epsilon_2)
    elif env == 'ug':
      alpha_1_over_alpha_0 = np.random.uniform(0, 1)
      alpha_1 = sigma*alpha_1_over_alpha_0
      alpha_2 = sigma - alpha_1
      prior_1 = np.random.dirichlet(np.array([alpha_1, alpha_2]))
      prior_2 = np.random.dirichlet(np.array([alpha_1, alpha_2]))
      r1, r2 = meta_ultimatum_game(a1, a2, prior_1, prior_2, eps_1=epsilon_1,
                                   eps_2=epsilon_2, tau=sigma_tol)

    X0 = np.vstack((X0, [sigma]))
    r = r1*is_player_1 + r2*(1 - is_player_1)
    y0 = np.hstack((y0, r))

  # Estimate conditional expectations
  lm0.fit(X0, y0)
  lm1.fit(X1, y1)

  q0 = lambda s: lm0.predict([[s]])
  q1 = lambda s: lm1.predict([[s]])

  # lm0.fit(X0, y0)
  # lm1.fit(X1, y1)
  # xrange_ = np.linspace(0, sigma_upper, 100).reshape(-1, 1)
  # y0_hat = lm0.predict(xrange_)
  # y1_hat = lm1.predict(xrange_)
  # # plt.scatter(X0, y0, label='ind payoffs')
  # plt.plot(xrange_, y0_hat, label='ind')
  # # plt.scatter(X1, y1, label='coop payoffs')
  # plt.plot(xrange_, y1_hat, label='cgs')
  # plt.xlabel('sigma')
  # plt.ylabel('Estimated expected payoff')
  # plt.title('Random forest estimates of conditional rewards\nunder each reporting policy')
  # plt.legend()
  # plt.show()
  # pdb.set_trace()

  return q0, q1


# ToDo: encapsulate environment settings in environments
def bandit(policy='cb', time_horizon=50, n=5, sigma_tol=1, sigma_upper=1.,
           env='coop', epsilon_1=1.0, epsilon_2=1., epsilon_21=1., epsilon_12=1., n_obs_per_strat=500):
  y1 = np.zeros(0)
  y2 = np.zeros(0)
  close_enough_lst = []
  welfare_lst = []

  if policy == 'cb':
    q0_1, q1_1 = learn_conditional_expectation(epsilon_1, epsilon_21, player=1, sigma_tol=sigma_tol,
                                               sigma_upper=sigma_upper,
                                               n_obs_per_strat=n_obs_per_strat,
                                               env=env)
    q0_2, q1_2 = learn_conditional_expectation(epsilon_12, epsilon_2, player=2, sigma_tol=sigma_tol,
                                               sigma_upper=sigma_upper, env=env,
                                               n_obs_per_strat=n_obs_per_strat)

  for t in range(time_horizon):
    # if np.random.random() < 1.0:
    #   true_u1_mean = np.array([[5, 2], [5, 2]])
    #   true_u2_mean = np.array([[0, 2], [0, 2]])
    # else:
    #   true_u1_mean = np.array([[0, 3], [0, 3]])
    #   true_u2_mean = np.array([[5, 3], [5, 3]])

    true_u1_mean = np.array([[-10, 0.5], [-3, -1]])
    true_u2_mean = np.array([[-10, -3], [0, -1]])

    # Draw context and take action
    sigma = np.random.uniform(0.0, sigma_upper)
    if env == 'ug':
      alpha_1_over_alpha_0 = np.random.uniform(0, 1)
      alpha_1 = sigma*alpha_1_over_alpha_0
      alpha_2 = sigma - alpha_1
      prior_1 = np.random.dirichlet(np.array([alpha_1, alpha_2]))
      prior_2 = np.random.dirichlet(np.array([alpha_1, alpha_2]))

    if policy == 'cb':
      a1 = q1_1(sigma) > q0_1(sigma)
      a2 = q1_2(sigma) > q0_2(sigma)
    elif policy=='ind':
      a1 = 0
      a2 = 0
    elif policy=='coop':
      a1 = 1
      a2 = 1

    if env == 'nash':
      r1, r2, close_enough_ = alternating(a1, a2, u1_mean=true_u1_mean, u2_mean=true_u2_mean, bias_2_1=bias_2_1,
                                         bias_2_2=bias_2_2, sigma_u=0, sigma_x=sigma, n=n, sigma_tol=sigma_tol)
    elif env == 'coop':
      r1, r2, close_enough_ = coop_bargaining(a1, a2, beta=5, sigma=sigma,
                                              tau=sigma_tol,
                                              epsilon_1=epsilon_1,
                                              epsilon_2=epsilon_2)
    elif env == 'ug':
      r1, r2 = meta_ultimatum_game(a1, a2, prior_1, prior_2, eps_1=epsilon_1,
                                   eps_2=epsilon_2, tau=sigma_tol)

    y1 = np.hstack((y1, r1))
    y2 = np.hstack((y2, r2))
    # close_enough_lst.append(close_enough_)
    welfare_lst.append(r1 + r2)

  return y1, y2, [], np.mean(welfare_lst)


def optimize_mechanism(policy='coop', env='coop', time_horizon=1000, n=5, mc_rep=10):
  if env == 'coop':
    epsilon_1 = 1.
    epsilon_2 = 1.
    tau_range = np.logspace(-1, 1, 5)
    sigma_upper = 1.
  elif env == 'ug':
    epsilon_1 = 0.5
    epsilon_2 = 0.1
    tau_range = np.linspace(0.1, 2, 5)
    sigma_upper = 1.

  for tau in tau_range:
    welfare_tau_lst = []
    for rep in range(mc_rep):
      _, _, _, welfare_tau_rep = bandit(policy=policy, time_horizon=time_horizon, n=n, sigma_tol=tau,
                                        sigma_upper=sigma_upper, env=env, epsilon_1=epsilon_1,
                                        epsilon_12=epsilon_1, epsilon_2=epsilon_2, epsilon_21=epsilon_2)
      welfare_tau_lst.append(welfare_tau_rep)
    welfare_tau_mean = np.mean(welfare_tau_lst)
    welfare_tau_se = np.std(welfare_tau_lst) / np.sqrt(len(welfare_tau_lst))
    print(welfare_tau_mean, welfare_tau_se)
  return


def optimize_mechanism_nash(env='coop', time_horizon=100, n=5, mc_rep=100, nA=10, eps_upper=1., policy='coop'):
  if env == 'coop':
    tau_range = np.logspace(-1, 1, 5)
  elif env == 'ug':
    tau_range = np.linspace(0.1, 2, 5)

  for tau in tau_range:
    res = nash_reporting_policy(env=env, time_horizon=time_horizon, n=n, mc_rep=mc_rep, nA=nA, tau=tau,
                                eps_upper=eps_upper, policy=policy)
    v1, v2 = res['v1'], res['v2']
    welfare = v1 + v2
    print(welfare)
  return


def optimize_reporting_policy(time_horizon=50, n=5, sigma_upper=1., sigma_tol=1., mc_rep=5):
  for epsilon_1 in [0.3, 0.5, 1, 2, 5]:
    rewards_lst = []
    close_enough_lst = []
    for rep in range(mc_rep):
      rewards_rep, close_enough_rep, _ = bandit(policy='cb', time_horizon=time_horizon, n=n, sigma_tol=sigma_tol,
                                                sigma_upper=sigma_upper, env='coop', epsilon_1=epsilon_1)
      rewards_lst.append(np.mean(rewards_rep))
      close_enough_lst.append(np.mean(close_enough_rep))
    rewards_mean = np.mean(rewards_lst)


def nash_reporting_policy(env='coop', time_horizon=100, n=5, mc_rep=100,
                          nA=10, tau=1., eps_upper=1., policy='coop'):
  # Compute payoff matrix
  if env=='coop':
    epsilon_1_space = np.linspace(0.1, eps_upper, nA)
    epsilon_2_space = np.linspace(0.1, eps_upper, nA)
    sigma_upper = 2.
  elif env=='ug':
    epsilon_1_space = np.linspace(0, eps_upper, nA)
    epsilon_2_space = np.linspace(0, eps_upper, nA)
    sigma_upper = 10.

  if policy == 'cb':
    nA = nA ** 2
    epsilon_1_prod = [(eps1, eps2) for eps1 in epsilon_1_space for eps2 in epsilon_2_space]
    epsilon_2_prod = [(eps2, eps1) for eps2 in epsilon_2_space for eps1 in epsilon_1_space]
  else:
    epsilon_1_prod = [(eps1, eps1) for eps1 in epsilon_1_space]
    epsilon_2_prod = [(eps2, eps2) for eps2 in epsilon_2_space]

  payoffs_1 = np.zeros((nA+1, nA+1))
  payoffs_2 = np.zeros((nA+1, nA+1))
  standard_errors_1 = np.zeros((nA+1, nA+1))
  standard_errors_2 = np.zeros((nA+1, nA+1))

  for i, (epsilon_1, epsilon_21) in enumerate(epsilon_1_prod):
    for j, (epsilon_2, epsilon_12) in enumerate(epsilon_2_prod):
      se_ij_1 = []
      se_ij_2 = []
      # print(i, j)
      for rep in range(mc_rep):
        rewards_rep_1, rewards_rep_2, _, _ = \
          bandit(policy=policy, time_horizon=time_horizon, n=n, sigma_tol=tau, sigma_upper=sigma_upper,
                env=env, epsilon_1=epsilon_1, epsilon_2=epsilon_2, epsilon_12=epsilon_12, epsilon_21=epsilon_21)
        payoffs_1[i, j] += np.mean(rewards_rep_1) / mc_rep
        payoffs_2[i, j] += np.mean(rewards_rep_2) / mc_rep
        se_ij_1 += list(rewards_rep_1)
        se_ij_2 += list(rewards_rep_2)
      standard_errors_1[i, j] = np.std(se_ij_1) / np.sqrt(len(se_ij_1))
      standard_errors_2[i, j] = np.std(se_ij_2) / np.sqrt(len(se_ij_2))

  # Get independent payoffs
  se_1 = []
  se_2 = []
  for rep in range(mc_rep):
    rewards_rep_1, rewards_rep_2, _, _ = \
      bandit(policy='ind', time_horizon=time_horizon, n=n, sigma_tol=tau, sigma_upper=sigma_upper,
             env=env, epsilon_1=epsilon_1, epsilon_2=epsilon_2)
    payoffs_1[nA, :] += np.mean(rewards_rep_1) / mc_rep
    payoffs_1[:-1, nA] += np.mean(rewards_rep_1) / mc_rep
    payoffs_2[nA, :] += np.mean(rewards_rep_2) / mc_rep
    payoffs_2[:-1, nA] += np.mean(rewards_rep_2) / mc_rep
    se_1 += list(rewards_rep_1)
    se_2 += list(rewards_rep_2)

  standard_errors_1[nA, :] = np.std(se_1) / np.sqrt(len(rewards_rep_1))
  standard_errors_1[:-1, nA] = np.std(se_1) / np.sqrt(len(rewards_rep_1))
  standard_errors_2[nA, :] = np.std(se_2) / np.sqrt(len(rewards_rep_2))
  standard_errors_2[:-1, nA] = np.std(se_2) / np.sqrt(len(rewards_rep_2))

  # print(standard_errors_1)
  # print(standard_errors_2)

  # Compute nash
  payoffs_1 = payoffs_1.round(2)
  payoffs_2 = payoffs_2.round(2)
  d1, d2 = payoffs_1[nA, nA], payoffs_2[nA, nA]
  # e1, e2, _ = get_nash_welfare_optimal_eq(nash.Game(payoffs_1, payoffs_2), d1, d2)
  e1, e2, _ = get_welfare_optimal_eq(nash.Game(payoffs_1, payoffs_2))
  se_1 = np.sqrt(np.dot(e1, np.dot(standard_errors_1**2, e2)))  # ToDo: check se calculation
  se_2 = np.sqrt(np.dot(e1, np.dot(standard_errors_2**2, e2)))
  v1, v2 = expected_payoffs(payoffs_1, payoffs_2, e1, e2)
  return {'epsilon_1_space': epsilon_1_space, 'epsilon_2_space':
          epsilon_2_space, 'e1': e1, 'e2': e2, 'v1': v1, 'v2': v2, 'payoffs_1': payoffs_1,
          'payoffs_2': payoffs_2, 'se_1': se_1, 'se_2': se_2}


def compare_policies(plot_name, env='coop', replicates=10, time_horizon=50, n_private_obs=5, sigma_tol=1, sigma_upper=1.):
  r1_list_cb = []
  r1_list_ind = []
  r1_list_coop = []
  r2_list_cb = []
  r2_list_ind = []
  r2_list_coop = []
  close_list_coop = []
  for _ in range(replicates):
    r1_list_cb_rep, r2_list_cb_rep, _, _ = bandit(policy='cb', n=n_private_obs, time_horizon=time_horizon, sigma_tol=sigma_tol,
                               env=env, sigma_upper=sigma_upper)
    r1_list_cb.append(r1_list_cb_rep)
    r2_list_cb.append(r2_list_cb_rep)
    r1_list_ind_rep, r2_list_ind_rep, _, _ = bandit(policy='ind', n=n_private_obs, time_horizon=time_horizon, sigma_tol=sigma_tol,
                                env=env, sigma_upper=sigma_upper)
    r1_list_ind.append(r1_list_ind_rep)
    r2_list_ind.append(r2_list_ind_rep)
    r1_list_coop_rep, r2_list_coop_rep, _, _ = bandit(policy='coop', n=n_private_obs, time_horizon=time_horizon,
                                          env=env, sigma_tol=sigma_tol, sigma_upper=sigma_upper)
    r1_list_coop.append(r1_list_coop_rep)
    r2_list_coop.append(r2_list_coop_rep)

  data = {'cb1': np.mean(r1_list_cb, axis=1),
          'ind1': np.mean(r1_list_ind, axis=1),
          'coop1': np.mean(r1_list_coop, axis=1),
          'cb2': np.mean(r2_list_cb, axis=1),
          'ind2': np.mean(r2_list_ind, axis=1),
          'coop2': np.mean(r2_list_coop, axis=1),
          'timepoint': np.arange(len(r1_list_coop[0]))}

  cb_mean1, cb_se1 = data['cb1'].mean(), data['cb1'].std() / np.sqrt(replicates)
  ind_mean1, ind_se1 = data['ind1'].mean(), data['ind1'].std() / np.sqrt(replicates)
  coop_mean1, coop_se1 = data['coop1'].mean(), data['coop1'].std() / np.sqrt(replicates)
  cb_mean2, cb_se2 = data['cb2'].mean(), data['cb2'].std() / np.sqrt(replicates)
  ind_mean2, ind_se2 = data['ind2'].mean(), data['ind2'].std() / np.sqrt(replicates)
  coop_mean2, coop_se2 = data['coop2'].mean(), data['coop2'].std() / np.sqrt(replicates)

  print('cb1: {}\nind: {}\ncoop: {}'.format((cb_mean1, cb_se1, cb_mean2, cb_se2),
                                            (ind_mean1, ind_se1, ind_mean2, ind_se2),
                                            (coop_mean1, coop_se1, coop_mean2, coop_se2)))

  # sns.lineplot(x=[i for _ in range(replicates) for i in
  #                 range(data['cb'].shape[1])], y=np.hstack(data['cb']),
  #             label='cb')
  # sns.lineplot(x=[i for _ in range(replicates) for i in
  #                 range(data['mab'].shape[1])], y=np.hstack(data['mab']),
  #              label='mab', color='k')
  # sns.lineplot(x=[i for _ in range(replicates) for i in
  #               range(data['cb'].shape[1])], y=np.hstack(data['ind']),
  #           label='ind')
  # sns.lineplot(x=[i for _ in range(replicates) for i in
  #               range(data['cb'].shape[1])], y=np.hstack(data['coop']),
  #           label='coop', color='r')
  # plt.legend()
  # plt.title('Tolerance={} Maximum sigma={}'.format(sigma_tol, sigma_upper))
  # plt.xlabel('Time')
  # plt.ylabel('Cumulative reward')
  # if plot_name is not None:
  #   plt.savefig('{}.png'.format(plot_name))
  # else:
  #   plt.show()
  # return


if __name__ == "__main__":
  # ToDo: Note that \beta's are now being drawn randomly rather than being set to the true value in
  # ToDo: learning the conditional expectation function, which is not reflected in the draft as of July 19 2020
  # sigma_tol_list = [2]
  # for sigma_tol in sigma_tol_list:
  #   compare_policies(None, env='ug', replicates=1, n_private_obs=2,
  #                    time_horizon=10000, sigma_tol=sigma_tol, sigma_upper=1)
  # res = nash_reporting_policy(env='ug', policy='cb', time_horizon=1000, n=2, mc_rep=1,
  #                             nA=4, tau=0.3, eps_upper=0.3)
  # print(res['payoffs_1'].round(2))
  # print(res['payoffs_2'].round(2))
  # print(res['v1'], res['se_1'])
  # print(res['v2'], res['se_2'])

  optimize_mechanism_nash(env='ug', policy='coop', time_horizon=1000, mc_rep=1, nA=3, eps_upper=1.0)
