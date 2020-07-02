import numpy as np
import nashpy as nash
from nash_unif import get_welfare_optimal_eq, expected_payoffs
import pdb
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import norm
from scipy.special import expit
import seaborn as sns


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


def coop_bargaining(a1, a2, beta=1, sigma=1, tau=1, p_L=0.8, p_U=1.2, epsilon_1=0.65, epsilon_2=0.8,
                    d1=0.1, d2=0.1):
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
  mu = np.log(beta) - sigma/2
  beta1_hat = np.random.lognormal(mean=mu, sigma=sigma)
  beta2_hat = np.random.lognormal(mean=mu, sigma=sigma)
  beta1_tilde = beta1_hat * np.sqrt(tau) * epsilon_1
  beta2_tilde = beta2_hat / (np.sqrt(tau) * epsilon_2)
  ratio = beta1_tilde / beta2_tilde
  close_enough = (ratio < tau)
  if a1 == 1 and a2 == 1:
    if close_enough:
      betaHat = np.sqrt(beta1_tilde * beta2_tilde)
      xHat = betaHat / 2
      r1 = beta * xHat
      r2 = -xHat**2
    else:
      r1, r2 = d1, d2
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


# ToDo: encapsulate environment settings in environments
def bandit(policy='cb', time_horizon=50, n=5, sigma_tol=1, sigma_upper=1.,
           env='coop'):
  a2 = 1  # Other player always plays collab
  X0 = np.zeros((0, 1))  # Will contain history of sigmas
  X1 = np.zeros((0, 1))
  y = np.zeros(0)
  y0 = np.zeros(0)  # Will contain history of rewards
  y1 = np.zeros(0)
  lm0 = DecisionTreeRegressor(max_depth=2, min_samples_split=0.2)
  lm1 = DecisionTreeRegressor(max_depth=2, min_samples_split=0.2)
  close_enough_lst = []

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
    bias_2 = np.random.uniform(0.0, 0.0)
    sigma = np.random.uniform(0, sigma_upper)

    if t < time_horizon:  # Choose at random in early stages
      a1 = np.random.choice(2)
    else:  # Thompson sampling
      if policy in ['cb', 'mab']:
        if policy == 'cb':
          lm0.fit(X0, y0)
          lm1.fit(X1, y1)
          q0 = lm0.predict([[sigma]])
          q1 = lm1.predict([[sigma]])
        elif policy == 'mab':
          q0 = y0.mean()
          q1 = y1.mean()
        if q1 > q0:
          if np.random.random() < 0.95:
            a1 = 1
          else:
            a1 = 0
        else:
          if np.random.random() < 0.95:
            a1 = 0
          else:
            a1 = 1
      elif policy=='ind':
        a1 = 0
      elif policy=='coop':
        a1 = 1

    # Observe reward
    bias_2_1 = np.array([[-5, 0], [-5, 0]])*bias_2
    bias_2_2 = np.array([[5, 0], [5, 0]])*bias_2

    if env == 'nash':
      r1, _, close_enough_ = alternating(a1, a2, u1_mean=true_u1_mean, u2_mean=true_u2_mean, bias_2_1=bias_2_1,
                                         bias_2_2=bias_2_2, sigma_u=0, sigma_x=sigma, n=n, sigma_tol=sigma_tol)
    elif env == 'coop':
      r1, _, close_enough_ = coop_bargaining(a1, a2, beta=5, sigma=sigma, tau=sigma_tol)

    # Update history
    if a1:
      X1 = np.vstack((X1, [sigma]))
      y1 = np.hstack((y1, r1))
    else:
      X0 = np.vstack((X0, [sigma]))
      y0 = np.hstack((y0, r1))
    y = np.hstack((y, r1))
    close_enough_lst.append(close_enough_)

  # lm0.fit(X0, y0)
  # lm1.fit(X1, y1)
  # xrange_ = np.linspace(0, sigma_upper, 100).reshape(-1, 1)
  # y0_hat = lm0.predict(xrange_)
  # y1_hat = lm1.predict(xrange_)
  # # plt.scatter(X0, y0, label='ind payoffs')
  # plt.plot(xrange_, y0_hat, label='ind')
  # # plt.scatter(X1, y1, label='coop payoffs')
  # plt.plot(xrange_, y1_hat, label='coop')
  # plt.xlabel('sigma')
  # plt.ylabel('Estimated expected payoff')
  # plt.title('Decision tree estimates of conditional rewards\nunder each reporting policy')
  # plt.legend()
  # plt.show()
  return y, close_enough_lst


def compare_policies(plot_name, replicates=10, time_horizon=50, n_private_obs=5, sigma_tol=1, sigma_upper=1.):
  r1_list_cb = []
  r1_list_mab = []
  r1_list_ind = []
  r1_list_coop = []
  close_list_coop = []
  for _ in range(replicates):
    r1_list_cb_rep, _ = bandit(policy='cb', n=n_private_obs, time_horizon=time_horizon, sigma_tol=sigma_tol,
                               sigma_upper=sigma_upper)
    r1_list_cb.append(r1_list_cb_rep)
    r1_list_mab_rep, _ = bandit(policy='mab', n=n_private_obs, time_horizon=time_horizon, sigma_tol=sigma_tol,
                                sigma_upper=sigma_upper)
    r1_list_mab.append(r1_list_mab_rep)
    r1_list_ind_rep, _ = bandit(policy='ind', n=n_private_obs, time_horizon=time_horizon, sigma_tol=sigma_tol,
                                sigma_upper=sigma_upper)
    r1_list_ind.append(r1_list_ind_rep)
    r1_list_coop_rep, close_coop = bandit(policy='coop', n=n_private_obs, time_horizon=time_horizon,
                                          sigma_tol=sigma_tol, sigma_upper=sigma_upper)
    r1_list_coop.append(r1_list_coop_rep)
    close_list_coop.append(np.mean(close_coop))

  print('prop close enough coop: {}'.format(np.mean(close_list_coop)))

  data = {'cb': np.cumsum(r1_list_cb, axis=1),
          'mab': np.cumsum(r1_list_mab, axis=1),
          'ind': np.cumsum(r1_list_ind, axis=1),
          'coop': np.cumsum(r1_list_coop, axis=1),
          'timepoint': np.arange(len(r1_list_coop[0]))}

  cb_mean, cb_se = data['cb'][:, -1].mean(), data['cb'][:, -1].std() / np.sqrt(replicates)
  mab_mean, mab_se = data['mab'][:, -1].mean(), data['mab'][:, -1].std() / np.sqrt(replicates)

  print('cb: {} mab: {}'.format((cb_mean, cb_se), (mab_mean, mab_se)))

  sns.lineplot(x=[i for _ in range(replicates) for i in
                  range(data['cb'].shape[1])], y=np.hstack(data['cb']),
              label='cb')
  sns.lineplot(x=[i for _ in range(replicates) for i in
                  range(data['mab'].shape[1])], y=np.hstack(data['mab']),
               label='mab', color='k')
  sns.lineplot(x=[i for _ in range(replicates) for i in
                range(data['cb'].shape[1])], y=np.hstack(data['ind']),
            label='ind')
  sns.lineplot(x=[i for _ in range(replicates) for i in
                range(data['cb'].shape[1])], y=np.hstack(data['coop']),
            label='coop', color='r')
  plt.legend()
  plt.title('Tolerance={} Maximum sigma={}'.format(sigma_tol, sigma_upper))
  plt.xlabel('Time')
  plt.ylabel('Cumulative reward')
  if plot_name is not None:
    plt.savefig('{}.png'.format(plot_name))
  else:
    plt.show()
  return


if __name__ == "__main__":
  sigma_tol_list = [2]
  for sigma_tol in sigma_tol_list:
    compare_policies(None, replicates=200, n_private_obs=2,
                     time_horizon=500, sigma_tol=sigma_tol, sigma_upper=1)
