import numpy as np
import warnings
from ultimatum import generate_ultimatum_data
from bayes_opt import BayesianOptimization
from scipy.optimize import minimize, Bounds
from scipy.special import expit
import pdb
np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore", category=UserWarning)


def big_model_lik(actions, splits, t, temp, f, p):
  log_lik = 0.
  for a, s in zip(actions, splits):
    soft_indicator_num = np.exp((0.5-t-s)*temp)
    soft_indicator = soft_indicator_num / (soft_indicator_num + 1)
    log_odds_a = s - f*soft_indicator
    odds_r = np.exp(p*soft_indicator)
    log_prob_a = log_odds_a - np.log(odds_r + np.exp(log_odds_a))
    log_prob_r = np.log(odds_r) - np.log(odds_r + np.exp(log_odds_a))
    log_lik += a*log_prob_a + (1-a)*log_prob_r
  return log_lik


def max_lik_big_model(actions, splits, penalty=0.01):
  def ll(theta):
    log_lik = big_model_lik(actions, splits, theta[0],
                            theta[1], theta[2], theta[3])
    return -log_lik + penalty*(theta[0]**2 + 1.1*theta[2]**2 + 0.9*theta[2]**2)

  res = minimize(ll, x0=[0.1, 1000., 0.01, 1.00], method='trust-constr',
                 bounds=Bounds([0., 0., 0., 0.], [1, 1500., 10., 10.]))
  print(res.success)
  return res.x.round(2)


def optimize_against_policy(t, f_list, prob_list):
  split_grid = np.linspace(0, 1, 20)
  best_val = -float('inf')
  best_s = None
  for s in split_grid:
    sval = 0.
    for f, prob in zip(f_list, prob_list):
      val = (1-s)*((s - f*(s < 0.5-t)) > 0)*prob
      sval += val / len(f_list)
    if sval > best_val:
      best_s = s
      best_val = sval
  return best_s, best_val


def adversary(actions, splits, t, temp, f_list, p_list):
  MULTIPLIER = 10.
  liks, const, probs = probs_from_params(actions, splits, t, temp,
                                         f_list, p_list)
  old_s, mean_val = optimize_against_policy(t, f_list, probs)

  def costs(theta):
    f0, p0 = theta[0], theta[1]
    new_lik = big_model_lik(actions, splits, t, temp, f0, p0)
    # TODO: think about this objective
    new_val = (1 - old_s)*((old_s - f0*(old_s < 0.5-t)) > 0)
    min_dist = np.min([np.abs(f0-f) + np.abs(p0-p)
                        for f, p in zip(f_list, p_list)])
    dist_indicator = np.exp((0.1-min_dist)*1000)
    payoff_diff = mean_val- new_val
    payoff_diff_indicator = np.exp((-payoff_diff*1000))
    log_denom = np.log(np.exp(new_lik) + const)
    return payoff_diff, new_lik, log_denom, dist_indicator, \
        payoff_diff_indicator

  def objective(theta):
    payoff_diff, new_lik, log_denom, dist_indicator, payoff_diff_indicator = \
        costs(theta)
    cost = np.log(np.max((payoff_diff, 0.00001))) + new_lik - log_denom - \
      dist_indicator - payoff_diff_indicator
    return -cost

  # bounds = {'f0': (0., 10.), 'p0': (0., 10.)}
  # explore = {'f0': [f], 'p0': [p]}
  # bo = BayesianOptimization(objective, bounds)
  # bo.explore(explore)
  # bo.maximize(init_points=2, n_iter=5)
  # best_param = bo.res['max']['max_params']
  res = minimize(objective, x0=[0.5, 0.5], method='trust-constr',
                 bounds=Bounds([0., 0.], [10., 10.]))
  best_param = res.x.round(2)
  best_cost = objective(best_param)
  _, _, _, dist_indicator, payoff_indicator = costs(best_param)
  feasible = np.max((payoff_indicator, dist_indicator)) < 1e3
  return best_param, best_cost, feasible


def probs_from_params(actions, splits, t, temp, f_list, p_list):
  log_liks = [big_model_lik(actions, splits, t, temp, f_, p_)
              for f_, p_ in zip(f_list, p_list)]
  liks = np.array([np.exp(ll) for ll in log_liks])
  const = liks.sum()
  probs = liks / const
  return liks, const, probs


def repeated_adversary(num_iter, actions, splits, t, temp, f, p):
  initial_s, initial_val = optimize_against_policy(t, [f], [1.])
  f_list = [f]
  p_list = [p]
  k = 0
  feasible = True
  while k < num_iter and feasible:
    new_param, new_cost, feasible = adversary(actions, splits, t, temp, f_list, p_list)
    if feasible:
      f_list.append(new_param[0])
      p_list.append(new_param[1])
    liks, const, probs = probs_from_params(actions, splits, t, temp, f_list, p_list)
    new_s, new_val = optimize_against_policy(t, f_list, probs)
    k += 1

  # Get new policy 
  new_params = [(f_, p_) for f_, p_ in zip(f_list, p_list)]
  liks, const, probs = probs_from_params(actions, splits, t, temp, f_list, p_list)
  return new_params, initial_s, new_s, probs


if __name__ == "__main__":
  N = 500
  # TODO: also need to incorporate uncertainty in t

  def real_ev(sp):
    return sp - (sp < 0.4)

  def real_policy(sp, st):
    num = np.exp(real_ev(sp))
    prob = num / (1 + num)
    return np.random.binomial(1, p=prob)

  splits, actions, _ = generate_ultimatum_data(real_policy, n=N)
  thetaHat = max_lik_big_model(actions, splits, penalty=0.05)
  print(thetaHat)
  new_params = repeated_adversary(5, actions, splits, thetaHat[0], thetaHat[1],
                                  thetaHat[2], thetaHat[3])
  print(new_params)
