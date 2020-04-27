import numpy as np
from ultimatum import generate_ultimatum_data
from bayes_opt import BayesianOptimization
from scipy.optimize import minimize, Bounds
import pdb
np.set_printoptions(suppress=True)


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
    return -log_lik + penalty*(1.1*theta[2]**2 + 0.9*theta[2]**2)

  res = minimize(ll, x0=[0.1, 900., 0.01, 1.00], method='trust-constr',
                 bounds=Bounds([0., 0., 0., 0.], [1, 1000., 10., 10.]))
  return res.x.round(2)


def optimize_against_policy(t, f_list, prob_list):
  split_grid = np.linspace(0, 1, 20)
  best_val = -float('inf')
  best_s = None
  for s in split_grid:
    sval = 0.
    for f, prob in zip(f_list, prob_list):
      val = (1-s)*(s - f*(s < 0.5-t) > 0)*prob
      sval += val / len(f_list)
    if sval > best_val:
      best_s = s
      best_val = val
  return best_s, best_val


def adversary(actions, splits, t, temp, f_list, p_list):
  MULTIPLIER = 10.
  log_liks = [big_model_lik(actions, splits, t, temp, f, p)
              for f, p in zip(f_list, p_list)]
  liks = np.array([np.exp(ll) for ll in log_liks])
  const = liks.sum()
  probs = liks / const
  _, mean_val = optimize_against_policy(t, f_list, probs)

  def objective(theta):
    f0, p0 = theta[0], theta[1]
    new_lik = big_model_lik(actions, splits, t, temp, f0, p0)
    # TODO: think about this objective
    new_s, _ = optimize_against_policy(t, [f0], [1.0])
    new_val = np.sum([(1 - new_s)*(new_s - f*(new_s < 0.5-t) > 0)*prob
                       for f, prob in zip(f_list, probs)])
    avg_dist = np.mean([np.abs(f0-f) + np.abs(p0-p)
                        for f, p in zip(f_list, p_list)])
    cost = np.log(np.max((mean_val- new_val, 0.001))) + new_lik \
        - np.log(np.exp(new_lik) + const) + np.log(avg_dist/2)
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
  return best_param


def repeated_adversary(num_iter, actions, splits, t, temp, f, p):
  initial_s, initial_val = optimize_against_policy(t, [f], [1.])
  f_list = [f]
  p_list = [p]
  for k in range(num_iter):
    new_param = adversary(actions, splits, t, temp, f_list, p_list)
    f_list.append(new_param[0])
    p_list.append(new_param[1])

  # Get new policy 
  new_params = [(f_, p_) for f_, p_ in zip(f_list, p_list)]
  log_liks = [big_model_lik(actions, splits, t, temp, f, p)
              for f, p in zip(f_list, p_list)]
  liks = np.array([np.exp(ll) for ll in log_liks])
  const = liks.sum()
  probs = liks / const
  new_s, new_val = optimize_against_policy(t, f_list, probs)

  return new_params, initial_s, new_s


if __name__ == "__main__":
  N = 100

  def real_ev(sp):
    return sp - (sp < 0.4)

  def real_policy(sp, st):
    num = np.exp(real_ev(sp))
    prob = num / (1 + num)
    return np.random.binomial(1, p=prob)

  splits, actions, _ = generate_ultimatum_data(real_policy, n=N)
  thetaHat = max_lik_big_model(actions, splits, penalty=0.05)
  new_params = repeated_adversary(5, actions, splits, thetaHat[0], thetaHat[1],
                                  thetaHat[2], thetaHat[3])
  print(new_params)
