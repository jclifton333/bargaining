import numpy as np
from scipy.special import expit
import pdb


def decisive_rep(p_fail_list, p_threat, D_list, T):
  u = 0
  for t in range(T):
    rando = np.random.random()
    p_fail = p_fail_list[t]
    if rando < p_fail:
      rando1 = np.random.random()
      if rando1 < p_threat:
        u = D_list[t]
      else:
        u = 0
      break
  return u


def decisive(p_fail_list, p_fail_decay_fn, T=100, n_rep=1e6, p_threat=0.1, p_fail_init_mu=0.1, p_fail_init_var=0.03):
  D_list = np.array([expit(0.1*(t-30)) for t in range(T)])
  u_mean = 0.
  n_rep = int(n_rep)
  for _ in range(n_rep):
    p_fail_list = get_p_fail_list(T, p_fail_init_mu, p_fail_init_var, p_fail_decay_fn)
    u = decisive_rep(p_fail_list, p_threat, D_list, T)
    u_mean += u / n_rep
  print(u_mean)


def solve_for_beta_params(mu, sigma_sq):
  alpha = ((1 - mu) / sigma_sq - 1/mu) * mu**2
  beta = alpha*(1/mu - 1)
  return alpha, beta


def p_fail_sigmoid_decay(p_fail_initial_, t, midpoint, steepness):
  p_success_initial = 1 - p_fail_initial_
  p_success_t = p_success_initial * expit(steepness * (t - midpoint))
  p_fail_t = 1 - p_success_t
  return p_fail_t


def p_fail_step_decay(p_fail_initial_, t, p_fail_final, num_initial):
  if t < num_initial:
    return p_fail_initial_
  else:
    return p_fail_final


def p_fail_constant(p_fail_initial_, t):
  return p_fail_initial_


def get_p_fail_list(T, p_fail_initial_mean, p_fail_initial_var, decay_function):
  p_fail_initial_alpha, p_fail_initial_beta = solve_for_beta_params(p_fail_initial_mean,
                                                                    p_fail_initial_var)
  p_fail_initial = np.random.beta(a=p_fail_initial_alpha, b=p_fail_initial_beta)
  p_fail_list = [decay_function(p_fail_initial, t) for t in range(T)]
  return p_fail_list


if __name__ == "__main__":
  T = 100
  p_fail_0 = np.ones(T) * 0.1
  p_fail_0[0] = 0.1
  decisive(p_fail_list=p_fail_0, T=T)
  for cutoff in [1, 50, 80]:
    p_fail_decay_fn = lambda p, t: p_fail_step_decay(p, t, 0.01, cutoff)
    decisive(p_fail_list=p_fail_1, T=T)






