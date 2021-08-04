import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import pdb


def simple(intervention=0, S=0.5, D0=0.01, P1_intervention_multiplier=1.):
  D1 = 1.
  P0 = 0.1 / np.power(10, intervention)
  P1 = 0.1 * np.power(P1_intervention_multiplier, intervention)

  u_mean = P0 * D0 + (1 - P0) * ((1 - S) * P1 * D1)
  return u_mean


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
  n_rep = 1e4
  axis_size = 20
  S = 0.2
  # S_list = np.linspace(0, 1, axis_size)
  D0_list = np.logspace(-3, 0, axis_size)
  P1_intervention_multiplier_list = 1 - np.logspace(-2, np.log10(0.9), axis_size)
  u_diff_mat = np.zeros((axis_size, axis_size))
  # for S_ix, S in enumerate(np.linspace(0, 1, axis_size)):
  for P1_ix, P1_intervention_multiplier in enumerate(P1_intervention_multiplier_list):
    for D0_ix, D0 in enumerate(D0_list):
      u0 = simple(intervention=0, S=S, D0=D0)
      u1 = simple(intervention=1, S=S, D0=D0, P1_intervention_multiplier=P1_intervention_multiplier)
      u_diff_rep = u0 - u1
      u_diff_mat[axis_size - P1_ix - 1, D0_ix] = u_diff_rep
  plt.imshow(u_diff_mat, cmap='viridis', extent=(-3, 0, -2, np.log10(0.9)))
  plt.ylabel('log (Late intervention effectiveness)')
  plt.xlabel('log (Expected early disvalue)')
  plt.colorbar(label='Expected value of intervention')
  plt.show()






