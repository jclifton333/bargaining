import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import pdb


def simple(intervention=0, n_rep=1e6, S=0.5, D0_distribution=(0.9, 0.1)):
  D0_list = [0.01, 1.]  # Early costs
  D1_list = [1.]  # Late costs # ToDo: should depend on D0
  D1_distribution = [1.]

  P0 = 0.1 / np.power(10, intervention)
  P1_list = [0.1 / np.power(10, intervention), 0.1]
  P1_distribution = [0.0, 1.0]

  u_mean = 0.
  n_rep = int(n_rep)
  for _ in range(n_rep):
    u = 0.

    P1 = np.random.choice(P1_list, p=P1_distribution)
    D0 = np.random.choice(D0_list, p=D0_distribution)
    D1 = np.random.choice(D1_list, p=D1_distribution)

    # Time 0
    if np.random.random() < P0:  # Conflict
      hegemon = True  # Assuming hegemon emerges from conflict
      u += D0
    else:  # No conflict
      if np.random.random() < S:
        hegemon = True
      else:
        hegemon = False

    # Time 1
    if not hegemon:
      if np.random.random() < P1:  # Conflict
        u += D1

    u_mean += u / n_rep

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
  n_rep = 1e5
  S_list = np.linspace(0, 1, 5)
  D0_probs = np.linspace(0, 1, 5)
  u_diff_mat = np.zeros((5, 5))
  for S_ix, S in enumerate(np.linspace(0, 1, 5)):
    for D0_ix, D0_prob in enumerate(D0_probs):
      D0_distribution = (1-D0_prob, D0_prob)
      u0 = simple(intervention=0, n_rep=n_rep, S=S, D0_distribution=D0_distribution)
      u1 = simple(intervention=1, n_rep=n_rep, S=S, D0_distribution=D0_distribution)
      u_diff_rep = u1 - u0
      u_diff_mat[S_ix, D0_ix] = u_diff_rep
  plt.imshow(u_diff_mat, cmap='viridis')
  plt.xlabel('Sprob')
  plt.ylabel('D0largeprob')
  plt.colorbar()
  plt.show()






