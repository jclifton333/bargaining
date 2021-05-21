import numpy as np
from scipy.special import expit
import pdb


def decisive_conflict_rep(p_fail_list, p_threat, D_list, T):
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


def decisive_conflict(p_fail_list, T=100, n_rep=1e7, p_threat=0.1):
  D_list = np.array([expit(0.1*(t-30)) for t in range(T)])
  u_mean = 0.
  n_rep = int(n_rep)
  for _ in range(n_rep):
    u = decisive_conflict_rep(p_fail_list, p_threat, D_list, T)
    u_mean += u / n_rep
  print(u_mean)


if __name__ == "__main__":
  T = 100
  p_fail_0 = np.ones(T) * 0.1
  p_fail_0[0] = 0.1
  decisive_conflict(p_fail_list=p_fail_0, T=T)
  p_fail_1 = np.zeros(T)
  p_fail_1[0] = 0.01
  decisive_conflict(p_fail_list=p_fail_1, T=T)






