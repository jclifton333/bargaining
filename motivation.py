import numpy as np
import nashpy
from itertools import product
import pdb


"""
https://www.pnas.org/content/111/Supplement_3/10810
Using replicator dynamics 
p_i' = p_i + p_i[ (A p)_i - q.Ap] 
q_j' = q_i + q_j[ (B q)_j - p.Bq] 
"""


def meta_bargaining_payoffs(m_i, w_i, m_j, w_j, miscoordination):
  prob_motivated = m_i * m_j
  motivated_payoff_i = prob_motivated * 3.0
  motivated_payoff_j = prob_motivated * 1.5
  if w_i == 'util':
    if w_j == 'util':
      u_i = motivated_payoff_i + (1 - prob_motivated) * 4.0
      u_j = motivated_payoff_j + (1 - prob_motivated) * 1.0
    elif w_j == 'ineq':
      u_i = motivated_payoff_i + (1 - prob_motivated) * miscoordination
      u_j = motivated_payoff_j + (1 - prob_motivated) * miscoordination
  elif w_i == 'ineq':
    if w_j == 'util':
      u_i = motivated_payoff_i + (1 - prob_motivated) * miscoordination
      u_j = motivated_payoff_j + (1 - prob_motivated) * miscoordination
    elif w_j == 'ineq':
      u_i = motivated_payoff_i + (1 - prob_motivated) * 2.0
      u_j = motivated_payoff_j + (1 - prob_motivated) * 2.0
  return u_i, u_j


def deliberation_payoff_matrix():
  """
  Strategies are (B(S), (Don't) deliberate, B(S) if other doesn't deliberate)
  Coding B as 0 and S as 1
  """
  bos_payoffs_i = np.array([[4, 0], [0, 2]])
  bos_payoffs_j = np.array([[1, 0], [0, 2]])
  compromise_payoff_i = 0.5*bos_payoffs_i[0, 0] + 0.5*bos_payoffs_i[1, 1]
  compromise_payoff_j = 0.5 * bos_payoffs_j[0, 0] + 0.5 * bos_payoffs_j[1, 1]
  deliberation_cost = 0.2
  n_strategy = 8
  payoff_matrix_i = np.zeros((n_strategy, n_strategy))
  payoff_matrix_j = np.zeros((n_strategy, n_strategy))

  strategies_i = list(product((0, 1), (0, 1), (0, 1)))
  strategies_j = list(product((0, 1), (0, 1), (0, 1)))

  i = 0
  for si_0, si_1, si_2 in strategies_i:
    j = 0
    for sj_0, sj_1, sj_2 in strategies_j:
      if si_1 == 0 and sj_1 == 0:
        payoff_matrix_i[i, j] = bos_payoffs_i[si_0, sj_0]
        payoff_matrix_j[i, j] = bos_payoffs_j[si_0, sj_0]
      elif si_1 == 0 and sj_1 == 1:
        payoff_matrix_i[i, j] = bos_payoffs_i[si_0, sj_2]
        payoff_matrix_j[i, j] = bos_payoffs_j[si_0, sj_2]
      elif si_1 == 1 and sj_1 == 0:
        payoff_matrix_i[i, j] = bos_payoffs_i[si_2, sj_0]
        payoff_matrix_j[i, j] = bos_payoffs_j[si_2, sj_0]
      elif si_1 == 1 and sj_1 == 1:
        payoff_matrix_i[i, j] = compromise_payoff_i - deliberation_cost
        payoff_matrix_j[i, j] = compromise_payoff_j - deliberation_cost
      j += 1
    i += 1

  return payoff_matrix_i, payoff_matrix_j



def meta_bargaining_payoff_matrix(grid_size=4, miscoordination=0.):
  """
  Player types are (welfare function, m) as m varies in [0.0, 0.2, ..., 0.8, 1.0]
  and welfare function varies in {ineq, util}.
  """
  n_strategy = grid_size * 2
  m_list = np.linspace(0, 1, grid_size)
  welfare_list = ['ineq', 'util']
  payoff_matrix_i = np.zeros((n_strategy, n_strategy))
  payoff_matrix_j = np.zeros((n_strategy, n_strategy))
  i = 0

  for w_i in welfare_list:
    for m_i in m_list:
      j = 0
      for w_j in welfare_list:
        for m_j in m_list:
          u_i, u_j = meta_bargaining_payoffs(m_i, w_i, m_j, w_j, miscoordination)
          payoff_matrix_i[i, j] = u_i
          payoff_matrix_j[i, j] = u_j
          j += 1
      i += 1
  return payoff_matrix_i, payoff_matrix_j


def clip_to_simplex(x):
  x = np.minimum(np.maximum(x, 0), 1)
  x = x / x.sum()
  return x


def asymmetric_replicator(A, B, n, reps=10, step_size=0.1):
  """
  p_i' = p_i + p_i[ (A p)_i - q.Ap]
  q_j' = q_i + q_j[ (B q)_j - p.Bq]
  """
  p = np.random.dirichlet(alpha=np.ones(n))
  q = np.random.dirichlet(alpha=np.ones(n))

  for _ in range(reps):
    Ap = np.dot(A, p)
    Bq = np.dot(B, q)
    qAp = np.dot(q, Ap)
    pBq = np.dot(p, Bq)

    # Get deltas
    delta_p = step_size * p * (Ap - qAp)
    delta_q = step_size * q * (Bq - pBq)
    print(np.dot(delta_p, delta_p))
    print(np.dot(delta_q, delta_q))

    # Update
    p = p + delta_p
    q = q + delta_q
    p = clip_to_simplex(p)
    q = clip_to_simplex(q)


def meta_bargaining_nash(miscoordination=0.):
  A, B = deliberation_payoff_matrix()
  G = nashpy.Game(A, B)
  eqs = [eq for eq in G.support_enumeration()]
  print(eqs)
  return eqs


def meta_bargaining_asymmetric_replicator(miscoordination=0., n_reps=100, grid_size=4):
  A, B = meta_bargaining_payoff_matrix(miscoordination=miscoordination, grid_size=grid_size)
  asymmetric_replicator(A, B, grid_size*2, reps=n_reps)


if __name__ == "__main__":
  eqs = meta_bargaining_nash()




