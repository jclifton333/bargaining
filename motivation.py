import numpy as np
import nashpy
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


def asymmetric_replicator(A, B, n, reps=10, step_size=0.01):
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
    p = p + step_size * p * (Ap - qAp)
    q = q + step_size * q * (Bq - pBq)
    print(p.round(2))
    print(q.round(2))


def meta_bargaining_nash(miscoordination=0.):
  A, B = meta_bargaining_payoff_matrix(miscoordination=miscoordination)
  G = nashpy.Game(A, B)
  eqs = [eq for eq in G.support_enumeration()]
  print(eqs)


def meta_bargaining_asymmetric_replicator(miscoordination=0., n_reps=10, grid_size=4):
  A, B = meta_bargaining_payoff_matrix(miscoordination=miscoordination, grid_size=grid_size)
  asymmetric_replicator(A, B, grid_size*2, reps=n_reps)


if __name__ == "__main__":
  meta_bargaining_nash()



