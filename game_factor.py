import docplex.mp
import docplex.mp.model as dcpm
import numpy as np
import pdb


def joint_action_metric(Ui, i, j):
  # Measure maximum regret of continuing to play i if counterpart deviates from j
  uij = Ui[i, j]
  nAi, nAj = Ui.shape
  max_regret = 0.
  for jprime in range(nAj):
    if jprime != j:
      jprime_payoffs = Ui[:, jprime]
      best_response_to_jprime = jprime_payoffs.argmax()
      regret = Ui[best_response_to_jprime, jprime] - uij
      if regret > max_regret:
        max_regret = regret
  return max_regret


def onehot(ix, n):
  v = np.zeros(n)
  v[ix] = 1
  return v


def extract_additive_matrices_from_parameter(beta, nA1, nA2):
  player_1_param = beta[:nA1]
  player_2_param = beta[nA1:(nA1+nA2)]
  player_1_matrix = np.array([player_1_param for _ in range(nA2)]).T
  player_2_matrix = np.array([player_2_param for _ in range(nA1)])
  return player_1_matrix, player_2_matrix


def fit_additive_approx_with_solver(U1, U2):
  nA1, nA2 = U1.shape
  nA = nA1 * nA2

  model = dcpm.Model(name='model')

  vars = {}

  # Add variables for each player
  for i in range(nA1):
    vars[(0, 0, i)] = model.continuous_var(name=f'{(0, 0, i)}', lb=0)
    vars[(1, 0, i)] = model.continuous_var(name=f'{(1, 0, i)}', lb=0)
    for j in range(nA2):
      if i == 0:
        vars[(0, 1, j)] = model.continuous_var(name=f'{(0, 1, j)}', lb=0)
        vars[(1, 1, j)] = model.continuous_var(name=f'{(1, 1, j)}', lb=0)
      vars[(2, i, j)] = model.continuous_var(name=f'{(2, i, j)}', lb=0)

  cost1 = model.sum( (U1[i, j] - vars[(0, 0, i)] - vars[(0, 1, j)] - vars[(2, i, j)])**2 for i in range(nA1)
                      for j in range(nA2))
  model.add_kpi(cost1, 'cost1')

  cost2 = model.sum( (U2[i, j] - vars[(1, 0, i)] - vars[(1, 1, j)] - vars[(2, i, j)])**2 for i in range(nA1)
                      for j in range(nA2))
  model.add_kpi(cost2, 'cost2')

  model.minimize(cost1 + cost2)
  sol = model.solve(url=None, key=None)

  # Get variables
  U11_hat = np.zeros_like(U1)
  U12_hat = np.zeros_like(U1)
  U1_hat = np.zeros_like(U1)

  for i in range(nA1):
    for j in range(nA2):
      U11_hat[i, j] = float(sol.get_value(f'{(0, 0, i)}'))
      U12_hat[i, j] = float(sol.get_value(f'{(0, 1, j)}'))
      U1_hat[i, j] = float(sol.get_value(f'{(2, i, j)}'))

  pdb.set_trace()
  return U11_hat, U12_hat, U1_hat


def fit_additive_approximation(U1, U2):
  nA1, nA2 = U1.shape
  nA = nA1 * nA2
  X = np.zeros((0, nA1 + nA2 + nA))
  y = np.zeros(0)

  # Collect player 1 data
  k = 0
  for i in range(nA1):
    for j in range(nA2):
      k += 1
      a1 = onehot(i, nA1)
      a2 = onehot(j, nA2)
      a = onehot(k, nA)
      X = np.vstack((X, np.concatenate((a1, a2, a))))
      y = np.hstack((y, U1[i, j]))

  # Collect player 2 data
  k = 0
  for i in range(nA1):
    for j in range(nA2):
      k += 1
      a1 = onehot(i, nA1)
      a2 = onehot(j, nA2)
      a = onehot(k, nA)
      X = np.vstack((X, np.concatenate((a1, a2, a))))
      y = np.hstack((y, U2[i, j]))

  XpX = np.dot(X.T, X)
  XpXinv = np.linalg.inv(XpX + 0.001*np.eye(nA1+nA2))
  Xpy = np.dot(X.T, y)
  beta = np.dot(XpXinv, Xpy)

  p1_matrix, p2_matrix = extract_additive_matrices_from_parameter(beta, nA1, nA2)

  return p1_matrix, p2_matrix


if __name__ == "__main__":
  bpm1 = np.array([[2, 1], [1, 0]])
  bpm2 = np.array([[2, 1], [1, 0]])

  ipd_bpm1 = np.array([[3, 2, 2, 1], [2, 1, 1, 0], [2, 1, 2, 1], [1, 0, 1, 0]])

  U11_hat, U12_hat, U1_hat = fit_additive_approx_with_solver(bpm1, bpm2)
