import docplex.mp
import docplex.mp.model as dcpm
import numpy as np
import pdb
np.set_printoptions(precision=2, suppress=True)


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


def fit_additive_approx_with_solver(U1, U2, weight=1., residual_weight=1.):
  nA1, nA2 = U1.shape
  nA = nA1 * nA2

  model = dcpm.Model(name='model')

  vars = {}

  # Add variables for each player
  for i in range(nA1):
    vars[(0, 0, i)] = model.continuous_var(name=f'{(0, 0, i)}', lb=-2)
    vars[(1, 0, i)] = model.continuous_var(name=f'{(1, 0, i)}')
    for j in range(nA2):
      if i == 0:
        vars[(0, 1, j)] = model.continuous_var(name=f'{(0, 1, j)}', lb=-2)
        vars[(1, 1, j)] = model.continuous_var(name=f'{(1, 1, j)}', lb=-2)
      vars[(2, i, j)] = model.continuous_var(name=f'{(2, i, j)}', lb=0)
      # vars[(3, i, j)] = model.continuous_var(name=f'{(3, i, j)}', lb=0)
      # vars[(4, i, j)] = model.continuous_var(name=f'{(4, i, j)}', lb=0)

      # Dummy variables for l1 penalty
      vars[(5, i, j)] = model.continuous_var(name=f'{(5, i, j)}', lb=0)
      vars[(6, i, j)] = model.continuous_var(name=f'{(6, i, j)}', lb=0)

  cost1 = model.sum( (U1[i, j] - vars[(0, 0, i)] - vars[(0, 1, j)] - vars[(2, i, j)])**2 for i in range(nA1)
                      for j in range(nA2))
  cost2 = model.sum( (U2[i, j] - vars[(1, 1, i)] - vars[(1, 0, j)] - vars[(2, i, j)])**2 for i in range(nA1)
                      for j in range(nA2))

  # cost3 = model.sum(vars[2, i, j]**2 for i in range(nA1) for j in range(nA2))
  cost3 = model.sum(vars[5, i, j] + vars[6, i, j] for i in range(nA1) for j in range(nA2))
  # cost4 = model.sum(vars[3, i, j]**2 for i in range(nA1) for j in range(nA2))
  # cost5 = model.sum(vars[4, i, j] ** 2 for i in range(nA1) for j in range(nA2))

  # Constraints for l1 dummy variables
  for i in range(nA1):
    for j in range(nA2):
      model.add_constraint(vars[5, i, j] - vars[6, i, j] - vars[2, i, j] == 0)

  model.minimize(cost1 + cost2 + weight*cost3)
  sol = model.solve(url=None, key=None)

  # Get variables
  U11_hat = np.zeros_like(U1).astype(float)
  U12_hat = np.zeros_like(U1).astype(float)
  U21_hat = np.zeros_like(U1).astype(float)
  U22_hat = np.zeros_like(U1).astype(float)
  U1_hat = np.zeros_like(U1).astype(float)

  for i in range(nA1):
    for j in range(nA2):
      U11_hat[i, j] = float(sol.get_value(f'{(0, 0, i)}'))
      U12_hat[i, j] = float(sol.get_value(f'{(0, 1, j)}'))
      U21_hat[i, j] = float(sol.get_value(f'{(1, 0, i)}'))
      U22_hat[i, j] = float(sol.get_value(f'{(1, 1, j)}'))
      U1_hat[i, j] = float(sol.get_value(f'{(2, i, j)}'))

  return U11_hat, U12_hat, U21_hat, U22_hat, U1_hat


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
  ipd_bpm2 = np.array([[3, 2, 2, 1], [2, 1, 1, 0], [2, 1, 2, 1], [1, 0, 1, 0]])

  cg1 = np.array([[0, 0, 2, 2], [0, 1, 2, 2], [0, 0, 0, 0], [0, 0, 0, 1]])
  cg2 = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [2, 2, 0, 0], [2, 2, 0, 1]])

  U11_hat, U12_hat, U21_hat, U22_hat, U1_hat = fit_additive_approx_with_solver(cg1, cg2, weight=1,
                                                                               residual_weight=0.0)
  print(U21_hat)
  print(U1_hat)
  print(U21_hat + U22_hat + U1_hat)
