import numpy as np
from scipy.optimize import minimize, Bounds
import pdb


def generate_ultimatum_data(policy, n=100):
  """
  :param policy: maps split to 0 (reject) or 1 (accept)
  """
  splits = np.random.uniform(size=n)
  actions = [policy(s) for s in splits]
  return splits, actions


def simple_boltzmann_ll(r, splits, actions, temp=1.):
  """
  Log likelihood of actions given splits, myopic Boltmzann
  policy,
  i.e. P(a | s) \propto exp( r*s )
  """
  log_lik = 0.
  for s, a in zip(splits, actions):
    u = s*r*temp
    if a:
      log_lik += (u - (1 + np.exp(u)))
    else:
      log_lik += 1 - (1 + np.exp(u))
  return -log_lik + r**2


def fairness_boltzmann_ll(r, f, t, splits, actions, temp=1.):
  """
  P(a | s) \propto exp( r*s - f*(2s-1)**2)
  """
  log_lik = 0.
  for s, a in zip(splits, actions):
    u = temp*(s*r - f*(s< 0.5-t))
    if a:
      log_lik += (u - (1 + np.exp(u)))
    else:
      log_lik += 1 - (1 + np.exp(u))
  return -log_lik + r**2 + f**2 + t**2

def repeated_ll(r, t, p, splits, actions, temp=1.):
  """
  Account for subject's beliefs about payoffs in the next game,
  given what they do now.
  P(~a | s) \propto exp( 1{s < 0.5 - t}*p )
  """
  log_lik = 0.
  for s, a in zip(splits, actions):
    u_rej = temp*(s < 0.5-t)*p
    u_acc = temp*(2*s*r)
    if a:
      log_lik += u_acc - (np.exp(u_rej) + np.exp(u_acc))
    else:
      log_lik += u_rej - (np.exp(u_rej) + np.exp(u_acc))
  return -log_lik + r**2 + t**2 + p**2


def maximize_all_likelihoods(splits, actions, temp=1.):
  def simple(r):
    return simple_boltzmann_ll(r, splits, actions, temp=temp)
  def fairness(theta):
    return fairness_boltzmann_ll(theta[0], theta[1], theta[2],
                                 splits, actions,
                                 temp=temp)
  def repeated(theta):
    return repeated_ll(theta[0], theta[1], theta[2], splits, actions,
                       temp=temp)

  simple_res = minimize(simple, x0=[0.5], method='trust-constr',
                        bounds=Bounds([-1], [1]))
  fairness_res = minimize(fairness, x0=np.ones(3)*0.5,
                          method='trust-constr',
                          bounds=Bounds([-1, -1, -1], [1, 1, 1]))
  repeated_res = minimize(repeated,
                          x0=np.ones(3)*0.5,
                          method='trust-constr',
                          bounds=Bounds([-1, -1, -1], [1, 1, 1]))
  print(simple_res.x[0], fairness_res.x[0], repeated_res.x[0])
  return


if __name__ == "__main__":
  real_policy = lambda s: s > 0.45
  splits, actions = generate_ultimatum_data(real_policy)
  maximize_all_likelihoods(splits, actions)







