import numpy as np


def generate_ultimatum_data(policy, n=100):
  """
  :param policy: maps split to 0 (reject) or 1 (accept)
  """
  splits = np.random.uniform(size=n)
  actions = [policy(s) for s in splits]
  return actions


def simple_boltzmann_ll(r, splits, actions, temp=1.):
  """
  Log likelihood of actions given splits, myopic Boltmzann
  policy,
  i.e. P(a | s) \propto exp( r*s )
  """
  log_lik = 0.
  for s, a in zip(splits, actions):
    u = r*s*temp
    if a:
      log_lik += (u - (1 + np.exp(u)))
    else:
      log_lik += 1 - (1 + np.exp(u))
  return log_lik


def fairness_boltzmann_ll(r, f, splits, actions, temp=1.):
  """
  P(a | s) \propto exp( r*s - f*(2s-1)**2)
  """
  log_lik = 0.
  for s, a in zip(splits, actions):
    u = temp*(r*s - f*(2*s-1)**2)
    if a:
      log_lik += (u - (1 + np.exp(u)))
    else:
      log_lik += 1 - (1 + np.exp(u))
  return log_lik

def repeated_ll(r, splits, actions, temp=1.):
  pass
