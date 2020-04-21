import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.special import expit
import pymc3 as pm
import matplotlib.pyplot as plt
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
      log_lik += (u - np.log(1 + np.exp(u)))
    else:
      log_lik += 1 - np.log(1 + np.exp(u))
  return -log_lik + r**2


def model_uncertainty(splits, actions, temp=1.):
  with pm.Model() as repeated_model:
    r = pm.Normal('r', mu=0, sd=1)
    p = pm.Gamma('p', alpha=1, beta=1)
    t = pm.Uniform('t', lower=0, upper=0.5)
    u_rej = (splits < 0.5-t)*p*r
    u_acc = 2*splits*r
    odds_rej = np.exp(u_rej)
    odds_acc = np.exp(u_acc)
    p = odds_acc / (odds_rej + odds_acc)
    a = pm.Binomial('a', 1, p, observed=actions)
    fitted = pm.fit(method='advi')
    trace_repeated = fitted.sample(2000)

  # with pm.Model() as simple_model:
  #   r = pm.Normal('r', mu=0, sd=1)
  #   p = np.exp(r*splits) / (1 + np.exp(r*splits))
  #   a = pm.Binomial('a', 1, p, observed=actions)
  #   trace_simple = pm.sample(2000, init='map')

  with pm.Model() as fairness_model:
    r = pm.Normal('r', mu=0, sd=1)
    t = pm.Uniform('t', lower=0, upper=0.5)
    f = pm.Normal('f', mu=0, sd=1)
    odds = np.exp(splits*r - f*(splits< 0.5-t))
    p = odds / (1 + odds)
    a = pm.Binomial('a', 1, p, observed=actions)
    fitted = pm.fit(method='advi')
    trace_fairness = fitted.sample(2000)

  model_dict = dict(zip([fairness_model, repeated_model],
                        [trace_fairness, trace_repeated]))
  comp = pm.compare(model_dict, method='BB-pseudo-BMA')
  return trace_fairness, trace_repeated, comp


def simple_bayes(splits, actions, temp=1.):
  with pm.Model() as model:
    r = pm.Normal('r', mu=0, sd=1)
    p = np.exp(r*splits) / (1 + np.exp(r*splits)) 
    a = pm.Binomial('a', 1, p, observed=actions)
    trace = pm.sample(20000, init='map')
  return trace



def fairness_boltzmann_ll(r, f, t, splits, actions, temp=1.):
  """
  P(a | s) \propto exp( r*s - f*(2s-1)**2)
  """
  log_lik = 0.
  for s, a in zip(splits, actions):
    u = temp*(s*r - f*(s< 0.5-t))
    if a:
      log_lik += (u - np.log(1 + np.exp(u)))
    else:
      log_lik += 1 - np.log(1 + np.exp(u))
  return -log_lik + r**2 + f**2 + t**2

def repeated_ll(r, t, p, splits, actions, temp=1.):
  """
  Account for subject's beliefs about payoffs in the next game,
  given what they do now.
  P(~a | s) \propto exp( 1{s < 0.5 - t}*p )
  """
  log_lik = 0.
  for s, a in zip(splits, actions):
    u_rej = temp*(s < 0.5-t)*p*r
    u_acc = temp*(2*s*r)
    if a:
      log_lik += u_acc - np.log(np.exp(u_rej) + np.exp(u_acc))
    else:
      log_lik += u_rej - np.log(np.exp(u_rej) + np.exp(u_acc))
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
                          bounds=Bounds([-1, -1, -3], [1, 1, 3]))
  print(simple_res.x[0], fairness_res.x[0], repeated_res.x[0])
  # print(repeated_res.x)
  return


if __name__ == "__main__":
  def real_policy(s):
    num = np.exp(s -5*(s < 0.5)*(1-s)**2)
    prob = num / (1 + num)
    return np.random.binomial(1, p=prob)
  # real_policy = lambda s: s > 0.4

  splits, actions = generate_ultimatum_data(real_policy, n=100)
  tf, tr, comp = model_uncertainty(splits, actions)

  recommended_actions = []
  evs = []
  wf, wr = comp['weight']
  s_range = np.linspace(0, 1, 20)
  for s in s_range:
    uf_0 = []
    ur_0 = []
    for pf in tf:
      uf = pf['r']*s - pf['f']*(s< 0.5-pf['t'])
      uf_0.append(uf)
    for pr in tr:
      ur = pr['r']*s
      ur_0.append(ur)
    ev = wf*np.mean(uf_0) + wr*np.mean(ur_0)
    evs.append(ev)
    recommended_actions.append(ev > 0)
  plt.plot(s_range, evs)
  plt.show()

    



  # maximize_all_likelihoods(splits, actions)








