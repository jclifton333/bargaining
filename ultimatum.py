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


def model_uncertainty(splits, actions, temp=1., sd=1.):
  with pm.Model() as repeated_model:
    r = pm.Normal('r', mu=0, sd=sd)
    # p = pm.Gamma('p', alpha=1, beta=1)
    # t = pm.Beta('t', alpha=2, beta=5)
    odds_a = np.exp(r*(splits > 0.4))
    # odds_r = np.exp(p*(splits < 0.5-t/2))
    odds_r = 1
    p = odds_a / (odds_r + odds_a)
    a = pm.Binomial('a', 1, p, observed=actions)
    fitted = pm.fit(method='advi')
    trace_repeated = fitted.sample(2000)
    # trace_repeated = pm.sample(200000, step=pm.Slice(), chains=2, cores=4)


  # with pm.Model() as simple_model:
  #   r = pm.Normal('r', mu=0, sd=1)
  #   p = np.exp(r*splits) / (1 + np.exp(r*splits))
  #   a = pm.Binomial('a', 1, p, observed=actions)
  #   trace_simple = pm.sample(2000, init='map')

  with pm.Model() as fairness_model:
    r = pm.Normal('r', mu=0, sd=sd)
    t = pm.Beta('t', alpha=2, beta=5)
    f = pm.Normal('f', mu=0, sd=sd)
    b = pm.Normal('b', mu=0, sd=sd)
    odds = np.exp(temp*(splits*r - f*(splits< 0.5-t/2) - b*(splits > 0.5+t/2)))
    p = odds / (1 + odds)
    a = pm.Binomial('a', 1, p, observed=actions)
    fitted = pm.fit(method='advi')
    trace_fairness = fitted.sample(2000)
    # trace_fairness = pm.sample(200000, step=pm.Slice(), chains=2, cores=4)

  fairness_model.name = 'fair'
  repeated_model.name = 'repeated'
  model_dict = dict(zip([fairness_model, repeated_model],
                        [trace_fairness, trace_repeated]))
  comp = pm.compare(model_dict, ic='LOO', method='BB-pseudo-BMA')
  pdb.set_trace()
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
  # TODO: use models from the literature
  # see https://www.sas.upenn.edu/~cb36/files/2010_anem.pdf
  np.random.seed(3)

  def real_policy(s):
    num = np.exp(s >0.4)
    prob = num / (1 + num)
    return np.random.binomial(1, p=prob)
    # return s > 0.4

  splits, actions = generate_ultimatum_data(real_policy, n=500)
  tf, tr, comp = model_uncertainty(splits, actions, sd=0.1, temp=5)

  recommended_actions = []
  priors_f = np.linspace(0.0, 0.5, 6)
  evs = [[] for _ in range(len(priors_f))]
  offerer_evs = [[] for _ in range(len(priors_f))]
  wf, wr = comp['weight']
  posteriors_f = [pf*wf*(wf+wr) / (pf*wf*(wf+wr) + (1-pf)*wr*(wf+wr)) for pf in
                  priors_f]
  s_range = np.linspace(0, 1, 20)
  for s in s_range:
    uf_0 = []
    ur_0 = []
    for pf in tf:
      # TODO: make sure this matches current version of model
      uf = pf['r']*s - pf['f']*(s< 0.5-pf['t']) - pf['b']*(s>0.5+pf['t'])
      uf_0.append(uf)
    for pr in tr:
      ur = pr['r']*s
      ur_0.append(ur)
    for i, post_f in enumerate(posteriors_f):
      post_dbn = post_f*np.array(uf_0) + (1-post_f)*np.array(ur_0)
      evs[i].append(np.median(post_dbn))
      offerer_evs[i].append((1-s)*np.mean(post_dbn > 0))

  print(wr, wf)
  pm.traceplot(tr)
  plt.show()
  for prior, ev in zip(priors_f, offerer_evs):
    plt.plot(s_range, ev, label=str(prior))
  plt.legend()
  plt.show()

    



  # maximize_all_likelihoods(splits, actions)








