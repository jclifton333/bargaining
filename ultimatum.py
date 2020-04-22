import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.special import expit
import pymc3 as pm
import matplotlib.pyplot as plt
import pdb
from itertools import product


def generate_ultimatum_data(policy, n=100):
  """
  :param policy: maps split to 0 (reject) or 1 (accept)
  """
  splits = np.random.uniform(size=n)
  stakes = np.random.lognormal(size=n)
  actions = [policy(sp, st) for sp, st in zip(splits, stakes)]
  return splits, actions, stakes


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


def big_model(splits, stakes, actions, p_prior='normal', c_prior='normal', sd=1.):
  with pm.Model() as model:
    r = pm.Gamma('r', alpha=1, beta=1)
    if p_prior == 'normal':
      p = pm.Normal('p', mu=0, sd=sd)
    else:
      p = pm.Gamma('p', alpha=1, beta=1)
    t = pm.Beta('t', alpha=2, beta=5)
    st = pm.Beta('st', alpha=1, beta=1)
    if c_prior == 'normal':
      c = pm.Normal('c', mu=0, sd=sd)
    else:
      c = pm.Gamma('c', alpha=1, beta=1)
    f = pm.Normal('f', mu=0, sd=sd)
    odds_a = np.exp(2*r*splits + c*stakes**st - f*(splits< 0.5-t/2))
    odds_r = np.exp(p*(splits < 0.5-t/2))
    p = odds_a / (odds_r + odds_a)
    a = pm.Binomial('a', 1, p, observed=actions)
    fitted = pm.fit(method='advi')
    trace_big = fitted.sample(2000)
    prior = pm.sample_prior_predictive(2000)
  return trace_big, prior

def model_uncertainty(splits, stakes, actions, temp=1., sd=1.):
  with pm.Model() as repeated_model:
    r = pm.Gamma('r', alpha=1, beta=1)
    p = pm.Gamma('p', alpha=1, beta=1)
    t = pm.Beta('t', alpha=2, beta=5)
    st = pm.Beta('st', alpha=1, beta=1)
    c = pm.Gamma('c', alpha=1, beta=1)
    odds_a = np.exp(2*r*splits + c*stakes**st)
    odds_r = np.exp(p*(splits < 0.5-t/2))
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
    r = pm.Gamma('r', alpha=1, beta=1)
    t = pm.Beta('t', alpha=2, beta=5)
    f = pm.Normal('f', mu=0, sd=sd)
    st = pm.Beta('st', alpha=1, beta=1)
    c = pm.Gamma('c', alpha=1, beta=1)
    odds = np.exp(c*stakes**st + splits*r - f*(splits< 0.5-t/2))
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

  def real_ev(sp, st):
    return st**(1/3)*sp - (0.4-sp)*(sp<0.4)

  def real_policy(sp, st):
    num = np.exp(real_ev(sp, st))
    prob = num / (1 + num)
    return np.random.binomial(1, p=prob)
    # return s > 0.4

  splits, actions, stakes = generate_ultimatum_data(real_policy, n=100)
  # tf, tr, comp = model_uncertainty(splits, stakes, actions, sd=0.1, temp=5)
  tb_list = []
  prior_list = []
  sds_list = [0.01, 0.1, 1]
  p_dbns_list = ['gamma', 'normal']
  c_dbns_list = ['gamma', 'normal']
  dbns_list = [(i, j) for i in p_dbns_list for j in c_dbns_list]
  for dbn in dbns_list:
    tb, prior = big_model(splits, stakes, actions, p_prior=dbn[0],
                          c_prior=dbn[1], sd=0.1)
    tb_list.append(tb)
    prior_list.append(prior)

  recommended_actions = []
  evs = [[] for _ in range(len(tb_list))]
  evs_prior = [[] for _ in range(len(tb_list))]
  s_range = np.linspace(0, 1, 20)
  scale = 1
  for s in s_range:
    u_list = []
    u_prior_list = []
    for i, tb in enumerate(tb_list):
      pri = prior_list[i]
      for j, post in enumerate(tb):
        # Get posterior expectations
        u_a = 2*post['r']*s + post['c']*scale**post['st'] - post['f']*(s< 0.5-post['t']/2)
        u_r = post['p']*(s < 0.5-post['t']/2)
        u_diff = u_a - u_r
        u_list.append(u_diff)

        # Get prior expectations
        u_a_pri = 2*pri['r'][j]*s + pri['c'][j]*scale**pri['st'][j] - \
          pri['f'][j]*(s< 0.5-pri['t'][j]/2)
        u_r_pri = pri['p'][j]*(s < 0.5-pri['t'][j]/2)
        u_diff_pri = u_a_pri - u_r_pri
        u_prior_list.append(u_diff_pri)

      evs[i].append(np.mean(u_list))
      evs_prior[i].append(np.mean(u_prior_list))

  # pm.traceplot(tb_list[0])
  # plt.show()
  for prior, ev, ev_prior in zip(list(dbns_list), evs, evs_prior):
    plt.plot(s_range, ev, label='{} post'.format(prior))
    plt.plot(s_range, ev_prior, label='{} prior'.format(prior))
  plt.plot(s_range, [real_ev(s, scale) for s in s_range], label='true')
  plt.legend()
  plt.show()

  # maximize_all_likelihoods(splits, actions)








