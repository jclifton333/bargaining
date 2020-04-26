import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.special import expit
import pymc3 as pm
import matplotlib.pyplot as plt
import pdb
from itertools import product
import seaborn as sns


def generate_ultimatum_data(policy, n=100):
  """
  :param policy: maps split to 0 (reject) or 1 (accept)
  """
  splits = np.random.uniform(size=n)
  # stakes = np.random.lognormal(size=n)
  stakes = np.random.uniform(1, 2, size=n)
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


def big_model(splits, stakes, actions, p_prior='normal', f_prior='uniform',
              gamma_param=1., sd=10., f_mean=1., unif_upper=10, st_t_param=1.):
  model = pm.Model()
  with model:
    # Specify priors
    t = pm.Uniform('t', lower=0, upper=1)
    # st = pm.Beta('st', alpha=1, beta=1)
    temp = pm.Uniform('temp', lower=0, upper=1000)
    if f_prior == 'normal':
      r = pm.Normal('r', mu=0, sd=sd)
      p = pm.Normal('p', mu=0, sd=sd)
      f = pm.Normal('f', mu=f_mean, sd=sd)
      # st_t = pm.Normal('st_t', mu=0, sd=sd)
    else:
      # r = pm.Uniform('r', lower=0, upper=unif_upper)
      # p = pm.Uniform('p', lower=0.0, upper=10)
      # f = pm.Uniform('f', lower=0.0, upper=10)
      p = pm.Lognormal('p', mu=1, sd=1)
      f = pm.Lognormal('f', mu=0.0, sd=1)
      # st_t = pm.Uniform('st_t', lower=0, upper=unif_upper)

    # Specify model
    soft_indicator_num = np.exp((0.5-t/2 - splits)*temp)
    soft_indicator = soft_indicator_num / (soft_indicator_num + 1)
    # soft_indicator = (0.5-t/2>splits)
    # soft_indicator = (0.4>splits)
    # odds_a = np.exp(2*r*splits - f*soft_indicator)
    odds_a = np.exp(splits - f*soft_indicator)
    odds_r = np.exp(p*soft_indicator)
    # odds_r = 1
    prob = odds_a / (odds_r + odds_a)
    a = pm.Binomial('a', 1, prob, observed=actions)

    # Fit and sample
    # fitted = pm.fit(method='fullrank_advi')
    # trace_big = fitted.sample(2000)
    trace_big = pm.sample(10000, chains=1, cores=4, seed=3)
    prior = pm.sample_prior_predictive(10000)
  return trace_big, prior, model

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
  np.random.seed(4)

  def real_ev(sp, st):
    return sp - (sp<0.4)

  def real_policy(sp, st):
    num = np.exp(real_ev(sp, st))
    prob = num / (1 + num)
    return np.random.binomial(1, p=prob)
    # return s > 0.4

  # splits, actions, stakes = generate_ultimatum_data(real_policy, n=10000)
  # tf, tr, comp = model_uncertainty(splits, stakes, actions, sd=0.1, temp=5)
  tb_list = []
  prior_list = []
  sds_list = [1]
  p_dbns_list = ['gamma']
  c_dbns_list = ['gamma']
  gamma_param_list = [1]
  f_mean_list = [0]
  st_t_param_list = [1]
  unif_upper_list = [10]
  dbns_list = unif_upper_list
  sample_size_list = [100]
  for sample_size in sample_size_list:
    splits, actions, stakes = generate_ultimatum_data(real_policy,
                                                      n=sample_size)
    tb, prior, model = big_model(splits, stakes, actions, unif_upper=10)
    tb_list.append(tb)
    prior_list.append(prior)

  recommended_actions = []
  evs = [[] for _ in range(len(tb_list))]
  evs_prior = [[] for _ in range(len(tb_list))]
  offerer_evs = [[] for _ in range(len(tb_list))]
  s_range = np.linspace(0, 1, 20)
  scale = 1 
  for s in s_range:
    ua_list = []
    ur_list = []
    ua_prior_list = []
    ur_prior_list = []
    udiff_list = []
    for i, tb in enumerate(tb_list):
      pri = prior_list[i]
      for j, post in enumerate(tb):
        # Get posterior expectations
        # u_a = post['r']*s - post['f']*(s< 0.5-post['t']/2)
        u_a = s - post['f']*(s< 0.5-post['t']/2)
        # u_r = scale*post['p']*(s < 0.5-post['t']/2)
        u_r = 0
        u_diff = u_a - u_r
        ua_list.append(u_a)
        ur_list.append(u_r)

        # Get prior expectations
        u_a_pri = s - pri['f'][j]*(s< 0.5-pri['t'][j]/2)
        # u_r_pri = scale*pri['p'][j]*(s < 0.5-pri['t'][j]/2)
        u_r_pri = 0
        ua_prior_list.append(u_a_pri)
        ur_prior_list.append(u_r_pri)

      accept_prob = np.mean(np.array(ua_list) - np.array(ur_list) > 0)
      accept_prob_prior = np.mean(np.array(ua_prior_list) - np.array(ur_prior_list) > 0)
      evs[i].append(accept_prob*(1-s)*scale)
      evs_prior[i].append(accept_prob_prior*(1-s)*scale)

  pm.traceplot(tb_list[0])
  # pm.traceplot(tb_list[1])
  plt.show()
  real_off_ev = lambda s: (real_ev(s, scale) > 0)*(1-s)*scale
  colors = ['b', 'g', 'r', 'k', 'y']
  for i in range(len(sample_size_list)):
    p_post = [var['p'] for var in tb_list[i]]
    sns.kdeplot(p_post, color=colors[i], 
                label='n={}'.format(sample_size_list[i]),
                shade=True)
  plt.title('Posterior densities for repeated play parameter')
  plt.xlabel('r')
  plt.ylabel('Density')
  plt.legend(loc=1)
  plt.show()

  for i in range(len(sample_size_list)):
    n = sample_size_list[i]
    ev = evs[i]
    ev_prior = evs_prior[i]
    color = colors[i]
    max_s = s_range[np.argmax(ev)]
    plt.plot(s_range, ev, label='n={}'.format(n), color=color)
    plt.plot([max_s], [np.min(evs)], marker='*', color=color)
    # plt.plot(s_range, ev_prior, label='n={}'.format(n), color=color,
    #          linestyle='dashed')
  # plt.plot(s_range, evs_prior[0], label='prior', color='k')
  plt.plot(s_range, [real_off_ev(s) for s in s_range], label='true')
  plt.xlabel('% to Responder')
  plt.ylabel('Proposer posterior expected utility')
  plt.title('Ultimatum (expected) payoffs after different sample sizes')
  plt.legend(loc=1)
  plt.show()

  # maximize_all_likelihoods(splits, actions)








