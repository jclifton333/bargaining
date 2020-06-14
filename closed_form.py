import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from ultimatum import generate_ultimatum_data
import pdb
import pymc3 as pm


def normal_posterior_mc(x, mu_f_0, mu_r_0, sigma_f=1, sigma_r=1, sigma=1):
  model = pm.Model()
  with model:
    mu_f = pm.Normal('mu_f', mu=mu_f_0, sd=sigma_f)
    mu_r = pm.Normal('mu_r', mu=mu_r_0, sd=sigma_r)
    x_ = pm.Normal('x', mu=mu_f+mu_r, sd=sigma, observed=x)
    trace = pm.sample(10000, chains=1)
  return trace


def normal_posterior(x, mu_f, mu_r, sigma_f=1, sigma_r=1., sigma=1):
  # Get posterior on f + r
  n = len(x)
  xbar = x.mean()
  denom = sigma_f**2 + sigma_r**2 + sigma**2
  mu_post_num = (sigma_r**2 + sigma**2)*mu_f + sigma_f**2*(xbar - mu_r)
  mu_f_post = mu_post_num / denom
  sigma_f_post = (sigma_r**2 + sigma**2)*sigma_f**2 / denom

  # Plot 
  # xs = np.linspace(-5, 5, 20)
  # fs = np.array([norm.pdf(x_, loc=mu_f_post, scale=sd_f_post) for x_ in xs])
  # plt.plot(xs, fs)
  # plt.show()

  return None, mu_f_post, sigma_f_post



if __name__ == "__main__":
  mu_f = 0.05
  mu_r = 0.15
  mu_f_list = [0.05, 0.1]
  mu_r_list = [0.15, 0.1]
  sigma_f = 0.04
  sigma_r = 0.04
  sigma = 0.01
  f_true = 0.15
  r_true = 0.05
  f_plus_r = f_true + r_true
  n = 10000
  s = np.random.uniform(0.0, 0.4, size=n)
  y = np.random.normal(loc=s-f_plus_r, scale=0.01, size=n)
  x = s-y
  mu_f_post_list = []
  sd_f_post_list = []
  for mu_f, mu_r in zip(mu_f_list, mu_r_list):
    _, mu_f_post, sd_f_post = normal_posterior(x, mu_f, mu_r,
                                                     sigma_f=sigma_f,
                                                     sigma_r=sigma_f,
                                                     sigma=sigma)
    mu_f_post_list.append(mu_f_post)
    sd_f_post_list.append(sd_f_post)
  # traces = []
  # for mu_f, mu_r in zip(mu_f_list, mu_r_list):
  #   trace = normal_posterior_mc(x, mu_f, mu_r, sigma_f=sigma_f,
  #                               sigma_r=sigma_r, sigma=sigma)
  #   traces.append(trace)

  def u_mc(s, trace, f=None):
    if f is None:
      posterior_payoffs = []
      for param in trace[1000:]:
        mu_f_draw = param['mu_f']
        x_f = np.random.normal(loc=mu_f_draw, scale=sigma)
        opponent_utility = s - x_f*(s < 0.4)
        utility = (1 - s)*(opponent_utility > 0)
        posterior_payoffs.append(utility)
      payoff_mean = np.mean(posterior_payoffs)
    else:
      opponent_utility = s - f*(s < 0.4)
      payoff_mean = (1-s)*(opponent_utility > 0)
    return payoff_mean

  def u(s, mu_f_post=None, sd_f_post=None, f=None):
    # Posterior predictive payoff for s
    if f is None:
      mu_f = np.random.normal(loc=mu_f_post, scale=sd_f_post, size=1000)
    else:
      mu_f = f*np.ones(1000)
    x_f = np.random.normal(loc=mu_f, scale=sigma)
    opponent_utility = s - x_f*(s < 0.4)
    payoff_dist = (1-s)*(opponent_utility > 0)
    payoff_mean = np.mean(payoff_dist)
    return payoff_mean

  srange = np.linspace(0, 1., 20)

  utrue_range = np.array([u(s, f=f_true) for s in srange])
  plt.plot(srange, utrue_range, label='true payoff')
  colors = ['b', 'g', 'r', 'k', 'y']
  for i in range(len(mu_f_post_list)):
    # trace = traces[i]
    color = colors[i]
    mu_f = mu_f_list[i]
    mu_r = mu_r_list[i]
    mu_f_post = mu_f_post_list[i]
    sd_f_post = sd_f_post_list[i]
    upost_range = np.array([u(s, mu_f_post=mu_f_post, sd_f_post=sd_f_post) for s in srange])
    argmax_s = srange[np.argmax(upost_range)]
    plt.plot(srange, upost_range,
             label='(mu_f, mu_r)={}'.format((mu_f, mu_r)),
             color=color)
    plt.plot([argmax_s], [np.min(utrue_range)], marker='*', markersize=9,
              color=color)
  plt.title('Proposer payoffs under different splits')
  plt.ylabel('(Posterior expected) payoff')
  plt.xlabel('Proportion to Responder')
  plt.legend(loc=1)
  plt.show()





