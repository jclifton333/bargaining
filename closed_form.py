import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from ultimatum import generate_ultimatum_data
import pdb


def normal_posterior(x, mu_f, mu_r, sigma_f=1, sigma_r=1., sigma=1):
  # Get posterior on f + r
  n = len(x)
  xbar = x.mean()
  mu_prior = mu_f + mu_r
  sigma_prior_sq = sigma_f**2 + sigma_r**2
  weight = sigma_prior_sq/(sigma**2/n + sigma_prior_sq)
  mu_post = weight*xbar + (1-weight)*mu_prior
  sigma_sq_post = 1 / ((1/sigma_prior_sq) + (n/sigma**2))

  # Get posterior on f
  mu_f_post = mu_post - mu_r
  sigma_sq_f_post = sigma_sq_post + sigma_r**2
  sd_f_post = np.sqrt(sigma_sq_f_post)

  # Plot 
  # xs = np.linspace(-5, 5, 20)
  # fs = np.array([norm.pdf(x_, loc=mu_f_post, scale=sd_f_post) for x_ in xs])
  # plt.plot(xs, fs)
  # plt.show()

  return mu_post, mu_f_post, sd_f_post


if __name__ == "__main__":
  mu_f = 0.8
  mu_r = 0.12
  f_true = 0.15
  f_plus_r = 0.2
  n = 1000
  s = np.random.uniform(0.0, 0.4, size=n)
  y = np.random.normal(loc=s-f_plus_r, scale=0.01, size=n)
  x = s-y
  mu_post, mu_f_post, sd_f_post = normal_posterior(x, mu_f, mu_r, sigma_f=0.01, sigma_r=0.01, sigma=0.01)

  def u(s, f=None):
    # Posterior predictive payoff for s
    if f is None:
      mu_f = np.random.normal(loc=mu_f_post, scale=sd_f_post, size=100)
      opponent_utility = s - mu_f*(s < 0.4)
      payoff_dist = (1-s)*(opponent_utility > 0)
      payoff_mean = np.mean(payoff_dist)
    else:
      opponent_utility = s - f*(s < 0.4)
      payoff_mean = (1-s)*(opponent_utility > 0)
    return payoff_mean

  srange = np.linspace(0, 1., 20)
  upost_range = np.array([u(s) for s in srange])
  utrue_range = np.array([u(s, f_true) for s in srange])
  plt.plot(srange, upost_range, label='post')
  plt.plot(srange, utrue_range, label='true')
  plt.legend()
  plt.show()





