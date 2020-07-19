import numpy as np
import matplotlib.pyplot as plt
from ultimatum import generate_ultimatum_data
import pdb


def u_responder(s, f):
  u = s - f*(s < 0.4)
  return u


def u_proposer(s, f):
  u_R = u_responder(s, f)
  u_P = (1 - s)*(u_R > 0)
  return u_P


def simple_proposer_posterior_expectation(s, prior, models):
  """
  Get posterior expectations, assuming unidentifiable
  models.

  :param models: tuples (F, I)
  """
  u_P_post = 0.
  for p, m in zip(prior, models):
    u_P_m = u_proposer(s, m[0])
    u_P_post += u_P_m*p
  return u_P_post


def random_priors(alpha=1, mc_rep=100):
  # Proposer is player 1

  models = [(0.05, 0.15), (0.15, 0.05)]
  F, I = 0.15, 0.05
  true_model = [(F, I)]
  srange = np.linspace(0, 1, 20)
  regret = 0.
  for _ in range(mc_rep):
    prior_1 = np.random.dirichlet(alpha*np.ones(2))
    prior_2 = np.random.dirichlet(alpha*np.ones(2))
    prior_comb = (prior_1 + prior_2) / 2
    u_post_1 = np.array([simple_proposer_posterior_expectation(s, prior_1, models) for s in
                         srange])
    u_post_comb = np.array([simple_proposer_posterior_expectation(s, prior_comb, models) for s in
                            srange])
    argmax_1 = srange[np.argmax(u_post_1)]
    argmax_comb = srange[np.argmax(u_post_comb)]

    accept_1 = (u_responder(argmax_1, F) > 0)
    accept_comb = (u_responder(argmax_comb, F) > 0)

    regret_rep = accept_comb*(1 - argmax_1) - accept_1*(1 - argmax_comb)
    regret += regret_rep / mc_rep
  print(regret)


def plot_payoff_curves():
  """
  Produces plot in identifiability doc (as of July 18 2020).
  """
  models = [(0.05, 0.15), (0.15, 0.05)]
  # models = [(0.00, 0.2), (0.2, 0.00)]
  prior_1 = (0.9, 0.1)
  prior_2 = (0.5, 0.5)
  priors = [prior_1, prior_2]
  true_model = [(0.15, 0.05)]
  # true_model = [(0.2, 0.00)]

  srange = np.linspace(0, 1, 20)
  colors = ['b', 'g', 'r', 'k', 'y']
  for i, prior in enumerate(priors):
    u_post = np.array([simple_proposer_posterior_expectation(s, prior, models) for s in
                       srange])
    plt.plot(srange, u_post, label='prior={}'.format(prior), color=colors[i])
    argmax_s = srange[np.argmax(u_post)]
    plt.plot([argmax_s], [np.min(u_post)], marker='*', markersize=9,
             color=colors[i])
  u_true = np.array([simple_proposer_posterior_expectation(s, [1.], true_model) for s in
                     srange])
  plt.title('Proposer payoffs under different splits')
  plt.ylabel('(Posterior expected) payoff')
  plt.xlabel('Proportion to Responder')
  plt.plot(srange, u_true, label='true model')
  plt.legend(loc=1)
  plt.show()


if __name__ == "__main__":
  random_priors(alpha=0.1)

