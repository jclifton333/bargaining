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


def meta_ultimatum_game(a1, a2, prior_1, prior_2, eps_1=0.05, eps_2=0.2, tau=0.3):
  # ToDo: assuming truthfully reported priors

  if a1 and a2 and np.abs(prior_1[0] - prior_2[0]) < tau:
    combine = True
  else:
    combine = False

  prior_1_distort = tau - eps_1
  prior_1[0] = np.min((prior_1[0] + prior_1_distort, 1.))
  prior_1[1] = 1 - prior_1[0]

  prior_2_distort = tau - eps_2
  prior_2[0] = np.max((prior_2[0] - prior_2_distort, 0.))
  prior_2[1] = 1 - prior_2[0]

  u_r, u_p = ultimatum_game(prior_1, prior_2, combine)
  return u_r, u_p


def ultimatum_game(prior_1, prior_2, combine):
  # Proposer is player 1

  models = [(0.05, 0.15), (0.15, 0.05)]
  F, I = 0.15, 0.05
  srange = np.linspace(0, 1, 20)

  if combine:
    prior_comb = (prior_1 + prior_2) / 2
    u_post_comb = np.array([simple_proposer_posterior_expectation(s, prior_comb, models) for s in
                            srange])
    offer_ = srange[np.argmax(u_post_comb)]
  else:
    u_post_1 = np.array([simple_proposer_posterior_expectation(s, prior_1, models) for s in
                         srange])
    offer_ = srange[np.argmax(u_post_1)]

  u_responder_ = u_responder(offer_, F)
  accept = (u_responder_ > 0)
  u_proposer_ = (1 - offer_)*accept

  return u_proposer_, u_responder_


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
  alpha_lst = np.logspace(-1, 1, 10)
  regrets_1 = []
  regrets_2 = []
  for alpha in alpha_lst:
    regret_1, regret_2 = random_priors(alpha=alpha, mc_rep=1000)
    regrets_1.append(regret_1)
    regrets_2.append(regret_2)
  plt.plot(alpha_lst, regrets_1, label='Proposer')
  plt.plot(alpha_lst, regrets_2, label='Responder')
  plt.xlabel('Dirichlet concentration parameter alpha')
  plt.ylabel('coop payoff - independent payoff')
  plt.legend()
  plt.show()

