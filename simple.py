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


if __name__ == "__main__":
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

