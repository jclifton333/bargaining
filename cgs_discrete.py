import numpy as np


# Define payoff matrices
G1_1 = np.array([[0, -1], [3, -5]])
G1_2 = np.array([[0, 2], [-1, -5]])
G2_1 = np.array([[0, -1], [2, -5]])
G2_2 = np.array([[0, 3], [-1, -5]])
G3_1 = np.array([[0, -1], [2, -5]])
G3_2 = np.array([[0, 4], [-1, -5]])


def report_ev(G_report_ix, G_observed_ix, G_lst_i, G_lst_j, mechanism_probs, prior_parameters=np.ones(3)):
  """
  Get the expected value of reporting G_report_ix, assuming the other player reported truthfully.

  :param G_report_ix: index of reported matrix
  :param G_observed_ix: index of player's observed matrix
  :param G_lst_i: list of payoff matrices for this player
  :param G_lst_j: list of payoff matrices for other player
  :param mechanism_probs: 3x3 stochastic matrix of 1 - (probability_of_default) given players' reports
  :param prior_parameters: parameters of Dirichlet prior over games
  """
  G_report = G_lst_i[G_report_ix]
  G_observed = G_lst_i[G_observed_ix]
  posterior_parameters = prior_parameters
  posterior_parameters[G_observed_ix] += 1
  posterior_probs = posterior_parameters / posterior_parameters.sum()

  report_ev_ = 0.
  for G_observed_j_ix, G_observed_j in enumerate(G_lst_j):
    mechanism_prob = mechanism_probs[G_observed_ix, G_observed_j_ix]
    posterior_prob = posterior_probs[G_observed_j_ix]

    # Average reported games
    G_avg_i = (G_lst_i[G_report_ix] + G_lst_i[G_observed_j_ix]) / 2
    G_avg_j = (G_lst_j[G_report_ix] + G_lst_j[G_observed_j_ix]) / 2

    # Get action profile under averaged game (assuming Chicken payoff structure)
    if G_avg_i[1, 0] + G_avg_j[1, 0] > G_avg_i[0, 1] + G_avg_j[0, 1]:
      a1, a2 = 1, 0
    else:
      a1, a2 = 0, 1

    # Get interim EV of resulting action profile
    # ToDo: should this be evaluated according to point estimate or posterior? Probably posterior
    # ToDo: can be optimized
    ev_under_action_profile = 0.
    for k in range(3):
      G_i = G_lst_i[k]
      ev_k = G_i[a1, a2]
      ev_under_action_profile += ev_k*posterior_probs[k]

    # ToDo: handle what happens if default
    report_ev_ += ev_under_action_profile*posterior_probs*posterior_prob

  return report_ev_


















