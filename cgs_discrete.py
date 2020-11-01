import numpy as np
import pdb


# Define payoff matrices
CRASH = -10
G1_1 = np.array([[0, -1], [3, CRASH]])
G1_2 = np.array([[0, 2], [-1, CRASH]])
G2_1 = np.array([[0, -1], [2, CRASH]])
G2_2 = np.array([[0, 3], [-1, CRASH]])
G3_1 = np.array([[0, -1], [2, CRASH]])
G3_2 = np.array([[0, 4], [-1, CRASH]])


def report_ev(G_report_ix, G_observed_ix, G_lst_i, G_lst_j, mechanism_probs, player_ix, prior_parameters=np.ones(3)):
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
  G_observed_i = G_lst_i[G_observed_ix]
  G_observed_j = G_lst_j[G_observed_ix]
  posterior_parameters = prior_parameters
  posterior_parameters[G_observed_ix] += 1
  posterior_probs = posterior_parameters / posterior_parameters.sum()

  report_ev_ = 0.

  if G_observed_i[1, 0] + G_observed_j[1, 0] > G_observed_i[0, 1] + G_observed_j[0, 1]:
    a1_i_default, a2_i_default = 1, 0
  else:
    a1_i_default, a2_i_default = 0, 1

  if player_ix == 1:
    a1_default = a1_i_default
  elif player_ix == 2:
    a2_default = a2_i_default

  for G_observed_j_ix in range(3):
    if player_ix == 1:
      mechanism_prob = mechanism_probs[G_report_ix, G_observed_j_ix]
    elif player_ix == 2:
      mechanism_prob = mechanism_probs[G_observed_j_ix, G_report_ix]
    posterior_prob = posterior_probs[G_observed_j_ix]

    # Average reported games
    G_avg_i = (G_lst_i[G_report_ix] + G_lst_i[G_observed_j_ix]) / 2
    G_avg_j = (G_lst_j[G_report_ix] + G_lst_j[G_observed_j_ix]) / 2

    # Get action profile under averaged game (assuming Chicken payoff structure)
    if G_avg_i[1, 0] + G_avg_j[1, 0] > G_avg_i[0, 1] + G_avg_j[0, 1]:
      a1, a2 = 1, 0
    else:
      a1, a2 = 0, 1

    # Get default player j action
    G_observed_j_i = G_lst_i[G_observed_j_ix]
    G_observed_j_j = G_lst_j[G_observed_j_ix]

    if G_observed_j_i[1, 0] + G_observed_j_j[1, 0] > G_observed_j_i[0, 1] + G_observed_j_j[0, 1]:
      a1_j_default, a2_j_default = 1, 0
    else:
      a1_j_default, a2_j_default = 0, 1

    if player_ix == 1:
      a2_default = a2_j_default
    elif player_ix == 2:
      a1_default = a1_j_default

    # Get interim EV of resulting action profile
    # ToDo: should this be evaluated according to point estimate or posterior? Probably posterior
    # ToDo: can be optimized
    ev_under_action_profile = 0.
    default_ev = 0.
    for k in range(3):
      G_i = G_lst_i[k]
      ev_k = G_i[a1, a2]
      ev_under_action_profile += ev_k*posterior_probs[k]
      default_ev += G_i[a1_default, a2_default]*posterior_probs[k]

    report_ev_ += ev_under_action_profile*posterior_prob*mechanism_prob + \
                  default_ev*posterior_prob*(1-mechanism_prob)
  return report_ev_


if __name__ == "__main__":
  G1_lst = [G1_1, G2_1, G3_1]
  G2_lst = [G1_2, G2_2, G3_2]
  mechanism_probs = np.eye(3)

  # Player 1 report payoffs
  player_ix = 1
  player_1_payoffs = np.zeros((3, 3))
  for report_ix in range(3):
    for observation_ix in range(3):
      report_ev_ = \
        report_ev(report_ix, observation_ix, G1_lst, G2_lst, mechanism_probs, player_ix, prior_parameters=np.ones(3))
      player_1_payoffs[observation_ix, report_ix] = report_ev_

  # Player 2 report payoffs
  player_ix = 2
  player_2_payoffs = np.zeros((3, 3))
  for report_ix in range(3):
    for observation_ix in range(3):
      report_ev_ = \
        report_ev(report_ix, observation_ix, G2_lst, G1_lst, mechanism_probs, player_ix, prior_parameters=np.ones(3))
      player_2_payoffs[observation_ix, report_ix] = report_ev_

  print('player 1:\n {}'.format(player_1_payoffs))
  print('player 2:\n {}'.format(player_2_payoffs))













