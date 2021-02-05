import numpy as np
import pdb


# Define payoff matrices
CRASH = -10
G1_1 = np.array([[-3, -1], [3, CRASH]])
G1_2 = np.array([[-3, 2], [-1, CRASH]])
G2_1 = np.array([[-3, -1], [2, CRASH]])
G2_2 = np.array([[-3, 3.1], [-1, CRASH]])
G3_1 = np.array([[-3, -1], [4, CRASH]]) / 3
G3_2 = np.array([[-3, 2], [-1, CRASH]]) / 3


def report_ev(G_report_ix, G_observed_ix, G_lst_i, G_lst_j, mechanism_probs, player_ix, prior_parameters=np.ones(3),
              temperature=None):
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
  default_ev_ = 0.

  if G_observed_i[1, 0] + G_observed_j[1, 0] > G_observed_i[0, 1] + G_observed_j[0, 1]:
    a1_i_default, a2_i_default = 1, 0
  else:
    a1_i_default, a2_i_default = 0, 1

  if player_ix == 1:
    a1_default = a1_i_default
  elif player_ix == 2:
    a2_default = a2_i_default

  for G_observed_j_ix in range(2):
    # if player_ix == 1:
    #   mechanism_prob = mechanism_probs[G_report_ix, G_observed_j_ix]
    # elif player_ix == 2:
    #   mechanism_prob = mechanism_probs[G_observed_j_ix, G_report_ix]
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
    ev_under_action_profile_j = 0.
    default_ev = 0.
    default_ev_j = 0.
    posterior_parameters_j = prior_parameters
    posterior_parameters_j[G_observed_j_ix] += 1
    posterior_probs_j = posterior_parameters_j / posterior_parameters_j.sum()
    for k in range(2):
      G_i = G_lst_i[k]
      G_j = G_lst_j[k]
      ev_k = G_i[a1, a2]
      ev_under_action_profile += ev_k*posterior_probs[k]
      ev_under_action_profile_j += G_j[a1, a2]*posterior_probs_j[k]
      default_ev += G_i[a1_default, a2_default]*posterior_probs[k]
      default_ev_j += G_j[a2_default, a2_default]*posterior_probs_j[k]
    # punish_ev = CRASH
    punish_ev = default_ev

    if temperature is None:
      transformed_posterior_probs_j = posterior_probs_j
      transformed_posterior_probs = posterior_probs
    else:
      exp_posterior_probs_j = np.exp(posterior_probs_j * temperature)
      transformed_posterior_probs_j = exp_posterior_probs_j / exp_posterior_probs_j.sum()
      exp_posterior_probs = np.exp(posterior_probs * temperature)
      transformed_posterior_probs = exp_posterior_probs / exp_posterior_probs.sum()

    # transformed_posterior_probs_j = posterior_probs_j
    # mechanism_prob = transformed_posterior_probs_j[G_report_ix]
    # mechanism_prob = 1
    mechanism_prob = transformed_posterior_probs_j[G_report_ix] * transformed_posterior_probs[G_observed_j_ix]
    value_j_ix = ev_under_action_profile*posterior_prob*mechanism_prob + \
                  punish_ev*posterior_prob*(1 - mechanism_prob)
    # print(f'obs i: {observation_ix} report i: {G_report_ix} report j: {G_observed_j_ix} value: {ev_under_action_profile} default_ev: {default_ev} prob: {posterior_prob} mechanism prob: {mechanism_prob}')
    report_ev_ += value_j_ix
    default_ev_ += default_ev*posterior_prob
  return report_ev_, default_ev_


def evaluate_at_temperature(temperature):
  G1_lst = [G1_1, G2_1]
  G2_lst = [G1_2, G2_2]
  mechanism_probs = np.eye(2)

  # Player 1 report payoffs
  player_ix = 1
  player_1_payoffs = np.zeros((2, 2))
  player_1_default_payoffs = np.zeros((2, 2))
  for observation_ix in range(2):
    for report_ix in range(2):
      report_ev_, default_ev_ = \
        report_ev(report_ix, observation_ix, G1_lst, G2_lst, mechanism_probs, player_ix,
                  prior_parameters=np.array([1, 1]), temperature=temperature)
      player_1_payoffs[observation_ix, report_ix] = report_ev_
      player_1_default_payoffs[observation_ix, report_ix] = default_ev_

  # Player 2 report payoffs
  player_ix = 2
  player_2_payoffs = np.zeros((2, 2))
  player_2_default_payoffs = np.zeros((2, 2))
  for observation_ix in range(2):
    for report_ix in range(2):
      report_ev_, default_ev_ = \
        report_ev(report_ix, observation_ix, G2_lst, G1_lst, mechanism_probs, player_ix,
                  prior_parameters=np.array([1, 1]), temperature=temperature)
      player_2_payoffs[observation_ix, report_ix] = report_ev_
      player_2_default_payoffs[observation_ix, report_ix] = default_ev_

  print('player 1:\n {}'.format(player_1_payoffs))
  print('player 1 default:\n {}'.format(player_1_default_payoffs))
  print('player 2:\n {}'.format(player_2_payoffs))
  print('player 2 default:\n {}'.format(player_2_default_payoffs))


if __name__ == "__main__":
  temperatures = [-10]
  for temperature in temperatures:
    print('temperature: {}'.format(temperature))
    evaluate_at_temperature(temperature)
