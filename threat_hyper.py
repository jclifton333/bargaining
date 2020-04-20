import numpy as np
import nashpy as nash
import matplotlib.pyplot as plt
import pdb

"""
Player and action coding: 

Player 0 = Target (ta) 
Player 1 = Threatener (Th)

Carry out = 1
Give in = 1 
"""

N = 30
N_GAME_REPS = 500
MEAN_CREDENCE = 0.07
V_TA_GIVE = -1
V_TA_CARRY = -20
V_TH_GIVE = 1
V_TH_CARRY = -5
PAYOFFS_TA = np.array([[0., V_TA_CARRY], [V_TA_GIVE, V_TA_GIVE]])
PAYOFFS_TH = np.array([[0., V_TH_CARRY], [V_TH_GIVE, V_TH_GIVE]])


def action_given_credence(p):
  return int(V_TA_GIVE > p*V_TA_CARRY)


def payoffs_from_actions(a_ta, a_th):
  u_ta = PAYOFFS_TA[a_ta, a_th]
  u_th = PAYOFFS_TH[a_ta, a_th]
  return u_ta, u_th


def payoffs_given_credences(p_ta, p_th):
  a_ta = action_given_credence(p_ta)
  a_th = action_given_credence(p_th)
  return payoffs_from_actions(a_ta, a_th)


def compare_expected_payoffs(obs_i, player_ix, n_rep=100, worst_case=False):
  p_i = np.mean(obs_i)
  payoffs_ind = []
  payoffs_avg = []
  for rep in range(n_rep):
    # obs_rep = np.random.choice(obs_i, len(obs_i), replace=True)  # Posterior draw for true credence
    weights = np.random.exponential(size=N)
    mean_rep = np.dot(weights, np.array(obs_i)) / np.sum(weights)
    obs_mi_rep = np.random.binomial(1, mean_rep, N)  # Posterior predictive draw for player -i's obs
    p_mi_rep = np.mean(obs_mi_rep)
    p_ta = p_i*(1-player_ix) + p_mi_rep*player_ix
    p_th = p_i*player_ix + p_mi_rep*(1-player_ix)
    p_avg = (p_ta + p_th) / 2
    payoffs_ind_rep = payoffs_given_credences(p_ta, p_th)[player_ix]
    payoffs_avg_rep = payoffs_given_credences(p_avg, p_avg)[player_ix]
    payoffs_ind.append(payoffs_ind_rep)
    payoffs_avg.append(payoffs_avg_rep)
  if worst_case:
    payoffs_avg_mean = np.min(payoffs_avg)
    payoffs_ind_mean = np.min(payoffs_ind)
  else:
    payoffs_avg_mean = np.mean(payoffs_avg)
    payoffs_ind_mean = np.mean(payoffs_ind)
  diff = payoffs_avg_mean - payoffs_ind_mean
  return diff


def all_strategies(obs_ta, obs_th, n_rep=100, worst_case=False):
  avg_is_better_ta = int(compare_expected_payoffs(obs_ta, 0, n_rep=n_rep, worst_case=worst_case) >= 0)
  avg_is_better_th = int(compare_expected_payoffs(obs_th, 1, n_rep=n_rep, worst_case=worst_case) >= 0)
  p_ta = np.mean(obs_ta)
  p_th = np.mean(obs_th)
  p_avg = (p_ta + p_th) / 2

  # Independent
  a_ta_ind = action_given_credence(p_ta)
  a_th_ind = action_given_credence(p_th)

  # Averaged
  a_ta_avg = action_given_credence(p_avg)
  a_th_avg = action_given_credence(p_avg)

  # Adaptive
  if avg_is_better_ta and avg_is_better_th:
    a_ta_ad = a_ta_avg
    a_th_ad = a_th_avg
  else:
    a_ta_ad = a_ta_ind
    a_th_ad = a_th_ind

  return a_ta_ind, a_th_ind, a_ta_avg, a_th_avg, a_ta_ad, a_th_ad, avg_is_better_th*avg_is_better_ta


if __name__ == "__main__":
  """For now, assuming Commit type"""
  u_ta_ind = []
  u_ta_avg = []
  u_ta_ad = []
  u_th_ind = []
  u_th_avg = []
  u_th_ad = []
  p_avg = []
  avg_is_better = []
  for rep in range(N_GAME_REPS):
    p_true = np.random.beta(a=7, b=100)
    obs_ta = np.random.binomial(1, p=p_true, size=N)
    obs_th = np.random.binomial(1, p=p_true, size=N)

    a_ta_ind, a_th_ind, a_ta_avg, a_th_avg, a_ta_ad, a_th_ad, avg_is_better_rep = \
      all_strategies(obs_ta, obs_th, n_rep=200, worst_case=False)
    u_ta_ind_rep, u_th_ind_rep = payoffs_from_actions(a_ta_ind, a_th_ind)
    u_ta_avg_rep, u_th_avg_rep = payoffs_from_actions(a_ta_avg, a_th_avg)
    u_ta_ad_rep, u_th_ad_rep = payoffs_from_actions(a_ta_ad, a_th_ad)

    p_avg_rep = (np.mean(obs_ta) + np.mean(obs_th)) / 2
    avg_is_better.append(avg_is_better_rep)
    p_avg.append(p_avg_rep - p_true)
    u_ta_ind.append(u_ta_ind_rep)
    u_ta_avg.append(u_ta_avg_rep)
    u_ta_ad.append(u_ta_ad_rep)
    u_th_ind.append(u_th_ind_rep)
    u_th_avg.append(u_th_avg_rep)
    u_th_ad.append(u_th_ad_rep)
  np.mean(np.array(p_avg)[np.where(avg_is_better == 1)])
  print('ta ind: {} ta avg: {} ta ad: {}\nth ind: {} th avg: {} th ad: {}'.format(np.mean(u_ta_ind), np.mean(u_ta_avg),
                                                                                  np.mean(u_ta_ad), np.mean(u_th_ind),
                                                                                  np.mean(u_th_avg), np.mean(u_th_ad)))


