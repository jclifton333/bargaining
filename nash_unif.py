import numpy as np
import nashpy as nash
import matplotlib.pyplot as plt
import pdb


def get_welfare_optimal_eq(game_res):
  best_welfare = -float('inf')
  best_v1 = -float('inf')
  best_eq = None
  for eq in game_res.support_enumeration():
    v1, v2 = game_res[eq]
    if v1 + v2 > best_welfare: 
      best_welfare = v1 + v2
      best_v1 = v1
      best_eq = eq
  return best_eq[0], best_eq[1], best_v1


def expected_payoffs(p1, p2, a1, a2):
  probs = np.outer(a1, a2)
  v1 = np.sum(np.multiply(p1, probs))
  v2 = np.sum(np.multiply(p2, probs))
  return v1, v2


def split_and_compare_perturbed_games(p_list, n_split=10, n_rep=10, player_ix=0):
  n_obs = len(p_list)
  mean_diffs = []
  for _ in range(n_split):
    split_indices = np.random.choice(len(p_list), len(p_list), replace=True)
    p_split = [p_list[ix] for ix in split_indices]
    p1_split, p2_split, p1_split_mean, p2_split_mean = get_payoffs_from_list(p_split)
    # Assuming only first row observed with noise
    p1_split = np.array(p1_split)
    p2_split = np.array(p2_split)
    scale = np.mean((np.mean(np.std(p1_split[:, 0, :], axis=0)), np.mean(np.std(p2_split[:, 0, :], axis=0))))
    # scale = 1. # ToDo: passing true scale for debugging
    diffs = []
    for rep in range(n_rep):
      diff = compare_perturbed_games(p1_split_mean, p2_split_mean, scale=scale / np.sqrt(n_obs), player_ix=player_ix)
      diffs.append(diff)
    mean_diffs.append(np.mean(diffs))
  mean_diff = np.mean(mean_diffs)
  return mean_diff


def get_payoffs_from_list(p_list):
  p1 = [p1 for p1, p2 in p_list]
  p2 = [p2 for p1, p2 in p_list]
  p1_mean = np.mean(p1, axis=0)
  p2_mean = np.mean(p2, axis=0)
  return p1, p2, p1_mean, p2_mean


def adaptive_strategy(p_list_1, p_list_2, n_split=40, n_rep=40):
  mean_diff_1 = split_and_compare_perturbed_games(p_list_1, n_split=n_split, n_rep=n_rep)
  mean_diff_2 = split_and_compare_perturbed_games(p_list_2, n_split=n_split, n_rep=n_rep, player_ix=1)
  averaging_is_better_1 = mean_diff_1 > 0
  averaging_is_better_2 = mean_diff_2 > 0
  _, _, p1_1, p2_1 = get_payoffs_from_list(p_list_1)
  _, _, p1_2, p2_2 = get_payoffs_from_list(p_list_2)
  p1_avg = (p1_1 + p2_1) / 2
  p2_avg = (p2_1 + p2_2) / 2

  # if averaging_is_better_1:
  #   a1 = get_welfare_optimal_eq(nash.Game(p1_avg, p2_avg))[0]
  # else:
  #   a1 = get_welfare_optimal_eq(nash.Game(p1_1, p2_1))[0]
  # if averaging_is_better_2:
  #   a2 = get_welfare_optimal_eq(nash.Game(p1_avg, p2_avg))[1]
  # else:
  #   a2 = get_welfare_optimal_eq(nash.Game(p1_2, p2_2))[1]

  if averaging_is_better_1 and averaging_is_better_2:
    a1, a2, _ = get_welfare_optimal_eq(nash.Game(p1_avg, p2_avg))
  else:
    a1 = get_welfare_optimal_eq(nash.Game(p1_1, p2_1))[0]
    a2 = get_welfare_optimal_eq(nash.Game(p1_2, p2_2))[1]
  return a1, a2, mean_diff_1, mean_diff_2


def compare_perturbed_games(p1, p2, loc=0., scale=0.3, bias_direction=1, player_ix=0):
  # Assuming only first row observed with noise
  scale_mat = np.array([[scale, scale], [0, 0]])
  p1_1 = p1 + np.random.normal(size=p1.shape, loc=loc, scale=scale_mat)
  p2_1 = p2 + np.random.normal(size=p2.shape, loc=loc, scale=scale_mat)
  p1_2 = p1 + np.random.normal(size=p1.shape, loc=loc*bias_direction, scale=scale_mat)
  p2_2 = p2 + np.random.normal(size=p2.shape, loc=loc*bias_direction, scale=scale_mat)
  # p1_1 = p1 + np.random.standard_t(size=p1.shape, df=2.1)
  # p2_1 = p2 + np.random.standard_t(size=p2.shape, df=2.1)
  # p1_2 = p1 + np.random.standard_t(size=p1.shape, df=2.1)
  # p2_2 = p2 + np.random.standard_t(size=p2.shape, df=2.1)

  p1_avg = (p1_1 + p1_2) / 2
  p2_avg = (p2_1 + p2_2) / 2

  a1 = get_welfare_optimal_eq(nash.Game(p1_1, p2_1))[0]
  a2 = get_welfare_optimal_eq(nash.Game(p1_2, p2_2))[1]
  a1_avg, a2_avg, _ = get_welfare_optimal_eq(nash.Game(p1_avg, p2_avg))

  # ToDo: Think more about justification for deciding based on this quantity
  v1_ind_ = expected_payoffs(p1, p2, a1, a2)[player_ix]
  v1_avg_ = expected_payoffs(p1, p2, a1_avg, a2_avg)[player_ix]
  return v1_avg_ - v1_ind_


def get_random_eq(game_res):
  game_values = [game_res[eq][0] for eq in game_res.support_enumeration()]
  return np.random.choice(game_values)


if __name__ == "__main__":
  # ToDo: version with both private and public signal
  # ToDo: implement threat game (although this seems structurally similar already?)
  # ToDo: implement heuristics for exploiting cooperators, and heuristics for punishing exploitation (penalizing by fit)
  """
  The method can be seen as creating a matrix with values for the row player of 
  
  u_C u_D
  u_D u_D  
  
  where C and D correspond to averaging and not-averaging*.
  
  and testing whether u_C > u_D; thus giving a weakly dominant strategy (with respect to these expectations). 
  Of course, this is still a hypergame and the estimates might be wrong, but we should try to prove
  that playing this hypergame is more robust than the original hypergame.
  
  *Also be aware that we have assumed the player
  commits to playing the equilibrium of the averaged game, if (C, C) is played. Assuming the players truthfully
  report their observations, this can be seen as rational in the sense that the new game is a better estimate than
  the one based on one's private obs, so trying to best-respond based on one's private obs is a bad idea.
  """

  GEN_SCALE = 1
  NOISE_SCALE = np.array([[20., 20.], [0, 0]])
  NOISE_BIAS = 0.0
  N_OBS = 20
  BIAS_FLIP_DIRECTION = 1  # Player 1 bias=NOISE_BIAS; Player 2 bias=BIAS_FLIP_DIRECTION*NOISE_BIAS

  num_games = 1
  num_reps = 50

  for i in range(num_games):
    adapts_i = []
    avgs_i = []
    inds_i = []
    errors_i = []
    p1 = np.array([[-10, 0], [-3, -1]])
    p2 = np.array([[-10, -3], [0, -1]])
    for j in range(num_reps):
      # Obs for player 1
      p1_1_list = [np.random.normal(loc=p1, scale=NOISE_SCALE) for obs in range(N_OBS)]
      p2_1_list = [np.random.normal(loc=p2, scale=NOISE_SCALE) for obs in range(N_OBS)]
      p_list_1 = list(zip(p1_1_list, p2_1_list))
      # Obs for player 2
      p1_2_list = [np.random.normal(loc=p1, scale=NOISE_SCALE) for obs in range(N_OBS)]
      p2_2_list = [np.random.normal(loc=p2, scale=NOISE_SCALE) for obs in range(N_OBS)]
      p_list_2 = list(zip(p1_2_list, p2_2_list))

      # ToDo: redundant, can be optimized
      p1_1_mean = np.mean(p1_1_list, axis=0)
      p2_1_mean = np.mean(p2_1_list, axis=0)
      p1_2_mean = np.mean(p1_2_list, axis=0)
      p2_2_mean = np.mean(p2_2_list, axis=0)
      p1_avg = (p1_1_mean + p1_2_mean) / 2
      p2_avg = (p2_1_mean + p2_2_mean) / 2

      a1_ind = get_welfare_optimal_eq(nash.Game(p1_1_mean, p2_1_mean))[0]
      a2_ind = get_welfare_optimal_eq(nash.Game(p1_2_mean, p2_2_mean))[1]
      a1_avg, a2_avg, _ = get_welfare_optimal_eq(nash.Game(p1_avg, p2_avg))

      # Get true difference
      v1_ind, v2_ind = expected_payoffs(p1, p2, a1_ind, a2_ind)
      v1_avg, v2_avg = expected_payoffs(p1, p2, a1_avg, a2_avg)
      true_diff = v1_avg - v1_ind

      # Get estimated difference
      a1_adapt, a2_adapt, est_diff_1, est_diff_2 = adaptive_strategy(p_list_1, p_list_2, n_split=100, n_rep=10)
      v1_adapt, v2_adapt = expected_payoffs(p1, p2, a1_adapt, a2_adapt)

      print('adapt: {} avg: {} ind: {}'.format(v1_adapt, v1_avg, v1_ind))
      error = est_diff_1 - true_diff
      errors_i.append(error)
      adapts_i.append(v1_adapt)
      avgs_i.append(v1_avg)
      inds_i.append(v1_ind)
    mean_adapts = np.mean(adapts_i)
    se_adapts = np.std(adapts_i) / np.sqrt(len(adapts_i))
    mean_avgs = np.mean(avgs_i)
    se_avgs = np.std(avgs_i) / np.sqrt(len(avgs_i))
    mean_inds = np.mean(inds_i)
    se_inds = np.std(inds_i) / np.sqrt(len(inds_i))
    print('MEAN adapt: {}\navg: {}\nind: {}'.format((mean_adapts, se_adapts), (mean_avgs, se_avgs),
                                                    (mean_inds, se_inds)))
