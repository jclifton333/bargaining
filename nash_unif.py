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


def split_and_compare_perturbed_games(p_list, n_split=10, n_rep=10):
  mean_diffs = []
  for _ in range(n_split):
    p_split = np.random.choice(p_list, len(p_list), replace=True)
    p1_split, p2_split, p1_split_mean, p2_split_mean = get_payoffs_from_list(p_split)
    scale = np.mean((np.mean(np.std(p1_split, axis=0)), np.mean(np.std(p2_split, axis=0))))
    diffs = []
    for rep in range(n_rep):
      diff = compare_perturbed_games(p1_split_mean, p2_split_mean, scale=scale)
      diffs.append(diff)
    mean_diffs.append(diffs)
  return mean_diff > 0


def get_payoffs_from_list(p_list):
  p1 = [p1 for p1, p2 in p_list]
  p2 = [p2 for p1, p2 in p_list]
  p1_mean = np.mean(p1, axis=0)
  p2_mean = np.mean(p2, axis=0)
  return p1, p2, p1_mean, p2_mean


def adaptive_strategy(p_list_1, p_list_2, n_split=10, n_rep=10):
  averaging_is_better_1 = split_and_compare_perturbed_games(p_list_1, n_split=n_split, n_rep=n_rep)
  averaging_is_better_2 = split_and_compare_perturbed_games(p_list_2, n_split=n_split, n_rep=n_rep)
  _, _, p1_1, p2_1 = get_payoffs_from_list(p_list_1)
  _, _, p2_1, p2_2 = get_payoffs_from_list(p_list_2)
  p1_avg = (p1_1 + p2_1) / 2
  p2_avg = (p2_1 + p2_2) / 2

  if averaging_is_better_1:
    a1 = get_welfare_optimal_eq(nash.Game(p1_avg, p2_avg))[0]
  else:
    a1 = get_welfare_optimal_eq(nash.Game(p1_1, p2_1))[0]
  if averaging_is_better_2:
    a2 = get_welfare_optimal_eq(nash.Game(p1_avg, p2_avg))[1]
  else:
    a2 = get_welfare_optimal_eq(nash.Game(p1_2, p2_2))[1]

  return a1, a2


def compare_perturbed_games(p1, p2, loc=0., scale=0.3, bias_direction=1):
  p1_1 = p1 + np.random.normal(size=p1.shape, loc=loc, scale=scale)
  p2_1 = p2 + np.random.normal(size=p2.shape, loc=loc, scale=scale)
  p1_2 = p1 + np.random.normal(size=p1.shape, loc=loc*bias_direction, scale=scale)
  p2_2 = p2 + np.random.normal(size=p2.shape, loc=loc*bias_direction, scale=scale)
  # p1_1 = p1 + np.random.standard_t(size=p1.shape, df=2.1)
  # p2_1 = p2 + np.random.standard_t(size=p2.shape, df=2.1)
  # p1_2 = p1 + np.random.standard_t(size=p1.shape, df=2.1)
  # p2_2 = p2 + np.random.standard_t(size=p2.shape, df=2.1)

  p1_avg = (p1_1 + p1_2) / 2
  p2_avg = (p2_1 + p2_2) / 2

  a1 = get_welfare_optimal_eq(nash.Game(p1_1, p2_1))[0]
  a2 = get_welfare_optimal_eq(nash.Game(p1_2, p2_2))[1]
  a1_avg, a2_avg, _ = get_welfare_optimal_eq(nash.Game(p1_avg, p2_avg))

  payoffs_hyper = expected_payoffs(p1, p2, a1, a2)
  payoffs_avg = expected_payoffs(p1, p2, a1_avg, a2_avg)

  return payoffs_avg[0] - payoffs_hyper[0]


def get_random_eq(game_res):
  game_values = [game_res[eq][0] for eq in game_res.support_enumeration()]
  return np.random.choice(game_values)


if __name__ == "__main__":
  # ToDo: implement adaptive (bootstrap) procedure
  # ToDo: version with both private and public signal
  GEN_SCALE = 1.
  NOISE_SCALE = 10
  NOISE_BIAS = 0.0
  BIAS_FLIP_DIRECTION = 1  # Player 1 bias=NOISE_BIAS; Player 2 bias=BIAS_FLIP_DIRECTION*NOISE_BIAS

  num_games = 20
  num_reps = 200

  for i in range(num_games):
    diffs_i = []
    p1 = np.random.normal(size=(2, 2), scale=GEN_SCALE)
    p2 = np.random.normal(size=(2, 2), scale=GEN_SCALE)
    for j in range(num_reps):
      diff = compare_perturbed_games(p1, p2, loc=NOISE_BIAS, scale=NOISE_SCALE, bias_direction=BIAS_FLIP_DIRECTION)
      diffs_i.append(diff)
    print(np.mean(diffs_i))
  # plt.hist(diffs)
  # plt.show()
