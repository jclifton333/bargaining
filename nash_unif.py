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


def compare_perturbed_games(p1, p2, loc=0., scale=0.3, bias_direction=1):
  p1_1 = p1 + np.random.normal(size=p1.shape, loc=loc, scale=scale)
  p2_1 = p2 + np.random.normal(size=p2.shape, loc=loc, scale=scale)
  p1_2 = p1 + np.random.normal(size=p1.shape, loc=loc*bias_direction, scale=scale)
  p2_2 = p2 + np.random.normal(size=p2.shape, loc=loc*bias_direction, scale=scale)
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
  GEN_SCALE = 1.
  NOISE_SCALE = 0.9
  NOISE_BIAS = 0.1
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
