import numpy as np
from scipy.special import expit, logit
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
  expected_effect = -1
  noise_multipliers = np.logspace(-1, 1, 10)
  base_probabilities = np.logspace(-5, -2, 10)
  results = np.zeros((10, 10))
  for noise_ix, noise_multiplier in enumerate(noise_multipliers):
    for prob_ix, base_prob in enumerate(base_probabilities):
      base_logit = logit(base_prob)
      noisy_logits = base_logit + np.random.normal(loc=expected_effect, scale=noise_multiplier, size=10000)
      expected_improvement = base_prob - np.mean(expit(noisy_logits))
      results[9-noise_ix, prob_ix] = expected_improvement
  sns.heatmap(results)
  # plt.show()

