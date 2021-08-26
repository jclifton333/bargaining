import numpy as np
from scipy.special import expit, logit
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
  expected_effect = -0.1
  noise_multipliers = np.linspace(0.01, 1, 100)
  results = np.zeros((100, 100))
  base_prob = 10**-5
  for noise_ix, noise_multiplier in enumerate(noise_multipliers):
    for replicate_ix in range(100):
      base_logit = logit(base_prob) + np.random.normal(loc=expected_effect, scale=0.01, size=10000)
      expected_base_prob = np.mean(expit(base_logit))
      noisy_logits = base_logit + np.random.normal(loc=expected_effect, scale=noise_multiplier, size=10000)
      expected_improvement = np.mean(expit(noisy_logits)) / base_prob
      results[99-noise_ix, replicate_ix] = expected_improvement

  log_noise = np.log10(noise_multipliers / np.abs(expected_effect)).round(1)
  log_noise = log_noise[::-1]
  noise = (noise_multipliers / np.abs(expected_effect)).round(1)
  noise = noise[::-1]
  ratios = results.mean(axis=1)
  plt.plot(noise, ratios)
  plt.hlines(1, noise.min(), noise.max(), linestyles='dashed')
  plt.xlabel(r'Noise-to-signal ratio $\sigma_{\Delta} / \delta$')
  plt.ylabel(r'$\mathbb{E}$[post-intervention prob] / $\mathbb{E}$[pre-intervention prob]')
  plt.show()
