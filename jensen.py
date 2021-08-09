import numpy as np
from scipy.special import expit, logit
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
  expected_effect = -1
  noise_multipliers = np.logspace(-1, 1, 100)
  base_probabilities = np.logspace(-5, -2, 100)
  results = np.zeros((100, 100))
  for noise_ix, noise_multiplier in enumerate(noise_multipliers):
    for prob_ix, base_prob in enumerate(base_probabilities):
      base_logit = logit(base_prob)
      noisy_logits = base_logit + np.random.normal(loc=expected_effect, scale=noise_multiplier, size=10000)
      expected_improvement = base_prob - np.mean(expit(noisy_logits))
      results[99-noise_ix, prob_ix] = expected_improvement

  xticklabels = []
  yticklabels = []
  log_base_probs = np.log10(base_probabilities).round(1)
  log_noise = np.log10(noise_multipliers).round(1)
  np.flip(log_noise)
  sign_log_abs = np.sign(results) * np.log10(np.abs(results))
  for i in range(100):
    if i % 10 == 0:
      xticklabels.append(str(log_base_probs[i]))
      yticklabels.append(str(log_noise[i]))
    else:
      xticklabels.append('')
      yticklabels.append('')
  ax = sns.heatmap(results, xticklabels=xticklabels, yticklabels=yticklabels)
  ax.set_xlabel('log default probability')
  ax.set_ylabel('log noise standard dev')
  # ax = sns.heatmap(results)
  # ax.axis([base_probabilities.min(), base_probabilities.max(), noise_multipliers.max(), noise_multipliers.min()])
  plt.show()

