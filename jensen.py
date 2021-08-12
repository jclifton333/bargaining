import numpy as np
from scipy.special import expit, logit
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
  expected_effect = -0.1
  # noise_multipliers = np.logspace(-2, 0, 100)
  noise_multipliers = np.linspace(0.01, 1, 100)
  base_probabilities = np.logspace(-5, -2, 100)
  results = np.zeros((100, 100))
  for noise_ix, noise_multiplier in enumerate(noise_multipliers):
    for prob_ix, base_prob in enumerate(base_probabilities):
      base_logit = logit(base_prob)
      noisy_logits = base_logit + np.random.normal(loc=expected_effect, scale=noise_multiplier, size=10000)
      expected_improvement = np.mean(expit(noisy_logits)) / base_prob
      results[99-noise_ix, prob_ix] = expected_improvement

  xticklabels = []
  yticklabels = []
  log_base_probs = np.log10(base_probabilities).round(1)
  log_noise = np.log10(noise_multipliers / np.abs(expected_effect)).round(1)
  log_noise = log_noise[::-1]
  noise = (noise_multipliers / np.abs(expected_effect)).round(1)
  noise = noise[::-1]
  # sign_log_abs = np.log10(np.abs(results))
  for i in range(100):
    if i % 10 == 0:
      xticklabels.append(str(log_base_probs[i]))
      # yticklabels.append(str(log_noise[i]))
      yticklabels.append(str(noise[i]))
    else:
      xticklabels.append('')
      yticklabels.append('')
  # ax = sns.heatmap(results, xticklabels=xticklabels, yticklabels=yticklabels, center=1, cmap='PiYG')
  # ax.set_xlabel('log default probability')
  # # ax.set_ylabel(r'$\log( \sigma / \delta$ )')
  # ax.set_ylabel(r'$\sigma / \delta$')
  # ax.collections[0].colorbar.set_label('expected intervention prob / base prob')
  # # ax = sns.heatmap(results)
  # # ax.axis([base_probabilities.min(), base_probabilities.max(), noise_multipliers.max(), noise_multipliers.min()])
  # plt.show()
  ratios = results.mean(axis=1)
  plt.plot(noise, ratios)
  plt.hlines(1, noise.min(), noise.max(), linestyles='dashed')
  plt.xlabel(r'Noise-to-signal ratio $\sigma / \delta$')
  plt.ylabel(r'$\mathbb{E}$[post-intervention probability] / default probability')
  plt.show()
