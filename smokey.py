import numpy as np
import matplotlib.pyplot as plt
import pdb


def decisive_conflict_hazard_model(alpha_hazard_rate=0.87, beta_hazard_rate=53.45):
  # For default using median=0.01 and 95 percentile=0.05 in Nix calculator https://observablehq.com/@ngd/beta-distributions

  N_YEARS = 20
  MC_REPLICATES = 1000
  alpha_hazard_rate = 0.87
  beta_hazard_rate = 53.45

  def decisive_conflict_prob_given_hazard_rate(hazard_rate):
    prob = 0.
    for T in range(N_YEARS):
      prob += (1 - np.exp(-hazard_rate*T)) / N_YEARS
    return prob

  draws = np.random.beta(a=alpha_hazard_rate, b=beta_hazard_rate, size=MC_REPLICATES)
  decisive_conflict_distribution = np.array([decisive_conflict_prob_given_hazard_rate(h) for h in draws])
  return decisive_conflict_distribution


def late_conflict_probs():
  outcomes = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
  n_outcomes = len(outcomes)
  probs = np.array([0.05, 0.225, 0.225, 0.225, 0.225, 0.05])




if __name__ == "__main__":
  central_params = (0.87, 53.45)

  # params for ad-hoc sensitivity analysis
  sensitivity_params = [central_params,
                        (1.41, 106.75),  # Median=0.01, 99 percentile=0.05
                        (0.5, 53.45)]    # Median=0.004, 95 percentile=0.04

  for a, b in sensitivity_params:
    dbn = decisive_conflict_hazard_model(alpha_hazard_rate=a, beta_hazard_rate=b)
    # plt.hist(dbn, alpha=0.5)
    print(np.percentile(dbn, [1, 50, 99]))
  # plt.show()

