import numpy as np
import matplotlib.pyplot as plt
import pdb


def pilot_model_helper(n_rep, early_scenario_probs_dbn, late_scenario_probs_dbn, late_disvalue_mean):
  disvalue = 0.
  for rep in range(N_REP):
    early_scenario_probs = early_scenario_probs_dbn()
    early_scenario = np.random.multinomial(n=1, pvals=early_scenario_probs)
    if early_scenario == 0:  # Early competition
      disvalue += 1 / N_REP
    elif early_scenario == 3:  # Move to late period
      late_scenario_probs = late_scenario_probs_dbn()
      late_scenario = np.random.binomial(1, late_scenario_probs)
      if late_scenario == 0:
        disvalue += late_disvalue_mean / N_REP

  return disvalue


def pilot_model():
  N_REP = 1000

  # Baseline
  early_scenario_probs_dbn_ = None
  late_scenario_probs_dbn_ = None
  late_disvalue_mean_ = None

  baseline_disvalue = pilot_model_helper(N_REP, early_scenario_probs_dbn_, late_scenario_probs_dbn_,
                                         late_disvalue_mean_)

  # With coopAI
  early_scenario_probs_dbn_cai = None
  late_scenario_probs_dbn_cai = None
  late_disvalue_mean_cai = None

  coopai_disvalue = pilot_model_helper(N_REP, early_scenario_probs_dbn_cai, late_scenario_probs_dbn_cai,
                                       late_disvalue_mean_cai)

  diff = baseline_disvalue - coopai_disvalue

  return


def probs_from_expert_survey():
  catastrophe_prob = 0.49 / (0.49 + 1.72)
  war_dbn = catastrophe_prob * 2 * np.random.beta(0.96, 6, size=1000)
  singleton_dbn = catastrophe_prob * np.random.beta(0.96, 6, size=1000)

  war_quantiles = np.percentile(war_dbn, [1, 25, 50, 75, 99])
  singleton_quantiles = np.percentile(singleton_dbn, [1, 25, 50, 99])

  hazard_model_war= np.random.beta(0.96, 7.58, size=1000)
  war_combined_quantiles = np.percentile(np.concatenate((war_dbn, hazard_model_war)), [1, 25, 50, 75, 99])

  print(f'war: {war_quantiles}\nsingleton: {singleton_quantiles}\nwar merged: {war_combined_quantiles}')


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
  # central_params = (0.87, 53.45)

  # # params for ad-hoc sensitivity analysis
  # sensitivity_params = [central_params,
  #                       (1.41, 106.75),  # Median=0.01, 99 percentile=0.05
  #                       (0.5, 53.45)]    # Median=0.004, 95 percentile=0.04

  # for a, b in sensitivity_params:
  #   dbn = decisive_conflict_hazard_model(alpha_hazard_rate=a, beta_hazard_rate=b)
  #   # plt.hist(dbn, alpha=0.5)
  #   print(np.percentile(dbn, [1, 50, 99]))
  # plt.show()
  probs_from_expert_survey()

