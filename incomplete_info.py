import numpy as np
from functools import partial
import nashpy
import matplotlib.pyplot as plt
import pdb

THREATENER_PROFILES = [(0, 0), (0, 1)]
TARGET_PROFILES = [(0, 0), (0, 1), (1, 0), (1, 1)]

def complete_information_payoffs(threatener_type, target_type, p, c, low_cost, high_cost):
  # Not commit, Low
  if threatener_type == 0 and target_type == 0:
    u_threatener= np.array([[1, p], [1, p]])
    u_target= np.array([[0, 1-p], [0, 1-p]])

  # Not commit, high
  if threatener_type == 0 and target_type == 1:
    u_threatener = np.array([[1, p], [1, p]])
    u_target = np.array([[0, 1 - p], [0, 1 - p]])

  # Commit, low
  if threatener_type == 1 and target_type == 0:
    u_threatener = np.array([[1, p], [1, p*(1-c)]])
    u_target = np.array([[0, 1-p], [0, 1-p - p*low_cost]])

  # Commit, high
  if threatener_type == 1 and target_type == 1:
    u_threatener = np.array([[1, p], [1, p * (1 - c)]])
    u_target = np.array([[0, 1 - p], [0, 1 - p - p * high_cost]])

  return u_threatener, u_target


def get_payoffs(threatener_profile, target_profile, commit_prior_if_committed, commit_prior_if_not_committed,
                cost_prior, low_cost, high_cost, p, c):
  """
  Threatener actions {Not commit, Commit} indexed {0, 1}
  Target actions {Give in, Fight} indexed {0, 1}
  """
  complete_info_partial = partial(complete_information_payoffs, p=p, c=c, low_cost=low_cost, high_cost=high_cost)
  u_threatener = 0.
  u_target = 0.

  for threatener_type_ix, threatener_action in enumerate(threatener_profile):
    for target_type_ix, target_action in enumerate(target_profile):
      if threatener_action == 0:
        commit_prior = commit_prior_if_not_committed
      else:
        commit_prior = commit_prior_if_committed

      u_threatener_type, u_target_type = complete_info_partial(threatener_type_ix, target_type_ix)
      u_threatener += cost_prior[target_type_ix] * commit_prior[threatener_type_ix] * \
                      u_threatener_type[threatener_action, target_action]
      u_target += cost_prior[target_type_ix] * commit_prior[threatener_type_ix] * \
                      u_target_type[threatener_action, target_action]

  return u_threatener, u_target


def create_payoff_matrix(p, c, low_cost, high_cost, commit_prior_if_committed, commit_prior_if_not_committed,
                         cost_prior):
  """
  Target type is in {L, H} for Low/High cost of threat execution
  Threatener type is in {N, C} for able to Commit/not able to commit

  Target actions are in {Give in, Fight}
  Threatener actions are in {Not commit, Commit}

  Target strategies
    (L, H) -> (G, G)
    (L, H) -> (G, F)
    (L, H) -> (F, G)
    (L, H) -> (F, F)
  Threatener strategies
    (C, N) -> (C, N)
    (C, N) -> (N, N)
  """
  payoffs_threatener = np.zeros((2, 4))
  payoffs_target = np.zeros((2, 4))

  get_payoffs_partial = partial(get_payoffs, commit_prior_if_committed=commit_prior_if_committed,
                                commit_prior_if_not_committed=commit_prior_if_not_committed,
                                cost_prior=cost_prior, low_cost=low_cost,
                                high_cost=high_cost, p=p, c=c)
  for threatener_profile_ix, threatener_profile in enumerate(THREATENER_PROFILES):
    for target_profile_ix, target_profile in enumerate(TARGET_PROFILES):
      u_threatener, u_target = get_payoffs_partial(threatener_profile, target_profile)
      payoffs_threatener[threatener_profile_ix, target_profile_ix] = u_threatener
      payoffs_target[threatener_profile_ix, target_profile_ix] = u_target

  return payoffs_threatener, payoffs_target


def threat_execution_probability_from_profile(threatener_strategy, target_strategy, commit_prior_if_committed,
                                              commit_prior_if_not_committed, cost_prior):
  # ToDo: think again about what priors to use to evaluate this; currently conditioning on commitment types
  threat_execution_probability = (threatener_strategy[1] == 1) * ((target_strategy[1] == 1) * cost_prior[1] + \
                                                                   (target_strategy[2] == 1) * cost_prior[0] + \
                                                                   (target_strategy[3] == 1))

  return threat_execution_probability


def get_index_from_onehot(onehot):
  for ix, x in enumerate(onehot):
    if x == 1:
      return ix


def get_interim_payoffs(threatener_strategy, target_strategy, commit_prior_if_committed, commit_prior_if_not_committed,
                        cost_prior, p, c, low_cost, high_cost):

  complete_information_payoffs_partial = partial(complete_information_payoffs, p=p, c=c, low_cost=low_cost,
                                                 high_cost=high_cost)
  threatener_actions = THREATENER_PROFILES[(threatener_strategy[1] == 1)]
  target_actions = TARGET_PROFILES[get_index_from_onehot(target_strategy)]

  # ToDo: use commit_prior_if_committed or not?
  # Threatener No commit interim payoffs
  u_threatener_nl, _ = complete_information_payoffs_partial(0, 0)[threatener_actions[0], target_actions[0]]
  u_threatener_nh, _ = complete_information_payoffs_partial(0, 1)[threatener_actions[0], target_actions[1]]
  u_threatener_n = cost_prior[0]*u_threatener_nl + cost_prior[1]*u_threatener_nh

  # Threatener commit interim payoffs
  u_threatener_cl, _ = complete_information_payoffs_partial(1, 0)[threatener_actions[1], target_actions[0]]
  u_threatener_ch, _ = complete_information_payoffs_partial(1, 1)[threatener_actions[1], target_actions[1]]
  u_threatener_c = cost_prior[0] * u_threatener_cl + cost_prior[1] * u_threatener_ch

  # Target low interim payoffs
  _, u_target_nl = complete_information_payoffs_partial(0, 0)[threatener_actions[0], target_actions[0]]
  _, u_target_cl = complete_information_payoffs_partial(1, 0)[threatener_actions[1], target_actions[0]]
  u_target_l = commit_prior_if_committed[0] * u_target_nl + commit_prior_if_committed[1] * u_target_cl # ToDo: priors don't make sense??

  # Target high interim payoffs
  _, u_target_nh = complete_information_payoffs_partial(0, 1)[threatener_actions[0], target_actions[1]]
  _, u_target_ch = complete_information_payoffs_partial(1, 1)[threatener_actions[1], target_actions[1]]
  u_target_h = commit_prior_if_committed[1] * u_target_nh + commit_prior_if_committed[1] * u_target_ch

  return {'u_threatener_n': u_threatener_n, 'u_threatener_c': u_threatener_c, 'u_target_l': u_target_l,
          'u_target_h': u_target_h}


def get_equilibria(p, c, low_cost, high_cost, commit_prior_if_committed, commit_prior_if_not_committed, cost_prior):
  payoffs_threatener, payoffs_target = create_payoff_matrix(p, c, low_cost, high_cost, commit_prior_if_committed,
                                                            commit_prior_if_not_committed, cost_prior)
  game = nashpy.Game(payoffs_threatener, payoffs_target)
  eqs = list(game.support_enumeration())
  # ToDo: check for multiple equilibria?
  return game, eqs[0]


def target_best_response(action_threatener, signal, p, c, low_cost, high_cost, prior_if_signal, prior_if_no_signal,
                         cost_prior):
  if signal == 0:
    p_carry_out = prior_if_no_signal[1]
  else:
    p_carry_out = prior_if_signal[1]

  # Get best response for target
  payoff_fight_l = (1 - p) - p*low_cost*p_carry_out
  payoff_fight_h = (1 - p) - p*high_cost*p_carry_out
  payoff_give_l = 0.
  payoff_give_h = 0.

  action_l = payoff_fight_l > payoff_give_l
  action_h = payoff_fight_h > payoff_give_h

  # Get corresponding payoffs for threatener
  if action_threatener == 0:
    payoff_threatener = cost_prior[0] * (action_l*p + (1 - action_l)) + cost_prior[1] * (action_h*p + (1 - action_h))
  else:
    payoff_threatener = cost_prior[0] * (action_l * (p - c) + (1 - action_l)) + cost_prior[1] * (
                        action_h * (p - c) + (1 - action_h))

  return action_l, action_h, payoff_threatener


def threatener_lookahead(p, c, low_cost, high_cost, prior_if_signal, prior_if_no_signal, prob_signal_if_commit,
                         prob_signal_if_no_commit, cost_prior):
  target_best_response_partial = partial(target_best_response, p=p, c=c, low_cost=low_cost, high_cost=high_cost,
                                         prior_if_signal=prior_if_signal, prior_if_no_signal=prior_if_no_signal,
                                         cost_prior=cost_prior)
  payoff_commit = prob_signal_if_commit * target_best_response_partial(1, 1)[2] + \
                    (1 - prob_signal_if_commit) * target_best_response_partial(1, 0)[2]
  payoff_dont_commit = prob_signal_if_no_commit * target_best_response_partial(0, 1)[2] + \
                        (1 - prob_signal_if_no_commit) * target_best_response_partial(0, 0)[2]
  commit_type_action = payoff_commit > payoff_dont_commit
  return 0, commit_type_action


def get_sequential_equilibrium(p, c, low_cost, high_cost, commit_prior_if_committed, commit_prior_if_not_committed,
                               cost_prior):
  pass


def randomize_between_priors(prior1, prior2, p):
  if p < np.random.random():
    return prior1
  else:
    return prior2


def cross_play(p, c, low_cost, high_cost, prior_on_high_commit_prior, low_commit_prior,
               prior_on_cost_prior, true_cost_prior, true_high_commit_prior, p_high_prior_given_commit=0.6,
               p_high_prior_given_not_commit=0.4, reps=10):

  true_game, _ = get_equilibria(p, c, low_cost, high_cost, true_high_commit_prior, low_commit_prior,
                                true_cost_prior)

  total_threat_execution_probability = 0.
  total_threat_made_probability = 0.
  cgs_total_threat_execution_probability = 0.
  default_payoffs = np.zeros(2)
  cgs_payoffs = np.zeros(2)
  prior_diffs = []
  for rep in range(reps):
    # Draw each player's priors
    threatener_cost_prior = prior_on_cost_prior()
    target_cost_prior = prior_on_cost_prior()

    threatener_high_prior = prior_on_high_commit_prior()
    target_high_prior = prior_on_high_commit_prior()

    threatener_commit_prior_if_committed = prior_on_commit_prior_if_committed()
    target_commit_prior_if_committed = prior_on_commit_prior_if_committed()
    prior_diffs.append(threatener_commit_prior_if_committed[0] - target_commit_prior_if_committed[0])

    # Get corresponding predictions
    _, threatener_eqs = get_equilibria(p, c, low_cost, high_cost, threatener_commit_prior_if_committed,
                                    commit_prior_if_not_committed, threatener_cost_prior)
    _, target_eqs = get_equilibria(p, c, low_cost, high_cost, target_commit_prior_if_committed,
                                commit_prior_if_not_committed, target_cost_prior)
    threat_execution_probability = threat_execution_probability_from_profile(threatener_eqs[0], target_eqs[1],
                                                                             true_commit_prior_if_committed,
                                                                             commit_prior_if_not_committed,
                                                                             true_cost_prior)
    total_threat_execution_probability += threat_execution_probability / reps
    total_threat_made_probability += threatener_eqs[0][1] / reps
    default_payoffs += true_game[(threatener_eqs[0], target_eqs[1])] / reps

    # Get outcome
    cgs_cost_prior = (threatener_cost_prior + target_cost_prior) / 2
    cgs_commit_prior_if_committed = (threatener_commit_prior_if_committed + target_commit_prior_if_committed) / 2
    _, cgs_eqs = get_equilibria(p, c, low_cost, high_cost, cgs_commit_prior_if_committed,
                                commit_prior_if_not_committed, cgs_cost_prior)
    cgs_threat_execution_probability = threat_execution_probability_from_profile(cgs_eqs[0], cgs_eqs[1],
                                                                                 true_commit_prior_if_committed,
                                                                                 commit_prior_if_not_committed,
                                                                                 true_cost_prior)
    cgs_total_threat_execution_probability += cgs_threat_execution_probability / reps
    cgs_payoffs += true_game[cgs_eqs] / reps

  print(default_payoffs, cgs_payoffs)

  return total_threat_execution_probability, cgs_total_threat_execution_probability, total_threat_made_probability, \
         prior_diffs


def get_prior_over_priors(true_prior, variance_multiplier, size=1):
  """
  beta distribution over prior[0], centered at true_prior[0]
  """
  mean = true_prior[0]
  variance = variance_multiplier * mean * (1 - mean)
  alpha = mean**2 * ((1-mean) / variance - (1 / mean))
  beta = alpha * ((1/mean) - 1)

  def prior_over_priors():
    p = np.random.beta(a=alpha, b=beta, size=size)
    if size == 1:
      p = p[0]
    return np.array([p, 1-p])

  return prior_over_priors


if __name__ == "__main__":
  # ToDo: enforce P(commitment type | commitment) > P(commitment type | no commitment)
  low_commit_prior = [0.9, 0.1]
  true_high_commit_prior = [0.5, 0.5]
  true_cost_prior = [0.9, 0.1]

  commit_prior_variance_multiplier = 0.9  # Must be < 1
  cost_prior_variance_multiplier = 0.9

  # Get distributions over credences
  prior_on_cost_prior = get_prior_over_priors(true_cost_prior, cost_prior_variance_multiplier)
  prior_on_high_commit_prior = get_prior_over_priors(true_high_commit_prior,
                                                     commit_prior_variance_multiplier)

  # ToDo: need to account for subgame perfection
  p = 0.5
  c = 0.1
  low_cost = 0.2
  high_cost = 1.
  threat_prob, cgs_threat_prob, threat_made_prob, prior_diffs = cross_play(p, c, low_cost, high_cost,
                                                                           prior_on_high_commit_prior,
                                                                           low_commit_prior,
                                                                           prior_on_cost_prior,
                                                                           true_cost_prior,
                                                                           true_commit_prior_if_committed, reps=100)
  print(threat_prob, cgs_threat_prob, threat_made_prob)
  # plt.hist(prior_diffs)
  # plt.show()


