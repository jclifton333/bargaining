import pdb
import numpy as np
from copy import copy
import matplotlib.pyplot as plt


# Frequency-dependent Moran process for the oligpoly game,
# with rational and behavioral types.


def level_k_action(my_pref, their_pref_estimate, k, m):
  """
  Ge level-2 action, where I best-respond to a model of the other player (given by their_pref_estimate)
  who is best-responding to my Nash strategy.
  """
  level0 = m / (2-k)
  br_to_level0 = subjective_best_response(level0, their_pref_estimate, k, m)
  br_to_br_to_level0 = subjective_best_response(br_to_level0, my_pref, k, m)
  return br_to_br_to_level0


def level_k_solve_for_preference(their_action, k, m):
  """
  Infer what the other player's preference is, given their action and the assumption they model me
  as playing Nash.
  """
  their_pref_estimate = (their_action - 0.25 - k*m / (4-2*k)) * (4-2*k) / (k*m)
  return their_pref_estimate


def level_k_interact(a, b, ahat, bhat, k, m, num_timesteps=30):
  mean_fitness_1 = 0.
  mean_fitness_2 = 0.
  for t in range(num_timesteps):
    x = level_k_action(a, bhat, k, m) # ToDo: use distribution instead of point estimate?
    y = level_k_action(b, ahat, k, m)
    new_ahat = level_k_solve_for_preference(x, k, m)
    new_bhat = level_k_solve_for_preference(y, k, m)
    ahat = ahat + (new_ahat - ahat) / (t + 2)
    bhat = bhat + (new_bhat - bhat) / (t + 2)
    f1 = oligopoly_fitness(x, y, k, m)
    f2 = oligopoly_fitness(y, x, k, m)
    mean_fitness_1 = mean_fitness_1 + (f1 - mean_fitness_1) / (t + 1)
    mean_fitness_2 = mean_fitness_2 + (f2 - mean_fitness_2) / (t + 1)
  return mean_fitness_1, mean_fitness_2


def oligopoly_fitness(my_action, their_action, k, m):
  f = my_action*(their_action*k + m - my_action)
  return f


def fitness_rational_types(a, b, k, m):
  xstar = m * (k * (a + 1) + 2) / (4 - k**2 * (a+1) * (b+1))
  ystar = m * (k * (b + 1) + 2) / (4 - k ** 2 * (a + 1) * (b + 1))
  f1 = oligopoly_fitness(xstar, ystar, k, m)
  f2 = oligopoly_fitness(ystar, xstar, k, m)
  return f1, f2, xstar, ystar


def subjective_best_response(their_action, my_pref, k, m):
  br = 0.25 * (1 + 2*k*their_action + 2*k*their_action*my_pref)
  return br


def interact(agent1, agent2, k, m, behavior_cost=0.2, rational_level_k=True):
  """
  Agents are tuples (bool, strategy), where
    bool = 0 indicates behavioral type, and strategy is an action in the base game.
    bool = 1 indicates rational type, and strategy is a scalar corresponding to subjective preference.
  """
  rational1, rational2 = agent1[0], agent2[0]
  strategy1, strategy2 = agent1[1], agent2[1]

  if rational1 and rational2:
    if rational_level_k:
      action1 = 0
      ahat = np.random.normal(0, 2) # ToDo: think about how to generate these
      bhat = np.random.normal(0, 2)
      f1, f2 = level_k_interact(strategy1, strategy2, ahat, bhat, k, m)
    else:
      f1, f2, action1, action2 = fitness_rational_types(strategy1, strategy2, k, m)
  elif not rational1 and not rational2:
    action1 = strategy1
    f1 = oligopoly_fitness(strategy1, strategy2, k, m) - behavior_cost
    f2 = oligopoly_fitness(strategy2, strategy1, k, m) - behavior_cost
  elif rational1 and not rational2:
    action1 = subjective_best_response(strategy2, strategy1, k, m)
    f1 = oligopoly_fitness(action1, strategy2, k, m)
    f2 = oligopoly_fitness(strategy2, action1, k, m) - behavior_cost
  elif not rational1 and rational2:
    action1 = strategy1
    action2 = subjective_best_response(strategy1, strategy2, k, m)
    f1 = oligopoly_fitness(strategy1, action2, k, m) - behavior_cost
    f2 = oligopoly_fitness(action2, strategy1, k, m)

  return f1, f2, action1


def draw_random_new_agent(behavior_prob=0.0):
  if np.random.uniform() < behavior_prob:
    bool = 0
    strategy = np.random.normal(0, 2)
  else:
    bool = 1
    strategy = np.random.normal(0, 2)
  type = (bool, strategy)
  return type


def draw_population(population_size):
  population = [draw_random_new_agent() for _ in range(population_size)]
  return population


def construct_game_matrix(population, k, m):
  population_size = len(population)
  game_matrix = np.zeros((population_size, population_size))
  for i, agent1 in enumerate(population):
    for j, agent2 in enumerate(population):
      f1, _, _ = interact(agent1, agent2, k, m)
      game_matrix[i, j] = f1
  return game_matrix


def new_generation(population, game_matrix, k, m, mutation_rate=0.2):
  population_size = len(population)

  # Replace member of population
  fitnesses = game_matrix.mean(axis=1)
  min_fitness = np.min(fitnesses)
  if np.allclose(fitnesses - min_fitness, 0):
    reproduction_ix = np.random.choice(population_size)
  else:
    reproduction_probs = (fitnesses - min_fitness) / (fitnesses - min_fitness).sum()
    reproduction_ix = np.random.choice(population_size, p=reproduction_probs)

  death_ix = np.random.choice(population_size)

  if np.random.uniform() < mutation_rate:
    new_agent = draw_random_new_agent()
  else:
    new_agent = copy(population[reproduction_ix])

  population[death_ix] = new_agent

  # Re-compute game matrix
  for i, agent in enumerate(population):
    f1, f2, _ = interact(new_agent, agent, k, m)
    game_matrix[death_ix, i] = f1
    game_matrix[i, death_ix] = f2

  proportion_rational = np.mean([a[0] for a in population])
  average_rational_strategy = np.mean([a[1] for a in population if a[0]])
  max_rational_strategy = np.max([a[1] for a in population if a[0]])
  statistics = {'prop_rational': proportion_rational, 'egal_welfare': min_fitness, 'util_welfare': np.mean(fitnesses),
                'average_rational_strategy': average_rational_strategy, 'max_rational_strategy': max_rational_strategy}

  return population, game_matrix, statistics


def run_moran_process(time_steps=5000, population_size=100, mutation_rate=0.01, k=-1.4, m=1.0):
  population = draw_population(population_size)
  game_matrix = construct_game_matrix(population, k, m)
  proportion_rational_series = []
  util_welfare_series = []
  rational_strategy_series = []

  for _ in range(time_steps):
    population, game_matrix, statistics = new_generation(population, game_matrix, k, m, mutation_rate=mutation_rate)
    
    proportion_rational_series.append(statistics['prop_rational'])
    util_welfare_series.append(statistics['util_welfare'])
    rational_strategy_series.append(statistics['average_rational_strategy'])

  benchmark_welfare = oligopoly_fitness(m / (2-k), m/(2-k), k, m)
  indirect_ess_welfare, _, _, _ = fitness_rational_types(k / (2 - k), k / (2-k), k, m)
  fig, ax1 = plt.subplots()
  ax1.set_ylabel('prop rational')
  ax1.plot(np.arange(time_steps), proportion_rational_series, color='red')
  ax2 = ax1.twinx()
  ax2.set_ylabel('welfare')
  ax2.set_ylim(np.min((benchmark_welfare, indirect_ess_welfare)) - 1,
               np.max((benchmark_welfare, indirect_ess_welfare)) + 1)
  ax2.plot(np.arange(time_steps), util_welfare_series, color='green')

  ax2.axhline(benchmark_welfare)
  ax2.axhline(indirect_ess_welfare)

  plt.show()

  return


if __name__ == "__main__":
  run_moran_process()


