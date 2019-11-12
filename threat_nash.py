import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nashpy as nash



def threat_game_nash(p_commit_1, p_commit_2, cost_1, cost_2):
  """

  :param p_commit_1:
  :param p_commit_2:
  :param cost_1:
  :param cost_2:
  :return:
  """
  # Define payoff matrices
  player_1 = [[0, -p_commit_2], [1, (1-p_commit_2) + p_commit_2*cost_1]]
  player_2 = [[0, 1], [-p_commit_1, (1-p_commit_1) + p_commit_1*cost_2]]

  # Compute and display equilibria
  game = nash.Game(player_1, player_2)

  # ToDo: assuming last eq is mixed!
  game_values = [game[eq] for eq in game.support_enumeration()][-1]
  return game_values


def nash_bargaining_solution(p_commit_1, p_commit_2, disagreement_points):
  """

  :param p_commit_1:
  :param p_commit_2:
  :param disagreement_points: Array [disagreement player i type commit for
                                      i =1, 2]
  :return:
  """
  def nash_welfare(q):
    player_1_nash_welfare = np.log(q - (1-q) + p_commit_2)*(1 - p_commit_1) + \
                            np.log(q - (1-q) - disagreement_points[0])*p_commit_1
    player_2_nash_welfare = np.log(1 - q - q + p_commit_1)*(1 - p_commit_2) + \
                            np.log(1 - q - q - disagreement_points[1])*p_commit_2
    return player_1_nash_welfare + player_2_nash_welfare

  # Brute force search for NBS
  pstar = None
  max_welfare = -float('inf')
  for p_ in np.linspace(0.01, 0.99, 10000):
    p_welfare = nash_welfare(p_)
    if p_welfare > max_welfare:
      pstar = p_
      max_welfare = p_welfare
  return pstar


def threat_game_bargaining(p_commit_1, p_commit_2, cost_1, cost_2):
  disagreement_points = threat_game_nash(p_commit_1, p_commit_2, cost_1, cost_2)
  p_nbs = nash_bargaining_solution(p_commit_1, p_commit_2, disagreement_points)
  return p_nbs


if __name__ == "__main__":
  solutions = []
  probs = np.linspace(0.0, 1.0, 100)
  for p in probs:
    s = threat_game_bargaining(p, 0.5, -5, -5)
    solutions.append(s)
  sns.set_style("darkgrid")
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title("Nash bargaining allocation as a function of threat credibility", fontsize=25)
  ax.set_xlabel('Prior that Player 1 has commitment device', fontsize=20)
  ax.set_ylabel('Bargaining proportion for Player 1', fontsize=20)
  for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
  for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
  ax.plot(probs, solutions)
  plt.show()
  # plt.plot(probs, solutions)
  # plt.xlabel('Prior that Player 1 has commitment device')
  # plt.ylabel('Bargaining proportion for Player 1')
  # plt.title("Nash bargaining allocation as a function of threat credibility")
  # plt.show()

