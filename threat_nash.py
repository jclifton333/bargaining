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

  print(np.array(player_1))
  print(np.array(player_2))

  # Compute and display equilibria
  game = nash.Game(player_1, player_2)
  for eq in game.support_enumeration():
    print(eq, game[eq[0], eq[1]])


if __name__ == "__main__":
  for p in np.linspace(0.1, 0.9, 9):
    threat_game_nash(p, 0.4, -10, -10)

