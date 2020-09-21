import nashpy as nash
import numpy as np
import pdb


def generate_latex_table(payoffs_1, payoffs_2, one_minus_alpha_lst=(0.55, 0.75, 0.95)):
  nA_1, nA_2 = payoffs_1.shape
  table = ''
  for i in range(nA_1):
    for j in range(nA_2):
      if j == 0:
        one_minus_alpha = one_minus_alpha_lst[i]
        table += '$\ltft_{{{}}}$ & '.format(one_minus_alpha)

      p1 = payoffs_1[i, j]
      p2 = payoffs_2[i, j]
      table += f'${p1},{p2}$'

      if j == nA_2 - 1:
        table += ' \\\\ \\hline \n'
      else:
        table += ' & '
  print(table)


if __name__ == "__main__":
  """
  Results from https://www.notion.so/Weekly-Report-2020-09-20-c29f682d2cd8474d91dd3f744eb6dd59. 
  
  Player 1: LTFT for 1-\alpha = 0.55, 0.75, 0.95
  Player 2: LTFT for 1-\alpha = 0.55, 0.75, 0.95 ; Exploiter for 1-\alpha = 0.55, 0.75, 0.95 
  """
  payoffs_1 = np.array([[-1.28, -0.97, -0.98, -1.03, -1.06, -0.91],
                         [-1.44, -1.13, -1.09, -1.19, -1.25, -1.21],
                         [-1.21, -1.12, -1.11, -1.20, -1.27, -1.41]])

  payoffs_2 = np.array([[-1.20, -1.48, -1.38, -1.41, -1.43, -1.81],
                         [-0.97, -1.1, -1.13, -1.11, -1.11, -1.35],
                         [-1.06, -1.1, -1.11, -1.06, -1.03, -0.98]])

  generate_latex_table(payoffs_1, payoffs_2)

  game = nash.Game(payoffs_1, payoffs_2)
  eqs = list(game.support_enumeration())
  print(eqs)


